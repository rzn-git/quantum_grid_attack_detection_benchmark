"""R5-G1: does the shipped standardization of the gas pressure channel cap gas accuracy?

Pre-registration: decision log 2026-11-19T08:39Z, "PRE-REGISTRATION R5-G1". The gas `pressure`
feature carries injected values up to 3.4e38 on attack rows; StandardScaler divides by their
spread (1.8e37), which maps every ordinary reading to one float (NS15 finding, same entry).

Two transforms, everything else equal (seed 0, the shipped gas row split, the tuned gas
hyperparameters, the 2,000-row cap for the PCA-8 models; helpers from run_gas_split_ladder):
  T0  shipped: median fill, StandardScaler (+ PCA-8);
  T1  pressure -> sign(p) * log1p(|p|) before the shipped steps (missing values stay missing).
Units: LightGBM full features (primary), the six classical PCA-8 models, QSVM-ZZ, QSVM angle
RY and the VQC, each under T0 and T1. Anchor: T0 LightGBM equals the gas split ladder's R arm
(0.6401) within 1e-6. Distinct training rows under each transform are reported.
Branch (LightGBM full, delta = T1 - T0): ARTIFACT if delta >= 0.05; NOT_AN_ARTIFACT if
|delta| <= 0.02; INTERMEDIATE otherwise. The PCA-8 contest under T1 is reported, no branch.

Run:
    CUDA_VISIBLE_DEVICES=-1 uv run python -m experiments.run_gas_pressure_transform \
        --workers 3 --threads 4
"""

from __future__ import annotations

import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=4)
THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
for _var in THREAD_VARS:
    os.environ[_var] = str(THREADS)

import json  # noqa: E402
import multiprocessing  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import psutil  # noqa: E402

import experiments.run_gas_split_ladder as gl  # noqa: E402
from qgridbench.data.gas import gas_config, gas_features, signed_log1p  # noqa: E402
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.features.reduce import (  # noqa: E402
    AngleScaler,
    build_preprocessor,
    fit_transform_splits,
)
from qgridbench.models.classical.zoo import (  # noqa: E402
    MODEL_NAMES,
    build,
    fit_model,
    stratified_cap,
)
from qgridbench.models.quantum.vqc import VQClassifier  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import set_all_seeds  # noqa: E402

log = get_logger(__name__)

VARIANT = gl.VARIANT
SEED = gl.SEED
QUBITS = gl.QUBITS
CAP = gl.CAP
LADDER_RUN = REPO_ROOT / "results/runs/20261116T184505Z_gas_split_ladder_gas_binary"
ANCHOR_TOL = 1e-6
ARTIFACT, NOT_ARTIFACT = 0.05, 0.02
PARTS = ("lgbm_full", "classical_pca8", "qsvm_zz", "qsvm_ry", "vqc")


def pressure_index() -> int:
    feats = gas_features(gas_config())
    if feats.count("pressure") != 1:
        raise AssertionError(f"expected one 'pressure' feature in {feats}")
    return feats.index("pressure")


def transform(X: np.ndarray, kind: str) -> np.ndarray:
    """T0 returns the stored matrix unchanged (float32, as the ladder's anchor fit saw it: the
    full-feature LightGBM fit moves with input precision); T1 changes only the pressure column,
    computed in float64 and stored back at the matrix's own dtype."""
    if kind == "T1":
        return signed_log1p(X, [pressure_index()])  # NaN stays NaN
    if kind != "T0":
        raise ValueError(kind)
    return np.array(X, copy=True)


def distinct_rows(X, y, sp) -> dict:
    out = {}
    for regime in ("full", "pca"):
        pipe = build_preprocessor(regime, n_components=QUBITS if regime == "pca" else None, seed=0)
        Z = pipe.fit_transform(X[sp["train"]])
        u, inv = np.unique(Z, axis=0, return_inverse=True)
        inv = inv.ravel()
        ytr = y[sp["train"]]
        conflicting = sum(
            1 for g in np.flatnonzero(np.bincount(inv) > 1) if len(set(ytr[inv == g])) > 1
        )
        out[regime] = {"distinct": int(len(u)), "label_conflicting_groups": int(conflicting)}
    return out


def run_part(spec: dict) -> dict:
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise AssertionError("worker thread variables differ from --threads")
    t0 = time.perf_counter()
    X, y, sp = load_processed(VARIANT)
    X = transform(X, spec["transform"])
    part = spec["part"]
    f1 = {}
    if part == "lgbm_full":
        f1["lgbm_full"] = gl.lgbm_f1(gl.lgbm_full_params(), X, y, sp)
    elif part == "qsvm_zz":
        bp = gl.pca_params()
        f1["qsvm_zz"] = gl.qsvm_f1(bp["qk_bw"], bp["qk_C"], X, y, sp, encoding="zz")
    elif part == "qsvm_ry":
        name = f"qk_angle_ry_{VARIANT}_q{QUBITS}"
        db = (REPO_ROOT / "results" / f"optuna_{name}.db").as_posix()
        ry = optuna.load_study(study_name=name, storage=f"sqlite:///{db}").best_params
        f1["qsvm_angle_ry"] = gl.qsvm_f1(ry["bandwidth"], ry["C"], X, y, sp, encoding="angle_ry")
    else:
        bp = gl.pca_params()
        Z = fit_transform_splits(build_preprocessor("pca", n_components=QUBITS, seed=SEED), X, sp)
        ytr, yva, yte = y[sp["train"]], y[sp["val"]], y[sp["test"]]
        sub = stratified_cap(Z["train"], ytr, CAP, seed=SEED)
        if part == "classical_pca8":
            for m in MODEL_NAMES:
                set_all_seeds(SEED)
                est = build(m, bp[m], SEED)
                fit_model(m, est, Z["train"][sub], ytr[sub])
                f1[m] = gl._f1(yte, est.predict(Z["test"]))
        elif part == "vqc":
            cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
            sc = AngleScaler().fit(Z["train"])
            va = stratified_cap(Z["val"], yva, cfg_q["qkernel"]["tuning_val_subset"], seed=SEED)
            set_all_seeds(SEED)
            vqc = VQClassifier(
                QUBITS,
                depth=bp["vqc_depth"],
                n_classes=2,
                lr=bp["vqc_lr"],
                batch_size=bp["vqc_batch"],
                max_epochs=cfg_q["vqc"]["max_epochs"],
                patience=cfg_q["vqc"]["patience"],
                seed=SEED,
            )
            vqc.fit(sc.transform(Z["train"][sub]), ytr[sub], sc.transform(Z["val"][va]), yva[va])
            f1["vqc"] = gl._f1(yte, vqc.predict(sc.transform(Z["test"])))
        else:
            raise ValueError(part)
    return {
        "unit": f"{spec['transform']}|{part}",
        "transform": spec["transform"],
        "part": part,
        "f1": f1,
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def verdict(units: dict) -> dict:
    f = {t: {} for t in ("T0", "T1")}
    for u in units.values():
        f[u["transform"]].update(u["f1"])
    d = f["T1"]["lgbm_full"] - f["T0"]["lgbm_full"]
    if d >= ARTIFACT:
        branch = "ARTIFACT"
    elif abs(d) <= NOT_ARTIFACT:
        branch = "NOT_AN_ARTIFACT"
    else:
        branch = "INTERMEDIATE"
    contest = {}
    for t in ("T0", "T1"):
        classical = {m: f[t][m] for m in MODEL_NAMES}
        quantum = {m: f[t][m] for m in ("qsvm_zz", "qsvm_angle_ry", "vqc")}
        bc, bq = max(classical, key=classical.get), max(quantum, key=quantum.get)
        contest[t] = {
            "best_classical": [bc, classical[bc]],
            "best_quantum": [bq, quantum[bq]],
            "delta": classical[bc] - quantum[bq],
        }
    return {
        "f1": f,
        "delta_lgbm_full": d,
        "delta_by_model": {m: f["T1"][m] - f["T0"][m] for m in f["T0"]},
        "contest_pca8": contest,
        "branch": branch,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--threads", type=int, required=True)
    args = ap.parse_args()
    if args.threads != THREADS:
        raise AssertionError("early thread parse differs from argparse")
    if args.workers * args.threads > psutil.cpu_count(logical=True):
        raise ValueError("workers x threads exceeds the logical CPU count")
    X, y, sp = load_processed(VARIANT)
    rows = {t: distinct_rows(transform(X, t), y, sp) for t in ("T0", "T1")}
    log.info("distinct training rows %s", json.dumps(rows))
    specs = [{"transform": t, "part": p} for p in reversed(PARTS) for t in ("T0", "T1")]
    cfg = {
        "experiment": "R5-G1",
        "preregistration": "decision log 2026-11-19T08:39Z, PRE-REGISTRATION R5-G1",
        "pressure_feature_index": pressure_index(),
        "T1": "pressure -> sign(p) * log1p(|p|) before median fill and StandardScaler",
        "seed": SEED,
        "workers": args.workers,
        "threads": args.threads,
        "ladder_run": LADDER_RUN.name,
    }
    with RunDir(f"gas_pressure_transform_{VARIANT}", config=cfg, seeds=[SEED]) as run:
        done: dict = {}
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futs = [ex.submit(run_part, s) for s in specs]
            for fut in as_completed(futs):
                r = fut.result()
                done[r["unit"]] = r
                run.write_json("results_partial.json", {"units": done})
                log.info("%s: %s (%.0f s)", r["unit"], json.dumps(r["f1"]), r["wall_s"])
        ladder_r = json.loads((LADDER_RUN / "results.json").read_text())["verdict"]["ladder"]["f1"][
            "R"
        ]
        anchor = {"T0_lgbm_minus_ladder_R": done["T0|lgbm_full"]["f1"]["lgbm_full"] - ladder_r}
        anchor["pass"] = abs(anchor["T0_lgbm_minus_ladder_R"]) <= ANCHOR_TOL
        v = verdict(done)
        run.write_json(
            "results.json",
            {
                "config": cfg,
                "distinct_training_rows": rows,
                "anchor": anchor,
                "verdict": v,
                "units": done,
            },
        )
        (run.path / "exports_provenance.json").write_text(
            json.dumps(
                {
                    "env": {k: os.environ.get(k) for k in (*THREAD_VARS, "CUDA_VISIBLE_DEVICES")},
                    "random_streams": ["seed 0 (set_all_seeds) per fit"],
                    "tuned_files": [
                        "results/optuna_cls_lgbm_gas_binary_full.db",
                        f"protocol.best_params_path({VARIANT!r}, {QUBITS})",
                        f"results/optuna_qk_angle_ry_{VARIANT}_q{QUBITS}.db",
                    ],
                    "notes": "per-model test macro-F1 only; seed 0; the shipped gas row split",
                },
                indent=1,
            )
        )
        run.metrics.update({"branch": v["branch"], "anchor_pass": anchor["pass"]})
        log.info("ANCHOR %s", json.dumps(anchor))
        log.info("VERDICT %s", json.dumps(v))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
