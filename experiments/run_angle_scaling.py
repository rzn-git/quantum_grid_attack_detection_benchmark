"""Does the quantum kernel's matched-contest score depend on how PCA outputs become angles?

Registered 2026-11-22T18:35Z (experiment `angle_scaling`, decision log; backlog row NS21). The
shipped `AngleScaler` is a train-fitted min-max into [-pi, pi], so a heavy-tailed PCA component
leaves the bulk of its rows in a few percent of the angle range. This runner retunes the QSVM
under two scalings that spread the bulk and under the shipped one (the control arm), through
the published tuning function (`run_qkernel`: 50 TPE trials over bandwidth and C on validation
macro-F1, one read of the full test split), with the published splits and training subset.

Arms:  minmax    the shipped scaling; must return the published score (control C1)
       clip      1st to 99th train percentile, then min-max
       quantile  train quantiles to a uniform law

One job per arm and encoding, so the six jobs run side by side:
    python -m experiments.run_angle_scaling --arm quantile --encoding zz --threads 3
After every job has an ok run, the registered reading:
    python -m experiments.run_angle_scaling --read

Registered reading (decision log): d = macro-F1 under an arm minus macro-F1 under `minmax`.
  SCALING_MATTERS  some encoding and new arm has d > THRESHOLD on test and d > 0 on validation
  NULL_SURVIVES    every other outcome
The matched-contest table keeps the shipped scaling under either branch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=None)
THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
if THREADS is not None:
    for _var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    ):
        os.environ[_var] = str(THREADS)

import numpy as np  # noqa: E402

from qgridbench.features.reduce import (  # noqa: E402
    ANGLE_RANGE,
    ClippedAngleScaler,
    QuantileAngleScaler,
)
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402

log = get_logger(__name__)

VARIANT = "binary"
ARMS = ("minmax", "clip", "quantile")
NEW_ARMS = ("clip", "quantile")
ENCODINGS = ("zz", "angle_ry")
PUBLISHED_RUNS = {
    8: "20260813T071427Z_quantum_qkernel_binary_q8",
    12: "20260813T074414Z_quantum_qkernel_binary_q12",
    16: "20260813T083622Z_quantum_qkernel_binary_q16",
}
# the arms registered at each dimensionality (amendment of 2026-11-22: 12 and 16 run the control
# arm and the quantile arm only)
ARMS_AT = {8: ARMS, 12: ("minmax", "quantile"), 16: ("minmax", "quantile")}
ANCHOR_TOL = 1e-6
THRESHOLD = 0.015  # about twice the QSVM's standard deviation over ten training draws (0.0072)
MARGIN = 0.02  # the margin of the gas accuracy rule: a lead under it reads as parity
RUNS = REPO_ROOT / "results" / "runs"


def published_xgb(qubits: int) -> float:
    """The matched contest's best model: XGBoost's ten-seed test macro-F1 at this dimensionality."""
    merged = RUNS / f"merged_rq1_{VARIANT}_pca{qubits}" / "results.json"
    return json.loads(merged.read_text())["xgb"]["aggregate"]["macro_f1"]["mean"]


def lead_reading(xgb: float, best_qsvm: float, margin: float = MARGIN) -> dict:
    """Registered for 12 and 16 dimensions: does XGBoost's lead over the best QSVM hold?"""
    lead = xgb - best_qsvm
    if lead >= margin:
        branch = "LEAD_HOLDS"
    elif lead <= -margin:
        branch = "QUANTUM_AHEAD"
    else:
        branch = "PARITY"
    return {"branch": branch, "lead": lead, "margin": margin, "xgb": xgb, "best_qsvm": best_qsvm}


def central_share(angles: np.ndarray, lo_pct: float = 5.0, hi_pct: float = 95.0) -> np.ndarray:
    """Share of the angle range that the central 90% of rows cover, per component."""
    lo, hi = np.percentile(angles, [lo_pct, hi_pct], axis=0)
    return (hi - lo) / (ANGLE_RANGE[1] - ANGLE_RANGE[0])


def arm_angles(reg, arm: str) -> tuple[dict[str, np.ndarray], object]:
    """The angle view of every split under one arm; every scaler is fitted on train only."""
    if arm == "minmax":
        return reg.angles, reg.angle_scaler
    scaler = ClippedAngleScaler() if arm == "clip" else QuantileAngleScaler()
    scaler.fit(reg.X["train"])
    return {s: scaler.transform(reg.X[s]) for s in ("train", "val", "test")}, scaler


def verdict(scores: dict, threshold: float = THRESHOLD) -> dict:
    """The registered reading. scores[encoding][arm] = {"test": float, "val": float}."""
    rows, hits = [], []
    for enc, by_arm in scores.items():
        base = by_arm["minmax"]
        for arm in (a for a in NEW_ARMS if a in by_arm):
            d_test = by_arm[arm]["test"] - base["test"]
            d_val = by_arm[arm]["val"] - base["val"]
            hit = bool(d_test > threshold and d_val > 0)
            rows.append(
                {"encoding": enc, "arm": arm, "d_test": d_test, "d_val": d_val, "matters": hit}
            )
            if hit:
                hits.append((enc, arm))
    return {
        "branch": "SCALING_MATTERS" if hits else "NULL_SURVIVES",
        "threshold": threshold,
        "differences": rows,
        "largest_d_test": max(r["d_test"] for r in rows),
        "smallest_d_test": min(r["d_test"] for r in rows),
    }


def run_job(arm: str, encoding: str, qubits: int) -> None:
    import optuna

    from experiments.exports import Exports
    from experiments.run_quantum import run_qkernel
    from qgridbench.protocol import prepare_regime

    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    cfg_c = load_yaml(REPO_ROOT / "configs" / "classical.yaml")
    storage_dir = REPO_ROOT / cfg_c["tuning"]["optuna_storage_dir"]

    reg = prepare_regime(VARIANT, "pca", n_components=qubits, seed=0)
    angles, scaler = arm_angles(reg, arm)
    lo, hi = ANGLE_RANGE
    for split, a in angles.items():
        if a.min() < lo - 1e-12 or a.max() > hi + 1e-12:
            raise ValueError(f"arm {arm}: angles of split {split} leave [-pi, pi]")
    share = central_share(angles["train"])
    share_minmax = central_share(reg.angles["train"])
    if arm in NEW_ARMS and not np.all(share > share_minmax):
        raise ValueError(f"arm {arm} does not spread the bulk on every component: {share}")
    reg_arm = dataclasses.replace(reg, angles=angles, angle_scaler=scaler)

    tag = f"{VARIANT}_q{qubits}_as_{arm}"
    study_name = f"qk_{encoding}_{tag}"
    storage = f"sqlite:///{(storage_dir / f'optuna_{study_name}.db').as_posix()}"
    config = {"arm": arm, "encoding": encoding, "qubits": qubits, "threads": THREADS}
    name = f"angle_scaling_{arm}_{encoding}_{VARIANT}_q{qubits}"
    with RunDir(name, config=config, seeds=[0]) as run:
        exports = Exports(VARIANT, tuned_file=None)
        exports.extra = {"threads": THREADS, "arm": arm, "study": study_name, "qubits": qubits}
        res = run_qkernel(reg_arm, encoding, qubits, [0], cfg_q, storage, tag, run, exports)
        study = optuna.load_study(study_name=study_name, storage=storage)
        out = {
            "arm": arm,
            "encoding": encoding,
            "qubits": qubits,
            "test_macro_f1": res["aggregate"]["macro_f1"]["mean"],
            "val_macro_f1_best_trial": study.best_value,
            "n_trials": len([t for t in study.trials if t.state.is_finished()]),
            "best_params": res["best_params"],
            "diagnostics": res["diagnostics"],
            "central_90_share_train": share.tolist(),
            "central_90_share_train_minmax": share_minmax.tolist(),
        }
        if arm == "minmax":
            pub = json.loads((RUNS / PUBLISHED_RUNS[qubits] / "results.json").read_text())
            published = pub[encoding]["aggregate"]["macro_f1"]["mean"]
            diff = abs(out["test_macro_f1"] - published)
            out["anchor"] = {
                "published_run": PUBLISHED_RUNS[qubits],
                "published": published,
                "abs_diff": diff,
                "tol": ANCHOR_TOL,
                "pass": bool(diff <= ANCHOR_TOL),
            }
        run.write_json("results.json", out)
        exports.write(run.path)
        run.metrics = {
            "test_macro_f1": out["test_macro_f1"],
            "val_macro_f1_best_trial": out["val_macro_f1_best_trial"],
        }
        print(json.dumps(out, indent=2))
        print(f"-> {run.path}")


def newest_ok(arm: str, encoding: str, qubits: int) -> tuple[str, dict]:
    name = f"_angle_scaling_{arm}_{encoding}_{VARIANT}_q{qubits}"
    for path in sorted(RUNS.glob(f"*{name}*"), reverse=True):
        metrics, results = path / "metrics.json", path / "results.json"
        if not (metrics.exists() and results.exists()):
            continue
        if json.loads(metrics.read_text()).get("status") != "ok":
            continue
        return path.name, json.loads(results.read_text())
    raise FileNotFoundError(f"no ok run for arm {arm}, encoding {encoding}")


def read(qubits: int) -> None:
    sources, scores, detail = {}, {}, {}
    arms = ARMS_AT[qubits]
    xgb = published_xgb(qubits)
    for enc in ENCODINGS:
        scores[enc], detail[enc] = {}, {}
        for arm in arms:
            run_id, res = newest_ok(arm, enc, qubits)
            sources[f"{arm}_{enc}"] = run_id
            scores[enc][arm] = {
                "test": res["test_macro_f1"],
                "val": res["val_macro_f1_best_trial"],
            }
            detail[enc][arm] = {
                "best_params": res["best_params"],
                "kta_quantum": res["diagnostics"]["kta_quantum"],
                "central_90_share_train": res["central_90_share_train"],
                "anchor": res.get("anchor"),
            }
    anchors_pass = all(detail[enc]["minmax"]["anchor"]["pass"] for enc in ENCODINGS)
    with RunDir(f"angle_scaling_verdict_{VARIANT}_q{qubits}", config={"qubits": qubits}) as run:
        out = {
            "sources": sources,
            "anchors_pass": anchors_pass,
            "scores": scores,
            "detail": detail,
            "published_xgb": xgb,
        }
        if anchors_pass:
            out["registered"] = verdict(scores)
            best = max(scores[e][a]["test"] for e in ENCODINGS for a in arms if a != "minmax")
            out["best_new_arm_test"] = best
            out["best_new_arm_below_xgb"] = bool(best < xgb)
            out["lead"] = lead_reading(
                xgb, max(scores[e][a]["test"] for e in ENCODINGS for a in arms)
            )
            out["max_kta_quantum"] = max(
                detail[e][a]["kta_quantum"] for e in ENCODINGS for a in arms
            )
            run.metrics = {
                "branch": out["registered"]["branch"],
                "lead_branch": out["lead"]["branch"],
            }
        else:
            out["registered"] = {"branch": "VOID_ANCHOR_FAILED"}
            run.metrics = {"branch": "VOID_ANCHOR_FAILED"}
        run.write_json("results.json", out)
        print(json.dumps(out, indent=2))
        print(f"-> {run.path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--encoding", choices=ENCODINGS)
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--threads", type=int, default=None, help="pin BLAS/OpenMP thread pools")
    ap.add_argument("--read", action="store_true", help="write the registered verdict run")
    args = ap.parse_args()
    if args.read:
        read(args.qubits)
        return
    if args.arm is None or args.encoding is None:
        ap.error("--arm and --encoding are required unless --read is given")
    run_job(args.arm, args.encoding, args.qubits)


if __name__ == "__main__":
    main()
