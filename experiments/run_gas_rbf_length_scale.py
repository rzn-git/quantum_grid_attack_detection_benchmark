"""R4-25: is the gas H-C1a KILL a length-scale effect? The gas RBF-SVM refit at gamma x {0.1, 1,
10, 100} of its tuned value (C fixed), each attacked by its own white-box PGD at 0.5 sigma.

Pre-registration: decision log 2026-11-17T00:10Z (R4-25, Tier A). Replays the RBF-SVM control
of P5a in `20261116T210229Z_whitebox_qsvm_gas_binary_q8` exactly: the same fit (FittedZoo's
rbf_svm on the shared 2,000-row kernel subset, seed 0), the same 500 evaluation rows
(`eval_subset_adv`, stratified, seed 0), the same attack seeds 0-9 and the same RNG stream
`spawn_rng(seed, "p5a")`. In P5a each epsilon draws one PGD random start for the QSVM and one
for the RBF-SVM, in that order, so the stream position of the RBF's 0.5-sigma start is reached
here by drawing the earlier starts without attacking (PGD consumes exactly one
`rng.uniform(-1, 1, size=X.shape)` per call). Only gamma changes between arms.

Comparator: the QSVM at its tuned bandwidth, read from the same artifact (fixed, not refit).
Readability: an arm whose clean macro-F1 lies within 0.05 of the stratified-random floor is
unreadable. Retention = mean attacked macro-F1 over seeds / clean macro-F1.
Branches: LENGTH-SCALE if retention rises by >= 0.30 from gamma x1 to the narrowest readable
arm; NOT LENGTH-SCALE if no readable arm moves by 0.10 from x1; INTERMEDIATE otherwise.
Control (blocking): gamma x1 reproduces the artifact's RBF cells (clean F1 and the ten 0.5-sigma
values) within 1e-6.

Units are gamma multipliers (independent); --workers/--threads, memory guard, resume from an
earlier run's units/.

Run:  uv run python -m experiments.run_gas_rbf_length_scale --workers 1 --threads 4
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
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from experiments.exports import Exports  # noqa: E402
from qgridbench.attacks.evasion import pgd  # noqa: E402
from qgridbench.attacks.gradients import GradProvider  # noqa: E402
from qgridbench.eval.baselines import trivial_baselines  # noqa: E402
from qgridbench.models.classical.zoo import (  # noqa: E402
    build,
    fit_model,
    predict_proba_timed,
    stratified_cap,
)
from qgridbench.protocol import best_params_path, kernel_subset, prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import set_all_seeds, spawn_rng  # noqa: E402

log = get_logger(__name__)

VARIANT = "gas_binary"
QUBITS = 8
ARTIFACT = "20261116T210229Z_whitebox_qsvm_gas_binary_q8"
MULTS = (0.1, 1.0, 10.0, 100.0)
EPS = 0.5
ATTACK_SEEDS = tuple(range(10))
READ_CUT = 0.05
RISE, MOVE = 0.30, 0.10
ANCHOR_TOL = 1e-6
MEM_FRACTION = 0.8


def _f1(y, proba) -> float:
    return float(f1_score(y, proba.argmax(1), average="macro"))


def unit(mult: float) -> dict:
    for v in THREAD_VARS:
        if os.environ.get(v) != str(THREADS):
            raise AssertionError(f"worker {os.getpid()}: {v}={os.environ.get(v)} != {THREADS}")
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    bp = json.loads(best_params_path(VARIANT, QUBITS).read_text())
    reg = prepare_regime(VARIANT, "pca", n_components=QUBITS, seed=0)
    sub = kernel_subset(reg, cfg_q["qkernel"]["subset_cap"], seed=0)
    Xtr, ytr = reg.X["train"][sub], reg.y["train"][sub]
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    X, y = reg.X["test"][ev], reg.y["test"][ev]
    scale = reg.feature_scale
    eps_grid = cfg_adv["evasion"]["epsilons"]
    pgd_cfg = cfg_adv["evasion"]["pgd"]
    if EPS not in eps_grid:
        raise ValueError(f"{EPS} not in the artifact's epsilon grid {eps_grid}")

    tuned = bp["rbf_svm"]
    set_all_seeds(0)
    est = build("rbf_svm", {**tuned, "gamma": tuned["gamma"] * mult}, 0)
    fit_model("rbf_svm", est, Xtr, ytr)
    gp = GradProvider(est, "rbf_svm")
    p_clean = predict_proba_timed(est, X)[0]
    clean = _f1(y, p_clean)
    attacked, arrays = [], {"ev": ev, "y": y, "clean": p_clean, "proba": {}, "crafted": {}}
    for a_seed in ATTACK_SEEDS:
        rng = spawn_rng(a_seed, "p5a")
        for e in eps_grid:
            rng.uniform(-1.0, 1.0, size=X.shape)  # P5a's QSVM start at this epsilon
            if e == EPS:
                break
            rng.uniform(-1.0, 1.0, size=X.shape)  # P5a's RBF start at this epsilon
        Xa = pgd(
            gp,
            X,
            y,
            EPS,
            scale,
            n_steps=pgd_cfg["n_steps"],
            step_frac=pgd_cfg["step_frac"],
            rng=rng,
        )
        p_att = predict_proba_timed(est, Xa)[0]
        attacked.append(_f1(y, p_att))
        arrays["proba"][a_seed], arrays["crafted"][a_seed] = p_att, Xa
    floor = trivial_baselines(ytr, y)["baseline_stratified_random"]["aggregate"]["macro_f1"]["mean"]
    mi = psutil.Process().memory_info()
    return {
        "mult": mult,
        "gamma": float(tuned["gamma"] * mult),
        "C": float(tuned["C"]),
        "clean_f1": clean,
        "attacked_f1_per_seed": attacked,
        "attacked_f1_mean": float(np.mean(attacked)),
        "retention": float(np.mean(attacked) / clean) if clean else float("nan"),
        "floor": floor,
        "readable": bool(abs(clean - floor) > READ_CUT),
        "n_support_vectors": int(est.support_vectors_.shape[0]),
        "peak_private_gb": getattr(mi, "private", mi.rss) / 2**30,
        "_arrays": arrays,
    }


def save_unit_arrays(path, a: dict) -> None:
    """A unit's predictions and crafted inputs, kept beside its JSON so a resume can export."""
    flat = {"ev": a["ev"], "y": a["y"], "clean": a["clean"]}
    for s_, v in a["proba"].items():
        flat[f"proba_s{s_}"] = v
        flat[f"crafted_s{s_}"] = a["crafted"][s_]
    np.savez_compressed(path, **flat)


def load_unit_arrays(path) -> dict:
    d = np.load(path)
    seeds = sorted(int(k[len("proba_s") :]) for k in d.files if k.startswith("proba_s"))
    return {
        "ev": d["ev"],
        "y": d["y"],
        "clean": d["clean"],
        "proba": {s_: d[f"proba_s{s_}"] for s_ in seeds},
        "crafted": {s_: d[f"crafted_s{s_}"] for s_ in seeds},
    }


def write_exports(run_dir, cells: dict, arrays: dict) -> None:
    """Round-4 exports (decision log 2026-11-17T02:18Z) beside results.json."""
    ex = Exports(VARIANT, best_params_path(VARIANT, QUBITS).relative_to(REPO_ROOT).as_posix())
    ex.stream("p5a")
    ex.extra["stream_position"] = (
        "spawn_rng(seed, 'p5a') advanced by the QSVM and RBF starts of the epsilons before 0.5 "
        "and the QSVM start at 0.5, as P5a consumed it"
    )
    ex.extra["artifact"] = ARTIFACT
    for mult, a in sorted(arrays.items()):
        model = f"rbf_svm_gamma_x{mult}"
        ex.predictions(0, "clean", model, a["clean"], a["y"], a["ev"])
        ex.metric(0, model, "clean", "macro_f1", cells[mult]["clean_f1"])
        for s_, proba in sorted(a["proba"].items()):
            cond = f"pgd_self@{EPS}"
            ex.crafted_input(s_, f"{cond}|gamma_x{mult}", model, a["crafted"][s_])
            ex.predictions(s_, cond, model, proba, a["y"], a["ev"])
        for s_, f1 in zip(ATTACK_SEEDS, cells[mult]["attacked_f1_per_seed"], strict=True):
            ex.metric(s_, model, f"pgd_self@{EPS}", "macro_f1", f1)
    ex.write(run_dir)


def anchor(cells: dict, art: dict) -> dict:
    rbf = art["p5a_whitebox_qsvm"]["rbf_control"]
    x1 = cells[1.0]
    pub = rbf["per_seed_pgd"][str(EPS)]
    diffs = [abs(a - b) for a, b in zip(x1["attacked_f1_per_seed"], pub, strict=True)]
    diffs.append(abs(x1["clean_f1"] - rbf["clean_f1"]))
    return {"max_abs_diff": max(diffs), "tol": ANCHOR_TOL, "pass": max(diffs) <= ANCHOR_TOL}


def verdict(cells: dict, art: dict) -> dict:
    x1 = cells[1.0]
    readable = [m for m in sorted(cells) if cells[m]["readable"]]
    narrow = [m for m in readable if m > 1.0]
    rise = cells[max(narrow)]["retention"] - x1["retention"] if narrow else None
    moves = {m: cells[m]["retention"] - x1["retention"] for m in readable if m != 1.0}
    if rise is not None and rise >= RISE:
        branch = "LENGTH-SCALE"
    elif all(abs(v) < MOVE for v in moves.values()):
        branch = "NOT LENGTH-SCALE"
    else:
        branch = "INTERMEDIATE"
    q = art["p5a_whitebox_qsvm"]
    return {
        "readable_mults": readable,
        "narrowest_readable": max(narrow) if narrow else None,
        "rise_x1_to_narrowest": rise,
        "retention_move_from_x1": moves,
        "x1_readable": x1["readable"],
        "branch": branch,
        "qsvm_comparator": {
            "clean_f1": q["clean_f1"],
            "whitebox_retention": q["whitebox_retention_at_half"],
            "artifact": ARTIFACT,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--worker-rss-gb", type=float, default=1.9, help="per-worker private bytes")
    ap.add_argument("--resume", default=None, help="earlier run dir; its finished units are kept")
    args = ap.parse_args()

    avail = psutil.virtual_memory().available / 2**30
    if args.workers * args.worker_rss_gb > MEM_FRACTION * avail:
        raise MemoryError(f"{args.workers} x {args.worker_rss_gb} GB > 80% of {avail:.1f} GB")
    art = json.loads((REPO_ROOT / "results" / "runs" / ARTIFACT / "results.json").read_text())
    done: dict[float, dict] = {}
    arrays: dict[float, dict] = {}
    if args.resume:
        for f in (REPO_ROOT / args.resume / "units").glob("*.json"):
            u = json.loads(f.read_text())
            done[u["mult"]] = u
            arrays[u["mult"]] = load_unit_arrays(f.with_name(f.stem + "_arrays.npz"))
    todo = [m for m in MULTS if m not in done]
    cfg = {**vars(args), "artifact": ARTIFACT, "mults": MULTS, "eps": EPS, "available_gb": avail}
    with RunDir(f"gas_rbf_length_scale_{VARIANT}", config=cfg, seeds=list(ATTACK_SEEDS)) as run:
        udir = run.path / "units"
        udir.mkdir()
        for u in done.values():
            (udir / f"x{u['mult']}.json").write_text(json.dumps(u, indent=1))
            save_unit_arrays(udir / f"x{u['mult']}_arrays.npz", arrays[u["mult"]])
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futs = {pool.submit(unit, m): m for m in todo}
            for fut in as_completed(futs):
                u = fut.result()
                arrays[u["mult"]] = u.pop("_arrays")
                done[u["mult"]] = u
                save_unit_arrays(udir / f"x{u['mult']}_arrays.npz", arrays[u["mult"]])
                (udir / f"x{u['mult']}.json").write_text(json.dumps(u, indent=1))
                log.info(
                    "gamma x%g: clean %.4f retention %.4f (peak %.2f GB)",
                    u["mult"],
                    u["clean_f1"],
                    u["retention"],
                    u["peak_private_gb"],
                )
        write_exports(run.path, done, arrays)
        anc = anchor(done, art)
        v = verdict(done, art)
        run.write_json("results.json", {"anchor": anc, "verdict": v, "cells": done})
        run.metrics.update({"anchor_pass": anc["pass"], "branch": v["branch"]})
        log.info("ANCHOR %s | VERDICT %s", json.dumps(anc), json.dumps(v, default=str))
        print(f"-> {run.path}")
        if not anc["pass"]:
            raise AssertionError(f"gamma x1 anchor failed: {anc}")


if __name__ == "__main__":
    main()
