"""N-SLOPE: is the leak-rate law the benchmark's or LightGBM's, and does slope unify it
with selective inflation?

PRE-REGISTERED 2026-09-02T04:02Z (Tier A). N-TEMP showed macro-F1 tracks the residual
near-duplicate rate at r = 0.99 for one tuned tree. This runs the same six split
constructions -- imported from that runner, so they are the identical objects -- across
five classical families at their published tuned hyperparameters, and asks whether each
family sits on the same curve with its own slope.

  H1 (law is the benchmark's): Pearson r >= 0.9 for every family.  KILL: r < 0.7 for >= 2.
  H2 (unification): steepest/shallowest slope >= 2x AND Spearman(slope, row score) >= 0.8.

Controls (blocking): LightGBM reproduces N-TEMP's six arm scores within 1e-3; every
family's shuffled arm lands within 0.02 of its own row-level arm. A constant predictor
fails the run.

Disclosure: the MLP is fitted with the balanced sample-weight correction, not its shipped
unweighted fit, which sits below the stratified-random floor.

Run:  python -m experiments.run_leak_rate_families [--smoke]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

from experiments.run_temporal_split import load_blocks, positional_splits
from qgridbench.data.preprocess import load_processed
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

VARIANT = "binary"
SEED = 0
NN_RADIUS = 0.1
FAMILIES = ["logreg", "rf", "xgb", "lgbm", "mlp"]
BALANCED_FIT = {"mlp"}  # shipped fit sits below the floor; see the module docstring
# N-TEMP's LightGBM arm scores, the blocking positive control
NTEMP_LGBM = {
    "R row-level": 0.9053,
    "T0 ordered": 0.6005,
    "T1 embargo": 0.5903,
    "S shuffled": 0.9054,
    "M middle": 0.6752,
    "V reversed": 0.6178,
}


def tuned_full_params(name: str) -> dict:
    """The published full-feature Optuna winner for one family."""
    db = (REPO_ROOT / "results" / f"optuna_cls_{name}_binary_full.db").as_posix()
    study = optuna.load_study(study_name=f"cls_{name}_binary_full", storage=f"sqlite:///{db}")
    return study.best_params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    X, y, block = load_blocks()
    Xp, yp, splits_row = load_processed(VARIANT)
    if X.shape != Xp.shape or not np.array_equal(y, yp):
        raise SystemExit("block reconstruction misaligned with the processed artifact")
    splits_row = {k: np.asarray(v) for k, v in splits_row.items()}

    rng = np.random.default_rng(SEED)
    t0, _ = positional_splits(block, "ordered", rng)
    t1, _ = positional_splits(block, "embargo", rng)
    sh, _ = positional_splits(block, "shuffled", rng)
    mid, _ = positional_splits(block, "middle", rng)
    rev, _ = positional_splits(block, "reversed", rng)
    arms = [
        ("R row-level", splits_row),
        ("T0 ordered", t0),
        ("T1 embargo", t1),
        ("S shuffled", sh),
        ("M middle", mid),
        ("V reversed", rev),
    ]

    families = FAMILIES[:2] if args.smoke else FAMILIES
    params = {f: tuned_full_params(f) for f in families}

    # transforms and the neighbour audit depend only on the arm, so compute each once
    per_arm = {}
    for name, sp in arms:
        pipe = build_preprocessor("full", n_components=None, seed=SEED)
        Z = fit_transform_splits(pipe, X, sp)
        d = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Z["train"]).kneighbors(Z["test"])[0]
        floor = trivial_baselines(y[sp["train"]], y[sp["test"]])
        per_arm[name] = {
            "Z": Z,
            "splits": sp,
            "nn": float((d.ravel() < NN_RADIUS).mean()),
            "floor": floor["baseline_stratified_random"]["aggregate"]["macro_f1"]["mean"],
        }
        log.info(
            "arm %-12s nn<0.1 %5.1f%% floor %.4f",
            name,
            100 * per_arm[name]["nn"],
            per_arm[name]["floor"],
        )

    rows = []
    for fam in families:
        for name, _ in arms:
            a = per_arm[name]
            set_all_seeds(SEED)
            est = build(fam, params[fam], SEED)
            fit_model(fam, est, a["Z"]["train"], y[a["splits"]["train"]], fam in BALANCED_FIT)
            pred = est.predict(a["Z"]["test"])
            if len(np.unique(pred)) < 2:
                raise SystemExit(f"DEGENERATE PREDICTOR: {fam} on {name}")
            f1 = float(f1_score(y[a["splits"]["test"]], pred, average="macro"))
            rows.append(
                {"family": fam, "arm": name, "macro_f1": f1, "nn": a["nn"], "floor": a["floor"]}
            )
            log.info("%-7s %-12s macro-F1 %.4f (nn %4.1f%%)", fam, name, f1, 100 * a["nn"])

    if args.smoke:
        print("SMOKE OK", len(rows), "cells")
        return

    by_fam = {}
    for fam in families:
        r = [x for x in rows if x["family"] == fam]
        nn = np.array([x["nn"] for x in r])
        f1 = np.array([x["macro_f1"] for x in r])
        slope, intercept = np.polyfit(nn, f1, 1)
        pear = float(stats.pearsonr(nn, f1)[0])
        row_score = next(x["macro_f1"] for x in r if x["arm"] == "R row-level")
        shuf = next(x["macro_f1"] for x in r if x["arm"] == "S shuffled")
        t0s = next(x["macro_f1"] for x in r if x["arm"] == "T0 ordered")
        by_fam[fam] = {
            "pearson_r": pear,
            "slope": float(slope),
            "intercept": float(intercept),
            "row_level": row_score,
            "shuffled": shuf,
            "temporal_T0": t0s,
            "leak_gap_row_minus_T0": row_score - t0s,
            "negative_control_pass": bool(abs(shuf - row_score) <= 0.02),
        }
        log.info(
            "%-7s r %.4f slope %.4f row %.4f T0 %.4f gap %.4f shuffled-ctrl %s",
            fam,
            pear,
            slope,
            row_score,
            t0s,
            row_score - t0s,
            by_fam[fam]["negative_control_pass"],
        )

    # blocking controls
    lg = {x["arm"]: x["macro_f1"] for x in rows if x["family"] == "lgbm"}
    pos_dev = max(abs(lg[k] - v) for k, v in NTEMP_LGBM.items())
    neg_fail = [f for f, v in by_fam.items() if not v["negative_control_pass"]]
    if pos_dev > 1e-3 or neg_fail:
        raise SystemExit(
            f"CONTROL FAILED (run VOID): lgbm max deviation from N-TEMP {pos_dev:.5f}; "
            f"shuffled-control failures {neg_fail}"
        )

    slopes = np.array([by_fam[f]["slope"] for f in families])
    row_scores = np.array([by_fam[f]["row_level"] for f in families])
    rs = np.array([by_fam[f]["pearson_r"] for f in families])
    sp_slope_row = float(stats.spearmanr(slopes, row_scores)[0])
    ratio = float(slopes.max() / slopes.min()) if slopes.min() > 0 else float("inf")
    verdict = {
        "families": families,
        "per_family": by_fam,
        "min_pearson_r": float(rs.min()),
        "h1_law_universal": bool((rs >= 0.9).all()),
        "kill_fires": bool((rs < 0.7).sum() >= 2),
        "slope_ratio_max_over_min": ratio,
        "spearman_slope_vs_row_level": sp_slope_row,
        "h2_unification": bool(ratio >= 2.0 and sp_slope_row >= 0.8),
        "controls": {"lgbm_max_dev_vs_ntemp": pos_dev, "shuffled_failures": neg_fail},
    }
    cfg = {
        "variant": VARIANT,
        "families": families,
        "seed": SEED,
        "nn_radius": NN_RADIUS,
        "balanced_fit": sorted(BALANCED_FIT),
        "preregistration": "decision log 2026-09-02T04:02Z (N-SLOPE)",
    }
    with RunDir(f"leak_rate_families_{VARIANT}", config=cfg, seeds=[SEED]) as run:
        run.metrics = verdict
        run.write_json("results.json", {"cells": rows, "verdict": verdict})
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2, default=float))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
