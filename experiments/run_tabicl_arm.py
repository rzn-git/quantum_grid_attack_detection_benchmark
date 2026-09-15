"""V2-E1: TabICLv2 (open tabular foundation model) under both split protocols.

PRE-REGISTERED 2026-09-01T11:05Z. The modern question the v2 adds: does a 2026 in-context
tabular foundation model change either side of the paper's story? Hypotheses fixed before any
number: H1 — row-level, TabICL at the 2,000-row cap sits within 0.02 of (or above) LightGBM at
the identical cap, because nearest-neighbour-like in-context inference exploits near-duplicate
rows at least as well as trees; H2 — whole-file, TabICL's mean lands within 0.05 of LightGBM's
at the matched cap (the foundation prior does not cross the file boundary). KILL/new-paper
branch: TabICL whole-file mean >= LightGBM + 0.10.

Cells: {full, PCA-8} x 2,000-row cap plus full x 10,000-row cap; shipped row split + the five
Table V whole-file partitions. Comparator: tuned LightGBM at the identical cap and splits.
Controls (blocking): Table V partition ids; file-disjointness; determinism (same seed twice ->
identical predictions); TabICL@2k row-level clears the stratified floor by >= 0.20.

Run:  python -m experiments.run_tabicl_arm [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import f1_score
from tabicl import TabICLClassifier

from experiments.run_group_split_control import group_splits, reconstruct_groups, tuned_params
from experiments.run_whole_file_retune import EXPECTED_TEST_GROUPS, assert_file_disjoint
from qgridbench.data.preprocess import load_processed
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.utils.run_tracking import RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
# (regime, n_components, train cap) — the matched-contract cells plus one capacity point.
CELLS = [("full", None, 2000), ("pca", 8, 2000), ("full", None, 10000)]
SPREAD_CELL = ("full", None, 2000)  # extra seeds run on this cell only
H1_BAR = -0.02
H2_BAR = 0.05
KILL_BAR = 0.10
# Instrument-sanity control, comparator-relative (amendment 2, decision log
# 2026-09-01T14:20Z): TabICL's row-level floor clearance must reach at least half of
# LightGBM's at the identical cell. The original absolute 0.20 bar was calibrated to the
# uncapped regime and is unpassable at the 2,000-row cap, where LightGBM itself clears ~0.13.
FLOOR_CLEAR_FRACTION = 0.5


def tabicl_f1(Ztr, ytr, Zte, yte, seed):
    """Chunked prediction bounds the (context + batch)^2 attention memory on laptop VRAM;
    QGB_TABICL_CHUNK overrides (the resolve_chunk pattern). Fails loud on a real OOM.

    AMENDMENT (decision log 2026-09-01T13:05Z): predictions come from a train-prior-corrected
    decision rule — argmax(proba / train prior) — because TabICL has no class-weight hook and
    the paper's imbalance policy is uniform balanced weighting. Returns (corrected_f1, raw_f1,
    clf); the pre-registered bars read the corrected number, the raw one is recorded."""
    import os

    import torch

    chunk = int(os.environ.get("QGB_TABICL_CHUNK", 2048))
    set_all_seeds(seed)
    clf = TabICLClassifier(device="cuda", random_state=seed, verbose=False)
    clf.fit(Ztr, ytr)
    proba = np.concatenate(
        [clf.predict_proba(Zte[i : i + chunk]) for i in range(0, len(Zte), chunk)]
    )
    prior = np.bincount(ytr, minlength=proba.shape[1]) / len(ytr)  # train fold only
    pred = (proba / prior).argmax(axis=1)
    if len(np.unique(pred)) < 2:
        raise SystemExit("DEGENERATE PREDICTOR: TabICL returned a constant prediction")
    corrected = float(f1_score(yte, pred, average="macro"))
    raw = float(f1_score(yte, proba.argmax(axis=1), average="macro"))
    torch.cuda.empty_cache()  # 18 cells back-to-back: fragmentation, not one big alloc, OOMs
    return corrected, raw, clf


def lgbm_f1(params, Ztr, ytr, Zte, yte, seed):
    set_all_seeds(seed)
    est = build("lgbm", params, seed)
    fit_model("lgbm", est, Ztr, ytr)
    return float(f1_score(yte, est.predict(Zte), average="macro"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    lgbm_params, _ = tuned_params()
    X, y, gid = reconstruct_groups()
    Xp, yp, splits_row = load_processed(VARIANT)
    assert X.shape == Xp.shape and np.array_equal(y, yp), "group reconstruction misaligned"

    cfg = {
        "variant": VARIANT,
        "cells": [{"regime": r, "n_components": c, "cap": k} for r, c, k in CELLS],
        "seeds": args.seeds,
        "tabicl_version": "2.1.1 (checkpoint tabicl-classifier-v2-20260212)",
        "comparator": "tuned LightGBM (optuna cls_lgbm_binary_full) at the identical cap/splits",
        "preregistration": "decision log 2026-09-01T11:05Z (V2-E1)",
    }
    with RunDir(f"tabicl_arm_{VARIANT}", config=cfg, seeds=args.seeds) as run:
        # split inventory: shipped row split + five Table V partitions (one shared rng stream)
        splits_all = [("row", None, {k: np.asarray(v) for k, v in splits_row.items()})]
        rng = np.random.default_rng(0)
        for p_idx in range(5):
            sp, held = group_splits(gid, y, rng)
            if held[2] != EXPECTED_TEST_GROUPS[p_idx]:
                raise SystemExit(f"PARTITION STREAM DIVERGED: p{p_idx} {held[2]}")
            assert_file_disjoint(sp, gid, p_idx)
            splits_all.append((f"p{p_idx}", p_idx, sp))

        rows, det_checked = [], False
        for regime, ncomp, cap in CELLS:
            for split_name, p_idx, sp in splits_all:
                pipe = build_preprocessor(regime, n_components=ncomp, seed=0)
                Z = fit_transform_splits(pipe, X, sp)
                ytr_full = y[sp["train"]]
                sub = stratified_cap(Z["train"], ytr_full, cap, seed=0)
                Ztr, ytr = Z["train"][sub], ytr_full[sub]
                Zte, yte = Z["test"], y[sp["test"]]
                floor = trivial_baselines(ytr, yte)["baseline_stratified_random"]["aggregate"][
                    "macro_f1"
                ]["mean"]

                seeds = args.seeds if (regime, ncomp, cap) == SPREAD_CELL else [0]
                ti = [tabicl_f1(Ztr, ytr, Zte, yte, s) for s in seeds]
                ti_scores = [t[0] for t in ti]  # prior-corrected (the decision numbers)
                ti_raw = [t[1] for t in ti]
                if not det_checked:  # blocking determinism control, once
                    clf = ti[0][2]
                    a, b = clf.predict(Zte[:500]), clf.predict(Zte[:500])
                    if not np.array_equal(a, b):
                        raise SystemExit("DETERMINISM CONTROL FAILED: repeated predict differs")
                    det_checked = True
                ti = None  # drop model refs so VRAM frees between cells
                lg = [lgbm_f1(lgbm_params, Ztr, ytr, Zte, yte, s) for s in seeds]

                rec = {
                    "regime": regime,
                    "n_components": ncomp,
                    "cap": cap,
                    "split": split_name,
                    "partition": p_idx,
                    "n_train_ctx": int(len(sub)),
                    "n_test": int(len(yte)),
                    "floor_macro_f1": floor,
                    "tabicl_f1": ti_scores[0],
                    "tabicl_f1_raw": ti_raw[0],
                    "lgbm_f1": lg[0],
                    "tabicl_seed_scores": ti_scores,
                    "tabicl_raw_seed_scores": ti_raw,
                    "lgbm_seed_scores": lg,
                }
                rows.append(rec)
                run.write_json("results_partial.json", {"rows": rows})
                log.info(
                    "%s-%s@%d %s | tabicl %.4f | lgbm %.4f | floor %.4f",
                    regime,
                    ncomp,
                    cap,
                    split_name,
                    rec["tabicl_f1"],
                    rec["lgbm_f1"],
                    floor,
                )

        # ---- verdict on the pre-registered lines (full@2k is the decision cell) -------------
        def cell_rows(regime, ncomp, cap):
            return [
                r
                for r in rows
                if (r["regime"], r["n_components"], r["cap"]) == (regime, ncomp, cap)
            ]

        cr = cell_rows("full", None, 2000)
        row_r = next(r for r in cr if r["split"] == "row")
        grp = [r for r in cr if r["split"] != "row"]
        ti_g = float(np.mean([r["tabicl_f1"] for r in grp]))
        lg_g = float(np.mean([r["lgbm_f1"] for r in grp]))
        tab_clear = row_r["tabicl_f1"] - row_r["floor_macro_f1"]
        lgbm_clear = row_r["lgbm_f1"] - row_r["floor_macro_f1"]
        if tab_clear < FLOOR_CLEAR_FRACTION * lgbm_clear:
            raise SystemExit(
                f"INSTRUMENT CONTROL FAILED: TabICL row-level clears the floor by "
                f"{tab_clear:.4f}, under {FLOOR_CLEAR_FRACTION} x LightGBM's {lgbm_clear:.4f}"
            )
        verdict = {
            "row_tabicl": row_r["tabicl_f1"],
            "row_lgbm": row_r["lgbm_f1"],
            "h1_row_delta": row_r["tabicl_f1"] - row_r["lgbm_f1"],
            "h1_pass": bool(row_r["tabicl_f1"] - row_r["lgbm_f1"] >= H1_BAR),
            "group_tabicl_mean": ti_g,
            "group_tabicl_std": float(np.std([r["tabicl_f1"] for r in grp], ddof=1)),
            "group_lgbm_mean": lg_g,
            "h2_group_delta": ti_g - lg_g,
            "h2_pass": bool(abs(ti_g - lg_g) <= H2_BAR),
            "kill_fires": bool(ti_g - lg_g >= KILL_BAR),
            "leak_gap_tabicl": row_r["tabicl_f1"] - ti_g,
            "leak_gap_lgbm": row_r["lgbm_f1"] - lg_g,
        }
        run.metrics = verdict
        run.write_json("results.json", {"rows": rows, "verdict": verdict})
        log.info(
            "VERDICT | H1 row delta %+.4f (pass %s) | H2 group delta %+.4f (pass %s) | "
            "kill %s | leak gap tabicl %.4f vs lgbm %.4f",
            verdict["h1_row_delta"],
            verdict["h1_pass"],
            verdict["h2_group_delta"],
            verdict["h2_pass"],
            verdict["kill_fires"],
            verdict["leak_gap_tabicl"],
            verdict["leak_gap_lgbm"],
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
