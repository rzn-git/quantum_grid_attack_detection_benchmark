"""P25: place the benchmark's OWN split rule on the leakage axis next to ours.

THE GAP THIS CLOSES. Hink et al. (ISRCS 2014), the paper that introduced this dataset, splits
rows *within* each of the 15 source files: "we partitioned the dataset into 10 sets randomly
selecting instances from each category ... for each learner and each dataset". This study's
headline protocol pools rows across files; the group control holds whole files out. That is
three partition rules, and only two have ever been measured here. Quoting the pooled-row cost
as the cost of the benchmark's own rule would attribute a measurement to a rule it was not
taken under.

H1 (pre-registered, decision log 2026-08-20T15:49Z): within-file and pooled-row are ONE leakage
regime, because rows inside a source file are near-duplicates and any rule leaving a file on both
sides of the boundary leaks the same way. Predictions: P1 within-file level within 0.02 of the
shipped 0.9053; P2 nearest-neighbour leakage >= 0.35 (row-level is 0.458, group is 0.047-0.058);
P3 the matched contrast does not flip sign. Kill criterion: P1 failing means three distinct rungs
and the paper may not quote 0.31 for Hink's rule.

Only the partition rule varies. Transforms, tuned hyperparameters, ratios and seeds are the ones
`run_group_split_control` already uses, imported rather than reimplemented, so the row-level and
whole-file arms recomputed here double as a positive control: they must land on the archived
2026-08-16 values or the run is suspect and the new arm is not interpreted.

Run:  python -m experiments.run_split_protocol_ladder [--partitions 3]
"""

from __future__ import annotations

import argparse

import numpy as np

from experiments.run_group_split_control import (
    VARIANT,
    audit,
    group_splits,
    reconstruct_groups,
    score_level,
    score_matched,
    tuned_params,
)
from qgridbench.data.preprocess import load_processed
from qgridbench.utils.run_tracking import RunDir, get_logger

log = get_logger(__name__)

# Archived reference values the control arms must reproduce
# (run 20260816T023620Z_group_split_control_binary).
ARCHIVED_ROW_LEVEL = 0.9053
ARCHIVED_GROUP_LEVEL_RANGE = (0.5837, 0.6033)


def within_file_splits(gid, y, rng):
    """Hink et al.'s rule: 60/20/20 inside EVERY source file, stratified by class.

    Every file contributes rows to train, validation and test — the opposite of holding whole
    files out, and a different mechanism from pooling (which scatters a file's rows only in
    expectation). Stratifying by class within the file mirrors "randomly selecting instances
    from each category".
    """
    tr, va, te = [], [], []
    for g in np.unique(gid):
        for c in np.unique(y):
            idx = np.flatnonzero((gid == g) & (y == c))
            if idx.size == 0:
                continue
            rng.shuffle(idx)
            n_tr = int(round(0.6 * idx.size))
            n_va = int(round(0.2 * idx.size))
            # keep test non-empty whenever the cell has enough rows to spare one
            if idx.size >= 3 and n_tr + n_va >= idx.size:
                n_va = max(0, idx.size - n_tr - 1)
            tr.append(idx[:n_tr])
            va.append(idx[n_tr : n_tr + n_va])
            te.append(idx[n_tr + n_va :])
    return {
        "train": np.sort(np.concatenate(tr)),
        "val": np.sort(np.concatenate(va)),
        "test": np.sort(np.concatenate(te)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--partitions", type=int, default=3)
    args = ap.parse_args()

    lgbm_params, best = tuned_params()
    X, y, gid = reconstruct_groups()
    Xp, yp, splits_row = load_processed(VARIANT)
    assert X.shape == Xp.shape, f"group reconstruction misaligned: {X.shape} vs {Xp.shape}"
    assert np.array_equal(y, yp), "label vector diverged from the shipped artifact"

    cfg = {
        "variant": VARIANT,
        "rules": ["row (shipped, pooled)", "within-file (Hink et al. 2014)", "whole-file (group)"],
        "n_partitions": args.partitions,
        "qk_bw": best["qk_bw"],
        "qk_C": best["qk_C"],
        "preregistration": "decision log 2026-08-20T15:49Z (P25)",
    }
    with RunDir(f"split_protocol_ladder_{VARIANT}", config=cfg) as run:
        rows = []

        def evaluate(rule, partition, splits):
            r = {
                "rule": rule,
                "partition": partition,
                "n_train": int(len(splits["train"])),
                "n_test": int(len(splits["test"])),
                "level_full_lgbm": score_level(lgbm_params, splits, X, y),
                **{f"matched_{k}": v for k, v in score_matched(best, splits, X, y).items()},
                **audit(splits, X, y),
            }
            rows.append(r)
            run.write_json("results_partial.json", {"rows": rows})
            log.info(
                "%-28s p%s | level %.4f | matched cls %.4f q %.4f contrast %+.4f | nn<0.1 %.1f%%",
                rule,
                partition,
                r["level_full_lgbm"],
                r["matched_classical"],
                r["matched_quantum"],
                r["matched_contrast"],
                100 * r["nn_frac_within_0.1"],
            )
            return r

        # control arm 1: the shipped pooled-row split, which must reproduce the archive
        evaluate("row (shipped, pooled)", None, {k: np.asarray(v) for k, v in splits_row.items()})

        # the new arm: the benchmark's own rule
        rng = np.random.default_rng(0)
        for p in range(args.partitions):
            evaluate("within-file (Hink)", p, within_file_splits(gid, y, rng))

        # control arm 2: whole-file holdout, same rng stream as the group control
        rng_g = np.random.default_rng(0)
        for p in range(args.partitions):
            splits, _ = group_splits(gid, y, rng_g)
            evaluate("whole-file (group)", p, splits)

        def level(rule):
            return np.array([r["level_full_lgbm"] for r in rows if r["rule"] == rule])

        def nn(rule):
            return np.array([r["nn_frac_within_0.1"] for r in rows if r["rule"] == rule])

        row_level = level("row (shipped, pooled)")[0]
        wf, gp = level("within-file (Hink)"), level("whole-file (group)")
        summary = {
            "row_level": float(row_level),
            "within_file_mean": float(wf.mean()),
            "within_file_std": float(wf.std(ddof=1)) if len(wf) > 1 else 0.0,
            "group_mean": float(gp.mean()),
            "group_std": float(gp.std(ddof=1)) if len(gp) > 1 else 0.0,
            "gap_withinfile_minus_row": float(wf.mean() - row_level),
            "drop_withinfile_minus_group": float(wf.mean() - gp.mean()),
            "nn_row": float(nn("row (shipped, pooled)")[0]),
            "nn_within_file_mean": float(nn("within-file (Hink)").mean()),
            "nn_group_mean": float(nn("whole-file (group)").mean()),
            "control_row_delta_vs_archive": float(row_level - ARCHIVED_ROW_LEVEL),
            "control_group_in_archived_range": bool(
                ARCHIVED_GROUP_LEVEL_RANGE[0] - 0.02
                <= gp.mean()
                <= ARCHIVED_GROUP_LEVEL_RANGE[1] + 0.02
            ),
        }
        p1 = abs(summary["gap_withinfile_minus_row"]) <= 0.02
        p2 = summary["nn_within_file_mean"] >= 0.35
        summary["P1_level_within_0.02"] = bool(p1)
        summary["P2_nn_leakage_ge_0.35"] = bool(p2)
        summary["verdict"] = (
            "ONE REGIME — within-file and pooled-row are empirically the same rule on this data"
            if p1 and p2
            else "THREE RUNGS — within-file is its own regime and must be reported separately"
            if not p1
            else "INSTRUMENT PROBLEM — levels agree but the leakage audit does not"
        )
        run.metrics = summary
        run.write_json("results.json", {"rows": rows, "summary": summary})
        log.info("VERDICT: %s", summary["verdict"])
        log.info(
            "row %.4f | within-file %.4f+-%.4f | group %.4f+-%.4f | nn %.3f / %.3f / %.3f",
            summary["row_level"],
            summary["within_file_mean"],
            summary["within_file_std"],
            summary["group_mean"],
            summary["group_std"],
            summary["nn_row"],
            summary["nn_within_file_mean"],
            summary["nn_group_mean"],
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
