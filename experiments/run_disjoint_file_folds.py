"""Disjoint five-fold over source files for the split table (E-DF).

Table I's whole-file partitions are drawn at random per partition, so they share test files.
Here the 15 files are permuted once (seed 0) into 5 folds of 3: fold k is test, fold k+1 (mod 5)
validation, the other 9 training, so every file is a test file exactly once. Each fold is scored
with `run_group_split_control`'s own functions (full-feature LightGBM level, the matched PCA-8
XGBoost-vs-ZZ-QSVM contrast on the identical 2,000 rows, the stratified-random floor and the
nearest-neighbour leakage audit). Pre-registration: decision log, "PRE-REGISTRATION ... E-DF".

Anchor (blocking): Table I's partition 0, re-scored by the same code, reproduces its level within
1e-6.

Usage: uv run python -m experiments.run_disjoint_file_folds --threads 8
"""

import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=8)
THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_var] = str(THREADS)

import json  # noqa: E402

import numpy as np  # noqa: E402

from experiments.run_group_split_control import (  # noqa: E402
    VARIANT,
    audit,
    group_splits,
    reconstruct_groups,
    score_level,
    score_matched,
    tuned_params,
)
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger  # noqa: E402

log = get_logger(__name__)

TABLE_I_RUN = "20260816T023620Z_group_split_control_binary"
ANCHOR_TOL = 1e-6
N_FOLDS = 5
FOLD_SEED = 0
LEVEL_HYP_TOL, LEVEL_KILL_TOL = 0.03, 0.05
CONTRAST_HYP_TOL = 0.03


def table_i_rows() -> list[dict]:
    path = REPO_ROOT / "results" / "runs" / TABLE_I_RUN / "results.json"
    d = json.loads(path.read_text())
    rows = d.get("rows") or d.get("results") or d
    return [r for r in rows if r["split"] == "group"]


def disjoint_folds(gid: np.ndarray) -> list[dict]:
    files = np.unique(gid)
    if len(files) % N_FOLDS:
        raise ValueError(f"{len(files)} files do not split into {N_FOLDS} equal folds")
    folds = np.random.default_rng(FOLD_SEED).permutation(files).reshape(N_FOLDS, -1)
    out = []
    for k in range(N_FOLDS):
        te, va = folds[k], folds[(k + 1) % N_FOLDS]
        tr = np.setdiff1d(files, np.concatenate([te, va]))
        idx = np.arange(len(gid))
        out.append(
            {
                "fold": k,
                "files": {
                    "train": tr.tolist(),
                    "val": sorted(va.tolist()),
                    "test": sorted(te.tolist()),
                },
                "splits": {
                    "train": idx[np.isin(gid, tr)],
                    "val": idx[np.isin(gid, va)],
                    "test": idx[np.isin(gid, te)],
                },
            }
        )
    tested = np.concatenate([f["files"]["test"] for f in out])
    if sorted(tested.tolist()) != sorted(files.tolist()):
        raise AssertionError("a file is not tested exactly once")
    return out


def mean_std(v):
    return {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=THREADS)
    args = ap.parse_args()

    lgbm_params, best = tuned_params()
    X, y, gid = reconstruct_groups()
    Xp, yp, _ = load_processed(VARIANT)
    assert X.shape == Xp.shape and np.array_equal(y, yp), "group reconstruction misaligned"
    ref = table_i_rows()

    config = {
        "threads": args.threads,
        "n_folds": N_FOLDS,
        "fold_seed": FOLD_SEED,
        "table_i_artifact": TABLE_I_RUN,
        "qk_bw": best["qk_bw"],
        "qk_C": best["qk_C"],
    }
    with RunDir(f"disjoint_file_folds_{VARIANT}", config=config) as run:
        sp0, held0 = group_splits(gid, y, np.random.default_rng(0))
        a0 = score_level(lgbm_params, sp0, X, y)
        ref0 = next(r for r in ref if r["partition"] == 0)["level_full_lgbm"]
        anchor = {"ours": a0, "table_i": ref0, "abs_diff": abs(a0 - ref0), "tol": ANCHOR_TOL}
        anchor["passed"] = anchor["abs_diff"] <= ANCHOR_TOL
        run.write_json("anchor.json", anchor)
        log.info("anchor partition 0: %.10f vs %.10f (diff %.2e)", a0, ref0, anchor["abs_diff"])
        if not anchor["passed"]:
            raise AssertionError(f"anchor failed: {anchor}")

        rows = []
        for f in disjoint_folds(gid):
            sp = f["splits"]
            r = {
                "fold": f["fold"],
                "files": f["files"],
                "n_test": int(len(sp["test"])),
                "test_positive_rate": float(y[sp["test"]].mean()),
                "level_full_lgbm": score_level(lgbm_params, sp, X, y),
                **{f"matched_{k}": v for k, v in score_matched(best, sp, X, y).items()},
                **audit(sp, X, y),
            }
            rows.append(r)
            run.write_json("results_partial.json", {"rows": rows})
            log.info(
                "fold %d test %s | level %.4f | matched cls %.4f q %.4f contrast %+.4f"
                " | floor %.4f",
                f["fold"],
                f["files"]["test"],
                r["level_full_lgbm"],
                r["matched_classical"],
                r["matched_quantum"],
                r["matched_contrast"],
                r["floor_macro_f1"],
            )

        level = mean_std([r["level_full_lgbm"] for r in rows])
        contrast = mean_std([r["matched_contrast"] for r in rows])
        ref_level = mean_std([r["level_full_lgbm"] for r in ref])
        ref_contrast = mean_std([r["matched_contrast"] for r in ref])
        d_level = level["mean"] - ref_level["mean"]
        if abs(d_level) > LEVEL_KILL_TOL:
            branch = "KILL"
        elif (
            abs(d_level) <= LEVEL_HYP_TOL
            and abs(contrast["mean"] - ref_contrast["mean"]) <= CONTRAST_HYP_TOL
        ):
            branch = "HYPOTHESIS"
        else:
            branch = "INTERMEDIATE"
        summary = {
            "disjoint": {"level": level, "contrast": contrast},
            "table_i": {"level": ref_level, "contrast": ref_contrast},
            "level_minus_table_i": d_level,
            "contrast_minus_table_i": contrast["mean"] - ref_contrast["mean"],
            "contrast_sign_counts": {
                "positive": int(sum(r["matched_contrast"] > 0 for r in rows)),
                "negative": int(sum(r["matched_contrast"] < 0 for r in rows)),
            },
            "floor": mean_std([r["floor_macro_f1"] for r in rows]),
            "branch": branch,
        }
        run.write_json("results.json", {"anchor": anchor, "rows": rows, "summary": summary})
        run.metrics.update(
            {"level_mean": level["mean"], "contrast_mean": contrast["mean"], "branch": branch}
        )
        log.info("SUMMARY %s", json.dumps(summary, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
