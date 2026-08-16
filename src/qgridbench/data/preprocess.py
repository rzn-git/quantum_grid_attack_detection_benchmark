"""Pool, clean, and split one dataset variant into model-ready arrays.

Pipeline (policies logged in experiment_decision_log.md):
  1. pool the 15 CSV groups (aggregation policy: "pooled")
  2. map string labels -> ints via configs/data.yaml (fail loud on unknowns)
  3. +/-inf -> NaN (imputation happens later, fit on train only — leakage rule)
  4. drop exact duplicate rows BEFORE splitting (prevents cross-split leakage)
  5. stratified 60/20/20 train/val/test split with the FIXED split seed;
     indices saved to data/processed/splits_<variant>.npz — all models use these.

No scaling/PCA here: every fitted transform lives in qgridbench.features and is
fit on the train split only at model-run time.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from qgridbench.data.validate import load_variant_frames
from qgridbench.utils.run_tracking import REPO_ROOT, get_logger, load_yaml

log = get_logger(__name__)


def preprocess_variant(variant: str) -> dict:
    cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    label_col = cfg["schema"]["label_column"]
    out_dir = REPO_ROOT / cfg["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled = pd.concat(load_variant_frames(variant), ignore_index=True)
    n_raw = len(pooled)

    # label mapping
    label_map = cfg["labels"].get(variant)
    if label_map is not None:
        unknown = set(pooled[label_col].unique()) - set(label_map)
        if unknown:
            raise ValueError(f"{variant}: unmapped label values {unknown}")
        y = pooled[label_col].map(label_map).to_numpy(dtype=np.int64)
    else:  # multiclass: raw integer scenario markers
        y = pooled[label_col].to_numpy(dtype=np.int64)

    X = pooled.drop(columns=[label_col]).to_numpy(dtype=np.float64)
    feature_names = [c for c in pooled.columns if c != label_col]

    # inf policy
    n_inf = int(np.isinf(X).sum())
    X[np.isinf(X)] = np.nan

    # dedupe BEFORE splitting (feature+label exact duplicates)
    df_all = pd.DataFrame(X)
    df_all["__y"] = y
    keep = ~df_all.duplicated()
    n_dupes = int((~keep).sum())
    X, y = X[keep.to_numpy()], y[keep.to_numpy()]

    # fixed stratified split
    ratios = cfg["preprocess"]["split"]["ratios"]
    seed = cfg["preprocess"]["split"]["seed"]
    idx = np.arange(len(y))
    idx_train, idx_tmp = train_test_split(idx, train_size=ratios[0], stratify=y, random_state=seed)
    rel_val = ratios[1] / (ratios[1] + ratios[2])
    idx_val, idx_test = train_test_split(
        idx_tmp, train_size=rel_val, stratify=y[idx_tmp], random_state=seed
    )

    np.savez_compressed(
        out_dir / f"{variant}.npz",
        X=X.astype(np.float32),
        y=y,
    )
    np.savez_compressed(
        out_dir / f"splits_{variant}.npz",
        train=np.sort(idx_train),
        val=np.sort(idx_val),
        test=np.sort(idx_test),
        split_seed=np.array([seed]),
    )
    (out_dir / f"meta_{variant}.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "n_raw_rows": n_raw,
                "n_inf_cells": n_inf,
                "n_duplicate_rows_dropped": n_dupes,
                "n_rows": int(len(y)),
                "n_features": X.shape[1],
                "feature_names": feature_names,
                "label_map": label_map,
                "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
                "split_sizes": {
                    "train": len(idx_train),
                    "val": len(idx_val),
                    "test": len(idx_test),
                },
                "split_seed": seed,
            },
            indent=2,
        )
    )
    summary = {
        "variant": variant,
        "rows": int(len(y)),
        "dupes_dropped": n_dupes,
        "inf_cells": n_inf,
        "train": len(idx_train),
        "val": len(idx_val),
        "test": len(idx_test),
    }
    log.info("%s", summary)
    return summary


def load_processed(variant: str):
    """Load (X, y, splits) for a variant. splits is a dict of sorted index arrays."""
    cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    proc = REPO_ROOT / cfg["paths"]["processed"]
    d = np.load(proc / f"{variant}.npz")
    s = np.load(proc / f"splits_{variant}.npz")
    return d["X"], d["y"], {k: s[k] for k in ("train", "val", "test")}


if __name__ == "__main__":
    import sys

    from qgridbench.utils.run_tracking import RunDir

    variants = sys.argv[1:] or ["binary", "triple"]
    with RunDir("preprocess", config={"variants": variants}) as run:
        results = {v: preprocess_variant(v) for v in variants}
        run.write_json("preprocess_summary.json", results)
        print(json.dumps(results, indent=2))
