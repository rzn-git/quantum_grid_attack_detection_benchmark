"""R5-G1 option B, boring-cause check: where the gas pressure signal restored by the signed log
comes from (decision log 2026-11-19T11:40Z; the dataset page warns that an earlier release let
pressure identify attacks through an operating-point pattern).

Reports, on the shipped split of `gas_binary_slog` (seed 0, LightGBM with balanced class weights,
200 trees, the same model for every slice):
  rows       per category (normal, NMRI, CMRI, MSCI, MPCI, MFCI, DoS, Recon): rows carrying a
             pressure value, in the raw capture and after deduplication, and the share inside
             the normal class's training range
  slices     macro-F1 of pressure alone, of the other fifteen features and of all sixteen, on all
             rows, on ordinary-magnitude rows (|pressure| <= 1e3 or missing) and on rows whose
             pressure lies inside the normal training range
  in_range   per category, the pressure-only detector's recall (specificity for normal) on
             in-range test rows

Run:  uv run python -m experiments.run_gas_pressure_signal --threads 4
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

from qgridbench.data.gas import gas_config, gas_features, load_aux, load_gas_frame
from qgridbench.data.preprocess import load_processed
from qgridbench.utils.run_tracking import RunDir

VARIANT, BASE = "gas_binary_slog", "gas_binary"
SEED = 0
ORDINARY_MAX = 1e3
CATEGORIES = {0: "normal", 1: "NMRI", 2: "CMRI", 3: "MSCI", 4: "MPCI", 5: "MFCI", 6: "DoS",
              7: "Recon"}  # fmt: skip


def detector(threads: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=200, random_state=SEED, n_jobs=threads, verbose=-1, class_weight="balanced"
    )


def fit_score(X, y, sp, cols, rows, threads) -> dict:
    tr, te = sp["train"][rows[sp["train"]]], sp["test"][rows[sp["test"]]]
    m = detector(threads).fit(X[tr][:, cols], y[tr])
    pred = m.predict(X[te][:, cols])
    return {
        "macro_f1": float(f1_score(y[te], pred, average="macro")),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "attack_share_test": float(y[te].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    cfg = gas_config()
    feats = gas_features(cfg)
    j = feats.index("pressure")
    X, y, sp = load_processed(VARIANT)
    p = load_processed(BASE)[0][:, j].astype(np.float64)  # raw pressure, before the log
    cat = load_aux(VARIANT)["categorized"]
    raw = load_gas_frame(cfg)
    raw_has = raw["pressure"].notna().to_numpy()
    raw_cat = raw["categorized"].to_numpy(dtype=np.int64)

    tr0 = sp["train"][(y[sp["train"]] == 0) & ~np.isnan(p[sp["train"]])]
    lo, hi = float(p[tr0].min()), float(p[tr0].max())
    has = ~np.isnan(p)
    in_range = has & (p >= lo) & (p <= hi)
    ordinary = ~has | (np.abs(np.where(has, p, 0.0)) <= ORDINARY_MAX)
    rows = {}
    for c, name in CATEGORIES.items():
        sel = cat == c
        rows[name] = {
            "raw_rows": int((raw_cat == c).sum()),
            "raw_rows_with_pressure": int(((raw_cat == c) & raw_has).sum()),
            "rows": int(sel.sum()),
            "rows_with_pressure": int((sel & has).sum()),
            "in_normal_range_share": float(in_range[sel].mean()),
            "rows_with_pressure_index": np.flatnonzero(sel & has).tolist() if c == 7 else None,
        }

    others = [i for i in range(len(feats)) if i != j]
    every = list(range(len(feats)))
    slices = {}
    for slice_name, mask in (
        ("all_rows", np.ones(len(y), bool)),
        ("ordinary_rows", ordinary),
        ("in_range_rows", in_range),
    ):
        slices[slice_name] = {
            "pressure_only": fit_score(X, y, sp, [j], mask, args.threads),
            "without_pressure": fit_score(X, y, sp, others, mask, args.threads),
            "all_features": fit_score(X, y, sp, every, mask, args.threads),
        }

    tr, te = sp["train"][in_range[sp["train"]]], sp["test"][in_range[sp["test"]]]
    pred = detector(args.threads).fit(X[tr][:, [j]], y[tr]).predict(X[te][:, [j]])
    per_cat = {}
    for c, name in CATEGORIES.items():
        sel = cat[te] == c
        if sel.sum():
            per_cat[name] = {
                "n_test": int(sel.sum()),
                ("specificity" if c == 0 else "recall"): float(
                    (pred[sel] == (0 if c == 0 else 1)).mean()
                ),
                "pressure_median": float(np.median(p[te][sel])),
            }

    cfg_out = {**vars(args), "variant": VARIANT, "seed": SEED, "normal_train_range": [lo, hi],
               "ordinary_max": ORDINARY_MAX}  # fmt: skip
    with RunDir(f"gas_pressure_signal_{VARIANT}", config=cfg_out, seeds=[SEED]) as run:
        out = {"rows": rows, "slices": slices, "in_range_per_category": per_cat}
        run.write_json("results.json", out)
        run.metrics.update(
            {
                "recon_rows_with_pressure": rows["Recon"]["rows_with_pressure"],
                "pressure_only_all_rows": slices["all_rows"]["pressure_only"]["macro_f1"],
                "pressure_only_in_range": slices["in_range_rows"]["pressure_only"]["macro_f1"],
            }
        )
        print(json.dumps({"rows": rows, "in_range_per_category": per_cat}, indent=1))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
