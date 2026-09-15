"""N-TEMP: decompose the 0.311 split drop into temporal-neighbour leakage and session identity.

PRE-REGISTERED 2026-09-02T02:20Z (Tier A). N-ORD showed rows within a released file are
order-preserving session samples and that 99% of leaked test rows have the adjacent row as
their nearest training neighbour. The matching protocol -- a temporal split -- is absent
from the paper's ladder (row-level 0.905, within-file 0.906, whole-file 0.594). This runs
it.

Design input (measured before the design was fixed): scenario order is essentially fixed
across files, so a naive positional split would be a SCENARIO holdout. Blocks are therefore
contiguous (file, scenario) runs, and the split is positional WITHIN each block, so every
file and every scenario appears on both sides and only temporal adjacency is broken.
Scenario ids come from the multiclass release, which is row-aligned with the binary one.

Arms: T0 ordered | T1 embargoed (10 rows either side of each internal boundary) |
S shuffled (negative control: order destroyed inside each block, same geometry) |
R row-level (positive control: the shipped split).

  H1: T0 <= 0.75 (>= half the drop recovered).   KILL: T0 >= 0.87.
Controls (blocking): R within 0.01 of 0.9056; S within 0.02 of R.

Run:  python -m experiments.run_temporal_split [--smoke]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

from experiments.run_group_split_control import tuned_params
from qgridbench.data.preprocess import load_processed
from qgridbench.data.validate import load_variant_frames
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
EMBARGO = 10
NN_RADIUS = 0.1
SEED = 0
PUBLISHED_ROW_LGBM = 0.9056  # MODEL_CARD 10-seed mean
ARCHIVED_GROUP_LGBM = 0.594  # whole-file 5-partition mean


def load_blocks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicated rows in released order, with a block id per row = (file, scenario) run.

    Mirrors reconstruct_groups' steps exactly, then attaches the row-aligned multiclass
    scenario label so contiguous runs can be identified.
    """
    cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    label_col = cfg["schema"]["label_column"]
    frames = load_variant_frames(VARIANT)
    multi = load_variant_frames("multiclass")
    if [len(f) for f in frames] != [len(m) for m in multi]:
        raise SystemExit("binary and multiclass releases are not row-aligned")

    gid_raw = np.concatenate([np.full(len(f), i, dtype=np.int64) for i, f in enumerate(frames)])
    scen_raw = np.concatenate([m[label_col].astype(str).to_numpy() for m in multi])
    pooled = pd.concat(frames, ignore_index=True)
    y = pooled[label_col].map(cfg["labels"].get(VARIANT)).to_numpy(dtype=np.int64)
    X = pooled.drop(columns=[label_col]).to_numpy(dtype=np.float64)
    X[np.isinf(X)] = np.nan

    df = pd.DataFrame(X)
    df["__y"] = y
    keep = (~df.duplicated()).to_numpy()
    X, y, gid, scen = X[keep], y[keep], gid_raw[keep], scen_raw[keep]

    # contiguous (file, scenario) runs over the kept rows, in released order
    change = np.ones(len(gid), dtype=bool)
    change[1:] = (gid[1:] != gid[:-1]) | (scen[1:] != scen[:-1])
    block = np.cumsum(change) - 1
    return X, y, block


def positional_splits(block: np.ndarray, mode: str, rng: np.random.Generator) -> tuple[dict, int]:
    """60/20/20 by POSITION inside each block; 'shuffled' permutes order within a block.

    The embargo is per-block adaptive (amendment 2026-09-02T02:26Z): a fixed 10-row cut
    empties a split on small blocks, so e_b = min(EMBARGO, max(0, m_b - 5)) with m_b the
    block's smallest split. Returns the splits and how many blocks got the full embargo.
    """
    tr, va, te = [], [], []
    full_embargo = 0
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        if mode == "shuffled":
            idx = idx[rng.permutation(len(idx))]
        n = len(idx)
        i, j = int(TRAIN_FRAC * n), int((TRAIN_FRAC + VAL_FRAC) * n)
        a, c, d = idx[:i], idx[i:j], idx[j:]
        if mode == "middle":
            # test = interior 20%, sandwiched by training rows (adjacency restored)
            lo, hi = int(0.4 * n), int(0.6 * n)
            d = idx[lo:hi]
            rest = np.concatenate([idx[:lo], idx[hi:]])
            k = int(0.75 * len(rest))
            a, c = rest[:k], rest[k:]
        if mode == "reversed":
            # train = last 60%, test = first 20% (direction flipped)
            a, c, d = idx[n - i :], idx[j - i : n - i], idx[: j - i]
        if mode == "embargo":
            e = min(EMBARGO, max(0, min(len(a), len(c), len(d)) - 5))
            full_embargo += int(e == EMBARGO)
            a, c, d = a[: len(a) - e] if e else a, c[e:], d[e:]
        if min(len(a), len(c), len(d)) == 0:
            raise SystemExit(f"block {b} has an empty split at n={n} (mode {mode})")
        tr.append(a)
        va.append(c)
        te.append(d)
    splits = {k: np.sort(np.concatenate(v)) for k, v in (("train", tr), ("val", va), ("test", te))}
    return splits, full_embargo


def score_arm(name: str, splits: dict, X: np.ndarray, y: np.ndarray, params: dict) -> dict:
    pipe = build_preprocessor("full", n_components=None, seed=SEED)
    Z = fit_transform_splits(pipe, X, splits)
    set_all_seeds(SEED)
    est = build("lgbm", params, SEED)
    fit_model("lgbm", est, Z["train"], y[splits["train"]])
    pred = est.predict(Z["test"])
    if len(np.unique(pred)) < 2:
        raise SystemExit(f"DEGENERATE PREDICTOR in arm {name}")
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Z["train"])
    d = nn.kneighbors(Z["test"])[0].ravel()
    floor = trivial_baselines(y[splits["train"]], y[splits["test"]])
    r = {
        "arm": name,
        "n_train": int(len(splits["train"])),
        "n_test": int(len(splits["test"])),
        "macro_f1": float(f1_score(y[splits["test"]], pred, average="macro")),
        "nn_frac_within_0.1": float((d < NN_RADIUS).mean()),
        "floor_macro_f1": floor["baseline_stratified_random"]["aggregate"]["macro_f1"]["mean"],
    }
    log.info(
        "%-10s | macro-F1 %.4f | nn<0.1 %.1f%% | floor %.4f | n_tr %d n_te %d",
        name,
        r["macro_f1"],
        100 * r["nn_frac_within_0.1"],
        r["floor_macro_f1"],
        r["n_train"],
        r["n_test"],
    )
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    params, _ = tuned_params()
    X, y, block = load_blocks()
    Xp, yp, splits_row = load_processed(VARIANT)
    if X.shape != Xp.shape or not np.array_equal(y, yp):
        raise SystemExit("block reconstruction misaligned with the processed artifact")
    splits_row = {k: np.asarray(v) for k, v in splits_row.items()}
    n_blocks = int(block.max()) + 1
    sizes = np.bincount(block)
    log.info(
        "blocks: %d | rows/block min %d median %d max %d",
        n_blocks,
        sizes.min(),
        int(np.median(sizes)),
        sizes.max(),
    )

    if args.smoke:
        rng = np.random.default_rng(SEED)
        sp, _ = positional_splits(block, "ordered", rng)
        spe, fe = positional_splits(block, "embargo", rng)
        print(
            "SMOKE blocks", n_blocks, {k: len(v) for k, v in sp.items()}, "full-embargo blocks", fe
        )
        return

    rng = np.random.default_rng(SEED)
    t0, _ = positional_splits(block, "ordered", rng)
    t1, full_embargo = positional_splits(block, "embargo", rng)
    sh, _ = positional_splits(block, "shuffled", rng)
    mid, _ = positional_splits(block, "middle", rng)
    rev, _ = positional_splits(block, "reversed", rng)
    log.info("blocks receiving the full %d-row embargo: %d of %d", EMBARGO, full_embargo, n_blocks)
    arms = [
        ("R row-level", splits_row),
        ("T0 ordered", t0),
        ("T1 embargo", t1),
        ("S shuffled", sh),
        ("M middle", mid),
        ("V reversed", rev),
    ]
    cfg = {
        "variant": VARIANT,
        "n_blocks": n_blocks,
        "full_embargo_blocks": full_embargo,
        "fractions": [TRAIN_FRAC, VAL_FRAC],
        "embargo": EMBARGO,
        "nn_radius": NN_RADIUS,
        "seed": SEED,
        "lgbm_source": "optuna study cls_lgbm_binary_full (best_params)",
        "preregistration": "decision log 2026-09-02T02:20Z (N-TEMP)",
    }
    with RunDir(f"temporal_split_{VARIANT}", config=cfg, seeds=[SEED]) as run:
        rows = [score_arm(n, s, X, y, params) for n, s in arms]
        run.write_json("results_partial.json", {"arms": rows})
        by = {r["arm"]: r for r in rows}
        R, T0, T1, S, M, V = (
            by[k]["macro_f1"]
            for k in (
                "R row-level",
                "T0 ordered",
                "T1 embargo",
                "S shuffled",
                "M middle",
                "V reversed",
            )
        )

        ctrl_pos = abs(R - PUBLISHED_ROW_LGBM) <= 0.01
        ctrl_neg = abs(S - R) <= 0.02
        if not (ctrl_pos and ctrl_neg):
            raise SystemExit(
                f"CONTROL FAILED (run VOID): R {R:.4f} vs published {PUBLISHED_ROW_LGBM} "
                f"(pass={ctrl_pos}); S {S:.4f} vs R (pass={ctrl_neg})"
            )

        drop = PUBLISHED_ROW_LGBM - ARCHIVED_GROUP_LGBM
        verdict = {
            "row_level": R,
            "temporal_ordered": T0,
            "temporal_embargo": T1,
            "shuffled_control": S,
            "whole_file_reference": ARCHIVED_GROUP_LGBM,
            "total_drop": drop,
            "recovered_by_temporal": R - T0,
            "recovered_fraction": (R - T0) / drop,
            "residual_session_identity": T0 - ARCHIVED_GROUP_LGBM,
            "embargo_effect": T0 - T1,
            "h1_temporal_dominates": bool(T0 <= 0.75),
            "kill_fires": bool(T0 >= 0.87),
            "secondary_nn_below_0.25": bool(by["T0 ordered"]["nn_frac_within_0.1"] < 0.25),
            "middle_test": M,
            "reversed_test": V,
            "adjacency_account": bool(M >= 0.85 and abs(V - T0) <= 0.05),
            "drift_account_kill": bool(M <= 0.70),
            "controls": {"positive_R": R, "negative_S_minus_R": S - R, "pass": True},
        }
        run.metrics = verdict
        run.write_json("results.json", {"arms": rows, "verdict": verdict})
        log.info(
            "VERDICT | T0 %.4f recovers %.1f%% of the %.3f drop | residual %.4f "
            "| embargo %+.4f | H1=%s KILL=%s",
            T0,
            100 * verdict["recovered_fraction"],
            drop,
            verdict["residual_session_identity"],
            verdict["embargo_effect"],
            verdict["h1_temporal_dominates"],
            verdict["kill_fires"],
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
