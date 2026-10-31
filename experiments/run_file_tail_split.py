"""N-TAIL: does an order-preserving FILE-TAIL split (Mamodiya et al.'s protocol) leak?

PRE-REGISTERED 2026-10-31T18:12Z (Tier A). Mamodiya et al. keep each file's record order and
split 60% train / 15% validation / 10% calibration / 15% test, reporting three-class
macro-F1 0.9635; the paper's order-based temporal split scores 0.6005 binary. The file-tail
protocol leaves a 25%-of-file gap between the last training row and the first test row, far
beyond the ~9 released rows in which the measured leak decays.

Arms, per variant (binary, triple):
  F0 file-tail, released order | F1 file-tail, rows shuffled within each file first |
  R row-level (positive control) | T0 temporal within (file, scenario) block (instrument).

  H1: F0 binary <= 0.65.   KILL: F0 binary >= 0.85.
  H2: F1 triple >= 0.90 and within 0.03 of R triple.
Controls (blocking): R binary within 0.01 of 0.9056; T0 binary within 0.002 of 0.6005.

Run:  python -m experiments.run_file_tail_split [--smoke]
"""

from __future__ import annotations

import argparse

import numpy as np
import optuna
import pandas as pd

from experiments.run_temporal_split import positional_splits, score_arm
from qgridbench.data.preprocess import load_processed
from qgridbench.data.validate import load_variant_frames
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml

log = get_logger(__name__)

VARIANTS = ("binary", "triple")
TRAIN_END, TEST_START = 0.60, 0.85  # 60% train | 15% val + 10% calibration dropped | 15% test
SEED = 0
PUBLISHED_ROW_LGBM = 0.9056  # MODEL_CARD 10-seed mean, binary
ARCHIVED_T0 = 0.6005  # 20260902T022541Z_temporal_split_binary, arm "T0 ordered"


def load_rows(variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicated rows in released order with file id, scenario id and (file, scenario) block.

    Same steps as run_temporal_split.load_blocks, parameterised by variant; the multiclass
    release is row-aligned with both binary and triple, so scenario ids come from it.
    """
    cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    label_col = cfg["schema"]["label_column"]
    frames = load_variant_frames(variant)
    multi = load_variant_frames("multiclass")
    if [len(f) for f in frames] != [len(m) for m in multi]:
        raise SystemExit(f"{variant} and multiclass releases are not row-aligned")

    gid_raw = np.concatenate([np.full(len(f), i, dtype=np.int64) for i, f in enumerate(frames)])
    scen_raw = np.concatenate([m[label_col].astype(str).to_numpy() for m in multi])
    pooled = pd.concat(frames, ignore_index=True)
    y = pooled[label_col].map(cfg["labels"].get(variant)).to_numpy()
    if pd.isna(y).any():
        raise SystemExit(f"unmapped labels in {variant}")
    y = y.astype(np.int64)
    X = pooled.drop(columns=[label_col]).to_numpy(dtype=np.float64)
    X[np.isinf(X)] = np.nan

    df = pd.DataFrame(X)
    df["__y"] = y
    keep = (~df.duplicated()).to_numpy()
    X, y, gid, scen = X[keep], y[keep], gid_raw[keep], scen_raw[keep]

    change = np.ones(len(gid), dtype=bool)
    change[1:] = (gid[1:] != gid[:-1]) | (scen[1:] != scen[:-1])
    block = np.cumsum(change) - 1
    return X, y, gid, scen, block


def file_tail_splits(gid: np.ndarray, shuffle: bool, rng: np.random.Generator) -> dict:
    """Per file: first 60% train, next 25% dropped (passed as 'val', never fit), last 15% test."""
    tr, va, te = [], [], []
    for g in np.unique(gid):
        idx = np.flatnonzero(gid == g)
        if shuffle:
            idx = idx[rng.permutation(len(idx))]
        n = len(idx)
        i, j = int(TRAIN_END * n), int(TEST_START * n)
        tr.append(idx[:i])
        va.append(idx[i:j])
        te.append(idx[j:])
    return {k: np.sort(np.concatenate(v)) for k, v in (("train", tr), ("val", va), ("test", te))}


def scenario_holdout_share(splits: dict, gid: np.ndarray, scen: np.ndarray) -> dict:
    """Share of test rows whose scenario has no training row in their file / in any file."""
    tr, te = splits["train"], splits["test"]
    seen_file = set(zip(gid[tr].tolist(), scen[tr].tolist(), strict=True))
    seen_any = set(scen[tr].tolist())
    unseen_file = np.array(
        [(g, s) not in seen_file for g, s in zip(gid[te], scen[te], strict=True)]
    )
    unseen_any = np.array([s not in seen_any for s in scen[te]])
    return {
        "test_unseen_scenario_in_file": float(unseen_file.mean()),
        "test_unseen_scenario_anywhere": float(unseen_any.mean()),
        "test_n_scenarios": int(len(np.unique(scen[te]))),
        "train_n_scenarios": int(len(seen_any)),
    }


def tuned_lgbm(variant: str) -> dict:
    name = f"cls_lgbm_{variant}_full"
    db = (REPO_ROOT / "results" / f"optuna_{name}.db").as_posix()
    return optuna.load_study(study_name=name, storage=f"sqlite:///{db}").best_params


def run_variant(variant: str, smoke: bool) -> None:
    X, y, gid, scen, block = load_rows(variant)
    Xp, yp, splits_row = load_processed(variant)
    if X.shape != Xp.shape or not np.array_equal(y, yp):
        raise SystemExit(f"{variant}: reconstruction misaligned with the processed artifact")
    splits_row = {k: np.asarray(v) for k, v in splits_row.items()}

    rng = np.random.default_rng(SEED)
    f0 = file_tail_splits(gid, shuffle=False, rng=rng)
    f1 = file_tail_splits(gid, shuffle=True, rng=rng)
    t0, _ = positional_splits(block, "ordered", np.random.default_rng(SEED))
    comp = {
        "F0": scenario_holdout_share(f0, gid, scen),
        "F1": scenario_holdout_share(f1, gid, scen),
    }
    log.info("%s | F0 test composition %s", variant, comp["F0"])
    if smoke:
        print("SMOKE", variant, {k: len(v) for k, v in f0.items()}, comp)
        return

    params = tuned_lgbm(variant)
    arms = [
        ("R row-level", splits_row),
        ("T0 temporal", t0),
        ("F0 file-tail", f0),
        ("F1 tail-shuffled", f1),
    ]
    cfg = {
        "variant": variant,
        "train_end": TRAIN_END,
        "test_start": TEST_START,
        "seed": SEED,
        "lgbm_source": f"optuna study cls_lgbm_{variant}_full (best_params)",
        "preregistration": "decision log 2026-10-31T18:12Z (N-TAIL)",
    }
    with RunDir(f"file_tail_split_{variant}", config=cfg, seeds=[SEED]) as run:
        rows = []
        for name, sp in arms:
            rows.append(score_arm(name, sp, X, y, params))
            run.write_json("results_partial.json", {"arms": rows})
        by = {r["arm"]: r["macro_f1"] for r in rows}
        metrics = {
            "row_level": by["R row-level"],
            "temporal_block": by["T0 temporal"],
            "file_tail": by["F0 file-tail"],
            "file_tail_shuffled": by["F1 tail-shuffled"],
            "composition": comp,
        }
        if variant == "binary":
            ctrl_r = abs(by["R row-level"] - PUBLISHED_ROW_LGBM) <= 0.01
            ctrl_t = abs(by["T0 temporal"] - ARCHIVED_T0) <= 0.002
            if not (ctrl_r and ctrl_t):
                raise SystemExit(
                    f"CONTROL FAILED (run VOID): R {by['R row-level']:.4f} (pass={ctrl_r}); "
                    f"T0 {by['T0 temporal']:.4f} vs {ARCHIVED_T0} (pass={ctrl_t})"
                )
            metrics["h1_law_holds"] = bool(by["F0 file-tail"] <= 0.65)
            metrics["kill_fires"] = bool(by["F0 file-tail"] >= 0.85)
        else:
            metrics["h2_shuffled_copy"] = bool(
                by["F1 tail-shuffled"] >= 0.90
                and abs(by["F1 tail-shuffled"] - by["R row-level"]) <= 0.03
            )
        run.metrics = metrics
        run.write_json("results.json", {"arms": rows, "verdict": metrics})
        log.info("%s VERDICT %s", variant, {k: v for k, v in metrics.items() if k != "composition"})
        print(f"-> {run.path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    for v in VARIANTS:
        run_variant(v, args.smoke)


if __name__ == "__main__":
    main()
