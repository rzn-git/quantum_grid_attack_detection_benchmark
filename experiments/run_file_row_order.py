"""N-ORD: is row order within a released file temporal? (leak-mechanism probe)

PRE-REGISTERED 2026-09-01T23:10Z (Tier B). The README says the fifteen sets were
"randomly sampled at one percent"; it does not say the order was destroyed, and the first
rows of a file vary smoothly under one scenario label. If rows are order-preserving
thinned samples of one session, the near-duplicate leak is temporal autocorrelation.

  (a) per released file: mean L2 distance between rows at lag k over the mean distance of
      random within-file pairs (ratio r_k), features standardized per file;
  (b) multiclass files: fraction of consecutive rows sharing the scenario label against
      the shuffled expectation sum(p_k^2);
  (c) shipped row-level split: among test rows whose nearest training neighbour lies
      within 0.1 in the standardized full-feature space the paper audits, the fraction
      whose neighbour is in the SAME file within +/-10 positions of released order.

  H1 (temporal order): r_1 <= 0.5 in >= 13/15 files AND r_k non-decreasing in k in >= 13/15.
  KILL: r_1 >= 0.8 in >= 8 files, or lag growth in < 8 files.
Controls (blocking): a shuffled copy of each file gives r_1 within 1.0 +/- 0.05; an AR(1)
series thinned 1% in order gives r_1 < 0.5 with monotone lag growth.

Run:  python -m experiments.run_file_row_order [--smoke]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.neighbors import NearestNeighbors

from experiments.run_group_split_control import reconstruct_groups
from qgridbench.data.preprocess import load_processed
from qgridbench.data.validate import load_variant_frames
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml

log = get_logger(__name__)

VARIANT = "binary"
LAGS = [1, 2, 5, 10, 50, 100]
N_RANDOM_PAIRS = 20_000
NEIGHBOUR_WINDOW = 10
NN_RADIUS = 0.1
AR_PHI, AR_N, AR_THIN = 0.999, 500_000, 100  # consecutive thinned correlation phi^100 = 0.905


def standardize_file(F: np.ndarray) -> np.ndarray:
    F = F.astype(np.float64, copy=True)
    F[np.isinf(F)] = np.nan
    med = np.nanmedian(F, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    rr, cc = np.where(np.isnan(F))
    F[rr, cc] = med[cc]
    sd = F.std(axis=0)
    sd[sd == 0] = 1.0
    return (F - F.mean(axis=0)) / sd


def lag_ratios(Z: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    n = len(Z)
    i = rng.integers(0, n, N_RANDOM_PAIRS)
    j = rng.integers(0, n, N_RANDOM_PAIRS)
    m = i != j
    rand = float(np.linalg.norm(Z[i[m]] - Z[j[m]], axis=1).mean())
    return {
        str(k): float(np.linalg.norm(Z[k:] - Z[:-k], axis=1).mean() / rand) for k in LAGS if k < n
    }


def is_nondecreasing(r: dict[str, float], tol: float = 0.01) -> bool:
    v = [r[str(k)] for k in LAGS if str(k) in r]
    return all(b >= a - tol for a, b in zip(v, v[1:]))


def label_runs(labels: np.ndarray) -> dict[str, float]:
    same = float((labels[1:] == labels[:-1]).mean())
    p = pd.Series(labels).value_counts(normalize=True).to_numpy()
    return {"consecutive_same": same, "shuffled_expectation": float((p**2).sum())}


def reconstruct_positions() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The exact steps of reconstruct_groups, also returning each kept row's file position."""
    cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    label_col = cfg["schema"]["label_column"]
    frames = load_variant_frames(VARIANT)
    gid_raw = np.concatenate([np.full(len(f), i, dtype=np.int64) for i, f in enumerate(frames)])
    pos_raw = np.concatenate([np.arange(len(f), dtype=np.int64) for f in frames])
    pooled = pd.concat(frames, ignore_index=True)
    y = pooled[label_col].map(cfg["labels"].get(VARIANT)).to_numpy(dtype=np.int64)
    X = pooled.drop(columns=[label_col]).to_numpy(dtype=np.float64)
    X[np.isinf(X)] = np.nan
    df = pd.DataFrame(X)
    df["__y"] = y
    keep = (~df.duplicated()).to_numpy()
    return X[keep], y[keep], gid_raw[keep], pos_raw[keep]


def positive_control(rng: np.random.Generator) -> dict[str, float]:
    e = rng.normal(size=(AR_N, 4))
    x = lfilter([1.0], [1.0, -AR_PHI], e, axis=0)[::AR_THIN]
    return lag_ratios(standardize_file(x), rng)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    label_col = load_yaml(REPO_ROOT / "configs" / "data.yaml")["schema"]["label_column"]

    frames = load_variant_frames(VARIANT)
    if args.smoke:
        frames = frames[:2]
    per_file, neg = [], []
    for f in frames:
        Z = standardize_file(f.drop(columns=[label_col]).to_numpy(dtype=np.float64))
        r = lag_ratios(Z, rng)
        r_sh = lag_ratios(Z[rng.permutation(len(Z))], rng)
        per_file.append(
            {
                "file": f.attrs["source_file"],
                "n": len(Z),
                "r": r,
                "nondecreasing": is_nondecreasing(r),
            }
        )
        neg.append(r_sh["1"])
        log.info(
            "%s n=%d r1=%.3f r100=%.3f mono=%s shuffled r1=%.3f",
            f.attrs["source_file"],
            len(Z),
            r["1"],
            r.get("100", float("nan")),
            is_nondecreasing(r),
            r_sh["1"],
        )

    runs = []
    for f in load_variant_frames("multiclass")[: (2 if args.smoke else None)]:
        runs.append(
            {"file": f.attrs["source_file"], **label_runs(f[label_col].astype(str).to_numpy())}
        )

    pos_ctrl = positive_control(rng)
    controls = {
        "negative_shuffled_r1_max_abs_dev": float(max(abs(v - 1.0) for v in neg)),
        "negative_pass": bool(all(abs(v - 1.0) <= 0.05 for v in neg)),
        "positive_ar1": pos_ctrl,
        "positive_pass": bool(pos_ctrl["1"] < 0.5 and is_nondecreasing(pos_ctrl)),
    }
    if args.smoke:
        print("SMOKE", per_file[0]["r"], runs[0], controls)
        return
    if not (controls["negative_pass"] and controls["positive_pass"]):
        raise SystemExit(f"CONTROL FAILED: {controls}")

    # (c) neighbour-lag under the shipped row-level split
    X, y, gid, pos = reconstruct_positions()
    Xg, _, gg = reconstruct_groups()
    assert X.shape == Xg.shape and np.array_equal(gid, gg), "position reconstruction diverged"
    Xp, yp, splits = load_processed(VARIANT)
    assert X.shape == Xp.shape and np.array_equal(y, yp), "processed artifact misaligned"
    splits = {k: np.asarray(v) for k, v in splits.items()}
    Z = fit_transform_splits(build_preprocessor("full", n_components=None, seed=0), X, splits)
    d, idx = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Z["train"]).kneighbors(Z["test"])
    d, idx = d.ravel(), idx.ravel()
    tr, te = splits["train"][idx], splits["test"]
    leak = d < NN_RADIUS
    same_file = gid[tr] == gid[te]
    gap = np.abs(pos[tr] - pos[te])
    near = same_file & (gap <= NEIGHBOUR_WINDOW)
    neighbour = {
        "n_test": int(len(te)),
        "leak_frac": float(leak.mean()),
        "leak_same_file_frac": float(same_file[leak].mean()),
        "leak_same_file_within_window_frac": float(near[leak].mean()),
        "leak_same_file_median_gap": float(np.median(gap[leak & same_file])),
        "nonleak_same_file_within_window_frac": float(near[~leak].mean()),
    }

    r1 = np.array([p["r"]["1"] for p in per_file])
    mono = np.array([p["nondecreasing"] for p in per_file])
    verdict = {
        "n_files": len(per_file),
        "files_r1_le_0.5": int((r1 <= 0.5).sum()),
        "files_r1_ge_0.8": int((r1 >= 0.8).sum()),
        "files_lag_nondecreasing": int(mono.sum()),
        "h1_temporal_order": bool((r1 <= 0.5).sum() >= 13 and mono.sum() >= 13),
        "kill_fires": bool((r1 >= 0.8).sum() >= 8 or mono.sum() < 8),
        "neighbour_secondary_pass": bool(neighbour["leak_same_file_within_window_frac"] >= 0.5),
        "controls": controls,
        "neighbour": neighbour,
    }
    cfg = {
        "variant": VARIANT,
        "lags": LAGS,
        "n_random_pairs": N_RANDOM_PAIRS,
        "window": NEIGHBOUR_WINDOW,
        "nn_radius": NN_RADIUS,
        "ar1": {"phi": AR_PHI, "n": AR_N, "thin": AR_THIN},
        "preregistration": "decision log 2026-09-01T23:10Z (N-ORD)",
    }
    with RunDir(f"file_row_order_{VARIANT}", config=cfg, seeds=[0]) as run:
        run.metrics = verdict
        run.write_json(
            "results.json", {"per_file": per_file, "label_runs": runs, "verdict": verdict}
        )
        log.info("VERDICT %s", verdict)
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
