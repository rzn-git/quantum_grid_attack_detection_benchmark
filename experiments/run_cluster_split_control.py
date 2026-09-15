"""E3 — near-duplicate cluster split: decompose the 0.311 into leakage vs distribution shift.

PRE-REGISTRATION (pass-4 workplan item E3; frozen before the run).
The paper's headline split effect (row-level 0.905 -> whole-file 0.594) is an UNDECOMPOSED
mixture: holding whole files out removes near-duplicate leakage AND introduces cross-file
distribution shift. A critic attributes it to shift and re-reads the row-level number as a
valid in-distribution score. This run separates them by removing the leakage WITHOUT
changing the population: split by near-duplicate CLUSTER, so near-copies cannot straddle
the boundary while every file still contributes to both sides.

DECISION RULE on s = mean test macro-F1 (anchors: row-level 0.905, whole-file 0.594):
  s <= 0.70            -> leakage-dominated; the "memorization" framing is confirmed
  s >= 0.85            -> shift-dominated; the framing needs rewording (v2 row)
  0.70 < s < 0.85      -> mixed; leakage share = (0.905 - s) / (0.905 - 0.594)
IDENTIFIABILITY KILL CRITERION: if >50% of pooled rows sit in clusters confined to a single
source file AND the cluster partition reproduces the file partition (adjusted Rand index vs
file ids > 0.9), the decomposition is UNIDENTIFIABLE on this data — that is the finding, and
the current paper wording stands as the honest maximum.

CONTROLS (must pass before the rule is read):
  internal  the cluster split's own nn_frac_within_0.1 must be ~0 (else clustering is bugged)
  positive  a WITHIN-cluster row split must reproduce ~0.905 (the leaky regime), confirming
            the instrument still sees the leak it is designed to remove

PREPROCESSING NOTE (deliberate, not a leak): the standardizer used to BUILD the clusters is
fit on all pooled rows, because clustering is a split-construction device — exactly like the
source-file ids, which also "see" every row. Every FITTED MODEL transform is refit on its own
train split per the leakage rule (fit_transform_splits), unchanged from the shipped runners.

Run:  python -m experiments.run_cluster_split_control --partitions 5
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph

from qgridbench.data.preprocess import load_processed
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)
VARIANT = "binary"
RADIUS = 0.1  # the shipped neighbour-audit radius

# Anchors the decision rule is written against (shipped artifacts, not recomputed here).
ANCHOR_ROW_LEVEL = 0.9053
ANCHOR_WHOLE_FILE = 0.5943


def tuned_lgbm_params() -> dict:
    db = (REPO_ROOT / "results" / "optuna_cls_lgbm_binary_full.db").as_posix()
    study = optuna.load_study(study_name="cls_lgbm_binary_full", storage=f"sqlite:///{db}")
    return study.best_params


def build_clusters(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Connected components of the radius-RADIUS graph in standardized full-feature space.

    Uses the SAME impute+standardize transform as the shipped neighbour audit
    (build_preprocessor("full")), so the radius means the same thing here as in the
    45.8%/5.0% figures. Fit on all pooled rows: split construction, not a fitted model
    (see module docstring).
    """
    Z = build_preprocessor("full", n_components=None, seed=0).fit_transform(X)
    log.info("building radius graph (r=%.2f) over %d rows ...", RADIUS, len(Z))
    graph = radius_neighbors_graph(Z, radius=RADIUS, mode="connectivity", n_jobs=-1)
    n_clusters, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    stats = {
        "n_clusters": int(n_clusters),
        "largest_cluster_size": int(sizes.max()),
        "largest_cluster_frac": float(sizes.max() / len(labels)),
        "singleton_clusters": int((sizes == 1).sum()),
        "singleton_row_frac": float((sizes == 1).sum() / len(labels)),
        "median_cluster_size": float(np.median(sizes)),
        "mean_cluster_size": float(sizes.mean()),
    }
    log.info("clusters: %s", stats)
    return labels, stats


def cluster_file_overlap(labels: np.ndarray, gid: np.ndarray) -> dict:
    """How far the cluster partition reproduces the file partition (kill-criterion input)."""
    sizes = np.bincount(labels)
    single_file_rows = 0
    for c in np.flatnonzero(sizes > 0):
        m = labels == c
        if len(np.unique(gid[m])) == 1:
            single_file_rows += int(m.sum())
    return {
        "frac_single_file_rows": float(single_file_rows / len(labels)),
        "adjusted_rand_vs_files": float(adjusted_rand_score(gid, labels)),
    }


def cluster_splits(labels: np.ndarray, rng: np.random.Generator) -> dict:
    """60/20/20 BY CLUSTER, assigned by ROW COUNT (clusters are very unequal in size)."""
    clusters = np.unique(labels)
    rng.shuffle(clusters)
    sizes = np.bincount(labels)
    n = len(labels)
    assign = {}
    running = 0
    for c in clusters:
        frac = running / n
        assign[c] = "train" if frac < 0.6 else ("val" if frac < 0.8 else "test")
        running += sizes[c]
    bucket = np.array([assign[c] for c in labels])
    idx = np.arange(n)
    return {k: idx[bucket == k] for k in ("train", "val", "test")}


def score_level(params: dict, splits: dict, X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    pipe = build_preprocessor("full", n_components=None, seed=seed)
    Z = fit_transform_splits(pipe, X, splits)
    set_all_seeds(seed)
    est = build("lgbm", params, seed)
    fit_model("lgbm", est, Z["train"], y[splits["train"]])
    return float(f1_score(y[splits["test"]], est.predict(Z["test"]), average="macro"))


def audit(splits: dict, X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    floor = trivial_baselines(y[splits["train"]], y[splits["test"]])
    pipe = build_preprocessor("full", n_components=None, seed=seed)
    Z = fit_transform_splits(pipe, X, splits)
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Z["train"])
    d = nn.kneighbors(Z["test"])[0].ravel()
    return {
        "floor_macro_f1": floor["baseline_stratified_random"]["aggregate"]["macro_f1"]["mean"],
        "nn_frac_within_0.1": float((d < RADIUS).mean()),
        "n_train": int(len(splits["train"])),
        "n_test": int(len(splits["test"])),
    }


def within_cluster_row_split(labels: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    """POSITIVE CONTROL: split rows at random INSIDE each cluster -> the leaky regime.

    Singleton clusters (49% of rows here) have no within-cluster split to make, so each is
    assigned to a bucket by a 60/20/20 draw — for those rows this IS the row-level protocol,
    which is the point: the control must land in the leaky regime, not construct a new one.
    A naive int(0.6*n) floor sends every singleton to test (measured: n_train 21,530 vs
    n_test 49,784) and silently builds a THIRD protocol, which is what this branch prevents.
    """
    idx = np.arange(len(y))
    buckets = {"train": [], "val": [], "test": []}
    names = np.array(["train", "val", "test"])
    for c in np.unique(labels):
        m = idx[labels == c]
        n = len(m)
        if n == 1:
            buckets[names[rng.choice(3, p=[0.6, 0.2, 0.2])]].append(m)
            continue
        m = m.copy()
        rng.shuffle(m)
        n_tr, n_va = max(1, int(round(0.6 * n))), None
        n_va = min(n - 1, n_tr + max(0, int(round(0.2 * n))))
        buckets["train"].append(m[:n_tr])
        buckets["val"].append(m[n_tr:n_va])
        buckets["test"].append(m[n_va:])
    return {k: np.sort(np.concatenate(v)) for k, v in buckets.items()}


def row_level_reference(y: np.ndarray) -> dict:
    """Second POSITIVE CONTROL: the shipped row-level split through THIS code path.

    Must reproduce the 0.905 anchor, which is what licenses reading the cluster-split
    number as a removal of leakage rather than as a property of this runner.
    """
    from qgridbench.data.preprocess import load_processed as _lp

    _, _, sp = _lp(VARIANT)
    return {k: np.asarray(v) for k, v in sp.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=5)
    args = ap.parse_args()

    params = tuned_lgbm_params()
    X, y, _ = load_processed(VARIANT)

    from experiments.run_group_split_control import reconstruct_groups

    Xg, yg, gid = reconstruct_groups()
    assert Xg.shape == X.shape and np.array_equal(yg, y), "group reconstruction misaligned"

    cfg = {
        "variant": VARIANT,
        "radius": RADIUS,
        "n_partitions": args.partitions,
        "lgbm_source": "optuna study cls_lgbm_binary_full (best_params)",
        "anchor_row_level": ANCHOR_ROW_LEVEL,
        "anchor_whole_file": ANCHOR_WHOLE_FILE,
    }

    with RunDir(f"cluster_split_control_{VARIANT}", config=cfg) as run:
        labels, cstats = build_clusters(X)
        overlap = cluster_file_overlap(labels, gid)
        cstats.update(overlap)
        run.write_json("cluster_stats.json", cstats)
        log.info("cluster-file overlap: %s", overlap)

        unidentifiable = (
            overlap["frac_single_file_rows"] > 0.5 and overlap["adjusted_rand_vs_files"] > 0.9
        )
        degenerate = cstats["largest_cluster_frac"] > 0.4

        rows = []
        if not degenerate:
            rng = np.random.default_rng(0)
            for p in range(args.partitions):
                sp = cluster_splits(labels, rng)
                r = {
                    "partition": p,
                    "level_full_lgbm": score_level(params, sp, X, y),
                    **audit(sp, X, y),
                }
                rows.append(r)
                run.write_json("results_partial.json", {"rows": rows})
                log.info(
                    "cluster p%d | level %.4f | floor %.4f | nn<0.1 %.2f%% | n_test %d",
                    p,
                    r["level_full_lgbm"],
                    r["floor_macro_f1"],
                    100 * r["nn_frac_within_0.1"],
                    r["n_test"],
                )
        else:
            log.warning(
                "DEGENERATE: largest cluster holds %.1f%% of rows — a 60/20/20 cluster split "
                "is impossible; reporting the structural finding instead",
                100 * cstats["largest_cluster_frac"],
            )

        # ---- positive controls: both must land in the LEAKY regime ------------------
        rng_c = np.random.default_rng(1)
        sp_wc = within_cluster_row_split(labels, y, rng_c)
        wc = {
            "level_full_lgbm": score_level(params, sp_wc, X, y),
            **audit(sp_wc, X, y),
        }
        log.info("POSITIVE CONTROL A (within-cluster row split): %s", wc)

        sp_row = row_level_reference(y)
        rowref = {
            "level_full_lgbm": score_level(params, sp_row, X, y),
            **audit(sp_row, X, y),
        }
        log.info("POSITIVE CONTROL B (shipped row-level split, anchor 0.9053): %s", rowref)

        # ---- verdict --------------------------------------------------------------
        if rows:
            s_vals = [r["level_full_lgbm"] for r in rows]
            s, s_std = float(np.mean(s_vals)), float(np.std(s_vals))
            nn_mean = float(np.mean([r["nn_frac_within_0.1"] for r in rows]))
            leakage_share = (ANCHOR_ROW_LEVEL - s) / (ANCHOR_ROW_LEVEL - ANCHOR_WHOLE_FILE)
            branch = (
                "leakage_dominated" if s <= 0.70 else ("shift_dominated" if s >= 0.85 else "mixed")
            )
        else:
            s = s_std = nn_mean = leakage_share = float("nan")
            branch = "degenerate_no_cluster_split"

        controls = {
            "internal_nn_near_zero": (None if not rows else bool(nn_mean < 0.05)),
            "positive_A_within_cluster_recovers_leak": bool(wc["level_full_lgbm"] >= 0.85),
            "positive_A_within_cluster_score": wc["level_full_lgbm"],
            "positive_A_train_test_balanced": bool(
                0.4 < wc["n_train"] / (wc["n_train"] + wc["n_test"]) < 0.8
            ),
            "positive_B_row_level_reproduces_anchor": bool(
                abs(rowref["level_full_lgbm"] - ANCHOR_ROW_LEVEL) < 0.005
            ),
            "positive_B_row_level_score": rowref["level_full_lgbm"],
        }
        controls["all_passed"] = bool(
            controls["internal_nn_near_zero"]
            and controls["positive_A_within_cluster_recovers_leak"]
            and controls["positive_B_row_level_reproduces_anchor"]
        )
        verdict = {
            "s_cluster_split_mean": s,
            "s_cluster_split_std": s_std,
            "leakage_share_of_0311": leakage_share,
            "branch": ("unidentifiable" if unidentifiable else branch),
            "unidentifiable": bool(unidentifiable),
            "degenerate": bool(degenerate),
            "controls": controls,
            "rule": "s<=0.70 leakage-dominated; s>=0.85 shift-dominated; else mixed",
        }
        run.write_json(
            "results.json",
            {
                "rows": rows,
                "cluster_stats": cstats,
                "positive_control_within_cluster": wc,
                "positive_control_row_level": rowref,
                "verdict": verdict,
            },
        )
        run.metrics = {"s_cluster_split_mean": s, "leakage_share": leakage_share}
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
