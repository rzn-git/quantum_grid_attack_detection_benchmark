"""N-DECAY: the leak has a length -- macro-F1 versus the positional gap between a test row
and its nearest training row, inside the scenario blocks of one released file.

PRE-REGISTRATION (decision log 2026-09-15T18:52Z; AMENDED 19:02Z as N-DECAY-2 and 19:09Z
as N-DECAY-2b; Tier A). Red-team pass 6 showed the "fraction of test rows within radius
0.1" law is a three-cluster statistic (Spearman 0.68 over seven constructions, order
inverted inside the 3-7% band) and that the cluster split (0.787 at 0.33%) is its
counterexample. The quantity underneath is the temporal gap g, in released rows, between a
test row and its nearest training row.

CONSTRUCTION. In each contiguous (file, scenario) block in released order the TRAINING band
is the middle 60% of rows; every other row of the block is a TEST row whose gap g is its
positional distance, within its file, to the nearest training row (g = 1 touching the
band). One tuned full-feature fit therefore samples every gap from 1 to ~28 on both sides
of the band. Curve criteria are judged on the COMPOSITION-MATCHED set: blocks with
>= MIN_BLOCK_MATCHED rows, whose flanks supply one test row at every gap 1..20 per side, so
gaps 1..20 share the same blocks by construction (the 18:52Z design judged flatness over
tail bins only large blocks populate, and that is what failed). Exponential fit
F(g) = a + b exp(-(g-1)/tau) on gaps 1..20; the embargo prescription is the first gap within
0.02 of the empirical asymptote. A released row is ~100 recorded samples (1% release).

ARMS. A: band arm, LightGBM. B: same geometry, order shuffled within block, gaps in the
permuted coordinate (negative control). C: middle-20% test band (kept as a diagnostic: its
gap-1 rows are the band's EDGE rows and are therefore one-sided, so it does not test
multiplicity -- the 19:02Z design error). D: arm A with tuned XGBoost (H4). E (2b): the
interleaved arm -- inside every block, rows at position 2 mod 5 test, rows at 4 mod 5 are
set aside, the rest (60%) train, so EVERY test row has training rows at gap 1 on both
sides at the row-level split's training density; this is the two-sided test of H3. Plus
two regenerated cluster-split partitions (same rule as E3), scored fresh and predicted from
their file-level gap distribution (positive control).

CONTROLS (blocking; VOID if either fails). Negative (2b form): arm B, matched set, gaps
1..20 -- |Spearman(gap, F)| <= 0.5, the observed max - min not extreme under a
gap-label permutation null (p >= 0.05 over N_PERM permutations of the nominal gap labels
among the same test rows), and the bin mean within 0.02 of arm B's overall test macro-F1.
(The 19:02Z form, max - min <= 0.03 over 20 bins of ~700 rows, is below the binomial noise
of such bins and cannot pass for a reason unrelated to gap.) Positive: both cluster
partitions predicted within 0.03 of their fresh scores.
H1: arm A matched: Spearman <= -0.9 over gaps 1..10; F(1) >= 0.85; asymptote (mean over
gaps 15..20) within 0.03 of 0.6005 or 0.594; tau < 20. KILL: F(1) <= 0.65.
H3 (2b): multiplicity delta = F_E(1) - F_A(1) >= 0.02 AND F_E(1) >= 0.89 (the row-level
regime is reached by two-sided gap-1 adjacency at 60% density). KILL: delta <= 0.005.
H4: tau_xgb / tau_lgbm within [0.5, 2].
Diagnostics (no criterion): all-blocks curves, before/after-band curves, arm C, prediction
tables over the archived constructions under file-level AND same-block gap definitions.

Run:  python -m experiments.run_leak_decay_curve [--smoke] [--no-cluster] [--families lgbm xgb]
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from experiments.run_cluster_split_control import build_clusters, cluster_splits
from experiments.run_group_split_control import group_splits, reconstruct_groups
from experiments.run_leak_rate_families import tuned_full_params
from experiments.run_temporal_split import load_blocks, positional_splits
from qgridbench.data.preprocess import load_processed
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import build_preprocessor
from qgridbench.models.classical.zoo import build, fit_model
from qgridbench.utils.run_tracking import RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
SEED = 0
BAND = (0.2, 0.8)  # the middle 60% of every block trains (arm A/B/D)
MIDDLE = (0.4, 0.6)  # the middle 20% of every block tests (arm C, diagnostic)
INTERLEAVE = (5, 2, 4)  # arm E: period, test offset, set-aside offset
MIN_ROWS = 200  # a gap enters a curve only with this many test rows
MIN_BLOCK_MATCHED = 100  # composition-matched set: blocks whose flanks reach gap 20
GAP_H1 = 10  # monotonicity is judged on gaps <= this
GAP_FIT = 20  # the exponential fit and the negative control use gaps <= this
ASYMPTOTE_GAPS = (15, 20)
N_PERM = 500  # gap-label permutations for the negative control's null
SAMPLES_PER_RELEASED_ROW = 100  # the release is a 1% sample of the recorded stream
TOL_PRED = 0.03
# Archived scores the prediction table is checked against
# (20260902T022541Z_temporal_split_binary; 20260816T023620Z_group_split_control_binary;
# E3 cluster split 20260822T074636Z_cluster_split_control_binary, five-partition mean).
ARCHIVED = {
    "R row-level": 0.9053,
    "T0 ordered": 0.6005,
    "T1 embargo": 0.5903,
    "S shuffled": 0.9054,
    "M middle": 0.6752,
    "V reversed": 0.6178,
    "whole-file p0": 0.6032,
}
ARCHIVED_CLUSTER_MEAN = 0.7874


def gaps_to_train(train_mask: np.ndarray, unit: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Gap from every row to its nearest TRAIN row in the same `unit` (file or block).

    `pos` is the true released index for real gaps, or the permuted coordinate for the
    negative control. Train rows get 0; rows whose unit has no train row get inf.
    """
    gap = np.full(len(unit), np.inf)
    for u in np.unique(unit):
        rows = np.flatnonzero(unit == u)
        order = np.argsort(pos[rows])
        rows = rows[order]
        p = pos[rows]
        tr = train_mask[rows]
        if not tr.any():
            continue
        ptr = p[tr]
        k = np.searchsorted(ptr, p)
        left = np.where(k > 0, p - ptr[np.clip(k - 1, 0, len(ptr) - 1)], np.inf)
        right = np.where(k < len(ptr), ptr[np.clip(k, 0, len(ptr) - 1)] - p, np.inf)
        gap[rows] = np.minimum(left, right)
    return gap


def band_split(block: np.ndarray, rng: np.random.Generator, shuffled: bool):
    """Middle-60% training band per block (arms A, B, D).

    Returns (train_mask, pos, side): `pos` is the coordinate gaps are measured in (the true
    released index, or the within-block permuted one for the control); `side` is -1 before
    the band, +1 after it, 0 inside.
    """
    n = len(block)
    pos = np.arange(n, dtype=np.int64)
    train = np.zeros(n, dtype=bool)
    side = np.zeros(n, dtype=np.int8)
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        if shuffled:
            perm = idx[rng.permutation(len(idx))]
            pos[perm] = idx  # row perm[j] takes the nominal position idx[j]
            idx = perm
        m = len(idx)
        lo, hi = int(BAND[0] * m), int(BAND[1] * m)
        train[idx[lo:hi]] = True
        side[idx[:lo]] = -1
        side[idx[hi:]] = 1
    return train, pos, side


def middle_split(block: np.ndarray) -> np.ndarray:
    """Arm C (diagnostic): the middle 20% of every block tests, everything else trains."""
    train = np.ones(len(block), dtype=bool)
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        m = len(idx)
        train[idx[int(MIDDLE[0] * m) : int(MIDDLE[1] * m)]] = False
    return train


def interleave_split(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arm E: inside every block, position 2 mod 5 tests, 4 mod 5 is set aside, rest trains.

    Every test row then has training rows at gap 1 on BOTH sides, with the training density
    (60%) of the row-level split. Returns (train_mask, test_mask); set-aside rows are neither.
    """
    period, t_off, v_off = INTERLEAVE
    train = np.zeros(len(block), dtype=bool)
    test = np.zeros(len(block), dtype=bool)
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        r = np.arange(len(idx)) % period
        test[idx[r == t_off]] = True
        train[idx[(r != t_off) & (r != v_off)]] = True
    return train, test


def fit_predict(family: str, train_mask: np.ndarray, X: np.ndarray, y: np.ndarray, params: dict):
    """Tuned full-feature model on the train rows; predictions for every other row."""
    pipe = build_preprocessor("full", n_components=None, seed=SEED)
    Ztr = pipe.fit_transform(X[train_mask])
    set_all_seeds(SEED)
    est = build(family, params, SEED)
    fit_model(family, est, Ztr, y[train_mask])
    pred = np.full(len(y), -1, dtype=np.int64)
    pred[~train_mask] = est.predict(pipe.transform(X[~train_mask]))
    return pred


def per_gap_curve(pred, y, gap, test_mask, y_train, max_gap: int | None = None) -> list[dict]:
    out = []
    gs = gap[test_mask]
    for g in np.unique(gs[np.isfinite(gs)]).astype(int):
        if max_gap is not None and g > max_gap:
            break
        sel = test_mask & (gap == g)
        n = int(sel.sum())
        if n < MIN_ROWS:
            continue
        floor = trivial_baselines(y_train, y[sel])["baseline_stratified_random"]["aggregate"]
        out.append(
            {
                "gap": int(g),
                "n": n,
                "macro_f1": float(f1_score(y[sel], pred[sel], average="macro")),
                "floor": float(floor["macro_f1"]["mean"]),
                "pos_rate": float(y[sel].mean()),
            }
        )
    return out


def permutation_null_range(pred, y, gap, test_mask, rng: np.random.Generator) -> dict:
    """Negative control's null: permute the nominal gap labels among the same test rows and
    recompute the per-gap macro-F1 range; p = share of null ranges >= the observed one."""
    sel = test_mask & np.isfinite(gap) & (gap <= GAP_FIT)
    yy, pp, gg = y[sel], pred[sel], gap[sel].astype(int)
    gaps = np.unique(gg)

    def rng_of(labels):
        vals = [f1_score(yy[labels == g], pp[labels == g], average="macro") for g in gaps]
        return max(vals) - min(vals)

    observed = rng_of(gg)
    null = np.array([rng_of(rng.permutation(gg)) for _ in range(N_PERM)])
    return {
        "observed_range": float(observed),
        "null_range_mean": float(null.mean()),
        "null_range_p95": float(np.percentile(null, 95)),
        "p_value": float((null >= observed).mean()),
        "n_perm": N_PERM,
    }


def fit_decay(pts: list[dict]) -> dict:
    g = np.array([p["gap"] for p in pts], dtype=float)
    f = np.array([p["macro_f1"] for p in pts])
    w = np.array([p["n"] for p in pts], dtype=float)

    def model(gg, a, b, tau):
        return a + b * np.exp(-(gg - 1) / tau)

    tail = np.mean(
        [p["macro_f1"] for p in pts if ASYMPTOTE_GAPS[0] <= p["gap"] <= ASYMPTOTE_GAPS[1]]
    )
    popt, _ = curve_fit(
        model,
        g,
        f,
        p0=[tail, f[0] - tail, 3.0],
        sigma=1 / np.sqrt(w),
        bounds=([0.0, -1.0, 0.05], [1.0, 1.0, 500.0]),
        maxfev=20000,
    )
    resid = f - model(g, *popt)
    within = [p["gap"] for p in pts if p["macro_f1"] - tail <= 0.02]
    return {
        "asymptote_fit": float(popt[0]),
        "amplitude_fit": float(popt[1]),
        "tau_released_rows": float(popt[2]),
        "tau_recorded_samples": float(popt[2] * SAMPLES_PER_RELEASED_ROW),
        "rmse_weighted": float(np.sqrt(np.average(resid**2, weights=w))),
        "asymptote_empirical": float(tail),
        "embargo_gap_within_0.02": int(min(within)) if within else None,
    }


def predict_arm(test_idx, train_idx, unit, pos_true, table, asymptote) -> dict:
    """Test-row-weighted mean of F(g) over the arm's TRUE gap distribution in `unit`."""
    train_mask = np.zeros(len(unit), dtype=bool)
    train_mask[train_idx] = True
    gs = gaps_to_train(train_mask, unit, pos_true)[test_idx]
    lookup = {p["gap"]: p["macro_f1"] for p in table}
    vals = np.array([lookup.get(int(g), asymptote) if np.isfinite(g) else asymptote for g in gs])
    fin = gs[np.isfinite(gs)]
    return {
        "predicted": float(vals.mean()),
        "n_test": int(len(test_idx)),
        "gap_median": float(np.median(fin)) if len(fin) else None,
        "frac_gap_1": float((gs == 1).mean()),
        "frac_gap_le_3": float((gs <= 3).mean()),
        "frac_inf": float((~np.isfinite(gs)).mean()),
    }


def curve_summary(pts: list[dict]) -> dict:
    return {p["gap"]: p["macro_f1"] for p in pts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="geometry only, no fits")
    ap.add_argument("--no-cluster", action="store_true", help="skip the cluster-split control")
    ap.add_argument("--cluster-partitions", type=int, default=2)
    ap.add_argument("--families", nargs="+", default=["lgbm", "xgb"])
    args = ap.parse_args()
    if args.families[0] != "lgbm":
        raise SystemExit("the first family must be lgbm: arms B, C and E are LightGBM-only")

    X, y, block = load_blocks()
    Xg, yg, gid = reconstruct_groups()
    if X.shape != Xg.shape or not np.array_equal(y, yg):
        raise SystemExit("block and group reconstructions disagree")
    Xp, yp, splits_row = load_processed(VARIANT)
    if X.shape != Xp.shape or not np.array_equal(y, yp):
        raise SystemExit("block reconstruction misaligned with the processed artifact")
    splits_row = {k: np.asarray(v) for k, v in splits_row.items()}
    n = len(y)
    pos_true = np.arange(n, dtype=np.int64)
    sizes = np.bincount(block)
    matched = sizes[block] >= MIN_BLOCK_MATCHED

    tr_a, pos_a, side_a = band_split(block, np.random.default_rng(SEED), shuffled=False)
    tr_b, pos_b, _ = band_split(block, np.random.default_rng(SEED), shuffled=True)
    tr_c = middle_split(block)
    tr_e, te_e = interleave_split(block)
    gap_a = gaps_to_train(tr_a, gid, pos_a)
    gap_b = gaps_to_train(tr_b, gid, pos_b)
    gap_c = gaps_to_train(tr_c, gid, pos_true)
    gap_e = gaps_to_train(tr_e, gid, pos_true)
    te_a, te_b, te_c = ~tr_a, ~tr_b, ~tr_c
    counts = {
        int(g): int(c)
        for g, c in zip(*np.unique(gap_a[te_a & matched], return_counts=True))
        if np.isfinite(g)
    }
    e_gap1_both = 0
    for i in np.flatnonzero(te_e):
        if (
            0 < i < n - 1
            and tr_e[i - 1]
            and tr_e[i + 1]
            and block[i - 1] == block[i] == block[i + 1]
        ):
            e_gap1_both += 1
    log.info(
        "blocks %d (matched >=%d rows: %d) | arm A train %d test %d (matched test %d, gap-1 %d) | "
        "arm C train %d test %d | arm E train %d test %d (gap-1 both sides: %d)",
        len(sizes),
        MIN_BLOCK_MATCHED,
        int((sizes >= MIN_BLOCK_MATCHED).sum()),
        int(tr_a.sum()),
        int(te_a.sum()),
        int((te_a & matched).sum()),
        counts.get(1, 0),
        int(tr_c.sum()),
        int(te_c.sum()),
        int(tr_e.sum()),
        int(te_e.sum()),
        e_gap1_both,
    )
    if args.smoke:
        print("SMOKE matched gap counts (arm A):", {g: counts[g] for g in sorted(counts)[:22]})
        ce = {
            int(g): int(c)
            for g, c in zip(*np.unique(gap_e[te_e], return_counts=True))
            if np.isfinite(g)
        }
        print(
            "SMOKE arm E gap counts:",
            ce,
            f"| train frac {tr_e.mean():.3f} test frac {te_e.mean():.3f}",
        )
        return

    params = {f: tuned_full_params(f) for f in args.families}
    cfg = {
        "variant": VARIANT,
        "band": list(BAND),
        "middle": list(MIDDLE),
        "interleave": list(INTERLEAVE),
        "min_rows_per_gap": MIN_ROWS,
        "min_block_matched": MIN_BLOCK_MATCHED,
        "n_blocks_matched": int((sizes >= MIN_BLOCK_MATCHED).sum()),
        "gap_h1": GAP_H1,
        "gap_fit": GAP_FIT,
        "asymptote_gaps": list(ASYMPTOTE_GAPS),
        "n_perm": N_PERM,
        "families": args.families,
        "seed": SEED,
        "params_source": "optuna studies cls_<family>_binary_full (best_params)",
        "cluster_partitions": 0 if args.no_cluster else args.cluster_partitions,
        "preregistration": "decision log 2026-09-15T19:09Z "
        "(N-DECAY-2b, amending 19:02Z and 18:52Z)",
    }
    with RunDir(f"leak_decay_curve_{VARIANT}", config=cfg, seeds=[SEED]) as run:
        out: dict = {"curves": {}, "fits": {}}
        preds = {}
        # ---- arm A per family ------------------------------------------------------------
        for fam in args.families:
            preds[("A", fam)] = fit_predict(fam, tr_a, X, y, params[fam])
            out["curves"][f"A_{fam}_matched"] = per_gap_curve(
                preds[("A", fam)], y, gap_a, te_a & matched, y[tr_a], GAP_FIT
            )
            out["curves"][f"A_{fam}_all"] = per_gap_curve(
                preds[("A", fam)], y, gap_a, te_a, y[tr_a]
            )
            out["fits"][f"A_{fam}"] = fit_decay(out["curves"][f"A_{fam}_matched"])
            for p in out["curves"][f"A_{fam}_matched"]:
                log.info(
                    "A %-4s gap %2d | n %5d | macro-F1 %.4f | floor %.3f",
                    fam,
                    p["gap"],
                    p["n"],
                    p["macro_f1"],
                    p["floor"],
                )
            log.info("A %-4s fit: %s", fam, out["fits"][f"A_{fam}"])
        # ---- arm B (negative control), C (diagnostic), E (two-sided) ----------------------
        preds[("B", "lgbm")] = fit_predict("lgbm", tr_b, X, y, params["lgbm"])
        out["curves"]["B_lgbm_matched"] = per_gap_curve(
            preds[("B", "lgbm")], y, gap_b, te_b & matched, y[tr_b], GAP_FIT
        )
        out["curves"]["B_lgbm_all"] = per_gap_curve(preds[("B", "lgbm")], y, gap_b, te_b, y[tr_b])
        preds[("C", "lgbm")] = fit_predict("lgbm", tr_c, X, y, params["lgbm"])
        out["curves"]["C_lgbm_matched"] = per_gap_curve(
            preds[("C", "lgbm")], y, gap_c, te_c & matched, y[tr_c], GAP_H1
        )
        preds[("E", "lgbm")] = fit_predict("lgbm", tr_e, X, y, params["lgbm"])
        out["curves"]["E_lgbm_matched"] = per_gap_curve(
            preds[("E", "lgbm")], y, gap_e, te_e & matched, y[tr_e], GAP_H1
        )
        out["curves"]["E_lgbm_all"] = per_gap_curve(
            preds[("E", "lgbm")], y, gap_e, te_e, y[tr_e], GAP_H1
        )
        out["sides"] = {
            name: per_gap_curve(
                preds[("A", "lgbm")], y, gap_a, te_a & matched & (side_a == s), y[tr_a], GAP_H1
            )
            for name, s in (("before", -1), ("after", 1))
        }
        overall = {
            "A_lgbm": float(f1_score(y[te_a], preds[("A", "lgbm")][te_a], average="macro")),
            "B_lgbm": float(f1_score(y[te_b], preds[("B", "lgbm")][te_b], average="macro")),
            "C_lgbm": float(f1_score(y[te_c], preds[("C", "lgbm")][te_c], average="macro")),
            "E_lgbm": float(f1_score(y[te_e], preds[("E", "lgbm")][te_e], average="macro")),
            "E_lgbm_matched": float(
                f1_score(y[te_e & matched], preds[("E", "lgbm")][te_e & matched], average="macro")
            ),
        }
        for p in out["curves"]["E_lgbm_matched"]:
            log.info(
                "E interleaved gap %d | n %5d | macro-F1 %.4f (all blocks overall %.4f)",
                p["gap"],
                p["n"],
                p["macro_f1"],
                overall["E_lgbm"],
            )
        null = permutation_null_range(
            preds[("B", "lgbm")], y, gap_b, te_b & matched, np.random.default_rng(SEED)
        )
        log.info("negative control null: %s", null)
        run.write_json("results_partial.json", {**out, "overall": overall, "negative_null": null})

        # ---- prediction tables (file-level and same-block gaps) ---------------------------
        curve_a = out["curves"]["A_lgbm_all"]
        asym = out["fits"]["A_lgbm"]["asymptote_empirical"]
        t0, _ = positional_splits(block, "ordered", np.random.default_rng(SEED))
        t1, _ = positional_splits(block, "embargo", np.random.default_rng(SEED))
        sh, _ = positional_splits(block, "shuffled", np.random.default_rng(SEED))
        mid, _ = positional_splits(block, "middle", np.random.default_rng(SEED))
        rev, _ = positional_splits(block, "reversed", np.random.default_rng(SEED))
        wf, held = group_splits(gid, y, np.random.default_rng(0))
        arms = {
            "R row-level": splits_row,
            "T0 ordered": t0,
            "T1 embargo": t1,
            "S shuffled": sh,
            "M middle": mid,
            "V reversed": rev,
            "whole-file p0": wf,
        }
        tables = {"file_gap": {}, "block_gap": {}}
        for name, sp in arms.items():
            for key, unit in (("file_gap", gid), ("block_gap", block)):
                r = predict_arm(sp["test"], sp["train"], unit, pos_true, curve_a, asym)
                r["actual_archived"] = ARCHIVED[name]
                r["abs_error"] = abs(r["predicted"] - ARCHIVED[name])
                tables[key][name] = r
            log.info(
                "%-14s archived %.4f | file-gap pred %.4f (err %.3f) | "
                "block-gap pred %.4f (err %.3f)",
                name,
                ARCHIVED[name],
                tables["file_gap"][name]["predicted"],
                tables["file_gap"][name]["abs_error"],
                tables["block_gap"][name]["predicted"],
                tables["block_gap"][name]["abs_error"],
            )

        # ---- cluster split: regenerated, scored fresh, predicted (positive control) -------
        cluster = []
        cluster_stats = {}
        if not args.no_cluster:
            labels, cstats = build_clusters(X)
            cluster_stats = {k: v for k, v in cstats.items() if isinstance(v, (int, float))}
            crng = np.random.default_rng(0)
            for k in range(args.cluster_partitions):
                sp = cluster_splits(labels, crng)
                tm = np.zeros(n, dtype=bool)
                tm[sp["train"]] = True
                pred_c = fit_predict("lgbm", tm, X, y, params["lgbm"])
                actual = float(f1_score(y[sp["test"]], pred_c[sp["test"]], average="macro"))
                r = predict_arm(sp["test"], sp["train"], gid, pos_true, curve_a, asym)
                r.update(
                    {
                        "partition": k,
                        "actual_fresh": actual,
                        "abs_error": abs(r["predicted"] - actual),
                    }
                )
                cluster.append(r)
                log.info(
                    "cluster p%d predicted %.4f | fresh %.4f | err %.4f",
                    k,
                    r["predicted"],
                    actual,
                    r["abs_error"],
                )

        # ---- controls and verdict ---------------------------------------------------------
        bpts = out["curves"]["B_lgbm_matched"]
        fb = np.array([p["macro_f1"] for p in bpts])
        rho_b = float(spearmanr([p["gap"] for p in bpts], fb)[0])
        ctrl_neg = bool(
            abs(rho_b) <= 0.5
            and null["p_value"] >= 0.05
            and abs(fb.mean() - overall["B_lgbm"]) <= 0.02
        )
        ctrl_pos = bool(cluster) and bool(all(c["abs_error"] <= TOL_PRED for c in cluster))
        fa = curve_summary(out["curves"]["A_lgbm_matched"])
        fc = curve_summary(out["curves"]["C_lgbm_matched"])
        fe = curve_summary(out["curves"]["E_lgbm_matched"])
        h1_pts = [(g, f) for g, f in fa.items() if g <= GAP_H1]
        rho = float(spearmanr([g for g, _ in h1_pts], [f for _, f in h1_pts])[0])
        fit_l = out["fits"]["A_lgbm"]
        f1_1, fe_1 = fa.get(1), fe.get(1)
        h1 = bool(
            f1_1 is not None
            and f1_1 >= 0.85
            and rho <= -0.9
            and (
                abs(fit_l["asymptote_empirical"] - 0.6005) <= 0.03
                or abs(fit_l["asymptote_empirical"] - 0.594) <= 0.03
            )
            and fit_l["tau_released_rows"] < 20
        )
        kill_h1 = bool(f1_1 is not None and f1_1 <= 0.65)
        delta = None if (fe_1 is None or f1_1 is None) else fe_1 - f1_1
        h3 = bool(delta is not None and delta >= 0.02 and fe_1 >= 0.89)
        kill_h3 = bool(delta is not None and delta <= 0.005)
        tau_ratio = None
        h4 = None
        if "xgb" in args.families:
            tau_ratio = out["fits"]["A_xgb"]["tau_released_rows"] / fit_l["tau_released_rows"]
            h4 = bool(0.5 <= tau_ratio <= 2.0)
        verdict = {
            "controls_pass": bool(ctrl_neg and ctrl_pos),
            "control_negative": {
                "pass": ctrl_neg,
                "spearman_gap_f1": rho_b,
                "bin_mean": float(fb.mean()),
                "bin_range": [float(fb.min()), float(fb.max())],
                **null,
            },
            "control_positive_cluster": ctrl_pos,
            "cluster_errors": [c["abs_error"] for c in cluster],
            "f1_gap": {str(g): fa[g] for g in sorted(fa) if g in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20)},
            "f_c_middle_band_diagnostic": {str(g): fc[g] for g in sorted(fc)},
            "f_e_interleaved_gap1": fe_1,
            "multiplicity_delta": delta,
            "overall": overall,
            "spearman_gap_f1_le_10_matched": rho,
            **{f"lgbm_{k}": v for k, v in fit_l.items()},
            **(
                {f"xgb_{k}": v for k, v in out["fits"]["A_xgb"].items()}
                if "xgb" in args.families
                else {}
            ),
            "tau_ratio_xgb_over_lgbm": tau_ratio,
            "h1_decay_law": h1,
            "kill_h1": kill_h1,
            "h3_two_sided_multiplicity": h3,
            "kill_h3": kill_h3,
            "h4_length_belongs_to_data": h4,
            "prediction_max_abs_error_file_gap": max(
                v["abs_error"] for v in tables["file_gap"].values()
            ),
            "prediction_max_abs_error_block_gap": max(
                v["abs_error"] for v in tables["block_gap"].values()
            ),
            "prediction_max_abs_error_block_gap_excl_dense": max(
                v["abs_error"]
                for k, v in tables["block_gap"].items()
                if k not in ("R row-level", "S shuffled")
            ),
            "cluster_archived_mean": ARCHIVED_CLUSTER_MEAN,
        }
        if not verdict["controls_pass"]:
            log.error("CONTROL FAILED (run VOID): negative=%s positive=%s", ctrl_neg, ctrl_pos)
        run.metrics = verdict
        run.write_json(
            "results.json",
            {
                **out,
                "overall": overall,
                "negative_null": null,
                "prediction_tables": tables,
                "cluster": cluster,
                "cluster_stats": cluster_stats,
                "held_out_files_p0": [int(h) for h in held[2]],
                "verdict": verdict,
            },
        )
        log.info(
            "VERDICT | F(1) %.4f F_E(1) %s delta %s | asymptote %.4f | "
            "tau %.2f rows (xgb ratio %s) | "
            "rho %.3f | controls=%s H1=%s H3=%s H4=%s",
            f1_1,
            None if fe_1 is None else round(fe_1, 4),
            None if delta is None else round(delta, 4),
            fit_l["asymptote_empirical"],
            fit_l["tau_released_rows"],
            None if tau_ratio is None else round(tau_ratio, 2),
            rho,
            verdict["controls_pass"],
            h1,
            h3,
            h4,
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
