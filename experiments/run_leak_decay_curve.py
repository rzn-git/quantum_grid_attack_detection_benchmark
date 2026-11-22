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

R4-10 SPIKES (decision log 2026-11-16T11:07Z, round-4 pre-registration; values, no branch).
(i) Leak length by scenario class: arm A's predictions re-aggregated by the three-class label
of each row's scenario (no-event, natural, attack; read off the row-aligned triple and
multiclass releases). A block holds one scenario, so a class subset holds one true label and
its macro-F1 is degenerate; the per-class statistic is the share of the class's test rows
predicted correctly (attack recall, natural and no-event specificity on the binary task),
fitted per gap with the same rule (matched set, gaps 1..20, the same exponential). (ii)
`--variant triple` runs the whole instrument -- arms, controls, verdict code -- on the
three-class release. Anchor (blocking, binary only): every family's arm-A tau and matched curve
equal the archived N-DECAY-2b run exactly. XGBoost's fit depends on its thread count and the
archive ran unpinned on 24 cores, so XGBoost fits in its own process at 24 threads
(--family-threads xgb=24, ruling 2026-11-16T13:05Z item 4) and its anchor blocks too; every
other family runs at --threads. At another XGBoost count its anchor is reported, not blocking.
Arm-A predictions are saved (arm_a_predictions.npz) so later re-aggregations need no refit.

Run:  uv run python -m experiments.run_leak_decay_curve --variant binary --threads 4 --workers 1
      [--smoke] [--no-cluster] [--families lgbm xgb]
"""

from __future__ import annotations

import argparse
import os
import sys

from experiments.family_threads import THREAD_VARS

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=4)
THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
if __name__ == "__main__":  # pin before numpy loads; importers (the gas runner) keep their own
    for _var in THREAD_VARS:
        os.environ[_var] = str(THREADS)

import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from experiments.family_threads import (  # noqa: E402
    check_thread_env,
    parse_family_threads,
    run_grouped,
)
from experiments.run_cluster_split_control import build_clusters, cluster_splits  # noqa: E402
from experiments.run_file_tail_split import load_rows  # noqa: E402
from experiments.run_group_split_control import group_splits  # noqa: E402
from experiments.run_temporal_split import positional_splits  # noqa: E402
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.data.validate import load_variant_frames  # noqa: E402
from qgridbench.eval.baselines import trivial_baselines  # noqa: E402
from qgridbench.features.reduce import build_preprocessor  # noqa: E402
from qgridbench.models.classical.zoo import build, fit_model  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import set_all_seeds  # noqa: E402

log = get_logger(__name__)

VARIANTS = ("binary", "triple")
ANCHOR_RUN = "20260915T190906Z_leak_decay_curve_binary"  # N-DECAY-2b (arm A as in 2's 190226Z)
ANCHOR_THREADS = 24  # the archive ran unpinned on a 24-core machine (its compute_env.json)
TRIPLE_ARCHIVE_RUN = "20261031T181540Z_file_tail_split_triple"  # R and T0 scored on triple
MEM_FRACTION = 0.8
UNIT_RSS_GB = 1.2  # one process holding the rows and fitting arm A (4-thread probe, R4-10)
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


def fit_predict(
    family: str,
    train_mask: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    n_jobs: int | None = None,
):
    """Tuned full-feature model on the train rows; predictions for every other row.

    `n_jobs` overrides the estimator's thread count (the zoo's -1 otherwise)."""
    pipe = build_preprocessor("full", n_components=None, seed=SEED)
    Ztr = pipe.fit_transform(X[train_mask])
    set_all_seeds(SEED)
    est = build(family, params, SEED)
    if n_jobs is not None:
        est.set_params(n_jobs=n_jobs)
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


def per_gap_class_curve(
    pred, y, gap, test_mask, max_gap: int | None = None, min_rows: int = MIN_ROWS
) -> list[dict]:
    """R4-10 (i): per-gap share of `test_mask` rows predicted correctly.

    `test_mask` selects one scenario class, whose rows share one true label in the binary task,
    so this is that class's recall (attack) or specificity (natural, no-event). The gap
    population rule is per_gap_curve's (MIN_ROWS test rows per gap); `min_rows=1` gives the
    unfiltered diagnostic curve of a class too small for the rule.
    """
    out = []
    gs = gap[test_mask]
    for g in np.unique(gs[np.isfinite(gs)]).astype(int):
        if max_gap is not None and g > max_gap:
            break
        sel = test_mask & (gap == g)
        n = int(sel.sum())
        if n < min_rows:
            continue
        out.append({"gap": int(g), "n": n, "accuracy": float((pred[sel] == y[sel]).mean())})
    return out


def scenario_class_map() -> dict[str, str]:
    """Scenario marker -> three-class label, read off the row-aligned triple and multiclass
    releases; fails loud if one marker carries two labels."""
    label_col = load_yaml(REPO_ROOT / "configs" / "data.yaml")["schema"]["label_column"]
    tri, multi = load_variant_frames("triple"), load_variant_frames("multiclass")
    if [len(f) for f in tri] != [len(m) for m in multi]:
        raise SystemExit("triple and multiclass releases are not row-aligned")
    pairs = pd.DataFrame(
        {
            "scen": np.concatenate([m[label_col].astype(str).to_numpy() for m in multi]),
            "cls": np.concatenate([f[label_col].astype(str).to_numpy() for f in tri]),
        }
    ).drop_duplicates()
    if pairs["scen"].duplicated().any():
        raise SystemExit(f"scenario markers with two three-class labels: {pairs}")
    return dict(zip(pairs["scen"], pairs["cls"], strict=True))


def row_classes(variant: str, y: np.ndarray, scen: np.ndarray) -> np.ndarray:
    """Three-class label name per row; checked against the variant's own labels."""
    labels = load_yaml(REPO_ROOT / "configs" / "data.yaml")["labels"]
    cmap = scenario_class_map()
    cls = np.array([cmap[s] for s in scen])
    if variant == "binary":
        expect = np.array([labels["binary"]["Attack" if c == "Attack" else "Natural"] for c in cls])
    else:
        expect = np.array([labels[variant][c] for c in cls])
    if not np.array_equal(expect, y):
        raise SystemExit(f"{variant}: scenario classes disagree with the row labels")
    return cls


def archived_scores(variant: str) -> dict[str, float | None]:
    """Archived scores of the prediction-table constructions; None where none was scored."""
    if variant == "binary":
        return dict(ARCHIVED)
    arms = json.loads(
        (REPO_ROOT / "results" / "runs" / TRIPLE_ARCHIVE_RUN / "results.json").read_text()
    )["arms"]
    got = {a["arm"]: a["macro_f1"] for a in arms}
    out: dict[str, float | None] = dict.fromkeys(ARCHIVED)
    out["R row-level"] = got["R row-level"]
    out["T0 ordered"] = got["T0 temporal"]  # positional_splits(block, "ordered"), same seed
    return out


def tuned_params(family: str, variant: str) -> dict:
    """The published full-feature Optuna winner for one family and variant."""
    name = f"cls_{family}_{variant}_full"
    db = (REPO_ROOT / "results" / f"optuna_{name}.db").as_posix()
    return optuna.load_study(study_name=name, storage=f"sqlite:///{db}").best_params


def anchor_check(
    fits: dict, curves: dict, families: list[str], family_threads: dict[str, int]
) -> dict:
    """Binary arm A against the archived N-DECAY-2b run: exact tau and exact matched curve.

    Blocking for every family fitted at a count that reproduces the archive. LightGBM
    reproduces at any count; XGBoost only at the archive's 24 (ruling 2026-11-16T13:05Z,
    item 4), so XGBoost at another count is reported, not raised.
    """
    root = REPO_ROOT / "results" / "runs" / ANCHOR_RUN
    arch = json.loads((root / "results.json").read_text())
    out = {}
    for fam in families:
        tau, tau_arch = (
            fits[f"A_{fam}"]["tau_released_rows"],
            arch["fits"][f"A_{fam}"]["tau_released_rows"],
        )
        exact = bool(
            tau == tau_arch and curves[f"A_{fam}_matched"] == arch["curves"][f"A_{fam}_matched"]
        )
        threads = family_threads.get(fam, THREADS)
        blocking = not (fam == "xgb" and threads != ANCHOR_THREADS)
        out[fam] = {
            "tau": tau,
            "tau_archived": tau_arch,
            "exact": exact,
            "blocking": blocking,
            "threads": threads,
            "archive_threads": ANCHOR_THREADS,
        }
        log.info("ANCHOR %s: %s", fam, out[fam])
        if blocking and not exact:
            raise SystemExit(f"ANCHOR FAILED ({fam}): tau {tau!r} vs archived {tau_arch!r}")
    return out


def _peak_rss_gb() -> float:
    mi = psutil.Process().memory_info()
    if sys.platform == "win32":
        return mi.peak_wset / 2**30
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def memory_guard(workers: int, rss_gb: float) -> dict:
    avail = psutil.virtual_memory().available / 2**30
    out = {"workers": workers, "unit_rss_gb": rss_gb, "available_gb": avail}
    log.info("memory: %d x %.2f GB vs %.2f GB available", workers, rss_gb, avail)
    if workers * rss_gb > MEM_FRACTION * avail:
        raise MemoryError(f"{workers} x {rss_gb} GB > {MEM_FRACTION:.0%} of {avail:.1f} GB free")
    return out


_W: dict = {}


def _init_worker(threads: int, X: np.ndarray, y: np.ndarray, params: dict, ft: dict) -> None:
    check_thread_env(threads)
    _W.update(X=X, y=y, params=params, family_threads=ft)


def _fit_unit(unit: tuple) -> tuple:
    """One independent fit: (key, family, train_mask) -> (key, predictions, wall s, peak GB)."""
    key, fam, train_mask = unit
    t = time.perf_counter()
    n_jobs = _W["family_threads"].get(fam)
    pred = fit_predict(fam, train_mask, _W["X"], _W["y"], _W["params"][fam], n_jobs)
    return key, pred, time.perf_counter() - t, _peak_rss_gb()


def unit_cache_path(cache_dir, unit: tuple, y: np.ndarray, params: dict, threads: int) -> Path:
    """Resume file of one fit, named by everything its predictions depend on: the arm, the
    family's tuned parameters, the train mask, the labels, the seed and the thread count."""
    key, fam, train_mask = unit
    h = hashlib.sha256()
    h.update(json.dumps([fam, params[fam], SEED, threads], sort_keys=True).encode())
    h.update(np.packbits(train_mask).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return cache_dir / f"{'_'.join(map(str, key))}_{h.hexdigest()[:16]}.npz"


def run_fits(
    units: list[tuple], X, y, params, workers: int, cache_dir, family_threads: dict[str, int]
) -> dict:
    """Every fit is independent (its own seed); one process per worker, results keyed.

    A family in `family_threads` fits in its own spawned process at that thread count; the
    rest run at --threads. Each finished fit is written to `cache_dir` at once, and a fit
    whose file exists is read back instead of refitted, so a stopped run resumes at its next
    unfinished arm.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    def threads_of(u: tuple) -> int:
        return family_threads.get(u[1], THREADS)

    done, todo, files = {}, [], {}
    for u in units:
        f = unit_cache_path(cache_dir, u, y, params, threads_of(u))
        if f.exists():
            z = np.load(f)
            done[u[0]] = (z["pred"], float(z["wall_s"]), float(z["peak_rss_gb"]))
            log.info("fit %-12s resumed from %s", "/".join(map(str, u[0])), f.name)
        else:
            todo.append(u)
            files[u[0]] = f

    def keep(u: tuple, res: tuple) -> None:
        key, pred, wall, rss = res
        np.savez(files[key], pred=pred, wall_s=wall, peak_rss_gb=rss)
        done[key] = (pred, wall, rss)
        log.info(
            "fit %-12s %.1f s | %d threads | peak RSS %.2f GB",
            "/".join(map(str, key)),
            wall,
            threads_of(u),
            rss,
        )

    run_grouped(
        todo,
        threads_of,
        _fit_unit,
        _init_worker,
        (X, y, params, family_threads),
        workers,
        THREADS,
        keep,
    )
    return {u[0]: done[u[0]] for u in units}


def fit_decay(pts: list[dict], key: str = "macro_f1") -> dict:
    g = np.array([p["gap"] for p in pts], dtype=float)
    f = np.array([p[key] for p in pts])
    w = np.array([p["n"] for p in pts], dtype=float)

    def model(gg, a, b, tau):
        return a + b * np.exp(-(gg - 1) / tau)

    tail = np.mean([p[key] for p in pts if ASYMPTOTE_GAPS[0] <= p["gap"] <= ASYMPTOTE_GAPS[1]])
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
    within = [p["gap"] for p in pts if p[key] - tail <= 0.02]
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
    ap.add_argument("--variant", choices=VARIANTS, default="binary")
    ap.add_argument("--threads", type=int, default=4, help="BLAS/OpenMP threads per process")
    ap.add_argument("--workers", type=int, default=1, help="processes fitting arms in parallel")
    ap.add_argument(
        "--family-threads",
        nargs="*",
        default=[f"xgb={ANCHOR_THREADS}"],
        help="FAMILY=N: fit that family at N threads in its own process (default xgb=24, the "
        "archive's count, so the XGBoost anchor is exact and blocking)",
    )
    args = ap.parse_args()
    if args.families[0] != "lgbm":
        raise SystemExit("the first family must be lgbm: arms B, C and E are LightGBM-only")
    variant = args.variant
    family_threads = parse_family_threads(args.family_threads)
    t_start = time.perf_counter()

    X, y, gid, scen, block = load_rows(variant)
    Xp, yp, splits_row = load_processed(variant)
    if X.shape != Xp.shape or not np.array_equal(y, yp):
        raise SystemExit("block reconstruction misaligned with the processed artifact")
    splits_row = {k: np.asarray(v) for k, v in splits_row.items()}
    cls = row_classes(variant, y, scen)
    n = len(y)
    pos_true = np.arange(n, dtype=np.int64)
    sizes = np.bincount(block)
    matched = sizes[block] >= MIN_BLOCK_MATCHED
    classes = list(load_yaml(REPO_ROOT / "configs" / "data.yaml")["labels"]["triple"])

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
        for c in classes:
            cm = te_a & matched & (cls == c)
            cc = {
                int(g): int(k)
                for g, k in zip(*np.unique(gap_a[cm], return_counts=True))
                if np.isfinite(g) and g <= GAP_FIT
            }
            print(
                f"SMOKE {variant} class {c}: blocks {len(np.unique(block[cls == c]))} "
                f"(matched {len(np.unique(block[(cls == c) & matched]))}) | "
                f"matched test rows per gap 1..{GAP_FIT}: {cc}"
            )
        print(f"SMOKE peak RSS {_peak_rss_gb():.2f} GB | {time.perf_counter() - t_start:.1f} s")
        return

    params = {f: tuned_params(f, variant) for f in args.families}
    mem = memory_guard(args.workers, UNIT_RSS_GB)
    cfg = {
        "variant": variant,
        "threads": THREADS,
        "family_threads": family_threads,
        "workers": args.workers,
        "memory_guard": mem,
        "classes": classes,
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
        "params_source": f"optuna studies cls_<family>_{variant}_full (best_params)",
        "cluster_partitions": 0 if args.no_cluster else args.cluster_partitions,
        "anchor_run": ANCHOR_RUN if variant == "binary" else None,
        "archived_scores": archived_scores(variant),
        "preregistration": "decision log 2026-09-15T19:09Z "
        "(N-DECAY-2b, amending 19:02Z and 18:52Z); R4-10 spikes 2026-11-16T11:07Z",
    }
    if args.no_cluster:
        labels = None
        cluster_stats = {}
    else:
        labels, cstats = build_clusters(X)
        cluster_stats = {k: v for k, v in cstats.items() if isinstance(v, (int, float))}
    cluster_parts = []
    if labels is not None:
        crng = np.random.default_rng(0)
        cluster_parts = [cluster_splits(labels, crng) for _ in range(args.cluster_partitions)]
    units = [(("A", fam), fam, tr_a) for fam in args.families]
    units += [(("B", "lgbm"), "lgbm", tr_b), (("C", "lgbm"), "lgbm", tr_c)]
    units += [(("E", "lgbm"), "lgbm", tr_e)]
    for k, sp in enumerate(cluster_parts):
        tm = np.zeros(n, dtype=bool)
        tm[sp["train"]] = True
        units.append((("cluster", k), "lgbm", tm))

    with RunDir(f"leak_decay_curve_{variant}", config=cfg, seeds=[SEED]) as run:
        out: dict = {"curves": {}, "fits": {}}
        fitted = run_fits(
            units,
            X,
            y,
            params,
            args.workers,
            REPO_ROOT / "models" / "leak_decay_curve" / variant,
            family_threads,
        )
        preds = {key: v[0] for key, v in fitted.items()}
        run.metrics.update(
            {
                "unit_wall_s": {"/".join(map(str, k)): v[1] for k, v in fitted.items()},
                "unit_peak_rss_gb": {"/".join(map(str, k)): v[2] for k, v in fitted.items()},
            }
        )
        # ---- arm A per family ------------------------------------------------------------
        for fam in args.families:
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
        anchor = None
        if variant == "binary":
            run.write_json("results_partial.json", out)
            anchor = anchor_check(out["fits"], out["curves"], args.families, family_threads)
        np.savez_compressed(
            run.path / "arm_a_predictions.npz",
            **{f"pred_{fam}": preds[("A", fam)] for fam in args.families},
            y=y,
            gap=gap_a,
            test=te_a,
            matched=matched,
            block=block,
            cls=cls,
        )
        # ---- R4-10 (i): arm A re-aggregated by scenario class ----------------------------
        out["class_curves"], out["class_fits"] = {}, {}
        for fam in args.families:
            for c in classes:
                for scope, mask in (("matched", te_a & matched), ("all", te_a)):
                    out["class_curves"][f"A_{fam}_{c}_{scope}"] = per_gap_class_curve(
                        preds[("A", fam)],
                        y,
                        gap_a,
                        mask & (cls == c),
                        GAP_FIT if scope == "matched" else None,
                    )
                pts = out["class_curves"][f"A_{fam}_{c}_matched"]
                if {p["gap"] for p in pts} >= set(range(1, GAP_FIT + 1)):
                    out["class_fits"][f"A_{fam}_{c}"] = fit_decay(pts, key="accuracy")
                else:
                    # below the population rule: the fit is not the instrument's; the unfiltered
                    # curve and its fit are kept as a labelled diagnostic, never as the tau
                    raw = per_gap_class_curve(
                        preds[("A", fam)], y, gap_a, te_a & matched & (cls == c), GAP_FIT, 1
                    )
                    out["class_curves"][f"A_{fam}_{c}_matched_unfiltered"] = raw
                    out["class_fits"][f"A_{fam}_{c}"] = {
                        "unsupported": f"gaps 1..{GAP_FIT} do not all reach {MIN_ROWS} "
                        "matched test rows",
                        "rows_per_gap": [p["n"] for p in raw],
                        "diagnostic_fit_below_min_rows": fit_decay(raw, key="accuracy"),
                    }
                    log.warning(
                        "class %s (%s): below the population rule on the matched set", c, fam
                    )
                log.info("A %-4s class %-8s fit: %s", fam, c, out["class_fits"][f"A_{fam}_{c}"])
        # ---- arm B (negative control), C (diagnostic), E (two-sided) ----------------------
        out["curves"]["B_lgbm_matched"] = per_gap_curve(
            preds[("B", "lgbm")], y, gap_b, te_b & matched, y[tr_b], GAP_FIT
        )
        out["curves"]["B_lgbm_all"] = per_gap_curve(preds[("B", "lgbm")], y, gap_b, te_b, y[tr_b])
        out["curves"]["C_lgbm_matched"] = per_gap_curve(
            preds[("C", "lgbm")], y, gap_c, te_c & matched, y[tr_c], GAP_H1
        )
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
        archived = archived_scores(variant)
        tables = {"file_gap": {}, "block_gap": {}}
        for name, sp in arms.items():
            for key, unit in (("file_gap", gid), ("block_gap", block)):
                r = predict_arm(sp["test"], sp["train"], unit, pos_true, curve_a, asym)
                r["actual_archived"] = archived[name]
                r["abs_error"] = (
                    None if archived[name] is None else abs(r["predicted"] - archived[name])
                )
                tables[key][name] = r
            log.info(
                "%-14s archived %s | file-gap pred %.4f (err %s) | block-gap pred %.4f (err %s)",
                name,
                archived[name],
                tables["file_gap"][name]["predicted"],
                tables["file_gap"][name]["abs_error"],
                tables["block_gap"][name]["predicted"],
                tables["block_gap"][name]["abs_error"],
            )

        # ---- cluster split: regenerated, scored fresh, predicted (positive control) -------
        cluster = []
        for k, sp in enumerate(cluster_parts):
            pred_c = preds[("cluster", k)]
            actual = float(f1_score(y[sp["test"]], pred_c[sp["test"]], average="macro"))
            r = predict_arm(sp["test"], sp["train"], gid, pos_true, curve_a, asym)
            r.update(
                {"partition": k, "actual_fresh": actual, "abs_error": abs(r["predicted"] - actual)}
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
                v["abs_error"] for v in tables["file_gap"].values() if v["abs_error"] is not None
            ),
            "prediction_max_abs_error_block_gap": max(
                v["abs_error"] for v in tables["block_gap"].values() if v["abs_error"] is not None
            ),
            "prediction_max_abs_error_block_gap_excl_dense": max(
                v["abs_error"]
                for k, v in tables["block_gap"].items()
                if k not in ("R row-level", "S shuffled") and v["abs_error"] is not None
            ),
            "cluster_archived_mean": ARCHIVED_CLUSTER_MEAN,
            "anchor": anchor,
            "class_tau_released_rows": {
                k: v.get("tau_released_rows") for k, v in out["class_fits"].items()
            },
        }
        if not verdict["controls_pass"]:
            log.error("CONTROL FAILED (run VOID): negative=%s positive=%s", ctrl_neg, ctrl_pos)
        run.metrics.update(verdict)
        run.metrics["run_peak_rss_gb"] = _peak_rss_gb()
        run.metrics["run_wall_s"] = time.perf_counter() - t_start
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
