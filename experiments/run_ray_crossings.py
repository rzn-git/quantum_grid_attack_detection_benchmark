"""P17e: measure the interleaving mechanism directly instead of inferring it.

THE CLAIM UNDER TEST (results.tex): "the fidelity kernel's decision regions interleave at fine
scale, so [HopSkipJump's] search terminates far away and the large distortion masquerades as
robustness." It is stated as fact and is load-bearing for three results, but it rests on two
INFERENTIAL legs (the random-perturbation control, the white-box ranking inversion) since
P22-final withdrew the restart-variance leg. This samples f densely along each perturbation ray
and counts sign changes -- the quantity the mechanism is about.

PRE-REGISTERED PREDICTIONS (interleaving account):
  H1 crossings-per-ray rises with bandwidth over the healthy->wide range.
  H2 the PARITY model explains the flip rate: flip <=> an ODD number of crossings, so the
     measured flip rate should equal P(odd) computed from the same rays. This is the sharp
     one -- it is what makes "many crossings" produce a flip-rate that saturates and FALLS
     while a single-crossing model keeps rising (the co-peak P17d could not reproduce).
  H3 the matched classical control (tuned RBF-SVM) shows far fewer crossings at its own
     accuracy optimum -- otherwise "interleaving" is not specific to the fidelity kernel.

KILL CRITERION (pre-registered): the account FAILS if either
  (a) crossings-per-ray is flat or falling in bandwidth across the healthy->wide range, or
  (b) the parity prediction misses the measured flip rate by > 0.10 absolute at the
      accuracy-optimal bandwidth.
Either outcome means the interleaving sentence in results.tex is not supported and must be
rewritten as an open question. A confirmation upgrades it from inference to measurement.

CONTROLS: (i) endpoint check -- the ray endpoint's sign must agree with the flip actually
measured by the deployed predict path, else the scan is not sampling the same function;
(ii) degenerate bandwidth (0.1) has collapsed to the prior, so it must show ~0 crossings --
a scan that reports interleaving there is measuring numerical noise, not geometry.

Run:  python -m experiments.run_ray_crossings --n-samples 200

Graduated from tmp/ because make_report consumes its run dir for the paper's
ray_crossings_binary_q8 table; the RunDir name keeps its original `probe_` prefix so
already-recorded results still resolve.

R4-17(b) (decision log 2026-11-16T11:34Z, round-4 part-B amendment). Two modes added:

  --dump-rays   rerun the scan and write each ray's crossing count, crossing sample indices and
                endpoint flip to `rays.npz`, under the RunDir `ray_crossings_dump_<variant>_q<q>`
                (no `probe_` prefix, so make_report never picks it up). Blocking anchor: every
                bandwidth's summary row equals the archived P17e row
                (`20260815T210829Z_probe_ray_crossings_binary_q8`) exactly. Bandwidth cells run
                in a process pool (`--workers`, `--threads` pinned before numpy is imported).
  --count-model QDIR CDIR
                fit the crossing-count models to the dumped rays of the quantum (QDIR) and the
                classical (CDIR, `run_ray_crossings_classical --dump-rays`) scans:
                  M0 Poisson; M1 negative binomial NB(mu, r), the independent-crossings null with
                  ray-varying rate; M2 pair model C = B + 2K, B ~ Bernoulli(q), K ~ NB.
                Statistic: d_par = observed odd-count share - P_NB(odd), P_NB(odd) =
                (1 - (r/(r+2 mu))^r)/2 (NB has no parity parameter, so this is a prediction),
                with a parametric-bootstrap 95% band from the fitted NB; and dAIC = AIC(M1) -
                AIC(M2).
                  PAIRING       at bandwidth 1.6: d_par <= -0.10, below the band, dAIC >= 4
                  KILL          d_par inside the band at 0.4, 0.8 and 1.6, and NB goodness of
                                fit p >= 0.05 at each
                  INTERMEDIATE  otherwise
                Bandwidth 3.0 is reported; all five cells (the ruled degeneracy rule keeps them)
                are a sensitivity reading. Blocking synthetic controls run first: replicates from
                M2 (q 0.34, K ~ Poisson(1.75)) must read PAIRING in >= 80%, replicates from
                NB(3.84, 1) must land inside the band in >= 90%.

Ruling 7 (decision log 2026-11-16T13:05Z): the controls failed at 100 rays (positive 0.79), so
`--controls-only 200 300 400` reruns the synthetic controls alone (no real ray) at each count and
names the smallest count with positive >= 0.80 and negative in-band >= 0.95, or
UNDERPOWERED_AT_400_RAYS. `--n-rays N` then dumps N rays per cell: the archived 100-ray block
unchanged first (its summary rows are the anchor), then N - 100 rays from the remaining test rows
on their own stream (`ray_set`). `--count-model` reruns the controls at the dumps' ray count.

Run:  python -m experiments.run_ray_crossings --controls-only 200 300 400 --workers 4 --threads 1
      python -m experiments.run_ray_crossings --dump-rays --n-rays <N> --workers 2 --threads 2
      OMP_NUM_THREADS=2 python -m experiments.run_ray_crossings_classical --dump-rays --n-rays <N>
      python -m experiments.run_ray_crossings --count-model <QDIR> <CDIR> --workers 3 --threads 2
"""

from __future__ import annotations

import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=None)
THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
if THREADS is not None:
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_var] = str(THREADS)

import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import gammaln  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from experiments.run_ablations import kernel_subset  # noqa: E402
from qgridbench.models.classical.zoo import stratified_cap  # noqa: E402
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel  # noqa: E402
from qgridbench.protocol import prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402

log = get_logger(__name__)

ARCHIVED_RUN = "results/runs/20260815T210829Z_probe_ray_crossings_binary_q8"
ARCHIVED_CLASSICAL_RUN = "results/runs/20260815T211558Z_probe_ray_crossings_classical_binary_q8"

# R4-17(b) registered thresholds and design constants (amendment 2026-11-16T11:34Z)
PAIR_DPAR = -0.10
PAIR_DAIC = 4.0
GOF_ALPHA = 0.05
IN_SCOPE_BW = (0.4, 0.8, 1.6)
PAIRING_BW = 1.6
N_BOOT = 2000
GOF_BINS = 10  # counts 0..9 plus a >= 10 tail bin
CTRL_REPLICATES = 200
CTRL_N_BOOT = 500
CTRL_POS = {"q": 0.34, "k_mean": 1.75, "min_pass": 0.80}
CTRL_NEG = {"mu": 3.84, "r": 1.0, "min_pass": 0.95}  # ruling 7, 2026-11-16T13:05Z
CTRL_RAY_COUNTS = (200, 300, 400)  # ruling 7: the smallest passing count is used
EXTRA_RAY_SEED = 1  # stream of the rays beyond the archived 100-ray block
LOG_R_BOUNDS = (-10.0, 16.0)  # r in [4.5e-5, 8.9e6]; the top is the Poisson limit
WORKER_RSS_GB = 0.86  # one quantum cell at 2 threads, tmp/r4_tier3/probe_ray_cost.json
MEM_FRACTION = 0.8


def memory_guard(workers: int, rss_gb: float) -> None:
    """W x peak RSS must fit in 80% of the RAM available now; fail loud otherwise."""
    import psutil

    avail = psutil.virtual_memory().available / 2**30
    log.info("memory: %d x %.2f GB vs %.2f GB available", workers, rss_gb, avail)
    if workers * rss_gb > MEM_FRACTION * avail:
        raise MemoryError(f"{workers} x {rss_gb:.2f} GB > {MEM_FRACTION:.0%} of {avail:.2f} GB")


def _fit_qsvm(A_tr, y_tr, enc, bw, C_grid):
    s_tr = compute_states(A_tr, enc, bandwidth=bw)
    K_tr = fidelity_kernel(s_tr, s_tr)

    def cv(C):
        sc = []
        for t, v in StratifiedKFold(3, shuffle=True, random_state=0).split(K_tr, y_tr):
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_tr[np.ix_(t, t)], y_tr[t])
            sc.append(f1_score(y_tr[v], clf.predict(K_tr[np.ix_(v, t)]), average="macro"))
        return float(np.mean(sc))

    best_C = max(C_grid, key=cv)
    return (
        s_tr,
        SVC(kernel="precomputed", C=best_C, class_weight="balanced").fit(K_tr, y_tr),
        best_C,
    )


def ray_stats(F: np.ndarray) -> dict:
    """Per-ray crossing counts, crossing sample indices and endpoint flips of F (samples, rays)."""
    signs = np.sign(F)
    signs[signs == 0] = 1.0
    changes = np.diff(signs, axis=0) != 0
    return {
        "counts": changes.sum(axis=0),
        "crossing_index": [np.flatnonzero(changes[:, j]) for j in range(F.shape[1])],
        "flip": np.sign(F[0]) != np.sign(F[-1]),
    }


def ray_set(reg, eval_subset: int, n_rays: int, radius: float) -> list[tuple]:
    """The ray blocks [(X, y, delta), ...]: the archived block first, then any extra rays.

    Block 0 is the archived P17e set exactly (stratified test cap `eval_subset`, seed 0; deltas
    from `default_rng(0)`). Rays beyond it (R4-17(b), ruling 7 of 2026-11-16T13:05Z) are a
    stratified draw from the REMAINING test rows (seed EXTRA_RAY_SEED) with deltas from their own
    `default_rng(EXTRA_RAY_SEED)` stream, so the archived block's rays are untouched. Blocks are
    evaluated separately so the archived block's decision values come from the same call shape.
    """
    Xt, yt = reg.X["test"], reg.y["test"]
    ev = stratified_cap(Xt, yt, eval_subset, seed=0)
    blocks = [(Xt[ev], yt[ev], np.random.default_rng(0).uniform(-1.0, 1.0, (len(ev), Xt.shape[1])))]
    if n_rays < eval_subset:
        raise ValueError(f"n_rays {n_rays} < the archived block of {eval_subset}")
    if n_rays > eval_subset:
        rest = np.setdiff1d(np.arange(len(yt)), ev)
        extra = rest[stratified_cap(Xt[rest], yt[rest], n_rays - eval_subset, seed=EXTRA_RAY_SEED)]
        d = np.random.default_rng(EXTRA_RAY_SEED).uniform(-1.0, 1.0, (len(extra), Xt.shape[1]))
        blocks.append((Xt[extra], yt[extra], d))
    return [(X, y, d * radius * reg.feature_scale) for X, y, d in blocks]


def summary_row(F: np.ndarray, y: np.ndarray, st: dict) -> dict:
    """P17e's per-bandwidth summary of one ray block (bandwidth and C are added by the caller)."""
    crossings = st["counts"]
    flip_measured = float(np.mean(st["flip"]))
    parity_pred = float(np.mean(crossings % 2 == 1))
    return {
        "clean_macro_f1": float(f1_score(y, (F[0] > 0).astype(int), average="macro")),
        "crossings_mean": float(crossings.mean()),
        "crossings_median": float(np.median(crossings)),
        "crossings_max": int(crossings.max()),
        "frac_rays_multi_crossing": float(np.mean(crossings > 1)),
        "flip_rate_measured": flip_measured,
        "flip_rate_parity_prediction": parity_pred,
        "parity_abs_error": abs(parity_pred - flip_measured),
    }


def scan_cell(variant: str, bw: float, n_samples: int, radius: float, n_rays: int) -> dict:
    """One bandwidth: fit the QSVM, sample f along every ray, return the summary rows and rays.

    `row` summarises the archived 100-ray block (the anchor); `row_all` all `n_rays` rays.
    """
    import psutil

    if THREADS is not None and os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise AssertionError(f"thread cap {os.environ.get('OMP_NUM_THREADS')} != {THREADS}")
    cfg = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")["ablations"]["bandwidth_fragility"]
    enc, q = cfg["cell"]
    reg = prepare_regime(variant, "pca", n_components=q, seed=0)
    sub = kernel_subset(reg, 2000, seed=0)
    A_tr, y_tr = reg.angles["train"][sub], reg.y["train"][sub]
    # ONE perturbation direction per point, shared across bandwidths -> paired comparison
    blocks = ray_set(reg, cfg["eval_subset"], n_rays, radius)
    ts = np.linspace(0.0, 1.0, n_samples)

    s_tr, clf, best_C = _fit_qsvm(A_tr, y_tr, enc, bw, cfg["svm_C_grid"])

    def f(X):
        A = reg.angle_scaler.transform(X)
        return clf.decision_function(fidelity_kernel(compute_states(A, enc, bw), s_tr))

    Fs = [np.stack([f(X + t * d) for t in ts]) for X, _y, d in blocks]
    F = np.concatenate(Fs, axis=1)
    y_all = np.concatenate([y for _X, y, _d in blocks])
    st = ray_stats(F)
    head = {"bandwidth": float(bw), "best_C": float(best_C)}
    row = {**head, **summary_row(Fs[0], blocks[0][1], ray_stats(Fs[0]))}
    row_all = {**head, **summary_row(F, y_all, st), "n_rays": int(F.shape[1])}
    rss = psutil.Process().memory_info()
    peak = (rss.peak_wset if sys.platform == "win32" else rss.rss) / 2**30
    return {"row": row, "row_all": row_all, "rays": st, "peak_rss_gb": float(peak)}


def save_rays(path: Path, cells: dict) -> None:
    """cells: {setting: ray_stats dict}. Ragged crossing indices are stored flat with offsets."""
    out = {}
    for key, st in cells.items():
        idx = st["crossing_index"]
        out[f"{key}__counts"] = np.asarray(st["counts"], dtype=np.int64)
        out[f"{key}__flip"] = np.asarray(st["flip"], dtype=bool)
        out[f"{key}__index_flat"] = (
            np.concatenate(idx).astype(np.int64) if idx else np.zeros(0, np.int64)
        )
        out[f"{key}__index_offsets"] = np.cumsum([0] + [len(i) for i in idx]).astype(np.int64)
    np.savez(path, **out)


def load_rays(path: Path) -> dict:
    z = np.load(path)
    keys = sorted({k.split("__")[0] for k in z.files}, key=float)
    out = {}
    for k in keys:
        off = z[f"{k}__index_offsets"]
        flat = z[f"{k}__index_flat"]
        out[float(k)] = {
            "counts": z[f"{k}__counts"],
            "flip": z[f"{k}__flip"],
            "crossing_index": [flat[off[i] : off[i + 1]] for i in range(len(off) - 1)],
        }
    return out


# ---- R4-17(b) crossing-count models ---------------------------------------------------------


def nb_loglik(c: np.ndarray, mu: float, r: float) -> float:
    c = np.asarray(c, dtype=float)
    if mu == 0.0:
        return 0.0 if np.all(c == 0) else -np.inf
    return float(
        np.sum(
            gammaln(c + r)
            - gammaln(r)
            - gammaln(c + 1)
            - r * np.log1p(mu / r)
            + c * (np.log(mu) - np.log(r + mu))
        )
    )


def fit_nb(c: np.ndarray) -> dict:
    """NB maximum likelihood. For fixed r the MLE of mu is the sample mean, so r is profiled."""
    c = np.asarray(c, dtype=float)
    mu = float(c.mean())
    if mu == 0.0:
        return {"mu": 0.0, "r": float("inf"), "ll": 0.0}
    res = minimize_scalar(
        lambda lr: -nb_loglik(c, mu, float(np.exp(lr))), bounds=LOG_R_BOUNDS, method="bounded"
    )
    r = float(np.exp(res.x))
    return {"mu": mu, "r": r, "ll": nb_loglik(c, mu, r)}


def nb_p_odd(mu: float, r: float) -> float:
    """P(odd) of NB(mu, r) = (1 - G(-1))/2, G(-1) = (r/(r+2 mu))^r; Poisson limit as r -> inf."""
    if mu == 0.0:
        return 0.0
    if not np.isfinite(r):
        return float((1.0 - np.exp(-2.0 * mu)) / 2.0)
    return float((1.0 - np.exp(-r * np.log1p(2.0 * mu / r))) / 2.0)


def fit_models(c: np.ndarray) -> dict:
    c = np.asarray(c, dtype=np.int64)
    lam = float(c.mean())
    ll0 = float(np.sum(c * np.log(lam) - lam - gammaln(c + 1))) if lam > 0 else 0.0
    m1 = fit_nb(c)
    n_odd = int(np.sum(c % 2 == 1))
    q = n_odd / len(c)
    llb = (n_odd * np.log(q) if n_odd else 0.0) + (
        (len(c) - n_odd) * np.log(1 - q) if n_odd < len(c) else 0.0
    )
    k = fit_nb(c // 2)
    ll2 = float(llb + k["ll"])
    return {
        "poisson": {"lambda": lam, "ll": ll0, "aic": 2 * 1 - 2 * ll0},
        "nb": {**m1, "aic": 2 * 2 - 2 * m1["ll"]},
        "pair": {"q": q, "k_mu": k["mu"], "k_r": k["r"], "ll": ll2, "aic": 2 * 3 - 2 * ll2},
        "p_odd_emp": q,
        "p_odd_nb": nb_p_odd(m1["mu"], m1["r"]),
    }


def _gof_stat(c: np.ndarray, mu: float, r: float) -> float:
    """Pearson chi-square of the count histogram (0..9, >=10) against NB(mu, r)."""
    from scipy.stats import nbinom, poisson

    obs = np.bincount(np.minimum(c, GOF_BINS), minlength=GOF_BINS + 1).astype(float)
    ks = np.arange(GOF_BINS)
    if mu == 0.0:
        pmf = np.r_[1.0, np.zeros(GOF_BINS)]
    elif not np.isfinite(r) or r > 1e6:
        pmf = np.r_[poisson.pmf(ks, mu), poisson.sf(GOF_BINS - 1, mu)]
    else:
        p = r / (r + mu)
        pmf = np.r_[nbinom.pmf(ks, r, p), nbinom.sf(GOF_BINS - 1, r, p)]
    exp_ = len(c) * pmf
    ok = exp_ > 0
    return float(np.sum((obs[ok] - exp_[ok]) ** 2 / exp_[ok]))


def _draw_nb(rng, mu, r, n):
    if mu == 0.0:
        return np.zeros(n, dtype=np.int64)
    if not np.isfinite(r) or r > 1e6:
        return rng.poisson(mu, n)
    return rng.negative_binomial(r, r / (r + mu), n)


def score_cell(c: np.ndarray, n_boot: int, seed: int, gof: bool = True) -> dict:
    """d_par with its NB parametric-bootstrap band, dAIC, NB goodness of fit, and the cell call."""
    c = np.asarray(c, dtype=np.int64)
    fit = fit_models(c)
    d_par = fit["p_odd_emp"] - fit["p_odd_nb"]
    rng = np.random.default_rng(seed)
    mu, r = fit["nb"]["mu"], fit["nb"]["r"]
    g_obs = _gof_stat(c, mu, r) if gof else None
    dpars, gstats = [], []
    for _ in range(n_boot):
        cb = _draw_nb(rng, mu, r, len(c))
        fb = fit_nb(cb)
        dpars.append(float(np.mean(cb % 2 == 1)) - nb_p_odd(fb["mu"], fb["r"]))
        if gof:
            gstats.append(_gof_stat(cb, fb["mu"], fb["r"]))
    lo, hi = (float(x) for x in np.percentile(dpars, [2.5, 97.5]))
    d_aic = fit["nb"]["aic"] - fit["pair"]["aic"]
    return {
        **fit,
        "d_par": float(d_par),
        "band": [lo, hi],
        "in_band": bool(lo <= d_par <= hi),
        "d_aic_nb_minus_pair": float(d_aic),
        "pairing_cell": bool(d_par <= PAIR_DPAR and d_par < lo and d_aic >= PAIR_DAIC),
        "nb_gof_p": (
            float((np.sum(np.array(gstats) >= g_obs) + 1) / (n_boot + 1)) if gof else None
        ),
    }


def _control_replicate(args: tuple) -> dict:
    kind, i, n_rays = args
    rng = np.random.default_rng([17, 0 if kind == "pos" else 1, i])
    if kind == "pos":
        b = rng.random(n_rays) < CTRL_POS["q"]
        c = b.astype(np.int64) + 2 * rng.poisson(CTRL_POS["k_mean"], n_rays)
    else:
        c = _draw_nb(rng, CTRL_NEG["mu"], CTRL_NEG["r"], n_rays)
    s = score_cell(c, CTRL_N_BOOT, seed=10_000 + i, gof=False)
    return {
        "kind": kind,
        "d_par": s["d_par"],
        "in_band": s["in_band"],
        "pairing": s["pairing_cell"],
    }


def synthetic_controls(workers: int, n_rays: int) -> dict:
    jobs = [(k, i, n_rays) for k in ("pos", "neg") for i in range(CTRL_REPLICATES)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_control_replicate, jobs, chunksize=8))
    pos = [r for r in res if r["kind"] == "pos"]
    neg = [r for r in res if r["kind"] == "neg"]
    out = {
        "positive_pairing_rate": float(np.mean([r["pairing"] for r in pos])),
        "negative_in_band_rate": float(np.mean([r["in_band"] for r in neg])),
        "positive_mean_d_par": float(np.mean([r["d_par"] for r in pos])),
        "negative_mean_d_par": float(np.mean([r["d_par"] for r in neg])),
        "d_par_not_identically_zero": bool(np.ptp([r["d_par"] for r in res]) > 0),
        "replicates": CTRL_REPLICATES,
        "n_boot": CTRL_N_BOOT,
        "n_rays": n_rays,
    }
    out["pass"] = bool(
        out["positive_pairing_rate"] >= CTRL_POS["min_pass"]
        and out["negative_in_band_rate"] >= CTRL_NEG["min_pass"]
        and out["d_par_not_identically_zero"]
    )
    return out


def gap_check(st: dict, n_samples: int, short: float = 0.05) -> dict:
    """Share of consecutive-crossing gaps < `short` of the ray, against uniform positions."""
    obs, exp_w, n_gaps = 0, 0.0, 0
    for idx in st["crossing_index"]:
        k = len(idx)
        if k < 2:
            continue
        pos = (np.asarray(idx) + 0.5) / (n_samples - 1)
        obs += int(np.sum(np.diff(pos) < short))
        exp_w += (k - 1) * (1.0 - (1.0 - short) ** k)
        n_gaps += k - 1
    return {
        "n_gaps": n_gaps,
        "obs_short_share": obs / n_gaps if n_gaps else None,
        "uniform_short_share": exp_w / n_gaps if n_gaps else None,
    }


def controls_sweep(workers: int, counts: list[int]) -> None:
    """Ruling 7: synthetic controls only (no real ray is read) at each rays-per-cell count; the
    smallest count passing both controls is the one the run of record dumps."""
    with RunDir("crossing_count_controls_binary_q8", config={"ray_counts": counts}) as run:
        sweep = {}
        for n in counts:
            sweep[str(n)] = synthetic_controls(workers, n)
            log.info("controls at %d rays: %s", n, sweep[str(n)])
            run.write_json("results_partial.json", sweep)
        passing = [n for n in counts if sweep[str(n)]["pass"]]
        chosen = min(passing) if passing else None
        res = {
            "sweep": sweep,
            "chosen_n_rays": chosen,
            "verdict": "use_n_rays" if chosen else "UNDERPOWERED_AT_400_RAYS",
            "rule": "smallest count with positive >= 0.80 and negative in-band >= 0.95",
        }
        run.write_json("results.json", res)
        run.metrics.update({"chosen_n_rays": chosen, "verdict": res["verdict"]})
        log.info("R4-17(b) controls sweep: %s", res["verdict"] + f" ({chosen})")
        print(f"-> {run.path}")


def count_model(qdir: str, cdir: str, workers: int, n_samples: int) -> None:
    qr = load_rays(REPO_ROOT / qdir / "rays.npz")
    cr = load_rays(REPO_ROOT / cdir / "rays.npz")
    n_rays = {len(st["counts"]) for rays in (qr, cr) for st in rays.values()}
    if len(n_rays) != 1:
        raise RuntimeError(f"dumps carry different ray counts per cell: {n_rays}")
    (n_rays,) = n_rays
    ctrl = synthetic_controls(workers, n_rays)
    log.info("synthetic controls at %d rays: %s", n_rays, ctrl)
    if not ctrl["pass"]:
        raise RuntimeError(f"R4-17(b) synthetic controls FAILED: {ctrl}")
    cfg = {
        "quantum_dump": qdir,
        "classical_dump": cdir,
        "n_rays": n_rays,
        "n_boot": N_BOOT,
        "amendment": "decision log 2026-11-16T11:34Z",
    }
    with RunDir("crossing_count_model_binary_q8", config=cfg, seeds=[0]) as run:
        cells = {"quantum": {}, "classical": {}}
        for fam, rays in (("quantum", qr), ("classical", cr)):
            for j, (setting, st) in enumerate(sorted(rays.items())):
                s = score_cell(st["counts"], N_BOOT, seed=j + (100 if fam == "classical" else 0))
                s["flip_rate"] = float(np.mean(st["flip"]))
                s["gap_check"] = gap_check(st, n_samples)
                cells[fam][str(setting)] = s
                log.info(
                    "%s %g: odd %.2f NB %.3f d_par %+.3f band [%+.3f, %+.3f] dAIC %.1f gof p %.3f",
                    fam,
                    setting,
                    s["p_odd_emp"],
                    s["p_odd_nb"],
                    s["d_par"],
                    *s["band"],
                    s["d_aic_nb_minus_pair"],
                    s["nb_gof_p"],
                )

        def branch(scope):
            q = cells["quantum"]
            if q[str(PAIRING_BW)]["pairing_cell"]:
                return "PAIRING"
            if all(q[str(b)]["in_band"] and q[str(b)]["nb_gof_p"] >= GOF_ALPHA for b in scope):
                return "KILL"
            return "INTERMEDIATE"

        all_bw = [float(k) for k in cells["quantum"]]
        res = {
            "branch": branch(IN_SCOPE_BW),
            "branch_sensitivity_all_five_cells": branch(all_bw),
            "controls": {"synthetic": ctrl, "pass": True},
            "cells": cells,
        }
        run.write_json("results.json", res)
        run.metrics.update(
            {"branch": res["branch"], **{"synthetic_" + k: v for k, v in ctrl.items()}}
        )
        log.info(
            "R4-17(b) BRANCH %s (all five: %s)",
            res["branch"],
            res["branch_sensitivity_all_five_cells"],
        )
        print(f"-> {run.path}")


def _anchor(rows: list[dict], archived_path: str) -> dict:
    arch = {
        r["bandwidth"]: r
        for r in json.loads((REPO_ROOT / archived_path / "results.json").read_text())["rows"]
    }
    diffs = {
        f"{r['bandwidth']}:{k}": (v, arch[r["bandwidth"]][k])
        for r in rows
        for k, v in r.items()
        if arch[r["bandwidth"]][k] != v
    }
    return {"archived": archived_path, "mismatches": diffs, "pass": not diffs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--n-samples", type=int, default=200, help="points sampled along each ray")
    ap.add_argument("--radius", type=float, default=0.5, help="ray length in sigma")
    ap.add_argument("--threads", type=int, default=None, help="pinned before numpy import")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dump-rays", action="store_true", help="R4-17(b): write rays.npz + anchor")
    ap.add_argument("--bandwidths", type=float, nargs="*", default=None)
    ap.add_argument("--count-model", nargs=2, metavar=("QDIR", "CDIR"), default=None)
    ap.add_argument(
        "--n-rays", type=int, default=None, help="rays per cell (default: the archived 100)"
    )
    ap.add_argument(
        "--controls-only",
        type=int,
        nargs="*",
        default=None,
        metavar="N_RAYS",
        help="ruling 7: synthetic controls at these ray counts (default 200 300 400); no real ray",
    )
    ap.add_argument(
        "--rss-gb",
        type=float,
        default=WORKER_RSS_GB,
        help="measured peak RSS per worker for the memory guard",
    )
    args = ap.parse_args()
    memory_guard(args.workers, args.rss_gb)

    if args.controls_only is not None:
        controls_sweep(args.workers, args.controls_only or list(CTRL_RAY_COUNTS))
        return
    if args.count_model:
        count_model(*args.count_model, args.workers, args.n_samples)
        return

    cfg = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")["ablations"]["bandwidth_fragility"]
    q = cfg["cell"][1]
    bws = args.bandwidths or cfg["bandwidths"]
    name = (
        f"{'ray_crossings_dump' if args.dump_rays else 'probe_ray_crossings'}_{args.variant}_q{q}"
    )
    n_rays = args.n_rays or cfg["eval_subset"]
    params = {k: v for k, v in vars(args).items() if k not in ("count_model", "controls_only")}

    with RunDir(name, config=params) as run:
        jobs = [(args.variant, bw, args.n_samples, args.radius, n_rays) for bw in bws]
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as ex:
                outs = list(ex.map(scan_cell, *zip(*jobs)))
        else:
            outs = [scan_cell(*j) for j in jobs]
        rows = [o["row"] for o in outs]
        for r in rows:
            log.info(
                "bw=%.2f  F1 %.3f | crossings mean %.2f (max %d, >1 on %.0f%%) | "
                "flip measured %.3f vs parity %.3f",
                r["bandwidth"],
                r["clean_macro_f1"],
                r["crossings_mean"],
                r["crossings_max"],
                100 * r["frac_rays_multi_crossing"],
                r["flip_rate_measured"],
                r["flip_rate_parity_prediction"],
            )

        if args.dump_rays:
            save_rays(
                run.path / "rays.npz", {str(r["bandwidth"]): o["rays"] for r, o in zip(rows, outs)}
            )
            anchor = _anchor(rows, ARCHIVED_RUN) if args.n_samples == 200 else None
            peak = {str(r["bandwidth"]): o["peak_rss_gb"] for r, o in zip(rows, outs)}
            run.write_json(
                "results.json",
                {
                    "rows": rows,
                    "rows_all_rays": [o["row_all"] for o in outs],
                    "anchor": anchor,
                    "anchor_note": "rows summarise the archived 100-ray block of every cell",
                    "peak_rss_gb": peak,
                },
            )
            run.metrics.update({"anchor_pass": None if anchor is None else anchor["pass"]})
            log.info("anchor %s | peak RSS per cell %s", anchor, peak)
            if anchor is not None and not anchor["pass"]:
                raise RuntimeError(f"R4-17(b) anchor FAILED: {anchor['mismatches']}")
            print(f"-> {run.path}")
            return

        # --- pre-registered verdict --------------------------------------------------
        healthy = [r for r in rows if r["bandwidth"] >= 0.4]
        cm = [r["crossings_mean"] for r in healthy]
        rising = all(b >= a - 1e-9 for a, b in zip(cm, cm[1:]))
        peak = max(rows, key=lambda r: r["clean_macro_f1"])
        verdict = {
            "H1_crossings_rise_with_bandwidth": bool(rising),
            "H1_crossings_healthy_range": cm,
            "H2_parity_error_at_accuracy_peak": peak["parity_abs_error"],
            "H2_parity_holds": bool(peak["parity_abs_error"] <= 0.10),
            "accuracy_peak_bandwidth": peak["bandwidth"],
            "control_degenerate_bw_crossings": rows[0]["crossings_mean"],
            "KILLED": bool(not rising or peak["parity_abs_error"] > 0.10),
        }
        run.write_json("results.json", {"rows": rows, "verdict": verdict})
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
