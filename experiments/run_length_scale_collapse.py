"""E-NS2: do the fidelity kernel's white-box cells fall on the RBF's length-scale curve?

P10a showed the RBF-SVM's attacked kernel-mass ratio collapses onto u = gamma * eps^2 *
||sigma||^2 (leave-one-gamma-out median error 0.0244) and F1 - floor follows it only roughly
(0.0465). This maps each fidelity-kernel (ZZ, 8 qubits) bandwidth to an effective RBF gamma
read from Gram-matrix decay and predicts every P9a cell from the RBF's interpolant at
u_Q = gamma_eff * eps^2 * ||sigma||^2. Pre-registration: decision log, 2026-11-16T09:19Z,
section E-NS2. No attack is run: P8a (8 gammas) and P9a (10 bandwidths) both hold every
radius, so nothing is recomputed, and a missing cell fails loud.

gamma_eff is the no-intercept least-squares slope of -log k(x, x') on ||x - x'||^2 over the
train subset's off-diagonal pairs (i < j). No intercept, chosen before any quantum fit was
seen: both kernels have k(x, x) = 1, so -log k is 0 at distance 0 by construction, and an
RBF is exactly -log k = gamma * d^2; a free intercept would let a non-RBF floor move the
slope away from the quantity u is built from. R^2 is the centred form,
1 - SS_res / sum (y - mean y)^2, so a line through the origin earns no credit for the mean.
The Gram is P9a's: `compute_states` at the default complex64 storage and `fidelity_kernel`
with exact=True (complex128 accumulation, decision log 2026-08-14T18:05Z), on the same PCA-8
inputs and the same 2,000-row `kernel_subset`. Distances are in the same PCA coordinates the
RBF was fitted in, and ||sigma||^2 is P10a's sum of squared per-feature train stds (77.1).

Controls: (i, blocking) P10a's leave-one-gamma-out errors are reproduced within 0.005;
(ii, blocking) the fitter returns gamma within 1% on the RBF's own Gram at every P8a gamma
and every fitted gamma_eff that float64 can represent without underflow (skipped ones are
listed); negative, u_wrong maps every bandwidth to the tuned RBF gamma and must be at least
2x worse on F1 - floor, or the test is discarded.

NS2b (`--gamma-fit near-pairs`, pre-registration 2026-11-16T11:07Z, R4-04): the same slope
fitted over the near pairs only, those with k >= 0.5 (the registered cut), so the kernel's
floor at large distance cannot set it. R^2 and the near-pair count are reported per bandwidth;
a bandwidth with fewer than 1,000 near pairs is flagged, never re-cut. Everything else (the
collapse test, controls (i) and (ii), the u_wrong negative control and the 0.10 kill line)
is E-NS2's. The default `--gamma-fit all-pairs` is E-NS2 unchanged.

R4-04c (`--gamma-fit ard`, amendment 2026-11-16T11:29Z, the last NS2 instrument): an
anisotropic length scale, -log k = sum_j gamma_j (x_j - x'_j)^2 fitted by non-negative least
squares over the same near pairs (k >= 0.5), one gamma_j per PCA dimension. Then
u_Q = eps^2 sum_j gamma_j sigma_j^2 with sigma the per-feature train std (the sigma of
||sigma||^2), carried as the equivalent isotropic gamma_eff = sum_j gamma_j sigma_j^2 /
||sigma||^2 so the collapse test is E-NS2's unchanged. Extra blocking control (iii): on the
RBF's own Gram every gamma_j returns gamma within 1%. R^2 and the gamma_j vector are reported.

Usage:
  uv run python -m experiments.run_length_scale_collapse --workers 2 --threads 1
  uv run python -m experiments.run_length_scale_collapse --smoke --workers 2 --threads 1
  uv run python -m experiments.run_length_scale_collapse --gamma-fit near-pairs \
      --workers 2 --threads 1
  uv run python -m experiments.run_length_scale_collapse --gamma-fit ard --workers 2 --threads 1
"""

import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=1)
THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_var] = str(THREADS)

import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402
from scipy.optimize import nnls  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402

from qgridbench.models.classical.zoo import stratified_cap  # noqa: E402
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel  # noqa: E402
from qgridbench.protocol import kernel_subset, prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402

log = get_logger(__name__)

P8A_RUN = "20260814T111131Z_whitebox_qsvm_binary_q8"
P9A_RUN = "20260814T180008Z_whitebox_qsvm_binary_q8"
P8A_KEY, P9A_KEY = "p8a_lengthscale_control", "p9a_quantum_bandwidth_regime"
BEST_PARAMS = "results/adv_best_params_binary_q8.json"
VARIANT, QUBITS = "binary", 8

P10A_SIG2 = 77.1  # P10a's ||sigma||^2, quoted to one decimal
P10A_LOGO = {"kernel_mass_ratio": 0.0244, "f1_minus_floor": 0.0465}
P10A_TOL = 0.005
GAMMA_REL_TOL = 0.01
R2_FLAG = 0.9
KILL_LINE = 0.10
NEG_POWER = 2.0
MEM_FRACTION = 0.8
TARGETS = ("f1_minus_floor", "kernel_mass_ratio")
NEAR_K = 0.5  # NS2b's registered near-pair cut, k >= 0.5
NEAR_MIN_PAIRS = 1000  # fewer near pairs than this is flagged, never re-cut


def _peak_rss_gb() -> float:
    mi = psutil.Process().memory_info()
    if sys.platform == "win32":
        return mi.peak_wset / 2**30
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _regime_rows(gram_rows: int, gram_seed: int):
    """P9a's train subset in PCA-8 coordinates, optionally stratified-subsampled."""
    reg = prepare_regime(VARIANT, "pca", n_components=QUBITS, seed=0)
    cap = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")["qkernel"]["subset_cap"]
    sub = kernel_subset(reg, cap, seed=0)
    X, y = reg.X["train"][sub], reg.y["train"][sub]
    if gram_rows < len(sub):
        keep = stratified_cap(X, y, gram_rows, seed=gram_seed)
        X = X[keep]
    return reg, X, int(len(sub))


def fit_gamma_eff(k_pairs: np.ndarray, d2_pairs: np.ndarray) -> dict:
    """No-intercept LS slope of -log k on d^2, with the centred R^2."""
    if (k_pairs <= 0).any():
        raise ValueError(f"{int((k_pairs <= 0).sum())} pairs have k <= 0; -log k is undefined")
    y = -np.log(k_pairs)
    g = float((d2_pairs * y).sum() / (d2_pairs * d2_pairs).sum())
    ss_res = float(((y - g * d2_pairs) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"gamma_eff": g, "r2": 1.0 - ss_res / ss_tot, "n_pairs": int(len(y))}


def fit_gamma_loc(k_pairs: np.ndarray, d2_pairs: np.ndarray) -> dict:
    """NS2b: the same slope over the near pairs only (k >= NEAR_K), with their count."""
    near = k_pairs >= NEAR_K
    n_near = int(near.sum())
    if n_near < 2:
        raise ValueError(f"{n_near} pairs have k >= {NEAR_K}; the near-pair slope is undefined")
    return {
        **fit_gamma_eff(k_pairs[near], d2_pairs[near]),
        "n_near_pairs": n_near,
        "few_near_pairs": bool(n_near < NEAR_MIN_PAIRS),
    }


GAMMA_FITS = {"all-pairs": fit_gamma_eff, "near-pairs": fit_gamma_loc}


def fit_gamma_ard(X: np.ndarray, K: np.ndarray, sigma: np.ndarray) -> dict:
    """R4-04c: NNLS fit of -log k = sum_j gamma_j (x_j - x'_j)^2 over the near pairs (i < j).

    The near pairs are streamed row by row into the d x d normal equations, and scipy's nnls
    runs on their Cholesky factor: ||A g - y||^2 = ||R g - c||^2 + const for R^T R = A^T A
    and R^T c = A^T y, so the solution is A's without the (pairs x d) matrix in memory.
    """
    X = X.astype(np.float64)
    n, d = X.shape
    AtA, Aty, y_sum, n_near = np.zeros((d, d)), np.zeros(d), 0.0, 0
    for i in range(n - 1):
        k = K[i, i + 1 :]
        near = k >= NEAR_K
        if near.any():
            D = (X[i + 1 :][near] - X[i]) ** 2
            y = -np.log(k[near])
            AtA += D.T @ D
            Aty += D.T @ y
            y_sum += float(y.sum())
            n_near += int(near.sum())
    if n_near < d:
        raise ValueError(f"{n_near} pairs have k >= {NEAR_K}; {d} gamma_j are undetermined")
    R = np.linalg.cholesky(AtA).T  # raises LinAlgError when A is rank-deficient
    g, _ = nnls(R, np.linalg.solve(R.T, Aty))
    y_mean, ss_res, ss_tot = y_sum / n_near, 0.0, 0.0
    for i in range(n - 1):
        k = K[i, i + 1 :]
        near = k >= NEAR_K
        if near.any():
            y = -np.log(k[near])
            ss_res += float(((y - ((X[i + 1 :][near] - X[i]) ** 2) @ g) ** 2).sum())
            ss_tot += float(((y - y_mean) ** 2).sum())
    s2 = np.asarray(sigma, dtype=np.float64) ** 2
    return {
        "gamma_eff": float((g * s2).sum() / s2.sum()),
        "gamma_j": g.tolist(),
        "r2": 1.0 - ss_res / ss_tot,
        "n_pairs": n_near,
        "n_near_pairs": n_near,
        "few_near_pairs": bool(n_near < NEAR_MIN_PAIRS),
    }


def fit_bandwidth(spec: dict) -> dict:
    """One unit: P9a's fidelity Gram at one bandwidth, then gamma_eff."""
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise AssertionError(f"worker thread cap {os.environ.get('OMP_NUM_THREADS')} != {THREADS}")
    t0 = time.perf_counter()
    reg, X, _ = _regime_rows(spec["gram_rows"], spec["gram_seed"])
    states = compute_states(reg.angle_scaler.transform(X), spec["encoding"], spec["bandwidth"])
    K = fidelity_kernel(states, states, exact=True)
    iu = np.triu_indices(len(X), 1)
    k = K[iu]
    if spec["gamma_fit"] == "ard":
        fit = fit_gamma_ard(X, K, reg.feature_scale)
    else:
        fit = GAMMA_FITS[spec["gamma_fit"]](k, pdist(X.astype(np.float64), "sqeuclidean"))
    return {
        "bandwidth": spec["bandwidth"],
        **fit,
        "r2_flag": bool(fit["r2"] < R2_FLAG),
        "n_rows": int(len(X)),
        "offdiag_k_mean": float(k.mean()),
        "offdiag_k_min": float(k.min()),
        "wall_s": round(time.perf_counter() - t0, 2),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "pid": os.getpid(),
    }


def memory_guard(workers: int, worker_rss: float, main_rss: float, budget: float) -> dict:
    avail = psutil.virtual_memory().available / 2**30
    need = workers * worker_rss
    out = {
        "workers": workers,
        "worker_peak_rss_gb": worker_rss,
        "main_rss_gb": main_rss,
        "need_gb": need + main_rss,
        "budget_gb": budget,
        "available_gb": avail,
    }
    log.info("memory: %s", out)
    if need > MEM_FRACTION * avail:
        raise MemoryError(f"{need:.2f} GB > {MEM_FRACTION:.0%} of {avail:.2f} GB available")
    if need + main_rss > budget:
        raise MemoryError(f"{need + main_rss:.2f} GB > the {budget:.2f} GB budget; lower --workers")
    return out


def load_cells(radii: list[float]) -> tuple[dict, list[dict], list[dict]]:
    """RBF (P8a) and fidelity (P9a) cells at every nonzero radius; a missing cell fails loud."""
    inv, rbf, fid = {}, [], []
    for run, key, blk, kind in (
        (P8A_RUN, P8A_KEY, "per_width", "rbf"),
        (P9A_RUN, P9A_KEY, "per_bandwidth", "fidelity"),
    ):
        path = REPO_ROOT / "results" / "runs" / run
        art = json.loads((path / "results.json").read_text())[key]
        params = load_yaml(path / "params.yaml")
        floor = art["constant_predictor_floor"]
        inv[kind] = {
            "artifact": run,
            "constant_predictor_floor": floor,
            "n_eval": art["n_eval"],
            "attack_seeds": params["seeds"],
            "encoding": params["config"]["encoding"],
            "units": {},
        }
        for unit, v in art[blk].items():
            eps_keys = sorted(v["curve"], key=float)
            missing = [e for e in radii if str(e) not in v["curve"]]
            if missing:
                raise KeyError(f"{run} {unit}: radii {missing} missing; recompute not implemented")
            inv[kind]["units"][unit] = {
                "eps": [float(e) for e in eps_keys],
                "metrics": sorted(v["curve"][eps_keys[-1]]),
            }
            for e in radii:
                c = v["curve"][str(e)]
                if abs(c["f1_minus_floor"] - (c["macro_f1"] - floor)) > 1e-12:
                    raise AssertionError(
                        f"{run} {unit} eps {e}: f1_minus_floor != macro_f1 - floor"
                    )
                row = {
                    "eps": e,
                    "macro_f1": c["macro_f1"],
                    "macro_f1_std": c["macro_f1_std"],
                    "f1_minus_floor": c["f1_minus_floor"],
                    "kernel_mass_ratio": c["kernel_mass_ratio"],
                }
                if kind == "rbf":
                    rbf.append({"gamma": v["gamma"], "gamma_multiplier": float(unit), **row})
                else:
                    fid.append({"bandwidth": float(unit), **row})
    return inv, rbf, fid


def interpolant(lu: np.ndarray, t: np.ndarray):
    """P10a's interpolant: sort on log10 u, average duplicate x, np.interp clamped flat."""
    order = np.argsort(lu)
    xs, ys = lu[order], t[order]
    xu, idx = np.unique(xs, return_inverse=True)
    yu = np.array([ys[idx == k].mean() for k in range(len(xu))])
    return xu, yu


def logo(rbf: list[dict], sig2: float, target: str) -> dict:
    """P10a's leave-one-gamma-out: pooled median |error| plus per-gamma medians."""
    gam = np.array([r["gamma"] for r in rbf])
    eps = np.array([r["eps"] for r in rbf])
    t = np.array([r[target] for r in rbf])
    lu = np.log10(gam * eps**2 * sig2)
    errs, per = [], {}
    for g in sorted(set(gam)):
        tr, te = gam != g, gam == g
        xu, yu = interpolant(lu[tr], t[tr])
        e = np.abs(np.interp(lu[te], xu, yu) - t[te])
        per[str(g)] = float(np.median(e))
        errs.extend(e.tolist())
    return {"median_abs_error": float(np.median(errs)), "per_gamma": per, "n": len(errs)}


def predict(rbf: list[dict], sig2: float, fid: list[dict], gmap: dict) -> dict:
    """Predict each fidelity cell from the RBF interpolant at u = gmap[bw] * eps^2 * sig2."""
    lu_r = np.log10(np.array([r["gamma"] * r["eps"] ** 2 * sig2 for r in rbf]))
    lo, hi = float(lu_r.min()), float(lu_r.max())
    cells = []
    for c in fid:
        lu = float(np.log10(gmap[c["bandwidth"]] * c["eps"] ** 2 * sig2))
        rec = {"bandwidth": c["bandwidth"], "eps": c["eps"], "log10_u": lu}
        rec["outside_hull"] = bool(lu < lo or lu > hi)
        for tg in TARGETS:
            xu, yu = interpolant(lu_r, np.array([r[tg] for r in rbf]))
            p = float(np.interp(lu, xu, yu))
            rec[f"{tg}_measured"] = c[tg]
            rec[f"{tg}_predicted"] = p
            rec[f"{tg}_abs_error"] = abs(p - c[tg])
        cells.append(rec)
    out = {"rbf_log10_u_hull": [lo, hi], "cells": cells}
    out["frac_outside_hull"] = float(np.mean([c["outside_hull"] for c in cells]))
    for tg in TARGETS:
        e = np.array([c[f"{tg}_abs_error"] for c in cells])
        bws = np.array([c["bandwidth"] for c in cells])
        out[tg] = {
            "median_abs_error": float(np.median(e)),
            "per_bandwidth": {str(b): float(np.median(e[bws == b])) for b in sorted(set(bws))},
        }
    return out


def rbf_self_control(
    X: np.ndarray, gammas: dict, gamma_fit: str = "all-pairs", sigma: np.ndarray | None = None
) -> dict:
    """Control (ii): fit gamma on exp(-gamma d^2), the P8a kernel row formula.

    The all-pairs fit skips a gamma whose far pairs underflow to 0; the near-pair fits never
    read those pairs (k >= 0.5). Under `ard`, control (ii) reads the equivalent isotropic
    gamma and control (iii) every gamma_j; a gamma with fewer near pairs than dimensions
    cannot be fitted and is skipped and listed, as underflowing ones are above.
    """
    d2 = pdist(X.astype(np.float64), "sqeuclidean")
    rows = {}
    for name, g in gammas.items():
        k = np.exp(-g * d2)
        if gamma_fit == "all-pairs" and (k <= 0).any():
            rows[name] = {"gamma": g, "skipped": f"{int((k <= 0).sum())} pairs underflow float64"}
            continue
        if gamma_fit == "ard" and (k >= NEAR_K).sum() < X.shape[1]:
            n_near = int((k >= NEAR_K).sum())
            rows[name] = {"gamma": g, "skipped": f"{n_near} near pairs < {X.shape[1]} gamma_j"}
            continue
        if gamma_fit == "ard":
            Kg = squareform(k)
            np.fill_diagonal(Kg, 1.0)
            fit = fit_gamma_ard(X, Kg, sigma)
        else:
            fit = GAMMA_FITS[gamma_fit](k, d2)
        rel = abs(fit["gamma_eff"] - g) / g
        rows[name] = {"gamma": g, **fit, "rel_error": rel, "passed": bool(rel <= GAMMA_REL_TOL)}
        if gamma_fit == "ard":
            rel_j = float(np.max(np.abs(np.array(fit["gamma_j"]) - g)) / g)
            rows[name]["rel_error_per_dim_max"] = rel_j
            rows[name]["per_dim_passed"] = bool(rel_j <= GAMMA_REL_TOL)
    tested = [r for r in rows.values() if "passed" in r]
    out = {
        "rows": rows,
        "n_tested": len(tested),
        "passed": bool(tested) and all(r["passed"] for r in tested),
    }
    if gamma_fit == "ard":
        out["per_dim_passed"] = bool(tested) and all(r["per_dim_passed"] for r in tested)
    return out


def figure(path, rbf, fid_pred, sig2, gamma_fit="all-pairs"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    gam = np.array([r["gamma"] for r in rbf])
    lu_r = np.log10(np.array([r["gamma"] * r["eps"] ** 2 * sig2 for r in rbf]))
    bws = sorted({c["bandwidth"] for c in fid_pred["cells"]})
    cmap_r, cmap_q = plt.get_cmap("viridis"), plt.get_cmap("plasma")
    lg = np.log10(gam)
    for ax, tg, lab in zip(axes, TARGETS, ("F1 - floor", "kernel-mass ratio"), strict=True):
        t = np.array([r[tg] for r in rbf])
        xu, yu = interpolant(lu_r, t)
        ax.plot(xu, yu, color="0.6", lw=1, zorder=1, label="RBF interpolant")
        sc = ax.scatter(lu_r, t, c=lg, cmap=cmap_r, s=22, zorder=2, label="RBF cells (P8a)")
        for i, b in enumerate(bws):
            cs = [c for c in fid_pred["cells"] if c["bandwidth"] == b]
            ax.scatter(
                [c["log10_u"] for c in cs],
                [c[f"{tg}_measured"] for c in cs],
                facecolors="none",
                edgecolors=cmap_q(i / max(len(bws) - 1, 1)),
                marker="D",
                s=34,
                zorder=3,
                label=f"fidelity bw {b:.3g}",
            )
        for x in fid_pred["rbf_log10_u_hull"]:
            ax.axvline(x, ls="--", color="0.8", lw=0.8)
        ax.set_xlabel("log10 u  (u = gamma * eps^2 * ||sigma||^2; gamma_eff for fidelity)")
        ax.set_ylabel(lab)
    fig.colorbar(sc, ax=axes[0], label="log10 RBF gamma")
    axes[1].legend(fontsize=6, loc="lower left", ncol=2)
    tag = {
        "all-pairs": "E-NS2",
        "near-pairs": "NS2b (near-pair gamma, k >= 0.5)",
        "ard": "R4-04c (per-dimension gamma, NNLS, k >= 0.5)",
    }[gamma_fit]
    fig.suptitle(f"{tag}: fidelity-kernel white-box cells against the RBF length-scale curve")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="2 bandwidths, 200-row Gram subsample")
    ap.add_argument("--gram-rows", type=int, default=None, help="default: whole 2,000-row subset")
    ap.add_argument("--gram-seed", type=int, default=0)
    ap.add_argument("--max-rss-gb", type=float, default=1.5, help="this job's RSS budget")
    ap.add_argument(
        "--gamma-fit",
        choices=sorted([*GAMMA_FITS, "ard"]),
        default="all-pairs",
        help="all-pairs: E-NS2's Gram-decay slope; near-pairs: NS2b's slope over k >= 0.5; "
        "ard: R4-04c's per-dimension NNLS over k >= 0.5",
    )
    args = ap.parse_args()
    gram_rows = args.gram_rows or (200 if args.smoke else 10**9)

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    radii = sorted(set(cfg_adv["evasion"]["epsilons"]) | {1.0, 2.0})  # P8a/P9a's grid
    inv, rbf, fid = load_cells(radii)
    reg, X, n_subset = _regime_rows(gram_rows, args.gram_seed)
    sig2 = float((reg.feature_scale**2).sum())  # P10a's convention
    if abs(sig2 - P10A_SIG2) > 0.05:
        raise AssertionError(f"||sigma||^2 {sig2:.3f} is not P10a's {P10A_SIG2}")
    bp = json.loads((REPO_ROOT / BEST_PARAMS).read_text())
    g_tuned = float(bp["rbf_svm"]["gamma"])
    if not any(abs(r["gamma"] - g_tuned) < 1e-12 for r in rbf):
        raise AssertionError(f"tuned RBF gamma {g_tuned} is not a P8a width")

    bandwidths = sorted({c["bandwidth"] for c in fid})
    if args.smoke:
        bandwidths = [bandwidths[0], bandwidths[-1]]
        fid = [c for c in fid if c["bandwidth"] in bandwidths]
    specs = [
        {
            "bandwidth": b,
            "encoding": inv["fidelity"]["encoding"],
            "gram_rows": gram_rows,
            "gram_seed": args.gram_seed,
            "gamma_fit": args.gamma_fit,
        }
        for b in bandwidths
    ]
    config = {
        **vars(args),
        "gram_rows_used": int(len(X)),
        "kernel_subset_size": n_subset,
        "inputs": {"p8a": P8A_RUN, "p9a": P9A_RUN, "best_params": BEST_PARAMS},
        "bandwidths": bandwidths,
        "radii": radii,
        "sigma_sq_norm": sig2,
        "gamma_eff_fit": "no-intercept LS slope of -log k on ||x-x'||^2, pairs i<j; centred R^2"
        + {
            "all-pairs": "",
            "near-pairs": f"; near pairs only, k >= {NEAR_K}",
            "ard": f"; per-dimension NNLS over near pairs, k >= {NEAR_K}; gamma_eff = "
            "sum_j gamma_j sigma_j^2 / ||sigma||^2",
        }[args.gamma_fit],
        "gram_path": "compute_states (complex64 storage) + fidelity_kernel(exact=True)",
    }
    name = (
        "length_scale_collapse_binary_q8"
        + {"all-pairs": "", "near-pairs": "_nearpairs", "ard": "_ard"}[args.gamma_fit]
        + ("_smoke" if args.smoke else "")
    )
    with RunDir(name, config=config, seeds=[args.gram_seed]) as run:
        out = {"config": config, "inventory": inv}
        main_rss = _peak_rss_gb()
        # the first unit runs alone to measure a worker's peak RSS before the pool widens
        with ProcessPoolExecutor(max_workers=1) as ex:
            fits = [ex.submit(fit_bandwidth, specs[0]).result()]
        out["memory_guard"] = memory_guard(
            args.workers, fits[0]["peak_rss_gb"], main_rss, args.max_rss_gb
        )
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(fit_bandwidth, specs[1:]):
                fits.append(r)
                run.write_json("results_partial.json", {"gamma_eff": fits})
        for r in fits:
            log.info(
                "bw %.4g: gamma_eff %.5f, R^2 %.3f (%.1fs)",
                *[r[k] for k in ("bandwidth", "gamma_eff", "r2", "wall_s")],
            )
        out["gamma_eff"] = {str(r["bandwidth"]): r for r in fits}
        gmap = {r["bandwidth"]: r["gamma_eff"] for r in fits}

        ctl1 = {tg: logo(rbf, sig2, tg) for tg in TARGETS}
        for tg in TARGETS:
            ctl1[tg]["p10a"] = P10A_LOGO[tg]
            ctl1[tg]["passed"] = bool(abs(ctl1[tg]["median_abs_error"] - P10A_LOGO[tg]) <= P10A_TOL)
        ctl1["passed"] = all(ctl1[tg]["passed"] for tg in TARGETS)
        ctl2 = rbf_self_control(
            X,
            {f"p8a_{r['gamma_multiplier']}": r["gamma"] for r in rbf if r["eps"] == radii[0]}
            | {f"gamma_eff_bw_{b}": g for b, g in gmap.items()},
            args.gamma_fit,
            reg.feature_scale,
        )
        test = predict(rbf, sig2, fid, gmap)
        wrong = predict(rbf, sig2, fid, dict.fromkeys(gmap, g_tuned))
        neg = {
            tg: {
                "median_abs_error_wrong": wrong[tg]["median_abs_error"],
                "median_abs_error_registered": test[tg]["median_abs_error"],
                "ratio": wrong[tg]["median_abs_error"] / test[tg]["median_abs_error"],
            }
            for tg in TARGETS
        }
        neg_pass = bool(neg["f1_minus_floor"]["ratio"] >= NEG_POWER)
        controls = {
            "positive_i_p10a_logo": ctl1,
            "positive_ii_rbf_self_fit": ctl2,
            "negative_u_wrong": {**neg, "tuned_rbf_gamma": g_tuned, "passed": neg_pass},
            "negative_u_wrong_frac_outside_hull": wrong["frac_outside_hull"],
        }
        blocked = not (ctl1["passed"] and ctl2["passed"])
        if args.gamma_fit == "ard":
            controls["positive_iii_ard_per_dim"] = {"passed": ctl2["per_dim_passed"]}
            blocked = blocked or not ctl2["per_dim_passed"]
        med = test["f1_minus_floor"]["median_abs_error"]
        if blocked:
            branch = "BLOCKED (positive control failed)"
        elif not neg_pass:
            branch = "DISCARDED (negative control: no power)"
        elif med <= KILL_LINE:
            branch = "HYPOTHESIS (one curve)"
        else:
            branch = "KILL"
        top = str(radii[-1])
        u_shape = {
            str(c["bandwidth"]): {
                "mass_measured": c["kernel_mass_ratio_measured"],
                "mass_predicted": c["kernel_mass_ratio_predicted"],
            }
            for c in test["cells"]
            if str(c["eps"]) == top
        }
        out.update(
            {
                "collapse_test": test,
                "negative_control_cells": wrong["cells"],
                "mass_at_top_radius": {"eps": radii[-1], "per_bandwidth": u_shape},
                "controls": controls,
                "branch": branch,
                "branch_statistic": {
                    "f1_minus_floor_median_abs_error": med,
                    "kill_line": KILL_LINE,
                },
                "peak_rss_gb": {
                    "main": round(_peak_rss_gb(), 3),
                    "workers": [r["peak_rss_gb"] for r in fits],
                },
            }
        )
        run.write_json("results.json", out)
        figure(run.path / "collapse_f1_mass_vs_log_u.png", rbf, test, sig2, args.gamma_fit)
        run.metrics.update(
            {
                "branch": branch,
                "f1_minus_floor_median_abs_error": med,
                "mass_median_abs_error": test["kernel_mass_ratio"]["median_abs_error"],
                "frac_outside_hull": test["frac_outside_hull"],
                "neg_control_ratio_f1": neg["f1_minus_floor"]["ratio"],
                "p10a_logo_mass": ctl1["kernel_mass_ratio"]["median_abs_error"],
                "p10a_logo_f1": ctl1["f1_minus_floor"]["median_abs_error"],
                "rbf_self_fit_passed": ctl2["passed"],
            }
        )
        log.info(
            "CONTROLS %s",
            json.dumps({k: v.get("passed") for k, v in controls.items() if isinstance(v, dict)}),
        )
        log.info("BRANCH %s (median |F1 - floor error| %.4f)", branch, med)
        print(f"-> {run.path}")
        if blocked:
            raise AssertionError(
                f"positive control failed: {json.dumps(controls, default=str)[:600]}"
            )


if __name__ == "__main__":
    main()
