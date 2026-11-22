"""R4-05: centred kernel alignment between each fidelity Gram and the tuned RBF Gram.

A spike with no branch (pre-registration 2026-11-16T11:07Z, R4-04/R4-05): for each of P9a's
10 bandwidths, the CKA between the fidelity-kernel Gram (ZZ, 8 qubits, P9a's evaluated path:
`compute_states` at complex64 storage and `fidelity_kernel(exact=True)`) and the tuned RBF
Gram (gamma 4.1909, P8a's kernel exp(-gamma ||x - x'||^2) in PCA-8 coordinates), on P9a's
2,000-row train `kernel_subset`. Also the CKA against the RBF at that bandwidth's fitted
gamma_loc (NS2b's near-pair slope, k >= 0.5). The Gram, the rows and gamma_loc are imported
from `run_length_scale_collapse`, never recomputed another way.

CKA(K, L) = HSIC_u(K, L) / sqrt(HSIC_u(K, K) HSIC_u(L, L)) on the unbiased HSIC estimator
(Song et al. 2012), which reads only the off-diagonal entries (amendment 2026-11-16T11:29Z).
The full-Gram form registered first could not pass its shuffled control: the tuned RBF Gram
is nearly diagonal and a row permutation keeps the diagonal (0.220 against 0.05).

Controls (blocking): CKA(RBF, RBF) = 1 within 1e-9; CKA(RBF, P RBF P^T) < 0.05, where P is
a seeded permutation of the rows (the RBF Gram of the shuffled rows, so row i of one Gram is
paired with another row of the other). The per-bandwidth CKA against that shuffled Gram is
reported beside the curve as its null (not registered, no threshold).

Usage:
  uv run python -m experiments.run_kernel_cka --workers 2 --threads 1
  uv run python -m experiments.run_kernel_cka --smoke --workers 2 --threads 1
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
from scipy.spatial.distance import pdist, squareform  # noqa: E402

from experiments.run_length_scale_collapse import (  # noqa: E402
    BEST_PARAMS,
    P9A_KEY,
    P9A_RUN,
    _peak_rss_gb,
    _regime_rows,
    fit_gamma_loc,
    memory_guard,
)
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402

log = get_logger(__name__)

SELF_TOL = 1e-9
SHUFFLE_MAX = 0.05
TUNED_GAMMA = 4.1909  # the brief's tuned RBF gamma, checked against BEST_PARAMS to 4 decimals


def hsic_unbiased(K: np.ndarray, L: np.ndarray) -> float:
    """Song et al. (2012), eq. 5: the unbiased HSIC estimator on zero-diagonal Grams.

    [tr(K~L~) + 1'K~1 1'L~1 / ((n-1)(n-2)) - 2/(n-2) 1'K~L~1] / (n(n-3)).
    """
    n = len(K)
    Kt, Lt = K.copy(), L.copy()
    np.fill_diagonal(Kt, 0.0)
    np.fill_diagonal(Lt, 0.0)
    rk, rl = Kt.sum(1), Lt.sum(0)
    term = (Kt * Lt.T).sum() + rk.sum() * rl.sum() / ((n - 1) * (n - 2)) - 2.0 * rk @ rl / (n - 2)
    return float(term / (n * (n - 3)))


def cka(K: np.ndarray, L: np.ndarray) -> float:
    """CKA on the unbiased HSIC estimator (diagonal terms removed)."""
    return hsic_unbiased(K, L) / float(np.sqrt(hsic_unbiased(K, K) * hsic_unbiased(L, L)))


def rbf_gram(d2_pairs: np.ndarray, gamma: float) -> np.ndarray:
    """exp(-gamma d^2) as a square Gram with unit diagonal (P8a's kernel)."""
    L = squareform(np.exp(-gamma * d2_pairs))
    np.fill_diagonal(L, 1.0)
    return L


def cka_bandwidth(spec: dict) -> dict:
    """One unit: P9a's fidelity Gram at one bandwidth, gamma_loc, and three CKAs."""
    if os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise AssertionError(f"worker thread cap {os.environ.get('OMP_NUM_THREADS')} != {THREADS}")
    t0 = time.perf_counter()
    reg, X, _ = _regime_rows(spec["gram_rows"], spec["gram_seed"])
    states = compute_states(reg.angle_scaler.transform(X), spec["encoding"], spec["bandwidth"])
    K = fidelity_kernel(states, states, exact=True)
    d2 = pdist(X.astype(np.float64), "sqeuclidean")
    loc = fit_gamma_loc(K[np.triu_indices(len(X), 1)], d2)
    L_tuned = rbf_gram(d2, spec["gamma_tuned"])
    perm = np.asarray(spec["perm"])
    return {
        "bandwidth": spec["bandwidth"],
        "cka_tuned_rbf": cka(K, L_tuned),
        "cka_rbf_at_gamma_loc": cka(K, rbf_gram(d2, loc["gamma_eff"])),
        "cka_shuffled_tuned_rbf_null": cka(K, L_tuned[perm][:, perm]),
        "gamma_loc": loc["gamma_eff"],
        "gamma_loc_r2": loc["r2"],
        "n_near_pairs": loc["n_near_pairs"],
        "few_near_pairs": loc["few_near_pairs"],
        "n_rows": int(len(X)),
        "wall_s": round(time.perf_counter() - t0, 2),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "pid": os.getpid(),
    }


def figure(path, rows: list[dict], tuned_bw: float):
    bw = [r["bandwidth"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(bw, [r["cka_tuned_rbf"] for r in rows], "o-", label="tuned RBF (gamma 4.19)")
    ax.plot(bw, [r["cka_rbf_at_gamma_loc"] for r in rows], "s-", label="RBF at gamma_loc")
    ax.plot(
        bw,
        [r["cka_shuffled_tuned_rbf_null"] for r in rows],
        "x--",
        color="0.5",
        label="shuffled tuned RBF (null)",
    )
    ax.axvline(tuned_bw, ls=":", color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("fidelity-kernel bandwidth (dotted: tuned)")
    ax.set_ylabel("CKA with the fidelity Gram")
    ax.set_ylim(min(0.0, *ax.get_ylim()), 1.02)
    ax.legend(fontsize=8)
    ax.set_title("R4-05: unbiased-HSIC CKA of the fidelity Gram against RBF Grams", fontsize=9)
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
    ap.add_argument("--perm-seed", type=int, default=0, help="the shuffled-row control's seed")
    ap.add_argument("--max-rss-gb", type=float, default=1.5, help="this job's RSS budget")
    args = ap.parse_args()
    gram_rows = args.gram_rows or (200 if args.smoke else 10**9)

    bp = json.loads((REPO_ROOT / BEST_PARAMS).read_text())
    g_tuned = float(bp["rbf_svm"]["gamma"])
    if round(g_tuned, 4) != TUNED_GAMMA:
        raise AssertionError(f"tuned RBF gamma {g_tuned} is not {TUNED_GAMMA}")
    p9a = REPO_ROOT / "results" / "runs" / P9A_RUN
    bandwidths = sorted(
        float(b) for b in json.loads((p9a / "results.json").read_text())[P9A_KEY]["per_bandwidth"]
    )
    if len(bandwidths) != 10:
        raise AssertionError(f"P9a has {len(bandwidths)} bandwidths, not 10")
    encoding = load_yaml(p9a / "params.yaml")["config"]["encoding"]
    tuned_bw = float(bp["qk_bw"])
    if args.smoke:
        bandwidths = [bandwidths[0], bandwidths[-1]]

    _, X, n_subset = _regime_rows(gram_rows, args.gram_seed)
    perm = np.random.default_rng(args.perm_seed).permutation(len(X))
    specs = [
        {
            "bandwidth": b,
            "encoding": encoding,
            "gram_rows": gram_rows,
            "gram_seed": args.gram_seed,
            "gamma_tuned": g_tuned,
            "perm": perm.tolist(),
        }
        for b in bandwidths
    ]
    config = {
        **vars(args),
        "gram_rows_used": int(len(X)),
        "kernel_subset_size": n_subset,
        "inputs": {"p9a": P9A_RUN, "best_params": BEST_PARAMS},
        "encoding": encoding,
        "bandwidths": bandwidths,
        "tuned_bandwidth": tuned_bw,
        "rbf_gamma_tuned": g_tuned,
        "cka": "HSIC_u(K, L) / sqrt(HSIC_u(K, K) HSIC_u(L, L)), unbiased HSIC "
        "(Song et al. 2012), diagonal terms removed",
        "gamma_loc_fit": "run_length_scale_collapse.fit_gamma_loc (near pairs, k >= 0.5)",
        "gram_path": "compute_states (complex64 storage) + fidelity_kernel(exact=True)",
        "shuffle": "P L P^T, P a permutation of the rows seeded by perm_seed",
    }
    name = "kernel_cka_binary_q8" + ("_smoke" if args.smoke else "")
    with RunDir(name, config=config, seeds=[args.gram_seed, args.perm_seed]) as run:
        out = {"config": config}
        d2 = pdist(X.astype(np.float64), "sqeuclidean")
        L = rbf_gram(d2, g_tuned)
        self_cka = cka(L, L)
        shuffled = cka(L, L[perm][:, perm])
        del L, d2
        controls = {
            "rbf_self": {
                "cka": self_cka,
                "tol": SELF_TOL,
                "passed": bool(abs(self_cka - 1) <= SELF_TOL),
            },
            "rbf_shuffled_rows": {
                "cka": shuffled,
                "max": SHUFFLE_MAX,
                "passed": bool(shuffled < SHUFFLE_MAX),
            },
        }
        log.info("CONTROLS %s", json.dumps(controls))
        main_rss = _peak_rss_gb()
        # the first unit runs alone to measure a worker's peak RSS before the pool widens
        with ProcessPoolExecutor(max_workers=1) as ex:
            rows = [ex.submit(cka_bandwidth, specs[0]).result()]
        out["memory_guard"] = memory_guard(
            args.workers, rows[0]["peak_rss_gb"], main_rss, args.max_rss_gb
        )
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(cka_bandwidth, specs[1:]):
                rows.append(r)
                run.write_json("results_partial.json", {"per_bandwidth": rows})
        for r in rows:
            log.info(
                "bw %.4g: CKA tuned %.4f, at gamma_loc %.4f, null %.4f (%.1fs)",
                *[
                    r[k]
                    for k in (
                        "bandwidth",
                        "cka_tuned_rbf",
                        "cka_rbf_at_gamma_loc",
                        "cka_shuffled_tuned_rbf_null",
                        "wall_s",
                    )
                ],
            )
        blocked = not all(c["passed"] for c in controls.values())
        out.update(
            {
                "controls": controls,
                "status": "BLOCKED (control failed)" if blocked else "controls passed",
                "per_bandwidth": {str(r["bandwidth"]): r for r in rows},
                "curve": {
                    "bandwidth": [r["bandwidth"] for r in rows],
                    "cka_tuned_rbf": [r["cka_tuned_rbf"] for r in rows],
                    "cka_rbf_at_gamma_loc": [r["cka_rbf_at_gamma_loc"] for r in rows],
                    "cka_shuffled_tuned_rbf_null": [r["cka_shuffled_tuned_rbf_null"] for r in rows],
                },
                "peak_rss_gb": {
                    "main": round(_peak_rss_gb(), 3),
                    "workers": [r["peak_rss_gb"] for r in rows],
                },
            }
        )
        run.write_json("results.json", out)
        figure(run.path / "cka_vs_bandwidth.png", rows, tuned_bw)
        run.metrics.update(
            {
                "status": out["status"],
                "cka_rbf_self": self_cka,
                "cka_rbf_shuffled": shuffled,
                "cka_tuned_rbf_at_tuned_bw": next(
                    (r["cka_tuned_rbf"] for r in rows if r["bandwidth"] == tuned_bw), None
                ),
            }
        )
        print(f"-> {run.path}")
        if blocked:
            raise AssertionError(f"control failed: {json.dumps(controls)}")


if __name__ == "__main__":
    main()
