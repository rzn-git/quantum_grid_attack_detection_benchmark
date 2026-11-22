"""E-NS8(b): where the MLP surrogate's transferable directions live under the grid physics.

Splits the MLP surrogate's input gradient into the part along the physical manifold of the
joint all-relay attack (E-M1c) and the part that violates the derived map, and reads each
part's cosine with every target family's own gradient (pre-registration, decision log
2026-11-16T09:19Z, E-NS8 part (b)).

Run (thread budget is W x T; --threads is read before numpy loads, in every worker):
    python -m experiments.run_transfer_direction_decomposition --smoke --threads 4
    python -m experiments.run_transfer_direction_decomposition --workers 1 --threads 4 \
        --worker-rss-gb 1.4

Rows and seeds are E-M1c's
(`results/runs/20261116T081940Z_consistent_relay_attack_binary_q8_frame_joint`): the same shared
builders (`build_shared`, the frame angle unit), the 500-row eval subset, seeds {0, 1, 2},
T = 4 (XGBoost); the full run requires every clean F1 to equal E-M1c's exactly.

Units: the joint attack's. A controlled channel (PM1..PM6, PA1..PA6, F, DF of each relay) is
measured in its attack unit (frame unit for PA1..PA6, raw train std scaler.scale_ otherwise);
a derived channel (PM7..PM12, PA7..PA12, PA:Z, PA:ZH) in its raw train std. A raw gradient
g_x becomes g_x * unit. Space: A1's channel set restricted to R1..R4 (asserted equal to the
four relays' 28 map channels each), D = 112.

Tangent space at each row: T = range(M), M = [I on the 56 controlled; J on the 56 derived],
J the block-diagonal Jacobian of each relay's derived map at the row's controlled values, from
the attack's own VJP path (`derived_vjp`, torch autograd, float64) in the unit metric. At a
deadband the map's `torch.where` branch gives the one-sided derivative the attack sees (an
off sequence channel has a zero Jacobian row); a magnitude on its floor (clean value 0) gets
the map's two-sided derivative, the floor being a box constraint the attack enforces, not a
manifold constraint; both are counted. Complete QR of M gives Q_on (D x 56) and Q_off
(D x 56); P_on = Q_on Q_on^T, P_off = I - P_on. A row is included when every relay's channels
are finite (the joint attack touches all four relays there); rows where any relay is left
untouched (a NaN relay channel, exactly the I1 == 0 rows) are excluded and counted.

Gradients at the map-consistent clean point X1 = JointSpace.rows(0), through the frozen
pipeline's exact delta map g_x = g_Z @ W / s (the imputer is the identity on finite values,
and every included row is finite on the 112 channels):
    g_s: MLP BCE (GradProvider "mlp").  Analytic targets: logistic regression (p - y) w,
    RBF-SVM (GradProvider "rbf_svm"), QSVM (QSVMWhiteBox, the gated white-box of
    make_whitebox), VQC (zoo.grad_provider("vqc")).
    Trees (rf, xgb, lgbm): Gaussian-smoothed antithetic finite differences of P(attack),
    times (1 - 2y) (the BCE ascent direction's sign), u ~ N(0, I_D) in the unit metric, the
    full pipeline evaluated at x +- sigma u. Every model sees x only through Z = Z0 + A u
    (A = W[:, ch] / s[ch] * unit, rank 8), so the component of u orthogonal to row(A) has
    zero mean in the estimator and is dropped: g = mean_i w_i P_A u_i, P_A the projector on
    row(A). Same smoothed function, same expectation, less variance; the plain estimator's
    MLP cosine is recorded beside it.

FD calibration (blocking, before any tree): on the MLP only, seed 0, every included row,
sigma in SIGMA_GRID x pairs in N_GRID (nested draws); the median cosine with the MLP's
analytic gradient is logged per cell. Rule: N = max(N_GRID) (the budget), sigma = the largest
sigma whose median cosine >= 0.9 at that N (most smoothing of the piecewise-constant trees
the MLP control admits; capped at the largest attack eps). The choice is written to
fd_choice.json; seed units read that file and re-check the MLP control at the choice before
estimating any tree.

Statistics per target: c_on = median over rows and seeds of cos(P_on g_s, P_on g_t), c_off the
same with P_off; the off-manifold share ||P_off g_s||^2 / ||g_s||^2 (median). Null: cosines of
independent Gaussian pairs in R^D projected into each row's subspace (95th percentile).
Branches: HYPOTHESIS when for >= 2 of 3 trees c_off - c_on >= 0.05 and c_off > null95_off
(plus the logistic-regression triangulation, c_on > null95_on, reported as its own pass or
fail); KILL when for >= 2 of 3 trees c_on >= c_off; INTERMEDIATE otherwise.

Blocking controls (each raises): projectors idempotent and P_on M = M to 1e-9; a step of h along
P_on g_s pushed back through the controlled channels leaves a derived-map residual of order 2
in h, along P_off g_s of order 1 (h = 1e-4 and 1e-5); every analytic gradient (MLP included)
matches central finite differences of its own loss on 3 rows x 5 channels within 1e-4 of the
row's largest component; the MLP FD-estimator control (median cosine >= 0.9); the full run's
clean F1 equal E-M1c's for every (seed, family).
"""

from __future__ import annotations

import os
import sys


def _early_threads(argv: list[str]) -> str | None:
    """--threads, read BEFORE numpy/torch load so the BLAS/OpenMP pools obey it."""
    for i, a in enumerate(argv):
        if a == "--threads":
            return argv[i + 1]
        if a.startswith("--threads="):
            return a.split("=", 1)[1]
    return None


_THREADS = _early_threads(sys.argv)
if _THREADS is None and not {"-h", "--help"} & set(sys.argv):
    raise SystemExit("--threads is required (it is applied before numpy is imported)")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
for _var in THREAD_VARS:
    os.environ[_var] = _THREADS or "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

from experiments.run_adversarial import FittedZoo, _f1  # noqa: E402
from experiments.run_channel_restricted_attack import (  # noqa: E402
    ENCODING,
    FAMILIES,
    QUBITS,
    RELAYS,
    SEEDS,
    VARIANT,
    make_whitebox,
)
from experiments.run_consistent_relay_attack import (  # noqa: E402
    TREES,
    JointSpace,
    Shared,
    _near_breakpoint,
    _peak_rss_gb,
    build_shared,
    derived_np,
    derived_vjp,
    memory_guard,
    wrap_deg,
    wrap_rad,
)
from qgridbench.attacks.gradients import mlp_forward  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger  # noqa: E402
from qgridbench.utils.seeding import spawn_rng  # noqa: E402

log = get_logger(__name__)

EM1C_RUN = (
    REPO_ROOT
    / "results"
    / "runs"
    / "20261116T081940Z_consistent_relay_attack_binary_q8_frame_joint"
)
EM1C_THREADS = 4  # XGBoost predictions depend on the thread count; E-M1c ran T = 4
SOURCE = "mlp"
ANALYTIC = ("logreg", "rbf_svm", "qsvm", "vqc")
TARGETS = ANALYTIC + TREES
SIGMA_GRID = (0.05, 0.1, 0.25, 0.5)  # unit metric; capped at the largest attack eps
N_GRID = (256, 1024, 4096)  # antithetic pairs; max(N_GRID) is the budgeted count
FD_COS_MIN = 0.9  # pre-registered MLP control on the estimator
MARGIN = 0.05  # pre-registered c_off - c_on
NULL_Q = 95.0
NULL_PAIRS_PER_ROW = 64
PROJ_TOL = 1e-9
STEP_H = (1e-4, 1e-5)
STEP_ROWS = 5
STEP_ORDER_ON_MIN = 1.8  # residual order along P_on g_s (O(h^2))
STEP_ORDER_OFF = (0.9, 1.1)  # residual order along P_off g_s (O(h))
GRAD_H, GRAD_TOL, GRAD_ROWS, GRAD_CH = 1e-5, 1e-4, 3, 5
CLIP_MARGIN = 1e-3  # rad: FD rows keep every encoding angle clear of the [-pi, pi] clip
EVALS_PER_CALL = 16384  # model rows per predict call in the smoothed estimator
MEM_FRACTION = 0.8


# --------------------------------------------------------------------------- pure helpers


def projectors(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Q_on (n, D, k) spanning range(M) and Q_off (n, D, D - k) its complement, by complete QR;
    with the blocking projector control (P_on idempotent, P_on M = M)."""
    k = M.shape[2]
    Q, R = np.linalg.qr(M, mode="complete")
    rdiag = np.abs(np.diagonal(R[:, :k, :k], axis1=1, axis2=2))
    q_on, q_off = Q[:, :, :k], Q[:, :, k:]
    P = q_on @ np.swapaxes(q_on, 1, 2)
    idem = float(np.abs(P @ P - P).max())
    keep = float(np.abs(P @ M - M).max() / max(1.0, float(np.abs(M).max())))
    orth = float(np.abs(np.swapaxes(q_on, 1, 2) @ q_off).max())
    out = {
        "dim_space": int(M.shape[1]),
        "dim_on": int(k),
        "dim_off": int(M.shape[1] - k),
        "min_abs_R_diag": float(rdiag.min()),
        "idempotence_max_abs": idem,
        "P_on_M_minus_M_max_rel": keep,
        "Q_on_T_Q_off_max_abs": orth,
        "tol": PROJ_TOL,
    }
    if rdiag.min() < 1e-8:
        raise AssertionError(f"projector control: M is rank-deficient ({out})")
    if not (idem <= PROJ_TOL and keep <= PROJ_TOL and orth <= PROJ_TOL):
        raise AssertionError(f"projector control failed: {out}")
    return q_on, q_off, out


def subspace_cos(Q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """cos(P a, P b) per row with P = Q Q^T (n, D, k); NaN where a projection is zero."""
    pa, pb = np.einsum("ndk,nd->nk", Q, a), np.einsum("ndk,nd->nk", Q, b)
    na, nb = np.linalg.norm(pa, axis=1), np.linalg.norm(pb, axis=1)
    ok = (na > 0) & (nb > 0)
    out = np.full(len(a), np.nan)
    out[ok] = (pa[ok] * pb[ok]).sum(axis=1) / (na[ok] * nb[ok])
    return out


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (na > 0) & (nb > 0)
    out = np.full(len(a), np.nan)
    out[ok] = (a[ok] * b[ok]).sum(axis=1) / (na[ok] * nb[ok])
    return out


def row_space_projector(A: np.ndarray) -> np.ndarray:
    """P_A = A^T (A A^T)^-1 A, the projector on row(A) (A is r x D, full row rank)."""
    G = A @ A.T
    if np.linalg.matrix_rank(G) != A.shape[0]:
        raise AssertionError("the pipeline map A is rank-deficient on the attackable channels")
    return A.T @ np.linalg.solve(G, A)


def smoothed_grads(f, Zc: np.ndarray, U: np.ndarray, A: np.ndarray, PA: np.ndarray, sigma, ns):
    """Antithetic Gaussian-smoothed gradient of f in the unit metric at each row.

    Zc (c, r) the rows' pipeline outputs, U (c, N, D) the draws, A (r, D) the pipeline map in
    unit coordinates, ns the nested pair counts read off the same draws. Returns
    {n: (projected (c, D), plain (c, D))}: projected = mean_i w_i P_A u_i, plain = mean_i w_i u_i,
    w_i = (f(Z + sigma A u_i) - f(Z - sigma A u_i)) / (2 sigma)."""
    c, N, _ = U.shape
    dZ = sigma * (U @ A.T)  # (c, N, r)
    Zs = np.concatenate([Zc[:, None, :] + dZ, Zc[:, None, :] - dZ], axis=1).reshape(-1, A.shape[0])
    fv = np.concatenate(
        [f(Zs[i : i + EVALS_PER_CALL]) for i in range(0, len(Zs), EVALS_PER_CALL)]
    ).reshape(c, 2 * N)
    w = (fv[:, :N] - fv[:, N:]) / (2.0 * sigma)
    out = {}
    for n in ns:
        plain = np.einsum("cn,cnd->cd", w[:, :n], U[:, :n]) / n
        out[n] = (plain @ PA, plain)
    return out


def null_cosines(q_on, q_off, rng, k: int, PA: np.ndarray | None = None) -> dict:
    """Cosines of independent Gaussian pairs in R^D projected into each row's two subspaces;
    with PA, the pairs are first projected on the pipeline's row space (information only)."""
    n, D, _ = q_on.shape
    a, b = rng.standard_normal((n, k, D)), rng.standard_normal((n, k, D))
    if PA is not None:
        a, b = a @ PA, b @ PA
    out = {}
    for name, Q in (("on", q_on), ("off", q_off)):
        pa, pb = np.einsum("ndj,nkd->nkj", Q, a), np.einsum("ndj,nkd->nkj", Q, b)
        cs = (pa * pb).sum(-1) / (np.linalg.norm(pa, axis=-1) * np.linalg.norm(pb, axis=-1))
        out[name] = cs.ravel()
    return out


def branch(stats: dict, null95: dict) -> dict:
    """The pre-registered reading over the three trees plus the logistic-regression check."""
    per = {}
    for t in TREES:
        c_on, c_off = stats[t]["c_on"], stats[t]["c_off"]
        per[t] = {
            "c_on": c_on,
            "c_off": c_off,
            "c_off_minus_c_on": c_off - c_on,
            "c_off_above_null95_off": bool(c_off > null95["off"]),
            "hypothesis_condition": bool(c_off - c_on >= MARGIN and c_off > null95["off"]),
            "kill_condition": bool(c_on >= c_off),
        }
    n_hyp = sum(p["hypothesis_condition"] for p in per.values())
    n_kill = sum(p["kill_condition"] for p in per.values())
    verdict = "HYPOTHESIS" if n_hyp >= 2 else "KILL" if n_kill >= 2 else "INTERMEDIATE"
    lr = stats["logreg"]["c_on"]
    return {
        "margin": MARGIN,
        "null95": null95,
        "per_tree": per,
        "n_trees_hypothesis": n_hyp,
        "n_trees_kill": n_kill,
        "verdict": verdict,
        "logreg_triangulation": {
            "c_on": lr,
            "null95_on": null95["on"],
            "pass": bool(lr > null95["on"]),
        },
    }


# --------------------------------------------------------------------------- geometry


class Geometry:
    """The joint attack's channel space, units and tangent space at each included row."""

    def __init__(self, sh: Shared, space: JointSpace):
        ctx = sh.ctx
        self.chans = np.concatenate(space.cols)  # per relay: 14 controlled, then 14 derived
        in_relays = np.array([c.startswith(tuple(f"R{r}" for r in RELAYS)) for c in ctx.columns])
        a1 = np.flatnonzero(ctx.arm_masks["A1"] & in_relays)
        if not np.array_equal(np.sort(self.chans), a1):
            raise AssertionError("A1 restricted to R1..R4 is not the four relays' map channels")
        self.a1_outside_relays = [
            c
            for c, m, r in zip(ctx.columns, ctx.arm_masks["A1"], in_relays, strict=True)
            if m and not r
        ]
        self.unit = np.concatenate([np.concatenate([s.sc, sh.s[s.rm.der_idx]]) for s in space.subs])
        self.D = len(self.chans)
        self.k = int(sum(len(s.ctrl) for s in space.subs))
        self.A = (ctx.W[:, self.chans] / sh.s[self.chans]) * self.unit[None, :]
        self.PA = row_space_projector(self.A)
        every = np.all([s.att for s in space.subs], axis=0)
        self.rows = np.flatnonzero(every)
        self.excluded = {
            "n_eval": int(len(every)),
            "n_included": int(every.sum()),
            "n_excluded_any_relay_untouched": int((~every).sum()),
            "n_untouched_by_relay": {f"R{s.rm.relay}": int((~s.att).sum()) for s in space.subs},
        }
        self.space, self.sh = space, sh
        self.X1 = space.rows(np.zeros_like(space.x0c))  # map-consistent clean rows
        self.Z1 = space.Z(self.X1)
        n = len(self.rows)
        M = np.zeros((n, self.D, self.k))
        floor_rows, off_rows, near_rows = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        for r, s in enumerate(space.subs):
            xc = self.X1[self.rows][:, s.ctrl]
            if not np.array_equal(xc, sh.X0imp[self.rows][:, s.ctrl]):
                raise AssertionError(f"R{s.rm.relay}: rows(0) moved a controlled channel")
            nd, nc = len(s.rm.der_idx), len(s.ctrl)
            J = np.empty((n, nd, nc))
            for o in range(nd):
                cot = np.zeros((n, nd))
                cot[:, o] = 1.0
                J[:, o, :] = derived_vjp(xc, cot, s.rm, "cpu")[1]
            Ju = J * s.sc[None, None, :] / sh.s[s.rm.der_idx][None, :, None]
            o0, c0 = 28 * r, nc * r
            M[:, o0 : o0 + nc, c0 : c0 + nc] = np.eye(nc)
            M[:, o0 + nc : o0 + nc + nd, c0 : c0 + nc] = Ju
            floor_rows |= (xc[:, :6] <= 0).any(axis=1)
            off_rows |= ~s._deadband_state(xc).all(axis=1)
            near_rows |= np.array([_near_breakpoint(s, x) for x in xc])
        self.M = M
        self.q_on, self.q_off, self.proj_control = projectors(M)
        self.counts = {
            "n_rows_any_controlled_magnitude_on_floor": int(floor_rows.sum()),
            "n_rows_any_sequence_channel_in_deadband": int(off_rows.sum()),
            "n_rows_near_a_breakpoint": int(near_rows.sum()),
        }
        self.near = near_rows

    def Z_rows(self, X: np.ndarray, e: int) -> np.ndarray:
        """The frozen pipeline's exact delta map for raw rows X perturbed from eval row e."""
        sh = self.sh
        return sh.ctx.Z0[e] + ((X - sh.X0imp[e]) / sh.s) @ sh.ctx.W.T

    def to_unit(self, gZ: np.ndarray) -> np.ndarray:
        """A PCA-space gradient (n, 8) as a gradient over the D channels in the unit metric."""
        return gZ @ self.A


# --------------------------------------------------------------------------- gradients


def analytic_grads(zoo, wbs, Z: np.ndarray, y: np.ndarray) -> dict:
    """BCE input gradients in PCA space: the MLP source and the four analytic targets."""
    lr = zoo.models["logreg"]
    if list(lr.classes_) != [0, 1] or lr.coef_.shape != (1, Z.shape[1]):
        raise AssertionError(f"logistic regression is not binary on {Z.shape[1]} inputs")
    p = lr.predict_proba(Z)[:, 1]
    return {
        SOURCE: wbs["mlp"](Z, y),
        "logreg": (p - y)[:, None] * lr.coef_[0][None, :],
        "rbf_svm": wbs["rbf_svm"](Z, y),
        "qsvm": wbs["qsvm"](Z, y),
        "vqc": wbs["vqc"](Z, y),
    }


def loss_fn(zoo, wbs, fam: str):
    """The per-row loss each analytic gradient differentiates, on PCA-space rows."""

    def mlp(Z, y):
        p = mlp_forward(zoo.models["mlp"], Z)[1]
        return -np.log(p[np.arange(len(y)), y])

    def logreg(Z, y):
        return -np.log(zoo.models["logreg"].predict_proba(Z)[np.arange(len(y)), y])

    def rbf(Z, y):
        f = zoo.models["rbf_svm"].decision_function(Z)
        return np.logaddexp(0.0, f) - y * f

    def qsvm(Z, y):
        p = 1.0 / (1.0 + np.exp(-wbs["qsvm"].decision_function(Z)))
        return -(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))

    def vqc(Z, y):
        m = zoo.models["vqc"]
        p = m.predict_proba(zoo._to_angles(Z))[np.arange(len(y)), y]
        return -np.log(p + 1e-9) if m.n_out == 1 else -np.log(p)

    return {"mlp": mlp, "logreg": logreg, "rbf_svm": rbf, "qsvm": qsvm, "vqc": vqc}[fam]


def control_analytic_fd(geo: Geometry, zoo, wbs, g_unit: dict, y: np.ndarray) -> dict:
    """Every analytic gradient against central differences of its own loss in the unit metric,
    3 rows (largest gradient, every encoding angle clear of the clip) x 5 channels each."""
    Z = geo.Z1[geo.rows]
    mm = zoo._angle._mm
    ang = Z * mm.scale_ + mm.min_
    clear = np.flatnonzero((np.abs(ang) < np.pi - CLIP_MARGIN).all(axis=1))
    clear = set(clear.tolist())
    out, worst_all = {}, 0.0
    for fam, g in g_unit.items():
        loss = loss_fn(zoo, wbs, fam)
        order = [i for i in np.argsort(-np.abs(g).max(axis=1)) if i in clear][:GRAD_ROWS]
        if len(order) < GRAD_ROWS:
            raise AssertionError(f"FD control {fam}: only {len(order)} rows clear of the clip")
        worst, per = 0.0, []
        for i in order:
            scale = float(np.abs(g[i]).max())
            for j in np.argsort(-np.abs(g[i]))[:GRAD_CH]:
                X = np.repeat(geo.X1[geo.rows[i]][None], 2, axis=0)
                X[0, geo.chans[j]] += GRAD_H * geo.unit[j]
                X[1, geo.chans[j]] -= GRAD_H * geo.unit[j]
                Lu, Ld = loss(geo.Z_rows(X, geo.rows[i]), np.array([y[i], y[i]]))
                fd = float((Lu - Ld) / (2 * GRAD_H))
                rel = abs(fd - g[i, j]) / max(scale, 1e-300)
                worst = max(worst, rel)
                per.append(
                    {
                        "row": int(geo.rows[i]),
                        "channel": int(j),
                        "grad": float(g[i, j]),
                        "fd": fd,
                        "rel": rel,
                    }
                )
        out[fam] = {"max_rel_err_over_row_max": worst, "checks": per}
        worst_all = max(worst_all, worst)
    out["max_rel_err"], out["tol"], out["h"] = worst_all, GRAD_TOL, GRAD_H
    if worst_all > GRAD_TOL:
        raise AssertionError(f"analytic FD control: {worst_all:.2e} > {GRAD_TOL}")
    return out


def control_step(geo: Geometry, gs: np.ndarray) -> dict:
    """A step h along P_on g_s (resp. P_off g_s), unit norm, pushed back through the controlled
    channels: residual of the derived channels against the map, in the unit metric (angles
    wrapped). Order log10(r(h) / r(h / 10)): >= 1.8 on, about 1 off."""
    space, sh = geo.space, geo.sh
    dirs = {
        "on": np.einsum("ndk,nk->nd", geo.q_on, np.einsum("ndk,nd->nk", geo.q_on, gs)),
        "off": np.einsum("ndk,nk->nd", geo.q_off, np.einsum("ndk,nd->nk", geo.q_off, gs)),
    }

    def residual(i: int, d: np.ndarray, h: float):
        y = h * d / np.linalg.norm(d)
        res, crossed = [], False
        for r, s in enumerate(space.subs):
            nc = len(s.ctrl)
            x0c = geo.X1[geo.rows[i], s.ctrl]
            xc = x0c + y[28 * r : 28 * r + nc] * s.sc
            crossed |= bool((s._deadband_state(xc[None]) != s._deadband_state(x0c[None])).any())
            dm = derived_np(xc[None], s.rm)[0]
            ds = (
                geo.X1[geo.rows[i], s.rm.der_idx]
                + y[28 * r + nc : 28 * r + 28] * sh.s[s.rm.der_idx]
            )
            diff = ds - dm
            diff[6:12] = wrap_deg(diff[6:12])
            diff[13] = wrap_rad(diff[13])
            res.append(diff / sh.s[s.rm.der_idx])
        return float(np.linalg.norm(np.concatenate(res))), crossed

    rows, per = [], []
    for i in range(len(geo.rows)):
        if (
            geo.near[i]
            or (geo.X1[geo.rows[i]][np.concatenate([s.ctrl[:6] for s in space.subs])] <= 0).any()
        ):
            continue
        r = {k: [residual(i, dirs[k][i], h) for h in STEP_H] for k in dirs}
        if any(c for v in r.values() for _, c in v):
            continue  # a deadband crossing under the step: not a smooth point
        rows.append(i)
        rec = {"row": int(geo.rows[i])}
        for k, v in r.items():
            (r1, _), (r2, _) = v
            rec[f"residual_{k}"] = [r1, r2]
            rec[f"order_{k}"] = float(np.log10(r1 / r2)) if r2 > 0 else float("inf")
        per.append(rec)
        if len(rows) == STEP_ROWS:
            break
    if len(rows) < STEP_ROWS:
        raise AssertionError(f"step control: only {len(rows)} smooth rows")
    ok_on = all(p["order_on"] >= STEP_ORDER_ON_MIN for p in per)
    ok_off = all(STEP_ORDER_OFF[0] <= p["order_off"] <= STEP_ORDER_OFF[1] for p in per)
    out = {
        "h": list(STEP_H),
        "order_on_min_required": STEP_ORDER_ON_MIN,
        "order_off_range": list(STEP_ORDER_OFF),
        "rows": per,
        "pass": bool(ok_on and ok_off),
    }
    if not out["pass"]:
        raise AssertionError(f"step control failed: {json.dumps(per)}")
    return out


def fd_draws(seed: int, tag: str, eval_rows: np.ndarray, N: int, D: int) -> np.ndarray:
    """(c, N, D) Gaussian draws, one stream per (seed, tag, eval row): chunking-invariant."""
    return np.stack(
        [spawn_rng(seed, f"ns8_fd_{tag}_r{int(e)}").standard_normal((N, D)) for e in eval_rows]
    )


def smoothed_target(geo, zoo, fam, y, sigma, ns, seed, tag) -> dict:
    """Smoothed-FD gradient of P(attack) times (1 - 2y) at every included row, for each n."""
    N = max(ns)
    chunk = max(1, EVALS_PER_CALL // (2 * N))
    Z = geo.Z1[geo.rows]
    sign = (1.0 - 2.0 * y)[:, None]
    out = {n: (np.empty((len(Z), geo.D)), np.empty((len(Z), geo.D))) for n in ns}
    for a in range(0, len(Z), chunk):
        b = min(a + chunk, len(Z))
        U = fd_draws(seed, tag, geo.rows[a:b], N, geo.D)
        g = smoothed_grads(
            lambda z: zoo.predict_proba(fam, z)[:, 1], Z[a:b], U, geo.A, geo.PA, sigma, ns
        )
        for n in ns:
            out[n][0][a:b] = g[n][0] * sign[a:b]
            out[n][1][a:b] = g[n][1] * sign[a:b]
    return out


# --------------------------------------------------------------------------- workers

_SH: Shared | None = None


def _init_worker(sh: Shared):
    global _SH
    want = str(sh.threads)
    got = {v: os.environ.get(v) for v in THREAD_VARS}
    if any(x != want for x in got.values()) or torch.get_num_threads() != sh.threads:
        raise AssertionError(
            f"worker {os.getpid()}: thread caps {got}, torch {torch.get_num_threads()}"
        )
    _SH = sh


def _setup(seed: int):
    sh = _SH
    ctx = sh.ctx
    zoo = FittedZoo(ctx.reg, ENCODING, QUBITS, seed, ctx.subset_cap, ctx.best_params)
    wbs, gate = make_whitebox(zoo, ctx)
    space = JointSpace("consistent", RELAYS, sh, "cpu")
    geo = Geometry(sh, space)
    return zoo, wbs, gate, geo


def run_calibration(spec: dict) -> dict:
    """The FD estimator on the MLP only (seed 0): median cosine per (sigma, N) and the choice."""
    t0 = time.perf_counter()
    seed = spec["seed"]
    zoo, wbs, _, geo = _setup(seed)
    y = _SH.ctx.y[geo.rows]
    gs = geo.to_unit(wbs["mlp"](geo.Z1[geo.rows], y))
    grid = []
    for sigma in SIGMA_GRID:
        est = smoothed_target(geo, zoo, SOURCE, y, sigma, N_GRID, seed, "calib")
        for n in N_GRID:
            cp, cpl = row_cos(est[n][0], gs), row_cos(est[n][1], gs)
            grid.append(
                {
                    "sigma": sigma,
                    "n_pairs": n,
                    "median_cos_projected": float(np.nanmedian(cp)),
                    "median_cos_plain": float(np.nanmedian(cpl)),
                    "n_rows": int(np.isfinite(cp).sum()),
                }
            )
        log.info(
            "calibration sigma %.3f: %s",
            sigma,
            [round(g["median_cos_projected"], 4) for g in grid[-len(N_GRID) :]],
        )
    n_pick = max(N_GRID)
    ok = [g for g in grid if g["n_pairs"] == n_pick and g["median_cos_projected"] >= FD_COS_MIN]
    if not ok:
        raise AssertionError(
            f"FD calibration: no sigma reaches median cos {FD_COS_MIN} at N={n_pick}: {grid}"
        )
    pick = max(ok, key=lambda g: g["sigma"])
    return {
        "seed": seed,
        "n_rows": int(len(geo.rows)),
        "grid": grid,
        "rule": "N = max(N_GRID); sigma = the largest sigma with MLP median cosine >= 0.9 at N",
        "sigma": pick["sigma"],
        "n_pairs": pick["n_pairs"],
        "mlp_median_cos": pick["median_cos_projected"],
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
    }


def run_seed(spec: dict) -> dict:
    """One seed: geometry, gradients, controls, then (choice read from disk) the trees."""
    t_unit = time.perf_counter()
    seed, sh = spec["seed"], _SH
    ctx = sh.ctx
    zoo, wbs, gate, geo = _setup(seed)
    t_fit = time.perf_counter() - t_unit
    clean = {f: _f1(ctx.y, zoo.clean_proba(f, ctx.Z0, ctx.y)) for f in FAMILIES}
    y = ctx.y[geo.rows]
    Z = geo.Z1[geo.rows]
    gZ = analytic_grads(zoo, wbs, Z, y)
    g = {k: geo.to_unit(v) for k, v in gZ.items()}
    ctl_fd = control_analytic_fd(geo, zoo, wbs, g, y)
    ctl_step = control_step(geo, g[SOURCE])
    log.info(
        "seed %d: geometry, analytic FD (%.1e) and step controls passed",
        seed,
        ctl_fd["max_rel_err"],
    )

    choice = json.loads(Path(spec["fd_choice_path"]).read_text())  # written before any tree
    sigma, npairs = choice["sigma"], choice["n_pairs"]
    timing = {}
    t0 = time.perf_counter()
    est = smoothed_target(geo, zoo, SOURCE, y, sigma, [npairs], seed, f"s{seed}")[npairs]
    timing["fd_mlp"] = round(time.perf_counter() - t0, 1)
    mlp_cos = row_cos(est[0], g[SOURCE])
    ctl_mlp = {
        "sigma": sigma,
        "n_pairs": npairs,
        "median_cos_projected": float(np.nanmedian(mlp_cos)),
        "median_cos_plain": float(np.nanmedian(row_cos(est[1], g[SOURCE]))),
        "min_required": FD_COS_MIN,
    }
    if not ctl_mlp["median_cos_projected"] >= FD_COS_MIN:
        raise AssertionError(f"seed {seed}: MLP FD control failed at the frozen choice: {ctl_mlp}")
    for fam in TREES:
        t0 = time.perf_counter()
        g[fam] = smoothed_target(geo, zoo, fam, y, sigma, [npairs], seed, f"s{seed}")[npairs][0]
        timing[f"fd_{fam}"] = round(time.perf_counter() - t0, 1)
        log.info("seed %d: %s smoothed gradient in %.0f s", seed, fam, timing[f"fd_{fam}"])

    gs = g[SOURCE]
    norm_s = np.linalg.norm(gs, axis=1)
    off_s = np.linalg.norm(np.einsum("ndk,nd->nk", geo.q_off, gs), axis=1) ** 2
    per_target = {}
    for t in TARGETS:
        gt = g[t]
        nt = np.linalg.norm(gt, axis=1)
        off_t = np.linalg.norm(np.einsum("ndk,nd->nk", geo.q_off, gt), axis=1) ** 2
        per_target[t] = {
            "cos_on": subspace_cos(geo.q_on, gs, gt).tolist(),
            "cos_off": subspace_cos(geo.q_off, gs, gt).tolist(),
            "cos_full": row_cos(gs, gt).tolist(),
            "off_share_target": np.where(
                nt > 0, off_t / np.where(nt > 0, nt, 1) ** 2, np.nan
            ).tolist(),
            "n_zero_gradient_rows": int((nt == 0).sum()),
        }
    nul = null_cosines(geo.q_on, geo.q_off, spawn_rng(seed, "ns8_null"), NULL_PAIRS_PER_ROW)
    nul_rs = null_cosines(
        geo.q_on, geo.q_off, spawn_rng(seed, "ns8_null_rowspace"), NULL_PAIRS_PER_ROW, geo.PA
    )
    return {
        "unit": f"s{seed}",
        "seed": seed,
        "eval_rows": geo.rows.tolist(),
        "clean_f1": clean,
        "geometry": {
            "dim_space": geo.D,
            "dim_on": geo.k,
            "dim_off": geo.D - geo.k,
            "dim_pipeline_row_space": int(np.linalg.matrix_rank(geo.A)),
            "a1_channels_outside_relays": geo.a1_outside_relays,
            **geo.excluded,
            **geo.counts,
            "n_attackable_rows_by_relay": {
                f"R{s.rm.relay}": int(s.att.sum()) for s in geo.space.subs
            },
        },
        "controls": {
            "projector": geo.proj_control,
            "step": ctl_step,
            "analytic_fd": ctl_fd,
            "mlp_fd_estimator": ctl_mlp,
            "qsvm_gate": gate,
        },
        "source_norm_zero_rows": int((norm_s == 0).sum()),
        "off_share_source": np.where(
            norm_s > 0, off_s / np.where(norm_s > 0, norm_s, 1) ** 2, np.nan
        ).tolist(),
        "per_target": per_target,
        "null": {k: v for k, v in nul.items()},
        "null_rowspace": {k: v for k, v in nul_rs.items()},
        "timing_s": {"zoo_fit_and_geometry": round(t_fit, 1), **timing},
        "wall_s": round(time.perf_counter() - t_unit, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "pid": os.getpid(),
    }


def pool(fn, specs, sh: Shared, workers: int, on_done) -> None:
    """Every finished unit is handed to on_done (which writes it); the first failure is raised
    after the pool drains, with its traceback."""
    failed = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(sh,),
    ) as ex:
        futs = {ex.submit(fn, sp): sp for sp in specs}
        for f in as_completed(futs):
            exc = f.exception()
            if exc is not None:
                log.error("unit %s failed", futs[f], exc_info=exc)
                failed.append((futs[f], exc))
                continue
            on_done(f.result())
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(specs)} units failed: {[s for s, _ in failed]}"
        ) from failed[0][1]


# --------------------------------------------------------------------------- parent


def _pct(x, q):
    return float(np.percentile(np.asarray(x), q))


def summarize(units: list[dict]) -> dict:
    units = sorted(units, key=lambda u: u["seed"])
    null = {k: np.concatenate([np.asarray(u["null"][k]) for u in units]) for k in ("on", "off")}
    null_rs = {
        k: np.concatenate([np.asarray(u["null_rowspace"][k]) for u in units]) for k in ("on", "off")
    }
    null95 = {k: _pct(v, NULL_Q) for k, v in null.items()}
    stats = {}
    for t in TARGETS:
        pooled = {
            k: np.concatenate([np.asarray(u["per_target"][t][k], dtype=float) for u in units])
            for k in ("cos_on", "cos_off", "cos_full", "off_share_target")
        }
        stats[t] = {
            "c_on": float(np.nanmedian(pooled["cos_on"])),
            "c_off": float(np.nanmedian(pooled["cos_off"])),
            "c_full": float(np.nanmedian(pooled["cos_full"])),
            "off_share_target_median": float(np.nanmedian(pooled["off_share_target"])),
            "n_rows_valid_on": int(np.isfinite(pooled["cos_on"]).sum()),
            "n_rows_valid_off": int(np.isfinite(pooled["cos_off"]).sum()),
            "n_rows_total": int(len(pooled["cos_on"])),
            "per_seed": {
                str(u["seed"]): {
                    "c_on": float(np.nanmedian(u["per_target"][t]["cos_on"])),
                    "c_off": float(np.nanmedian(u["per_target"][t]["cos_off"])),
                    "n_zero_gradient_rows": u["per_target"][t]["n_zero_gradient_rows"],
                }
                for u in units
            },
        }
    share = np.concatenate([np.asarray(u["off_share_source"], dtype=float) for u in units])
    return {
        "null": {
            "registered": {
                "pairs": "independent N(0, I_D) pairs projected into each row's subspace",
                "pairs_per_row": NULL_PAIRS_PER_ROW,
                "n": {k: int(len(v)) for k, v in null.items()},
                "p95": null95,
                "p50": {k: _pct(v, 50) for k, v in null.items()},
            },
            "information_pipeline_row_space": {
                "pairs": "N(0, I_D) pairs projected on row(A) (where every gradient lives), then "
                "into each subspace; not used by the branch",
                "p95": {k: _pct(v, NULL_Q) for k, v in null_rs.items()},
                "p50": {k: _pct(v, 50) for k, v in null_rs.items()},
            },
        },
        "off_share_source": {
            "median": float(np.nanmedian(share)),
            "per_seed_median": {
                str(u["seed"]): float(np.nanmedian(u["off_share_source"])) for u in units
            },
        },
        "per_target": stats,
        "branch": branch(stats, null95),
    }


def em1c_anchor(units: list[dict]) -> dict:
    """The full run is on E-M1c's rows, seeds and zoo: every clean F1 and attackable count equal."""
    res = json.loads((EM1C_RUN / "results.json").read_text())
    ref = {(x["seed"], x["family"]): x["clean_f1"] for x in res["records"] if x["mode"] == "joint"}
    meta = {m["unit"]: m["spaces"]["joint"]["by_relay"] for m in res["units_meta"]}
    diff = []
    for u in units:
        for f, v in u["clean_f1"].items():
            if ref[(u["seed"], f)] != v:
                diff.append(
                    {"seed": u["seed"], "family": f, "em1c": ref[(u["seed"], f)], "here": v}
                )
        by = meta[f"R1+R2+R3+R4_s{u['seed']}"]
        for r, n in u["geometry"]["n_attackable_rows_by_relay"].items():
            if by[r]["n_attackable_rows"] != n:
                diff.append(
                    {"seed": u["seed"], "relay": r, "em1c": by[r]["n_attackable_rows"], "here": n}
                )
    out = {"run": str(EM1C_RUN.relative_to(REPO_ROOT)), "rule": "exact equality", "differ": diff}
    out["passed"] = not diff
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--threads", type=int, required=True, help="BLAS/OpenMP/torch threads per worker"
    )
    ap.add_argument("--workers", type=int, default=1, help="parallel seed units")
    ap.add_argument("--smoke", action="store_true", help="40 eval rows, seed 0")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--worker-rss-gb", type=float, help="per-worker peak RSS measured by smoke")
    ap.add_argument("--resume", type=Path, help="results_partial.json of an interrupted run")
    args = ap.parse_args()
    if str(args.threads) != _THREADS:
        raise AssertionError(f"early thread parse {_THREADS} != argparse {args.threads}")
    T, W = args.threads, 1 if args.smoke else args.workers
    if T != EM1C_THREADS:
        raise SystemExit(f"--threads must be E-M1c's {EM1C_THREADS} (XGBoost is thread-dependent)")
    n_cpu = psutil.cpu_count(logical=True)
    avail = psutil.virtual_memory().available / 2**30
    print(f"workers W={W} x threads T={T} of {n_cpu}; {avail:.1f} GB available", flush=True)
    if W * T > n_cpu:
        raise ValueError(f"W x T = {W * T} exceeds the {n_cpu} logical CPUs")
    seeds = [0] if args.smoke else args.seeds
    sh, diag = build_shared(SimpleNamespace(smoke=args.smoke, angle_unit="frame"), T, RELAYS)
    em1c_cfg = json.loads((EM1C_RUN / "results.json").read_text())["config"]
    if not args.smoke and (len(sh.ctx.y) != em1c_cfg["n_eval"] or seeds != em1c_cfg["seeds"]):
        raise AssertionError("the full run must use E-M1c's eval rows and seeds")
    cfg = {
        "experiment": "E-NS8(b)",
        "preregistration": "experiment_decision_log.md 2026-11-16T09:19Z, E-NS8 part (b)",
        "em1c_run": str(EM1C_RUN.relative_to(REPO_ROOT)),
        "threads": T,
        "workers": W,
        "smoke": args.smoke,
        "seeds": seeds,
        "relays": RELAYS,
        "angle_unit": "frame",
        "n_eval": int(len(sh.ctx.y)),
        "source": SOURCE,
        "targets": list(TARGETS),
        "sigma_grid": list(SIGMA_GRID),
        "n_grid": list(N_GRID),
        "fd_cos_min": FD_COS_MIN,
        "margin": MARGIN,
        "null_pairs_per_row": NULL_PAIRS_PER_ROW,
        "step_h": list(STEP_H),
        "grad_fd": {"h": GRAD_H, "tol": GRAD_TOL, "rows": GRAD_ROWS, "channels": GRAD_CH},
        "resume": str(args.resume) if args.resume else None,
    }
    name = f"transfer_direction_decomposition_{VARIANT}_q{QUBITS}" + (
        "_smoke" if args.smoke else ""
    )
    with RunDir(name, config=cfg, seeds=seeds) as run:
        run.write_json("controls_map.json", diag)
        if args.smoke:
            rss = None
        else:
            if args.worker_rss_gb is None:
                raise SystemExit("--worker-rss-gb is required (smoke prints the measured value)")
            rss = memory_guard(W, args.worker_rss_gb, "seed units")
        state = {"calibration": None, "units": []}
        if args.resume:
            state = json.loads(args.resume.read_text())
            if state["calibration"] is None:
                raise AssertionError("resume: the partial file carries no FD calibration")
        # 1. the FD choice, on the MLP only, written before any tree is estimated
        if state["calibration"] is None:
            pool(
                run_calibration, [{"seed": 0}], sh, 1, on_done=lambda r: state.update(calibration=r)
            )
        cal = state["calibration"]
        choice_path = run.write_json(
            "fd_choice.json",
            {"sigma": cal["sigma"], "n_pairs": cal["n_pairs"], "rule": cal["rule"]},
        )
        run.write_json("results_partial.json", state)
        log.info(
            "FD choice frozen: sigma %.3f, N %d (MLP median cos %.4f)",
            cal["sigma"],
            cal["n_pairs"],
            cal["mlp_median_cos"],
        )
        print(
            f"calibration peak RSS {cal['peak_rss_gb']:.2f} GB, {cal['wall_s']:.0f} s", flush=True
        )
        # 2. the seed units
        have = {u["seed"] for u in state["units"]}
        specs = [{"seed": s, "fd_choice_path": str(choice_path)} for s in seeds if s not in have]

        def save(res):
            state["units"].append(res)
            run.write_json("results_partial.json", state)
            print(
                f"unit {res['unit']}: {res['wall_s']:.0f} s, peak RSS {res['peak_rss_gb']:.2f} GB",
                flush=True,
            )

        pool(run_seed, specs, sh, W, on_done=save)
        units = sorted(state["units"], key=lambda u: u["seed"])
        out = summarize(units)
        anchor = None if args.smoke else em1c_anchor(units)
        results = {
            "config": cfg,
            "memory_guard": rss,
            "controls_map": diag,
            "fd_calibration": cal,
            "em1c_anchor": anchor,
            "geometry": {u["unit"]: u["geometry"] for u in units},
            "controls": {u["unit"]: u["controls"] for u in units},
            **out,
            "units_meta": [
                {
                    k: u[k]
                    for k in (
                        "unit",
                        "timing_s",
                        "wall_s",
                        "peak_rss_gb",
                        "pid",
                        "source_norm_zero_rows",
                    )
                }
                for u in units
            ],
            "per_row": {
                u["unit"]: {
                    "eval_rows": u["eval_rows"],
                    "off_share_source": u["off_share_source"],
                    "per_target": u["per_target"],
                }
                for u in units
            },
        }
        run.write_json("results.json", results)
        b = out["branch"]
        run.metrics.update(
            {
                "verdict": b["verdict"],
                "logreg_triangulation_pass": b["logreg_triangulation"]["pass"],
                "null95": b["null95"],
                "off_share_source_median": out["off_share_source"]["median"],
                **{f"{t}_c_on": out["per_target"][t]["c_on"] for t in TARGETS},
                **{f"{t}_c_off": out["per_target"][t]["c_off"] for t in TARGETS},
                "fd_sigma": cal["sigma"],
                "fd_n_pairs": cal["n_pairs"],
            }
        )
        log.info("branch: %s", json.dumps(b))
        print(f"-> {run.path}")
        if anchor is not None and not anchor["passed"]:
            raise AssertionError(f"E-M1c anchor failed: {anchor['differ']}")


if __name__ == "__main__":
    main()
