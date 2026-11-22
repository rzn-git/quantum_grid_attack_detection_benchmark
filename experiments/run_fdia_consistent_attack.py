"""R4-22 H3: NS6 on the synthetic FDIA variants, a state-constrained attacker against an
unconstrained one, with the chi-square flag rate of each (the registered secondary statistic).

Pre-registration: decision log 2026-11-16T12:17Z, "PRE-RUN AMENDMENT (R4-22", H3 and ruling 5;
design note tmp/review_shihav_20261116/r4_22_fdia_design.md, section 5.2.

Models: the PCA-8 eight-family zoo at the variant's tuned parameters (`FittedZoo`, the Table III
fit), on the stratified 500-row test subset; the frozen pipeline impute -> StandardScaler ->
PCA(8) maps a raw measurement perturbation dz exactly: dZ = (dz / s) @ W.T (Stage 1's map).
  U  unconstrained (Stage 1's A1 analogue): L-inf PGD on the standardised measurements u
     (every channel, radius eps in train std), the field-standard feature-space attacker.
  C  state-constrained (the E-M1c analogue): PGD on a state offset d over all 2n - 1 states
     (non-reference angles, voltage magnitudes), |d_j| <= eps * sigma_j with sigma_j the train
     std of the estimated state j; the measurement moves along the AC map,
     z' = z + h(x_hat + d) - h(x_hat), x_hat the row's own WLS estimate (stored by the
     generator), so z' stays on the measurement manifold; gradient J(x_hat + d)^T (g_u / s).
Sources: the MLP surrogate (primary), white-box QSVM, RBF-SVM and VQC; each crafted set is
scored on every family. Random nulls: +-eps signs per channel (U) and +-eps sigma_j per state
mapped through h (C). Secondary arm (no branch): the MLP's U perturbation rescaled per row to
the C perturbation's standardised L2 norm. Tree-surrogate arm (R4-07(b)'s branches): the zoo's
XGBoost attacked with the Square schedule (V2-E2's `square_attack`, 600 iterations) in U and in C.

Statistics (seed means): E(attacker, tree, eps) = random F1 - transfer F1, attacker's own space.
H3 at eps 0.5 over rf, xgb, lgbm: SURVIVES if E(C) >= 0.05 for >= 2 trees; PHYSICS if E(C) < 0.05
for >= 2 trees that have E(U) >= 0.05; NOT_TESTABLE if E(U) < 0.05 for >= 2 trees; else MIXED.
Tree arm at eps 0.5: TREE_TRANSFER_SURVIVES if E_xgb(C, rf) or E_xgb(C, lgbm) >= 0.05, else
PHYSICS_REMOVES_IT_FOR_TREES_TOO. Ruling 5: the chi-square flag rate (alpha 0.01; a
non-converged estimate is a flag) of the U and C rows (MLP source and random), per eps, against
the unperturbed rows; reported, no branch. Matched-budget norms: median standardised L-inf and
L2 of du per cell.

Controls: mapping (Stage 1's control 2, raises); zero offset maps to zero change and leaves
every family's predictions bit-identical (raises); the C-space gradient against central finite
differences of the RBF-SVM loss, 3 rows x 5 states, <= 1e-4 relative (raises); stealth, each C
row's detector decision equals its unperturbed row's on >= 99% of rows at every eps and source
(raised after results.json is written); control 5, a white-box cell is admitted only if its
attacked F1 is below its random F1 (recorded); the tree arm's positive control, XGBoost's own
Square F1 under its random F1 at every eps in C (raised after results.json is written).

Units are seeds (each fits its own zoo; every stream depends on (seed, stream name) only), run
in --workers spawned processes of --threads threads, with a memory guard and --resume.

Run (thread budget W x T; --threads is read before numpy loads, in every worker):
    uv run python -m experiments.run_fdia_consistent_attack --variant fdia14_binary --smoke \
        --best-params tmp/r4_22_fdia/smoke_best_params.json --threads 4
    uv run python -m experiments.run_fdia_consistent_attack --variant fdia118_binary \
        --workers 2 --threads 4 --worker-rss-gb 1.5
"""

from __future__ import annotations

import os
import sys


def _early_threads(argv: list[str]) -> str | None:
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
import pickle  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from experiments.exports import Exports  # noqa: E402
from experiments.run_adversarial import FittedZoo, _f1  # noqa: E402
from experiments.run_channel_restricted_attack import (  # noqa: E402
    control_mapping,
    make_whitebox,
    stratified_random_floor,
    u_space_grad,
)
from experiments.run_consistent_relay_attack import _peak_rss_gb, memory_guard  # noqa: E402
from experiments.run_square_blackbox import square_attack  # noqa: E402
from qgridbench.attacks.evasion import pgd  # noqa: E402
from qgridbench.attacks.gradients import GradProvider  # noqa: E402
from qgridbench.data.fdia_synth import Case, fdia_config, load_aux  # noqa: E402
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.features.reduce import build_preprocessor  # noqa: E402
from qgridbench.models.classical.zoo import stratified_cap  # noqa: E402
from qgridbench.protocol import best_params_path, prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import spawn_rng  # noqa: E402

log = get_logger(__name__)

QUBITS = 8
ENCODING = "zz"
FAMILIES = ["logreg", "rbf_svm", "rf", "xgb", "lgbm", "mlp", "qsvm", "vqc"]
SOURCES = ["mlp", "qsvm", "rbf_svm", "vqc"]
TREES = ["rf", "xgb", "lgbm"]
EPS_GRID = [0.1, 0.25, 0.5]
SEEDS = [0, 1, 2]
E_MARGIN = 0.05  # E-NS8(a)'s margin
STEALTH_MIN = 0.99
GRAD_TOL = 1e-4  # Stage 1's control 4 tolerance
SQUARE_ITER = 600  # V2-E2
SMOKE = {"n_eval": 40, "n_steps": 3, "square_iter": 20, "eps": [0.5], "seeds": [0]}


@dataclass
class Ctx:
    variant: str
    case_name: str
    reg: object
    best_params: dict
    subset_cap: int
    W: np.ndarray  # (8, m) PCA components
    s: np.ndarray  # (m,) StandardScaler scale_
    Z0: np.ndarray  # (n_eval, 8)
    U0: np.ndarray  # (n_eval, m) standardised measurements
    z0: np.ndarray  # (n_eval, m) raw measurements, float64
    xhat: np.ndarray  # (n_eval, n_state) WLS estimates
    sig_state: np.ndarray  # (n_state,) train std of the estimated states
    y: np.ndarray
    floor: float
    n_steps: int
    step_frac: float
    square_iter: int
    eps_grid: list
    alpha: float
    threads: int
    test_pos: np.ndarray  # eval rows' positions in the variant's test split
    tuned_file: str


# ------------------------------------------------------------------------- the two spaces


def state_map(case: Case, xhat: np.ndarray, h0: np.ndarray, D: np.ndarray) -> np.ndarray:
    """dz = h(x_hat + D) - h(x_hat), row-wise."""
    return case.h_rows(xhat + D) - h0


def c_space_grad(grad_z, ctx: Ctx, case: Case, h0: np.ndarray):
    """A PCA-space provider wrapped into a state-offset one: g_D = J(x_hat + D)^T (g_u / s)."""
    gu_fn = u_space_grad(grad_z, ctx)

    def g(D, y):
        U = ctx.U0 + state_map(case, ctx.xhat, h0, D) / ctx.s
        gz = gu_fn(U, y) / ctx.s
        return np.stack([case.jac_x(ctx.xhat[i] + D[i]).T @ gz[i] for i in range(len(D))])

    return g


def bdd_flags(case: Case, z: np.ndarray, thr: float) -> np.ndarray:
    """True where the chi-square test flags the row (or the estimate does not converge)."""
    out = np.empty(len(z), dtype=bool)
    for i, zi in enumerate(z):
        J, ok, _ = case.wls(zi)
        out[i] = (not ok) or J > thr
    return out


def norms(dU: np.ndarray) -> dict:
    return {
        "du_linf_median": float(np.median(np.abs(dU).max(axis=1))),
        "du_l2_median": float(np.median(np.linalg.norm(dU, axis=1))),
    }


# ---------------------------------------------------------------------------- statistics


def h3_branch(e_u: dict, e_c: dict) -> str:
    """e_u / e_c: {tree: E at eps 0.5}."""
    if sum(e_c[t] >= E_MARGIN for t in TREES) >= 2:
        return "SURVIVES"
    if sum(e_c[t] < E_MARGIN and e_u[t] >= E_MARGIN for t in TREES) >= 2:
        return "PHYSICS"
    if sum(e_u[t] < E_MARGIN for t in TREES) >= 2:
        return "NOT_TESTABLE"
    return "MIXED"


def tree_branch(e_rf: float, e_lgbm: float) -> str:
    if e_rf >= E_MARGIN or e_lgbm >= E_MARGIN:
        return "TREE_TRANSFER_SURVIVES"
    return "PHYSICS_REMOVES_IT_FOR_TREES_TOO"


def seed_mean(records, **key) -> float:
    v = [r["attacked_f1"] for r in records if all(r[k] == x for k, x in key.items())]
    if not v:
        raise KeyError(f"no records for {key}")
    return float(np.mean(v))


def summarize(records: list[dict], bdd: list[dict], ctx_eps: list) -> dict:
    E = {}
    for att in ("U", "C"):
        for eps in ctx_eps:
            for fam in FAMILIES:
                rnd = seed_mean(records, attacker=att, eps=eps, family=fam, kind="random")
                for src in [*SOURCES, "xgb_square"]:
                    tr = seed_mean(
                        records, attacker=att, eps=eps, family=fam, kind=f"transfer_{src}"
                    )
                    E[f"{att}|{eps}|{src}|{fam}"] = rnd - tr
    top = max(ctx_eps)
    e_u = {t: E[f"U|{top}|mlp|{t}"] for t in TREES}
    e_c = {t: E[f"C|{top}|mlp|{t}"] for t in TREES}
    ctl5 = []
    for att in ("U", "C"):
        for eps in ctx_eps:
            for src in SOURCES:
                wb = seed_mean(records, attacker=att, eps=eps, family=src, kind=f"transfer_{src}")
                rd = seed_mean(records, attacker=att, eps=eps, family=src, kind="random")
                ctl5.append(
                    {
                        "attacker": att,
                        "eps": eps,
                        "source": src,
                        "whitebox_f1": wb,
                        "random_f1": rd,
                        "admitted": wb < rd,
                    }
                )
    flag = {}
    for b in bdd:
        k = f"{b['attacker']}|{b['eps']}|{b['source']}"
        flag.setdefault(k, []).append(b)
    flag_rates = {
        k: {
            "flag_rate": float(np.mean([b["flag_rate"] for b in v])),
            "clean_flag_rate": float(np.mean([b["clean_flag_rate"] for b in v])),
            "agreement": float(np.mean([b["agreement"] for b in v])),
            "min_agreement_over_seeds": float(np.min([b["agreement"] for b in v])),
        }
        for k, v in flag.items()
    }
    square_ctl = {
        eps: {
            "xgb_square_f1": seed_mean(
                records, attacker="C", eps=eps, family="xgb", kind="transfer_xgb_square"
            ),
            "xgb_random_f1": seed_mean(records, attacker="C", eps=eps, family="xgb", kind="random"),
        }
        for eps in ctx_eps
    }
    return {
        "E": E,
        "H3": {
            "E_U": e_u,
            "E_C": e_c,
            "consistency_effect": {t: e_u[t] - e_c[t] for t in TREES},
            "branch": h3_branch(e_u, e_c),
            "eps": top,
        },
        "tree_arm": {
            "E_xgb_C_rf": E[f"C|{top}|xgb_square|rf"],
            "E_xgb_C_lgbm": E[f"C|{top}|xgb_square|lgbm"],
            "E_xgb_U_rf": E[f"U|{top}|xgb_square|rf"],
            "E_xgb_U_lgbm": E[f"U|{top}|xgb_square|lgbm"],
            "branch": tree_branch(E[f"C|{top}|xgb_square|rf"], E[f"C|{top}|xgb_square|lgbm"]),
            "positive_control": square_ctl,
        },
        "control5": ctl5,
        "bdd_flag_rates": flag_rates,
    }


# -------------------------------------------------------------------------------- context


def build_ctx(args) -> Ctx:
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    fcfg = fdia_config()
    bp = args.best_params or best_params_path(args.variant, QUBITS)
    best = json.loads((REPO_ROOT / bp).read_text())
    Xraw, y, splits = load_processed(args.variant)
    aux = load_aux(args.variant)
    reg = prepare_regime(args.variant, "pca", n_components=QUBITS, seed=0)
    pipe = build_preprocessor("pca", n_components=QUBITS, seed=0)
    pipe.fit(Xraw[splits["train"]])
    mapping = control_mapping(Xraw, splits, pipe, reg)
    n_eval = SMOKE["n_eval"] if args.smoke else cfg_adv["eval_subset_adv"]
    ev = stratified_cap(reg.X["test"], reg.y["test"], n_eval, seed=0)
    rows = splits["test"][ev]
    z0 = Xraw[rows].astype(np.float64)
    scaler = pipe.named_steps["scale"]
    U0 = scaler.transform(pipe.named_steps["impute"].transform(z0))
    ctx = Ctx(
        variant=args.variant,
        case_name=fcfg["variants"][args.variant]["case"],
        reg=reg,
        best_params=best,
        subset_cap=cfg_q["qkernel"]["subset_cap"],
        W=pipe.named_steps["pca"].components_.astype(np.float64),
        s=scaler.scale_.astype(np.float64),
        Z0=reg.X["test"][ev].astype(np.float64),
        U0=U0,
        z0=z0,
        xhat=aux["x_hat"][rows].astype(np.float64),
        sig_state=aux["x_hat"][splits["train"]].std(axis=0),
        y=reg.y["test"][ev],
        floor=stratified_random_floor(reg.y["train"], reg.y["test"][ev]),
        n_steps=SMOKE["n_steps"] if args.smoke else cfg_adv["evasion"]["pgd"]["n_steps"],
        step_frac=cfg_adv["evasion"]["pgd"]["step_frac"],
        square_iter=SMOKE["square_iter"] if args.smoke else SQUARE_ITER,
        eps_grid=SMOKE["eps"] if args.smoke else EPS_GRID,
        alpha=fcfg["estimator"]["alpha"],
        threads=int(_THREADS),
        test_pos=ev,
        tuned_file=Path(bp).resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
    )
    if np.any(ctx.sig_state <= 0):
        raise ValueError("a state has zero train std")
    mapping["eval_u_vs_regime_max_abs"] = float(  # float64 u vs the float32 regime: recorded
        np.abs(pipe.named_steps["pca"].transform(U0) - ctx.Z0).max()
    )
    return ctx, mapping


# ---------------------------------------------------------------------------------- units

_CTX: Ctx | None = None


def _init_worker(ctx: Ctx):
    global _CTX
    got = {v: os.environ.get(v) for v in THREAD_VARS}
    if any(x != str(ctx.threads) for x in got.values()):
        raise AssertionError(f"worker {os.getpid()}: thread caps {got}, expected {ctx.threads}")
    _CTX = ctx


def control_zero_and_gradient(zoo, ctx: Ctx, case: Case, h0: np.ndarray) -> dict:
    zero = state_map(case, ctx.xhat, h0, np.zeros_like(ctx.xhat))
    if np.any(zero != 0.0):
        raise AssertionError("control: a zero state offset changed the measurements")
    Zz = ctx.Z0 + (zero / ctx.s) @ ctx.W.T
    if not np.array_equal(Zz, ctx.Z0):
        raise AssertionError("control: a zero state offset moved the model inputs")
    # identical inputs; the forest averages its trees across threads, so its probabilities may
    # differ in the last bits between calls: labels must match, probabilities within 1e-12
    proba_diff = {}
    for fam in FAMILIES:
        pa, pb = zoo.predict_proba(fam, Zz), zoo.clean_proba(fam, ctx.Z0, ctx.y)
        proba_diff[fam] = float(np.abs(pa - pb).max())
        if not np.array_equal(pa.argmax(1), pb.argmax(1)) or proba_diff[fam] > 1e-12:
            raise AssertionError(f"control: zero offset moved {fam}'s predictions")
    svc = zoo.models["rbf_svm"]
    sv, a = svc.support_vectors_, svc.dual_coef_[0]
    gamma = svc._gamma

    def loss(D):
        Z = ctx.Z0 + (state_map(case, ctx.xhat, h0, D) / ctx.s) @ ctx.W.T
        f = np.exp(-gamma * ((Z[:, None, :] - sv[None]) ** 2).sum(axis=2)) @ a + svc.intercept_[0]
        return np.logaddexp(0.0, f) - ctx.y * f

    g_fn = c_space_grad(GradProvider(svc, "rbf_svm"), ctx, case, h0)
    D = spawn_rng(0, "fdia_ctl_grad").normal(0.0, 0.25, size=ctx.xhat.shape) * ctx.sig_state
    g = g_fn(D, ctx.y)
    worst, hstep = 0.0, 1e-6
    for i in np.argsort(-np.abs(g).max(axis=1))[:3]:
        for j in np.argsort(-np.abs(g[i]))[:5]:
            up, dn = D.copy(), D.copy()
            up[i, j] += hstep
            dn[i, j] -= hstep
            fd = (loss(up)[i] - loss(dn)[i]) / (2 * hstep)
            worst = max(worst, abs(fd - g[i, j]) / max(abs(fd), abs(g[i, j]), 1e-12))
    if worst > GRAD_TOL:
        raise AssertionError(f"control: C-space gradient off by {worst:.2e} relative")
    return {"zero_offset_proba_max_abs_diff": proba_diff, "grad_max_rel_err": float(worst)}


def run_seed(seed: int) -> dict:
    ctx = _CTX
    t0 = time.perf_counter()
    zoo = FittedZoo(ctx.reg, ENCODING, QUBITS, seed, ctx.subset_cap, ctx.best_params)
    t_fit = time.perf_counter() - t0
    wbs, gate = make_whitebox(zoo, ctx)
    case = Case(ctx.case_name, fdia_config())
    h0 = case.h_rows(ctx.xhat)
    ctl = control_zero_and_gradient(zoo, ctx, case, h0)
    thr = case.threshold(ctx.alpha)
    clean_flags = bdd_flags(case, ctx.z0, thr)
    y, W, s = ctx.y, ctx.W, ctx.s
    clean_p = {f: zoo.clean_proba(f, ctx.Z0, ctx.y) for f in FAMILIES}
    clean = {f: _f1(y, clean_p[f]) for f in FAMILIES}
    records, bdd = [], []
    ex = {"preds": [("clean", f, clean_p[f]) for f in FAMILIES], "crafted": []}

    def score(att, eps, kind, dU, extra=None, state=None):
        Z = ctx.Z0 + dU @ W.T
        nm = norms(dU)
        ex["crafted"].append((f"{att}@{eps}", kind, ctx.U0 + dU))
        if state is not None:
            ex["crafted"].append((f"{att}_state_offset@{eps}", kind, state))
        for fam in FAMILIES:
            proba = zoo.predict_proba(fam, Z)
            ex["preds"].append((f"{att}@{eps}:{kind}", fam, proba))
            records.append(
                {
                    "seed": seed,
                    "attacker": att,
                    "eps": eps,
                    "family": fam,
                    "kind": kind,
                    "clean_f1": clean[fam],
                    "attacked_f1": _f1(y, proba),
                    **nm,
                    **(extra or {}),
                }
            )

    def flags(att, eps, src, dz):
        f = bdd_flags(case, ctx.z0 + dz, thr)
        bdd.append(
            {
                "seed": seed,
                "attacker": att,
                "eps": eps,
                "source": src,
                "flag_rate": float(f.mean()),
                "clean_flag_rate": float(clean_flags.mean()),
                "agreement": float(np.mean(f == clean_flags)),
            }
        )

    for eps in ctx.eps_grid:
        for src in SOURCES:
            dU_u = (
                pgd(
                    u_space_grad(wbs[src], ctx),
                    ctx.U0,
                    y,
                    eps,
                    np.ones(ctx.U0.shape[1]),
                    n_steps=ctx.n_steps,
                    step_frac=ctx.step_frac,
                    rng=spawn_rng(seed, f"fdia_U_{src}_{eps}"),
                )
                - ctx.U0
            )
            D = pgd(
                c_space_grad(wbs[src], ctx, case, h0),
                np.zeros_like(ctx.xhat),
                y,
                eps,
                ctx.sig_state,
                n_steps=ctx.n_steps,
                step_frac=ctx.step_frac,
                rng=spawn_rng(seed, f"fdia_C_{src}_{eps}"),
            )
            dz_c = state_map(case, ctx.xhat, h0, D)
            dU_c = dz_c / s
            score("U", eps, f"transfer_{src}", dU_u)
            score("C", eps, f"transfer_{src}", dU_c, state=D)
            flags("C", eps, src, dz_c)
            if src == "mlp":
                flags("U", eps, src, dU_u * s)
                ratio = np.linalg.norm(dU_c, axis=1) / np.linalg.norm(dU_u, axis=1)
                score("U_l2_matched", eps, "transfer_mlp", dU_u * ratio[:, None])
        dU_r = spawn_rng(seed, f"fdia_U_random_{eps}").choice([-1.0, 1.0], size=ctx.U0.shape) * eps
        D_r = (
            spawn_rng(seed, f"fdia_C_random_{eps}").choice([-1.0, 1.0], size=ctx.xhat.shape)
            * eps
            * ctx.sig_state
        )
        dz_r = state_map(case, ctx.xhat, h0, D_r)
        score("U", eps, "random", dU_r)
        score("C", eps, "random", dz_r / s, state=D_r)
        flags("U", eps, "random", dU_r * s)
        flags("C", eps, "random", dz_r)

        def p_u(U):
            return zoo.predict_proba("xgb", ctx.Z0 + (U - ctx.U0) @ W.T)

        def p_c(D):
            return zoo.predict_proba("xgb", ctx.Z0 + (state_map(case, ctx.xhat, h0, D) / s) @ W.T)

        Ua, _ = square_attack(
            p_u,
            ctx.U0,
            y,
            eps,
            np.ones(ctx.U0.shape[1]),
            spawn_rng(seed, f"fdia_sq_U_{eps}"),
            ctx.square_iter,
        )
        Da, q = square_attack(
            p_c,
            np.zeros_like(ctx.xhat),
            y,
            eps,
            ctx.sig_state,
            spawn_rng(seed, f"fdia_sq_C_{eps}"),
            ctx.square_iter,
        )
        score("U", eps, "transfer_xgb_square", Ua - ctx.U0, {"queries": q})
        score(
            "C",
            eps,
            "transfer_xgb_square",
            state_map(case, ctx.xhat, h0, Da) / s,
            {"queries": q},
            state=Da,
        )
        log.info("seed %d eps %.2f done (%.0f s)", seed, eps, time.perf_counter() - t0)
    return {
        "seed": seed,
        "records": records,
        "bdd": bdd,
        "control": ctl,
        "gate": gate,
        "clean_flag_rate": float(clean_flags.mean()),
        "fit_s": round(t_fit, 1),
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_rss_gb": _peak_rss_gb(),
        "exports": ex,
    }


def stream_names(seed: int, eps_grid: list) -> list[str]:
    """Every spawn_rng stream a unit draws from, in the unit's order."""
    out = ["fdia_ctl_grad"]
    for eps in eps_grid:
        for src in SOURCES:
            out += [f"fdia_U_{src}_{eps}", f"fdia_C_{src}_{eps}"]
        out += [f"fdia_U_random_{eps}", f"fdia_C_random_{eps}", f"fdia_sq_U_{eps}"]
        out += [f"fdia_sq_C_{eps}"]
    return [f"{seed}:{n}" for n in out]


def write_exports(run_path, args, ctx: Ctx, units: list[dict], resume_dir=None) -> None:
    """per_seed.csv, predictions.parquet, crafted_inputs.npz and exports_provenance.json
    (operator exports rule, decision log 2026-11-17T02:18Z) from the per-seed pickles."""
    exp = Exports(args.variant, ctx.tuned_file)
    for u in units:
        sd = u["seed"]
        for nm in stream_names(sd, ctx.eps_grid):
            exp.stream(nm)
        src_dir = run_path if (run_path / f"unit_exports_seed{sd}.pkl").exists() else resume_dir
        with open(src_dir / f"unit_exports_seed{sd}.pkl", "rb") as fh:
            ue = pickle.load(fh)
        for cond, fam, proba in ue["preds"]:
            exp.predictions(sd, cond, fam, proba, ctx.y, ctx.test_pos)
        for cond, src, arr in ue["crafted"]:
            exp.crafted_input(sd, cond, src, arr)
        for r in u["records"]:
            cond = f"{r['attacker']}@{r['eps']}:{r['kind']}"
            for k in ("clean_f1", "attacked_f1", "du_linf_median", "du_l2_median", "queries"):
                if k in r:
                    exp.metric(sd, r["family"], cond, k, r[k])
        for b in u["bdd"]:
            cond = f"{b['attacker']}@{b['eps']}:{b['source']}"
            for k in ("flag_rate", "clean_flag_rate", "agreement"):
                exp.metric(sd, "chi2_bdd", cond, k, b[k])
    exp.extra = {
        "threads": ctx.threads,
        "workers": args.workers,
        "crafted_space": "standardised measurements u (U0 + du); *_state_offset* keys hold d",
        "eval_rows": "test_pos is the position in the variant's test split",
    }
    exp.write(run_path)
    for u in units:
        pk = run_path / f"unit_exports_seed{u['seed']}.pkl"
        if pk.exists():
            pk.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["fdia14_binary", "fdia118_binary"])
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker-rss-gb", type=float, default=1.5)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--best-params", default=None, help="smoke only: a tuned-parameter JSON")
    ap.add_argument("--resume", default=None, help="run dir whose results_partial.json to reuse")
    args = ap.parse_args()
    if args.best_params and not args.smoke:
        raise ValueError("--best-params is for the smoke; a full run reads the variant's own file")
    seeds = args.seeds or (SMOKE["seeds"] if args.smoke else SEEDS)
    ctx, mapping = build_ctx(args)
    mem = memory_guard(args.workers, args.worker_rss_gb, "fdia consistent attack")
    config = {
        **vars(args),
        "seeds": seeds,
        "eps_grid": ctx.eps_grid,
        "n_eval": len(ctx.y),
        "n_steps": ctx.n_steps,
        "square_iter": ctx.square_iter,
        "floor": ctx.floor,
        "memory": mem,
    }
    name = f"fdia_consistent_attack_{args.variant}" + ("_smoke" if args.smoke else "")
    done = []
    if args.resume:
        done = json.loads((REPO_ROOT / args.resume / "results_partial.json").read_text())["units"]
    with RunDir(name, config=config, seeds=seeds) as run:
        run.write_json("mapping_control.json", mapping)
        todo = [sd for sd in seeds if sd not in {u["seed"] for u in done}]
        units = list(done)

        def on_done(u):  # arrays go to a per-seed pickle, the rest to results_partial.json
            with open(run.path / f"unit_exports_seed{u['seed']}.pkl", "wb") as fh:
                pickle.dump(u.pop("exports"), fh)
            units.append(u)
            run.write_json("results_partial.json", {"units": units})
            log.info(
                "seed %d finished: %.0f s, peak %.2f GB", u["seed"], u["wall_s"], u["peak_rss_gb"]
            )

        failed = []
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(ctx,),
        ) as ex:
            futs = {ex.submit(run_seed, sd): sd for sd in todo}
            for f in as_completed(futs):
                if f.exception() is not None:
                    log.error("seed %d failed", futs[f], exc_info=f.exception())
                    failed.append(futs[f])
                    continue
                on_done(f.result())
        if failed:
            raise RuntimeError(f"seeds {failed} failed; finished ones are in results_partial.json")
        units.sort(key=lambda u: u["seed"])
        records = [r for u in units for r in u["records"]]
        bdd = [b for u in units for b in u["bdd"]]
        summary = summarize(records, bdd, ctx.eps_grid)
        run.write_json("results.json", {"units": units, "summary": summary})
        write_exports(run.path, args, ctx, units, REPO_ROOT / args.resume if args.resume else None)
        run.metrics.update(
            {
                "H3": summary["H3"]["branch"],
                "tree_arm": summary["tree_arm"]["branch"],
                "peak_rss_gb": max(u["peak_rss_gb"] for u in units),
                "unit_wall_s": [u["wall_s"] for u in units],
            }
        )
        log.info("H3 %s", json.dumps(summary["H3"]))
        log.info("tree arm %s", json.dumps(summary["tree_arm"]))
        bad = {
            k: v
            for k, v in summary["bdd_flag_rates"].items()
            if k.startswith("C|") and v["min_agreement_over_seeds"] < STEALTH_MIN
        }
        if bad:
            raise AssertionError(f"stealth control failed: {json.dumps(bad)}")
        pc = summary["tree_arm"]["positive_control"]
        if not all(v["xgb_square_f1"] < v["xgb_random_f1"] for v in pc.values()):
            raise AssertionError(f"tree-arm positive control failed: {json.dumps(pc)}")
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
