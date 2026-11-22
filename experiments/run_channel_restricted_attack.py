"""E-M1a: does the white-box kernel contrast survive a PMU-channel-restricted attacker?

The published attacks perturb the models' PCA-8 input directly, which grants the attacker
every one of the 8 PCA coordinates. A grid attacker controls measurement CHANNELS instead:
the physical PMU channels (phasors, frequency, impedance), not the integer-valued logs,
Snort alerts, control-panel counters or relay status flags. This runner perturbs the
standardized RAW 128-channel space u (the StandardScaler output of the frozen pipeline
impute -> StandardScaler -> PCA(8), fit on train only) inside an L-inf ball over the channels
the attacker controls, and maps the perturbation EXACTLY into the models' input:

    delta_Z = delta_u @ W.T      (W = pca.components_, 8 x 128, PCA not whitened)
    g_u     = g_Z @ W            (chain rule through the same linear map)

so the existing PCA-space gradient providers are reused unchanged and `attacks.evasion.pgd`
runs on the 128-d space with a per-channel scale (1 for controlled channels, 1e-12 for blocked
ones; pgd requires scale > 0, and the mask control proves blocked channels do not move).

Arms: A0 all 128 channels; A1 physical PMU channels only; A2-R1..R4 the physical channels of
one relay's PMU. eps in {0.1, 0.25, 0.5} raw sigma (L-inf), PGD steps/step_frac from
configs/adversarial.yaml, model seeds {0, 1, 2}. White-box attacks on QSVM, RBF-SVM, MLP and
VQC; the MLP-crafted perturbation is transferred to every family. Reported per seed / arm /
eps / family: clean and attacked macro-F1, retention, F1 minus the stratified-random floor
(Table III's definition, 2,000 draws), the induced PCA-space radius in PCA sigma units, and the
channel count. Pre-registered in experiment_decision_log.md (E-M1a); Stage 2 (physical
consistency projection) is parked and not implemented here.

Blocking controls (each raises): 1 anchor (published PCA-space attack at eps 0.5, model seed
0, reproduces QSVM 0.064 and RBF 0.685 retention within 0.03); 2 mapping (rebuilt pipeline
equals the regime features; delta_u @ W.T equals the pipeline's response to a raw
perturbation, both within 1e-9); 3 mask (blocked channels move < 1e-9; allowed channels stay
inside the eps ball); 4 gradient (central finite differences of the RBF-SVM loss in u agree
with g_u within 1e-4); 5 negative (a random sign perturbation in the same masked ball does
less damage than PGD, per white-box family; evaluated at the end on the seed means, after the
results are written).

Run:  python -m experiments.run_channel_restricted_attack --threads 10
      python -m experiments.run_channel_restricted_attack --smoke --threads 10
      python -m experiments.run_channel_restricted_attack --anchor-only --threads 10
"""

from __future__ import annotations

import os
import sys


def _early_threads(argv: list[str]) -> str:
    """Thread budget, read from argv BEFORE numpy/torch load so the BLAS/OpenMP pools obey it."""
    for i, a in enumerate(argv):
        if a == "--threads":
            return argv[i + 1]
        if a.startswith("--threads="):
            return a.split("=", 1)[1]
    return "10"


_THREADS = _early_threads(sys.argv)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_var] = _THREADS  # LOKY_MAX_CPU_COUNT caps joblib's n_jobs=-1 (RF, LightGBM)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from experiments.run_adversarial import CLASSICAL, FittedZoo, _f1  # noqa: E402
from experiments.run_feature_block_ablation import block_of  # noqa: E402
from qgridbench.attacks.evasion import pgd  # noqa: E402
from qgridbench.attacks.gradients import GradProvider  # noqa: E402
from qgridbench.attacks.qsvm_whitebox import QSVMWhiteBox, assert_matches_reference  # noqa: E402
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.data.validate import load_variant_frames  # noqa: E402
from qgridbench.features.reduce import build_preprocessor  # noqa: E402
from qgridbench.models.classical.zoo import stratified_cap  # noqa: E402
from qgridbench.protocol import prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import spawn_rng  # noqa: E402

log = get_logger(__name__)

VARIANT = "binary"
QUBITS = 8
ENCODING = "zz"
FAMILIES = CLASSICAL + ["qsvm", "vqc"]
WHITEBOX = ["mlp", "rbf_svm", "qsvm", "vqc"]
NONPHYSICAL_BLOCKS = {"status_flag", "control_panel_log", "relay_log", "snort_log"}
BLOCK_SCALE = 1e-12  # pgd needs scale > 0; blocked channels are checked to stay put (control 3)
EPS_GRID = [0.1, 0.25, 0.5]
SEEDS = [0, 1, 2]
RELAYS = [1, 2, 3, 4]
# pre-registered anchor (control 1): published PCA-space white-box retention at eps 0.5
ANCHOR_EPS, ANCHOR_TOL = 0.5, 0.03
ANCHOR_PUBLISHED = {"qsvm": 0.064, "rbf_svm": 0.685}
MAP_TOL, GRAD_TOL = 1e-9, 1e-4
BRANCH_SURVIVES, BRANCH_ATTENUATED = 0.3, 0.1
METRICS = ["clean_f1", "attacked_f1", "retention", "f1_minus_floor", "radius_mean", "radius_max"]


@dataclass
class Context:
    """Everything fixed across seeds: the frozen pipeline, the eval subset and the arms."""

    reg: object
    cfg_adv: dict
    best_params: dict
    subset_cap: int
    W: np.ndarray  # (8, 128) float64
    Z0: np.ndarray  # (n_eval, 8) published regime features of the eval rows
    U0: np.ndarray  # (n_eval, 128) standardized raw channels of the same rows
    y: np.ndarray
    floor: float
    columns: list[str]
    excluded: np.ndarray  # bool (128,): channels fixed in A1/A2 (non-physical)
    arm_masks: dict[str, np.ndarray]
    n_steps: int
    step_frac: float
    map_control: dict
    partition: dict | None = None  # channel counts and the A1 exclusion list, for the results


def channel_partition(Xraw, splits, columns):
    """Discrete channels exactly as run_physical_realizability does, cross-checked by name."""
    train_raw = Xraw[splits["train"]]
    finite = np.isfinite(train_raw)
    is_discrete = np.array(
        [
            bool(np.all(np.equal(np.mod(train_raw[finite[:, j], j], 1), 0)))
            for j in range(train_raw.shape[1])
        ]
    )
    named_nonphysical = np.array([block_of(c) in NONPHYSICAL_BLOCKS for c in columns])
    # A1 fixes every channel that is integer-valued OR named a log / Snort / control-panel /
    # status-flag channel; the two rules are compared and any disagreement is reported
    disagree = [
        (c, bool(d), bool(n))
        for c, d, n in zip(columns, is_discrete, named_nonphysical, strict=True)
        if d != n
    ]
    log.info(
        "channels: %d discrete (integer rule), %d named non-physical, disagreements %s",
        is_discrete.sum(),
        named_nonphysical.sum(),
        disagree,
    )
    return is_discrete | named_nonphysical, is_discrete, named_nonphysical, disagree


def build_arms(columns, excluded):
    masks = {"A0": np.ones(len(columns), dtype=bool), "A1": ~excluded}
    for r in RELAYS:
        in_relay = np.array([c.startswith((f"R{r}-", f"R{r}:")) for c in columns])
        masks[f"A2-R{r}"] = in_relay & ~excluded
    for name, m in masks.items():
        if not m.any():
            raise ValueError(f"arm {name} has no channels")
    return masks


def stratified_random_floor(y_train, y_eval, draws=2000):
    """Table III's F1-floor: mean macro-F1 of prior-matched random labels on the eval subset."""
    cls, cnt = np.unique(y_train, return_counts=True)
    rng = np.random.default_rng(0)
    return float(
        np.mean(
            [
                f1_score(
                    y_eval, rng.choice(cls, size=len(y_eval), p=cnt / cnt.sum()), average="macro"
                )
                for _ in range(draws)
            ]
        )
    )


def control_mapping(Xraw, splits, pipe, reg):
    """Control 2: the rebuilt pipeline reproduces the regime, and delta_u @ W.T is exact."""
    scaler, pca = pipe.named_steps["scale"], pipe.named_steps["pca"]
    if pca.whiten:
        raise AssertionError("PCA is whitened: delta_Z = delta_u @ W.T would not hold")
    W = pca.components_.astype(np.float64)
    out = {}
    for s in ("train", "val", "test"):
        err = float(np.abs(pipe.transform(Xraw[splits[s]]) - reg.X[s]).max())
        out[f"pipeline_vs_regime_{s}_max_abs"] = err
        if err > MAP_TOL:
            raise AssertionError(
                f"control 2a: rebuilt pipeline differs from reg.X[{s}] by {err:.2e}"
            )
    raw_test = Xraw[splits["test"]].astype(np.float64)
    rows = np.flatnonzero(np.isfinite(raw_test).all(axis=1))[:200]
    if len(rows) < 50:
        raise AssertionError(f"control 2b: only {len(rows)} NaN-free test rows")
    raw = raw_test[rows]
    du = spawn_rng(0, "m1a_map").normal(0.0, 0.5, size=raw.shape)
    dz_pipe = pipe.transform(raw + du * scaler.scale_) - pipe.transform(raw)
    err = float(np.abs(dz_pipe - du @ W.T).max())
    out.update({"delta_map_max_abs": err, "n_rows": int(len(rows))})
    if err > MAP_TOL:
        raise AssertionError(f"control 2b: delta_u @ W.T differs from the pipeline by {err:.2e}")
    # informational: float64 u of the eval rows vs the float32 regime features (not a gate)
    log.info("control 2 passed: %s", json.dumps(out))
    return out


def build_context(args) -> Context:
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    best_params = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    Xraw, y, splits = load_processed(VARIANT)
    reg = prepare_regime(VARIANT, "pca", n_components=QUBITS, seed=0)
    pipe = build_preprocessor("pca", n_components=QUBITS, seed=0)
    pipe.fit(Xraw[splits["train"]])
    map_control = control_mapping(Xraw, splits, pipe, reg)

    label_col = load_yaml(REPO_ROOT / "configs" / "data.yaml")["schema"]["label_column"]
    columns = [c for c in load_variant_frames(VARIANT)[0].columns if c != label_col]
    if len(columns) != Xraw.shape[1]:
        raise AssertionError(f"{len(columns)} columns vs {Xraw.shape[1]} features")
    excluded, is_discrete, named, disagree = channel_partition(Xraw, splits, columns)
    arm_masks = build_arms(columns, excluded)

    n_eval = 40 if args.smoke else cfg_adv["eval_subset_adv"]
    ev_idx = stratified_cap(reg.X["test"], reg.y["test"], n_eval, seed=0)
    imputer, scaler = pipe.named_steps["impute"], pipe.named_steps["scale"]
    U0 = scaler.transform(imputer.transform(Xraw[splits["test"]][ev_idx].astype(np.float64)))
    Z0 = reg.X["test"][ev_idx].astype(np.float64)
    y_ev = reg.y["test"][ev_idx]
    W = pipe.named_steps["pca"].components_.astype(np.float64)
    map_control["eval_u_vs_regime_max_abs"] = float(
        np.abs(pipe.named_steps["pca"].transform(U0) - Z0).max()
    )  # float64 u against the float32 regime features: precision gap, recorded not gated
    ctx = Context(
        reg=reg,
        cfg_adv=cfg_adv,
        best_params=best_params,
        subset_cap=cfg_q["qkernel"]["subset_cap"],
        W=W,
        Z0=Z0,
        U0=U0,
        y=y_ev,
        floor=stratified_random_floor(reg.y["train"], y_ev),
        columns=columns,
        excluded=excluded,
        arm_masks=arm_masks,
        n_steps=3 if args.smoke else cfg_adv["evasion"]["pgd"]["n_steps"],
        step_frac=cfg_adv["evasion"]["pgd"]["step_frac"],
        map_control=map_control,
    )
    ctx.partition = {  # attached for the results file
        "n_discrete_integer_rule": int(is_discrete.sum()),
        "n_named_nonphysical": int(named.sum()),
        "disagreements_name_vs_integer": [
            {"column": c, "integer_rule": d, "named_nonphysical": n} for c, d, n in disagree
        ],
        "excluded_from_A1": [c for c, e in zip(columns, excluded, strict=True) if e],
        "arm_channels": {k: int(m.sum()) for k, m in arm_masks.items()},
    }
    log.info("arm channel counts: %s", json.dumps(ctx.partition["arm_channels"]))
    log.info("excluded from A1: %s", ctx.partition["excluded_from_A1"])
    log.info("eval rows %d, stratified-random floor %.4f", len(y_ev), ctx.floor)
    return ctx


def u_space_grad(grad_z, ctx: Context):
    """Wrap a PCA-space gradient provider into a u-space one: g_u = grad_Z(Z0 + (U-U0) W.T) @ W."""

    def g(U, y):
        return grad_z(ctx.Z0 + (U - ctx.U0) @ ctx.W.T, y) @ ctx.W

    return g


def make_whitebox(zoo, ctx: Context):
    svc = zoo.models["qsvm"]
    bw = ctx.best_params.get("qk_bw", 1.0)
    wb = QSVMWhiteBox(svc, zoo._states_tr, zoo._angle, ENCODING, bw)
    gate_idx = stratified_cap(ctx.reg.X["test"], ctx.reg.y["test"], 40, seed=0)
    gate = assert_matches_reference(
        wb, svc, zoo._states_tr, zoo._angle, ctx.reg.X["test"][gate_idx]
    )
    return {
        "qsvm": wb,
        "rbf_svm": GradProvider(zoo.models["rbf_svm"], "rbf_svm"),
        "mlp": zoo.grad_provider("mlp"),
        "vqc": zoo.grad_provider("vqc"),
    }, gate


def control_gradient(zoo, ctx: Context, h=1e-5):
    """Control 4: central FD of the RBF-SVM BCE loss in u against g_u (3 points x 5 channels)."""
    svc = zoo.models["rbf_svm"]
    sv, alpha = svc.support_vectors_, svc.dual_coef_[0]
    gamma = svc._gamma if hasattr(svc, "_gamma") else svc.gamma

    def loss(Z, y):  # BCE(sigmoid(f), y) with f the decision the GradProvider differentiates
        f = np.exp(-gamma * ((Z[:, None, :] - sv[None, :, :]) ** 2).sum(axis=2)) @ alpha
        f = f + svc.intercept_[0]
        return np.logaddexp(0.0, f) - y * f

    gu = u_space_grad(GradProvider(svc, "rbf_svm"), ctx)
    rng = spawn_rng(0, "m1a_ctl4")
    U = ctx.U0 + rng.normal(0.0, 0.25, size=ctx.U0.shape)
    g = gu(U, ctx.y)
    rows = np.argsort(-np.abs(g).max(axis=1))[:3]  # unsaturated points, so FD has signal
    worst = 0.0
    for i in rows:
        for j in np.argsort(-np.abs(g[i]))[:5]:
            up, dn = U.copy(), U.copy()
            up[i, j] += h
            dn[i, j] -= h
            fd = (
                loss(ctx.Z0 + (up - ctx.U0) @ ctx.W.T, ctx.y)[i]
                - loss(ctx.Z0 + (dn - ctx.U0) @ ctx.W.T, ctx.y)[i]
            ) / (2 * h)
            worst = max(worst, abs(fd - g[i, j]) / max(abs(fd), abs(g[i, j]), 1e-12))
    out = {"max_rel_err": float(worst), "rows": [int(r) for r in rows], "h": h}
    if worst > GRAD_TOL:
        raise AssertionError(
            f"control 4: u-space gradient off by {worst:.2e} relative > {GRAD_TOL}"
        )
    log.info("control 4 passed: %s", json.dumps(out))
    return out


def control_anchor(zoo, ctx: Context, wbs):
    """Control 1: the published PCA-space white-box PGD at eps 0.5 (p5a setup) is reproduced."""
    reg = ctx.reg
    ev = stratified_cap(reg.X["test"], reg.y["test"], ctx.cfg_adv["eval_subset_adv"], seed=0)
    X, y = reg.X["test"][ev], reg.y["test"][ev]
    rng = spawn_rng(0, "m1a_anchor")
    out = {"n_eval": int(len(ev)), "eps": ANCHOR_EPS, "published": ANCHOR_PUBLISHED, "timing_s": {}}
    for fam in ("qsvm", "rbf_svm"):
        t0 = time.perf_counter()
        Xa = pgd(
            wbs[fam],
            X,
            y,
            ANCHOR_EPS,
            reg.feature_scale,
            n_steps=ctx.cfg_adv["evasion"]["pgd"]["n_steps"],
            step_frac=ctx.step_frac,
            rng=rng,
        )
        out["timing_s"][fam] = round(time.perf_counter() - t0, 1)
        clean, adv = _f1(y, zoo.clean_proba(fam, X, y)), _f1(y, zoo.predict_proba(fam, Xa))
        out[fam] = {"clean_f1": clean, "attacked_f1": adv, "retention": adv / clean}
        log.info(
            "anchor %s: clean %.3f -> %.3f, retention %.3f (published %.3f), %.0f s",
            fam,
            clean,
            adv,
            adv / clean,
            ANCHOR_PUBLISHED[fam],
            out["timing_s"][fam],
        )
    out["passed"] = all(
        abs(out[f]["retention"] - ANCHOR_PUBLISHED[f]) <= ANCHOR_TOL for f in ANCHOR_PUBLISHED
    )
    return out


def run_seed(seed, ctx: Context, arms, eps_grid, anchor: bool, run: RunDir):
    t_seed = time.perf_counter()
    zoo = FittedZoo(ctx.reg, ENCODING, QUBITS, seed, ctx.subset_cap, ctx.best_params)
    t_fit = time.perf_counter() - t_seed
    wbs, gate = make_whitebox(zoo, ctx)
    log.info(
        "seed %d: zoo fitted in %.0f s; QSVM gate fwd %.2e grad %.2e",
        seed,
        t_fit,
        gate["forward_rel_err"],
        gate["grad_rel_err"],
    )
    grad_ctl = control_gradient(zoo, ctx)
    anchor_res = None
    if anchor:
        anchor_res = control_anchor(zoo, ctx, wbs)
        run.write_json("anchor.json", anchor_res)
        if not anchor_res["passed"]:
            raise AssertionError(f"control 1 (anchor) failed: {json.dumps(anchor_res)}")

    Z0, U0, y, W = ctx.Z0, ctx.U0, ctx.y, ctx.W
    scale_z = ctx.reg.feature_scale
    clean = {f: _f1(y, zoo.clean_proba(f, Z0, y)) for f in FAMILIES}
    rng = spawn_rng(seed, "m1a")
    rng_rand = spawn_rng(seed, "m1a_random")
    records, timing = [], defaultdict(float)

    def record(arm, eps, fam, kind, Zadv, dZ, n_ch):
        f1 = _f1(y, zoo.predict_proba(fam, Zadv))
        if clean[fam] <= 0:
            raise ValueError(f"clean macro-F1 of {fam} is {clean[fam]}")
        radius = (np.abs(dZ) / scale_z).max(axis=1)  # per row: max_k |dZ_k| / PCA sigma_k
        records.append(
            {
                "seed": seed,
                "arm": arm,
                "eps": eps,
                "family": fam,
                "kind": kind,
                "n_channels": n_ch,
                "clean_f1": clean[fam],
                "attacked_f1": f1,
                "retention": f1 / clean[fam],
                "f1_minus_floor": f1 - ctx.floor,
                "radius_mean": float(radius.mean()),
                "radius_max": float(radius.max()),
            }
        )

    for arm in arms:
        mask = ctx.arm_masks[arm]
        n_ch = int(mask.sum())
        scale_u = np.where(mask, 1.0, BLOCK_SCALE)
        for eps in eps_grid:
            for src in ("mlp", "qsvm", "rbf_svm", "vqc"):
                t0 = time.perf_counter()
                U = pgd(
                    u_space_grad(wbs[src], ctx),
                    U0,
                    y,
                    eps,
                    scale_u,
                    n_steps=ctx.n_steps,
                    step_frac=ctx.step_frac,
                    rng=rng,
                )
                timing[src] += time.perf_counter() - t0
                d = U - U0
                if (~mask).any() and np.abs(d[:, ~mask]).max() >= 1e-9:  # control 3
                    raise AssertionError(f"control 3: blocked channel moved in {arm} ({src})")
                if np.abs(d[:, mask]).max() > eps + 1e-9:
                    raise AssertionError(f"control 3: eps ball exceeded in {arm} ({src})")
                dZ = d @ W.T
                Zadv = Z0 + dZ
                record(arm, eps, src, "whitebox", Zadv, dZ, n_ch)
                if src == "mlp":  # the MLP surrogate's perturbation, applied to every family
                    for fam in FAMILIES:
                        record(arm, eps, fam, "transfer_mlp", Zadv, dZ, n_ch)
            # control 5 reference: random signs of magnitude eps on the controlled channels
            d = rng_rand.choice([-1.0, 1.0], size=U0.shape) * eps * mask
            dZ = d @ W.T
            for fam in FAMILIES:
                record(arm, eps, fam, "random", Z0 + dZ, dZ, n_ch)
            log.info(
                "seed %d | %s eps %.2f done (%.0f s elapsed)",
                seed,
                arm,
                eps,
                time.perf_counter() - t_seed,
            )
    return {
        "seed": seed,
        "records": records,
        "gate": gate,
        "control4_gradient": grad_ctl,
        "anchor": anchor_res,
        "timing_s": {"zoo_fit": round(t_fit, 1), **{k: round(v, 1) for k, v in timing.items()}},
        "wall_s": round(time.perf_counter() - t_seed, 1),
    }


def _ms(vals):
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


def aggregate(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["arm"], r["eps"], r["family"], r["kind"])].append(r)
    rows = []
    for (arm, eps, fam, kind), rs in groups.items():
        row = {
            "arm": arm,
            "eps": eps,
            "family": fam,
            "kind": kind,
            "n_channels": rs[0]["n_channels"],
            "n_seeds": len(rs),
        }
        row.update({m: _ms([r[m] for r in rs]) for m in METRICS})
        rows.append(row)
    return rows


def branch(delta_by_eps: dict[float, float]) -> str:
    """Pre-registered reading (E-M1a), headline at the largest eps."""
    top = delta_by_eps[max(delta_by_eps)]
    if top >= BRANCH_SURVIVES:
        return "survives"
    if top >= BRANCH_ATTENUATED:
        return "survives_attenuated"
    if all(v < BRANCH_ATTENUATED for v in delta_by_eps.values()):
        return "kill"
    return "unspecified: delta < 0.1 at the largest eps but >= 0.1 at another"


def contrast(records, arms, eps_grid):
    """Delta = retention(rbf_svm white-box) - retention(qsvm white-box), per arm and eps."""
    by = {}
    for r in records:
        if r["kind"] == "whitebox" and r["family"] in ("qsvm", "rbf_svm"):
            by[(r["seed"], r["arm"], r["eps"], r["family"])] = r["retention"]
    seeds = sorted({k[0] for k in by})
    per_arm = {}
    for arm in arms:
        per_arm[arm] = {
            str(e): _ms([by[(s, arm, e, "rbf_svm")] - by[(s, arm, e, "qsvm")] for s in seeds])
            for e in eps_grid
        }
    out = {"per_arm": per_arm}
    a1 = {e: per_arm["A1"][str(e)]["mean"] for e in eps_grid} if "A1" in per_arm else None
    if a1:
        out["A1_branch"] = branch(a1)
    relays = [a for a in arms if a.startswith("A2-")]
    if relays:
        mean_rel = {
            e: float(np.mean([per_arm[a][str(e)]["mean"] for a in relays])) for e in eps_grid
        }
        worst = min(relays, key=lambda a: per_arm[a][str(max(eps_grid))]["mean"])
        out["A2"] = {
            "mean_over_relays_delta": mean_rel,
            "mean_over_relays_branch": branch(mean_rel),
            "worst_relay": worst,
            "worst_relay_delta": {e: per_arm[worst][str(e)]["mean"] for e in eps_grid},
            "worst_relay_branch": branch({e: per_arm[worst][str(e)]["mean"] for e in eps_grid}),
            "per_relay_branch": {
                a: branch({e: per_arm[a][str(e)]["mean"] for e in eps_grid}) for a in relays
            },
        }
    return out


def control_negative(summary, arms, eps_grid):
    """Control 5: random signs in the same masked ball damage less than PGD (white-box families)."""
    idx = {(r["arm"], r["eps"], r["family"], r["kind"]): r for r in summary}
    cells = []
    for arm in arms:
        for eps in eps_grid:
            for fam in WHITEBOX:
                pgd_f1 = idx[(arm, eps, fam, "whitebox")]["attacked_f1"]["mean"]
                rnd_f1 = idx[(arm, eps, fam, "random")]["attacked_f1"]["mean"]
                cells.append(
                    {
                        "arm": arm,
                        "eps": eps,
                        "family": fam,
                        "pgd_f1": pgd_f1,
                        "random_f1": rnd_f1,
                        "pass": bool(rnd_f1 > pgd_f1),
                    }
                )
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=10, help="BLAS/OpenMP/joblib thread cap")
    ap.add_argument("--smoke", action="store_true", help="40 rows, seed 0, A0+A1, eps 0.5, 3 steps")
    ap.add_argument("--anchor-only", action="store_true", help="control 1 only, then stop")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--resume", type=Path, help="results_partial.json of an interrupted run")
    args = ap.parse_args()

    ctx = build_context(args)
    suffix = "_smoke" if args.smoke else "_anchor" if args.anchor_only else ""
    name = f"channel_restricted_attack_{VARIANT}_q{QUBITS}{suffix}"

    if args.anchor_only:
        with RunDir(name, config=vars(args) | {"threads": _THREADS}, seeds=[0]) as run:
            zoo = FittedZoo(ctx.reg, ENCODING, QUBITS, 0, ctx.subset_cap, ctx.best_params)
            wbs, gate = make_whitebox(zoo, ctx)
            res = control_anchor(zoo, ctx, wbs) | {"qsvm_gate": gate}
            run.write_json("anchor.json", res)
            if not res["passed"]:
                raise AssertionError(f"control 1 (anchor) failed: {json.dumps(res)}")
            print(f"-> {run.path}")
        return

    seeds = [0] if args.smoke else args.seeds
    arms = ["A0", "A1"] if args.smoke else list(ctx.arm_masks)
    eps_grid = [0.5] if args.smoke else EPS_GRID
    cfg = vars(args) | {
        "resume": str(args.resume) if args.resume else None,
        "threads": _THREADS,
        "arms": arms,
        "eps_grid": eps_grid,
        "n_steps": ctx.n_steps,
        "step_frac": ctx.step_frac,
        "n_eval": int(len(ctx.y)),
        "block_scale": BLOCK_SCALE,
    }
    with RunDir(name, config=cfg, seeds=seeds) as run:
        per_seed = []
        if args.resume:
            per_seed = json.loads(args.resume.read_text())["per_seed"]
            log.info("resuming: seeds already done %s", [p["seed"] for p in per_seed])
        done = {p["seed"] for p in per_seed}
        for s in seeds:
            if s in done:
                continue
            # the anchor (control 1) is blocking and runs once, on model seed 0, unless smoke
            per_seed.append(
                run_seed(s, ctx, arms, eps_grid, anchor=(s == 0 and not args.smoke), run=run)
            )
            run.write_json("results_partial.json", {"per_seed": per_seed})
        records = [r for p in per_seed for r in p["records"]]
        summary = aggregate(records)
        ctl5 = control_negative(summary, arms, eps_grid)
        results = {
            "config": cfg,
            "constant_predictor_floor": ctx.floor,
            "partition": ctx.partition,
            "control2_mapping": ctx.map_control,
            "control4_gradient": [p["control4_gradient"] for p in per_seed],
            "control1_anchor": next((p["anchor"] for p in per_seed if p["anchor"]), None),
            "qsvm_gate": [p["gate"] for p in per_seed],
            "contrast": contrast(records, arms, eps_grid),
            "control5_negative": ctl5,
            "timing_s": {p["seed"]: p["timing_s"] | {"wall": p["wall_s"]} for p in per_seed},
            "summary": summary,
            "per_seed": per_seed,
        }
        run.write_json("results.json", results)
        for arm, d in results["contrast"]["per_arm"].items():
            log.info(
                "Delta(rbf - qsvm retention) %s: %s",
                arm,
                {e: f"{v['mean']:+.3f}+/-{v['std']:.3f}" for e, v in d.items()},
            )
        log.info("A1 branch: %s", results["contrast"].get("A1_branch"))
        log.info("A2: %s", json.dumps(results["contrast"].get("A2")))
        failed = [c for c in ctl5 if not c["pass"]]
        print(f"-> {run.path}")
        if failed:
            raise AssertionError(
                f"control 5 failed (random >= PGD damage) in {len(failed)} cells: {failed}"
            )


if __name__ == "__main__":
    main()
