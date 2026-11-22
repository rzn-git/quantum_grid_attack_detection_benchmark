"""R4-14: a QUPID-type Gaussian noise defense (PixelDP construction) under the multi-axis suite.

Every family is refit on a noisy copy of the shared 2,000-row training subset and predicts with
the mean of predict_proba over M seeded Gaussian draws added to its pre-encoding input; sigma is
in the attack's own unit (`reg.feature_scale`, the per-component train std), chosen per family on
validation only. The defense is then read with E12b's two-step rule: step 1, is it real under an
adaptive attack (PGD whose gradient averages K fresh noise draws, expectation over transformation,
crafted against the defended model itself); step 2, only if it is, does the cross-family picture
move. QUPID's depolarizing half is not run: at QUPID's placement it rescales the readout by
(1 - 4p/3), which the trainable head absorbs (decision log 2026-11-16T11:39Z, the R4-14 amendment
and its analytic note). Pre-registration: that entry; design note
`tmp/review_shihav_20261116/r4_14_qupid_dp_design.md`.

Run (the anchor reproduces Table III bit for bit at 3 threads; --threads is applied before numpy
loads, in every worker):
    python -m experiments.run_noise_defense --smoke --threads 3
    python -m experiments.run_noise_defense --workers 2 --threads 3 --worker-rss-gb 1.2

Stages, each unit's result written as it finishes (resume with --resume <results_partial.json>):
  A  selection, units (seed, sigma), seeds {0, 1, 2}, sigma in SIGMAS, on the 500-row validation
     stratified_cap(seed=0): every family's clean macro-F1 (M-draw mean) and attacked macro-F1 at
     eps 0.5: EOT-PGD (K draws) for the white-box families (MLP, VQC, QSVM, RBF-SVM), the
     undefended-MLP-surrogate PGD set for logreg, RF, XGBoost and LightGBM. A sigma is eligible
     if its seed-mean clean macro-F1 is at least the undefended one minus 0.05 and its clean
     predictions are finite and not constant; the selected sigma maximises the seed-mean attacked
     macro-F1 (ties to the smaller sigma). sigma 0 is always eligible.
  B0 the prediction-draw check (seed 0, validation, before any test row is read): each family
     at its selected sigma, clean macro-F1 with M = 16 draws against 64. Ruling 6 (decision log
     2026-11-16T13:05Z): if any family differs by more than 0.005, the 64-draw prediction is
     primary in EVERY stage-B cell and the 16-draw prediction of the same inputs is reported
     beside it (with its own step-1 reading); the PGD gradient keeps its 16-draw EOT.
  B  evaluation, units (seed), on the 500-row test subset (Table III's), touched once, after
     selection: the undefended zoo replays Table III's RNG stream exactly (the anchor), then each
     family at its selected sigma: clean, EOT-PGD white-box at eps {0.1, 0.25, 0.5}, the
     single-draw (K = 1) masking check at 0.5, the zero-radius check, Table III's own
     MLP-surrogate set (transfer), benign sensor noise at 30, 20, 10 dB, and random signs at 0.5.
     Seed 0 also runs the EOT adequacy check (K at 64).
The black-box stage (Square against the M-draw predictor) is not in this runner: it runs only if
step 1 says the defense is real, and is reported before it is launched.

Decision rules (fixed in the amendment):
  step 1  REAL if, for at least one of MLP, VQC, QSVM, RBF-SVM, the seed-mean defended minus
          undefended macro-F1 under white-box PGD at 0.5 sigma (EOT, crafted against the defended
          model) is >= 0.05; OBFUSCATED if none clears it under EOT but one clears it with K = 1;
          else NULL. Only REAL opens step 2.
  step 2  per axis (white-box, transfer, sensor noise; black-box in its own stage) and quantum
          family q in {VQC, QSVM}: ND = (clean_q - attacked_q) / mean classical damage (logreg,
          RBF-SVM, RF, XGBoost, LightGBM; for the white-box axis their MLP-surrogate transfer
          damage, as in E12b); the axis MOVES if |ND_defended - ND_undefended| >= 0.10 for
          either q; if the mean classical damage is < 0.02 in either arm, it moves if
          |delta damage_q - delta mean classical damage| >= 0.05. Accuracy moves if the sign of
          (best quantum - best classical) clean macro-F1 flips with |difference| >= 0.02.
          The sensor-noise axis is read on the mean damage over the three SNR levels.
VQC configuration (R4-23 ruling 2, decision log 2026-11-16T13:33Z): the VQC uses Table I's
configuration (the vqc_* keys of results/adv_best_params_binary_q8_table1.json, which must equal
R4-23's recorded depth-6 configuration); every other family uses Table III's tuned file.
Controls: anchor (blocking, full run) every family's clean and published attacked macro-F1 at 0.5
and its sensor-noise macro-F1 reproduce Table III within 1e-6 at seeds 0-2, except the VQC, whose
clean and white-box macro-F1 reproduce R4-23's depth-6 arm (its sensor noise is not anchored);
zero-radius (blocking) EOT-PGD at eps 0 returns the defended clean macro-F1 exactly; the
standalone undefended MLP used for the validation surrogate set equals the zoo's MLP (blocking);
the prediction-draw check (stage B0; it decides the primary draw count under ruling 6, it does
not block); EOT adequacy (blocking, seed 0) EOT-PGD attacked macro-F1 at K = 16 and 64 within
0.02; attack-strength ordering (reported) EOT attacked <= single-draw attacked + 0.02;
random-sign null (reported); the clean predictor guard on every defended model's clean test
predictions (blocking); the QSVM white-box gate on every QSVM.
"""

from __future__ import annotations

import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, required=True)
THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
for _var in THREAD_VARS:
    os.environ[_var] = str(THREADS)
# VQC statevector chunk, fixed (vqc.resolve_chunk's documented override). Left to resolve_chunk it
# sizes to 35% of available RAM, which on this host is 4,096 rows: one EOT-stacked gradient call
# (16 draws x 500 rows) then holds about 0.9 GB of backprop statevectors, the run's excess over its
# 1.3 GB budget. The value is per-row arithmetic only; the VQC anchor checks that nothing moved.
VQC_CHUNK = 256
os.environ["QGB_VQC_CHUNK"] = str(VQC_CHUNK)

import copy  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from experiments.run_adversarial import CLASSICAL, FittedZoo  # noqa: E402
from experiments.run_square_blackbox import square_attack  # noqa: E402
from experiments.runner_guard import assert_clean_predictor  # noqa: E402
from qgridbench.attacks.evasion import fgsm, gaussian_sensor_noise, pgd  # noqa: E402
from qgridbench.attacks.gradients import GradProvider  # noqa: E402
from qgridbench.attacks.qsvm_whitebox import QSVMWhiteBox, assert_matches_reference  # noqa: E402
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap  # noqa: E402
from qgridbench.protocol import best_params_path, kernel_subset, prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import spawn_rng  # noqa: E402

log = get_logger(__name__)

VARIANT, QUBITS, ENCODING = "binary", 8, "zz"  # Table III's regime
TABLE_III_RUN = "20260813T183448Z_adversarial_binary_q8"
# R4-23 ruling 2 (decision log 2026-11-16T13:33Z): the VQC uses Table I's configuration, read from
# this file; every other family keeps Table III's tuned file. The undefended VQC is anchored to
# R4-23's depth-6 arm, every other family to Table III.
TABLE1_PARAMS = REPO_ROOT / "results" / "adv_best_params_binary_q8_table1.json"
R4_23_RUN = REPO_ROOT / "results" / "runs" / "20261116T122057Z_vqc_config_check_binary_q8"
R4_23_DEPTH6 = R4_23_RUN / "table_iii_depth6" / "results.json"
VQC_KEYS = ("vqc_depth", "vqc_lr", "vqc_batch", "vqc_epochs")
SEEDS = [0, 1, 2]
SIGMAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)  # attack units
M_DRAWS, K_DRAWS, ADEQ_DRAWS = 16, 16, 64
EPS, EPS_CURVE = 0.5, (0.1, 0.25, 0.5)
FAMILIES = CLASSICAL + ["qsvm", "vqc"]
WHITEBOX = ("mlp", "vqc", "qsvm", "rbf_svm")
CLASSICAL_TARGETS = ("logreg", "rbf_svm", "rf", "xgb", "lgbm")  # E12b's normalisation set
QUANTUM = ("vqc", "qsvm")
COST_CAP, STEP1_BAR, PICTURE_BAR = 0.05, 0.05, 0.10
SMALL_DEN, SMALL_DEN_BAR, ACC_BAR = 0.02, 0.05, 0.02
ANCHOR_TOL, MC_TOL, EOT_TOL, ORDER_TOL = 1e-6, 0.005, 0.02, 0.02
# attack-strength control (decision log 2026-11-17, R4-14 interim entry)
STRENGTH_RESTARTS, STRENGTH_STEPS, STRENGTH_STEP_FRAC = 5, 100, 0.025
SQUARE_ITERS = 600
ATTACKS = ("pgd20", "pgd100x5", "square")
FLOOR = 0.415  # the constant-predictor macro-F1 the lead's entry reads against
DRAW_BLOCK = 8  # draws per stacked predict_proba call (memory only; the mean is the same)
# rows per gradient call inside the EOT stack (a multiple of VQC_CHUNK and of the QSVM provider's
# 256-row chunk). rbf_svm_loss_input_grad broadcasts a (rows, support vectors, 8) float64 array:
# on the whole 8,000-row stack that is about 1.5 GB at once.
EOT_BLOCK = 1024
MEM_FRACTION = 0.8
SMOKE = {"sigmas": (0.0, 0.5), "n": 40, "n_steps": 3, "m": 4, "k": 4, "adeq": 8, "seeds": [0]}


def _f1(y, proba) -> float:
    return float(f1_score(y, np.asarray(proba).argmax(1), average="macro"))


def _peak_rss_gb() -> float:
    return psutil.Process().memory_info().peak_wset / 2**30


def _peak_private_gb() -> float:
    """Peak private bytes (Windows PeakPagefileUsage, the commit charge), what the admission
    guard budgets; the working-set peak above can be lower when pages are trimmed."""
    return psutil.Process().memory_info().peak_pagefile / 2**30


# --------------------------------------------------------------------------- context


_CTX: dict = {}


def ctx(smoke: bool) -> dict:
    """Per-process context: configs, tuned parameters, the regime and both eval subsets."""
    if not _CTX:
        cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
        cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
        reg = prepare_regime(VARIANT, "pca", n_components=QUBITS, seed=0)
        n = SMOKE["n"] if smoke else cfg_adv["eval_subset_adv"]
        te = stratified_cap(reg.X["test"], reg.y["test"], n, seed=0)
        va = stratified_cap(reg.X["val"], reg.y["val"], n, seed=0)
        _CTX.update(
            cfg_adv=cfg_adv,
            cap=cfg_q["qkernel"]["subset_cap"],
            bp=tuned_params(),
            reg=reg,
            scale=reg.feature_scale,
            Xte=reg.X["test"][te],
            te_idx=te,
            yte=reg.y["test"][te],
            Xva=reg.X["val"][va],
            yva=reg.y["val"][va],
            n_steps=SMOKE["n_steps"] if smoke else cfg_adv["evasion"]["pgd"]["n_steps"],
            step_frac=cfg_adv["evasion"]["pgd"]["step_frac"],
            m=SMOKE["m"] if smoke else M_DRAWS,
            k=SMOKE["k"] if smoke else K_DRAWS,
            adeq=SMOKE["adeq"] if smoke else ADEQ_DRAWS,
        )
    return _CTX


def tuned_params(table3=None, table1=TABLE1_PARAMS, r4_23=R4_23_RUN / "results.json") -> dict:
    """Table III's tuned file with the VQC keys of Table I's file (ruling 2). The merged VQC
    configuration must be exactly the one R4-23's depth-6 arm recorded, or this raises."""
    table3 = table3 or best_params_path(VARIANT, QUBITS)
    bp = json.loads(Path(table3).read_text())
    over = {k: v for k, v in json.loads(Path(table1).read_text()).items() if k.startswith("vqc_")}
    missing = set(over) - set(bp)
    if not over or missing:
        raise KeyError(f"Table I file VQC keys {sorted(over)}; absent from Table III's: {missing}")
    merged = {**bp, **over}
    ref = {
        json.dumps(u["vqc_config"], sort_keys=True)
        for u in json.loads(Path(r4_23).read_text())["units"]
        if u["arm"] == "depth6"
    }
    if len(ref) != 1:
        raise AssertionError(f"R4-23's depth-6 arm records {len(ref)} VQC configurations")
    ours = {k: merged[k] for k in VQC_KEYS}
    if ours != json.loads(ref.pop()):
        raise AssertionError(f"VQC configuration {ours} differs from R4-23's depth-6 arm")
    return merged


def _init_worker(threads: int) -> None:
    got = {v: os.environ.get(v) for v in THREAD_VARS}
    if any(x != str(threads) for x in got.values()):
        raise AssertionError(f"worker {os.getpid()}: thread caps {got}, expected {threads}")
    torch.set_num_threads(threads)


# --------------------------------------------------------------------------- the defense


def noisy_regime(reg, sigma: float, seed: int, cap: int):
    """A shallow Regime copy whose training subset rows carry one seeded Gaussian draw each, in
    attack units; every other row, split, scaler and label is the original object."""
    if sigma == 0.0:
        return reg
    sub = kernel_subset(reg, cap, seed=0)
    Xtr = reg.X["train"].copy()
    rng = spawn_rng(seed, f"nd_train_{sigma}")
    Xtr[sub] = Xtr[sub] + sigma * reg.feature_scale * rng.standard_normal((len(sub), Xtr.shape[1]))
    r = copy.copy(reg)
    r.X = dict(reg.X)
    r.X["train"] = Xtr
    if not np.array_equal(kernel_subset(r, cap, seed=0), sub):
        raise AssertionError("the noisy regime selects a different training subset")
    return r


def fit_zoo(c: dict, sigma: float, seed: int) -> FittedZoo:
    reg = noisy_regime(c["reg"], sigma, seed, c["cap"])
    return FittedZoo(reg, ENCODING, QUBITS, seed, c["cap"], c["bp"])


class Defended:
    """One family's M-draw mean predictor. The draws E (M, n, d) are fixed per (seed, family,
    rows), so the predictor is deterministic on those rows; sigma 0 is the undefended model."""

    def __init__(self, zoo: FittedZoo, fam: str, sigma: float, scale: np.ndarray, E):
        self.zoo, self.fam, self.sigma, self.scale, self.E = zoo, fam, sigma, scale, E

    def proba(self, X: np.ndarray) -> np.ndarray:
        if self.sigma == 0.0:
            return self.zoo.predict_proba(self.fam, X)
        if X.shape != self.E.shape[1:]:
            raise AssertionError(f"rows {X.shape} differ from the draws {self.E.shape[1:]}")
        m, n, d = self.E.shape
        total = 0.0
        for b in range(0, m, DRAW_BLOCK):  # blocks of draws bound the stacked batch's memory
            E = self.E[b : b + DRAW_BLOCK]
            Xs = (X[None] + self.sigma * self.scale * E).reshape(len(E) * n, d)
            total = total + self.zoo.predict_proba(self.fam, Xs).reshape(len(E), n, -1).sum(0)
        return total / m


def draws(seed: int, tag: str, m: int, X: np.ndarray) -> np.ndarray:
    return spawn_rng(seed, tag).standard_normal((m, *X.shape))


def eot(grad_fn, sigma: float, scale: np.ndarray, k: int, rng: np.random.Generator):
    """Expectation over transformation: the input gradient averaged over k fresh noise draws per
    call. The k copies are stacked and passed to the provider in blocks of EOT_BLOCK rows; every
    provider's gradient is per row, so blocking bounds memory without changing a value."""
    if sigma == 0.0:
        return grad_fn

    def g(X, y):
        n, d = X.shape
        Xs = (X[None] + sigma * scale * rng.standard_normal((k, n, d))).reshape(k * n, d)
        ys = np.tile(y, k)
        out = np.concatenate(
            [
                grad_fn(Xs[i : i + EOT_BLOCK], ys[i : i + EOT_BLOCK])
                for i in range(0, k * n, EOT_BLOCK)
            ]
        )
        return out.reshape(k, n, d).mean(axis=0)

    return g


def provider(zoo: FittedZoo, fam: str, bw: float):
    """White-box input-gradient provider in the pre-encoding space; the QSVM's is gated against
    the deployed kernel SVC on 40 test rows."""
    if fam in ("mlp", "vqc"):
        return zoo.grad_provider(fam), None
    if fam == "rbf_svm":
        return GradProvider(zoo.models["rbf_svm"], "rbf_svm"), None
    wb = QSVMWhiteBox(zoo.models["qsvm"], zoo._states_tr, zoo._angle, ENCODING, bw)
    reg = zoo.reg
    gi = stratified_cap(reg.X["test"], reg.y["test"], 40, seed=0)
    gate = assert_matches_reference(
        wb, zoo.models["qsvm"], zoo._states_tr, zoo._angle, reg.X["test"][gi]
    )
    return wb, gate


def attack(c, zoo, fam, sigma, X, y, eps, k, seed, tag):
    gp, gate = provider(zoo, fam, c["bp"].get("qk_bw", 1.0))
    g = eot(gp, sigma, c["scale"], k, spawn_rng(seed, f"nd_eot_{tag}"))
    Xa = pgd(
        g,
        X,
        y,
        eps,
        c["scale"],
        n_steps=c["n_steps"],
        step_frac=c["step_frac"],
        rng=spawn_rng(seed, f"nd_pgd_{tag}"),
    )
    return Xa, gate


def clean_ok(y, proba) -> bool:
    """Eligibility (selection only): finite, and more than one predicted class."""
    return bool(np.isfinite(proba).all() and len(np.unique(proba.argmax(1))) > 1)


# --------------------------------------------------------------------------- stage A


def surrogate_mlp(c: dict, seed: int):
    """The undefended MLP of this seed, fitted alone (random_state = seed), as in FittedZoo."""
    sub = kernel_subset(c["reg"], c["cap"], seed=0)
    est = build("mlp", c["bp"].get("mlp", {}), seed)
    fit_model("mlp", est, c["reg"].X["train"][sub], c["reg"].y["train"][sub])
    return est


def unit_a(spec: dict) -> dict:
    seed, sigma, smoke = spec["seed"], spec["sigma"], spec["smoke"]
    c = ctx(smoke)
    t0 = time.perf_counter()
    X, y, scale = c["Xva"], c["yva"], c["scale"]
    zoo = fit_zoo(c, sigma, seed)
    t_fit = time.perf_counter() - t0
    mlp = surrogate_mlp(c, seed)
    out = {"stage": "A", "seed": seed, "sigma": sigma, "families": {}}
    if sigma == 0.0:  # the standalone MLP must be the zoo's MLP
        same = np.array_equal(mlp.predict_proba(X), zoo.models["mlp"].predict_proba(X))
        out["standalone_mlp_equals_zoo"] = bool(same)
    X_sh = pgd(
        GradProvider(mlp, "mlp"),
        X,
        y,
        EPS,
        scale,
        n_steps=c["n_steps"],
        step_frac=c["step_frac"],
        rng=spawn_rng(seed, "nd_sur_val"),
    )
    for fam in FAMILIES:
        E = draws(seed, f"nd_mc_val_{fam}_{sigma}", c["m"], X)
        D = Defended(zoo, fam, sigma, scale, E)
        p = D.proba(X)
        rec = {"clean": _f1(y, p), "clean_ok": clean_ok(y, p), "transfer": _f1(y, D.proba(X_sh))}
        if fam in WHITEBOX:
            Xa, gate = attack(c, zoo, fam, sigma, X, y, EPS, c["k"], seed, f"val_{fam}_{sigma}")
            rec["whitebox"] = _f1(y, D.proba(Xa))
            if gate is not None:
                rec["qsvm_gate"] = gate
        rec["attacked"] = rec["whitebox"] if fam in WHITEBOX else rec["transfer"]
        out["families"][fam] = rec
    out |= {
        "zoo_fit_s": round(t_fit, 1),
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "peak_private_gb": round(_peak_private_gb(), 3),
        "pid": os.getpid(),
    }
    log.info(
        "A seed %d sigma %.2f: %s (%.0f s)",
        seed,
        sigma,
        {f: round(r["attacked"], 3) for f, r in out["families"].items()},
        out["wall_s"],
    )
    return out


def select(units_a: list[dict], sigmas) -> dict:
    """Per family: eligible sigmas (seed-mean clean >= undefended - cap, clean_ok at every seed)
    and the one with the largest seed-mean attacked validation macro-F1 (ties: smaller sigma)."""
    by = {(u["seed"], u["sigma"]): u for u in units_a}
    seeds = sorted({u["seed"] for u in units_a})
    out = {}
    for fam in FAMILIES:
        rows = {}
        for s in sigmas:
            recs = [by[(sd, s)]["families"][fam] for sd in seeds]
            rows[s] = {
                "clean": float(np.mean([r["clean"] for r in recs])),
                "attacked": float(np.mean([r["attacked"] for r in recs])),
                "clean_ok_all_seeds": all(r["clean_ok"] for r in recs),
            }
        base = rows[0.0]["clean"]
        elig = [
            s
            for s in sigmas
            if s == 0.0 or (rows[s]["clean_ok_all_seeds"] and rows[s]["clean"] >= base - COST_CAP)
        ]
        best = max(elig, key=lambda s: (rows[s]["attacked"], -s))
        out[fam] = {
            "selected_sigma": best,
            "eligible": elig,
            "by_sigma": {str(s): v for s, v in rows.items()},
        }
    return out


# --------------------------------------------------------------------------- stage B


def replay_table_iii(c: dict, zoo: FittedZoo, seed: int) -> dict:
    """run_adversarial's evasion_and_transfer + sensor_noise RNG consumption for one seed, with
    the crafted sets kept: the anchor, and Table III's own MLP-surrogate set at eps 0.5."""
    cfg = c["cfg_adv"]
    X, y, scale = c["Xte"], c["yte"], c["scale"]
    rng = spawn_rng(seed, "adv")
    grid = cfg["evasion"]["epsilons"]
    top = max(grid)
    kw = {"n_steps": c["n_steps"], "step_frac": c["step_frac"]}
    wb = {}
    for eps in grid:
        for src in cfg["transfer_matrix"]["sources"]:
            gp = zoo.grad_provider(src)
            fgsm(gp, X, y, eps, scale)  # no randomness; call parity
            Xa = pgd(gp, X, y, eps, scale, rng=rng, **kw)
            if eps == top:
                wb[src] = Xa
    gp = zoo.grad_provider(cfg["evasion"]["surrogate_model"])
    shared = None
    for eps in grid:
        Xa = pgd(gp, X, y, eps, scale, rng=rng, **kw)
        if eps == top:
            shared = Xa
    for _ in FAMILIES:  # the transfer-matrix row: one PGD (one random start) per target family
        pgd(gp, X, y, top, scale, rng=rng, **kw)
    noise = {}
    for snr in cfg["sensor_noise"]["snr_db"]:
        Xn = gaussian_sensor_noise(X, snr, rng, scale)
        noise[str(snr)] = {f: _f1(y, zoo.predict_proba(f, Xn)) for f in FAMILIES}
    published = {f: _f1(y, zoo.predict_proba(f, wb[f] if f in wb else shared)) for f in FAMILIES}
    return {"wb_sets": wb, "shared": shared, "published_pgd_0.5": published, "noise": noise}


def stage_b_draw_tag(fam: str, m: int, c: dict) -> str:
    """Stage-B prediction draws: M draws keep their tag; adequacy draws have their own."""
    return f"nd_mc_test_{fam}" if m == c["m"] else f"nd_mc_test{m}_{fam}"


def evaluate(c, zoo, fam, sigma, seed, shared, adequacy: bool, m_primary: int, m_beside) -> dict:
    """One family at one sigma on the test subset, every axis but black-box. Every number is the
    m_primary-draw prediction; with m_beside set (ruling 6), the m_beside-draw prediction of the
    same inputs is reported beside it under "beside". The PGD gradient keeps its K-draw EOT."""
    X, y, scale = c["Xte"], c["yte"], c["scale"]
    D = Defended(
        zoo, fam, sigma, scale, draws(seed, stage_b_draw_tag(fam, m_primary, c), m_primary, X)
    )
    Db = None
    if m_beside is not None:
        Eb = draws(seed, stage_b_draw_tag(fam, m_beside, c), m_beside, X)
        Db = Defended(zoo, fam, sigma, scale, Eb)
    rec, bes = {"sigma": sigma, "prediction_draws": m_primary}, {"prediction_draws": m_beside}

    def put(path, Xin):
        """Score Xin with the primary predictor (and the beside one) under the key path."""
        for target, P in ((rec, D), (bes, Db)):
            if P is None:
                continue
            for k in path[:-1]:
                target = target.setdefault(k, {})
            target[path[-1]] = _f1(y, P.proba(Xin))

    p = D.proba(X)
    assert_clean_predictor(f"{fam} sigma {sigma}", y, p, p.argmax(1))
    put(("clean",), X)
    put(("transfer",), shared)
    for snr in c["cfg_adv"]["sensor_noise"]["snr_db"]:
        Xn = gaussian_sensor_noise(X, snr, spawn_rng(seed, f"nd_snr_{snr}"), scale)
        put(("noise", str(snr)), Xn)
    sign = spawn_rng(seed, "nd_rand").choice([-1.0, 1.0], size=X.shape)
    put(("random_sign_0.5",), X + EPS * scale * sign)
    if fam in WHITEBOX:
        for eps in EPS_CURVE:
            Xa, gate = attack(c, zoo, fam, sigma, X, y, eps, c["k"], seed, f"test_{fam}_{eps}")
            put(("whitebox", str(eps)), Xa)
        if gate is not None:
            rec["qsvm_gate"] = gate
        Xa, _ = attack(c, zoo, fam, sigma, X, y, EPS, 1, seed, f"test_k1_{fam}")
        put(("whitebox_single_draw_0.5",), Xa)
        # at eps 0 PGD returns X whatever the gradient, so one draw suffices
        X0, _ = attack(c, zoo, fam, sigma, X, y, 0.0, 1, seed, f"test_zero_{fam}")
        rec["zero_radius_equals_clean"] = bool(_f1(y, D.proba(X0)) == rec["clean"])
        if adequacy and sigma > 0:
            Xa, _ = attack(c, zoo, fam, sigma, X, y, EPS, c["adeq"], seed, f"test_adeq_{fam}")
            rec["whitebox_0.5_k_adequacy"] = _f1(y, D.proba(Xa))
    if Db is not None:
        rec["beside"] = bes
    return rec


def unit_b0(spec: dict) -> dict:
    """The prediction-draw check (seed 0, validation): each family at its selected sigma > 0,
    clean macro-F1 with the M-draw mean against the adequacy-draw mean."""
    seed, smoke, sel = spec["seed"], spec["smoke"], spec["selected"]
    c = ctx(smoke)
    t0 = time.perf_counter()
    Xv, yv, scale = c["Xva"], c["yva"], c["scale"]
    check = {}
    for sigma in sorted({sel[f] for f in FAMILIES} - {0.0}):
        zoo = fit_zoo(c, sigma, seed)
        for fam in [f for f in FAMILIES if sel[f] == sigma]:
            Em = draws(seed, f"nd_mc_val_{fam}_{sigma}", c["m"], Xv)
            Ea = draws(seed, f"nd_mc_adeq_{fam}", c["adeq"], Xv)
            fm = _f1(yv, Defended(zoo, fam, sigma, scale, Em).proba(Xv))
            fa = _f1(yv, Defended(zoo, fam, sigma, scale, Ea).proba(Xv))
            check[fam] = {"sigma": sigma, "m": fm, "adeq": fa, "abs_diff": abs(fm - fa)}
        del zoo
    out = {
        "stage": "B0",
        "seed": seed,
        "mc_check": check,
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "peak_private_gb": round(_peak_private_gb(), 3),
        "pid": os.getpid(),
    }
    log.info("B0 prediction-draw check: %s", {f: round(v["abs_diff"], 4) for f, v in check.items()})
    return out


def prediction_draws(b0: dict, m: int, adeq: int) -> dict:
    """Ruling 6 (decision log 2026-11-16T13:05Z): if the M-against-adequacy-draw check exceeds
    MC_TOL in any cell, the adequacy-draw prediction is primary in every stage-B cell and the
    M-draw prediction is reported beside it; otherwise M draws are primary."""
    worst = max((v["abs_diff"] for v in b0["mc_check"].values()), default=0.0)
    switch = worst > MC_TOL
    return {
        "primary": adeq if switch else m,
        "beside": m if switch else None,
        "max_abs_diff": worst,
        "tol": MC_TOL,
        "switched": switch,
    }


def unit_b(spec: dict) -> dict:
    seed, smoke, sel = spec["seed"], spec["smoke"], spec["selected"]
    c = ctx(smoke)
    t0 = time.perf_counter()
    zoo0 = fit_zoo(c, 0.0, seed)
    X, y = c["Xte"], c["yte"]
    rep = replay_table_iii(c, zoo0, seed)
    und = {}
    for fam in FAMILIES:
        p = zoo0.clean_proba(fam, X, y)
        rec = {"clean": _f1(y, p), "transfer": _f1(y, zoo0.predict_proba(fam, rep["shared"]))}
        rec["noise"] = {
            str(snr): _f1(
                y,
                zoo0.predict_proba(
                    fam, gaussian_sensor_noise(X, snr, spawn_rng(seed, f"nd_snr_{snr}"), c["scale"])
                ),
            )
            for snr in c["cfg_adv"]["sensor_noise"]["snr_db"]
        }
        sign = spawn_rng(seed, "nd_rand").choice([-1.0, 1.0], size=X.shape)
        rec["random_sign_0.5"] = _f1(y, zoo0.predict_proba(fam, X + EPS * c["scale"] * sign))
        if fam in ("mlp", "vqc"):  # Table III's own white-box sets (anchored)
            rec["whitebox_0.5"] = _f1(y, zoo0.predict_proba(fam, rep["wb_sets"][fam]))
        elif fam in WHITEBOX:
            Xa, gate = attack(c, zoo0, fam, 0.0, X, y, EPS, 1, seed, f"test_und_{fam}")
            rec["whitebox_0.5"] = _f1(y, zoo0.predict_proba(fam, Xa))
        und[fam] = rec
    defended = {}
    for sigma in sorted({sel[f] for f in FAMILIES}):
        zoo = zoo0 if sigma == 0.0 else fit_zoo(c, sigma, seed)
        for fam in [f for f in FAMILIES if sel[f] == sigma]:
            if sigma == 0.0:
                defended[fam] = {"sigma": 0.0, "undefended": True}
                continue
            defended[fam] = evaluate(
                c,
                zoo,
                fam,
                sigma,
                seed,
                rep["shared"],
                spec["adequacy"],
                spec["m_primary"],
                spec["m_beside"],
            )
        del zoo
    out = {
        "stage": "B",
        "seed": seed,
        "anchor_values": {
            "clean": {f: und[f]["clean"] for f in FAMILIES},
            "published_pgd_0.5": rep["published_pgd_0.5"],
            "noise": rep["noise"],
        },
        "undefended": und,
        "defended": defended,
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "peak_private_gb": round(_peak_private_gb(), 3),
        "pid": os.getpid(),
    }
    log.info("B seed %d done (%.0f s)", seed, out["wall_s"])
    return out


def unit_anchor(spec: dict) -> dict:
    """Anchor only (ruling on R4-14, option a): the undefended zoo of one seed and the Table III
    replay, giving the anchor cells and nothing else."""
    seed = spec["seed"]
    c = ctx(False)
    t0 = time.perf_counter()
    zoo0 = fit_zoo(c, 0.0, seed)
    X, y = c["Xte"], c["yte"]
    rep = replay_table_iii(c, zoo0, seed)
    out = {
        "stage": "anchor",
        "seed": seed,
        "anchor_values": {
            "clean": {f: _f1(y, zoo0.clean_proba(f, X, y)) for f in FAMILIES},
            "published_pgd_0.5": rep["published_pgd_0.5"],
            "noise": rep["noise"],
        },
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "peak_private_gb": round(_peak_private_gb(), 3),
        "pid": os.getpid(),
    }
    log.info("anchor seed %d done (%.0f s)", seed, out["wall_s"])
    return out


def main_anchor_only(args, rss: float, avail: float) -> None:
    """Recheck all 111 anchor cells of a finished run with the corrected replay (seeds 0-2)."""
    cfg = {
        "mode": "anchor_only",
        "threads": args.threads,
        "workers": 1,
        "worker_rss_gb": rss,
        "available_gb_at_launch": avail,
        "vqc_chunk": VQC_CHUNK,
        "seeds": SEEDS,
        "anchor_for_run": str(args.anchor_only),
        "reason": "the full run's replay drew one transfer-matrix PGD start where Table III drew "
        "one per family (fixed in 9c7ab78), shifting only its sensor-noise anchor cells",
        "table_iii_run": TABLE_III_RUN,
        "vqc_reference": str(R4_23_DEPTH6.relative_to(REPO_ROOT)),
    }
    with RunDir(f"noise_defense_anchor_{VARIANT}_q{QUBITS}", config=cfg, seeds=SEEDS) as run:
        units: list = []

        def save():
            run.write_json("results_partial.json", {"anchor": units})

        run_units(unit_anchor, [{"seed": s} for s in SEEDS], 1, args.threads, units, save)
        units.sort(key=lambda u: u["seed"])
        anc = anchor(units)
        peaks = {
            "peak_rss_gb": max(u["peak_rss_gb"] for u in units),
            "peak_private_gb": max(u["peak_private_gb"] for u in units),
        }
        run.write_json(
            "results.json", {"config": cfg, "anchor": anc, "peaks": peaks, "units": units}
        )
        run.metrics.update(
            {
                "anchor_pass": anc["pass"],
                "max_abs_diff": anc["max_abs_diff"],
                "n_cells": anc["n_cells"],
                **peaks,
            }
        )
        log.info("anchor: %s", json.dumps({k: v for k, v in anc.items()}))
        print(f"-> {run.path}")
        if not anc["pass"]:
            raise AssertionError(f"anchor failed: {anc['worst']} by {anc['max_abs_diff']:.2e}")


def ce_loss(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row cross-entropy of the deployed prediction at the true label."""
    return -np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0))


def pgd_restarts(make_grad, predict, X, y, eps, scale, seed: int, tag: str) -> np.ndarray:
    """PGD with STRENGTH_RESTARTS random restarts of STRENGTH_STEPS steps at STRENGTH_STEP_FRAC;
    per row, the restart whose end point has the highest loss under the deployed predictor.
    make_grad(r) returns restart r's gradient function (its own EOT stream when defended)."""
    best_X, best_l = None, None
    for r in range(STRENGTH_RESTARTS):
        Xa = pgd(
            make_grad(r),
            X,
            y,
            eps,
            scale,
            n_steps=STRENGTH_STEPS,
            step_frac=STRENGTH_STEP_FRAC,
            rng=spawn_rng(seed, f"{tag}_start{r}"),
        )
        loss = ce_loss(predict(Xa), y)
        if best_X is None:
            best_X, best_l = Xa.copy(), loss
            continue
        up = loss > best_l
        best_X[up], best_l[up] = Xa[up], loss[up]
    return best_X


def run_cells(run_dir: Path) -> tuple[dict, dict]:
    """The full run's selection and its stage-B units by seed."""
    sel = json.loads((run_dir / "selection.json").read_text())
    units = json.loads((run_dir / "results_partial.json").read_text())["B"]
    return {f: v["selected_sigma"] for f, v in sel.items()}, {u["seed"]: u for u in units}


def unit_strength(spec: dict) -> dict:
    """Attack-strength control on the VQC (decision log 2026-11-17, R4-14 interim entry): the
    undefended and the defended VQC at eps 0.5, PGD-20 (the run's, reproduced), PGD-100 x 5
    restarts (step 0.025 eps, EOT over K draws when defended) and Square (600 iterations, the
    deployed predictor's scores: the 64-draw mean when defended). Every scored input and its
    probabilities go to <array_dir>/strength_s<seed>.npz (the exports rule)."""
    seed, sigma = spec["seed"], spec["sigma_vqc"]
    c = ctx(False)
    t0 = time.perf_counter()
    X, y, scale = c["Xte"], c["yte"], c["scale"]
    arrays: dict = {}

    def score(arm, cond, predict, Xin):
        p = predict(Xin)
        arrays[f"vqc|{arm}|{cond}|proba"] = p
        if cond != "clean":
            arrays[f"vqc|{arm}|{cond}|X"] = Xin
        return _f1(y, p)

    zoo0 = fit_zoo(c, 0.0, seed)
    rep = replay_table_iii(c, zoo0, seed)
    gp0 = zoo0.grad_provider("vqc")

    def p0(Z):
        return zoo0.predict_proba("vqc", Z)

    p_clean = zoo0.clean_proba("vqc", X, y)  # the runner guard on the clean prediction
    und = {
        "clean": score("undefended", "clean", lambda Z: p_clean, X),
        "pgd20": score("undefended", "pgd20", p0, rep["wb_sets"]["vqc"]),
    }
    Xa = pgd_restarts(lambda r: gp0, p0, X, y, EPS, scale, seed, "nd_str_und")
    und["pgd100x5"] = score("undefended", "pgd100x5", p0, Xa)
    Xs, q = square_attack(p0, X, y, EPS, scale, spawn_rng(seed, "nd_sq_und_vqc"), SQUARE_ITERS)
    und |= {"square": score("undefended", "square", p0, Xs), "square_queries": q}
    zoo0 = rep = None  # free the undefended zoo before the defended fit
    zoo = fit_zoo(c, sigma, seed)
    m = c["adeq"]
    D = Defended(zoo, "vqc", sigma, scale, draws(seed, stage_b_draw_tag("vqc", m, c), m, X))
    gp, _ = provider(zoo, "vqc", c["bp"].get("qk_bw", 1.0))
    Xa, _ = attack(c, zoo, "vqc", sigma, X, y, EPS, c["k"], seed, f"test_vqc_{EPS}")
    pd_clean = D.proba(X)  # the draws are fixed at construction, so this equals score's call
    assert_clean_predictor(f"vqc sigma {sigma} (defended)", y, pd_clean, pd_clean.argmax(1))
    dfd = {
        "clean": score("defended", "clean", lambda Z: pd_clean, X),
        "pgd20": score("defended", "pgd20", D.proba, Xa),
    }

    def make_grad(r):
        return eot(gp, sigma, scale, c["k"], spawn_rng(seed, f"nd_str_def_eot{r}"))

    Xa = pgd_restarts(make_grad, D.proba, X, y, EPS, scale, seed, "nd_str_def")
    dfd["pgd100x5"] = score("defended", "pgd100x5", D.proba, Xa)
    Xs, q = square_attack(D.proba, X, y, EPS, scale, spawn_rng(seed, "nd_sq_def_vqc"), SQUARE_ITERS)
    dfd |= {"square": score("defended", "square", D.proba, Xs), "square_queries": q}
    dfd |= {"sigma": sigma, "draws": m}
    np.savez_compressed(Path(spec["array_dir"]) / f"strength_s{seed}.npz", **arrays)
    out = {
        "stage": "strength",
        "seed": seed,
        "undefended": und,
        "defended": dfd,
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "peak_private_gb": round(_peak_private_gb(), 3),
        "pid": os.getpid(),
    }
    log.info("strength seed %d: und %s def %s (%.0f s)", seed, und, dfd, out["wall_s"])
    return out


def unit_axis(spec: dict) -> dict:
    """Square cells for the step-2 black-box axis: each classical target and the QSVM at its
    selected sigma (a sigma > 0 also gets its undefended cell); the VQC's are unit_strength's.
    Inputs and probabilities go to <array_dir>/axis_s<seed>.npz."""
    seed, sel = spec["seed"], spec["selected"]
    c = ctx(False)
    t0 = time.perf_counter()
    X, y, scale = c["Xte"], c["yte"], c["scale"]
    fams = list(CLASSICAL_TARGETS) + ["qsvm"]
    cells: dict = {f: {} for f in fams}
    arrays: dict = {}
    for sigma in sorted({0.0} | {sel[f] for f in fams}):
        zoo = fit_zoo(c, sigma, seed)
        for f in fams:
            if sigma == 0.0:
                arm = "undefended"
            elif sel[f] == sigma:
                arm = "defended"
            else:
                continue
            m = c["adeq"]
            E = None if sigma == 0.0 else draws(seed, stage_b_draw_tag(f, m, c), m, X)
            D = Defended(zoo, f, sigma, scale, E)
            Xs, q = square_attack(
                D.proba, X, y, EPS, scale, spawn_rng(seed, f"nd_sq_{arm}_{f}"), SQUARE_ITERS
            )
            pc, ps = D.proba(X), D.proba(Xs)
            assert_clean_predictor(f"{f} {arm} sigma {sigma}", y, pc, pc.argmax(1))
            arrays |= {
                f"{f}|{arm}|clean|proba": pc,
                f"{f}|{arm}|square|proba": ps,
                f"{f}|{arm}|square|X": Xs,
            }
            cells[f][arm] = {
                "clean": _f1(y, pc),
                "square": _f1(y, ps),
                "square_queries": q,
                "sigma": sigma,
            }
        del zoo
    for f in fams:
        if "defended" not in cells[f]:
            cells[f]["defended"] = cells[f]["undefended"]
    np.savez_compressed(Path(spec["array_dir"]) / f"axis_s{seed}.npz", **arrays)
    out = {
        "stage": "axis",
        "seed": seed,
        "cells": cells,
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "peak_private_gb": round(_peak_private_gb(), 3),
        "pid": os.getpid(),
    }
    log.info("axis seed %d done (%.0f s)", seed, out["wall_s"])
    return out


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eval_row_ids(c: dict) -> np.ndarray:
    """Each evaluation row's index into the processed dataset (the splits file's test array at
    the stratified_cap positions)."""
    proc = REPO_ROOT / load_yaml(REPO_ROOT / "configs" / "data.yaml")["paths"]["processed"]
    split = np.load(proc / f"splits_{VARIANT}.npz")["test"]
    return split[c["te_idx"]]


def write_exports(run_path: Path, strength: list[dict], axis: list[dict], row_ids, y) -> dict:
    """The exports rule (decision log 2026-11-17T02:18Z): per_seed.csv (every macro-F1 and query
    count per seed x family x arm x condition), predictions.parquet (per evaluation row: its
    dataset id, true label, the probability of the attack class and the predicted label per
    seed x family x arm x condition) and crafted_inputs.npz (every attacked input)."""
    import pandas as pd

    rows = []
    for u in strength:
        for arm in ("undefended", "defended"):
            for cond in ("clean", *ATTACKS):
                rows.append(
                    {
                        "seed": u["seed"],
                        "family": "vqc",
                        "arm": arm,
                        "condition": cond,
                        "metric": "macro_f1",
                        "value": u[arm][cond],
                    }
                )
            rows.append(
                {
                    "seed": u["seed"],
                    "family": "vqc",
                    "arm": arm,
                    "condition": "square",
                    "metric": "queries_per_row",
                    "value": u[arm]["square_queries"],
                }
            )
    for u in axis:
        for f, arms in u["cells"].items():
            for arm, rec in arms.items():
                for cond in ("clean", "square"):
                    rows.append(
                        {
                            "seed": u["seed"],
                            "family": f,
                            "arm": arm,
                            "condition": cond,
                            "metric": "macro_f1",
                            "value": rec[cond],
                        }
                    )
                rows.append(
                    {
                        "seed": u["seed"],
                        "family": f,
                        "arm": arm,
                        "condition": "square",
                        "metric": "queries_per_row",
                        "value": rec["square_queries"],
                    }
                )
    pd.DataFrame(rows).to_csv(run_path / "per_seed.csv", index=False)
    preds, crafted = [], {}
    for stage, units in (("strength", strength), ("axis", axis)):
        for u in units:
            with np.load(run_path / "unit_arrays" / f"{stage}_s{u['seed']}.npz") as z:
                for key in z.files:
                    fam, arm, cond, kind = key.split("|")
                    if kind == "X":
                        crafted[f"{u['seed']}|{fam}|{arm}|{cond}"] = z[key]
                        continue
                    p = z[key]
                    preds.append(
                        pd.DataFrame(
                            {
                                "seed": u["seed"],
                                "family": fam,
                                "arm": arm,
                                "condition": cond,
                                "row_id": row_ids,
                                "y_true": y,
                                "prob_attack": p[:, 1],
                                "pred": p.argmax(1),
                            }
                        )
                    )
    pd.concat(preds, ignore_index=True).to_parquet(run_path / "predictions.parquet", index=False)
    np.savez_compressed(run_path / "crafted_inputs.npz", **crafted)
    return {
        "per_seed_rows": len(rows),
        "prediction_rows": int(sum(len(d) for d in preds)),
        "crafted_arrays": len(crafted),
    }


def strength_reading(units: list[dict], run_b: dict) -> dict:
    """The control's checks and statistic. Reproduction: each arm's clean and PGD-20 macro-F1
    equal the full run's within 1e-6. No attack scores above clean. Statistic: per arm, the
    strongest attack is the one with the lowest seed-mean attacked macro-F1; HOLDS if defended
    minus undefended under it is >= STEP1_BAR, else COLLAPSES."""
    repro, above = {}, []
    for u in units:
        r = run_b[u["seed"]]
        ref = {
            "undefended": {
                "clean": r["undefended"]["vqc"]["clean"],
                "pgd20": r["undefended"]["vqc"]["whitebox_0.5"],
            },
            "defended": {
                "clean": r["defended"]["vqc"]["clean"],
                "pgd20": r["defended"]["vqc"]["whitebox"]["0.5"],
            },
        }
        for arm in ("undefended", "defended"):
            for k in ("clean", "pgd20"):
                repro[f"{u['seed']}|{arm}|{k}"] = abs(u[arm][k] - ref[arm][k])
            above += [f"{u['seed']}|{arm}|{a}" for a in ATTACKS if u[arm][a] > u[arm]["clean"]]
    mean = {
        arm: {k: float(np.mean([u[arm][k] for u in units])) for k in ("clean", *ATTACKS)}
        for arm in ("undefended", "defended")
    }
    strongest = {arm: min(ATTACKS, key=lambda a, arm=arm: mean[arm][a]) for arm in mean}
    gain = mean["defended"][strongest["defended"]] - mean["undefended"][strongest["undefended"]]
    return {
        "reproduction_max_abs_diff": max(repro.values()),
        "reproduction_pass": bool(max(repro.values()) <= ANCHOR_TOL),
        "reproduction_cells": repro,
        "attacks_scoring_above_clean": above,
        "no_attack_above_clean": not above,
        "seed_mean": mean,
        "strongest_attack": strongest,
        "gain_under_strongest": gain,
        "verdict": "HOLDS" if gain >= STEP1_BAR else "COLLAPSES",
        "bar": STEP1_BAR,
        "defended_attacked_under_strongest": mean["defended"][strongest["defended"]],
        "floor": FLOOR,
        "defended_above_floor": bool(mean["defended"][strongest["defended"]] > FLOOR),
    }


def blackbox_axis(strength: list[dict], axis: list[dict]) -> dict:
    """The step-2 black-box axis from the Square cells, with step 2's ND and small-denominator
    rules (seed means; VQC cells from the strength units)."""
    by = {u["seed"]: u for u in strength}
    rows = {}
    for f in list(CLASSICAL_TARGETS) + ["qsvm", "vqc"]:
        rows[f] = {}
        for arm in ("undefended", "defended"):
            recs = [by[u["seed"]][arm] if f == "vqc" else u["cells"][f][arm] for u in axis]
            rows[f][arm] = {
                "clean": float(np.mean([r["clean"] for r in recs])),
                "square": float(np.mean([r["square"] for r in recs])),
            }
    den = {
        arm: float(
            np.mean([rows[c][arm]["clean"] - rows[c][arm]["square"] for c in CLASSICAL_TARGETS])
        )
        for arm in ("undefended", "defended")
    }
    q_rows = {}
    for q in QUANTUM:
        dmg = {arm: rows[q][arm]["clean"] - rows[q][arm]["square"] for arm in den}
        if min(den.values()) < SMALL_DEN:
            delta = (dmg["defended"] - dmg["undefended"]) - (den["defended"] - den["undefended"])
            q_rows[q] = {
                "rule": "small denominator",
                "delta": delta,
                "damage": dmg,
                "moves": bool(abs(delta) >= SMALL_DEN_BAR),
            }
        else:
            nd = {arm: dmg[arm] / den[arm] for arm in den}
            q_rows[q] = {
                "rule": "ND",
                "nd": nd,
                "delta": nd["defended"] - nd["undefended"],
                "damage": dmg,
                "moves": bool(abs(nd["defended"] - nd["undefended"]) >= PICTURE_BAR),
            }
    return {
        "cells_seed_mean": rows,
        "classical_damage": den,
        "quantum": q_rows,
        "moves": any(r["moves"] for r in q_rows.values()),
    }


def main_strength(args, rss: float, avail: float) -> None:
    run_dir = args.strength_control
    sel, run_b = run_cells(run_dir)
    cfg = {
        "mode": "strength_control",
        "threads": args.threads,
        "workers": args.workers,
        "worker_rss_gb": rss,
        "available_gb_at_launch": avail,
        "vqc_chunk": VQC_CHUNK,
        "eot_block_rows": EOT_BLOCK,
        "seeds": SEEDS,
        "for_run": str(run_dir),
        "selected_sigma": sel,
        "eps": EPS,
        "pgd_restarts": STRENGTH_RESTARTS,
        "pgd_steps": STRENGTH_STEPS,
        "pgd_step_frac": STRENGTH_STEP_FRAC,
        "square_iterations": SQUARE_ITERS,
        "preregistration": "decision log 2026-11-17, R4-14 interim entry (attack-strength control)",
        "qgb_vqc_chunk": os.environ["QGB_VQC_CHUNK"],
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "prediction_draws_defended": ADEQ_DRAWS,
        "eot_draws": K_DRAWS,
        "tuned_files": {
            str(best_params_path(VARIANT, QUBITS).relative_to(REPO_ROOT)): sha256(
                best_params_path(VARIANT, QUBITS)
            ),
            str(TABLE1_PARAMS.relative_to(REPO_ROOT)): sha256(TABLE1_PARAMS),
        },
        "full_run_inputs_sha256": {
            n: sha256(run_dir / n) for n in ("selection.json", "results_partial.json")
        },
        "random_streams": {
            "undefended_vqc_pgd20": 'spawn_rng(seed, "adv") (the Table III replay)',
            "pgd100x5_starts": "nd_str_und_start{r}, nd_str_def_start{r}, r = 0..4",
            "pgd100x5_eot": "nd_str_def_eot{r}",
            "defended_pgd20": f"nd_pgd_test_vqc_{EPS}, nd_eot_test_vqc_{EPS}",
            "prediction_draws": "nd_mc_test64_<family>",
            "square": "nd_sq_<arm>_<family>",
            "training_noise": "nd_train_<sigma>",
        },
        "eval_rows": "stratified_cap(test, 500, seed=0); ids in predictions.parquet row_id",
    }
    with RunDir(f"noise_defense_strength_{VARIANT}_q{QUBITS}", config=cfg, seeds=SEEDS) as run:
        prior = json.loads(args.resume.read_text()) if args.resume else {"strength": [], "axis": []}
        S, A = prior["strength"], prior["axis"]
        array_dir = run.path / "unit_arrays"
        array_dir.mkdir(exist_ok=True)
        if args.resume:  # a resumed run copies the finished units' arrays beside its own
            import shutil

            for f in (args.resume.parent / "unit_arrays").glob("*.npz"):
                shutil.copy2(f, array_dir / f.name)

        def save():
            run.write_json("results_partial.json", {"strength": S, "axis": A})

        have = {u["seed"] for u in S}
        specs = [
            {"seed": s, "sigma_vqc": sel["vqc"], "array_dir": str(array_dir)}
            for s in SEEDS
            if s not in have
        ]
        run_units(unit_strength, specs, args.workers, args.threads, S, save)
        S.sort(key=lambda u: u["seed"])
        reading = strength_reading(S, run_b)
        run.write_json("strength_control.json", reading)
        log.info(
            "strength control: %s",
            json.dumps(
                {
                    k: reading[k]
                    for k in (
                        "verdict",
                        "gain_under_strongest",
                        "strongest_attack",
                        "reproduction_pass",
                        "no_attack_above_clean",
                    )
                }
            ),
        )
        have = {u["seed"] for u in A}
        specs = [
            {"seed": s, "selected": sel, "array_dir": str(array_dir)}
            for s in SEEDS
            if s not in have
        ]
        run_units(unit_axis, specs, args.workers, args.threads, A, save)
        A.sort(key=lambda u: u["seed"])
        axis = blackbox_axis(S, A)
        c = ctx(False)
        exports = write_exports(run.path, S, A, eval_row_ids(c), c["yte"])
        peaks = {
            "peak_rss_gb": max(u["peak_rss_gb"] for u in S + A),
            "peak_private_gb": max(u["peak_private_gb"] for u in S + A),
        }
        run.write_json(
            "results.json",
            {
                "config": cfg,
                "strength_control": reading,
                "step2_blackbox_axis": axis,
                "peaks": peaks,
                "exports": exports,
                "strength_units": S,
                "axis_units": A,
            },
        )
        run.metrics.update(
            {
                "verdict": reading["verdict"],
                "gain_under_strongest": reading["gain_under_strongest"],
                "blackbox_axis_moves": axis["moves"],
                **peaks,
            }
        )
        print(f"-> {run.path}")
        failed = [k for k in ("reproduction_pass", "no_attack_above_clean") if not reading[k]]
        if failed:
            raise AssertionError(f"strength-control checks failed: {failed}")


# --------------------------------------------------------------------------- reading


def anchor(units_b: list[dict]) -> dict:
    """Every family but the VQC against Table III (clean, published attacked at 0.5, sensor noise);
    the VQC against R4-23's depth-6 arm (clean, white-box PGD at 0.5). That arm ran no sensor-noise
    stage, so the VQC's noise cells are not anchored and are listed as such."""
    pub = json.loads((REPO_ROOT / "results" / "runs" / TABLE_III_RUN / "results.json").read_text())
    pub = pub["per_seed"]
    d6 = json.loads(R4_23_DEPTH6.read_text())["per_seed"]
    diffs = {}
    for u in units_b:
        s, a = u["seed"], u["anchor_values"]
        for f in FAMILIES:
            ev = (d6 if f == "vqc" else pub)[s]["evasion"]
            diffs[f"{s}|{f}|clean"] = a["clean"][f] - ev["clean"][f]
            diffs[f"{s}|{f}|pgd_0.5"] = a["published_pgd_0.5"][f] - ev["curves"][f]["pgd"]["0.5"]
            if f == "vqc":
                continue
            for snr, v in a["noise"].items():
                diffs[f"{s}|{f}|noise_{snr}"] = v[f] - pub[s]["sensor_noise"][f][snr]
    worst = max(diffs, key=lambda k: abs(diffs[k]))
    return {
        "max_abs_diff": abs(diffs[worst]),
        "worst": worst,
        "tol": ANCHOR_TOL,
        "n_cells": len(diffs),
        "references": {"vqc": str(R4_23_DEPTH6.relative_to(REPO_ROOT)), "others": TABLE_III_RUN},
        "not_anchored": "vqc sensor noise (the R4-23 depth-6 arm has no sensor-noise stage)",
        "pass": abs(diffs[worst]) <= ANCHOR_TOL,
    }


def merged(units_b: list[dict]) -> dict:
    """Per family, seed-mean defended and undefended numbers on every axis; a family whose
    selected sigma is 0 is its undefended model on every axis."""
    seeds = sorted(u["seed"] for u in units_b)
    by = {u["seed"]: u for u in units_b}

    def mean(fn):
        return float(np.mean([fn(by[s]) for s in seeds]))

    out = {}
    for f in FAMILIES:

        def und(key, f=f):
            return lambda u: u["undefended"][f][key]

        def noise_mean(rec):
            return float(np.mean(list(rec["noise"].values())))

        u_rec = {
            "clean": mean(und("clean")),
            "transfer": mean(und("transfer")),
            "noise_mean": mean(lambda u, f=f: noise_mean(u["undefended"][f])),
            "random_sign_0.5": mean(und("random_sign_0.5")),
        }
        if f in WHITEBOX:
            u_rec["whitebox_0.5"] = mean(und("whitebox_0.5"))
        d0 = by[seeds[0]]["defended"][f]
        if d0.get("undefended"):
            d_rec = dict(u_rec) | {"sigma": 0.0}
            if f in WHITEBOX:
                d_rec["whitebox_single_draw_0.5"] = u_rec["whitebox_0.5"]
        else:

            def dv(fn, f=f):
                return mean(lambda u: fn(u["defended"][f]))

            d_rec = {
                "sigma": d0["sigma"],
                "clean": dv(lambda r: r["clean"]),
                "transfer": dv(lambda r: r["transfer"]),
                "noise_mean": dv(noise_mean),
                "random_sign_0.5": dv(lambda r: r["random_sign_0.5"]),
            }
            if f in WHITEBOX:
                d_rec["whitebox_0.5"] = dv(lambda r: r["whitebox"]["0.5"])
                d_rec["whitebox_curve"] = {
                    e: dv(lambda r, e=e: r["whitebox"][e]) for e in d0["whitebox"]
                }
                d_rec["whitebox_single_draw_0.5"] = dv(lambda r: r["whitebox_single_draw_0.5"])
        out[f] = {"undefended": u_rec, "defended": d_rec}
    return out


def beside_units(units_b: list[dict]) -> list[dict]:
    """Stage-B units with every defended record replaced by its beside (M-draw) values."""
    out = []
    for u in units_b:
        v = dict(u)
        v["defended"] = {
            f: (d if "beside" not in d else d["beside"] | {"sigma": d["sigma"]})
            for f, d in u["defended"].items()
        }
        out.append(v)
    return out


def step1(m: dict) -> dict:
    gains = {
        f: m[f]["defended"]["whitebox_0.5"] - m[f]["undefended"]["whitebox_0.5"] for f in WHITEBOX
    }
    single = {
        f: m[f]["defended"]["whitebox_single_draw_0.5"] - m[f]["undefended"]["whitebox_0.5"]
        for f in WHITEBOX
    }
    if any(g >= STEP1_BAR for g in gains.values()):
        verdict = "REAL"
    elif any(g >= STEP1_BAR for g in single.values()):
        verdict = "OBFUSCATED"
    else:
        verdict = "NULL"
    return {"verdict": verdict, "gain_eot": gains, "gain_single_draw": single, "bar": STEP1_BAR}


def step2(m: dict) -> dict:
    """Computed for the record; read only when step 1 is REAL (black-box axis in its own stage)."""

    def cls_damage(arm, key):
        return float(np.mean([m[c][arm]["clean"] - m[c][arm][key] for c in CLASSICAL_TARGETS]))

    axes = {}
    for axis, qkey, ckey in (
        ("whitebox", "whitebox_0.5", "transfer"),
        ("transfer", "transfer", "transfer"),
        ("sensor_noise", "noise_mean", "noise_mean"),
    ):
        den = {arm: cls_damage(arm, ckey) for arm in ("undefended", "defended")}
        rows = {}
        for q in QUANTUM:
            dmg = {arm: m[q][arm]["clean"] - m[q][arm][qkey] for arm in den}
            if min(den.values()) < SMALL_DEN:
                delta = (dmg["defended"] - dmg["undefended"]) - (
                    den["defended"] - den["undefended"]
                )
                rows[q] = {
                    "rule": "small denominator",
                    "delta": delta,
                    "moves": bool(abs(delta) >= SMALL_DEN_BAR),
                    "damage": dmg,
                }
            else:
                nd = {arm: dmg[arm] / den[arm] for arm in den}
                rows[q] = {
                    "rule": "ND",
                    "nd": nd,
                    "delta": nd["defended"] - nd["undefended"],
                    "moves": bool(abs(nd["defended"] - nd["undefended"]) >= PICTURE_BAR),
                    "damage": dmg,
                }
        axes[axis] = {
            "classical_damage": den,
            "quantum": rows,
            "moves": any(r["moves"] for r in rows.values()),
        }

    def gap(arm):
        return max(m[q][arm]["clean"] for q in QUANTUM) - max(m[c][arm]["clean"] for c in CLASSICAL)

    g_u, g_d = gap("undefended"), gap("defended")
    axes["accuracy"] = {
        "gap_undefended": g_u,
        "gap_defended": g_d,
        "moves": bool(np.sign(g_u) != np.sign(g_d) and abs(g_d) >= ACC_BAR),
    }
    return {"axes": axes, "moves": any(a["moves"] for a in axes.values())}


def controls(units_a, units_b, b0: dict, smoke: bool) -> dict:
    out = {}
    a0 = [u for u in units_a if u["sigma"] == 0.0]
    out["standalone_mlp_equals_zoo"] = all(u["standalone_mlp_equals_zoo"] for u in a0)
    zr = [
        d["zero_radius_equals_clean"]
        for u in units_b
        for d in u["defended"].values()
        if "zero_radius_equals_clean" in d
    ]
    out["zero_radius_equals_clean"] = all(zr) if zr else "no defended white-box family"
    adeq_k, order = {}, {}
    for u in units_b:
        for f, d in u["defended"].items():
            if "whitebox_0.5_k_adequacy" in d:
                adeq_k[f] = abs(d["whitebox"]["0.5"] - d["whitebox_0.5_k_adequacy"])
            if "whitebox_single_draw_0.5" in d:
                order[f"{u['seed']}|{f}"] = d["whitebox"]["0.5"] - d["whitebox_single_draw_0.5"]
    out["prediction_draw_check_abs_diff"] = {f: v["abs_diff"] for f, v in b0["mc_check"].items()}
    out["eot_adequacy_max_abs_diff"] = adeq_k
    out["eot_adequacy"] = all(v <= EOT_TOL for v in adeq_k.values()) if adeq_k else "n/a"
    out["attack_strength_ordering_eot_minus_single"] = order
    out["attack_strength_ordering_reported"] = all(v <= ORDER_TOL for v in order.values())
    return out


# --------------------------------------------------------------------------- parent


def run_units(fn, specs, workers, threads, done: list, save) -> None:
    """Units in a spawn pool; with one worker they run in this process, so the whole job is one
    process and its peak RSS is the unit's (no idle parent holding its own imports)."""
    if workers == 1:
        for sp in specs:
            done.append(fn(sp))
            save()
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(threads,),
    ) as ex:
        futs = {ex.submit(fn, sp): sp for sp in specs}
        for f in as_completed(futs):
            done.append(f.result())  # a failed unit raises here, with its traceback
            save()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker-rss-gb", type=float, help="per-worker peak RSS measured by smoke")
    ap.add_argument("--resume", type=Path, help="results_partial.json of an interrupted run")
    ap.add_argument(
        "--strength-control",
        type=Path,
        help="a finished run directory: the VQC attack-strength control and the black-box axis",
    )
    ap.add_argument(
        "--anchor-only",
        type=Path,
        help="a finished run directory: recheck its 111 anchor cells only (seeds 0-2)",
    )
    args = ap.parse_args()
    _init_worker(args.threads)
    W, T = args.workers, args.threads
    if W * T > psutil.cpu_count(logical=True):
        raise ValueError(f"W x T = {W * T} exceeds the logical CPUs")
    rss = args.worker_rss_gb or (1.2 if args.smoke else None)
    if rss is None:
        raise SystemExit("--worker-rss-gb is required (smoke prints the measured value)")
    avail = psutil.virtual_memory().available / 2**30
    if W * rss > MEM_FRACTION * avail:
        raise MemoryError(f"{W} x {rss} GB > {MEM_FRACTION:.0%} of {avail:.1f} GB free")
    seeds = SMOKE["seeds"] if args.smoke else SEEDS
    sigmas = SMOKE["sigmas"] if args.smoke else SIGMAS
    cfg = {
        "threads": T,
        "workers": W,
        "worker_rss_gb": rss,
        "available_gb_at_launch": avail,
        "smoke": args.smoke,
        "seeds": seeds,
        "sigmas_attack_units": list(sigmas),
        "encoding": ENCODING,
        "vqc_chunk": VQC_CHUNK,
        "eot_block_rows": EOT_BLOCK,
        "m_draws": SMOKE["m"] if args.smoke else M_DRAWS,
        "k_draws": SMOKE["k"] if args.smoke else K_DRAWS,
        "adequacy_draws": SMOKE["adeq"] if args.smoke else ADEQ_DRAWS,
        "eps": EPS,
        "eps_curve": list(EPS_CURVE),
        "clean_cost_cap": COST_CAP,
        "table_iii_run": TABLE_III_RUN,
        "vqc_configuration": {
            "rule": "Table I (R4-23 ruling 2, decision log 2026-11-16T13:33Z)",
            "params": str(TABLE1_PARAMS.relative_to(REPO_ROOT)),
            "anchor": str(R4_23_DEPTH6.relative_to(REPO_ROOT)),
        },
        "preregistration": "decision log 2026-11-16T11:39Z, amendment R4-14",
        "resume": str(args.resume) if args.resume else None,
    }
    if args.anchor_only:
        main_anchor_only(args, rss, avail)
        return
    if args.strength_control:
        main_strength(args, rss, avail)
        return
    name = f"noise_defense_{VARIANT}_q{QUBITS}" + ("_smoke" if args.smoke else "")
    with RunDir(name, config=cfg, seeds=seeds) as run:
        empty = {"A": [], "B0": [], "B": []}
        prior = json.loads(args.resume.read_text()) if args.resume else empty
        A, B0, B = prior["A"], prior["B0"], prior["B"]

        def save():
            run.write_json("results_partial.json", {"A": A, "B0": B0, "B": B})

        have = {(u["seed"], u["sigma"]) for u in A}
        specs = [
            {"seed": s, "sigma": g, "smoke": args.smoke}
            for s in seeds
            for g in sigmas
            if (s, g) not in have
        ]
        specs.sort(key=lambda sp: -sp["sigma"])  # the costly refits first
        run_units(unit_a, specs, W, T, A, save)
        sel = select(A, sigmas)
        run.write_json("selection.json", sel)
        chosen = {f: sel[f]["selected_sigma"] for f in FAMILIES}
        log.info("selected sigma: %s", chosen)
        if not B0:
            b0_spec = {"seed": seeds[0], "smoke": args.smoke, "selected": chosen}
            run_units(unit_b0, [b0_spec], 1, T, B0, save)
        draws_rule = prediction_draws(B0[0], cfg["m_draws"], cfg["adequacy_draws"])
        log.info("prediction draws: %s", draws_rule)
        have_b = {u["seed"] for u in B}
        specs = [
            {
                "seed": s,
                "smoke": args.smoke,
                "selected": chosen,
                "adequacy": s == seeds[0],
                "m_primary": draws_rule["primary"],
                "m_beside": draws_rule["beside"],
            }
            for s in seeds
            if s not in have_b
        ]
        run_units(unit_b, specs, W, T, B, save)
        B.sort(key=lambda u: u["seed"])
        anc = (
            {"applied": False, "reason": "smoke rows and steps differ"} if args.smoke else anchor(B)
        )
        m = merged(B)
        s1 = step1(m)
        s2 = step2(m)
        ctl = controls(A, B, B0[0], args.smoke)
        beside = None
        if draws_rule["beside"] is not None:
            mb = merged(beside_units(B))
            beside = {"prediction_draws": draws_rule["beside"], "step1": step1(mb), "merged": mb}
        blocking = {
            "anchor": anc.get("pass", "not applied (smoke)"),
            "standalone_mlp_equals_zoo": ctl["standalone_mlp_equals_zoo"],
            "zero_radius_equals_clean": ctl["zero_radius_equals_clean"],
            "eot_adequacy": ctl["eot_adequacy"],
        }
        results = {
            "config": cfg,
            "blocking_controls": blocking,
            "controls": ctl,
            "anchor": anc,
            "selection": sel,
            "prediction_draws": draws_rule,
            "beside_m_draws": beside,
            "step1": s1,
            "step2": s2
            if s1["verdict"] == "REAL"
            else {"status": "UNTESTED (step 1 not REAL)", "computed_for_record": s2},
            "merged": m,
            "units_meta": [
                {k: u[k] for k in ("stage", "seed", "wall_s", "peak_rss_gb", "pid")}
                | ({"sigma": u["sigma"], "zoo_fit_s": u["zoo_fit_s"]} if u["stage"] == "A" else {})
                for u in A + B0 + B
            ],
        }
        run.write_json("results.json", results)
        run.metrics.update(
            {
                "step1": s1["verdict"],
                "blocking_controls": blocking,
                "selected_sigma": chosen,
                "prediction_draws_primary": draws_rule["primary"],
            }
        )
        log.info("step 1: %s", json.dumps(s1))
        log.info("controls: %s", json.dumps(blocking))
        print(f"-> {run.path}")
        failed = [k for k, v in blocking.items() if v is False]
        if failed:
            raise AssertionError(f"blocking controls failed: {failed}")


if __name__ == "__main__":
    main()
