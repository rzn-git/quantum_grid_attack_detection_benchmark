"""R4-08: per-minibatch Madry adversarial training of the MLP and the VQC (binary, q8, zz).

Trains each differentiable family with every minibatch replaced by its PGD maximiser against the
CURRENT parameters (eps 0.5 sigma in the standardized pre-encoding space, 10 inner steps, the
project's step rule), then attacks each defended model with white-box PGD crafted against itself
and reads E12b's two decision rules. Run (thread budget is W x T; --threads is read before numpy
loads, in every worker):
    python -m experiments.run_madry_per_minibatch --smoke --seeds 0 1 --threads 3
    python -m experiments.run_madry_per_minibatch --workers 2 --threads 3 --worker-rss-gb 1.2
(seeds 0-9 by default, ruling 2 of 2026-11-16T13:05Z; E12b's seeds 0-2 are read separately)
    python -m experiments.run_madry_per_minibatch --resume <run>/results_partial.json ...

Pre-registration: decision log 2026-11-16T11:07Z (round 4, part A, R4-08). E12b
(`run_madry_adversarial_training.py`, finding 2026-08-23T20:31Z) ran round-based min-max: craft a
PGD set against the current model, refit on clean + adversarial, five times. This is the faithful
form: the inner maximisation runs on every minibatch, and the step is taken on the maximiser only.

Training.
  MLP  sklearn `partial_fit` on each maximised minibatch, with the tuned MLP's architecture,
       Adam settings, batch size and epoch cap (`build("mlp", ...)`, max_iter 300). Early stopping
       is the tuned recipe's own rule, replayed outside sklearn because `partial_fit` refuses
       `early_stopping=True`: a stratified 10% of the training rows held out, accuracy on them
       after every epoch, stop when it has not beaten the best by `tol` for more than
       `n_iter_no_change` epochs, restore the best epoch. sklearn creates the parameters on the
       first `partial_fit` call, so the first minibatch of the first epoch is the one step taken
       on clean rows; every later minibatch is replaced by its maximiser. The inner gradient is
       the white-box attacks' MLP path (`GradProvider(mlp, "mlp")`).
  VQC  `VQClassifier.fit` (PennyLane torch interface, backprop) with `batch_inputs` returning
       each minibatch's maximiser in angles; the tuned epochs, batch, patience 8 and early
       stopping on clean validation macro-F1 (the 2,000-row val subset), as FittedZoo trains it.
       The inner gradient is FittedZoo's VQC path (angles chain-ruled to PCA space).
  At eps 0 the maximiser is the batch itself (pgd's start and box collapse to X; asserted in the
  test), so the eps-0 arm skips the inner loop and is the training loop alone (control b).

Hyperparameters: `--best-params`, by default the Table I file (VQC depth 6, ruling 2 of
2026-11-16T13:33Z); the depth-4 file E12b used reproduces E12b's arm. Control (a)'s VQC anchor
follows the file: E12b's per-seed undefended VQC for depth 4, R4-23's depth-6 arm
(`20261116T122057Z_vqc_config_check_binary_q8`) for depth 6; the MLP's is E12b's either way.

Evaluation (E12b's, so the undefended arm is E12b's undefended arm): the FittedZoo of each seed,
the 500-row test subset, the configured PGD (configs/adversarial.yaml, 20 steps) at eps 0.1, 0.3,
0.5, E12b's RNG stream names for the undefended and classical cells. Each defended model is
attacked with examples crafted against itself (control c), through the same FittedZoo paths
(a shallow copy of the zoo with the family's model swapped).

Decision rules (E12b's, unchanged):
  (1) the defense is real if, under white-box PGD at 0.5 sigma, defended minus undefended
      macro-F1 >= 0.05 for either family;
  (2) only if (1) holds: the cross-family picture moves if the VQC's damage normalised by the
      mean classical (undefended, MLP-surrogate transfer) damage shifts by >= 0.10.
Controls: (a) the undefended arm reproduces E12b's undefended figures (and the published
figures E12b gated on) within 0.03; (b) eps-0 training reproduces standard training's clean
macro-F1 within 0.02; (c) self-crafted attacks; zero-radius no-op. Gradient-masking checks
(reported): a random +-eps-sign null at 0.5 sigma, and attacked F1 at 10, 20 and 50 PGD steps.
A unit is one seed (its FittedZoo fits once and serves both families); every RNG stream is keyed
by (seed, family, purpose), so no number depends on scheduling.
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
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_var] = _THREADS or "1"

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.utils import shuffle as sk_shuffle  # noqa: E402

from experiments.run_adversarial import FittedZoo, _f1  # noqa: E402
from experiments.run_consistent_relay_attack import (  # noqa: E402
    THREAD_VARS,
    _peak_rss_gb,
    memory_guard,
)
from experiments.run_madry_adversarial_training import (  # noqa: E402
    CLASSICAL_TARGETS,
    CONTROL_TOL,
    DEFENSE_BAR,
    PICTURE_BAR,
    PUBLISHED,
)
from qgridbench.attacks.evasion import pgd  # noqa: E402
from qgridbench.attacks.gradients import GradProvider  # noqa: E402
from qgridbench.models.classical.zoo import build, stratified_cap  # noqa: E402
from qgridbench.models.quantum.vqc import VQClassifier  # noqa: E402
from qgridbench.protocol import best_params_path, prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import set_all_seeds, spawn_rng  # noqa: E402

log = get_logger(__name__)

VARIANT, QUBITS, ENCODING = "binary", 8, "zz"
E12B_RUN = REPO_ROOT / "results" / "runs" / "20260823T201500Z_madry_adversarial_training_binary_q8"
E12B_SEEDS = [0, 1, 2]
# The Table I VQC (depth 6) is the reported VQC from 2026-11-16T13:33Z (R4-23 MATERIAL, ruling 2);
# its tuned-parameter file differs from best_params_path's only in vqc_depth and vqc_lr.
TABLE1_PARAMS = REPO_ROOT / "results" / f"adv_best_params_{VARIANT}_q{QUBITS}_table1.json"
R4_23_RUN = REPO_ROOT / "results" / "runs" / "20261116T122057Z_vqc_config_check_binary_q8"
TRAIN_EPS, INNER_STEPS = 0.5, 10
EVAL_EPS = [0.1, 0.3, 0.5]
CHECK_STEPS = [10, 50]  # beside the configured steps, for the monotonicity check
EPS0_TOL = 0.02
VAL_CAP = 2000  # FittedZoo's VQC early-stopping subset
FAMILIES = ("mlp", "vqc")


@dataclass
class Shared:
    threads: int
    best: dict
    subset_cap: int
    eval_n: int
    val_cap: int
    mlp_max_epochs: int | None  # None: the tuned recipe's max_iter
    train_eps: float
    inner_steps: int
    eval_eps: list
    pgd_steps: int
    step_frac: float


_SHARED: Shared | None = None
_REG = None


def _init_worker(sh: Shared):
    global _SHARED
    want = str(sh.threads)
    got = {v: os.environ.get(v) for v in THREAD_VARS}
    if any(x != want for x in got.values()) or torch.get_num_threads() != sh.threads:
        raise AssertionError(
            f"worker {os.getpid()}: thread caps {got}, torch {torch.get_num_threads()}, "
            f"expected {want}"
        )
    _SHARED = sh


# ------------------------------------------------------------------------------ training


def train_mlp_madry(params, seed, X, y, eps, scale, inner_steps, step_frac, max_epochs=None):
    """Per-minibatch adversarial training of the tuned sklearn MLP; returns (mlp, trace).

    Every random draw follows `MLPClassifier.fit`'s order on one `RandomState(seed)` (pre-run
    amendment, decision log 2026-11-16T18:55Z): the weight initialisation (made by the first
    `partial_fit` from its own `RandomState(seed)`, so the shared stream is advanced past the same
    number of draws), then the stratified hold-out split, then one shuffle per epoch. With eps 0
    the loop therefore is sklearn's fit (asserted in the test); only the inner maximiser differs.
    """
    mlp = build("mlp", params, seed)
    patience, tol, frac = mlp.n_iter_no_change, mlp.tol, mlp.validation_fraction
    max_epochs = mlp.max_iter if max_epochs is None else max_epochs
    if not mlp.early_stopping:
        raise AssertionError("the tuned MLP recipe early-stops; this replay assumes it")
    mlp.set_params(early_stopping=False)
    classes = np.unique(y)
    units = [X.shape[1], *mlp.hidden_layer_sizes, 1 if len(classes) == 2 else len(classes)]
    rs = np.random.RandomState(seed)
    rs.uniform(size=sum(a * b + b for a, b in zip(units[:-1], units[1:])))  # _init_coef's draws
    Xf, Xes, yf, yes = train_test_split(X, y, random_state=rs, test_size=frac, stratify=y)
    bs = mlp.batch_size
    order = np.arange(len(yf))
    inner_rng = spawn_rng(seed, f"r408_mlp_inner_{eps}")
    best_score, stale, best, scores, epoch = -np.inf, 0, None, [], 0
    for epoch in range(1, max_epochs + 1):
        order = sk_shuffle(order, random_state=rs)
        for s0 in range(0, len(order), bs):
            b = order[s0 : s0 + bs]
            xb = Xf[b]
            if eps > 0 and hasattr(mlp, "coefs_"):
                xb = pgd(
                    GradProvider(mlp, "mlp"),
                    xb,
                    yf[b],
                    eps,
                    scale,
                    n_steps=inner_steps,
                    step_frac=step_frac,
                    rng=inner_rng,
                )
            mlp.partial_fit(xb, yf[b], classes=classes)
        score = float(accuracy_score(yes, mlp.predict(Xes)))
        scores.append(score)
        stale = stale + 1 if score < best_score + tol else 0
        if score > best_score:
            best_score = score
            best = ([c.copy() for c in mlp.coefs_], [i.copy() for i in mlp.intercepts_])
        if stale > patience:
            break
    mlp.coefs_, mlp.intercepts_ = best
    trace = {
        "epochs": epoch,
        "best_epoch": int(np.argmax(scores)) + 1,
        "best_val_accuracy": best_score,
        "val_accuracy": scores,
        "clean_steps": 1 if eps > 0 else None,
    }
    return mlp, trace


def train_vqc_madry(zoo, best, seed, X, y, Xva, yva, eps, scale, inner_steps, step_frac):
    """Per-minibatch adversarial training of the tuned VQC through VQClassifier.fit's hook."""
    set_all_seeds(seed)
    vqc = VQClassifier(
        zoo.n_qubits,
        depth=best["vqc_depth"],
        n_classes=zoo.n_classes,
        lr=best["vqc_lr"],
        batch_size=best["vqc_batch"],
        max_epochs=best["vqc_epochs"],
        patience=8,
        seed=seed,
    )
    gp = with_models(zoo, vqc=vqc).grad_provider("vqc")
    inner_rng = spawn_rng(seed, f"r408_vqc_inner_{eps}")

    def maximiser(b):
        if eps == 0:
            return zoo._to_angles(X[b])
        adv = pgd(
            gp, X[b], y[b], eps, scale, n_steps=inner_steps, step_frac=step_frac, rng=inner_rng
        )
        return zoo._to_angles(adv)

    vqc.fit(zoo._to_angles(X), y, zoo._to_angles(Xva), yva, batch_inputs=maximiser)
    trace = {
        "epochs": len(vqc.history["epoch_loss"]),
        "best_epoch": int(vqc.best_epoch_) + 1,
        "best_val_macro_f1": float(vqc.best_val_f1_),
        "val_macro_f1": vqc.history["val_macro_f1"],
        "grad_norm_var": vqc.history["grad_norm_var"],
    }
    return vqc, trace


def with_models(zoo, **models):
    """A shallow copy of `zoo` with some families' models swapped, so predictions, clean guards
    and input gradients of a defended model run through FittedZoo's own paths."""
    z = copy.copy(zoo)
    z.models = {**zoo.models, **models}
    return z


# ------------------------------------------------------------------------------ one unit


def _attack(z, fam, X, y, eps, scale, n_steps, step_frac, stream, seed):
    a = pgd(
        z.grad_provider(fam),
        X,
        y,
        eps,
        scale,
        n_steps=n_steps,
        step_frac=step_frac,
        rng=spawn_rng(seed, stream),
    )
    return _f1(y, z.predict_proba(fam, a))


def clean_record(z, fam, X, y, guard: bool) -> dict:
    """Clean macro-F1 with the predicted-class counts; `guard` routes through the runner guard."""
    proba = z.clean_proba(fam, X, y) if guard else z.predict_proba(fam, X)
    pred = proba.argmax(1)
    return {
        "clean": _f1(y, proba),
        "clean_pred_counts": {int(k): int(n) for k, n in zip(*np.unique(pred, return_counts=True))},
        "constant_on_clean": bool(np.unique(pred).size == 1),
    }


def random_sign_f1(z, fam, X, y, eps, scale, stream, seed):
    """The same-radius null: every feature moved to a random corner of the eps box."""
    sign = spawn_rng(seed, stream).choice([-1.0, 1.0], size=X.shape)
    return _f1(y, z.predict_proba(fam, X + eps * scale * sign))


def run_unit(spec: dict) -> dict:
    global _REG
    sh, s = _SHARED, spec["seed"]
    t_unit, timing = time.perf_counter(), {}
    if _REG is None:
        _REG = prepare_regime(VARIANT, "pca", n_components=QUBITS, seed=0)
    reg = _REG
    scale = reg.feature_scale
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    ev = stratified_cap(reg.X["test"], reg.y["test"], sh.eval_n, seed=0)
    Xte, yte = reg.X["test"][ev], reg.y["test"][ev]
    va = stratified_cap(reg.X["val"], reg.y["val"], sh.val_cap, seed=0)
    Xva, yva = reg.X["val"][va], reg.y["val"][va]

    t = time.perf_counter()
    zoo = FittedZoo(
        reg, ENCODING, QUBITS, s, min(sh.subset_cap, cfg_q["qkernel"]["subset_cap"]), sh.best
    )
    timing["fitted_zoo"] = time.perf_counter() - t
    Xtr, ytr = zoo.Xtr, zoo.ytr
    kw = dict(inner_steps=sh.inner_steps, step_frac=sh.step_frac)

    trained, traces = {}, {}
    for arm, eps in (("defended", sh.train_eps), ("eps0", 0.0)):
        t = time.perf_counter()
        set_all_seeds(s)
        m, traces[f"mlp_{arm}"] = train_mlp_madry(
            sh.best["mlp"], s, Xtr, ytr, eps, scale, max_epochs=sh.mlp_max_epochs, **kw
        )
        timing[f"mlp_{arm}_train"] = time.perf_counter() - t
        t = time.perf_counter()
        v, traces[f"vqc_{arm}"] = train_vqc_madry(
            zoo, sh.best, s, Xtr, ytr, Xva, yva, eps, scale, **kw
        )
        timing[f"vqc_{arm}_train"] = time.perf_counter() - t
        trained[arm] = with_models(zoo, mlp=m, vqc=v)
        log.info("seed %d %s trained | %s", s, arm, {k: round(x, 1) for k, x in timing.items()})

    t = time.perf_counter()
    dz, ez = trained["defended"], trained["eps0"]
    ev_kw = dict(n_steps=sh.pgd_steps, step_frac=sh.step_frac, seed=s)
    cell = {
        "seed": s,
        "train": traces,
        "undefended": {},
        "defended": {},
        "eps0": {},
        "classical": {},
    }
    for fam in FAMILIES:
        # A defended model that predicts one class on clean rows is a known end state of
        # adversarial training (the trivially robust constant classifier), not a void run: it is
        # recorded, not raised. So is the tuned MLP's own fit (the undefended arm, which the eps-0
        # arm reproduces): the unweighted recipe can restore an all-positive epoch, and such a
        # seed counts as collapsed in both arms (amendment 2026-11-16T18:55Z). The VQC's
        # undefended and eps-0 arms keep the clean guard.
        cell["defended"][fam] = clean_record(dz, fam, Xte, yte, guard=False)
        cell["undefended"][fam] = clean_record(zoo, fam, Xte, yte, guard=fam != "mlp")
        cell["eps0"][fam] = clean_record(ez, fam, Xte, yte, guard=fam != "mlp")
        cell["defended"][fam]["zero_radius"] = _attack(
            dz, fam, Xte, yte, 0.0, scale, stream=f"zero_{fam}", **ev_kw
        )
    for e in sh.eval_eps:
        ek = str(e)
        for fam in FAMILIES:
            cell["undefended"][fam][ek] = _attack(
                zoo, fam, Xte, yte, e, scale, stream=f"und_{fam}_{e}", **ev_kw
            )
            cell["defended"][fam][ek] = _attack(
                dz, fam, Xte, yte, e, scale, stream=f"def_{fam}_{e}", **ev_kw
            )
        for c in CLASSICAL_TARGETS:
            sur = pgd(
                zoo.grad_provider("mlp"),
                Xte,
                yte,
                e,
                scale,
                n_steps=sh.pgd_steps,
                step_frac=sh.step_frac,
                rng=spawn_rng(s, f"sur_{c}_{e}"),
            )
            cell["classical"].setdefault(c, {})["clean"] = _f1(yte, zoo.clean_proba(c, Xte, yte))
            cell["classical"][c][ek] = _f1(yte, zoo.predict_proba(c, sur))
    # gradient-masking checks at the decision radius
    e = max(sh.eval_eps)
    for fam in FAMILIES:
        for arm, z in (("undefended", zoo), ("defended", dz)):
            cell[arm][fam]["random_sign"] = random_sign_f1(
                z, fam, Xte, yte, e, scale, f"rand_{fam}_{e}", s
            )
        cell["defended"][fam]["by_steps"] = {
            str(n): _attack(
                dz,
                fam,
                Xte,
                yte,
                e,
                scale,
                n_steps=n,
                step_frac=sh.step_frac,
                seed=s,
                stream=f"def_{fam}_{e}_steps{n}",
            )
            for n in CHECK_STEPS
        }
        cell["defended"][fam]["by_steps"][str(sh.pgd_steps)] = cell["defended"][fam][str(e)]
    timing["evaluate"] = time.perf_counter() - t
    cell["timing_s"] = timing
    cell["wall_s"] = time.perf_counter() - t_unit
    cell["peak_rss_gb"] = _peak_rss_gb()
    cell["pid"] = os.getpid()
    log.info("seed %d done in %.0f s, peak RSS %.2f GB", s, cell["wall_s"], cell["peak_rss_gb"])
    return cell


def run_pool(specs, sh: Shared, workers: int, on_done) -> None:
    """Every finished unit goes to on_done (which writes it); the first failure is raised after
    the pool drains, so one failed unit does not discard the others. One worker runs the units
    in this process, so a run's memory is one process rather than a parent plus a worker."""
    if workers == 1:
        _init_worker(sh)
        for sp in specs:
            on_done(run_unit(sp))
        return
    failed = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(sh,),
    ) as ex:
        futs = {ex.submit(run_unit, sp): sp for sp in specs}
        for f in as_completed(futs):
            exc = f.exception()
            if exc is not None:
                log.error("unit %s failed", futs[f], exc_info=exc)
                failed.append((futs[f], exc))
                continue
            on_done(f.result())
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(specs)} units failed (finished units are in "
            f"results_partial.json): {[sp for sp, _ in failed]}"
        ) from failed[0][1]


# ------------------------------------------------------------------------------ reading


def _ms(xs) -> dict:
    xs = [float(x) for x in xs]
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "per_seed": xs}


def read_rules(cells: list[dict], E: str) -> dict:
    """E12b's statistics and rules over `cells` (seed means; std across seeds), with the
    collapse reading of ruling 2 (decision log 2026-11-16T13:05Z): a defended model that predicts
    one class on clean rows is COLLAPSED for that seed; rule (1)'s gain counts only that family's
    non-collapsed seeds, and a family collapsed on a majority of seeds reads "COLLAPSED, not a
    defense" and cannot make rule (1) fire. A constant predictor scores about the stratified
    floor clean and attacked alike, so without this reading it would pass the bar undefended."""
    fam_out = {}
    for fam in FAMILIES:
        col_def = [c["seed"] for c in cells if c["defended"][fam]["constant_on_clean"]]
        col_std = [c["seed"] for c in cells if c["undefended"][fam]["constant_on_clean"]]
        collapsed = sorted(set(col_def) | set(col_std))
        kept = [c for c in cells if c["seed"] not in collapsed]
        g = {
            "clean_undefended": _ms([c["undefended"][fam]["clean"] for c in cells]),
            "clean_defended": _ms([c["defended"][fam]["clean"] for c in cells]),
            "attacked_undefended": _ms([c["undefended"][fam][E] for c in cells]),
            "attacked_defended": _ms([c["defended"][fam][E] for c in cells]),
            "robustness_gain": _ms(
                [c["defended"][fam][E] - c["undefended"][fam][E] for c in cells]
            ),
        }
        g["clean_cost"] = g["clean_undefended"]["mean"] - g["clean_defended"]["mean"]
        g["collapsed_seeds"] = collapsed
        g["collapsed_defended_seeds"] = col_def
        g["collapsed_standard_fit_seeds"] = col_std
        g["collapsed_majority"] = bool(len(collapsed) > len(cells) / 2)
        g["robustness_gain_non_collapsed"] = (
            _ms([c["defended"][fam][E] - c["undefended"][fam][E] for c in kept]) if kept else None
        )
        if g["collapsed_majority"]:
            g["verdict"] = "COLLAPSED, not a defense"
        elif g["robustness_gain_non_collapsed"]["mean"] >= DEFENSE_BAR:
            g["verdict"] = "defense (non-collapsed gain >= the bar)"
        else:
            g["verdict"] = "no defense (non-collapsed gain < the bar)"
        fam_out[fam] = g
    cls_damage = float(
        np.mean(
            [
                c["classical"][k]["clean"] - c["classical"][k][E]
                for c in cells
                for k in CLASSICAL_TARGETS
            ]
        )
    )
    v = fam_out["vqc"]
    nd_und = (v["clean_undefended"]["mean"] - v["attacked_undefended"]["mean"]) / cls_damage
    nd_def = (v["clean_defended"]["mean"] - v["attacked_defended"]["mean"]) / cls_damage
    real = any(fam_out[f]["verdict"].startswith("defense") for f in FAMILIES)
    collapsed_fams = [f for f in FAMILIES if fam_out[f]["collapsed_majority"]]
    if real:
        branch = "per_minibatch_defense_is_real"
    elif collapsed_fams:
        branch = f"per_minibatch_defense_is_a_NULL ({', '.join(collapsed_fams)} COLLAPSED)"
    else:
        branch = "per_minibatch_defense_is_a_NULL"
    return {
        "seeds": [c["seed"] for c in cells],
        "families": fam_out,
        "mean_classical_damage": cls_damage,
        "normalized_vqc_damage_undefended": nd_und,
        "normalized_vqc_damage_defended": nd_def,
        "normalized_vqc_damage_shift": nd_def - nd_und,
        "q1_defense_is_real": real,
        "q1_rule": "a family counts only if it is not collapsed on a majority of seeds, and its "
        "gain is the mean over its non-collapsed seeds (ruling 2, 2026-11-16T13:05Z)",
        "family_verdicts": {f: fam_out[f]["verdict"] for f in FAMILIES},
        "q2_picture_changed": bool(abs(nd_def - nd_und) >= PICTURE_BAR) if real else None,
        "q2_status": "read" if real else "UNTESTED - the defense itself is a null",
        "branch": branch,
    }


def vqc_anchor(best: dict) -> str:
    """Which archived undefended VQC control (a) checks against, from the VQC configuration run:
    E12b's for the depth-4 file, R4-23's depth-6 arm for the Table I file; anything else fails."""
    cfg = (best["vqc_depth"], best["vqc_lr"])
    for name, path in (("e12b", best_params_path(VARIANT, QUBITS)), ("r4_23", TABLE1_PARAMS)):
        ref = json.loads(path.read_text())
        if cfg == (ref["vqc_depth"], ref["vqc_lr"]):
            return name
    raise ValueError(f"no archived undefended VQC for depth {cfg[0]}, lr {cfg[1]}")


def undefended_references(e12b: dict, r423: dict | None, anchor: str, E: str) -> dict:
    """Per family, {seed: {"clean", E}} of the archived undefended arm control (a) compares
    against: the MLP always E12b's; the VQC E12b's (depth 4) or R4-23's depth-6 Table III arm."""
    e12b_ref = {
        c["seed"]: {f: {k: c["undefended"][f][k] for k in ("clean", E)} for f in FAMILIES}
        for c in e12b["per_seed"]
    }
    refs = {"mlp": {s: v["mlp"] for s, v in e12b_ref.items()}}
    if anchor == "e12b":
        refs["vqc"] = {s: v["vqc"] for s, v in e12b_ref.items()}
    else:
        refs["vqc"] = {
            u["seed"]: {
                "clean": u["evasion"]["clean"]["vqc"],
                E: u["evasion"]["curves"]["vqc"]["pgd"][E],
            }
            for u in r423["units"]
            if u["arm"] == "depth6"
        }
    return refs


def read_controls(cells: list[dict], refs: dict, anchor: str, E: str) -> dict:
    by_seed = {c["seed"]: c for c in cells}
    sub = [by_seed[s] for s in E12B_SEEDS if s in by_seed]
    out: dict = {}
    # (a) the undefended arm against the archived undefended arm on the same seeds: the MLP
    # against E12b (and the published MLP figures E12b gated on), the VQC against `anchor`
    a_rows = {}
    for fam in FAMILIES:
        for key in ("clean", E):
            ours = [c["undefended"][fam][key] for c in sub]
            ref = [refs[fam][c["seed"]][key] for c in sub]
            a_rows[f"{fam}_{key}"] = {
                "reference": "e12b" if fam == "mlp" else anchor,
                "measured_mean": float(np.mean(ours)) if ours else None,
                "reference_mean": float(np.mean(ref)) if ref else None,
                "max_abs_diff_per_seed": float(np.max(np.abs(np.subtract(ours, ref))))
                if ours
                else None,
            }
    pub = {"mlp_clean": "mlp_clean", "mlp_pgd_0.5": f"mlp_{E}"}
    if anchor == "e12b":  # the published VQC figure is the depth-4 model's
        pub["vqc_clean"] = "vqc_clean"
    out["a_undefended"] = {
        "seeds": [c["seed"] for c in sub],
        "vqc_anchor": anchor,
        "vs_archived": a_rows,
        "vs_published": {
            k: {"measured": a_rows[v]["measured_mean"], "published": PUBLISHED[k]}
            for k, v in pub.items()
        },
        "tol": CONTROL_TOL,
    }
    out["a_pass"] = bool(
        sub
        and all(
            abs(r["measured_mean"] - r["reference_mean"]) <= CONTROL_TOL for r in a_rows.values()
        )
        and all(
            abs(r["measured"] - r["published"]) <= CONTROL_TOL
            for r in out["a_undefended"]["vs_published"].values()
        )
    )
    # (b) eps-0 training against standard training, clean macro-F1
    # read over seeds where neither arm collapsed; the collapse counts are reported beside it
    b = {}
    for fam in FAMILIES:
        col_e0 = [c["seed"] for c in cells if c["eps0"][fam]["constant_on_clean"]]
        col_st = [c["seed"] for c in cells if c["undefended"][fam]["constant_on_clean"]]
        kept = [c for c in cells if c["seed"] not in set(col_e0) | set(col_st)]
        e0 = [c["eps0"][fam]["clean"] for c in kept]
        st = [c["undefended"][fam]["clean"] for c in kept]
        b[fam] = {
            "seeds_read": [c["seed"] for c in kept],
            "collapsed_eps0_seeds": col_e0,
            "collapsed_standard_fit_seeds": col_st,
            "eps0_clean": _ms(e0) if kept else None,
            "standard_clean": _ms(st) if kept else None,
            "mean_diff": float(np.mean(e0) - np.mean(st)) if kept else None,
        }
    out["b_eps0"] = {**b, "tol": EPS0_TOL}
    out["b_pass"] = bool(
        all(b[f]["mean_diff"] is not None and abs(b[f]["mean_diff"]) <= EPS0_TOL for f in FAMILIES)
    )
    out["c_self_crafted"] = "every defended model is attacked with PGD through its own gradient"
    out["zero_radius_no_op"] = {
        f: bool(all(c["defended"][f]["zero_radius"] == c["defended"][f]["clean"] for c in cells))
        for f in FAMILIES
    }
    out["defended_constant_on_clean_seeds"] = {
        f: [c["seed"] for c in cells if c["defended"][f]["constant_on_clean"]] for f in FAMILIES
    }
    # gradient masking (reported)
    mask = {}
    for fam in FAMILIES:
        steps = sorted(cells[0]["defended"][fam]["by_steps"], key=int)
        by = {n: _ms([c["defended"][fam]["by_steps"][n] for c in cells]) for n in steps}
        rnd_def = _ms([c["defended"][fam]["random_sign"] for c in cells])
        rnd_und = _ms([c["undefended"][fam]["random_sign"] for c in cells])
        lo, hi = str(min(CHECK_STEPS)), str(max(CHECK_STEPS))
        mask[fam] = {
            "defended_by_steps": by,
            "steps_monotone_non_increasing": bool(
                all(by[a]["mean"] >= by[b]["mean"] for a, b in zip(steps, steps[1:]))
            ),
            f"steps{hi}_le_steps{lo}": bool(by[hi]["mean"] <= by[lo]["mean"]),
            "random_sign_defended": rnd_def,
            "random_sign_undefended": rnd_und,
            "pgd_defended": _ms([c["defended"][fam][E] for c in cells]),
            "random_does_less_damage_than_pgd": bool(
                rnd_def["mean"] > np.mean([c["defended"][fam][E] for c in cells])
            ),
        }
    out["gradient_masking"] = mask
    return out


# ------------------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker-rss-gb", type=float, help="measured peak RSS of one unit (smoke)")
    ap.add_argument(
        "--max-rss-gb",
        type=float,
        help="stop after the unit whose measured peak RSS passes this (it is saved; --resume)",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--best-params", type=Path, default=TABLE1_PARAMS)
    ap.add_argument("--resume", type=Path, help="results_partial.json of an interrupted run")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="small data: 600-row train subset, 120 test rows, 200 val rows, 2 VQC epochs",
    )
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    best = json.loads(args.best_params.read_text())
    anchor = vqc_anchor(best)
    if args.smoke:
        best = {**best, "vqc_epochs": 2}
    sh = Shared(
        threads=args.threads,
        best=best,
        subset_cap=600 if args.smoke else 10**9,  # at 200 rows the eps-0 MLP can be constant
        eval_n=120 if args.smoke else cfg_adv["eval_subset_adv"],
        val_cap=200 if args.smoke else VAL_CAP,
        mlp_max_epochs=None,  # the tuned cap in smoke too: short caps leave the MLP constant
        train_eps=TRAIN_EPS,
        inner_steps=INNER_STEPS,
        eval_eps=EVAL_EPS,
        pgd_steps=cfg_adv["evasion"]["pgd"]["n_steps"],
        step_frac=cfg_adv["evasion"]["pgd"]["step_frac"],
    )
    cfg = {
        "item": "R4-08 per-minibatch Madry adversarial training (decision log 2026-11-16T11:07Z)",
        "variant": VARIANT,
        "qubits": QUBITS,
        "encoding": ENCODING,
        "seeds": args.seeds,
        "best_params": str(args.best_params),
        "best_params_content": best,
        "smoke": args.smoke,
        "workers": args.workers,
        "threads": args.threads,
        "qgb_vqc_chunk": os.environ.get("QGB_VQC_CHUNK"),  # resolve_chunk override (None: sized)
        "train_eps": sh.train_eps,
        "inner_steps": sh.inner_steps,
        "eval_eps": sh.eval_eps,
        "eval_pgd_steps": sh.pgd_steps,
        "step_frac": sh.step_frac,
        "check_steps": CHECK_STEPS,
        "eval_n": sh.eval_n,
        "val_cap": sh.val_cap,
        "subset_cap": sh.subset_cap,
        "mlp_max_epochs": sh.mlp_max_epochs,
        "e12b_run": str(E12B_RUN.relative_to(REPO_ROOT)),
        "vqc_anchor": anchor,
        "r4_23_run": str(R4_23_RUN.relative_to(REPO_ROOT)) if anchor == "r4_23" else None,
        "defense_bar": DEFENSE_BAR,
        "picture_bar": PICTURE_BAR,
        "control_tol": CONTROL_TOL,
        "eps0_tol": EPS0_TOL,
        "resume": str(args.resume) if args.resume else None,
    }
    name = f"madry_per_minibatch_{VARIANT}_q{QUBITS}" + ("_smoke" if args.smoke else "")
    with RunDir(name, config=cfg, seeds=args.seeds) as run:
        done: list[dict] = []
        if args.resume:
            prev = json.loads(args.resume.read_text())
            keep = (
                "best_params_content",
                "train_eps",
                "inner_steps",
                "eval_eps",
                "eval_n",
                "smoke",
            )
            if any(prev["config"][k] != cfg[k] for k in keep):
                raise AssertionError(f"resume: {args.resume} was run under another configuration")
            done = prev["units"]
            log.info("resuming: seeds already done %s", [u["seed"] for u in done])
        have = {u["seed"] for u in done}
        todo = [{"seed": s} for s in args.seeds if s not in have]
        if args.worker_rss_gb is not None:
            run.metrics.update(memory_guard(args.workers, args.worker_rss_gb, "full run"))
        elif not args.smoke:
            raise SystemExit("--worker-rss-gb is required (the smoke prints the measured value)")

        def save(cell):
            done.append(cell)
            run.write_json("results_partial.json", {"config": cfg, "units": done})
            if args.max_rss_gb is not None and cell["peak_rss_gb"] > args.max_rss_gb:
                raise MemoryError(
                    f"seed {cell['seed']} peaked at {cell['peak_rss_gb']:.2f} GB > the "
                    f"{args.max_rss_gb} GB budget; finished units are in results_partial.json"
                )

        run_pool(todo, sh, args.workers, on_done=save)
        cells = sorted(done, key=lambda c: c["seed"])
        E = str(max(sh.eval_eps))
        e12b = json.loads((E12B_RUN / "results.json").read_text())
        r423 = json.loads((R4_23_RUN / "results.json").read_text()) if anchor == "r4_23" else None
        refs = undefended_references(e12b, r423, anchor, E)
        sub = [c for c in cells if c["seed"] in E12B_SEEDS]
        results = {
            "config": cfg,
            "rules_all_seeds": read_rules(cells, E),
            "rules_e12b_seeds": read_rules(sub, E) if sub else None,
            "e12b_reference": {
                "summary": e12b["summary"],
                "mean_classical_damage": e12b["mean_classical_damage"],
                "normalized_vqc_damage_undefended": e12b["normalized_vqc_damage_undefended"],
                "normalized_vqc_damage_defended": e12b["normalized_vqc_damage_defended"],
                "vqc_note": "E12b's VQC is the depth-4 configuration; at Table I's depth 6 the "
                "undefended VQC anchor is R4-23's depth-6 arm (ruling 2026-11-16T13:33Z)",
            },
            "controls": read_controls(cells, refs, anchor, E),
            "units_meta": [
                {k: c[k] for k in ("seed", "timing_s", "wall_s", "peak_rss_gb", "pid")}
                for c in cells
            ],
            "per_seed": cells,
        }
        run.write_json("results.json", results)
        ra, ctl = results["rules_all_seeds"], results["controls"]
        run.metrics.update(
            {
                "mlp_robustness_gain": ra["families"]["mlp"]["robustness_gain"]["mean"],
                "vqc_robustness_gain": ra["families"]["vqc"]["robustness_gain"]["mean"],
                "branch": ra["branch"],
                "family_verdicts": ra["family_verdicts"],
                "collapsed_seeds": {f: ra["families"][f]["collapsed_seeds"] for f in FAMILIES},
                "control_a_pass": ctl["a_pass"],
                "control_b_pass": ctl["b_pass"],
                "peak_rss_gb_max": max(c["peak_rss_gb"] for c in cells),
            }
        )
        log.info(
            "BRANCH %s | %s | gain mlp %+.4f vqc %+.4f | nd %.3f -> %.3f | a %s b %s | "
            "collapsed %s",
            ra["branch"],
            ra["family_verdicts"],
            ra["families"]["mlp"]["robustness_gain"]["mean"],
            ra["families"]["vqc"]["robustness_gain"]["mean"],
            ra["normalized_vqc_damage_undefended"],
            ra["normalized_vqc_damage_defended"],
            ctl["a_pass"],
            ctl["b_pass"],
            ctl["defended_constant_on_clean_seeds"],
        )
        print(f"-> {run.path}", flush=True)
        if not args.smoke:
            bad = [k for k in ("a_pass", "b_pass") if not ctl[k]]
            bad += [f"zero_radius_{f}" for f, ok in ctl["zero_radius_no_op"].items() if not ok]
            if bad:
                raise AssertionError(f"controls failed (results written): {bad}")


if __name__ == "__main__":
    main()
