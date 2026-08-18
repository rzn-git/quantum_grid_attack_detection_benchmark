"""Phase 4: adversarial robustness — evasion, transfer matrix, black-box, noise, poisoning.

All attacks in the standardized PRE-ENCODING (PCA) feature space (comparability
contract). Differentiable models (MLP, VQC) are attacked white-box; trees/kernel
SVMs are attacked by transfer from the MLP surrogate and by black-box HopSkipJump.

Deliverables (study protocol §7):
  - evasion degradation curves (macro-F1 vs epsilon) per model family
  - source x target transferability matrix
  - black-box (HopSkipJump) robustness, identical per family, capped query budget
  - random-perturbation control for that black-box table (REQUIRED alongside it:
    HopSkipJump is a minimum-distortion attack and over-states fidelity-kernel
    robustness ~3-8x, because the kernel's decision regions interleave at a scale
    its line search cannot navigate -- see the 2026-08-14T01:10Z decision-log entry)
  - benign Gaussian sensor-noise ablation at 3 SNR levels
  - label-noise poisoning at {5,10,20}%

This script consumes fitted models produced by a shared fit step so every attack
sees identical parameters. Models are refit here at seed 0 on the shared PCA
subset for a single robustness run (the seeded 10x sweep is driven by --seeds).

Run:  python -m experiments.run_adversarial --variant binary --qubits 8
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from qgridbench.attacks.blackbox import hopskipjump_attack
from qgridbench.attacks.evasion import fgsm, gaussian_sensor_noise, pgd
from qgridbench.attacks.gradients import GradProvider
from qgridbench.attacks.label_noise import flip_labels
from qgridbench.models.classical.zoo import (
    build,
    fit_model,
    predict_proba_timed,
    stratified_cap,
)
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
from qgridbench.models.quantum.vqc import VQClassifier
from qgridbench.protocol import kernel_subset, prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml, mlflow_log
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)

# model families evaluated for robustness
CLASSICAL = ["logreg", "rbf_svm", "rf", "xgb", "lgbm", "mlp"]


class FittedZoo:
    """Fits every family on the shared PCA subset at one seed; exposes a uniform
    predict_proba(pre-encoding X) and white-box gradients where available."""

    def __init__(self, reg, encoding, n_qubits, seed, subset_cap, best_params):
        self.reg, self.encoding, self.n_qubits, self.seed = reg, encoding, n_qubits, seed
        self.n_classes = len(np.unique(reg.y["train"]))
        sub = kernel_subset(reg, subset_cap, seed=0)
        self.Xtr, self.ytr = reg.X["train"][sub], reg.y["train"][sub]
        self.models = {}
        self.best_params = best_params

        set_all_seeds(seed)
        for name in CLASSICAL:
            est = build(name, best_params.get(name, {}), seed)
            fit_model(name, est, self.Xtr, self.ytr)
            self.models[name] = est

        # QSVM (precomputed fidelity kernel over the same subset)
        self._angle = reg.angle_scaler  # train-fitted in prepare_regime (leakage rule)
        self.A_tr = self._to_angles(self.Xtr)
        states_tr = compute_states(self.A_tr, encoding, bandwidth=best_params.get("qk_bw", 1.0))
        self._states_tr = states_tr
        K_tr = fidelity_kernel(states_tr, states_tr)
        qk = _svc_on_kernel(self.n_classes, best_params.get("qk_C", 1.0)).fit(K_tr, self.ytr)
        self.models["qsvm"] = qk

        # VQC (val subsample for early stopping — bounds GPU memory, same protocol
        # as the accuracy runs)
        set_all_seeds(seed)
        vqc = VQClassifier(
            n_qubits,
            depth=best_params.get("vqc_depth", 4),
            n_classes=self.n_classes,
            lr=best_params.get("vqc_lr", 0.05),
            batch_size=best_params.get("vqc_batch", 64),
            max_epochs=best_params.get("vqc_epochs", 40),
            patience=8,
            seed=seed,
        )
        va = stratified_cap(reg.X["val"], reg.y["val"], 2000, seed=0)
        vqc.fit(self.A_tr, self.ytr, self._to_angles(reg.X["val"][va]), reg.y["val"][va])
        self.models["vqc"] = vqc

    def _to_angles(self, X):
        return self._angle.transform(X)

    def predict_proba(self, name, X):
        """X is pre-encoding PCA-space; route through each family's pipeline."""
        m = self.models[name]
        if name == "vqc":
            return m.predict_proba(self._to_angles(X))
        if name == "qsvm":
            states = compute_states(
                self._to_angles(X), self.encoding, bandwidth=self.best_params.get("qk_bw", 1.0)
            )
            K = fidelity_kernel(states, self._states_tr)
            return _svc_proba(m, K, self.n_classes)
        # shared zoo path: handles estimators fit without probability calibration
        # (rbf_svm uses a decision-margin sigmoid — see models/classical/zoo.py)
        proba, _latency = predict_proba_timed(m, X)
        return proba

    def grad_provider(self, name):
        if name == "mlp":
            return GradProvider(self.models["mlp"], "mlp")
        if name == "vqc":
            # gradient wrt PCA space = chain rule through the linear angle scaler
            scale = self._angle._mm.scale_
            base = GradProvider(self.models["vqc"], "vqc")

            def g(X, y):
                return base(self._to_angles(X), y) * scale  # d(angles)/d(X) = scale

            return g
        raise ValueError(name)


def _svc_on_kernel(n_classes, C):
    base = SVC(kernel="precomputed", C=C, class_weight="balanced")
    return OneVsRestClassifier(base) if n_classes > 2 else base


def _svc_proba(clf, K, n_classes):
    d = clf.decision_function(K)
    if n_classes == 2:
        p1 = 1.0 / (1.0 + np.exp(-d))
        return np.column_stack([1 - p1, p1])
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _f1(y, proba):
    from sklearn.metrics import f1_score

    return float(f1_score(y, proba.argmax(1), average="macro"))


def evasion_and_transfer(zoo, X, y, cfg_adv, rng, scale):
    """Returns clean F1, per-epsilon degradation, and source x target matrix (at max eps).

    `scale` is the per-feature train std: epsilon is a budget in STANDARDIZED
    units, which on PCA outputs is not the same as raw coordinates (see
    attacks/evasion.py SCALE CONTRACT)."""
    families = CLASSICAL + ["qsvm", "vqc"]
    eps_grid = cfg_adv["evasion"]["epsilons"]
    clean = {f: _f1(y, zoo.predict_proba(f, X)) for f in families}

    # white-box crafted sets per differentiable source, per epsilon
    curves = {f: {"clean": clean[f], "fgsm": {}, "pgd": {}} for f in families}
    transfer = {}  # (source -> {target -> f1}) at max epsilon
    max_eps = max(eps_grid)
    sources = cfg_adv["transfer_matrix"]["sources"]  # differentiable: mlp, vqc

    for eps in eps_grid:
        # each differentiable model attacks ITSELF (white-box)
        for src in sources:
            gp = zoo.grad_provider(src)
            X_fgsm = fgsm(gp, X, y, eps, scale)
            X_pgd = pgd(
                gp,
                X,
                y,
                eps,
                scale,
                n_steps=cfg_adv["evasion"]["pgd"]["n_steps"],
                step_frac=cfg_adv["evasion"]["pgd"]["step_frac"],
                rng=rng,
            )
            curves[src]["fgsm"][eps] = _f1(y, zoo.predict_proba(src, X_fgsm))
            curves[src]["pgd"][eps] = _f1(y, zoo.predict_proba(src, X_pgd))
            if eps == max_eps:
                transfer[src] = {t: _f1(y, zoo.predict_proba(t, X_pgd)) for t in families}

    # non-differentiable families: transfer from MLP surrogate (their curves)
    surrogate = cfg_adv["evasion"]["surrogate_model"]
    gp = zoo.grad_provider(surrogate)
    for eps in eps_grid:
        X_pgd = pgd(
            gp,
            X,
            y,
            eps,
            scale,
            n_steps=cfg_adv["evasion"]["pgd"]["n_steps"],
            step_frac=cfg_adv["evasion"]["pgd"]["step_frac"],
            rng=rng,
        )
        X_fgsm = fgsm(gp, X, y, eps, scale)
        for t in families:
            if t in sources:
                continue
            curves[t]["pgd"][eps] = _f1(y, zoo.predict_proba(t, X_pgd))
            curves[t]["fgsm"][eps] = _f1(y, zoo.predict_proba(t, X_fgsm))
    transfer[f"{surrogate}_surrogate"] = {
        t: _f1(
            y,
            zoo.predict_proba(
                t,
                pgd(
                    gp,
                    X,
                    y,
                    max_eps,
                    scale,
                    n_steps=cfg_adv["evasion"]["pgd"]["n_steps"],
                    step_frac=cfg_adv["evasion"]["pgd"]["step_frac"],
                    rng=rng,
                ),
            ),
        )
        for t in families
    }
    return {"clean": clean, "curves": curves, "transfer_matrix": transfer}


def blackbox(zoo, X, y, cfg_adv, scale, families=None, on_family=None):
    """Query-based HopSkipJump, identical budget for every family.

    HSJ is a MINIMUM-DISTORTION attack, so the headline cross-family number is the
    L-inf distortion it needs to flip a prediction at a fixed query budget (larger
    = more robust). We additionally threshold that distortion at the evasion
    epsilon grid, which puts black-box results on the same axis as the white-box /
    transfer curves: a sample counts as attacked only if HSJ found an adversarial
    point within epsilon of it under the budget.
    """
    families = families or CLASSICAL + ["qsvm", "vqc"]
    bb = cfg_adv["blackbox"]
    Xc, yc, n = X, y, len(X)  # caller passes the stratified subset; never re-slice
    eps_grid = cfg_adv["evasion"]["epsilons"]
    out = {}
    for f in families:
        Xa, budget = hopskipjump_attack(
            lambda z, _f=f: zoo.predict_proba(_f, z),
            Xc,
            zoo.n_classes,
            max_iter=bb["max_iter"],
            max_eval=bb["max_eval"],
            init_eval=bb["init_eval"],
        )
        # per-sample L-inf distortion in STANDARDIZED units, so it is comparable
        # across families and lives on the same axis as the evasion epsilon grid
        dist = (np.abs(Xa - Xc) / scale).max(axis=1)
        # attacked only where HSJ got inside the epsilon ball within budget
        f1_at_eps = {}
        for e in eps_grid:
            Xe = np.where((dist <= e)[:, None], Xa, Xc)
            f1_at_eps[str(e)] = _f1(yc, zoo.predict_proba(f, Xe))
        out[f] = {
            "clean_f1": _f1(yc, zoo.predict_proba(f, Xc)),
            "adv_f1": _f1(yc, zoo.predict_proba(f, Xa)),  # unbounded distortion
            "f1_at_eps": f1_at_eps,
            "median_linf_distortion": float(np.median(dist)),
            "mean_linf_distortion": float(dist.mean()),
            "queries_per_sample": budget["queries_per_sample"],
            "n_eval": int(n),
        }
        log.info(
            "blackbox %s: clean %.3f -> adv %.3f | median Linf %.3f | %.0f q/sample",
            f,
            out[f]["clean_f1"],
            out[f]["adv_f1"],
            out[f]["median_linf_distortion"],
            out[f]["queries_per_sample"],
        )
        if on_family:
            on_family(f, out[f])  # checkpoint: never lose a finished family
    return out


def random_perturbation_control(zoo, X, y, cfg_adv, rng, scale, n_draws=3):
    """REQUIRED control for the black-box table — never report one without the other.

    HopSkipJump is a MINIMUM-DISTORTION attack: it starts from a random adversarial
    point and binary-searches back toward the original. That procedure assumes the
    decision regions are separated by a boundary a line search can find. For a
    fidelity-kernel SVM they are not — clean and adversarial regions interleave at
    fine scale — so HSJ terminates far away and reports a large distortion that
    reads as robustness. Measured here: HSJ needs 2.818 sigma against the QSVM,
    while *random* perturbation flips 35% of its predictions at 0.2 sigma, the
    highest flip rate of any family. Without this control the black-box table
    overstates fidelity-kernel robustness by roughly 3-8x.

    Reports, per family and radius: prediction flip rate (attackable at all?) and
    mean decision margin relative to clean (is the model going uninformative, or
    staying confident?). Those two separate a genuinely robust model from one the
    attack simply cannot navigate.
    """
    families = CLASSICAL + ["qsvm", "vqc"]
    radii = cfg_adv["evasion"]["epsilons"]
    base_proba = {f: zoo.predict_proba(f, X) for f in families}
    base_pred = {f: p.argmax(1) for f, p in base_proba.items()}
    base_margin = {f: float(np.abs(p[:, 1] - 0.5).mean()) for f, p in base_proba.items()}

    out = {f: {"clean_margin": base_margin[f], "by_radius": {}} for f in families}
    for r in radii:
        flips = {f: [] for f in families}
        margins = {f: [] for f in families}
        for _ in range(n_draws):
            d = rng.standard_normal(X.shape)
            d /= np.abs(d).max(axis=1, keepdims=True)  # L-inf normalised direction
            Xp = X + r * scale * d
            for f in families:
                proba = zoo.predict_proba(f, Xp)
                flips[f].append(float((proba.argmax(1) != base_pred[f]).mean()))
                margins[f].append(float(np.abs(proba[:, 1] - 0.5).mean()))
        for f in families:
            out[f]["by_radius"][str(r)] = {
                "flip_rate": float(np.mean(flips[f])),
                "margin_ratio": (
                    float(np.mean(margins[f]) / base_margin[f]) if base_margin[f] else float("nan")
                ),
            }
    return out


def sensor_noise(zoo, X, y, cfg_adv, rng, scale):
    families = CLASSICAL + ["qsvm", "vqc"]
    out = {f: {} for f in families}
    for snr in cfg_adv["sensor_noise"]["snr_db"]:
        Xn = gaussian_sensor_noise(X, snr, rng, scale)
        for f in families:
            out[f][snr] = _f1(y, zoo.predict_proba(f, Xn))
    return out


def poisoning(reg, encoding, n_qubits, cfg_adv, subset_cap, best_params, seed):
    """Flip training labels, refit each family, measure clean-test degradation."""
    out = {}
    for rate in [0.0] + cfg_adv["poisoning"]["label_flip_rates"]:
        rng = spawn_rng(seed, f"poison_{rate}")
        reg2 = reg
        sub = kernel_subset(reg2, subset_cap, seed=0)
        y_flipped = flip_labels(reg2.y["train"][sub], rate, rng)
        reg_poisoned = _reg_with_labels(reg2, sub, y_flipped)
        zoo = FittedZoo(reg_poisoned, encoding, n_qubits, seed, subset_cap, best_params)
        Xte, yte = reg.X["test"], reg.y["test"]
        out[rate] = {f: _f1(yte, zoo.predict_proba(f, Xte)) for f in CLASSICAL + ["qsvm", "vqc"]}
    return out


def _reg_with_labels(reg, sub, y_sub):
    """Shallow copy of a Regime whose train labels (on the subset) are replaced.

    The subset is what every family trains on, so we overlay flipped labels onto
    the full train label array at the subset indices."""
    import copy

    r = copy.copy(reg)
    r.y = dict(reg.y)
    y_train = reg.y["train"].copy()
    y_train[sub] = y_sub
    r.y["train"] = y_train
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="angle_ry")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--best-params", default=None, help="JSON file of tuned params per family")
    ap.add_argument("--skip-blackbox", action="store_true")
    ap.add_argument(
        "--bb-families",
        nargs="*",
        default=None,
        help="run the black-box stage for this family subset only and checkpoint each "
        "result to results/blackbox_<tag>/. Families are independent, so one process "
        "per family runs them concurrently; merged by --merge-blackbox.",
    )
    ap.add_argument(
        "--merge-blackbox",
        action="store_true",
        help="collect results/blackbox_<tag>/*.json into one run dir and exit",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="subset of {evasion,noise,blackbox,random,poison}; blackbox is the expensive one "
        "and is normally run as its own job",
    )
    args = ap.parse_args()
    do = set(args.only) if args.only else {"evasion", "noise", "blackbox", "random", "poison"}
    if args.skip_blackbox:
        do.discard("blackbox")

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_c = load_yaml(REPO_ROOT / "configs" / "classical.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    seeds = args.seeds or cfg_c["evaluation"]["seeds"]
    subset_cap = cfg_q["qkernel"]["subset_cap"]
    best_params = json.loads(open(args.best_params).read()) if args.best_params else {}

    tag = f"{args.variant}_q{args.qubits}"
    bb_dir = REPO_ROOT / "results" / f"blackbox_{tag}"

    if args.merge_blackbox:
        merged = {p.stem: json.loads(p.read_text()) for p in sorted(bb_dir.glob("*.json"))}
        if not merged:
            raise SystemExit(f"no per-family black-box results under {bb_dir}")
        with RunDir(f"adversarial_{tag}_blackbox", config=vars(args), seeds=seeds) as run:
            run.write_json("results.json", {"per_seed": [{"blackbox": merged}], "poisoning": None})
            log.info("merged %d families -> %s", len(merged), run.path)
        return

    reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
    scale = reg.feature_scale  # attack budget unit — see attacks/evasion.py SCALE CONTRACT
    log.info(
        "attack scale (train per-feature std): min %.3f max %.3f mean %.3f",
        scale.min(),
        scale.max(),
        scale.mean(),
    )
    # Cost-control subsets are STRATIFIED SAMPLES, never head slices. The saved
    # split indices are sorted, so X["test"][:n] is the first n rows in original
    # file order, not a sample of the test distribution: on this dataset that slice
    # carries a 93.4% attack rate against a true 71.0%, which deflates macro-F1 and
    # manufactures "F1 rises under attack" artifacts. stratified_cap is the shared
    # subsetting primitive already used for the kernel cap.
    ev_idx = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    bb_idx = stratified_cap(
        reg.X["test"], reg.y["test"], cfg_adv["blackbox"]["eval_subset"], seed=0
    )
    log.info(
        "eval subsets: evasion n=%d (positive rate %.3f), black-box n=%d (%.3f); "
        "full test n=%d (%.3f)",
        len(ev_idx),
        reg.y["test"][ev_idx].mean(),
        len(bb_idx),
        reg.y["test"][bb_idx].mean(),
        len(reg.y["test"]),
        reg.y["test"].mean(),
    )

    name = f"adversarial_{tag}" + ("" if len(do) > 1 else f"_{next(iter(do))}")
    with RunDir(name, config={**vars(args), "stages": sorted(do)}, seeds=seeds) as run:
        per_seed = []
        for s in seeds:
            rng = spawn_rng(s, "adv")
            zoo = FittedZoo(reg, args.encoding, args.qubits, s, subset_cap, best_params)
            Xte, yte = reg.X["test"][ev_idx], reg.y["test"][ev_idx]
            res = {}
            if "evasion" in do:
                res["evasion"] = evasion_and_transfer(zoo, Xte, yte, cfg_adv, rng, scale)
            if "noise" in do:
                res["sensor_noise"] = sensor_noise(zoo, Xte, yte, cfg_adv, rng, scale)
            # control for the black-box table; cheap (predictions only, no attack)
            if "random" in do:
                res["random_control"] = random_perturbation_control(
                    zoo, reg.X["test"][bb_idx], reg.y["test"][bb_idx], cfg_adv, rng, scale
                )
            # HopSkipJump is deterministic given (model, seed) and expensive:
            # run once at the first seed; evasion/noise curves carry the seed spread
            if "blackbox" in do and s == seeds[0]:
                bb_dir.mkdir(parents=True, exist_ok=True)

                def checkpoint(fam, rec):
                    (bb_dir / f"{fam}.json").write_text(json.dumps(rec, indent=2))

                res["blackbox"] = blackbox(
                    zoo,
                    reg.X["test"][bb_idx],
                    reg.y["test"][bb_idx],
                    cfg_adv,
                    scale,
                    families=args.bb_families,
                    on_family=checkpoint,
                )
            per_seed.append(res)
            run.write_json("results_partial.json", {"per_seed": per_seed})  # checkpointed
            log.info(
                "seed %d done | %s",
                s,
                {k: round(v, 3) for k, v in res.get("evasion", {}).get("clean", {}).items()}
                or sorted(res),
            )
        poison = (
            poisoning(reg, args.encoding, args.qubits, cfg_adv, subset_cap, best_params, seeds[0])
            if "poison" in do
            else None
        )
        run.write_json("results.json", {"per_seed": per_seed, "poisoning": poison})
        mlflow_log(
            f"adversarial_{tag}",
            vars(args),
            {f"clean_{k}": v for k, v in per_seed[0]["evasion"]["clean"].items()},
            tags={"phase": "adversarial"},
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
