"""Surrogate control: does the transfer asymmetry survive a correctly fitted surrogate?

THE CONTROL BEHIND SEC. "The transfer asymmetry is an artifact of the surrogate". Every
transfer number in this study is crafted on the MLP, and `MLPClassifier` takes no
`class_weight` argument, so that model alone is fitted with no imbalance handling on a
71%-positive task (see `CLASS_BALANCE_ROUTE` in models/classical/zoo.py). This script runs
the identical protocol with the surrogate refitted through `sample_weight` and reports both.

Only the surrogate's FIT changes between those two arms. Same estimator class, same tuned
hyperparameters, same seeds, same subsets, same encoding, and the same hand-rolled
input-gradient path (`mlp_loss_input_grad`) used for every published transfer number -- so
any difference is the fit and not the attack.

THIRD ARM (P11a, decision log 2026-08-15T19:52Z): a surrogate of a DIFFERENT FUNCTION CLASS.
Two fits of one estimator cannot distinguish "this defect caused the artifact" from "the
artifact is specific to neural-net surrogates", so the RBF-SVM is attacked as a third source:
different function class, exact finite-difference-verified input gradient
(`rbf_svm_loss_input_grad`), and `class_weight` in its balance route, so it carries no
imbalance defect by construction. Ordering the three arms by how far each surrogate clears
the stratified-random floor makes the result a dose-response rather than a before/after.

Deliverables: per-target damage under each surrogate, the VQC-crafted reverse direction,
and the forward/reverse asymmetry ratio that the paper quotes.

Run:  python -m experiments.run_surrogate_control --variant binary --qubits 8
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.run_adversarial import CLASSICAL, FittedZoo, _f1
from qgridbench.attacks.evasion import pgd
from qgridbench.attacks.gradients import GradProvider
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)

CLASSICAL_TARGETS = ["logreg", "rbf_svm", "rf", "xgb", "lgbm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="zz")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--best-params", default="results/adv_best_params_binary_q8.json")
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    best = json.loads((REPO_ROOT / args.best_params).read_text())
    eps = max(cfg_adv["evasion"]["epsilons"])
    pgd_cfg = cfg_adv["evasion"]["pgd"]

    reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
    scale = reg.feature_scale
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    Xte, yte = reg.X["test"][ev], reg.y["test"][ev]
    floor = trivial_baselines(reg.y["train"], yte)["baseline_stratified_random"]["aggregate"][
        "macro_f1"
    ]["mean"]
    families = CLASSICAL + ["qsvm", "vqc"]

    tag = f"{args.variant}_q{args.qubits}"
    with RunDir(f"surrogate_control_{tag}", config=vars(args), seeds=args.seeds) as run:
        per_seed = []
        for s in args.seeds:
            rng = spawn_rng(s, "adv")  # same stream the published evasion run used
            zoo = FittedZoo(
                reg, args.encoding, args.qubits, s, cfg_q["qkernel"]["subset_cap"], best
            )

            # the corrected surrogate: same class, same params, one extra argument
            set_all_seeds(s)
            fixed = build("mlp", best["mlp"], s)
            fit_model("mlp", fixed, zoo.Xtr, zoo.ytr, balance_via_sample_weight=True)

            def proba(name, X, _f=fixed):
                return _f.predict_proba(X) if name == "mlp_balanced" else zoo.predict_proba(name, X)

            models = families + ["mlp_balanced"]
            sets = {"clean": Xte}
            # ORDER IS LOAD-BEARING: `rng` is one stream shared by the calls below, so an arm
            # inserted ahead of another shifts that one's random PGD start and silently moves
            # a published number. The third arm therefore goes LAST, leaving the mlp,
            # mlp_balanced and vqc draws bit-identical to the run the paper already reports.
            for src, gp in [
                ("mlp", zoo.grad_provider("mlp")),
                ("mlp_balanced", GradProvider(fixed, "mlp")),
                ("vqc", zoo.grad_provider("vqc")),
                # third arm: different function class, no imbalance defect by construction
                ("rbf_svm", GradProvider(zoo.models["rbf_svm"], "rbf_svm")),
            ]:
                sets[f"pgd_{src}"] = pgd(
                    gp,
                    Xte,
                    yte,
                    eps,
                    scale,
                    n_steps=pgd_cfg["n_steps"],
                    step_frac=pgd_cfg["step_frac"],
                    rng=rng,
                )
            f1 = {k: {m: _f1(yte, proba(m, X)) for m in models} for k, X in sets.items()}
            pos = {m: float(proba(m, Xte).argmax(1).mean()) for m in ("mlp", "mlp_balanced")}
            per_seed.append({"f1": f1, "surrogate_positive_rate": pos})
            run.write_json("results_partial.json", {"per_seed": per_seed})
            log.info(
                "seed %d | surrogate clean %.3f -> %.3f | VQC under it %.3f -> %.3f",
                s,
                f1["clean"]["mlp"],
                f1["clean"]["mlp_balanced"],
                f1["pgd_mlp"]["vqc"],
                f1["pgd_mlp_balanced"]["vqc"],
            )

        def mn(fn):
            return float(np.mean([fn(p) for p in per_seed]))

        summary = {"floor": floor, "eps": eps, "positive_rate_eval": float(yte.mean())}
        for src, name in (
            ("pgd_mlp", "mlp"),
            ("pgd_mlp_balanced", "mlp_balanced"),
            ("pgd_rbf_svm", "rbf_svm"),
        ):
            vq_c = mn(lambda p: p["f1"]["clean"]["vqc"])
            vq_a = mn(lambda p, s=src: p["f1"][s]["vqc"])
            cls_c = float(
                np.mean([mn(lambda p, t=t: p["f1"]["clean"][t]) for t in CLASSICAL_TARGETS])
            )
            cls_a = float(
                np.mean([mn(lambda p, s=src, t=t: p["f1"][s][t]) for t in CLASSICAL_TARGETS])
            )
            # a surrogate is white-box against itself, which would inflate its own classical
            # mean; the peer column therefore excludes it. Identical to the full set for the
            # two MLP arms, since CLASSICAL_TARGETS does not contain the MLP.
            peers = [t for t in CLASSICAL_TARGETS if t != name]
            peer_c = float(np.mean([mn(lambda p, t=t: p["f1"]["clean"][t]) for t in peers]))
            peer_a = float(np.mean([mn(lambda p, s=src, t=t: p["f1"][s][t]) for t in peers]))
            rev_c = mn(lambda p, n=name: p["f1"]["clean"][n])
            rev_a = mn(lambda p, n=name: p["f1"]["pgd_vqc"][n])
            summary[name] = {
                "surrogate_clean": rev_c,
                "surrogate_clean_minus_floor": rev_c - floor,
                "surrogate_positive_rate": mn(lambda p, n=name: p["surrogate_positive_rate"][n])
                if name in ("mlp", "mlp_balanced")
                else None,
                "vqc_damage": vq_c - vq_a,
                "classical_damage": cls_c - cls_a,
                "classical_peer_damage": peer_c - peer_a,
                "damage_ratio_vqc_over_classical": (vq_c - vq_a) / (cls_c - cls_a),
                "reverse_damage": rev_c - rev_a,
                "forward_over_reverse": (vq_c - vq_a) / max(rev_c - rev_a, 1e-9),
                "per_target_adv_f1": {t: mn(lambda p, s=src, t=t: p["f1"][s][t]) for t in families},
            }
        run.write_json("results.json", {"per_seed": per_seed, "summary": summary})
        log.info("SUMMARY\n%s", json.dumps(summary, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
