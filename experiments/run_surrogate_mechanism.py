"""P19e: WHY does an unweighted surrogate inflate the transfer asymmetry?

P19d (10 seeds) showed the C1 asymmetry is a surrogate artefact: with the published
sklearn MLP (which sklearn cannot class-weight) the VQC takes 0.344 macro-F1 of damage
against a classical mean of 0.208 and a 9.5x forward/reverse asymmetry; with an
identical-architecture class-WEIGHTED MLP the classical damage is unchanged (0.202) while
the VQC's falls to 0.133 and the asymmetry collapses to 1.03x. A finding that reverses a
headline needs a mechanism, not just a number.

HYPOTHESIS (pre-registered before running). An MLP fitted without class weights on a
71%-positive task collapses toward the majority class, so its input-gradient field is
dominated by ONE global direction -- "push everything toward natural" -- rather than by
point-specific boundary structure. Under sign-based PGD every point then receives nearly
the same corner of the L-inf ball. That is a class-prior shift, not a transferred decision
boundary, and it maximally damages any model whose own decisions are near chance (the
VQC), while a tree ensemble with real structure resists it.

PREDICTIONS, each with its own falsifier:
  P1  the unweighted surrogates predict the majority class far more often than the
      weighted one on clean data.
  P2  their perturbations are far more globally aligned: the shared-direction share
      ||mean_i delta_i|| / mean_i ||delta_i|| is high for the unweighted surrogates and
      materially lower for the weighted one.
  P3  the damage they do to the VQC is one-directional: attacked VQC predictions move
      overwhelmingly toward one class, whereas the weighted surrogate's attack moves them
      both ways.
KILL: if P2 fails -- shared-direction share within ~0.05 across surrogates -- the
class-prior account is wrong and the artefact needs a different explanation, which is
logged as an open mechanism rather than papered over.

Run: python -m experiments.run_surrogate_mechanism

Graduated from tmp/ because the appendix reports its numbers: the shared-direction share
that separates the unweighted from the class-weighted surrogate is the mechanism behind the
corrected C1 asymmetry, so the producer has to be a file the repository carries. The RunDir
name is unchanged, so the existing run directories still match.
"""

from __future__ import annotations

import json

import numpy as np

from experiments.run_adversarial import FittedZoo
from experiments.run_review_claims import WeightedMLP
from qgridbench.attacks.evasion import pgd
from qgridbench.attacks.gradients import GradProvider
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)

SEEDS = [0, 1, 2]
EPS = 0.5
ENCODING = "zz"


def main():
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    best_params = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    mp = best_params["mlp"]

    reg = prepare_regime("binary", "pca", n_components=8, seed=0)
    scale = reg.feature_scale
    ev_idx = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    va_idx = stratified_cap(reg.X["val"], reg.y["val"], 2000, seed=0)
    Xte, yte = reg.X["test"][ev_idx], reg.y["test"][ev_idx]
    Xva, yva = reg.X["val"][va_idx], reg.y["val"][va_idx]

    with RunDir("probe_surrogate_mechanism_binary_q8", config={"eps": EPS}, seeds=SEEDS) as run:
        per_seed = []
        for s in SEEDS:
            rng = spawn_rng(s, "adv")
            zoo = FittedZoo(reg, ENCODING, 8, s, cfg_q["qkernel"]["subset_cap"], best_params)
            set_all_seeds(s)
            wmlp = WeightedMLP(
                mp["h1"], mp["h2"], mp["alpha"], mp["learning_rate_init"], mp["batch_size"], s
            ).fit(zoo.Xtr, zoo.ytr, Xva, yva)
            set_all_seeds(s)
            umlp = WeightedMLP(
                mp["h1"],
                mp["h2"],
                mp["alpha"],
                mp["learning_rate_init"],
                mp["batch_size"],
                s,
                weighted=False,
            ).fit(zoo.Xtr, zoo.ytr, Xva, yva)

            sources = {
                "mlp": (zoo.grad_provider("mlp"), lambda X: zoo.predict_proba("mlp", X)),
                "mlp_torch_unweighted": (GradProvider(umlp, "vqc"), umlp.predict_proba),
                "mlp_weighted": (GradProvider(wmlp, "vqc"), wmlp.predict_proba),
            }
            vqc_clean_pred = zoo.predict_proba("vqc", Xte).argmax(1)
            rec = {"positive_rate_eval": float(yte.mean())}
            for name, (gp, pp) in sources.items():
                Xadv = pgd(
                    gp,
                    Xte,
                    yte,
                    EPS,
                    scale,
                    n_steps=cfg_adv["evasion"]["pgd"]["n_steps"],
                    step_frac=cfg_adv["evasion"]["pgd"]["step_frac"],
                    rng=rng,
                )
                delta = (Xadv - Xte) / scale  # in sigma units
                mean_dir = delta.mean(axis=0)
                # P2: how much of the per-point perturbation is one shared global shift
                shared = float(np.linalg.norm(mean_dir) / np.linalg.norm(delta, axis=1).mean())
                # sign agreement with the modal sign pattern, per coordinate
                modal_sign = np.sign(mean_dir)
                sign_agree = float((np.sign(delta) == modal_sign).mean())
                # P1: clean prediction balance of the surrogate itself
                pred = pp(Xte).argmax(1)
                # P3: which way the attack moves the VQC's predictions
                vqc_adv_pred = zoo.predict_proba("vqc", Xadv).argmax(1)
                moved = vqc_adv_pred != vqc_clean_pred
                to_pos = float((vqc_adv_pred[moved] == 1).mean()) if moved.any() else float("nan")
                rec[name] = {
                    "surrogate_positive_rate": float(pred.mean()),
                    "shared_direction_share": shared,
                    "modal_sign_agreement": sign_agree,
                    "vqc_flip_rate": float(moved.mean()),
                    "vqc_flips_toward_positive": to_pos,
                }
            per_seed.append(rec)
            run.write_json("results_partial.json", {"per_seed": per_seed})
            log.info("seed %d | %s", s, json.dumps(rec))

        summary = {}
        for name in ["mlp", "mlp_torch_unweighted", "mlp_weighted"]:
            summary[name] = {
                k: float(np.mean([p[name][k] for p in per_seed])) for k in per_seed[0][name]
            }
        gap = (
            summary["mlp"]["shared_direction_share"]
            - summary["mlp_weighted"]["shared_direction_share"]
        )
        verdict = (
            "CLASS-PRIOR ACCOUNT SUPPORTED"
            if gap > 0.05
            else "KILLED — shared-direction share does not separate the surrogates"
        )
        run.write_json(
            "results.json",
            {
                "per_seed": per_seed,
                "summary": summary,
                "shared_direction_gap": gap,
                "verdict": verdict,
            },
        )
        log.info("VERDICT: %s (gap %.3f)\n%s", verdict, gap, json.dumps(summary, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
