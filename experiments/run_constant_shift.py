"""P23: is the class-prior artefact a property of the ENCODING rather than of the surrogate?

WHY THIS EXISTS. P19d/P19e found that an unweighted MLP surrogate manufactures C1's transfer
asymmetry, and attributed it to a class-prior push that "flattens any near-chance model". A free
re-analysis of P19f killed that attribution: across 7 targets, artefact damage does NOT track
distance from the random floor (Spearman 0.107, p = 0.82), and the counterexample is decisive —
logreg sits 0.012 above the floor with artefact damage of -0.028 while the VQC sits AT the floor
with +0.230, 10.4x any other family. So the VQC is not susceptible for being weak.

HYPOTHESIS (pre-registered). The VQC is the only model in the zoo whose decision function is a
BOUNDED PERIODIC function of the input: angle encoding means a constant displacement in feature
space becomes a constant phase shift, which rotates every point's PauliZ expectation coherently.
For a monotone decision function (logreg, trees) a constant displacement is only a threshold
offset and moves few points. If that is right, the artefact is a property of the ENCODING and the
surrogate was merely the thing that happened to supply a global direction.

TEST. Strip the attack down to its global component: apply ONE constant vector to every test
point, at the headline budget, and measure damage per family. No per-point gradient, no training,
no surrogate in the loop at evaluation time.
  arms: `constant_prior`  — the mean PGD direction from the broken surrogate, rescaled to
                            L-inf = 0.5 sigma and applied identically to all points
        `constant_random` — a matched-norm constant random direction (is any constant shift
                            enough, or must it be THIS direction?)
        `perpoint_random` — matched-norm per-point random (the study's existing control)
        `pgd_broken`      — the full per-point attack, for reference

PREDICTIONS, each falsifiable:
  P1 the VQC's damage under `constant_prior` is a large fraction of its damage under the full
     per-point attack, while for logreg and the tree families it is a small fraction.
  P2 the VQC is damaged more by `constant_prior` than by `constant_random` at matched norm --
     otherwise ANY constant shift breaks it and the class-prior story adds nothing.
KILL: if every family loses comparable macro-F1 under `constant_prior`, a global shift is not
what singles out the VQC and the encoding account is wrong. If the VQC's constant-shift damage is
within noise of its matched-norm random damage, the direction does not matter and the account is
also wrong.

Run:  python -m experiments.run_constant_shift

Graduated from tmp/ because the appendix reports its numbers: the class-prior account is
rejected on its evidence (Spearman 0.107, p = 0.82; the global component moves the VQC by
-0.012 against 0.241 for the per-point residual), and a single constant displacement
reproduces 94% of the RBF-SVM's damage. The RunDir name keeps its original `probe_` prefix
so already-recorded results still resolve.
"""

from __future__ import annotations

import json

import numpy as np

from experiments.run_adversarial import CLASSICAL, FittedZoo
from experiments.run_review_claims import floors, macro_f1
from qgridbench.attacks.evasion import pgd
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import spawn_rng

log = get_logger(__name__)

SEEDS = [0, 1, 2]
EPS = 0.5
ENCODING = "zz"


def main():
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())

    reg = prepare_regime("binary", "pca", n_components=8, seed=0)
    scale = reg.feature_scale
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    Xte, yte = reg.X["test"][ev], reg.y["test"][ev]
    fl = floors(reg.y["train"], yte)["stratified_random"]
    families = CLASSICAL + ["qsvm", "vqc"]

    with RunDir("probe_constant_shift_binary_q8", config={"eps": EPS}, seeds=SEEDS) as run:
        per_seed = []
        for s in SEEDS:
            rng = spawn_rng(s, "adv")
            zoo = FittedZoo(reg, ENCODING, 8, s, cfg_q["qkernel"]["subset_cap"], best)

            # the full per-point attack from the BROKEN surrogate (the published one)
            Xpgd = pgd(
                zoo.grad_provider("mlp"),
                Xte,
                yte,
                EPS,
                scale,
                n_steps=cfg_adv["evasion"]["pgd"]["n_steps"],
                step_frac=cfg_adv["evasion"]["pgd"]["step_frac"],
                rng=rng,
            )
            delta = Xpgd - Xte

            def at_budget(v):
                """rescale a direction so its L-inf budget is exactly EPS * scale"""
                u = v / np.abs(v / scale).max()
                return u * EPS

            d_prior = at_budget(delta.mean(axis=0))  # ONE vector, the global component
            d_rand = at_budget(rng.standard_normal(Xte.shape[1]))  # ONE random vector
            pp = rng.standard_normal(Xte.shape)
            pp = pp / np.abs(pp).max(axis=1, keepdims=True)
            # THE COMPLEMENT: strip the global component instead of keeping it, so the two
            # arms decompose the attack rather than sampling one half of it.
            resid = delta - delta.mean(axis=0)
            sets = {
                "clean": Xte,
                "constant_prior": Xte + d_prior,
                "constant_random": Xte + d_rand,
                "perpoint_residual": Xte + resid / np.abs(resid / scale).max() * EPS,
                "perpoint_random": Xte + EPS * scale * pp,
                "pgd_broken": Xpgd,
            }
            f1 = {
                k: {f: macro_f1(yte, zoo.predict_proba(f, X)) for f in families}
                for k, X in sets.items()
            }
            per_seed.append(f1)
            run.write_json("results_partial.json", {"per_seed": per_seed})
            log.info(
                "seed %d | constant_prior %s",
                s,
                {f: round(f1["constant_prior"][f], 3) for f in families},
            )

        def mean(k, f):
            return float(np.mean([p[k][f] for p in per_seed]))

        summary = {}
        for f in families:
            c = mean("clean", f)
            dp = c - mean("constant_prior", f)
            dr = c - mean("constant_random", f)
            dres = c - mean("perpoint_residual", f)
            dpp = c - mean("perpoint_random", f)
            dfull = c - mean("pgd_broken", f)
            summary[f] = {
                "clean": c,
                "clean_minus_floor": c - fl,
                "damage_constant_prior": dp,
                "damage_constant_random": dr,
                "damage_perpoint_residual": dres,
                "damage_perpoint_random": dpp,
                "residual_share_of_full": dres / dfull if dfull > 1e-9 else float("nan"),
                "damage_full_pgd": dfull,
                "constant_share_of_full": dp / dfull if dfull > 1e-9 else float("nan"),
                "prior_over_random_constant": dp / dr if abs(dr) > 1e-9 else float("nan"),
            }
        run.write_json("results.json", {"per_seed": per_seed, "summary": summary, "floor": fl})
        print(
            f"\n{'family':<9}{'clean-fl':>9}{'const-prior':>12}{'const-rand':>11}"
            f"{'residual':>10}{'pp-rand':>9}"
            f"{'full PGD':>10}{'const/full':>11}{'resid/full':>11}"
        )
        for f in families:
            v = summary[f]
            print(
                f"{f:<9}{v['clean_minus_floor']:>9.3f}{v['damage_constant_prior']:>12.3f}"
                f"{v['damage_constant_random']:>11.3f}"
                f"{v['damage_perpoint_residual']:>10.3f}{v['damage_perpoint_random']:>9.3f}"
                f"{v['damage_full_pgd']:>10.3f}{v['constant_share_of_full']:>11.2f}"
                f"{v['residual_share_of_full']:>11.2f}"
            )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
