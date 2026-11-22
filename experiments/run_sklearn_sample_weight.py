"""P19f: does the ONE-ARGUMENT fix reproduce P19d? sklearn MLP + `sample_weight`.

P19d diagnosed the surrogate defect and fixed it with a torch re-implementation, on the stated
grounds that "sklearn's MLPClassifier supports neither class_weight nor sample_weight" — which
is what `models/classical/zoo.py`'s own docstring says. A concurrent session checked that claim
(decision log 2026-08-15T10:40Z) and found it half wrong on the installed environment: on
scikit-learn 1.9.0 `class_weight` is indeed absent from the constructor, but `fit()` DOES accept
`sample_weight`. Verified here again by signature inspection before running.

That changes the recommended remedy from "re-implement the MLP" to "pass one argument", so it
has to be tested rather than assumed.

H: fitting the PUBLISHED sklearn MLP — same class, same tuned hyperparameters, same seeds — with
`compute_sample_weight("balanced", y)` reproduces the torch weighted arm: clean macro-F1 above
the 0.5024 floor (torch arm: 0.552), and the C1 asymmetry collapsing from 9.54x toward 1x.
KILL: if the sklearn+sample_weight arm stays below the floor, `sample_weight` is accepted but
ineffective for this purpose, the torch path stays on the critical path, and the zoo fix is a
re-implementation rather than an argument.

Why this arm is BETTER than the torch one if it works: attacks are crafted through the existing,
already-tested hand-rolled `mlp_loss_input_grad` path (`attacks/gradients.py`), so the corrected
surrogate uses the published gradient code and the "is it the re-implementation?" control that
P19d had to spend an arm on becomes unnecessary.

Run:  uv run python -m experiments.run_sklearn_sample_weight [seeds ...] [--best-params FILE]
      [--run-tag TAG]
      (default seeds 0..9, the published run
      `20260815T104438Z_probe_sklearn_sample_weight_binary_q8`)

Graduated from `tmp/probe_sklearn_sample_weight.py` (R4-24, ruling 8 of 2026-11-16) because the
appendix reports its numbers: the sample-weight fit's 0.76 forward/reverse ratio, the torch
fit's 1.06, and the step-1 re-analysis of P23 (artefact damage +0.230 for the VQC, Spearman
0.107). The computation is unchanged; the RunDir name keeps the probe's name so the published
run still resolves.
"""

from __future__ import annotations

import argparse
import inspect
import json

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from experiments.exports import Exports
from experiments.run_adversarial import CLASSICAL, FittedZoo
from experiments.run_review_claims import WeightedMLP, floors, macro_f1
from experiments.runner_guard import assert_clean_predictor
from experiments.tuned_file import add_tuned_file_args, load_tuned_file, run_name
from qgridbench.attacks.evasion import pgd
from qgridbench.attacks.gradients import GradProvider
from qgridbench.models.classical.zoo import build, stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)

SEEDS = list(range(10))
EPS = 0.5
ENCODING = "zz"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seeds", type=int, nargs="*", default=SEEDS)
    add_tuned_file_args(ap)
    args = ap.parse_args()
    seeds = args.seeds or SEEDS
    if "sample_weight" not in inspect.signature(MLPClassifier.fit).parameters:
        raise SystemExit("this environment's MLPClassifier.fit has no sample_weight — probe moot")

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    bp_rel, best = load_tuned_file(args.best_params, "binary", 8)
    mp = best["mlp"]

    reg = prepare_regime("binary", "pca", n_components=8, seed=0)
    scale = reg.feature_scale
    ev_idx = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    va_idx = stratified_cap(reg.X["val"], reg.y["val"], 2000, seed=0)
    Xte, yte = reg.X["test"][ev_idx], reg.y["test"][ev_idx]
    Xva, yva = reg.X["val"][va_idx], reg.y["val"][va_idx]
    fl = floors(reg.y["train"], yte)["stratified_random"]

    families = CLASSICAL + ["qsvm", "vqc"]
    CLS = ["logreg", "rbf_svm", "rf", "xgb", "lgbm"]

    with RunDir(
        run_name("probe_sklearn_sample_weight_binary_q8", args.run_tag),
        config={"eps": EPS, "best_params": bp_rel},
        seeds=seeds,
    ) as run:
        per_seed = []
        ex = Exports("binary", bp_rel)
        for s in seeds:
            rng = spawn_rng(s, ex.stream("adv"))
            zoo = FittedZoo(reg, ENCODING, 8, s, cfg_q["qkernel"]["subset_cap"], best)

            # THE ARM UNDER TEST: published estimator, published hyperparameters, one argument.
            set_all_seeds(s)
            mlp_sw = build("mlp", mp, s)
            mlp_sw.fit(zoo.Xtr, zoo.ytr, sample_weight=compute_sample_weight("balanced", zoo.ytr))

            # reference arm: the torch weighted MLP P19d used
            set_all_seeds(s)
            wmlp = WeightedMLP(
                mp["h1"], mp["h2"], mp["alpha"], mp["learning_rate_init"], mp["batch_size"], s
            ).fit(zoo.Xtr, zoo.ytr, Xva, yva)

            extra = {"mlp_sample_weight": mlp_sw, "mlp_torch_weighted": wmlp}

            def proba(name, X, _e=extra):
                p = _e[name].predict_proba(X) if name in _e else zoo.predict_proba(name, X)
                if X is Xte:  # the clean set: a constant or non-finite predictor fails the run
                    assert_clean_predictor(name, yte, p, p.argmax(1))
                return p

            all_models = families + list(extra)
            sets = {"clean": Xte}
            for src, gp in [
                ("mlp", zoo.grad_provider("mlp")),
                # published gradient path, corrected model
                ("mlp_sample_weight", GradProvider(mlp_sw, "mlp")),
                ("mlp_torch_weighted", GradProvider(wmlp, "vqc")),
                ("vqc", zoo.grad_provider("vqc")),
            ]:
                sets[f"pgd_{src}"] = pgd(
                    gp,
                    Xte,
                    yte,
                    EPS,
                    scale,
                    n_steps=cfg_adv["evasion"]["pgd"]["n_steps"],
                    step_frac=cfg_adv["evasion"]["pgd"]["step_frac"],
                    rng=rng,
                )

            src = {k: k.removeprefix("pgd_") for k in sets if k != "clean"}
            f1 = ex.score(s, sets, all_models, proba, yte, ev_idx, macro_f1, src)
            pos_rate = {
                m: float(proba(m, Xte).argmax(1).mean())
                for m in ["mlp", "mlp_sample_weight", "mlp_torch_weighted"]
            }
            per_seed.append({"f1": f1, "surrogate_positive_rate": pos_rate})
            run.write_json("results_partial.json", {"per_seed": per_seed})
            log.info(
                "seed %d | clean mlp %.3f | +sample_weight %.3f | torch %.3f | pos-rate %s",
                s,
                f1["clean"]["mlp"],
                f1["clean"]["mlp_sample_weight"],
                f1["clean"]["mlp_torch_weighted"],
                {k: round(v, 3) for k, v in pos_rate.items()},
            )

        def mean(path):
            return float(np.mean([path(p) for p in per_seed]))

        summary = {"floor": fl}
        for src, self_name in [
            ("pgd_mlp", "mlp"),
            ("pgd_mlp_sample_weight", "mlp_sample_weight"),
            ("pgd_mlp_torch_weighted", "mlp_torch_weighted"),
        ]:
            vq = mean(lambda p, s=src: p["f1"][s]["vqc"])
            vq_cln = mean(lambda p: p["f1"]["clean"]["vqc"])
            cls_adv = float(np.mean([mean(lambda p, s=src, t=t: p["f1"][s][t]) for t in CLS]))
            cls_cln = float(np.mean([mean(lambda p, t=t: p["f1"]["clean"][t]) for t in CLS]))
            rev = mean(lambda p, n=self_name: p["f1"]["pgd_vqc"][n])
            rev_cln = mean(lambda p, n=self_name: p["f1"]["clean"][n])
            summary[self_name] = {
                "clean": rev_cln,
                "clean_minus_floor": rev_cln - fl,
                "positive_rate": mean(lambda p, n=self_name: p["surrogate_positive_rate"][n]),
                "vqc_adv": vq,
                "classical_mean_adv": cls_adv,
                "vqc_damage": vq_cln - vq,
                "classical_damage": cls_cln - cls_adv,
                "damage_ratio": (vq_cln - vq) / (cls_cln - cls_adv),
                "forward_over_reverse": (vq_cln - vq) / max(rev_cln - rev, 1e-9),
                "own_adv": mean(lambda p, s=src, n=self_name: p["f1"][s][n]),
            }
        ex.write(run.path)
        run.write_json("results.json", {"per_seed": per_seed, "summary": summary})
        log.info("SUMMARY\n%s", json.dumps(summary, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
