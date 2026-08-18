"""P19g: the operating-point result at DEPLOYMENT-GRADE accuracy (full 128 features).

P19b measured detection at a fixed false-alarm budget in the matched PCA-8 regime and found
0.3-4.1% detection at 1% FPR, with the attack driving the realised false-alarm rate to
0.465 (MLP). That regime sits near the random floor, so an operator can dismiss it. This probe
repeats the measurement where the models actually work (LightGBM 0.906 macro-F1) and decides
whether the finding is a property of a weak regime or of the detectors themselves.

H: at deployment-grade accuracy the detectors ARE usable at a 1% false-alarm budget (TPR well
above the PCA-8 regime's 4%), and the attack still converts the failure into a false-alarm flood
rather than into silence.
KILL: if the realised FPR under attack stays near its clean value, the false-alarm-flood finding
is an artefact of the near-chance regime and must be scoped to it (or dropped).

RBF-SVM is excluded: sklearn's single-threaded SVC on 47k x 128 was the cause of a 3.2 h overrun
in an earlier round (decision log 2026-08-15T07:45Z) and it is not needed for the claim.

Run:  python -m experiments.run_operating_point_full_features

Graduated from tmp/ because the appendix reports its numbers: full-feature detection at a
1% false-alarm budget (LightGBM 0.652, random forest 0.663) and the false-alarm flood under
transferred PGD (0.007/0.002/0.014 to 0.920/0.940/0.924). The RunDir name keeps its original
`probe_` prefix so already-recorded results still resolve.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score

from experiments.run_review_claims import floors, tpr_at_fpr
from qgridbench.attacks.evasion import pgd
from qgridbench.attacks.gradients import GradProvider
from qgridbench.models.classical.zoo import build, fit_model, predict_proba_timed, stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)

SEEDS = [0, 1, 2]
EPS = 0.5
MODELS = ["logreg", "rf", "xgb", "lgbm", "mlp"]
FPR_TARGETS = [0.01, 0.05]


def main():
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    src = json.loads(
        (REPO_ROOT / "results/runs/merged_accuracy_binary_full/results.json").read_text()
    )
    best = {m: src[m]["best_params"] for m in MODELS}

    reg = prepare_regime("binary", "full", seed=0)
    scale = reg.feature_scale
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    va = stratified_cap(reg.X["val"], reg.y["val"], 5000, seed=0)
    Xte, yte = reg.X["test"][ev], reg.y["test"][ev]
    Xva, yva = reg.X["val"][va], reg.y["val"][va]
    fl = floors(reg.y["train"], yte)["stratified_random"]

    with RunDir("probe_operating_point_full_binary", config={"eps": EPS}, seeds=SEEDS) as run:
        per_seed = []
        for s in SEEDS:
            rng = spawn_rng(s, "opfull")
            fitted = {}
            for m in MODELS:
                set_all_seeds(s)
                est = build(m, best[m], s)
                fit_model(m, est, reg.X["train"], reg.y["train"])
                fitted[m] = est
            log.info("seed %d | models fitted", s)

            def proba(m, X):
                return predict_proba_timed(fitted[m], X)[0]

            gp = GradProvider(fitted["mlp"], "mlp")
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
            sets = {"clean": Xte, "pgd_mlp": Xadv}

            rec = {"f1": {}, "op": {}}
            for m in MODELS:
                rec["f1"][m] = {
                    k: float(f1_score(yte, proba(m, X).argmax(1), average="macro"))
                    for k, X in sets.items()
                }
                sv = proba(m, Xva)[:, 1]
                rec["op"][m] = {
                    str(t): {
                        k: tpr_at_fpr(sv, yva, proba(m, X)[:, 1], yte, t) for k, X in sets.items()
                    }
                    for t in FPR_TARGETS
                }
            per_seed.append(rec)
            run.write_json("results_partial.json", {"per_seed": per_seed})
            log.info(
                "seed %d | clean F1 %s", s, {m: round(rec["f1"][m]["clean"], 3) for m in MODELS}
            )

        def mn(f):
            return float(np.mean([f(p) for p in per_seed]))

        summary = {"floor": fl}
        for m in MODELS:
            summary[m] = {
                "clean_f1": mn(lambda p, m=m: p["f1"][m]["clean"]),
                "adv_f1": mn(lambda p, m=m: p["f1"][m]["pgd_mlp"]),
                **{
                    f"tpr@{t}_clean": mn(lambda p, m=m, t=t: p["op"][m][str(t)]["clean"]["tpr"])
                    for t in FPR_TARGETS
                },
                **{
                    f"tpr@{t}_adv": mn(lambda p, m=m, t=t: p["op"][m][str(t)]["pgd_mlp"]["tpr"])
                    for t in FPR_TARGETS
                },
                **{
                    f"fpr@{t}_clean": mn(lambda p, m=m, t=t: p["op"][m][str(t)]["clean"]["fpr"])
                    for t in FPR_TARGETS
                },
                **{
                    f"fpr@{t}_adv": mn(lambda p, m=m, t=t: p["op"][m][str(t)]["pgd_mlp"]["fpr"])
                    for t in FPR_TARGETS
                },
            }
        run.write_json("results.json", {"per_seed": per_seed, "summary": summary})
        print(
            f"\n{'model':<8}{'cleanF1':>9}{'advF1':>8}{'TPR@1%':>9}{'TPR@1%adv':>11}"
            f"{'FPR@1%':>9}{'FPR@1%adv':>11}"
        )
        for m in MODELS:
            v = summary[m]
            print(
                f"{m:<8}{v['clean_f1']:>9.3f}{v['adv_f1']:>8.3f}{v['tpr@0.01_clean']:>9.3f}"
                f"{v['tpr@0.01_adv']:>11.3f}{v['fpr@0.01_clean']:>9.3f}{v['fpr@0.01_adv']:>11.3f}"
            )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
