"""P22: is the QSVM's 2.818 sigma HopSkipJump distortion restart-robust?

External review: "HSJ's outcome depends on its random initialization; with interleaved
decision regions (your own finding), restart variance could be large. A few restarts on
the QSVM cells makes the 'attack is fooled' claim restart-robust."

The published black-box table is one HSJ run per family at np.random.seed(0)
(attacks/blackbox.py). HSJ starts from a random adversarial point and binary-searches
back, so the initialisation is a real degree of freedom -- and the paper's own mechanism
claim (decision regions interleave at fine scale, so the line search terminates far away)
predicts that a different start could terminate somewhere else entirely.

H: the QSVM's median L-inf distortion varies materially across HSJ initialisations.
PRE-REGISTERED KILL CRITERION, decided before looking:
  - restart-robust if the QSVM's median distortion has coefficient of variation <= 0.10
    across restarts AND the QSVM stays the highest-distortion family in every restart;
    the 2.818 sigma figure then stands as published.
  - otherwise the number must be reported as mean +- std over restarts, and the
    "hardest target" wording scoped to the restart budget.
CONTROLS: the MLP (0.867 sigma) and RBF-SVM (0.000 sigma) get the identical treatment,
so restart variance specific to the fidelity kernel is separable from HSJ's variance in
general.

Run: python -m experiments.run_hopskipjump_restarts [n_restarts] [family ...]

Graduated from tmp/ because the paper reports its numbers: the abstract's "an unseeded
black-box attack moves 75% between restarts" and the Results section's 2.808-4.917 sigma
range both come from this measurement, so the producer has to be a file the repository
carries. The RunDir name is unchanged, so the existing run directories still match.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from experiments.run_adversarial import FittedZoo
from qgridbench.attacks.blackbox import hopskipjump_attack
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml

log = get_logger(__name__)

DEFAULT_FAMILIES = ["qsvm", "mlp", "rbf_svm"]
ENCODING = "zz"
SEED_FIT = 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "n_restarts",
        nargs="?",
        type=int,
        default=5,
        help="HopSkipJump restarts per family (default 5)",
    )
    ap.add_argument(
        "families",
        nargs="*",
        default=DEFAULT_FAMILIES,
        help=f"families to attack (default: {' '.join(DEFAULT_FAMILIES)})",
    )
    args = ap.parse_args()
    n_restarts = args.n_restarts
    families = args.families or DEFAULT_FAMILIES
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    bb = cfg_adv["blackbox"]
    best_params = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())

    reg = prepare_regime("binary", "pca", n_components=8, seed=0)
    scale = reg.feature_scale
    bb_idx = stratified_cap(reg.X["test"], reg.y["test"], bb["eval_subset"], seed=0)
    Xc = reg.X["test"][bb_idx]

    zoo = FittedZoo(reg, ENCODING, 8, SEED_FIT, cfg_q["qkernel"]["subset_cap"], best_params)

    with RunDir(
        "probe_hsj_restarts_binary_q8",
        config={"n_restarts": n_restarts, "families": families, **bb},
        seeds=list(range(n_restarts)),
    ) as run:
        out = {f: [] for f in families}
        for f in families:
            for r in range(n_restarts):
                t0 = time.perf_counter()
                Xa, budget = hopskipjump_attack(
                    lambda z, _f=f: zoo.predict_proba(_f, z),
                    Xc,
                    zoo.n_classes,
                    max_iter=bb["max_iter"],
                    max_eval=bb["max_eval"],
                    init_eval=bb["init_eval"],
                    seed=r,  # THE variable: ART's random start
                )
                dist = (np.abs(Xa - Xc) / scale).max(axis=1)
                rec = {
                    "restart": r,
                    "median_linf_sigma": float(np.median(dist)),
                    "mean_linf_sigma": float(dist.mean()),
                    "queries_per_sample": budget["queries_per_sample"],
                    "seconds": round(time.perf_counter() - t0, 1),
                }
                out[f].append(rec)
                run.write_json("results_partial.json", out)
                log.info(
                    "%s restart %d -> median %.3f sigma (%.0f s)",
                    f,
                    r,
                    rec["median_linf_sigma"],
                    rec["seconds"],
                )

        summary = {}
        for f in families:
            m = np.array([r["median_linf_sigma"] for r in out[f]])
            summary[f] = {
                "mean": float(m.mean()),
                "std": float(m.std(ddof=1)),
                "min": float(m.min()),
                "max": float(m.max()),
                "cv": float(m.std(ddof=1) / m.mean()) if m.mean() else float("nan"),
            }
        # does the QSVM stay the hardest target in every restart?
        ordering_holds = (
            all(
                out["qsvm"][r]["median_linf_sigma"]
                > max(out[f][r]["median_linf_sigma"] for f in families if f != "qsvm")
                for r in range(n_restarts)
            )
            if "qsvm" in families and len(families) > 1
            else None
        )
        verdict = (
            "RESTART-ROBUST — published figure stands"
            if "qsvm" in summary and summary["qsvm"]["cv"] <= 0.10 and ordering_holds
            else "NOT RESTART-ROBUST — report mean +- std and scope the wording"
        )
        run.write_json(
            "results.json",
            {
                "per_family": out,
                "summary": summary,
                "qsvm_highest_every_restart": ordering_holds,
                "verdict": verdict,
                "published_qsvm": 2.8185,
            },
        )
        log.info("VERDICT: %s | %s", verdict, json.dumps(summary))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
