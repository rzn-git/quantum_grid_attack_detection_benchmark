"""E10 — does the HopSkipJump budget, not decision geometry, produce the QSVM's large distortion?

PRE-REGISTRATION (pass-4 workplan item E10; frozen before the run).
The shipped black-box budget is ~600 queries/sample, one to two orders below HopSkipJump's
design regime (ART defaults: max_iter 50, max_eval 10,000), and it was set by measured
DEFENDER LATENCY rather than by attack convergence. The paper attributes the fidelity
kernel's large median distortion (2.818 sigma published, restart mean 3.920 +/- 1.035) to
decision regions that interleave at fine scale, and supports that with a budget-independent
measurement (26% of rays cross more than one boundary against the classical kernel's 17% at
matched accuracy optima). What is unquantified is how much of the distortion is simply
starvation.

DECISION RULE on the QSVM's restart-mean median L-inf distortion at the 5x budget:
  stays >= 2.0 sigma while the RBF stays <= 0.9 sigma -> the geometry attribution stands
  falls <= 1.0 sigma                                  -> the checklist row is reworded
                                                         "query budget + measurement geometry"
  otherwise                                           -> report the budget-dependence as measured

CONTROL (must pass before the rule is read): the SHIPPED-budget arm, re-run here over
restarts, must land inside the published restart range for the QSVM (2.808--4.917 sigma). If
it does not, this runner differs from the published one and no budget comparison is licensed.

HopSkipJump is NOT reproducible from a seed (ART draws its initial adversarial point from an
unseeded generator -- the paper's own finding), so every figure here is a restart mean +/- std
and never a single draw.

Cost was MEASURED before committing (run_hsj_budget_pilot, 2026-08-22): ~1.2 h for both
families at the 5x budget over 100 points x 5 restarts, against the workplan's "days" guess.

Run:  python -m experiments.run_hsj_budget_arm
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from experiments.run_adversarial import FittedZoo, _f1
from qgridbench.attacks.blackbox import hopskipjump_attack
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml

log = get_logger(__name__)

PUBLISHED_QSVM_RESTART_RANGE = (2.808, 4.917)
PUBLISHED_QSVM_RESTART_MEAN = 3.920


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--qubits", type=int, default=8)
    # The published 2.818 sigma figure comes from the ZZ run with TUNED parameters
    # (results/runs/20260813T183453Z_adversarial_binary_q8_blackbox: encoding zz,
    # best_params adv_best_params_binary_q8.json). A sibling artifact of the same date
    # ran angle_ry with best_params null and is NOT the published cell — matching it
    # gives median distortion 0.165 instead of 2.818, which is how this was caught.
    ap.add_argument("--encoding", default="zz")
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--budget-mult", type=int, default=5)
    ap.add_argument("--families", nargs="*", default=["rbf_svm", "qsvm"])
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    subset_cap = cfg_q["qkernel"]["subset_cap"]
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    bb = cfg_adv["blackbox"]

    reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
    zoo = FittedZoo(reg, args.encoding, args.qubits, 0, subset_cap, best)
    idx = stratified_cap(reg.X["test"], reg.y["test"], bb["eval_subset"], seed=0)
    X, y = reg.X["test"][idx], reg.y["test"][idx]
    scale = reg.feature_scale
    m = args.budget_mult

    budgets = {
        "shipped": dict(
            max_iter=bb["max_iter"], max_eval=bb["max_eval"], init_eval=bb["init_eval"]
        ),
        f"{m}x": dict(
            max_iter=bb["max_iter"] * m, max_eval=bb["max_eval"] * m, init_eval=bb["init_eval"]
        ),
    }

    cfg = {
        "variant": args.variant,
        "qubits": args.qubits,
        "encoding": args.encoding,
        "n_eval": int(len(y)),
        "restarts": args.restarts,
        "budgets": budgets,
        "published_qsvm_restart_range": list(PUBLISHED_QSVM_RESTART_RANGE),
        "note": "HSJ is unseeded by construction; every figure is a restart mean +/- std",
    }

    with RunDir(f"hsj_budget_arm_{args.variant}_q{args.qubits}", config=cfg) as run:
        out: dict = {}
        for fam in args.families:
            out[fam] = {}
            for name, b in budgets.items():
                meds, advs, qs = [], [], []
                t0 = time.perf_counter()
                for r in range(args.restarts):
                    Xa, budget = hopskipjump_attack(
                        lambda z, _f=fam: zoo.predict_proba(_f, z), X, zoo.n_classes, **b
                    )
                    dist = (np.abs(Xa - X) / scale).max(axis=1)
                    meds.append(float(np.median(dist)))
                    advs.append(_f1(y, zoo.predict_proba(fam, Xa)))
                    qs.append(budget["queries_per_sample"])
                    run.write_json(
                        "results_partial.json",
                        {"family": fam, "budget": name, "restarts_done": r + 1, "medians": meds},
                    )
                out[fam][name] = {
                    "median_linf_mean": float(np.mean(meds)),
                    "median_linf_std": float(np.std(meds, ddof=1)),
                    "median_linf_per_restart": meds,
                    "adv_f1_mean": float(np.mean(advs)),
                    "queries_per_sample": float(np.mean(qs)),
                    "cv": float(np.std(meds, ddof=1) / max(np.mean(meds), 1e-12)),
                    "wall_s": time.perf_counter() - t0,
                }
                log.info(
                    "%-8s %-8s | median Linf %.3f +/- %.3f (CV %.2f) | %.0f q/sample | %.0f s",
                    fam,
                    name,
                    out[fam][name]["median_linf_mean"],
                    out[fam][name]["median_linf_std"],
                    out[fam][name]["cv"],
                    out[fam][name]["queries_per_sample"],
                    out[fam][name]["wall_s"],
                )

        q_ship = out["qsvm"]["shipped"]["median_linf_mean"]
        q_big = out["qsvm"][f"{m}x"]["median_linf_mean"]
        r_big = out["rbf_svm"][f"{m}x"]["median_linf_mean"]
        lo, hi = PUBLISHED_QSVM_RESTART_RANGE
        control_pass = bool(lo <= q_ship <= hi)
        branch = (
            "geometry_attribution_stands"
            if (q_big >= 2.0 and r_big <= 0.9)
            else ("reword_budget_plus_geometry" if q_big <= 1.0 else "report_budget_dependence")
        )
        verdict = {
            "control_shipped_qsvm_mean": q_ship,
            "control_published_range": list(PUBLISHED_QSVM_RESTART_RANGE),
            "control_pass": control_pass,
            "qsvm_shipped": q_ship,
            "qsvm_big_budget": q_big,
            "rbf_big_budget": r_big,
            "qsvm_over_rbf_at_big_budget": float(q_big / max(r_big, 1e-12)),
            "qsvm_budget_ratio": float(q_big / max(q_ship, 1e-12)),
            "branch": branch,
            "rule": "qsvm >= 2.0 and rbf <= 0.9 -> geometry stands; qsvm <= 1.0 -> reword",
        }
        run.write_json("results.json", {"per_family": out, "verdict": verdict})
        run.metrics = {"qsvm_big_budget": q_big, "rbf_big_budget": r_big}
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
