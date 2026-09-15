"""E10 pilot — project the wall-clock of a 5x HopSkipJump budget before committing to it.

PRE-REGISTRATION (pass-4 workplan item E10; the pilot is MANDATORY there).
The shipped black-box budget is ~600 queries/sample (max_iter 10, max_eval 200, init_eval 20),
one to two orders below HopSkipJump's design regime, and it was set by measured DEFENDER
LATENCY rather than by attack convergence. The open question is whether the QSVM's large
reported distortion is decision geometry or budget starvation. Before spending days on that,
this pilot measures the actual cost on a few points and extrapolates to the full arm
(100 points x 5 restarts x 2 kernel machines).

This run makes NO scientific claim. Its output is a projection the operator can act on, and
a same-budget sanity check that the measured distortion is in the published range.

Run:  python -m experiments.run_hsj_budget_pilot --n-points 5
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

FULL_POINTS = 100
FULL_RESTARTS = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="angle_ry")  # the published black-box config
    ap.add_argument("--n-points", type=int, default=5)
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
    X, y = reg.X["test"][idx][: args.n_points], reg.y["test"][idx][: args.n_points]
    scale = reg.feature_scale
    log.info("pilot on %d points, families %s", len(y), args.families)

    budgets = {
        "shipped": dict(
            max_iter=bb["max_iter"], max_eval=bb["max_eval"], init_eval=bb["init_eval"]
        ),
        "5x": dict(
            max_iter=bb["max_iter"] * 5, max_eval=bb["max_eval"] * 5, init_eval=bb["init_eval"]
        ),
    }

    cfg = {
        "variant": args.variant,
        "qubits": args.qubits,
        "encoding": args.encoding,
        "n_points_pilot": len(y),
        "families": args.families,
        "budgets": budgets,
        "projection_target": {"points": FULL_POINTS, "restarts": FULL_RESTARTS},
        "purpose": "wall-clock projection only; no scientific claim",
    }

    with RunDir(f"hsj_budget_pilot_{args.variant}_q{args.qubits}", config=cfg) as run:
        out = {}
        for fam in args.families:
            out[fam] = {}
            for name, b in budgets.items():
                t0 = time.perf_counter()
                Xa, budget = hopskipjump_attack(
                    lambda z, _f=fam: zoo.predict_proba(_f, z), X, zoo.n_classes, **b
                )
                dt = time.perf_counter() - t0
                dist = (np.abs(Xa - X) / scale).max(axis=1)
                out[fam][name] = {
                    "seconds_total": dt,
                    "seconds_per_point": dt / len(y),
                    "queries_per_sample": budget["queries_per_sample"],
                    "median_linf_distortion": float(np.median(dist)),
                    "adv_f1": _f1(y, zoo.predict_proba(fam, Xa)),
                    "projected_hours_full_arm": dt / len(y) * FULL_POINTS * FULL_RESTARTS / 3600.0,
                }
                log.info(
                    "%-8s %-8s | %.1f s/point | %.0f q/sample | median Linf %.3f | "
                    "projected %.2f h for %d pts x %d restarts",
                    fam,
                    name,
                    out[fam][name]["seconds_per_point"],
                    out[fam][name]["queries_per_sample"],
                    out[fam][name]["median_linf_distortion"],
                    out[fam][name]["projected_hours_full_arm"],
                    FULL_POINTS,
                    FULL_RESTARTS,
                )
                run.write_json("results_partial.json", {"pilot": out})

        total_h = sum(out[f]["5x"]["projected_hours_full_arm"] for f in args.families)
        verdict = {
            "projected_hours_5x_both_families": total_h,
            "projected_hours_shipped_both_families": sum(
                out[f]["shipped"]["projected_hours_full_arm"] for f in args.families
            ),
            "recommendation": (
                "run" if total_h <= 12 else ("reduce to 50 points" if total_h <= 48 else "escalate")
            ),
            "note": "projection is linear in points x restarts; HSJ cost is dominated by "
            "defender query latency, which differs ~1000x across families",
        }
        run.write_json("results.json", {"pilot": out, "verdict": verdict})
        run.metrics = {"projected_hours_5x": total_h}
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
