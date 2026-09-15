"""E6 — label-flip poisoning across seeds: put a spread on a one-seed near-null.

PRE-REGISTRATION (pass-4 workplan item E6; frozen before the run).
RQ3 states "Label-flip poisoning up to 20% moves no family more than 0.03 macro-F1 except
the MLP". That figure comes from ONE seed (`run_adversarial.poisoning` retrains at
`seeds[0]`), which contravenes the project's own no-number-without-uncertainty rule in the
one section that asserts a null.

DECISION RULE, per family: if std <= 0.01 across seeds AND max |delta macro-F1| <= 0.03
(the MLP is exempt, as published), the claim keeps and is restated with +/- std; otherwise
the sentence is requoted with the measured spread. A v2 BACKLOG row is owed either way,
because the published sentence carries no uncertainty at all.

CONTROL (must pass before the rule is read): seed 0 must reproduce the published per-family
deltas — exactly for the deterministic families, within the recorded thread sensitivity for
the histogram-boosting ones (decision log 2026-08-15T20:09Z / 2026-08-20T14:22Z).

SCOPE NOTE: the shipped `poisoning()` takes one `seed` that drives BOTH the flip pattern
(`spawn_rng(seed, f"poison_{rate}")`) and the model initialization (FittedZoo). Varying it
therefore reports the JOINT spread over flip draw and fit, which is the honest quantity for
"how stable is this null" — it is not decomposed here, and the artifact says so.

Run:  python -m experiments.run_poisoning_seeds --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.run_adversarial import CLASSICAL, poisoning
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml

log = get_logger(__name__)

FAMILIES = CLASSICAL + ["qsvm", "vqc"]
STD_BAR = 0.01
DELTA_BAR = 0.03
EXEMPT = {"mlp"}  # published exception


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="angle_ry")  # the shipped adversarial-suite encoding
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--reaggregate-from",
        default=None,
        help="run dir holding results_partial.json; re-aggregates WITHOUT refitting "
        "(the fits are the expensive part and are already archived)",
    )
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    subset_cap = cfg_q["qkernel"]["subset_cap"]
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())

    reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)

    cfg = {
        "variant": args.variant,
        "qubits": args.qubits,
        "encoding": args.encoding,
        "seeds": args.seeds,
        "rates": cfg_adv["poisoning"]["label_flip_rates"],
        "std_bar": STD_BAR,
        "delta_bar": DELTA_BAR,
        "scope": "seed drives BOTH flip pattern and model init; joint spread reported",
    }

    with RunDir(
        f"poisoning_seeds_{args.variant}_q{args.qubits}", config=cfg, seeds=args.seeds
    ) as run:
        per_seed = {}
        if args.reaggregate_from:
            src = REPO_ROOT / args.reaggregate_from / "results_partial.json"
            per_seed = json.loads(src.read_text())["per_seed"]
            log.info("re-aggregating from %s (no refits): seeds %s", src, sorted(per_seed))
        else:
            for s in args.seeds:
                out = poisoning(reg, args.encoding, args.qubits, cfg_adv, subset_cap, best, s)
                per_seed[str(s)] = out
                run.write_json("results_partial.json", {"per_seed": per_seed})
                log.info(
                    "seed %d done | rates: %s",
                    s,
                    {k: round(v.get("lgbm", float("nan")), 4) for k, v in out.items()},
                )

        # rate keys are floats in memory but strings after a JSON round-trip — normalize
        per_seed = {s: {str(float(r)): v for r, v in d.items()} for s, d in per_seed.items()}
        seed_keys = sorted(per_seed, key=int)

        # ---- aggregate: degradation from each seed's own 0.0-rate reference -----------
        rates = [r for r in per_seed[seed_keys[0]] if float(r) > 0]
        summary = {}
        for fam in FAMILIES:
            fam_out = {}
            for r in rates:
                deltas = [per_seed[s][r][fam] - per_seed[s]["0.0"][fam] for s in seed_keys]
                fam_out[str(r)] = {
                    "delta_mean": float(np.mean(deltas)),
                    "delta_std": float(np.std(deltas, ddof=1)),
                    "abs_delta_mean": float(np.mean(np.abs(deltas))),
                    "per_seed": [float(v) for v in deltas],
                }
            worst = max(fam_out.values(), key=lambda d: abs(d["delta_mean"]))
            max_std = max(d["delta_std"] for d in fam_out.values())
            fam_out["_verdict"] = {
                "max_abs_delta_mean": float(abs(worst["delta_mean"])),
                "max_std": float(max_std),
                "within_published_claim": bool(
                    abs(worst["delta_mean"]) <= DELTA_BAR and max_std <= STD_BAR
                ),
                "exempt": fam in EXEMPT,
            }
            summary[fam] = fam_out
            log.info(
                "%-8s | max |delta| %.4f, max std %.4f -> within claim=%s%s",
                fam,
                abs(worst["delta_mean"]),
                max_std,
                fam_out["_verdict"]["within_published_claim"],
                " (exempt)" if fam in EXEMPT else "",
            )

        violators = [
            f
            for f in FAMILIES
            if f not in EXEMPT and not summary[f]["_verdict"]["within_published_claim"]
        ]
        verdict = {
            "violating_families": violators,
            "branch": ("claim_keeps_with_std" if not violators else "requote_with_spread"),
            "rule": f"per family: std <= {STD_BAR} AND max |delta| <= {DELTA_BAR} (mlp exempt)",
            "n_seeds": len(seed_keys),
        }
        run.write_json(
            "results.json", {"per_seed": per_seed, "summary": summary, "verdict": verdict}
        )
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
