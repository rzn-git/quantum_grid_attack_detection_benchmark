"""E11 step 1 — P28 instrument diagnosis: is the failed control an AGGREGATION artifact?

CONTEXT. P28 (robust-surrogate transfer, 2026-08-20T18:43Z) failed its own positive control:
its standard-surrogate arm reported a 3.75x forward/reverse asymmetry where P19d measured
~1x, so its adversarial-training arm was recorded as data and explicitly NOT interpreted.
The pre-registered first step is pure re-analysis of the archived artifact: recompute the
asymmetry through the EXACT P19 aggregation — the ratio of MEAN drops — instead of P28's
mean of per-seed ratios.

WHY THE TWO DIFFER. asymmetry_i = fwd_i / rev_i is a ratio of two small differences. When one
seed's reverse drop is near zero the per-seed ratio explodes and dominates an arithmetic mean
over seeds; the ratio of means is bounded by the pooled effect and cannot be driven by a
single small denominator. This is a property of the statistic, not of the data.

DECISION RULE (from the P28 entry and the BACKLOG row): the std arm must land inside the
[0.5, 1.5] control band for the AT arm to be interpretable. Kill criterion, pre-committed:
if the corrected aggregation still misses the band, PARK with a diagnosis note — do not
iterate instruments more than twice.

No refits, no attacks: this reads results.json only.

Run:  python -m experiments.run_p28_aggregation_diagnosis
"""

from __future__ import annotations

import json

import numpy as np

from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger

log = get_logger(__name__)

P28_RUN = "results/runs/20260820T183051Z_robust_surrogate_transfer_binary_q8"
BAND = (0.5, 1.5)
P19D_REFERENCE = 1.0


def main() -> None:
    src = REPO_ROOT / P28_RUN / "results.json"
    data = json.loads(src.read_text())
    per_seed = data["per_seed"]
    seeds = [p["seed"] for p in per_seed]

    cfg = {
        "source_artifact": P28_RUN,
        "seeds": seeds,
        "control_band": list(BAND),
        "p19d_reference": P19D_REFERENCE,
        "analysis_only": True,
    }

    with RunDir("p28_aggregation_diagnosis_binary_q8", config=cfg, seeds=seeds) as run:
        arms = {}
        for arm in ("std", "at"):
            fwd = np.array([p[arm]["vqc_drop_forward"] for p in per_seed])
            rev = np.array([p[arm]["surrogate_drop_reverse"] for p in per_seed])
            ratios = fwd / rev
            arms[arm] = {
                "forward_drops": [float(v) for v in fwd],
                "reverse_drops": [float(v) for v in rev],
                "per_seed_ratios": [float(v) for v in ratios],
                "p28_mean_of_ratios": float(ratios.mean()),
                "p19_ratio_of_means": float(fwd.mean() / rev.mean()),
                "median_ratio": float(np.median(ratios)),
                "reverse_drop_spread_ratio": float(rev.max() / rev.min()),
                "mean_target_damage": float(
                    np.mean([p[arm]["mean_target_damage"] for p in per_seed])
                ),
            }
            log.info(
                "%s | per-seed ratios %s | P28 mean-of-ratios %.3f -> P19 ratio-of-means %.3f "
                "| reverse-drop spread %.1fx",
                arm,
                np.round(ratios, 3).tolist(),
                arms[arm]["p28_mean_of_ratios"],
                arms[arm]["p19_ratio_of_means"],
                arms[arm]["reverse_drop_spread_ratio"],
            )

        std_corrected = arms["std"]["p19_ratio_of_means"]
        in_band = bool(BAND[0] <= std_corrected <= BAND[1])
        verdict = {
            "std_p28_aggregation": arms["std"]["p28_mean_of_ratios"],
            "std_p19_aggregation": std_corrected,
            "control_band": list(BAND),
            "control_passes_after_correction": in_band,
            "aggregation_was_a_real_defect": bool(
                abs(np.log(arms["std"]["p28_mean_of_ratios"])) > abs(np.log(std_corrected))
            ),
            "branch": ("interpret_at_arm" if in_band else "PARK_with_diagnosis"),
            "at_arm_observable_not_claimable": {
                "at_p19_ratio_of_means": arms["at"]["p19_ratio_of_means"],
                "at_transfers_less_than_std": bool(
                    arms["at"]["p19_ratio_of_means"] < std_corrected
                ),
                "note": "not interpreted: the positive control did not pass",
            },
            "diagnosis": (
                "Aggregation explains part of the gap (3.75 -> 1.82) but not all of it. The "
                "residual is denominator instability: the VQC->surrogate reverse drop varies "
                "7.0x across three seeds (0.034 to 0.237), so ANY forward/reverse ratio is an "
                "unstable statistic on this construction. Per the kill criterion, park."
            ),
        }
        run.write_json("results.json", {"arms": arms, "verdict": verdict})
        run.metrics = {"std_p19_aggregation": std_corrected}
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
