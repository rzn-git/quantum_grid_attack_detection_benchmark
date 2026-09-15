"""Worked example: reproduce the published 0.905 / 0.594 result through the audit tool.

Verifies the tool against the QuantumGridBench headline (arXiv:2608.15617) before it is
trusted on new data: tuned LightGBM on the MSU/ORNL power system attack data (binary variant)
scores ~0.905 macro-F1 under the shipped row-level split and ~0.594 mean under five whole-file
partitions, with the neighbour audit explaining the gap (45.8% of test rows within 0.1 of a
train row under the row split, ~5.0% under whole-file holdout).

Every expected value below is the full-precision figure from the archived control artifact
`results/runs/20260816T023620Z_group_split_control_binary/results.json`; the example FAILS
LOUD if any reproduction drifts past its tolerance.

Requires the project's processed data and the tuned LightGBM Optuna study
(`results/optuna_cls_lgbm_binary_full.db`) — run the pipeline through the classical tuning
stage first if they are missing.

Run from the repository root:
    python -m tools.leakage_audit.examples.reproduce_msu_ornl
"""

from __future__ import annotations

import json

from experiments.run_group_split_control import reconstruct_groups, tuned_params
from qgridbench.data.preprocess import load_processed
from qgridbench.models.classical.zoo import build
from tools.leakage_audit import paired_protocol

# Archived control artifact 20260816T023620Z_group_split_control_binary (full precision).
EXPECTED = {
    "row_score": 0.9053182941463003,
    "row_nn": 0.45801965037641956,
    "group_scores": {
        0: 0.6032023226950904,
        1: 0.5837062936031531,
        2: 0.6033207159131841,
        3: 0.593682888509598,
        4: 0.5884017615020977,
    },
    "group_nn_mean": 0.05018,
    "held_out_files": {0: [1, 8, 13], 1: [3, 7, 9], 2: [1, 10, 11], 3: [5, 9, 14], 4: [1, 11, 12]},
}
SCORE_TOL = 1e-6  # the fits are deterministic; anything larger is a code-path divergence
NN_TOL = 1e-6


def main() -> None:
    lgbm_params, _ = tuned_params()
    X, y, gid = reconstruct_groups()
    _, _, splits_row = load_processed("binary")

    result = paired_protocol(
        build("lgbm", lgbm_params, seed=0),
        X,
        y,
        gid,
        n_partitions=5,
        seed=0,
        row_split_indices=splits_row,
    )
    print(json.dumps(result, indent=2))

    failures = []

    def check(name: str, got: float, want: float, tol: float) -> None:
        status = "ok" if abs(got - want) <= tol else "FAIL"
        print(f"{status:4s} {name}: got {got:.6f}, expected {want:.6f} (tol {tol:g})")
        if status == "FAIL":
            failures.append(name)

    check("row-level macro-F1", result["row_level"]["score"], EXPECTED["row_score"], SCORE_TOL)
    check(
        "row-level nn<0.1 fraction",
        result["row_level"]["nn_frac_within_threshold"],
        EXPECTED["row_nn"],
        NN_TOL,
    )
    for p in result["whole_group"]["partitions"]:
        i = p["partition"]
        if p["held_out_groups"] != EXPECTED["held_out_files"][i]:
            failures.append(f"partition {i} held-out files")
            print(
                f"FAIL partition {i} held-out files: got {p['held_out_groups']}, "
                f"expected {EXPECTED['held_out_files'][i]}"
            )
        check(f"whole-file p{i} macro-F1", p["score"], EXPECTED["group_scores"][i], SCORE_TOL)
    check(
        "whole-file nn<0.1 fraction (mean)",
        result["whole_group"]["nn_frac_within_threshold_mean"],
        EXPECTED["group_nn_mean"],
        1e-4,
    )

    delta = result["delta_row_minus_group"]
    print(
        f"\nrow-level {result['row_level']['score']:.4f}  vs  whole-file "
        f"{result['whole_group']['score_mean']:.4f} ± {result['whole_group']['score_std']:.4f}"
        f"  ->  delta {delta:.4f}"
    )
    if failures:
        raise SystemExit(f"REPRODUCTION FAILED: {failures}")
    print("PASS: the tool reproduces the published row-level and whole-file results.")


if __name__ == "__main__":
    main()
