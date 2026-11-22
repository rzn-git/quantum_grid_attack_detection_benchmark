"""Does the regularization still change the QSVM's failure once the attack is read inside the
ball? The verdict of the sweep registered in the decision log at 2026-11-20T12:16Z (archive-only).

P11c part B read each value of C at one radius, the full budget, under the distance-seeking
objective, and found the kernel constant at C of 10 or less. Stage p11e of
experiments/run_whitebox_qsvm.py attacked the same fits at every grid radius under both
objectives (one run per value of C). This reads those runs, attacks nothing, and writes one run:

  - per C and objective, the reading at the budget and at the worst radius inside it, with
    the end-state label at every radius (experiments.read_worst_radius.read_sweep);
  - per C, the worst reading over both objectives, because the attacker picks the objective
    as well as the radius, beside the random control at that radius and the reading of an
    attacker who picks the radius per input;
  - the registered branch. STANDS UNDER R: at every C of 10 or less the worst reading is not
    inverted, at the tuned C it is, and the tuned C reads at least 0.05 macro-F1 lower in the
    mean and in at least 9 of the 10 attack starts. FALLS UNDER R: every C reads inverted at
    its worst radius and at three radii or more. MIXED: anything else.

Two stored runs anchor it, or the run fails: mass descent at the budget over part B's attack
seeds must reproduce part B, and loss ascent at the tuned C over P9a's attack seeds must
reproduce the tuned-bandwidth cell of P9a.

Run: python -m experiments.read_regularization_radii --runs <run> <run> <run> <run>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.read_worst_radius import RUNS, _anchor, _json, read_sweep
from qgridbench.utils.run_tracking import RunDir, get_logger, load_yaml

log = get_logger(__name__)

BLOCK = "p11e_regularization_radii"
PART_B_RUN = "20260814T200349Z_whitebox_qsvm_binary_q8"
P9A_RUN = "20260814T180008Z_whitebox_qsvm_binary_q8"
SHARED = (
    "constant_predictor_floor",
    "n_eval",
    "budget_sigma",
    "radii",
    "n_steps",
    "tuned_bandwidth",
    "tuned_C",
    "c_grid",
    "attack_seeds",
)
AXES = ("per_c", "per_c_loss", "per_c_mass", "per_c_random")
OBJECTIVES = ("loss", "mass")
# the paper's claim is about C of 10 or less; the bars are the ones registered for the kernel
# contrast with whole files held out (0.05 in the mean and in 9 of 10 starts) and the
# persistence rule of the end-state sweeps (a label holds over at least three radii)
SMALL_C_MAX = 10.0
GAP_MIN, SEEDS_MIN, MIN_RADII = 0.05, 9, 3
# anchors: two rows of the 200 in macro-F1; the stored spread of the decision-spread ratio
ANCHOR_F1, ANCHOR_SPREAD = 0.01, 0.02


def merge(blocks: list[dict]) -> dict:
    """One block from the per-C runs: shared fields agree, every C of the grid appears once."""
    first = blocks[0]
    out = {k: first[k] for k in SHARED} | {a: {} for a in AXES}
    for b in blocks:
        differing = [k for k in SHARED if b[k] != first[k]]
        if differing:
            raise ValueError(f"the runs disagree on {differing}")
        for a in AXES:
            for C, v in b[a].items():
                if C in out[a]:
                    raise ValueError(f"C {C} appears in two runs")
                out[a][C] = v
    got = sorted(float(C) for C in out["per_c"])
    if got != [float(c) for c in first["c_grid"]]:
        raise ValueError(f"the runs cover C {got}, the grid is {first['c_grid']}")
    return out


def check_anchors(
    block: dict, part_b: dict, p9a: dict, p9a_seeds: list[int], anchors: list
) -> None:
    """The two stored runs this sweep must reproduce where the streams coincide."""
    budget = next(
        r
        for r in block["per_c_mass"][str(block["tuned_C"])]["curve"]
        if float(r) == block["budget_sigma"]
    )
    for C, stored in part_b.items():
        rows = {r["seed"]: r for r in block["per_c_mass"][C]["curve"][budget]["per_seed"]}
        seeds = [p["seed"] for p in stored["per_seed"]]
        for key, tol in (("dec_spread_ratio", ANCHOR_SPREAD), ("f1_minus_floor", ANCHOR_F1)):
            _anchor(
                anchors,
                f"mass descent at the budget, C {C}, {key}, starts {seeds}",
                float(np.mean([rows[s][key] for s in seeds])),
                stored["mean"][key]["mean"],
                tol,
            )
        _anchor(
            anchors,
            f"clean macro-F1, C {C}",
            block["per_c"][C]["clean_macro_f1"],
            stored["clean_macro_f1"],
            ANCHOR_F1,
        )
    cell = p9a["per_bandwidth"][str(block["tuned_bandwidth"])]["curve"]
    for r, c in block["per_c_loss"][str(block["tuned_C"])]["curve"].items():
        rows = {p["seed"]: p["macro_f1"] for p in c["per_seed"]}
        _anchor(
            anchors,
            f"loss ascent at the tuned C, {r} sigma, starts {p9a_seeds}",
            float(np.mean([rows[s] for s in p9a_seeds])),
            cell[r]["macro_f1"],
            ANCHOR_F1,
        )


def read_c(block: dict, C: str, sweeps: dict) -> dict:
    """One value of C: each objective's reading, and the worst over both objectives."""
    per_obj = {k: sweeps[k]["settings"][C] for k in OBJECTIVES}
    # the first minimum: a tie goes to loss ascent, the attack every other reading uses
    kind = min(OBJECTIVES, key=lambda k: per_obj[k]["worst"]["f1_minus_floor"])
    w = per_obj[kind]["worst"]
    curve = block[f"per_c_{kind}"][C]["curve"]
    key = next(r for r in curve if float(r) == w["radius"])
    per_start = [float(p["macro_f1"]) for p in curve[key]["per_seed"]]
    fit = block["per_c"][C]
    return {
        "clean_macro_f1": fit["clean_macro_f1"],
        "degenerate": fit["degenerate"],
        "n_support": fit["n_support"],
        **per_obj,
        "worst": {
            "objective": kind,
            "radius": w["radius"],
            "macro_f1": float(np.mean(per_start)),
            "macro_f1_std": float(np.std(per_start, ddof=1)) if len(per_start) > 1 else 0.0,
            "f1_minus_floor": w["f1_minus_floor"],
            "end_state": w["end_state"],
            "n_radii_inverted": per_obj[kind]["n_radii_inverted"],
            "per_start": per_start,
            "random_control_macro_f1": block["per_c_random"][C]["curve"][key]["macro_f1"],
        },
        "per_input_union": fit["per_input_union"],
    }


def verdict(readings: dict, tuned: str) -> dict:
    """The registered branch (module docstring) and the numbers it rests on."""
    small = sorted((C for C in readings if float(C) <= SMALL_C_MAX), key=float)
    t = np.asarray(readings[tuned]["worst"]["per_start"])
    gaps = {}
    for C in small:
        d = np.asarray(readings[C]["worst"]["per_start"]) - t
        gaps[C] = {
            "mean": float(d.mean()),
            "std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "starts_at_least_gap_min": int((d >= GAP_MIN).sum()),
            "per_start": d.tolist(),
        }
    inverted = {C: r["worst"]["end_state"] == "inverted" for C, r in readings.items()}
    persistent = {C: r["worst"]["n_radii_inverted"] >= MIN_RADII for C, r in readings.items()}
    stands = (
        not any(inverted[C] for C in small)
        and inverted[tuned]
        and all(
            g["mean"] >= GAP_MIN and g["starts_at_least_gap_min"] >= SEEDS_MIN
            for g in gaps.values()
        )
    )
    falls = all(inverted.values()) and all(persistent.values())
    return {
        "branch": "STANDS UNDER R" if stands else "FALLS UNDER R" if falls else "MIXED",
        "small_c": small,
        "tuned_c": tuned,
        "inverted_at_worst": inverted,
        "inverted_at_min_radii_or_more": persistent,
        "small_c_minus_tuned_at_worst": gaps,
        "gap_min": GAP_MIN,
        "seeds_min": SEEDS_MIN,
        "persistence_min_radii": MIN_RADII,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, help="the p11e runs, one per C")
    ap.add_argument("--part-b-run", default=PART_B_RUN)
    ap.add_argument("--p9a-run", default=P9A_RUN)
    args = ap.parse_args()
    sources: dict = {}
    anchors: list = []
    block = merge([_json(RUNS / r / "results.json", sources)[BLOCK] for r in args.runs])
    part_b = _json(RUNS / args.part_b_run / "results.json", sources)[
        "p11c_objective_and_regularization"
    ]["part_b_regularization"]
    p9a = _json(RUNS / args.p9a_run / "results.json", sources)["p9a_quantum_bandwidth_regime"]
    p9a_seeds = [int(s) for s in load_yaml(RUNS / args.p9a_run / "params.yaml")["seeds"]]
    check_anchors(block, part_b, p9a, p9a_seeds, anchors)
    degenerate = [C for C, v in block["per_c"].items() if v["degenerate"]]
    if degenerate:
        raise ValueError(f"clean model within 0.05 of the floor at C {degenerate}: nothing to read")

    budget = float(block["budget_sigma"])
    sweeps = {
        k: read_sweep(
            {
                "constant_predictor_floor": block["constant_predictor_floor"],
                "per_c": block[f"per_c_{k}"],
            },
            "per_c",
            budget,
            MIN_RADII,
        )
        for k in (*OBJECTIVES, "random")
    }
    tuned = str(block["tuned_C"])
    readings = {C: read_c(block, C, sweeps) for C in sorted(block["per_c"], key=float)}
    out = {
        "rule": (
            "per C the worst reading is the lowest mean attacked macro-F1 over the attack "
            "starts, over the grid radii at or below the budget and over the two objectives"
        ),
        **{k: block[k] for k in SHARED},
        "source_runs": list(args.runs),
        "verdict": verdict(readings, tuned),
        "per_c": readings,
        "random_control": sweeps["random"]["settings"],
        "numeric_gates": {C: v["numeric_gate"] for C, v in block["per_c"].items()},
        "anchors": anchors,
        "source_files_sha256": sources,
    }
    with RunDir("regularization_radii_verdict", config=vars(args)) as run:
        rows = ["C,objective,radius,start,macro_f1,mass_ratio,dec_spread_ratio,modal_frac"]
        for kind in (*OBJECTIVES, "random"):
            for C, v in block[f"per_c_{kind}"].items():
                for radius, c in v["curve"].items():
                    rows += [
                        f"{C},{kind},{radius},{p['seed']},{p['macro_f1']!r},{p['mass_ratio']!r},"
                        f"{p['dec_spread_ratio']!r},{p['modal_frac']!r}"
                        for p in c["per_seed"]
                    ]
        Path(run.path, "per_start.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        run.write_json("results.json", out)
        for C, r in readings.items():
            w = r["worst"]
            log.info(
                "C %s: worst %+.3f against the floor (%s, %s at %.2f sigma, inverted at %d radii)",
                C,
                w["f1_minus_floor"],
                w["end_state"],
                w["objective"],
                w["radius"],
                w["n_radii_inverted"],
            )
        log.info("branch: %s", out["verdict"]["branch"])
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
