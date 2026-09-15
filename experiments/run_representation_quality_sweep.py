"""N2d — does the quantum/classical gap track REPRESENTATION QUALITY? A within-run sweep.

PRE-REGISTRATION (BACKLOG `Validate` row, winds review 2026-08-23T09:04Z; re-ranked 15:13Z; frozen
before the run). N2c observed the quantum arms gaining more than the classical arm on the
PCA-8 -> supervised-8 step (+0.0443 kernel, +0.0569 VQC, against +0.0341 XGBoost). Across every
representation in hand the story does NOT hold: the quantum/classical ratio runs 0.68 (random-8) ->
0.46 (PCA-8) -> 0.91 (supervised-8), non-monotone, and the random-8 point comes from a DIFFERENT
artifact than the other two. Three points, one not strictly comparable, non-monotone — no claim in
either direction. This run puts the whole quality range inside ONE artifact.

TWO DEVIATIONS FROM THE 09:04Z SKETCH, both stated before the run rather than discovered after.

  (1) THE QUALITY AXIS IS LABEL CORRUPTION, NOT A RECONSTRUCTION-TO-SUPERVISION BLEND. The sketch
      put label fraction 0 at "reconstruction-only", which needs a decoder and therefore a SECOND
      code path — and a second code path is exactly what would stop the label-fraction-1.0 arm from
      reproducing N2c BIT-IDENTICALLY, which is the blocking control and the reason the sweep is
      worth running at all. Instead, one mechanism spans the range: at fraction f, a 1-f share of
      the encoder's training labels is replaced by a uniform random draw. At f = 1.0 the label
      vector is the untouched original, so the call into N2b's `fit_file_adversarial` is literally
      the same call N2c made. At f = 0 the encoder is driven by pure label noise. Nothing about the
      encoder code changes at any point on the sweep. The reconstruction-only point is not lost to
      the record — N2b measured it at +0.0260 (`autoenc8`) — it is simply not on this axis, and the
      random projection supplies the within-run bottom anchor the cross-artifact defect demanded.

  (2) THE RATIO IS FLOORED, BECAUSE E11 ALREADY PAID FOR THIS LESSON. The registered statistic is
      the quantum/classical ratio, and at the bottom of a quality sweep the classical excess
      approaches zero — precisely the regime E11/P28 (2026-08-22T08:12Z) found makes a ratio
      structurally unstable, whose winds check demanded "a bounded statistic ... chosen BEFORE the
      run". The denominator is therefore floored at 0.01 in absolute value, and the DIFFERENCE
      (classical minus quantum) is reported alongside as the bounded companion. The decision rule is
      read on the pre-registered floored ratio. If the two statistics disagree, that disagreement is
      the finding and NEITHER is read as settled.

DECISION RULE, fixed now (unchanged from 09:04Z apart from the flooring above):
  |Spearman rho| >= 0.5 between classical excess and the quantum/classical ratio
        -> the gap tracks representation quality; N2c's observation is a real relationship
  |Spearman rho| <  0.5
        -> no relationship; the N2c observation is a two-point artifact and the paper says nothing
           about it

The correlation unit is the (partition, representation) CELL — 6 representations x 3 partitions = 18
points — because a 6-point Spearman over representation means has almost no power. The 6-point
aggregate is reported too, and both are stated whichever way they land.

CONTROLS.
  positive (blocking)  the label-fraction-1.0 arm must reproduce N2c's supervised-8 figures —
                       XGBoost +0.06325 and fidelity kernel +0.05777 — within 5e-4, at 12 threads.
                       The thread count is written into the pre-registration rather than left to the
                       day's scheduling; that is the D2b lesson, which cost a run.
  negative             the random-projection arm pins the bottom of the range WITHIN this artifact,
                       which is the whole point: it is what fixes the cross-run defect in the
                       three-point table above.
  reported             each arm's classical excess IS the quality axis, so a sweep that failed to
                       span a range is visible rather than silently correlated against noise.

SCOPE. The VQC is deliberately excluded: at roughly five minutes a fit it would add hours, and the
kernel carries the question. Stated rather than silently dropped.

EFFICIENCY NOTE, and the blocking control covers it. The fidelity-kernel arm is deterministic given
(Ztr, Zte) — N2c's own artifact shows an identical `qsvm_excess` at all three seeds within every
partition — so the kernel is computed ONCE per (partition, representation) and recorded for each
seed, rather than three times. If that assumption were wrong the label-fraction-1.0 arm would miss
N2c's +0.05777 and the run would fail its blocking control.

Run:  OMP_NUM_THREADS=12 python -m experiments.run_representation_quality_sweep
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from experiments.run_group_split_control import group_splits, reconstruct_groups
from experiments.run_nonlinear_embedding import DIM, encode, fit_file_adversarial
from experiments.run_supervised_repr_quantum import qsvm_excess, tuned
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.utils.run_tracking import RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
CAP = 2000
ENCODER_FIT_N = 3000  # the same stratified subsample N2b and N2c fit their encoders on
LABEL_FRACTIONS = [0.0, 0.01, 0.05, 0.25, 1.0]
N2C_XGB_SUPERVISED = 0.06324948105905161  # 20260823T075450Z_supervised_repr_quantum_binary
N2C_QSVM_SUPERVISED = 0.05777171815279374
POS_TOL = 5e-4
RHO_BAR = 0.5
DENOM_FLOOR = 0.01  # E11/P28 lesson: a ratio needs a floored denominator, chosen before the run


def corrupt_labels(y: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Keep a `fraction` share of true labels; replace the rest with a uniform random draw.

    At fraction 1.0 the ORIGINAL array is returned unchanged and no random number is drawn, so the
    downstream call is byte-for-byte the call N2c made — which is what the blocking control tests.
    """
    if fraction >= 1.0:
        return y
    keep = rng.random(len(y)) < fraction
    return np.where(keep, y, rng.integers(0, 2, size=len(y))).astype(y.dtype)


def xgb_excess(params, Ztr, ytr, Zte, yte, floor, seeds) -> list[float]:
    out = []
    for s in seeds:
        set_all_seeds(s)
        m = build("xgb", params, s)
        fit_model("xgb", m, Ztr, ytr)
        out.append(float(f1_score(yte, m.predict(Zte), average="macro")) - floor)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    best = tuned()
    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # shipped stream — partitions match every prior run
    parts = [(p, group_splits(gid, y, rng_part)[0]) for p in range(args.partitions)]

    arms = [f"label{f:g}" for f in LABEL_FRACTIONS] + ["random8"]
    cfg = {
        "variant": VARIANT,
        "dim": DIM,
        "cap": CAP,
        "encoder_fit_n": ENCODER_FIT_N,
        "label_fractions": LABEL_FRACTIONS,
        "arms": arms,
        "seeds": args.seeds,
        "rho_bar": RHO_BAR,
        "denominator_floor": DENOM_FLOOR,
        "n2c_reference": {"xgb": N2C_XGB_SUPERVISED, "qsvm": N2C_QSVM_SUPERVISED},
        "vqc": "excluded by pre-registration — the kernel carries the question",
        "note": "quality axis is LABEL CORRUPTION on one code path; see the module docstring",
    }

    with RunDir(f"representation_quality_sweep_{VARIANT}", config=cfg, seeds=args.seeds) as run:
        cells: list[dict] = []
        for p, sp in parts:
            pipe = build_preprocessor("full", n_components=None, seed=0)
            Z = fit_transform_splits(pipe, X, sp)
            ytr_all, yte = y[sp["train"]], y[sp["test"]]
            gtr = gid[sp["train"]]
            floor = trivial_baselines(ytr_all, yte)["baseline_stratified_random"]["aggregate"][
                "macro_f1"
            ]["mean"]
            sub = stratified_cap(Z["train"], ytr_all, min(ENCODER_FIT_N, len(ytr_all)), seed=0)

            reps: dict = {}
            for f in LABEL_FRACTIONS:
                rng_lab = np.random.default_rng(9000 + p)  # independent of the torch stream
                y_used = corrupt_labels(ytr_all[sub], f, rng_lab)
                enc, _acc, _ch = fit_file_adversarial(
                    Z["train"][sub], y_used, gtr[sub], seed=0, lam_max=0.0
                )
                reps[f"label{f:g}"] = (encode(enc, Z["train"]), encode(enc, Z["test"]))

            r = np.random.default_rng(7000)  # identical construction to N2b's random8 arm
            R = r.standard_normal((Z["train"].shape[1], DIM)) / np.sqrt(Z["train"].shape[1])
            reps["random8"] = (Z["train"] @ R, Z["test"] @ R)

            for arm, (Ctr_full, Cte) in reps.items():
                idx = stratified_cap(Ctr_full, ytr_all, CAP, seed=0)
                Ctr, ytr = Ctr_full[idx], ytr_all[idx]
                xg = xgb_excess(best["xgb"], Ctr, ytr, Cte, yte, floor, args.seeds)
                qk = qsvm_excess(Ctr, ytr, Cte, yte, best, args.seeds[0]) - floor
                cells.append(
                    {
                        "partition": p,
                        "arm": arm,
                        "floor": floor,
                        "classical_excess": float(np.mean(xg)),
                        "classical_excess_by_seed": xg,
                        "quantum_excess": qk,
                    }
                )
                log.info(
                    "p%d %-10s classical %+0.4f | quantum %+0.4f", p, arm, float(np.mean(xg)), qk
                )
                run.write_json("results_partial.json", {"cells": cells})

        for c in cells:
            denom = max(abs(c["classical_excess"]), DENOM_FLOOR)
            c["ratio_floored"] = c["quantum_excess"] / denom
            c["difference"] = c["classical_excess"] - c["quantum_excess"]

        # ---- controls -------------------------------------------------------------------
        sup = [c for c in cells if c["arm"] == "label1"]
        pos_x = float(np.mean([c["classical_excess"] for c in sup]))
        pos_q = float(np.mean([c["quantum_excess"] for c in sup]))
        controls = {
            "positive_label1_xgb": pos_x,
            "positive_label1_qsvm": pos_q,
            "positive_reference": {"xgb": N2C_XGB_SUPERVISED, "qsvm": N2C_QSVM_SUPERVISED},
            "positive_delta_xgb": abs(pos_x - N2C_XGB_SUPERVISED),
            "positive_delta_qsvm": abs(pos_q - N2C_QSVM_SUPERVISED),
            "positive_pass": bool(
                abs(pos_x - N2C_XGB_SUPERVISED) <= POS_TOL
                and abs(pos_q - N2C_QSVM_SUPERVISED) <= POS_TOL
            ),
            "positive_note": "label fraction 1.0 is N2c's exact call; anything else means the "
            "sweep is not anchored to the finding it was built to test",
            "negative_random8_classical": float(
                np.mean([c["classical_excess"] for c in cells if c["arm"] == "random8"])
            ),
            "negative_random8_quantum": float(
                np.mean([c["quantum_excess"] for c in cells if c["arm"] == "random8"])
            ),
        }
        span = [c["classical_excess"] for c in cells]
        controls["reported_quality_span"] = [float(min(span)), float(max(span))]
        controls["pass"] = controls["positive_pass"]
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        # ---- the pre-registered statistic, and its bounded companion ---------------------
        xs = [c["classical_excess"] for c in cells]
        rho_cell, p_cell = spearmanr(xs, [c["ratio_floored"] for c in cells])
        rho_diff, p_diff = spearmanr(xs, [c["difference"] for c in cells])
        by_arm = {
            a: {
                "classical": float(
                    np.mean([c["classical_excess"] for c in cells if c["arm"] == a])
                ),
                "quantum": float(np.mean([c["quantum_excess"] for c in cells if c["arm"] == a])),
                "ratio_floored": float(
                    np.mean([c["ratio_floored"] for c in cells if c["arm"] == a])
                ),
                "difference": float(np.mean([c["difference"] for c in cells if c["arm"] == a])),
            }
            for a in arms
        }
        rho_arm, p_arm = spearmanr(
            [v["classical"] for v in by_arm.values()],
            [v["ratio_floored"] for v in by_arm.values()],
        )
        agrees = bool((abs(rho_cell) >= RHO_BAR) == (abs(rho_diff) >= RHO_BAR))
        verdict = {
            "controls": controls,
            "rho_cell_level": float(rho_cell),
            "p_cell_level": float(p_cell),
            "n_cells": len(cells),
            "rho_arm_level": float(rho_arm),
            "p_arm_level": float(p_arm),
            "rho_difference_companion": float(rho_diff),
            "p_difference_companion": float(p_diff),
            "statistics_agree": agrees,
            "rho_bar": RHO_BAR,
            "by_arm": by_arm,
            "branch": (
                "gap_tracks_representation_quality"
                if abs(rho_cell) >= RHO_BAR
                else "no_representation_quality_relationship"
            ),
            "rule": f"|rho| >= {RHO_BAR} on the floored ratio -> the gap tracks quality",
            "cells": cells,
        }
        run.write_json("results.json", verdict)
        run.metrics = {"rho_cell_level": float(rho_cell), "rho_arm_level": float(rho_arm)}
        log.info(
            "VERDICT %s | rho(cell) %+0.3f p=%.4f | rho(arm) %+0.3f | companion %+0.3f | agree %s",
            verdict["branch"],
            rho_cell,
            p_cell,
            rho_arm,
            rho_diff,
            agrees,
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
