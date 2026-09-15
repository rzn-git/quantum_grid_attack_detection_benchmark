"""W1 — leakage-free dimensional ladder: is there ANY quantum-reachable regime above chance?

PRE-REGISTRATION (pass-4 workplan item W1; frozen before the run).
E7 closed with "the surrogate instance is not testable leakage-free" because every arm sat
below its own stratified-random floor at PCA-8 on a whole-file split. That is a property of the
CAPPED 8-DIMENSIONAL regime, not a proof about the whole-file protocol: the full-feature level
arm clears its floor by roughly 0.095 on the same partitions. This run measures
excess-over-floor as a function of dimensionality and turns "untestable" into a LOCATED
statement.

TWO HYPOTHESES, both with rules fixed before the run:
  H1 (dimensional)  excess-over-floor rises monotonically 8 -> 12 -> 16 -> full, so PCA
                    compression, not the whole-file protocol, is what removes the
                    cross-file-transferable signal.
                    confirmed  monotone non-decreasing AND (full - pca8) >= 0.05
                    refuted    pca16 within 0.02 of pca8 while full clears both by >= 0.05
                               (the signal lives in specific raw features, not in dimension)
  H2 (testability)  the surrogate instance becomes testable at the smallest d where BOTH the
                    MLP surrogate and the VQC clear floor + 0.05. If no d in {8,12,16}
                    qualifies for the MLP, the VQC leg is NOT run — the instance is dead
                    without a surrogate — and E7's verdict upgrades from "not testable at
                    matched dimensionality" to "not testable at any quantum-reachable
                    dimensionality", which is the stronger statement.

TRAINING-SET CONTRACT. The ladder is run at the matched 2,000-row cap for every regime so
that dimensionality is the only thing that changes across rungs. A separate full-train
LightGBM cell per partition carries the LEVEL contract (E5's `score_level`), because that is
the number the 0.095 headroom comes from; the two are reported side by side and never mixed.

CONTROLS (blocking, read before any rule).
  positive-1  the full-train LightGBM cell must reproduce E5's archived per-partition
              `level_full_lgbm` at seed 0 within 1e-3 (same partitions, same stream)
  positive-2  the PCA-8 unweighted MLP at partition 0 must land within 0.02 of E7's
              measured -0.080 excess-over-floor
  negative    the stratified-random floor must be INVARIANT across regimes within one
              partition (it reads labels only) and sit within 0.01 of 0.499

Run:  python -m experiments.run_dimensional_ladder --partitions 3 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import f1_score

from experiments.run_group_split_control import group_splits, reconstruct_groups
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
CAP = 2000  # the matched contract: identical training-set size on every rung
REGIMES = [("pca", 8), ("pca", 12), ("pca", 16), ("full", None)]
# "mlp_balanced" is the same estimator fit with the imbalance policy its class carries no
# argument for (run_surrogate_control's corrected arm), kept distinct from the published "mlp".
MODELS = ["mlp", "mlp_balanced", "xgb", "lgbm", "rbf_svm"]

E5_ARTIFACT = "results/runs/20260822T090134Z_group_split_power_binary"
E7_MLP_PCA8_P0_EXCESS = -0.080  # decision log 2026-08-22T09:45Z (10-seed mean)
FLOOR_REFERENCE = 0.499
QUALIFY = 0.05  # excess-over-floor a model must clear to be worth attacking


def regime_key(regime: str, d: int | None) -> str:
    return regime if regime == "full" else f"pca{d}"


def tuned(model: str, regime: str, d: int | None) -> dict:
    """Hyperparameters tuned FOR THIS REGIME (not the q8 parameters reused off-regime)."""
    name = f"cls_{model}_{VARIANT}_{regime_key(regime, d)}"
    db = (REPO_ROOT / "results" / f"optuna_{name}.db").as_posix()
    return optuna.load_study(study_name=name, storage=f"sqlite:///{db}").best_params


def score(model: str, params: dict, Ztr, ytr, Zte, yte, seed: int) -> float:
    set_all_seeds(seed)
    est = build("mlp" if model == "mlp_balanced" else model, params, seed)
    fit_model(
        "mlp" if model == "mlp_balanced" else model,
        est,
        Ztr,
        ytr,
        balance_via_sample_weight=(model == "mlp_balanced"),
    )
    return float(f1_score(yte, est.predict(Zte), average="macro"))


def e5_reference() -> dict[int, float]:
    """Archived per-partition level numbers at seed 0 (the reproduction anchor)."""
    src = Path(REPO_ROOT / E5_ARTIFACT / "results.json")
    rows = json.loads(src.read_text())["rows"]
    return {r["partition"]: r["level_full_lgbm"] for r in rows if r["seed"] == 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    X, y, gid = reconstruct_groups()
    rng = np.random.default_rng(0)  # shipped stream: partition sets match every group control
    parts = []
    for p in range(args.partitions):
        sp, held = group_splits(gid, y, rng)
        parts.append((p, sp, held[2]))

    cfg = {
        "variant": VARIANT,
        "cap": CAP,
        "regimes": [regime_key(r, d) for r, d in REGIMES],
        "models": MODELS,
        "partitions": args.partitions,
        "seeds": args.seeds,
        "qualify_threshold": QUALIFY,
        "e5_artifact": E5_ARTIFACT,
        "e7_mlp_pca8_p0_excess": E7_MLP_PCA8_P0_EXCESS,
    }

    with RunDir(f"dimensional_ladder_{VARIANT}", config=cfg, seeds=args.seeds) as run:
        e5 = e5_reference()
        rows: list[dict] = []
        level_rows: list[dict] = []

        for p, sp, held in parts:
            ytr_all, yte = y[sp["train"]], y[sp["test"]]
            floor = trivial_baselines(ytr_all, yte)["baseline_stratified_random"]["aggregate"][
                "macro_f1"
            ]["mean"]

            for regime, d in REGIMES:
                pipe = build_preprocessor(regime, n_components=d, seed=0)
                Z = fit_transform_splits(pipe, X, sp)
                sub = stratified_cap(Z["train"], ytr_all, CAP, seed=0)
                Ztr, ytr = Z["train"][sub], ytr_all[sub]

                for model in MODELS:
                    params = tuned("mlp" if model == "mlp_balanced" else model, regime, d)
                    for s in args.seeds:
                        f1 = score(model, params, Ztr, ytr, Z["test"], yte, s)
                        rows.append(
                            {
                                "partition": p,
                                "regime": regime_key(regime, d),
                                "dim": d if d else int(X.shape[1]),
                                "model": model,
                                "seed": s,
                                "clean_f1": f1,
                                "floor": floor,
                                "excess": f1 - floor,
                                "n_train": int(len(ytr)),
                            }
                        )
                    ex = np.mean([r["excess"] for r in rows[-len(args.seeds) :]])
                    log.info(
                        "p%d %-6s %-12s excess %+0.4f (floor %.4f)",
                        p,
                        regime_key(regime, d),
                        model,
                        ex,
                        floor,
                    )
                    run.write_json("results_partial.json", {"rows": rows})

                # LEVEL contract: full features, FULL train split, tuned LightGBM (E5's arm)
                if regime == "full":
                    for s in args.seeds:
                        f1 = score(
                            "lgbm", tuned("lgbm", regime, d), Z["train"], ytr_all, Z["test"], yte, s
                        )
                        level_rows.append(
                            {
                                "partition": p,
                                "seed": s,
                                "level_full_lgbm": f1,
                                "floor": floor,
                                "excess": f1 - floor,
                                "n_train": int(len(ytr_all)),
                            }
                        )
                    log.info(
                        "p%d LEVEL full-train lgbm %.4f (E5 archived %.4f)",
                        p,
                        level_rows[-len(args.seeds)]["level_full_lgbm"],
                        e5.get(p, float("nan")),
                    )
                    run.write_json("results_partial.json", {"rows": rows, "level_rows": level_rows})

        # ---- controls -------------------------------------------------------------
        lvl_s0 = {r["partition"]: r["level_full_lgbm"] for r in level_rows if r["seed"] == 0}
        pos1 = {
            str(p): {"measured": v, "archived": e5.get(p), "delta": v - e5[p]}
            for p, v in lvl_s0.items()
            if p in e5
        }
        pos1_ok = all(abs(c["delta"]) < 1e-3 for c in pos1.values())

        mlp_p0 = [r["excess"] for r in rows if r["partition"] == 0 and r["model"] == "mlp"]
        mlp_p0_pca8 = [
            r["excess"]
            for r in rows
            if r["partition"] == 0 and r["model"] == "mlp" and r["regime"] == "pca8"
        ]
        pos2_delta = float(np.mean(mlp_p0_pca8) - E7_MLP_PCA8_P0_EXCESS)
        pos2_ok = abs(pos2_delta) <= 0.02

        floors = {}
        for r in rows:
            floors.setdefault(r["partition"], set()).add(round(r["floor"], 12))
        neg_invariant = all(len(v) == 1 for v in floors.values())
        neg_band = all(abs(next(iter(v)) - FLOOR_REFERENCE) <= 0.01 for v in floors.values())

        controls = {
            "positive_1_level_reproduction": pos1,
            "positive_1_pass": bool(pos1_ok),
            "positive_2_mlp_pca8_p0_excess": float(np.mean(mlp_p0_pca8)),
            "positive_2_delta_vs_e7": pos2_delta,
            "positive_2_pass": bool(pos2_ok),
            "negative_floor_invariant_across_regimes": bool(neg_invariant),
            "negative_floor_in_band": bool(neg_band),
            "pass": bool(pos1_ok and pos2_ok and neg_invariant and neg_band),
            "note_mlp_p0_all_regimes": [round(float(v), 4) for v in mlp_p0],
        }
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        # ---- H1: does excess rise with dimensionality? ----------------------------
        order = [regime_key(r, d) for r, d in REGIMES]

        def mean_excess(reg: str, model: str | None = None) -> float:
            v = [
                r["excess"]
                for r in rows
                if r["regime"] == reg and (model is None or r["model"] == model)
            ]
            return float(np.mean(v)) if v else float("nan")

        # H1 is read on the BEST model per rung: the question is whether the regime admits
        # above-floor detection at all, not whether a particular estimator likes it.
        best_per_regime = {}
        for reg in order:
            per_model = {m: mean_excess(reg, m) for m in MODELS}
            top = max(per_model, key=lambda m: per_model[m])
            best_per_regime[reg] = {"model": top, "excess": per_model[top], "all": per_model}

        ladder = [best_per_regime[r]["excess"] for r in order]
        monotone = all(ladder[i + 1] >= ladder[i] - 1e-9 for i in range(len(ladder) - 1))
        gap_full_pca8 = ladder[-1] - ladder[0]
        gap_pca16_pca8 = ladder[2] - ladder[0]
        h1 = {
            "order": order,
            "best_excess_per_regime": {r: best_per_regime[r] for r in order},
            "monotone_non_decreasing": bool(monotone),
            "full_minus_pca8": float(gap_full_pca8),
            "pca16_minus_pca8": float(gap_pca16_pca8),
            "branch": (
                "confirmed_dimensional"
                if (monotone and gap_full_pca8 >= 0.05)
                else (
                    "refuted_signal_in_raw_features"
                    if (abs(gap_pca16_pca8) <= 0.02 and gap_full_pca8 >= 0.05)
                    else "inconclusive"
                )
            ),
        }

        # ---- H2: is any quantum-reachable rung testable? --------------------------
        quantum_reachable = ["pca8", "pca12", "pca16"]
        mlp_by_reg = {r: mean_excess(r, "mlp") for r in quantum_reachable}
        qualifying = [r for r in quantum_reachable if mlp_by_reg[r] >= QUALIFY]
        h2 = {
            "mlp_excess_by_regime": mlp_by_reg,
            "qualify_threshold": QUALIFY,
            "qualifying_regimes": qualifying,
            "smallest_qualifying_d": qualifying[0] if qualifying else None,
            "stage2_vqc_leg_runs": bool(qualifying),
            "verdict": (
                "surrogate instance testable at " + qualifying[0]
                if qualifying
                else "not testable at ANY quantum-reachable dimensionality (8/12/16): the "
                "surrogate itself never clears floor+0.05, so no VQC leg is warranted"
            ),
        }

        # full-train level headroom, reported beside the capped ladder, never mixed with it
        level_excess = float(np.mean([r["excess"] for r in level_rows]))
        verdict = {
            "controls": controls,
            "h1": h1,
            "h2": h2,
            "level_full_train_excess": level_excess,
            "capped_full_excess": ladder[-1],
            "data_vs_dimension": float(level_excess - ladder[-1]),
            "rule": (
                "H1: monotone AND full-pca8>=0.05 -> dimensional; pca16~pca8 AND full>=0.05 "
                "-> raw-feature. H2: smallest d with MLP excess >= 0.05 gates the VQC leg."
            ),
        }
        run.write_json(
            "results.json",
            {"rows": rows, "level_rows": level_rows, "verdict": verdict},
        )
        run.metrics = {
            "best_excess_pca8": ladder[0],
            "best_excess_full": ladder[-1],
            "level_full_train_excess": level_excess,
        }
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
