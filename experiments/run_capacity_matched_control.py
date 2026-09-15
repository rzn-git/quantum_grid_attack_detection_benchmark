"""N-CAP: capacity-matched control on the leak-gap mechanism.

PRE-REGISTERED 2026-09-01T18:05Z (Tier B). Question: is the deep models' small
row-vs-whole-file gap an ARCHITECTURE effect, or does ANY model at matched row-level fit
show the same gap? A fixed a-priori ladder of LightGBM capacity rungs (stumps -> the tuned
Optuna winner) is scored under BOTH protocols; the rung whose row-level mean lands nearest
FTTransformer-default's 0.6363 is the matched read.

  KILL (generic fit-strength/ceiling account): matched-rung gap <= 0.08.
  SURVIVE (architecture account):              matched-rung gap >= 0.12.

Controls (blocking): tuned rung reproduces row 0.906 within 0.01 and whole-file mean 0.594
within 0.015; the ladder brackets 0.6363 row-level; Table V partition ids + file
disjointness asserted; a constant prediction FAILS the run.

Run:  python -m experiments.run_capacity_matched_control [--smoke]
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import f1_score

from experiments.run_group_split_control import group_splits, reconstruct_groups, tuned_params
from experiments.run_whole_file_retune import EXPECTED_TEST_GROUPS, assert_file_disjoint
from qgridbench.data.preprocess import load_processed
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model
from qgridbench.utils.run_tracking import RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
ROW_SEEDS = [0, 1, 2]
N_PARTITIONS = 5

# Anchors from the runs this control interrogates (results.json of each artifact).
FTT_ROW = 0.6363  # 20260901T114347Z_deep_tabular_ftt_default_binary, 10-seed mean
FTT_GAP = 0.0404
PUBLISHED_LGBM_ROW = 0.9056  # MODEL_CARD 10-seed mean
ARCHIVED_LGBM_GROUP = 0.594  # group_split_control 5-partition mean

# The fixed a-priori ladder (weakest -> tuned winner appended at runtime). Plain
# LGBMClassifier kwargs; build() adds class_weight="balanced" and the seed.
LADDER = [
    ("cap0", {"n_estimators": 1, "num_leaves": 2, "learning_rate": 0.5, "min_child_samples": 500}),
    ("cap1", {"n_estimators": 3, "num_leaves": 2, "learning_rate": 0.3, "min_child_samples": 200}),
    ("cap2", {"n_estimators": 5, "num_leaves": 2, "learning_rate": 0.2, "min_child_samples": 100}),
    ("cap3", {"n_estimators": 10, "num_leaves": 2, "learning_rate": 0.1, "min_child_samples": 100}),
    ("cap4", {"n_estimators": 20, "num_leaves": 2, "learning_rate": 0.1, "min_child_samples": 50}),
    ("cap5", {"n_estimators": 40, "num_leaves": 3, "learning_rate": 0.1, "min_child_samples": 50}),
    ("cap6", {"n_estimators": 80, "num_leaves": 6, "learning_rate": 0.1, "min_child_samples": 20}),
    (
        "cap7",
        {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.1, "min_child_samples": 10},
    ),
]


def score(params: dict, Z: dict, y: np.ndarray, splits: dict, seed: int) -> float:
    set_all_seeds(seed)
    est = build("lgbm", params, seed)
    fit_model("lgbm", est, Z["train"], y[splits["train"]])
    pred = est.predict(Z["test"])
    if len(np.unique(pred)) < 2:
        raise SystemExit(f"DEGENERATE PREDICTOR: params {params} seed {seed} predict one class")
    return float(f1_score(y[splits["test"]], pred, average="macro"))


def matched_read(rungs: list[dict]) -> dict:
    """The rung nearest FTT_ROW row-level, interpolating when no rung is within 0.03."""
    rows = np.array([r["row_mean"] for r in rungs])
    gaps = np.array([r["leak_gap"] for r in rungs])
    if rows.min() > FTT_ROW or rows.max() < FTT_ROW:
        raise SystemExit(
            f"LADDER DOES NOT BRACKET {FTT_ROW}: row range [{rows.min():.4f}, {rows.max():.4f}]"
        )
    i = int(np.argmin(np.abs(rows - FTT_ROW)))
    if abs(rows[i] - FTT_ROW) <= 0.03:
        return {
            "mode": "nearest",
            "rung": rungs[i]["name"],
            "row": float(rows[i]),
            "gap": float(gaps[i]),
        }
    below = int(np.where(rows <= FTT_ROW)[0][np.argmax(rows[rows <= FTT_ROW])])
    above = int(np.where(rows >= FTT_ROW)[0][np.argmin(rows[rows >= FTT_ROW])])
    w = (FTT_ROW - rows[below]) / (rows[above] - rows[below])
    return {
        "mode": "interpolated",
        "rung": f"{rungs[below]['name']}~{rungs[above]['name']}",
        "row": FTT_ROW,
        "gap": float((1 - w) * gaps[below] + w * gaps[above]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end pass, no artifacts")
    args = ap.parse_args()

    X, y, gid = reconstruct_groups()
    Xp, yp, splits_row = load_processed(VARIANT)
    assert X.shape == Xp.shape and np.array_equal(y, yp), "group reconstruction misaligned"
    splits_row = {k: np.asarray(v) for k, v in splits_row.items()}

    if args.smoke:
        sub = {k: v[:3000] for k, v in splits_row.items()}
        Z = fit_transform_splits(build_preprocessor("full", n_components=None, seed=0), X, sub)
        for name, params in LADDER[:2]:
            print(f"SMOKE {name}: row {score(params, Z, y, sub, 0):.4f} on 3k-row subset")
        print("SMOKE OK")
        return

    tuned, _ = tuned_params()
    ladder = LADDER + [("tuned", tuned)]
    cfg = {
        "variant": VARIANT,
        "ladder": {n: p for n, p in ladder},
        "row_seeds": ROW_SEEDS,
        "n_partitions": N_PARTITIONS,
        "anchors": {"ftt_row": FTT_ROW, "ftt_gap": FTT_GAP},
        "preregistration": "decision log 2026-09-01T18:05Z (N-CAP)",
    }
    with RunDir(f"capacity_matched_control_{VARIANT}", config=cfg, seeds=ROW_SEEDS) as run:
        # Transforms are rung-independent: fit each split's preprocessor once, reuse.
        Z_row = fit_transform_splits(
            build_preprocessor("full", n_components=None, seed=0), X, splits_row
        )
        rng = np.random.default_rng(0)
        partitions = []
        for p_idx in range(N_PARTITIONS):
            sp, held = group_splits(gid, y, rng)
            if held[2] != EXPECTED_TEST_GROUPS[p_idx]:
                raise SystemExit(f"PARTITION STREAM DIVERGED: p{p_idx} {held[2]}")
            assert_file_disjoint(sp, gid, p_idx)
            Zg = fit_transform_splits(build_preprocessor("full", n_components=None, seed=0), X, sp)
            partitions.append((sp, Zg, held[2]))

        rungs = []
        for name, params in ladder:
            row_scores = [score(params, Z_row, y, splits_row, s) for s in ROW_SEEDS]
            grp_scores = [score(params, Zg, y, sp, 0) for sp, Zg, _ in partitions]
            r = {
                "name": name,
                "params": params,
                "row_scores": row_scores,
                "row_mean": float(np.mean(row_scores)),
                "group_scores": grp_scores,
                "group_mean": float(np.mean(grp_scores)),
                "group_std": float(np.std(grp_scores, ddof=1)),
                "leak_gap": float(np.mean(row_scores) - np.mean(grp_scores)),
            }
            rungs.append(r)
            log.info(
                "%s | row %.4f | group %.4f±%.4f | gap %.4f",
                name,
                r["row_mean"],
                r["group_mean"],
                r["group_std"],
                r["leak_gap"],
            )
            run.write_json("results_partial.json", {"rungs": rungs})

        # Blocking positive control: the tuned rung anchors both ends of the instrument.
        t = rungs[-1]
        ctrl_row = abs(t["row_mean"] - PUBLISHED_LGBM_ROW) <= 0.01
        ctrl_grp = abs(t["group_mean"] - ARCHIVED_LGBM_GROUP) <= 0.015
        if not (ctrl_row and ctrl_grp):
            raise SystemExit(
                f"POSITIVE CONTROL FAILED: tuned row {t['row_mean']:.4f} "
                f"(pub {PUBLISHED_LGBM_ROW}), group {t['group_mean']:.4f} "
                f"(archived {ARCHIVED_LGBM_GROUP})"
            )

        m = matched_read(rungs)
        grp_means = np.array([r["group_mean"] for r in rungs])
        verdict = {
            "matched": m,
            "kill_fires": bool(m["gap"] <= 0.08),
            "survives": bool(m["gap"] >= 0.12),
            "inconclusive": bool(0.08 < m["gap"] < 0.12),
            "ceiling_flat_band": {
                "group_min": float(grp_means.min()),
                "group_max": float(grp_means.max()),
                "within_0.02_of_0.594": [
                    r["name"] for r in rungs if abs(r["group_mean"] - ARCHIVED_LGBM_GROUP) <= 0.02
                ],
            },
            "positive_control": {"tuned_row": t["row_mean"], "tuned_group": t["group_mean"]},
        }
        run.metrics = verdict
        run.write_json("results.json", {"rungs": rungs, "verdict": verdict})
        log.info(
            "VERDICT | matched rung %s (row %.4f) gap %.4f | kill=%s survive=%s",
            m["rung"],
            m["row"],
            m["gap"],
            verdict["kill_fires"],
            verdict["survives"],
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
