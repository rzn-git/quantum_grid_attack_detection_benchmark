"""E1 — label-ambiguity audit: is the accuracy ceiling label noise or row-level ambiguity?

PRE-REGISTRATION (pass-4 workplan item E1; frozen before the run).
The paper's abstract says a positive control "explains the accuracy null: the labels, not
the pipeline". The relabel control localizes the failure to LABEL-KERNEL ALIGNMENT; it
cannot separate (i) mislabeled rows from (ii) irreducible class overlap at row granularity
from (iii) clean labels this feature map cannot align with. (ii) is directly measurable and
was never measured. This run measures it.

Metrics, fixed before running:
  rho_conflict  fraction of pooled rows in an exact-feature-duplicate group carrying BOTH
                labels (the dedup keys feature+label, so these rows SURVIVE preprocessing
                and can straddle splits)
  delta_nn      fraction of TEST rows whose nearest TRAIN neighbour within 0.1 (standardized
                full-feature space, the shipped audit's metric) carries the OPPOSITE label
  delta_nn_grp  the same under one whole-file partition (deployment granularity)

DECISION RULE: delta_nn >= 0.05 OR rho_conflict >= 0.02 -> "material ambiguity" branch;
both below -> "alignment isolated" branch.

CONTROLS (must pass before the rule is read):
  positive  the shipped 45.8% within-0.1 neighbour fraction reproduces within 0.5 points
  negative  same-label NN fraction == 1 - opposite fraction (join is not off-by-one)

NOTE on dtype: data/processed/*.npz stores X as float32 while the preprocess dedup ran on
float64, so float32 exact-duplicate groups are a SUPERSET — this widens rho_conflict, i.e.
the estimate is conservative in the direction of finding ambiguity. Recorded in the artifact.

Run:  python -m experiments.run_label_ambiguity_audit
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from qgridbench.data.preprocess import load_processed
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.utils.run_tracking import RunDir, get_logger

log = get_logger(__name__)
VARIANT = "binary"
RADIUS = 0.1  # the shipped neighbour-audit radius (run_group_split_control.audit)
PUBLISHED_NN_WITHIN_0_1 = 0.458  # row-level, results.tex / group_split_control artifact


def duplicate_label_conflict(X: np.ndarray, y: np.ndarray) -> dict:
    """Exact-feature-duplicate groups carrying more than one label."""
    # group rows by exact feature-vector identity via a byte view (no float compare)
    Xc = np.ascontiguousarray(X)
    keys = np.array([r.tobytes() for r in Xc])
    order = np.argsort(keys, kind="stable")
    keys_sorted, y_sorted = keys[order], y[order]

    boundaries = np.flatnonzero(np.r_[True, keys_sorted[1:] != keys_sorted[:-1], True])
    n_conflict_rows = 0
    n_conflict_groups = 0
    n_dup_rows = 0
    n_dup_groups = 0
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        size = e - s
        if size < 2:
            continue
        n_dup_groups += 1
        n_dup_rows += int(size)
        if len(np.unique(y_sorted[s:e])) > 1:
            n_conflict_groups += 1
            n_conflict_rows += int(size)
    return {
        "rho_conflict": float(n_conflict_rows / len(y)),
        "n_conflict_rows": n_conflict_rows,
        "n_conflict_groups": n_conflict_groups,
        "n_duplicate_rows": n_dup_rows,
        "n_duplicate_groups": n_dup_groups,
        "dup_row_fraction": float(n_dup_rows / len(y)),
    }


def neighbour_label_agreement(Z_train, y_train, Z_test, y_test) -> dict:
    """Nearest-train-neighbour label agreement for test rows inside the radius."""
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Z_train)
    dist, idx = nn.kneighbors(Z_test)
    dist, idx = dist.ravel(), idx.ravel()
    near = dist < RADIUS
    y_nb = y_train[idx]
    opposite = (y_nb != y_test) & near
    same = (y_nb == y_test) & near
    n_near = int(near.sum())
    return {
        "nn_frac_within_0.1": float(near.mean()),
        "n_test": int(len(y_test)),
        "n_near": n_near,
        # delta_nn: opposite-label share AMONG the near pairs (the ambiguity rate)
        "delta_nn": float(opposite.sum() / n_near) if n_near else float("nan"),
        "delta_nn_of_all_test": float(opposite.mean()),
        # negative control: shares must sum to 1 over the near set
        "same_label_share_near": float(same.sum() / n_near) if n_near else float("nan"),
    }


def main() -> None:
    X, y, splits = load_processed(VARIANT)
    log.info("loaded %s: X %s, y %s", VARIANT, X.shape, y.shape)

    cfg = {
        "variant": VARIANT,
        "radius": RADIUS,
        "x_dtype_on_disk": str(X.dtype),
        "dtype_note": (
            "dedup at preprocess ran on float64; float32 duplicate groups are a superset, "
            "so rho_conflict is conservative (biased toward finding ambiguity)"
        ),
    }

    with RunDir(f"label_ambiguity_{VARIANT}", config=cfg) as run:
        # ---- (a) exact-feature-duplicate label conflict, pooled -------------------
        dup = duplicate_label_conflict(X, y)
        log.info("duplicate-conflict: %s", dup)

        # ---- (b) row-level split: nearest-train-neighbour label agreement ---------
        pipe = build_preprocessor("full", n_components=None, seed=0)
        Z = fit_transform_splits(pipe, X, {k: np.asarray(v) for k, v in splits.items()})
        row_nn = neighbour_label_agreement(
            Z["train"], y[splits["train"]], Z["test"], y[splits["test"]]
        )
        log.info("row-level NN: %s", row_nn)

        # ---- (c) whole-file partition p0, same measurement -----------------------
        from experiments.run_group_split_control import group_splits, reconstruct_groups

        Xg, yg, gid = reconstruct_groups()
        assert Xg.shape == X.shape, "group reconstruction misaligned with shipped X"
        assert np.array_equal(yg, y), "label vector diverged from the shipped artifact"
        rng = np.random.default_rng(0)  # same stream as the shipped group control
        sp_g, held = group_splits(gid, yg, rng)
        pipe_g = build_preprocessor("full", n_components=None, seed=0)
        Zg = fit_transform_splits(pipe_g, Xg, sp_g)
        grp_nn = neighbour_label_agreement(
            Zg["train"], yg[sp_g["train"]], Zg["test"], yg[sp_g["test"]]
        )
        log.info("group p0 NN (held-out files %s): %s", held[2], grp_nn)

        # ---- controls ------------------------------------------------------------
        pos_delta = row_nn["nn_frac_within_0.1"] - PUBLISHED_NN_WITHIN_0_1
        pos_ok = abs(pos_delta) <= 0.005
        neg_ok = bool(
            np.isclose(row_nn["delta_nn"] + row_nn["same_label_share_near"], 1.0, atol=1e-9)
        )
        controls = {
            "positive_nn_reproduction_delta": float(pos_delta),
            "positive_pass": bool(pos_ok),
            "negative_shares_sum_to_one": neg_ok,
            "negative_pass": neg_ok,
        }
        log.info("controls: %s", controls)

        # ---- pre-registered decision rule ---------------------------------------
        material = (row_nn["delta_nn"] >= 0.05) or (dup["rho_conflict"] >= 0.02)
        verdict = {
            "controls_passed": bool(pos_ok and neg_ok),
            "rho_conflict": dup["rho_conflict"],
            "delta_nn_row": row_nn["delta_nn"],
            "delta_nn_group_p0": grp_nn["delta_nn"],
            "branch": ("material_ambiguity" if material else "alignment_isolated"),
            "rule": "delta_nn >= 0.05 OR rho_conflict >= 0.02 -> material_ambiguity",
        }

        run.write_json(
            "results.json",
            {
                "duplicate_conflict": dup,
                "row_level_nn": row_nn,
                "group_p0_nn": {**grp_nn, "held_out_files": held[2]},
                "controls": controls,
                "verdict": verdict,
            },
        )
        run.metrics = {
            "rho_conflict": dup["rho_conflict"],
            "delta_nn_row": row_nn["delta_nn"],
            "delta_nn_group_p0": grp_nn["delta_nn"],
        }
        log.info("VERDICT: %s", verdict)
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
