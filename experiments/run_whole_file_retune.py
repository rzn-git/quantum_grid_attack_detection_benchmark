"""P27/P27b: re-tune LightGBM under the whole-file protocol — is the 0.311 drop tuning-limited?

THE GAP THIS CLOSES. results.tex discloses it: "All group-split arms reuse the
hyperparameters tuned under the row-level protocol; whether re-tuning under the whole-file
protocol recovers part of the drop is unmeasured." This runner measures it: an equal-budget
(50-trial) Optuna search per whole-file partition, tuned ON that partition's held-out VAL
files, evaluated once on its TEST files, against the archived row-tuned results on the
identical partitions.

P27b extension (pre-registered 2026-09-01T01:10Z, follow-up brief Task 1): all FIVE Table V
partitions; the winner and the row-tuned comparator both retrained across seeds {0..9} with
recovery as the seed-paired difference; a blocking row-level reproduction control (the
published 0.905318 must reproduce before any partition runs); and a blocking file-disjointness
assertion on every partition's train/val/test group sets.

Pre-registered decision lines (decision log 2026-08-20T18:27Z, inherited by P27b): recovery
< 0.02 -> the drop is not tuning-limited and the paper's sentence upgrades to the measured
figure; recovery >= 0.05 -> the 0.311 attribution is scoped with the measured recovery. The
archived comparators are recomputed in-run as positive controls and must match the archive.

Run:  python -m experiments.run_whole_file_retune [--partitions 5] [--trials 50]
"""

from __future__ import annotations

import argparse

import numpy as np
import optuna
from sklearn.metrics import f1_score

from experiments.run_group_split_control import group_splits, reconstruct_groups, tuned_params
from qgridbench.data.preprocess import load_processed
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, suggest_params
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
TUNING_SEEDS = [0, 1, 2]  # the paper's seed protocol: search on {0,1,2}
FINAL_SEEDS = list(range(10))  # winner + comparator retrained across all 10 seeds (P27b)
# Archived row-tuned results on the identical partitions
# (run 20260816T023620Z_group_split_control_binary).
ARCHIVED_ROW_TUNED = {0: 0.6032, 1: 0.5837, 2: 0.6033, 3: 0.5937, 4: 0.5884}
# The published row-level headline from the same archived control (level_row, seed 0).
ARCHIVED_ROW_LEVEL = 0.9053182941463003
# Table V held-out file ids — the partition stream must reproduce these exactly.
EXPECTED_TEST_GROUPS = {0: [1, 8, 13], 1: [3, 7, 9], 2: [1, 10, 11], 3: [5, 9, 14], 4: [1, 11, 12]}


def fit_and_score(params, Z, y, splits, score_split, seed):
    set_all_seeds(seed)
    est = build("lgbm", params, seed)
    fit_model("lgbm", est, Z["train"], y[splits["train"]])
    return float(f1_score(y[splits[score_split]], est.predict(Z[score_split]), average="macro"))


def assert_file_disjoint(splits, gid, p_idx):
    """The brief's rule: assert file-disjointness in code, do not assume it."""
    g = {k: set(np.unique(gid[idx]).tolist()) for k, idx in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = g[a] & g[b]
        if overlap:
            raise SystemExit(
                f"SPLIT NOT FILE-DISJOINT: p{p_idx} {a} and {b} share files {sorted(overlap)}"
            )
    return {k: sorted(v) for k, v in g.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--partitions", type=int, default=5)
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

    row_tuned_params, _ = tuned_params()
    X, y, gid = reconstruct_groups()

    cfg = {
        "variant": VARIANT,
        "n_partitions": args.partitions,
        "n_trials": args.trials,
        "tuning_seeds": TUNING_SEEDS,
        "final_seeds": FINAL_SEEDS,
        "preregistration": "decision log 2026-08-20T18:27Z (P27) + 2026-09-01T01:10Z (P27b)",
    }
    with RunDir(f"whole_file_retune_{VARIANT}", config=cfg, seeds=FINAL_SEEDS) as run:
        # Blocking control 0 (P27b): the published row-level number reproduces before any
        # partition runs — catches environment drift since the archived control.
        Xp, yp, splits_row = load_processed(VARIANT)
        assert X.shape == Xp.shape and np.array_equal(y, yp), "group reconstruction misaligned"
        Z_row = fit_transform_splits(
            build_preprocessor("full", n_components=None, seed=0), X, splits_row
        )
        row_level = fit_and_score(row_tuned_params, Z_row, y, splits_row, "test", seed=0)
        d_row = row_level - ARCHIVED_ROW_LEVEL
        log.info("row-level reproduction %.10f (archive delta %+.2e)", row_level, d_row)
        if abs(d_row) > 1e-6:
            raise SystemExit(
                f"ROW-LEVEL REPRODUCTION FAILED: {row_level:.10f} vs archived "
                f"{ARCHIVED_ROW_LEVEL:.10f} — environment drift; run discarded"
            )
        del Z_row

        rng = np.random.default_rng(0)  # same stream as the archived control -> same partitions
        rows = []
        for p_idx in range(args.partitions):
            splits, held = group_splits(gid, y, rng)
            if held[2] != EXPECTED_TEST_GROUPS[p_idx]:
                raise SystemExit(
                    f"PARTITION STREAM DIVERGED: p{p_idx} test files {held[2]} vs "
                    f"expected {EXPECTED_TEST_GROUPS[p_idx]}"
                )
            file_sets = assert_file_disjoint(splits, gid, p_idx)
            pipe = build_preprocessor("full", n_components=None, seed=0)
            Z = fit_transform_splits(pipe, X, splits)

            # positive control: the row-tuned config on this partition must match the archive
            row_tuned_test = fit_and_score(row_tuned_params, Z, y, splits, "test", seed=0)
            d_ctrl = row_tuned_test - ARCHIVED_ROW_TUNED[p_idx]
            log.info(
                "p%d row-tuned control %.4f (archive delta %+.4f)", p_idx, row_tuned_test, d_ctrl
            )
            if abs(d_ctrl) > 0.002:
                raise SystemExit(
                    f"POSITIVE CONTROL FAILED: p{p_idx} row-tuned {row_tuned_test:.4f} vs "
                    f"archived {ARCHIVED_ROW_TUNED[p_idx]:.4f} — run discarded"
                )

            def objective(trial):
                params = suggest_params("lgbm", trial)
                return float(
                    np.mean([fit_and_score(params, Z, y, splits, "val", s) for s in TUNING_SEEDS])
                )

            db = (REPO_ROOT / "results" / f"optuna_group_retune_{VARIANT}_p{p_idx}.db").as_posix()
            study = optuna.create_study(
                direction="maximize",
                study_name=f"group_retune_{VARIANT}_p{p_idx}",
                storage=f"sqlite:///{db}",
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=0),
            )
            remaining = args.trials - len([t for t in study.trials if t.state.is_finished()])
            if remaining > 0:
                study.optimize(objective, n_trials=remaining, show_progress_bar=False)

            # P27b final evaluation: both configurations across all 10 seeds, seed-paired.
            row_seed = [
                fit_and_score(row_tuned_params, Z, y, splits, "test", s) for s in FINAL_SEEDS
            ]
            grp_seed = [
                fit_and_score(study.best_params, Z, y, splits, "test", s) for s in FINAL_SEEDS
            ]
            rec_seed = [g - r for g, r in zip(grp_seed, row_seed)]
            rec = {
                "partition": p_idx,
                "file_sets": file_sets,
                "test_groups": held[2],
                "row_tuned_test": row_tuned_test,
                "group_tuned_test": grp_seed[0],
                "recovery_seed0": grp_seed[0] - row_tuned_test,
                "row_tuned_seed_scores": row_seed,
                "group_tuned_seed_scores": grp_seed,
                "row_tuned_mean": float(np.mean(row_seed)),
                "row_tuned_std": float(np.std(row_seed, ddof=1)),
                "group_tuned_mean": float(np.mean(grp_seed)),
                "group_tuned_std": float(np.std(grp_seed, ddof=1)),
                "recovery_mean": float(np.mean(rec_seed)),
                "recovery_std": float(np.std(rec_seed, ddof=1)),
                "group_tuned_val": study.best_value,
                "best_params": study.best_params,
                "n_trials": len([t for t in study.trials if t.state.is_finished()]),
            }
            rows.append(rec)
            run.write_json("results_partial.json", {"rows": rows})
            log.info(
                "p%d | row-tuned %.4f±%.4f -> group-tuned %.4f±%.4f | recovery %+.4f±%.4f",
                p_idx,
                rec["row_tuned_mean"],
                rec["row_tuned_std"],
                rec["group_tuned_mean"],
                rec["group_tuned_std"],
                rec["recovery_mean"],
                rec["recovery_std"],
            )

        recov = np.array([r["recovery_mean"] for r in rows])
        summary = {
            "row_level_reproduction": row_level,
            "row_tuned_mean": float(np.mean([r["row_tuned_mean"] for r in rows])),
            "group_tuned_mean": float(np.mean([r["group_tuned_mean"] for r in rows])),
            "recovery_mean": float(recov.mean()),
            "recovery_std": float(recov.std(ddof=1)) if len(recov) > 1 else 0.0,
            "recovery_per_partition": [float(v) for v in recov],
        }
        r = summary["recovery_mean"]
        summary["verdict"] = (
            "H1 — recovery < 0.02: the drop is not tuning-limited"
            if r < 0.02
            else "KILL — recovery >= 0.05: the 0.311 attribution must be scoped"
            if r >= 0.05
            else "INTERMEDIATE — report as measured, scope proportionally"
        )
        run.metrics = summary
        run.write_json("results.json", {"rows": rows, "summary": summary})
        log.info(
            "VERDICT: %s (recovery %+.4f +- %.4f)", summary["verdict"], r, summary["recovery_std"]
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
