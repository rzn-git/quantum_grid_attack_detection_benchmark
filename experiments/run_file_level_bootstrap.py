"""X1 — resample FILES, not partitions: are the whole-file intervals too narrow?

PRE-REGISTRATION (pass-4 workplan item X1; frozen before the run).
W3 established that file identity is the unit of generalization on this dataset. If that is
true, this project's own uncertainty arithmetic over-counts: E5's headline interval
(+0.032, 95% CI [0.026, 0.039]) is a percentile bootstrap over 20 whole-file PARTITIONS, and
those partitions are not independent draws. Measured on E5's own artifact: 20 partitions x 3
held-out files spread over 15 files gives 60 slots, each file appearing 2-7 times, and 49% of
partition pairs share at least one test file. Resampling partitions therefore treats a shared
file as fresh evidence. This run recomputes the same intervals with the file as the resampling
unit. NO MODEL IS REFIT — it reads E5's archived rows.

HYPOTHESIS. The file-level interval is materially wider than the partition-level one.

DECISION RULE (primary: the cluster bootstrap over file-level means).
  excludes 0 -> the queued v2 sentence for the classical arm stands as drafted
  includes 0 -> the sentence changes again and the paper gains a note about its own
                resampling unit; ESCALATE to the operator rather than rewording
Reported for the classical arm, the quantum arm, and the contrast, at each seed and pooled.

TRIANGULATION (required before the rule is read). Two independent file-level estimators must
agree, or the widening is not trustworthy:
  (a) cluster bootstrap over the 15 file-level means (each partition's statistic credited
      equally to its 3 held-out files)
  (b) delete-one-file jackknife: for each file, average over the partitions whose test set does
      NOT contain it, then the standard (n-1)/n jackknife variance
Agreement criterion, fixed now: the two half-widths agree within 30% of the larger.

CONTROL (blocking). The partition-level bootstrap recomputed here from the artifact must
reproduce E5's archived CI to 1e-9 — same values, same resamples, same seed — so any widening
is attributable to the resampling unit and nothing else.

Run:  python -m experiments.run_file_level_bootstrap
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger

log = get_logger(__name__)

E5_ARTIFACT = "results/runs/20260822T090134Z_group_split_power_binary"
BOOT_RESAMPLES = 10000  # E5's count, reused so the control can be exact
BOOT_SEED = 12345  # E5's bootstrap seed
AGREEMENT = 0.30  # max relative disagreement between the two file-level estimators
ARMS = {
    "classical_minus_floor": lambda r: r["matched_classical"] - r["floor_macro_f1"],
    "quantum_minus_floor": lambda r: r["matched_quantum"] - r["floor_macro_f1"],
    "contrast_classical_minus_quantum": lambda r: r["matched_contrast"],
    "level_full_lgbm": lambda r: r["level_full_lgbm"],
}


def percentile_ci(draws: np.ndarray) -> dict:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "half_width": float((hi - lo) / 2),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def partition_bootstrap(values: np.ndarray, rng: np.random.Generator) -> dict:
    """E5's estimator, recomputed here so the control is exact."""
    draws = rng.choice(values, size=(BOOT_RESAMPLES, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "n_units": int(len(values)), **percentile_ci(draws)}


def file_means(rows: list[dict], stat) -> dict[int, float]:
    """Credit each partition's statistic equally to the 3 files it held out."""
    acc = defaultdict(list)
    for r in rows:
        v = stat(r)
        for f in r["test_groups"]:
            acc[int(f)].append(v)
    return {f: float(np.mean(v)) for f, v in sorted(acc.items())}


def file_cluster_bootstrap(rows: list[dict], stat, rng: np.random.Generator) -> dict:
    """(a) Resample the 15 FILES with replacement; the file mean is the unit."""
    fm = file_means(rows, stat)
    vals = np.array(list(fm.values()), dtype=float)
    draws = rng.choice(vals, size=(BOOT_RESAMPLES, len(vals)), replace=True).mean(axis=1)
    return {
        "mean": float(vals.mean()),
        "n_units": int(len(vals)),
        "per_file": fm,
        **percentile_ci(draws),
    }


def file_jackknife(rows: list[dict], stat) -> dict:
    """(b) Delete-one-file jackknife over the partitions that do not hold that file out."""
    files = sorted({int(f) for r in rows for f in r["test_groups"]})
    theta = float(np.mean([stat(r) for r in rows]))
    reps = []
    for f in files:
        keep = [stat(r) for r in rows if f not in r["test_groups"]]
        if not keep:
            continue
        reps.append(float(np.mean(keep)))
    reps_arr = np.array(reps, dtype=float)
    n = len(reps_arr)
    var = (n - 1) / n * float(((reps_arr - reps_arr.mean()) ** 2).sum())
    se = float(np.sqrt(var))
    return {
        "mean": theta,
        "n_units": n,
        "se": se,
        "ci_lo": theta - 1.96 * se,
        "ci_hi": theta + 1.96 * se,
        "half_width": 1.96 * se,
        "excludes_zero": bool(abs(theta) > 1.96 * se),
        "replicates": reps,
    }


def overlap_audit(rows: list[dict]) -> dict:
    counts: dict[int, int] = defaultdict(int)
    for r in rows:
        for f in r["test_groups"]:
            counts[int(f)] += 1
    sets = [set(r["test_groups"]) for r in rows]
    shared = [len(sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return {
        "n_partitions": len(rows),
        "n_files": len(counts),
        "slots": int(sum(counts.values())),
        "appearances_per_file": dict(sorted(counts.items())),
        "mean_shared_files_per_pair": float(np.mean(shared)),
        "share_of_pairs_sharing_a_file": float(np.mean([s > 0 for s in shared])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=E5_ARTIFACT)
    args = ap.parse_args()

    rows_all = json.loads((REPO_ROOT / args.source / "results.json").read_text())["rows"]
    archived = json.loads((REPO_ROOT / args.source / "results.json").read_text())["verdict"]["cis"]
    seeds = sorted({r["seed"] for r in rows_all})

    cfg = {
        "source_artifact": args.source,
        "boot_resamples": BOOT_RESAMPLES,
        "boot_seed": BOOT_SEED,
        "agreement_tolerance": AGREEMENT,
        "note": "re-analysis only; no model is refit",
    }

    with RunDir("file_level_bootstrap_binary", config=cfg, seeds=seeds) as run:
        audit = overlap_audit([r for r in rows_all if r["seed"] == seeds[0]])
        log.info("OVERLAP AUDIT: %s", json.dumps(audit, indent=2))

        out: dict = {"overlap_audit": audit, "per_seed": {}}
        # E5 consumed ONE rng for every CI in sequence; the control reproduces that order.
        rng_ctrl = np.random.default_rng(BOOT_SEED)
        rng_file = np.random.default_rng(BOOT_SEED)
        control_deltas = {}

        for s in seeds:
            rows = [r for r in rows_all if r["seed"] == s]
            per_arm = {}
            for arm, stat in ARMS.items():
                vals = np.array([stat(r) for r in rows], dtype=float)
                part = partition_bootstrap(vals, rng_ctrl)
                clus = file_cluster_bootstrap(rows, stat, rng_file)
                jack = file_jackknife(rows, stat)
                widen = clus["half_width"] / max(part["half_width"], 1e-12)
                agree = abs(clus["half_width"] - jack["half_width"]) / max(
                    clus["half_width"], jack["half_width"], 1e-12
                )
                per_arm[arm] = {
                    "partition_bootstrap": part,
                    "file_cluster_bootstrap": clus,
                    "file_jackknife": jack,
                    "widening_factor": float(widen),
                    "estimator_disagreement": float(agree),
                    "estimators_agree": bool(agree <= AGREEMENT),
                }
                key = {
                    "classical_minus_floor": "classical_minus_floor",
                    "quantum_minus_floor": "quantum_minus_floor",
                    "contrast_classical_minus_quantum": "contrast_classical_minus_quantum",
                    "level_full_lgbm": "level_full_lgbm",
                }[arm]
                a = archived[f"seed{s}"][key]
                control_deltas[f"seed{s}_{arm}"] = {
                    "ci_lo_delta": part["ci_lo"] - a["ci_lo"],
                    "ci_hi_delta": part["ci_hi"] - a["ci_hi"],
                    "mean_delta": part["mean"] - a["mean"],
                }
                log.info(
                    "s%d %-34s partition [%.4f, %.4f] -> file [%.4f, %.4f] (%.2fx wider, "
                    "jackknife half-width %.4f)",
                    s,
                    arm,
                    part["ci_lo"],
                    part["ci_hi"],
                    clus["ci_lo"],
                    clus["ci_hi"],
                    widen,
                    jack["half_width"],
                )
            out["per_seed"][f"seed{s}"] = per_arm

        ctrl_ok = all(
            abs(v["ci_lo_delta"]) < 1e-9
            and abs(v["ci_hi_delta"]) < 1e-9
            and abs(v["mean_delta"]) < 1e-9
            for v in control_deltas.values()
        )
        controls = {
            "partition_ci_reproduction": control_deltas,
            "pass": bool(ctrl_ok),
            "note": (
                "E5 drew all its CIs from one generator in a fixed order; this run consumes it "
                "the same way, so an exact match confirms the re-analysis reads the artifact "
                "correctly and any widening comes from the resampling unit alone"
            ),
        }
        log.info("CONTROL exact-reproduction pass: %s", ctrl_ok)

        primary = out["per_seed"][f"seed{seeds[0]}"]["classical_minus_floor"]
        verdict = {
            "controls": controls,
            "primary_arm": "classical_minus_floor (seed 0, the paper's primary)",
            "partition_ci": [
                primary["partition_bootstrap"]["ci_lo"],
                primary["partition_bootstrap"]["ci_hi"],
            ],
            "file_ci": [
                primary["file_cluster_bootstrap"]["ci_lo"],
                primary["file_cluster_bootstrap"]["ci_hi"],
            ],
            "widening_factor": primary["widening_factor"],
            "estimators_agree": primary["estimators_agree"],
            "still_excludes_zero": primary["file_cluster_bootstrap"]["excludes_zero"],
            "branch": (
                "v2_sentence_stands"
                if primary["file_cluster_bootstrap"]["excludes_zero"]
                else "ESCALATE_sentence_changes_again"
            ),
            "arms_flipping_significance": [
                f"seed{s}:{arm}"
                for s in seeds
                for arm, d in out["per_seed"][f"seed{s}"].items()
                if d["partition_bootstrap"]["excludes_zero"]
                != d["file_cluster_bootstrap"]["excludes_zero"]
            ],
            "rule": "file-level cluster bootstrap CI for (matched_classical - floor)",
        }
        run.write_json("results.json", {**out, "verdict": verdict})
        run.metrics = {
            "widening_factor_primary": primary["widening_factor"],
            "file_ci_lo_primary": primary["file_cluster_bootstrap"]["ci_lo"],
        }
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
