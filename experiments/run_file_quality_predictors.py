"""D2 — is per-file training value PREDICTABLE in advance, without touching the held-out files?

PRE-REGISTRATION (BACKLOG `Validate` row D2, winds review 2026-08-23T00:49Z; frozen before the
run). D1 showed the spread across WHICH single file you train on (0.035-0.054 within a partition)
is two to four times the +0.0149 gained by going from one training file to nine. Every lever the
engineering-boundary map measures concerns volume, representation or dimensionality — none concerns
SELECTION. If per-file value is predictable from a statistic computable without the held-out files,
a practitioner can CHOOSE source files, which would be the first actionable mitigation this project
has found. If it is not, the boundary claim reaches its sharpest form: you can neither collect nor
choose your way out of the gap.

METHOD. Two arms, both on the same three whole-file partitions and the same shipped partition
stream.

  REPLICATION ARM  reproduces D1's k=1 procedure EXACTLY for the nine (partition, file) pairs D1
                   sampled — same 20% within-file holdout carve-out keyed to D1's subset index,
                   same `stratified_cap(..., 2000, seed=0)`, same seeds. This exists only to
                   satisfy the pre-registered control; it is not the science.
  SWEEP ARM        every one of the partition's nine training files, trained individually on
                   `stratified_cap(file_rows, 2000, seed=0)` with NO holdout carve-out — the
                   cleaner sampling, applied identically to all 27 (partition, file) cells.

Tuned XGBoost, full features, 3 seeds, scored on the partition's fixed held-out FILES. The floor is
the partition-level stratified-random baseline D1 used, so excesses are directly comparable.

FOUR PREDICTORS, every one computable WITHOUT the held-out files:
  n_rows          the file's row count
  minority_share  its class balance
  spread          mean feature variance in the partition's preprocessed space
  centrality      mean Euclidean distance from this file's centroid to the OTHER TRAINING files'
                  centroids — how typical it is among the files a practitioner already holds

DECISION RULE, fixed now, on the SWEEP arm, Spearman rho pooled over n = 27:
  some predictor reaches |rho| >= 0.5  -> file quality IS predictable; selection is a real
                                         mitigation and the paper says which statistic to use
  no predictor reaches |rho| >= 0.5    -> file quality is NOT predictable from cheap statistics;
                                         you can neither collect nor choose your way out

CONTROLS.
  positive (blocking)  the replication arm must reproduce D1's nine k=1 excesses to 1e-6. Same code
                       path, same seeds, so anything larger means this is not what D1 measured.
  negative             a permutation null: 2,000 shuffles of the excess against each predictor,
                       reporting the 95th percentile of |rho|, so the 0.5 bar is read against a
                       null computed ON THIS DATA rather than an assumed one.
  reported             per-partition rho (n = 9 each) alongside the pooled figure, and the pooled
                       figure recomputed on within-partition ranks, so a pooled correlation cannot
                       be an artifact of partitions having different floors.

TRIANGULATION. W3 concluded the cross-file drop is GENERIC file idiosyncrasy rather than distance
to the particular held-out files. `centrality` tests the training-side mirror of that: if being
typical among training files predicts value, W3 gains a mechanism it did not have; if it does not,
W3 is corroborated from a direction it never looked.

Run:  python -m experiments.run_file_quality_predictors
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import f1_score

from experiments.run_group_split_control import group_splits, reconstruct_groups
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
CAP = 2000
D1_RUN = "results/runs/20260822T211713Z_file_diversity_curve_binary"
D1_TOL = 1e-6
RHO_BAR = 0.5
N_PERM = 2000
PREDICTORS = ("n_rows", "minority_share", "spread", "centrality")


def full_regime_params(model: str) -> dict:
    name = f"cls_{model}_{VARIANT}_full"
    db = (REPO_ROOT / "results" / f"optuna_{name}.db").as_posix()
    return optuna.load_study(study_name=name, storage=f"sqlite:///{db}").best_params


def d1_pairs() -> dict:
    """(partition, subset_index) -> file, recovered from D1's own rows."""
    rows = json.loads((REPO_ROOT / D1_RUN / "results.json").read_text())["rows"]
    out = {}
    for x in rows:
        if x["k"] == 1 and x["arm"] == "A" and x["model"] == "xgb":
            out.setdefault((x["partition"], x["subset"]), x["files"][0])
    return out


def d1_excess() -> dict:
    """(partition, file, seed) -> the excess D1 recorded, for the blocking control."""
    rows = json.loads((REPO_ROOT / D1_RUN / "results.json").read_text())["rows"]
    return {
        (x["partition"], x["files"][0], x["seed"]): x["crossfile_excess"]
        for x in rows
        if x["k"] == 1 and x["arm"] == "A" and x["model"] == "xgb"
    }


def score(params: dict, Ztr, ytr, Zte, yte, seed: int) -> float:
    set_all_seeds(seed)
    m = build("xgb", params, seed)
    fit_model("xgb", m, Ztr, ytr)
    return float(f1_score(yte, m.predict(Zte), average="macro"))


def predictors_for(Ztr_file: np.ndarray, y_file: np.ndarray, centroids: dict, f: int) -> dict:
    """Every value here uses TRAINING data only — the held-out files are never touched."""
    c = np.bincount(y_file, minlength=2)
    others = [v for k, v in centroids.items() if k != f]
    return {
        "n_rows": float(len(y_file)),
        "minority_share": float(c.min() / c.sum()),
        "spread": float(np.mean(np.var(Ztr_file, axis=0))),
        "centrality": float(np.mean([np.linalg.norm(centroids[f] - o) for o in others])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    params = full_regime_params("xgb")
    pairs, d1x = d1_pairs(), d1_excess()

    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # shipped stream — partitions must match every prior run
    parts = []
    for p in range(args.partitions):
        sp, _held = group_splits(gid, y, rng_part)
        parts.append((p, sp))

    cfg = {
        "variant": VARIANT,
        "cap": CAP,
        "predictors": list(PREDICTORS),
        "rho_bar": RHO_BAR,
        "n_permutations": N_PERM,
        "d1_reference": D1_RUN,
        "partitions": args.partitions,
        "seeds": args.seeds,
        "note": "every predictor uses TRAINING files only; held-out files are never touched",
    }

    with RunDir(f"file_quality_predictors_{VARIANT}", config=cfg, seeds=args.seeds) as run:
        sweep: list[dict] = []
        replication: list[dict] = []

        for p, sp in parts:
            pipe = build_preprocessor("full", n_components=None, seed=0)
            Z = fit_transform_splits(pipe, X, sp)
            ytr_all, yte = y[sp["train"]], y[sp["test"]]
            gtr = gid[sp["train"]]
            files = sorted(np.unique(gtr).tolist())
            floor = trivial_baselines(ytr_all, yte)["baseline_stratified_random"]["aggregate"][
                "macro_f1"
            ]["mean"]
            centroids = {f: Z["train"][gtr == f].mean(axis=0) for f in files}

            # ---- REPLICATION ARM: D1's exact k=1 path, for the blocking control -------------
            for (pp, si), f in pairs.items():
                if pp != p:
                    continue
                m_sub = gtr == f
                Zsub, ysub = Z["train"][m_sub], ytr_all[m_sub]
                perm = np.random.default_rng(100 + si).permutation(len(ysub))
                n_hold = max(200, int(0.2 * len(ysub)))
                Zpool, ypool = Zsub[perm[n_hold:]], ysub[perm[n_hold:]]
                idx = stratified_cap(Zpool, ypool, min(CAP, len(ypool)), seed=0)
                for s in args.seeds:
                    e = score(params, Zpool[idx], ypool[idx], Z["test"], yte, s) - floor
                    replication.append(
                        {
                            "partition": p,
                            "file": f,
                            "subset": si,
                            "seed": s,
                            "excess": e,
                            "d1_excess": d1x[(p, f, s)],
                            "delta": e - d1x[(p, f, s)],
                        }
                    )

            # ---- SWEEP ARM: every training file, identical sampling --------------------------
            for f in files:
                m_sub = gtr == f
                Zsub, ysub = Z["train"][m_sub], ytr_all[m_sub]
                idx = stratified_cap(Zsub, ysub, min(CAP, len(ysub)), seed=0)
                Ztr, ytr = Zsub[idx], ysub[idx]
                ex = [score(params, Ztr, ytr, Z["test"], yte, s) - floor for s in args.seeds]
                sweep.append(
                    {
                        "partition": p,
                        "file": f,
                        "floor": floor,
                        "excess": float(np.mean(ex)),
                        "excess_by_seed": ex,
                        "n_train": int(len(ytr)),
                        **predictors_for(Zsub, ysub, centroids, f),
                    }
                )
                log.info(
                    "p%d file %-3d excess %+0.4f | rows %d minority %.3f centrality %.2f",
                    p,
                    f,
                    sweep[-1]["excess"],
                    int(sweep[-1]["n_rows"]),
                    sweep[-1]["minority_share"],
                    sweep[-1]["centrality"],
                )
                run.write_json("results_partial.json", {"sweep": sweep})

        # ---- controls -------------------------------------------------------------------
        worst = max(abs(r["delta"]) for r in replication)
        controls = {
            "positive_d1_replication_n": len(replication),
            "positive_worst_abs_delta": float(worst),
            "positive_pass": bool(worst <= D1_TOL),
            "positive_note": (
                "the replication arm reruns D1's exact k=1 path (20% holdout carve-out keyed to "
                "D1's subset index); the sweep arm deliberately omits that carve-out"
            ),
        }

        # ---- correlations ----------------------------------------------------------------
        exc = np.array([r["excess"] for r in sweep])
        part = np.array([r["partition"] for r in sweep])
        # within-partition ranks, so a pooled rho cannot be an artifact of differing floors
        exc_ranked = np.concatenate([rankdata(exc[part == p]) for p in sorted(set(part.tolist()))])
        order = np.concatenate([np.where(part == p)[0] for p in sorted(set(part.tolist()))])

        corr = {}
        rng_perm = np.random.default_rng(0)
        for name in PREDICTORS:
            v = np.array([r[name] for r in sweep])
            rho, pv = spearmanr(v, exc)
            per_part = {}
            for p in sorted(set(part.tolist())):
                m = part == p
                r_p, _ = spearmanr(v[m], exc[m])
                per_part[str(p)] = float(r_p)
            rho_ranked, _ = spearmanr(v[order], exc_ranked)
            null = np.array(
                [abs(spearmanr(v, rng_perm.permutation(exc))[0]) for _ in range(N_PERM)]
            )
            corr[name] = {
                "rho_pooled": float(rho),
                "p_value": float(pv),
                "abs_rho": float(abs(rho)),
                "rho_within_partition_ranks": float(rho_ranked),
                "rho_per_partition": per_part,
                "null_abs_rho_p95": float(np.percentile(null, 95)),
                "beats_null_p95": bool(abs(rho) > np.percentile(null, 95)),
                "clears_bar": bool(abs(rho) >= RHO_BAR),
            }
        controls["negative_permutation_null_p95"] = {
            k: v["null_abs_rho_p95"] for k, v in corr.items()
        }
        controls["negative_note"] = (
            "the 0.5 bar is read against a null computed on this data, not an assumed one"
        )
        controls["pass"] = controls["positive_pass"]
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        best = max(PREDICTORS, key=lambda k: corr[k]["abs_rho"])
        verdict = {
            "controls": controls,
            "correlations": corr,
            "best_predictor": best,
            "best_abs_rho": corr[best]["abs_rho"],
            "rho_bar": RHO_BAR,
            "spread_within_partition": {
                str(p): float(exc[part == p].max() - exc[part == p].min())
                for p in sorted(set(part.tolist()))
            },
            "excess_range": [float(exc.min()), float(exc.max())],
            "branch": (
                "file_quality_IS_predictable"
                if corr[best]["abs_rho"] >= RHO_BAR
                else "file_quality_is_NOT_predictable"
            ),
            "rule": f"some predictor reaching |rho| >= {RHO_BAR} pooled over n=27 -> predictable",
            "sweep": sweep,
            "replication": replication,
        }
        run.write_json("results.json", verdict)
        run.metrics = {"best_abs_rho": corr[best]["abs_rho"], "n_cells": len(sweep)}
        log.info(
            "VERDICT %s | best %s |rho| %.3f (bar %.2f) | excess range %.4f..%.4f",
            verdict["branch"],
            best,
            corr[best]["abs_rho"],
            RHO_BAR,
            exc.min(),
            exc.max(),
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
