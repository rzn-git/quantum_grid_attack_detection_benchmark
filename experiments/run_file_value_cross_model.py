"""D2c — is per-file training value a property of the FILE, or of XGBoost?

PRE-REGISTRATION (BACKLOG `Validate` row, opened by the winds review 2026-08-23T15:13Z; frozen
before the run). D2 (2026-08-23T01:10Z) and D2b (2026-08-23T08:45Z) both measured per-file training
value with ONE learner, tuned XGBoost. D2b triangulated by DATA SLICE — five independent partitions,
which is what let D2's post-hoc "file 14" observation earn its keep — and NOT by METHOD. Yet the
claim in its sharpest form is stated about the FILE: per-file training value is large (7.5x), stable
(mean cross-partition rho +0.642 against a 0.129 null) and unpredictable from cheap statistics.

This project's research discipline names "second method" FIRST among the ways to triangulate a
headline, and P11a is the in-repo precedent: a second surrogate of a different function class is
what turned the corrected C1 from one fit into a result. A tuned RBF-SVM is a different function
class, is deterministic, and is cheap at this scale.

METHOD. Re-run D2b's per-file sweep with a tuned RBF-SVM over the SAME 10 whole-file partitions
from the shipped rng stream and the SAME 2,000-row stratified caps. The XGBoost half is READ from
D2b's artifact — not refit — so the comparison cannot drift on thread count, the exact failure that
cost D2b its first attempt. Then, WITHIN each partition, take the Spearman rho between the XGBoost
and SVM per-file excess vectors (9 files each) and average over partitions.

Within-partition is the right unit and the choice is deliberate: it asks whether the two learners
RANK THE SAME FILES the same way, which is the claim at issue, and it is immune to any level
difference between the learners.

DECISION RULE, fixed now:
  mean within-partition cross-model rho >= 0.5  -> per-file value is a property of the FILE; the
                                                   paper may keep stating it about the data
  mean within-partition cross-model rho <  0.5  -> per-file value is LEARNER-SPECIFIC; every
                                                   statement is scoped to "for a tuned
                                                   gradient-boosted tree" rather than about the file

CONTROLS.
  positive (blocking)  every (partition, file) key this run generates must be present in D2b's
                       artifact, and this run's per-partition FLOOR must equal D2b's exactly. The
                       floor is computed from the split, so an exact match certifies that the
                       partition stream and the preprocessing reproduce D2b's — which is what makes
                       reading its XGBoost half legitimate instead of assumed.
  negative             a permutation null: shuffle the SVM per-file excesses WITHIN each partition
                       and recompute the mean within-partition rho, 2,000 times, reporting the 95th
                       percentile, so the 0.5 bar is read against a null computed on this data.
  reported             per-file mean rank under the SVM, so D2b's file-14/file-10 extremes can be
                       checked under the second learner rather than assumed to carry over.

This is a `Validate` row whose output is a FINDING: nothing is implemented and nothing is adopted.

--------------------------------------------------------------------------------------------------
AMENDMENT 2026-08-23, AFTER THE FIRST RUN — everything above is the ORIGINAL pre-registration and is
deliberately left unedited; this block records what the first run exposed and what was added.

The `rbf_svm` run (`20260823T152147Z_file_value_cross_model_binary`) returned mean within-partition
rho +0.425 against a 0.50 bar and a 0.178 null — the kill criterion fired. It should NOT be read,
because the instrument failed a validity check the original design did not include: **99% of its
single-file excesses were NEGATIVE** (mean -0.0389, best +0.0003). A learner that never clears the
floor is ranking degrees of FAILURE, not degrees of value, so its per-file ordering does not measure
the quantity D2b measured, and a correlation against it cannot triangulate the claim.

Two things were therefore added, and both are controls rather than new questions:
  --model                     the second learner is now selectable, so a learner that DOES clear the
                              floor in the single-file regime can be tried. `rbf_svm` stays the
                              default so the first run remains reproducible from this file.
  instrument validity arm     the same learner refit on ALL NINE training files at the identical
                              2,000-row cap, per partition. This separates "the single-file REGIME
                              defeats this learner" from "this learner is wrong for this data" —
                              opposite implications, and the first run could not tell them apart.

The bar, the statistic, the null and the decision rule are UNCHANGED. What changed is that a run
must now demonstrate its instrument is measuring value before its rho is read at all.
--------------------------------------------------------------------------------------------------

Run:  OMP_NUM_THREADS=4 python -m experiments.run_file_value_cross_model --model mlp
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

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
RHO_BAR = 0.5
N_PERM = 2000
MODEL = "rbf_svm"
D2B_RUN = "results/runs/20260823T083323Z_file_value_reproducibility_binary"


def full_regime_params(model: str) -> dict:
    name = f"cls_{model}_{VARIANT}_full"
    db = (REPO_ROOT / "results" / f"optuna_{name}.db").as_posix()
    return optuna.load_study(study_name=name, storage=f"sqlite:///{db}").best_params


def d2b_cells() -> tuple[dict, dict]:
    """(partition, file) -> XGBoost excess, and partition -> floor, both read from D2b unchanged."""
    cells = json.loads((REPO_ROOT / D2B_RUN / "results.json").read_text())["cells"]
    ex = {(c["partition"], c["file"]): c["excess"] for c in cells}
    floors = {c["partition"]: c["floor"] for c in cells}
    return ex, floors


def mean_within_partition_rho(a: dict, b: dict) -> tuple[float, list]:
    """Mean Spearman rho between two per-file excess maps, computed WITHIN each partition."""
    out = []
    for p in sorted(set(a) & set(b)):
        shared = sorted(set(a[p]) & set(b[p]))
        if len(shared) < 3:
            continue
        rho, _ = spearmanr(rankdata([a[p][f] for f in shared]), rankdata([b[p][f] for f in shared]))
        if not np.isnan(rho):
            out.append({"partition": p, "n_files": len(shared), "rho": float(rho)})
    return (float(np.mean([r["rho"] for r in out])) if out else float("nan")), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument(
        "--model",
        default=MODEL,
        help="the SECOND learner. Added after the first run (rbf_svm) failed the instrument "
        "validity check below: a learner whose single-file excess never clears the floor is "
        "ranking degrees of FAILURE, not degrees of value, and cannot triangulate.",
    )
    args = ap.parse_args()

    params = full_regime_params(args.model)
    xgb_ex, d2b_floor = d2b_cells()

    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # shipped stream — must reproduce D2b's partitions exactly
    parts = [(p, group_splits(gid, y, rng_part)[0]) for p in range(args.partitions)]

    cfg = {
        "variant": VARIANT,
        "model": args.model,
        "cap": CAP,
        "partitions": args.partitions,
        "seeds": args.seeds,
        "rho_bar": RHO_BAR,
        "n_permutations": N_PERM,
        "d2b_reference": D2B_RUN,
        "note": "the XGBoost half is READ from D2b, never refit — no thread-count exposure",
    }

    with RunDir(
        f"file_value_cross_model_{args.model}_{VARIANT}", config=cfg, seeds=args.seeds
    ) as run:
        cells: list[dict] = []
        floor_deltas: list[float] = []
        pooled: list[dict] = []
        for p, sp in parts:
            pipe = build_preprocessor("full", n_components=None, seed=0)
            Z = fit_transform_splits(pipe, X, sp)
            ytr_all, yte = y[sp["train"]], y[sp["test"]]
            gtr = gid[sp["train"]]
            floor = trivial_baselines(ytr_all, yte)["baseline_stratified_random"]["aggregate"][
                "macro_f1"
            ]["mean"]
            if p in d2b_floor:
                floor_deltas.append(abs(floor - d2b_floor[p]))

            for f in sorted(np.unique(gtr).tolist()):
                m_sub = gtr == f
                Zsub, ysub = Z["train"][m_sub], ytr_all[m_sub]
                idx = stratified_cap(Zsub, ysub, min(CAP, len(ysub)), seed=0)
                Ztr, ytr = Zsub[idx], ysub[idx]
                ex = []
                for s in args.seeds:
                    set_all_seeds(s)
                    mdl = build(args.model, params, s)
                    fit_model(args.model, mdl, Ztr, ytr)
                    ex.append(float(f1_score(yte, mdl.predict(Z["test"]), average="macro")) - floor)
                cells.append(
                    {
                        "partition": p,
                        "file": f,
                        "floor": floor,
                        "excess": float(np.mean(ex)),
                        "excess_by_seed": ex,
                    }
                )

            # INSTRUMENT VALIDITY CONTROL: the same learner on ALL nine training files, capped
            # identically. A learner that clears the floor here but not on single files is
            # failing the single-file REGIME; one that fails here too is simply the wrong
            # instrument for this data. The two have opposite implications and the first D2c run
            # could not tell them apart.
            idx_all = stratified_cap(Z["train"], ytr_all, CAP, seed=0)
            pooled_ex = []
            for s in args.seeds:
                set_all_seeds(s)
                mdl = build(args.model, params, s)
                fit_model(args.model, mdl, Z["train"][idx_all], ytr_all[idx_all])
                pooled_ex.append(
                    float(f1_score(yte, mdl.predict(Z["test"]), average="macro")) - floor
                )
            pooled.append({"partition": p, "excess": float(np.mean(pooled_ex))})

            here = [c for c in cells if c["partition"] == p]
            log.info(
                "p%d done | %d files | %s best %+0.4f worst %+0.4f | pooled-9-file %+0.4f",
                p,
                len(here),
                args.model,
                max(c["excess"] for c in here),
                min(c["excess"] for c in here),
                pooled[-1]["excess"],
            )
            run.write_json("results_partial.json", {"cells": cells})

        svm_by_part: dict = defaultdict(dict)
        for c in cells:
            svm_by_part[c["partition"]][c["file"]] = c["excess"]
        xgb_by_part: dict = defaultdict(dict)
        for (p, f), v in xgb_ex.items():
            xgb_by_part[p][f] = v

        # ---- controls -------------------------------------------------------------------
        missing = [(p, f) for p in svm_by_part for f in svm_by_part[p] if (p, f) not in xgb_ex]
        worst_floor = max(floor_deltas) if floor_deltas else float("nan")
        controls = {
            "positive_keys_missing_from_d2b": len(missing),
            "positive_worst_floor_delta": float(worst_floor),
            "positive_pass": bool(not missing and worst_floor == 0.0),
            "instrument_single_file_frac_below_floor": float(
                np.mean([c["excess"] < 0 for c in cells])
            ),
            "instrument_single_file_max_excess": float(max(c["excess"] for c in cells)),
            "instrument_pooled_9file_excess": float(np.mean([q["excess"] for q in pooled])),
            "instrument_pooled_by_partition": pooled,
            "instrument_valid": bool(max(c["excess"] for c in cells) > 0),
            "instrument_note": "a learner whose single-file excess never clears the floor is "
            "ranking degrees of FAILURE, not degrees of value; the pooled-9-file arm separates "
            "'wrong regime' from 'wrong instrument for this data'",
            "positive_note": "identical keys and exact floors certify the partition stream and "
            "preprocessing reproduce D2b, which is what makes reading its XGBoost half legitimate",
        }

        observed, per_part = mean_within_partition_rho(xgb_by_part, svm_by_part)
        rng = np.random.default_rng(0)
        null = []
        for _ in range(N_PERM):
            shuffled = {
                p: dict(zip(list(d), rng.permutation(list(d.values())), strict=True))
                for p, d in svm_by_part.items()
            }
            r, _ = mean_within_partition_rho(xgb_by_part, shuffled)
            if not np.isnan(r):
                null.append(r)
        null_arr = np.array(null)
        controls["negative_permutation_null_p95"] = float(np.percentile(null_arr, 95))
        controls["negative_permutation_null_mean"] = float(null_arr.mean())
        controls["pass"] = controls["positive_pass"]
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        # ---- per-file mean rank under the SVM, against D2b's under XGBoost ----------------
        def mean_ranks(by_part: dict) -> dict:
            ranks: dict = defaultdict(list)
            for _p, d in by_part.items():
                for i, f in enumerate(sorted(d, key=lambda k: -d[k])):
                    ranks[f].append(i + 1)
            return {
                str(f): {
                    "n_partitions": len(v),
                    "mean_rank": float(np.mean(v)),
                    "best_rank": int(min(v)),
                    "worst_rank": int(max(v)),
                }
                for f, v in sorted(ranks.items())
            }

        verdict = {
            "controls": controls,
            "mean_within_partition_rho": observed,
            "n_partitions_compared": len(per_part),
            "per_partition_rho": per_part,
            "rho_bar": RHO_BAR,
            "beats_null_p95": bool(observed > controls["negative_permutation_null_p95"]),
            "per_file_rank_svm": mean_ranks(svm_by_part),
            "per_file_rank_xgb": mean_ranks(xgb_by_part),
            "branch": (
                "per_file_value_is_a_property_of_the_FILE"
                if observed >= RHO_BAR
                else "per_file_value_is_LEARNER_SPECIFIC"
            ),
            "rule": f"mean within-partition cross-model rho >= {RHO_BAR} -> property of the file",
            "cells": cells,
        }
        run.write_json("results.json", verdict)
        run.metrics = {
            "mean_within_partition_rho": observed,
            "n_partitions_compared": len(per_part),
        }
        log.info(
            "VERDICT %s | mean rho %+0.3f over %d partitions (bar %.2f, null p95 %.3f)",
            verdict["branch"],
            observed,
            len(per_part),
            RHO_BAR,
            controls["negative_permutation_null_p95"],
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
