"""D2b — is per-file training value REPRODUCIBLE across partitions, or is it resampling noise?

PRE-REGISTRATION (BACKLOG `Spike` row, opened by the D2 finding 2026-08-23T01:10Z; frozen before the
run). D2 established that per-file training value varies 7.5x and that no cheap statistic predicts
it. It did NOT establish whether that variation is a stable property of a file or noise, and the two
readings differ for the paper: a stable property that cannot be predicted is a sharper claim than
noise. D2's evidence was weak and inconsistent — cross-partition Spearman rho of +0.143, +0.400 and
+0.900 on only 4 to 6 shared files — with one suggestive detail noticed after the fact, that file 14
ranked 2nd, 1st and 1st across three DIFFERENT held-out targets.

METHOD, and the design note from the row is the whole point: the limit was SHARED FILES PER PAIR,
not seed noise, so this buys more PARTITIONS rather than more seeds. Ten whole-file partitions drawn
from the shipped stream; in each, train on every one of its nine training files individually
(2,000 stratified rows, tuned XGBoost, full features) and score on that partition's fixed held-out
FILES. Then, for every pair of partitions sharing at least MIN_SHARED files, rank their shared files
by excess and take Spearman rho.

DECISION RULE, fixed now:
  mean cross-partition rho > 0.5   -> per-file value is REPRODUCIBLE; the paper may say the effect
                                      is a stable property of a file that cannot be predicted
  mean cross-partition rho <= 0.5  -> not shown to be reproducible; the paper says only that
                                      per-file value is large and unpredictable, and says nothing
                                      about stability

CONTROLS.
  positive (blocking)  partitions 0-2 come from the same shipped rng stream D2 used, with identical
                       sampling, so their 27 (file, seed) cells must reproduce D2's excesses
                       BIT-IDENTICALLY. Anything else means this is not measuring what D2 measured.
  negative             a permutation null: shuffle the per-file excesses WITHIN each partition and
                       recompute the mean cross-partition rho, 2,000 times, reporting the 95th
                       percentile — so the 0.5 bar is read against a null computed on this data.
  reported             per-file mean rank across the partitions each file appears in, which is how
                       the post-hoc "file 14" observation from D2 gets tested rather than repeated.

This is a SPIKE: the output is a finding, never an implementation.

Run:  python -m experiments.run_file_value_reproducibility
"""

from __future__ import annotations

import argparse
import itertools
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
MIN_SHARED = 4
RHO_BAR = 0.5
N_PERM = 2000
D2_RUN = "results/runs/20260823T010202Z_file_quality_predictors_binary"


def full_regime_params(model: str) -> dict:
    name = f"cls_{model}_{VARIANT}_full"
    db = (REPO_ROOT / "results" / f"optuna_{name}.db").as_posix()
    return optuna.load_study(study_name=name, storage=f"sqlite:///{db}").best_params


def d2_cells() -> dict:
    """(partition, file) -> mean excess, for the blocking control."""
    sweep = json.loads((REPO_ROOT / D2_RUN / "results.json").read_text())["sweep"]
    return {(x["partition"], x["file"]): x["excess"] for x in sweep}


def mean_pair_rho(by_part: dict, min_shared: int = MIN_SHARED) -> tuple[float, list]:
    """Mean Spearman rho over every partition pair sharing at least `min_shared` files."""
    pairs = []
    for a, b in itertools.combinations(sorted(by_part), 2):
        shared = sorted(set(by_part[a]) & set(by_part[b]))
        if len(shared) < min_shared:
            continue
        ra = rankdata([by_part[a][f] for f in shared])
        rb = rankdata([by_part[b][f] for f in shared])
        rho, _ = spearmanr(ra, rb)
        if not np.isnan(rho):
            pairs.append({"a": a, "b": b, "n_shared": len(shared), "rho": float(rho)})
    return (float(np.mean([p["rho"] for p in pairs])) if pairs else float("nan")), pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    params = full_regime_params("xgb")
    d2 = d2_cells()

    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # shipped stream — partitions 0-2 must match D2 exactly
    parts = []
    for p in range(args.partitions):
        sp, _held = group_splits(gid, y, rng_part)
        parts.append((p, sp))

    cfg = {
        "variant": VARIANT,
        "cap": CAP,
        "partitions": args.partitions,
        "seeds": args.seeds,
        "min_shared_files": MIN_SHARED,
        "rho_bar": RHO_BAR,
        "n_permutations": N_PERM,
        "d2_reference": D2_RUN,
        "note": "buys PARTITIONS, not seeds — the D2 limit was shared files per pair",
    }

    with RunDir(f"file_value_reproducibility_{VARIANT}", config=cfg, seeds=args.seeds) as run:
        cells: list[dict] = []
        for p, sp in parts:
            pipe = build_preprocessor("full", n_components=None, seed=0)
            Z = fit_transform_splits(pipe, X, sp)
            ytr_all, yte = y[sp["train"]], y[sp["test"]]
            gtr = gid[sp["train"]]
            floor = trivial_baselines(ytr_all, yte)["baseline_stratified_random"]["aggregate"][
                "macro_f1"
            ]["mean"]

            for f in sorted(np.unique(gtr).tolist()):
                m_sub = gtr == f
                Zsub, ysub = Z["train"][m_sub], ytr_all[m_sub]
                idx = stratified_cap(Zsub, ysub, min(CAP, len(ysub)), seed=0)
                Ztr, ytr = Zsub[idx], ysub[idx]
                ex = []
                for s in args.seeds:
                    set_all_seeds(s)
                    mdl = build("xgb", params, s)
                    fit_model("xgb", mdl, Ztr, ytr)
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
            log.info(
                "p%d done | %d files | best %+0.4f worst %+0.4f",
                p,
                len([c for c in cells if c["partition"] == p]),
                max(c["excess"] for c in cells if c["partition"] == p),
                min(c["excess"] for c in cells if c["partition"] == p),
            )
            run.write_json("results_partial.json", {"cells": cells})

        by_part = defaultdict(dict)
        for c in cells:
            by_part[c["partition"]][c["file"]] = c["excess"]

        # ---- controls ------------------------------------------------------------------
        deltas = {
            f"p{p}_f{f}": by_part[p][f] - d2[(p, f)]
            for (p, f) in d2
            if p in by_part and f in by_part[p]
        }
        worst = max(abs(v) for v in deltas.values()) if deltas else float("nan")
        controls = {
            "positive_d2_cells_checked": len(deltas),
            "positive_worst_abs_delta": float(worst),
            "positive_pass": bool(worst == 0.0),
            "positive_note": "same shipped stream and identical sampling — expect bit-identical",
        }

        observed, pairs = mean_pair_rho(by_part)
        rng = np.random.default_rng(0)
        null = []
        for _ in range(N_PERM):
            shuffled = {}
            for p, d in by_part.items():
                files = list(d)
                vals = rng.permutation(list(d.values()))
                shuffled[p] = dict(zip(files, vals, strict=True))
            r, _ = mean_pair_rho(shuffled)
            if not np.isnan(r):
                null.append(r)
        null = np.array(null)
        controls["negative_permutation_null_p95"] = float(np.percentile(null, 95))
        controls["negative_permutation_null_mean"] = float(null.mean())
        controls["pass"] = controls["positive_pass"]
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        # ---- per-file mean rank: tests D2's post-hoc "file 14" observation ---------------
        ranks = defaultdict(list)
        for p, d in by_part.items():
            order = sorted(d, key=lambda f: -d[f])
            for i, f in enumerate(order):
                ranks[f].append(i + 1)
        per_file = {
            str(f): {
                "n_partitions": len(v),
                "mean_rank": float(np.mean(v)),
                "best_rank": int(min(v)),
                "worst_rank": int(max(v)),
                "ranks": v,
            }
            for f, v in sorted(ranks.items())
        }

        verdict = {
            "controls": controls,
            "mean_cross_partition_rho": observed,
            "n_pairs": len(pairs),
            "pairs": pairs,
            "rho_bar": RHO_BAR,
            "beats_null_p95": bool(observed > controls["negative_permutation_null_p95"]),
            "per_file_rank": per_file,
            "branch": (
                "per_file_value_IS_reproducible"
                if observed > RHO_BAR
                else "reproducibility_NOT_shown"
            ),
            "rule": f"mean cross-partition rho > {RHO_BAR} -> reproducible",
            "cells": cells,
        }
        run.write_json("results.json", verdict)
        run.metrics = {"mean_cross_partition_rho": observed, "n_pairs": len(pairs)}
        log.info(
            "VERDICT %s | mean rho %+0.3f over %d pairs (bar %.2f, null p95 %.3f)",
            verdict["branch"],
            observed,
            len(pairs),
            RHO_BAR,
            controls["negative_permutation_null_p95"],
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
