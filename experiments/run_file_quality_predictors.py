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

R4-16 RE-SCORE (decision log 2026-11-16T11:34Z, round-4 part-B amendment): `--rescore-from <D2b
run>` fits nothing. It reads D2b's 90 per-(partition, file) XGBoost excesses (10 partitions),
recomputes the four predictors above per (partition, file) with `predictors_for` on the same
shipped partition stream, averages predictor and excess per FILE over the partitions the file
trains in (n = 15), and scores Spearman rho against two permutation nulls of the per-file excess:
each predictor's own p95, and the family-wise p95 of the max |rho| over the four predictors.
  PREDICTABLE      some predictor: |rho| >= 0.50 AND family-wise p < 0.05
  INTERMEDIATE     otherwise, some predictor beats its own null (own p < 0.05)
  NOT PREDICTABLE  no predictor beats its own null
Blocking anchors: the 10 partitions reproduce D2b's (partition, file) keys exactly; n_rows and
minority_share rho reproduce D2e (+0.6393, +0.3321) to 1e-4; partitions 0-2 reproduce D2's archived
predictor values to 1e-9. Supplementary second learner: the same rhos against D2c's archived MLP
per-file excess (`--mlp-from`). Serial by measured profile: the whole re-score is under a minute
(one partition's preprocessing 0.54 s, peak 0.57 GB; tmp/r4_tier3 probe), below pool start-up cost.

Run:  python -m experiments.run_file_quality_predictors --rescore-from \
        results/runs/20260823T083323Z_file_value_reproducibility_binary
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
D2_RUN = "results/runs/20260823T010202Z_file_quality_predictors_binary"  # R4-16 anchor 3
D2C_MLP_RUN = "results/runs/20260823T154055Z_file_value_cross_model_mlp_binary"
N_PERM_RESCORE = 10_000
D2_PRED_TOL = 1e-9


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


def perm_rhos(preds: np.ndarray, y: np.ndarray, n_perm: int, rng) -> np.ndarray:
    """|Spearman rho| of each predictor row against n_perm permutations of y: (n_perm, n_pred).

    Spearman is Pearson on ranks, and permuting y then ranking equals permuting y's ranks, so
    one rank vector is permuted and correlated with every predictor in one product.
    """
    rx = np.apply_along_axis(rankdata, 1, preds)
    rx = (rx - rx.mean(1, keepdims=True)) / rx.std(1, keepdims=True)
    ry = rankdata(y)
    perm = np.stack([rng.permutation(ry) for _ in range(n_perm)])
    perm = (perm - perm.mean(1, keepdims=True)) / perm.std(1, keepdims=True)
    return np.abs(perm @ rx.T) / len(y)


def rescore_verdict(rho: dict, null: np.ndarray, names: tuple, bar: float) -> dict:
    """Own-null and family-wise p per predictor, and the R4-16 branch."""
    fw_max = null.max(axis=1)
    out = {}
    for j, nm in enumerate(names):
        a = abs(rho[nm])
        out[nm] = {
            "rho": rho[nm],
            "own_p": float((np.sum(null[:, j] >= a) + 1) / (len(null) + 1)),
            "own_p95": float(np.percentile(null[:, j], 95)),
            "fw_p": float((np.sum(fw_max >= a) + 1) / (len(null) + 1)),
        }
    fw_p95 = float(np.percentile(fw_max, 95))
    if any(abs(v["rho"]) >= bar and v["fw_p"] < 0.05 for v in out.values()):
        branch = "PREDICTABLE"
    elif any(v["own_p"] < 0.05 for v in out.values()):
        branch = "INTERMEDIATE"
    else:
        branch = "NOT_PREDICTABLE"
    return {"per_predictor": out, "familywise_p95": fw_p95, "branch": branch}


def per_file_means(cells: list[dict], key: str) -> dict:
    acc: dict = {}
    for c in cells:
        acc.setdefault(int(c["file"]), []).append(float(c[key]))
    return {f: float(np.mean(v)) for f, v in sorted(acc.items())}


def rescore_n15(src: str, mlp_src: str, n_perm: int, seed: int) -> None:
    """R4-16: D2's four predictors re-scored on per-file mean excess at n = 15 (no fitting)."""
    from experiments.runner_guard import assert_clean_predictor

    cells = json.loads((REPO_ROOT / src / "results.json").read_text())["cells"]
    mlp_cells = json.loads((REPO_ROOT / mlp_src / "results.json").read_text())["cells"]
    d2 = json.loads((REPO_ROOT / D2_RUN / "results.json").read_text())["sweep"]
    parts_n = 1 + max(int(c["partition"]) for c in cells)
    want_keys = {(int(c["partition"]), int(c["file"])) for c in cells}

    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # shipped stream, as D2 / D2b / D2c
    rows = []
    for p in range(parts_n):
        sp, _held = group_splits(gid, y, rng_part)
        Z = fit_transform_splits(build_preprocessor("full", n_components=None, seed=0), X, sp)
        gtr, ytr = gid[sp["train"]], y[sp["train"]]
        files = sorted(np.unique(gtr).tolist())
        centroids = {f: Z["train"][gtr == f].mean(axis=0) for f in files}
        for f in files:
            m = gtr == f
            rows.append(
                {"partition": p, "file": f, **predictors_for(Z["train"][m], ytr[m], centroids, f)}
            )

    got_keys = {(r["partition"], r["file"]) for r in rows}
    if got_keys != want_keys:
        raise RuntimeError(f"partition keys differ from {src}: {got_keys ^ want_keys}")
    d2_by = {(x["partition"], x["file"]): x for x in d2}
    d2_dev = max(
        abs(r[k] - d2_by[(r["partition"], r["file"])][k])
        for r in rows
        if (r["partition"], r["file"]) in d2_by
        for k in PREDICTORS
    )
    if d2_dev > D2_PRED_TOL:
        raise RuntimeError(f"predictors do not reproduce D2's archived values: max dev {d2_dev}")

    files = sorted({r["file"] for r in rows})
    pred = {k: per_file_means(rows, k) for k in PREDICTORS}
    xgb = per_file_means(cells, "excess")
    mlp = per_file_means(mlp_cells, "excess")
    if set(mlp) != set(xgb):
        raise RuntimeError("D2c MLP cells do not cover the same files as D2b")
    P = np.array([[pred[k][f] for f in files] for k in PREDICTORS])
    yx = np.array([xgb[f] for f in files])
    ym = np.array([mlp[f] for f in files])
    for k, row in zip(PREDICTORS, P):
        assert_clean_predictor(k, yx, row, row)

    rho_x = {k: float(spearmanr(P[i], yx).statistic) for i, k in enumerate(PREDICTORS)}
    rho_m = {k: float(spearmanr(P[i], ym).statistic) for i, k in enumerate(PREDICTORS)}
    anchors = {"n_rows": 0.6393, "minority_share": 0.3321}  # D2e, logged 2026-08-24T01:01Z
    a_dev = {k: abs(rho_x[k] - v) for k, v in anchors.items()}
    if max(a_dev.values()) > 1e-4:
        raise RuntimeError(f"D2e anchors not reproduced: {a_dev}")

    rng = np.random.default_rng(seed)
    v_x = rescore_verdict(rho_x, perm_rhos(P, yx, n_perm, rng), PREDICTORS, RHO_BAR)
    v_m = rescore_verdict(rho_m, perm_rhos(P, ym, n_perm, rng), PREDICTORS, RHO_BAR)
    rand = rng.normal(size=len(files))

    # sensitivity (reported only): leave one partition out, and within-partition mean ranks
    lopo = {k: [] for k in PREDICTORS}
    for p in range(parts_n):
        keep_c = [c for c in cells if int(c["partition"]) != p]
        keep_r = [r for r in rows if r["partition"] != p]
        ex = per_file_means(keep_c, "excess")
        fs = sorted(ex)
        for k in PREDICTORS:
            pk = per_file_means(keep_r, k)
            lopo[k].append(float(spearmanr([pk[f] for f in fs], [ex[f] for f in fs]).statistic))
    ranked = []
    for p in range(parts_n):
        cp = [c for c in cells if int(c["partition"]) == p]
        r = rankdata([-float(c["excess"]) for c in cp])
        ranked += [{"file": c["file"], "rank": float(rr)} for c, rr in zip(cp, r)]
    mean_rank = per_file_means(ranked, "rank")
    within = {
        k: float(spearmanr(P[i], [-mean_rank[f] for f in files]).statistic)
        for i, k in enumerate(PREDICTORS)
    }

    cfg = {
        "mode": "rescore_n15",
        "source": src,
        "mlp_source": mlp_src,
        "n_permutations": n_perm,
        "perm_seed": seed,
        "rho_bar": RHO_BAR,
        "amendment": "decision log 2026-11-16T11:34Z",
    }
    with RunDir(f"file_quality_rescore_n15_{VARIANT}", config=cfg, seeds=[seed]) as run:
        res = {
            "branch": v_x["branch"],
            "xgb_primary": v_x,
            "mlp_supplementary": v_m,
            "controls": {
                "partition_keys_match": True,
                "d2_predictor_max_abs_dev": float(d2_dev),
                "d2e_anchor_abs_dev": a_dev,
                "random_predictor_abs_rho": abs(float(spearmanr(rand, yx).statistic)),
                "pass": True,
            },
            "sensitivity": {
                "leave_one_partition_out_rho": {
                    k: [float(min(v)), float(max(v))] for k, v in lopo.items()
                },
                "within_partition_mean_rank_rho": within,
                "n_partitions_per_file": {
                    str(f): sum(1 for c in cells if int(c["file"]) == f) for f in files
                },
            },
            "per_file": {
                str(f): {**{k: pred[k][f] for k in PREDICTORS}, "xgb": xgb[f], "mlp": mlp[f]}
                for f in files
            },
        }
        run.write_json("results.json", res)
        run.metrics.update(
            {
                "branch": v_x["branch"],
                "familywise_p95": v_x["familywise_p95"],
                **{f"rho_{k}": rho_x[k] for k in PREDICTORS},
            }
        )
        log.info(
            "R4-16 %s | fw p95 %.3f | %s",
            v_x["branch"],
            v_x["familywise_p95"],
            {k: round(v, 3) for k, v in rho_x.items()},
        )
        print(f"-> {run.path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--rescore-from", default=None, help="D2b run dir: R4-16 n=15 re-score")
    ap.add_argument("--mlp-from", default=D2C_MLP_RUN, help="D2c MLP run dir (supplementary)")
    ap.add_argument("--n-perm", type=int, default=N_PERM_RESCORE)
    args = ap.parse_args()
    if args.rescore_from:
        rescore_n15(args.rescore_from, args.mlp_from, args.n_perm, seed=0)
        return

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
