"""D2d — does an OPTIMAL-TRANSPORT valuation predict per-file value where naive statistics failed?

PRE-REGISTRATION (BACKLOG `Validate` row, winds review 2026-08-23T16:28Z, opened by the literature
check at 16:23Z; frozen before the run). TIER B under the experiment gate added to study
protocol §11 on 2026-08-23 — exploratory, may OPEN a claim; run authorised by the operator.

WHY. D2 (2026-08-23T01:10Z) concluded that per-file training value is unpredictable, and tested
exactly FOUR predictors to get there: `n_rows` (|rho| 0.441), `minority_share` (0.365), `spread`
(0.259) and `centrality` (0.188), against a 0.50 bar. Every one is a naive summary statistic; the
closest in spirit, `centrality`, is a Euclidean gap between class-agnostic file centroids — a
first-moment-only proxy. The 16:23Z literature check found that the data-valuation field's answer to
exactly this problem is an OPTIMAL-TRANSPORT distance (LAVA's class-wise Wasserstein discrepancy,
plus the scalable successors SAVA and KAIROS), developed precisely because valuation is otherwise
learner-dependent. So "per-file value is unpredictable" currently means "unpredictable by four
proxies a valuation researcher would not have tried", which is materially weaker than the claim a
follow-up would want to make.

DECISION RULE, fixed now:
  |rho| <  0.5  -> even a purpose-built distributional predictor fails; the claim reaches its
                   STRONGEST form — it survives the field's own instrument, not merely four proxies
  |rho| >= 0.5  -> the claim is WRONG as stated and becomes "unpredictable from cheap statistics but
                   predictable by optimal-transport valuation" — a different practitioner message,
                   because you CAN then choose your way out and the price is an OT computation

HONESTY CONSTRAINT, fixed before the run and NOT negotiable afterwards. `POT` is not a project
dependency, so the instrument is a SLICED-Wasserstein approximation: L random unit directions, exact
one-dimensional W1 per direction via sorted quantiles, averaged. Sliced Wasserstein is a standard,
published approximation to the Wasserstein distance and it is **NOT LAVA**. Any write-up says "an
optimal-transport distributional distance" and never the method's name, unless the real algorithm is
implemented and its paper read. The class-wise structure IS reproduced (the distance is computed
within each label and combined by class share), because that is the part which distinguishes this
from a plain distributional gap and from D2's `centrality`.

PREDICTORS, all computed from TRAINING DATA ONLY — the held-out files are never touched:
  ot_classwise   class-wise sliced-Wasserstein between file f's capped rows and the POOLED other
                 training files' capped rows. The new instrument.
  ot_pooled      the same distance ignoring labels — isolates whether any class-wise structure
                 matters, or whether a plain distributional gap carries it
  n_rows, minority_share, spread, centrality   D2's four, recomputed here so all six sit on one
                 sample and the comparison is not cross-artifact

CONTROLS.
  positive (blocking)  restricted to partitions 0-2 — D2's exact partitions — the four naive
                       predictors must reproduce D2's published |rho| values (0.441 / 0.365 /
                       0.259 / 0.188) within 5e-3. This certifies the harness computes what D2 did;
                       without it, any OT number is uninterpretable against D2's bar.
  negative             a random per-file score, and a permutation null over the excesses, both
                       giving the 95th percentile of |rho| reachable by chance on this sample.
  reported             per-partition rho for the OT predictor, so a pooled correlation carried by
                       ONE partition is visible rather than hidden inside an average.

Excesses are READ from D2b's artifact and never refit, so this run carries no thread-count exposure.

Run:  OMP_NUM_THREADS=4 python -m experiments.run_file_value_ot_predictor
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from experiments.run_group_split_control import group_splits, reconstruct_groups
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger

log = get_logger(__name__)

VARIANT = "binary"
CAP = 2000
RHO_BAR = 0.5
N_PROJ = 256
N_PERM = 2000
D2B_RUN = "results/runs/20260823T083323Z_file_value_reproducibility_binary"
D2_PUBLISHED = {  # decision log 2026-08-23T01:10Z, |rho_pooled| over partitions 0-2
    "n_rows": 0.44107911476378464,
    "minority_share": 0.3651682053526683,
    "spread": 0.25946275946275943,
    "centrality": 0.18803418803418803,
}
D2_PARTITIONS = {0, 1, 2}
POS_TOL = 5e-3


def sliced_w1(
    A: np.ndarray, B: np.ndarray, rng: np.random.Generator, n_proj: int = N_PROJ
) -> float:
    """Sliced 1-Wasserstein: mean over random directions of the exact 1-D W1 on that projection.

    The 1-D W1 between two empirical measures is the L1 gap between their quantile functions, which
    for equal-size samples is the mean absolute difference of the sorted projections. The samples
    here differ in size, so both are resampled onto a COMMON quantile grid first — the standard
    treatment, and the one that keeps the distance from being driven by sample-size mismatch.
    """
    d = A.shape[1]
    dirs = rng.standard_normal((d, n_proj))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    PA = np.sort(A @ dirs, axis=0)
    PB = np.sort(B @ dirs, axis=0)
    q = np.linspace(0.0, 1.0, 200)
    qa = np.quantile(PA, q, axis=0)
    qb = np.quantile(PB, q, axis=0)
    return float(np.mean(np.abs(qa - qb)))


def ot_predictors(
    Zf: np.ndarray, yf: np.ndarray, Zr: np.ndarray, yr: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """Class-wise and label-agnostic sliced-Wasserstein between one file and the pooled rest."""
    parts, weights = [], []
    for c in (0, 1):
        a, b = Zf[yf == c], Zr[yr == c]
        if len(a) < 10 or len(b) < 10:
            continue
        parts.append(sliced_w1(a, b, rng))
        weights.append(float((yr == c).mean()))
    classwise = float(np.average(parts, weights=weights)) if parts else float("nan")
    return classwise, sliced_w1(Zf, Zr, rng)


def naive_predictors(Zf: np.ndarray, yf: np.ndarray, centroids: dict, f: int) -> dict:
    """D2's four, recomputed verbatim so the positive control is a like-for-like check."""
    c = np.bincount(yf, minlength=2)
    others = [v for k, v in centroids.items() if k != f]
    return {
        "n_rows": float(len(yf)),
        "minority_share": float(c.min() / c.sum()),
        "spread": float(np.mean(np.var(Zf, axis=0))),
        "centrality": float(np.mean([np.linalg.norm(centroids[f] - o) for o in others])),
    }


def abs_rho(x: list[float], y: list[float]) -> tuple[float, float]:
    r, p = spearmanr(x, y)
    return (float("nan"), float("nan")) if np.isnan(r) else (abs(float(r)), float(p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=10)
    args = ap.parse_args()

    cells = json.loads((REPO_ROOT / D2B_RUN / "results.json").read_text())["cells"]
    excess = {(c["partition"], c["file"]): c["excess"] for c in cells}

    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # shipped stream — must reproduce D2b's partitions
    parts = [(p, group_splits(gid, y, rng_part)[0]) for p in range(args.partitions)]

    cfg = {
        "variant": VARIANT,
        "cap": CAP,
        "n_projections": N_PROJ,
        "rho_bar": RHO_BAR,
        "d2b_reference": D2B_RUN,
        "d2_published": D2_PUBLISHED,
        "instrument": "sliced-Wasserstein approximation — NOT LAVA; see the module docstring",
        "note": "excesses READ from D2b, never refit — no thread-count exposure",
    }

    with RunDir(f"file_value_ot_predictor_{VARIANT}", config=cfg) as run:
        rows: list[dict] = []
        for p, sp in parts:
            pipe = build_preprocessor("full", n_components=None, seed=0)
            Z = fit_transform_splits(pipe, X, sp)
            ytr_all = y[sp["train"]]
            gtr = gid[sp["train"]]
            files = sorted(np.unique(gtr).tolist())

            # TWO views per file, and keeping them distinct is what the first run got wrong.
            # D2 computed its four predictors on the UNCAPPED file rows (`Zsub` at
            # run_file_quality_predictors.py:212) and trained on the capped subset. Computing the
            # naive predictors on the capped rows makes `n_rows` constant at 2,000 — its Spearman
            # is then undefined — and shifts `spread` and `centrality`. The blocking control caught
            # exactly that. Naive predictors therefore use `full`; the OT distance uses `capped`,
            # because it is bounded by the sample it is computed on and must be comparable across
            # files of very different sizes.
            full, capped = {}, {}
            for f in files:
                m = gtr == f
                Zm, ym = Z["train"][m], ytr_all[m]
                full[f] = (Zm, ym)
                idx = stratified_cap(Zm, ym, min(CAP, int(m.sum())), seed=0)
                capped[f] = (Zm[idx], ym[idx])
            centroids = {f: Z["train"][gtr == f].mean(axis=0) for f in files}  # D2's exact line

            for f in files:
                Zf, yf = capped[f]
                oth = [k for k in files if k != f]
                Zr = np.concatenate([capped[k][0] for k in oth])
                yr = np.concatenate([capped[k][1] for k in oth])
                sub = stratified_cap(Zr, yr, min(CAP, len(yr)), seed=0)
                rng = np.random.default_rng(1234)  # same directions for every cell
                cw, pooled = ot_predictors(Zf, yf, Zr[sub], yr[sub], rng)
                row = {"partition": p, "file": f, "excess": excess[(p, f)]}
                row.update(naive_predictors(full[f][0], full[f][1], centroids, f))
                row["ot_classwise"] = cw
                row["ot_pooled"] = pooled
                rows.append(row)
            log.info("p%d done | %d files", p, len(files))
            run.write_json("results_partial.json", {"rows": rows})

        names = ["ot_classwise", "ot_pooled", "n_rows", "minority_share", "spread", "centrality"]
        ex = [r["excess"] for r in rows]
        corr = {}
        for n in names:
            a, pv = abs_rho([r[n] for r in rows], ex)
            per = {}
            for p in sorted({r["partition"] for r in rows}):
                sel = [r for r in rows if r["partition"] == p]
                per[str(p)] = abs_rho([r[n] for r in sel], [r["excess"] for r in sel])[0]
            corr[n] = {
                "abs_rho_pooled": a,
                "p_value": pv,
                "per_partition_abs_rho": per,
                "clears_bar": bool(a >= RHO_BAR),
            }

        # ---- controls -------------------------------------------------------------------
        d2 = [r for r in rows if r["partition"] in D2_PARTITIONS]
        d2ex = [r["excess"] for r in d2]
        repro = {
            n: {
                "here": abs_rho([r[n] for r in d2], d2ex)[0],
                "d2_published": D2_PUBLISHED[n],
                "delta": abs(abs_rho([r[n] for r in d2], d2ex)[0] - D2_PUBLISHED[n]),
            }
            for n in D2_PUBLISHED
        }
        rng = np.random.default_rng(0)
        null = [abs_rho(list(rng.permutation(ex)), ex)[0] for _ in range(N_PERM)]
        rand_score = abs_rho(list(rng.standard_normal(len(rows))), ex)[0]
        controls = {
            "positive_d2_reproduction": repro,
            "positive_worst_delta": float(max(v["delta"] for v in repro.values())),
            "positive_pass": bool(max(v["delta"] for v in repro.values()) <= POS_TOL),
            "positive_note": "partitions 0-2 are D2's exactly; the four naive predictors must "
            "reproduce its published |rho| or no OT number is readable against D2's bar",
            "negative_permutation_null_abs_rho_p95": float(np.percentile(null, 95)),
            "negative_random_score_abs_rho": rand_score,
        }
        controls["pass"] = controls["positive_pass"]
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        best = max(names, key=lambda n: corr[n]["abs_rho_pooled"])
        verdict = {
            "controls": controls,
            "correlations": corr,
            "best_predictor": best,
            "best_abs_rho": corr[best]["abs_rho_pooled"],
            "ot_classwise_abs_rho": corr["ot_classwise"]["abs_rho_pooled"],
            "rho_bar": RHO_BAR,
            "n_cells": len(rows),
            "branch": (
                "OT_PREDICTS_per_file_value"
                if corr["ot_classwise"]["abs_rho_pooled"] >= RHO_BAR
                else "unpredictable_SURVIVES_the_OT_instrument"
            ),
            "rule": f"|rho| of ot_classwise >= {RHO_BAR} -> the unpredictability claim is WRONG",
            "rows": rows,
        }
        run.write_json("results.json", verdict)
        run.metrics = {
            "ot_classwise_abs_rho": corr["ot_classwise"]["abs_rho_pooled"],
            "best_abs_rho": corr[best]["abs_rho_pooled"],
        }
        log.info(
            "VERDICT %s | ot_classwise |rho| %.3f (bar %.2f, null p95 %.3f) | best overall %s %.3f",
            verdict["branch"],
            corr["ot_classwise"]["abs_rho_pooled"],
            RHO_BAR,
            controls["negative_permutation_null_abs_rho_p95"],
            best,
            corr[best]["abs_rho_pooled"],
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
