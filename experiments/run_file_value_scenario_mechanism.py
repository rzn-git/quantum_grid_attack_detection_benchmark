"""D2e — is the 3.5x spread in per-file training value explained by SCENARIO COMPOSITION?

PRE-REGISTRATION (BACKLOG `Spike` row "WHY is one source file worth 7.5x another?", opened from
the D2b finding 2026-08-23T08:45Z; frozen before the run). D2b established that per-file training
value is LARGE, REPRODUCIBLE (mean cross-partition rho +0.642 against a 0.129 null) and
UNPREDICTABLE from four cheap statistics (D2: best |rho| 0.441 against a 0.50 bar). D2 ruled out
row count, class balance, feature spread and centrality. If an explanation exists it should lie in
WHAT EACH FILE CONTAINS — which of the dataset's 37 staged scenarios, and in what proportion — not
in summary statistics of its feature matrix. This is the last place in the data where the mechanism
could hide before the claim reaches its final form.

WHY THIS IS ANSWERABLE AT ALL. The binary variant carries only Natural/Attack, so scenario identity
is not in the data D2b trained on. The multiclass variant ships the SAME 15 files with the SAME rows
and a 37-value `marker`; attaching it to the binary rows by position is what makes the taxonomy
visible. That alignment is an assumption, so it is a BLOCKING control below rather than a remark.

METHOD. No model is trained. D2b's per-(partition, file) excesses are READ from its artifact and
never refit, so this run carries no thread-count or seed exposure and cannot drift from what D2b
measured. Scenario composition is computed on the rows that SURVIVE preprocessing — the same
dedup mask `reconstruct_groups` applies — because a scenario dropped by dedup was never trained on.

FIVE PREDICTORS, every one computable from the taxonomy WITHOUT the held-out files:
  n_scenarios             distinct scenario ids present in the file
  scenario_entropy        Shannon entropy (nats) of the file's scenario row-shares — taxonomy
                          COVERAGE, the natural reading of "this file teaches more"
  attack_scenario_entropy the same restricted to attack scenarios, renormalized — coverage of the
                          part of the taxonomy the detector actually has to separate
  rare_scenario_share     share of the file's rows in scenarios whose POOLED count is in the bottom
                          tercile — does carrying rare material make a file valuable
  scenario_chi2_to_pool   sum over scenarios of (p_file - p_pool)^2 / p_pool — how ATYPICAL the
                          file's taxonomy coverage is, the taxonomy-side mirror of D2's `centrality`

DECISION RULE, fixed now, Spearman rho against per-file MEAN excess, n = 15:
  some predictor reaches |rho| >= 0.5 AND clears its own permutation p95
      -> the mechanism IS in the taxonomy; the paper names the statistic and file selection becomes
         an actionable mitigation after all
  no predictor does
      -> KILL CRITERION FIRES: the mechanism is not in the taxonomy either, and the claim reaches
         its final form — per-file value is large, stable, and unexplained by anything cheap

SECONDARY (the row's literal question). Do the rank-1 and rank-9 files separate on scenario mix by
more than a random pair of files? Chi-square distance between their scenario distributions, read as
a percentile of all 105 file pairs. Bar: >= 90th percentile. This can fire when the pooled rho does
not, and it is the weaker evidence of the two — a single pair, chosen because it is extreme.

CONTROLS.
  positive (blocking) A   the binary and multiclass frames must agree cell-for-cell on finite
                          features to 1e-6 across all 15 files, and no scenario id may map to BOTH
                          Natural and Attack. Without this the scenario labels are attached to the
                          wrong rows and every number below is noise. (The ARFF export rounds to six
                          decimals, so exact equality is the wrong test and 1e-6 is the right one.)
  positive (blocking) B   the per-file mean excesses recomputed from D2b's artifact must rank file
                          14 first and file 10 last, and D2b's own `per_file_rank` must agree — this
                          certifies the cells being explained are the cells the backlog row names.
  negative                2,000-shuffle permutation null per predictor giving the 95th percentile of
                          |rho| reachable by chance at n = 15, plus a random per-file score scored
                          identically. At n = 15 the null is WIDE, and reading the 0.5 bar without
                          it would be the error D2 avoided.
  reported                the full per-file scenario table, so the composition behind any verdict is
                          auditable rather than summarized; and the correlation recomputed on
                          within-partition ranks, so a pooled result cannot be an artifact of
                          partitions having different floors.

NAMING TRAP, recorded because it is easy to get wrong downstream: group ids index
`load_variant_frames`, which sorts file names LEXICOGRAPHICALLY. Group 14 is `data9.csv` and
group 10 is `data5.csv` — not `data14` and `data10`. Every id below is a group id.

AMENDMENT 2026-08-24, AFTER THE FIRST RUN (which is VOID). Two changes, both recorded rather than
folded in silently:

  (1) BUG FIX, mandatory. `attack_scenario_entropy` was computed by testing string composition keys
      against INTEGER attack ids, so the membership test never matched, every file scored a
      constant 0, and the predictor measured nothing. The pre-registered five-predictor rule is
      unchanged; the first run simply did not execute it.

  (2) POST-HOC DIAGNOSTICS, and they are NOT part of the decision rule. The first run put
      `scenario_entropy` at rho -0.632, i.e. files covering MORE of the taxonomy are worth LESS.
      That is surprising, and under the standing discipline a surprising result is a bug until an
      independent path reproduces it and the boring causes are named. Three checks are added, all
      REPORTED ONLY, none able to change the branch:
        - each predictor's correlation with `n_rows`, the statistic D2 already tested, so an
          entropy signal that is really row count in disguise is visible rather than inferred
        - `rows_per_scenario`, the depth reading of the same fact: D2b trained on a 2,000-row
          stratified cap, so a file spread across more scenarios contributes fewer rows to each
        - entropy recomputed on the CAPPED subset D2b actually trained on, an independent path to
          the same quantity; if the effect is a property of the cap it must survive there

Run:  OMP_NUM_THREADS=4 python -m experiments.run_file_value_scenario_mechanism
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qgridbench.data.validate import load_variant_frames
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
D2B_RUN = "results/runs/20260823T083323Z_file_value_reproducibility_binary"
RHO_BAR = 0.5  # D2's bar, kept identical so the two runs are directly comparable
ALIGN_TOL = 1e-6  # ARFF rounds to six decimals; this is the rounding envelope, not a fudge
N_PERM = 2000
PAIR_PCTL_BAR = 90.0
CAP = 2000  # D2b's training cap; the capped-subset diagnostic must use the same value
TOP_FILE, BOTTOM_FILE = 14, 10  # group ids from D2b's ranking; data9.csv and data5.csv


def aligned_scenarios():
    """Scenario id per row of the binary variant, with the alignment control's evidence.

    Returns (frames_binary, scenario_per_file, control_dict). The control is computed here rather
    than asserted, so its numbers land in the artifact whether it passes or fails.
    """
    fb = load_variant_frames("binary")
    fm = load_variant_frames("multiclass")
    label_col = load_yaml(REPO_ROOT / "configs" / "data.yaml")["schema"]["label_column"]

    worst = 0.0
    shape_ok = len(fb) == len(fm)
    for a, b in zip(fb, fm):
        A = a.drop(columns=[label_col]).to_numpy(float)
        B = b.drop(columns=[label_col]).to_numpy(float)
        if A.shape != B.shape:
            shape_ok = False
            continue
        finite = np.isfinite(A) & np.isfinite(B)
        if finite.any():
            worst = max(worst, float(np.abs(A[finite] - B[finite]).max()))

    # No scenario id may carry both binary labels; a shuffled join would produce many that do.
    cross = defaultdict(Counter)
    for a, b in zip(fb, fm):
        for bm, mm in zip(a[label_col], b[label_col]):
            cross[str(mm).strip()][str(bm).strip()] += 1
    ambiguous = sorted(k for k, v in cross.items() if len(v) > 1)
    natural = sorted((int(k) for k, v in cross.items() if "Natural" in v))
    attack = sorted((int(k) for k, v in cross.items() if "Attack" in v))

    control = {
        "shapes_match": bool(shape_ok),
        "worst_abs_feature_delta": worst,
        "tol": ALIGN_TOL,
        "ambiguous_scenario_ids": ambiguous,
        "natural_scenarios": natural,
        "attack_scenarios": attack,
        "pass": bool(shape_ok and worst <= ALIGN_TOL and not ambiguous),
        "note": (
            "worst delta is the ARFF six-decimal rounding envelope; zero ambiguous ids across all "
            "rows is the independent evidence that the join is positional-correct"
        ),
    }
    scen = [b[label_col].astype(str).str.strip().to_numpy() for b in fm]
    # STRING ids: the composition counters are keyed by the raw marker text, and comparing
    # those against int ids silently matches nothing. This bug voided the first run.
    return fb, scen, control, {str(a) for a in attack}


def surviving_scenarios(frames, scen):
    """Scenario ids per file AFTER preprocess dedup — mirrors reconstruct_groups exactly."""
    cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    label_col = cfg["schema"]["label_column"]
    gid_raw = np.concatenate([np.full(len(f), i, dtype=np.int64) for i, f in enumerate(frames)])
    scen_raw = np.concatenate(scen)
    pooled = pd.concat(frames, ignore_index=True)

    label_map = cfg["labels"].get(VARIANT)
    y = pooled[label_col].map(label_map).to_numpy(dtype=np.int64)
    X = pooled.drop(columns=[label_col]).to_numpy(dtype=np.float64)
    X[np.isinf(X)] = np.nan

    df = pd.DataFrame(X)
    df["__y"] = y
    keep = (~df.duplicated()).to_numpy()
    return gid_raw[keep], scen_raw[keep], y[keep], int((~keep).sum())


def predictors(gid, scen, attack_ids):
    """Five taxonomy predictors per file, plus the raw composition table."""
    files = sorted(set(gid.tolist()))
    pooled = Counter(scen.tolist())
    counts = {f: Counter(scen[gid == f].tolist()) for f in files}

    # "Rare" is a property of the POOLED corpus, so it is knowable without the held-out files.
    by_size = sorted(pooled, key=lambda s: pooled[s])
    rare = set(by_size[: max(1, len(by_size) // 3)])
    n_pool = sum(pooled.values())
    p_pool = {s: pooled[s] / n_pool for s in pooled}

    out = {}
    for f in files:
        c = counts[f]
        n = sum(c.values())
        p = np.array([c[s] / n for s in c])
        atk = {s: v for s, v in c.items() if s in attack_ids}
        n_atk = sum(atk.values())
        pa = np.array([v / n_atk for v in atk.values()]) if n_atk else np.array([1.0])
        chi2 = sum((c.get(s, 0) / n - p_pool[s]) ** 2 / p_pool[s] for s in p_pool)
        out[f] = {
            "n_rows": n,
            "n_scenarios": float(len(c)),
            "scenario_entropy": float(-(p * np.log(p)).sum()),
            "attack_scenario_entropy": float(-(pa * np.log(pa)).sum()),
            "rare_scenario_share": float(sum(v for s, v in c.items() if s in rare) / n),
            # POST-HOC diagnostic, not in the decision rule: the depth reading of coverage.
            "rows_per_scenario": float(n / len(c)),
            "scenario_chi2_to_pool": float(chi2),
            "composition": {s: c[s] for s in sorted(c, key=lambda x: int(x))},
        }
    return out, sorted(rare, key=int)


def perm_null(x, yv, rng, n=N_PERM):
    """95th percentile of |rho| reachable by shuffling the excess against a fixed predictor."""
    vals = []
    yv = np.asarray(yv, dtype=float)
    for _ in range(n):
        r = spearmanr(x, rng.permutation(yv)).statistic
        vals.append(abs(r) if np.isfinite(r) else 0.0)
    return float(np.percentile(vals, 95)), float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = set_all_seeds(args.seed)

    cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    log.info("D2e — scenario-composition mechanism for per-file training value")

    with RunDir(f"file_value_scenario_mechanism_{VARIANT}", config=cfg, seeds=[args.seed]) as run:
        _body(run, rng)


def _body(run, rng):
    # ---- blocking control A: the join the whole run rests on -------------------------------
    frames, scen, align, attack_ids = aligned_scenarios()
    log.info(
        "alignment control: worst delta %.2e, ambiguous ids %s, pass=%s",
        align["worst_abs_feature_delta"],
        align["ambiguous_scenario_ids"],
        align["pass"],
    )
    if not align["pass"]:
        run.write_json(
            "results.json", {"controls": {"alignment": align}, "status": "ABORTED_ON_CONTROL"}
        )
        raise SystemExit("alignment control FAILED — scenario labels cannot be trusted")

    gid, scen_kept, y_kept, n_dropped = surviving_scenarios(frames, scen)
    log.info("dedup dropped %d rows; %d survive", n_dropped, len(gid))

    # ---- blocking control B: the cells being explained are D2b's cells ----------------------
    d2b = json.load(open(REPO_ROOT / D2B_RUN / "results.json", encoding="utf-8"))
    by_file = defaultdict(list)
    for c in d2b["cells"]:
        by_file[int(c["file"])].append(float(c["excess"]))
    mean_excess = {f: float(np.mean(v)) for f, v in by_file.items()}
    order = sorted(mean_excess, key=lambda f: -mean_excess[f])
    ranks = d2b["per_file_rank"]
    ctrl_b = {
        "top_file": order[0],
        "bottom_file": order[-1],
        "expected_top": TOP_FILE,
        "expected_bottom": BOTTOM_FILE,
        "mean_excess_top": mean_excess[order[0]],
        "mean_excess_bottom": mean_excess[order[-1]],
        "ratio_mean_excess": mean_excess[order[0]] / mean_excess[order[-1]],
        "ratio_max_over_min_cell": (
            max(c["excess"] for c in d2b["cells"]) / min(c["excess"] for c in d2b["cells"])
        ),
        "top_rank1_count": sum(1 for r in ranks[str(TOP_FILE)]["ranks"] if r == 1),
        "top_n_partitions": ranks[str(TOP_FILE)]["n_partitions"],
        "bottom_rank9_count": sum(1 for r in ranks[str(BOTTOM_FILE)]["ranks"] if r == 9),
        "bottom_n_partitions": ranks[str(BOTTOM_FILE)]["n_partitions"],
        "pass": order[0] == TOP_FILE and order[-1] == BOTTOM_FILE,
    }
    log.info(
        "D2b integrity: top=%d bottom=%d ratio=%.2fx pass=%s",
        ctrl_b["top_file"],
        ctrl_b["bottom_file"],
        ctrl_b["ratio_mean_excess"],
        ctrl_b["pass"],
    )
    if not ctrl_b["pass"]:
        run.write_json(
            "results.json",
            {"controls": {"alignment": align, "d2b": ctrl_b}, "status": "ABORTED_ON_CONTROL"},
        )
        raise SystemExit("D2b integrity control FAILED — not the cells the row names")

    # ---- predictors and the primary correlation --------------------------------------------
    preds, rare_ids = predictors(gid, scen_kept, attack_ids)
    files = sorted(mean_excess)  # only files that appear as training files in D2b
    yv = [mean_excess[f] for f in files]
    names = [
        "n_scenarios",
        "scenario_entropy",
        "attack_scenario_entropy",
        "rare_scenario_share",
        "scenario_chi2_to_pool",
    ]

    primary = {}
    for nm in names:
        x = [preds[f][nm] for f in files]
        rho = spearmanr(x, yv).statistic
        p95, pmean = perm_null(x, yv, rng)
        primary[nm] = {
            "rho": float(rho) if np.isfinite(rho) else 0.0,
            "abs_rho": abs(float(rho)) if np.isfinite(rho) else 0.0,
            "perm_null_p95": p95,
            "perm_null_mean": pmean,
            "clears_bar": abs(float(rho)) >= RHO_BAR if np.isfinite(rho) else False,
            "clears_null": (abs(float(rho)) > p95) if np.isfinite(rho) else False,
        }
        log.info("  %-24s rho %+.3f  (null p95 %.3f)", nm, primary[nm]["rho"], p95)

    # ---- POST-HOC diagnostics (reported only; cannot change the branch) ---------------------
    n_rows_vec = [preds[f]["n_rows"] for f in files]
    confound = {
        nm: float(spearmanr([preds[f][nm] for f in files], n_rows_vec).statistic)
        for nm in names
        if np.isfinite(spearmanr([preds[f][nm] for f in files], n_rows_vec).statistic)
    }
    depth_rho = spearmanr([preds[f]["rows_per_scenario"] for f in files], yv).statistic

    # INDEPENDENT PATH: entropy on the 2,000-row stratified cap D2b actually trained on, not on
    # the whole file. Same quantity, different sample; if the effect is a property of the cap it
    # has to survive here, and if it is an artifact of whole-file counts it should not.
    capped_ent = {}
    for f in files:
        m = gid == f
        sc_f, y_f = scen_kept[m], y_kept[m]
        idx = stratified_cap(np.zeros((len(y_f), 1)), y_f, CAP, seed=0)
        cc = Counter(sc_f[idx].tolist())
        tot = sum(cc.values())
        pc = np.array([v / tot for v in cc.values()])
        capped_ent[f] = float(-(pc * np.log(pc)).sum())
    capped_rho = spearmanr([capped_ent[f] for f in files], yv).statistic
    posthoc = {
        "note": "reported only; declared in the amendment; none of these can change the branch",
        "predictor_vs_n_rows_rho": confound,
        "rows_per_scenario_rho": float(depth_rho) if np.isfinite(depth_rho) else 0.0,
        "capped_subset_entropy_rho": float(capped_rho) if np.isfinite(capped_rho) else 0.0,
        "capped_subset_entropy": {str(f): capped_ent[f] for f in capped_ent},
    }
    log.info(
        "post-hoc: entropy-vs-nrows %+.3f | rows_per_scenario %+.3f | capped-subset entropy %+.3f",
        confound.get("scenario_entropy", float("nan")),
        posthoc["rows_per_scenario_rho"],
        posthoc["capped_subset_entropy_rho"],
    )

    rand_scores = rng.normal(size=len(files))
    rand_rho = spearmanr(rand_scores, yv).statistic
    negative = {
        "random_predictor_abs_rho": abs(float(rand_rho)),
        "worst_perm_p95": max(v["perm_null_p95"] for v in primary.values()),
    }

    # ---- reported: within-partition ranks, so a pooled result cannot be a floor artifact ----
    per_part = defaultdict(dict)
    for c in d2b["cells"]:
        per_part[int(c["partition"])][int(c["file"])] = float(c["excess"])
    rank_pairs = []
    for p, cells in per_part.items():
        fs = sorted(cells, key=lambda f: -cells[f])
        for r, f in enumerate(fs, 1):
            rank_pairs.append((f, r))
    rank_by_file = defaultdict(list)
    for f, r in rank_pairs:
        rank_by_file[f].append(r)
    mean_rank = {f: float(np.mean(v)) for f, v in rank_by_file.items()}
    within = {}
    for nm in names:
        x = [preds[f][nm] for f in files]
        rr = spearmanr(x, [-mean_rank[f] for f in files]).statistic
        within[nm] = float(rr) if np.isfinite(rr) else 0.0

    # ---- secondary: does the extreme pair separate more than a random pair? -----------------
    all_files = sorted(preds)
    keys = sorted({s for f in all_files for s in preds[f]["composition"]}, key=int)

    def dist(a, b):
        na = sum(preds[a]["composition"].values())
        nb = sum(preds[b]["composition"].values())
        pa = np.array([preds[a]["composition"].get(s, 0) / na for s in keys])
        pb = np.array([preds[b]["composition"].get(s, 0) / nb for s in keys])
        m = (pa + pb) / 2
        ok = m > 0
        return float((((pa[ok] - pb[ok]) ** 2) / m[ok]).sum())

    pair_d = [dist(a, b) for i, a in enumerate(all_files) for b in all_files[i + 1 :]]
    target = dist(TOP_FILE, BOTTOM_FILE)
    pctl = float(100.0 * np.mean([d <= target for d in pair_d]))
    secondary = {
        "pair": [TOP_FILE, BOTTOM_FILE],
        "chi2_distance": target,
        "percentile_among_all_pairs": pctl,
        "n_pairs": len(pair_d),
        "median_pair_distance": float(np.median(pair_d)),
        "bar": PAIR_PCTL_BAR,
        "separates": pctl >= PAIR_PCTL_BAR,
    }
    log.info(
        "secondary: pair chi2 %.4f -> %.1fth percentile of %d pairs", target, pctl, len(pair_d)
    )

    best = max(primary, key=lambda k: primary[k]["abs_rho"])
    mechanism_found = any(v["clears_bar"] and v["clears_null"] for v in primary.values())
    branch = (
        "mechanism_IS_in_the_taxonomy"
        if mechanism_found
        else "KILL_CRITERION_FIRES_mechanism_not_in_taxonomy"
    )

    results = {
        "controls": {
            "alignment": align,
            "d2b_integrity": ctrl_b,
            "negative": negative,
            "pass": align["pass"] and ctrl_b["pass"],
        },
        "primary": primary,
        "best_predictor": best,
        "best_abs_rho": primary[best]["abs_rho"],
        "rho_bar": RHO_BAR,
        "within_partition_rank_rho": within,
        "posthoc_diagnostics": posthoc,
        "secondary_pair_separation": secondary,
        "per_file": {str(f): {k: v for k, v in preds[f].items()} for f in preds},
        "mean_excess": {str(f): mean_excess[f] for f in mean_excess},
        "mean_rank": {str(f): mean_rank[f] for f in mean_rank},
        "rare_scenario_ids": rare_ids,
        "n_rows_dropped_by_dedup": n_dropped,
        "branch": branch,
        "rule": f"any |rho| >= {RHO_BAR} AND above its permutation p95 -> mechanism is in taxonomy",
    }
    run.write_json("results.json", results)
    run.metrics = {
        "best_predictor": best,
        "best_abs_rho": primary[best]["abs_rho"],
        "mechanism_found": mechanism_found,
        "pair_percentile": secondary["percentile_among_all_pairs"],
    }
    log.info("BRANCH: %s (best %s, |rho| %.3f)", branch, best, primary[best]["abs_rho"])


if __name__ == "__main__":
    main()
