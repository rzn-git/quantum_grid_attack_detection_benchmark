"""Generate every paper table + figure from finalized run artifacts.

No number is ever hand-typed: this script is the ONLY writer of paper/tables/*.tex
and figures/*.pdf. Re-runnable and idempotent — pointing it at the current run
dirs regenerates the full reporting surface.

Headline statistics (spec section 5): paired Wilcoxon signed-rank across seeds with
Holm-Bonferroni correction over the family of headline comparisons, plus effect
sizes.

Run:  python -m experiments.make_report
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from qgridbench.eval.baselines import trivial_baselines
from qgridbench.eval.stats import holm_bonferroni, paired_wilcoxon, robustness_stats
from qgridbench.eval.tables import (
    MULTICLASS_CONFUSION_FILES,
    TABLE_DIR,
    accuracy_table,
    blackbox_table,
    cost_table,
    diagnostics_combined_table,
    group_split_partitions_table,
    group_split_table,
    hsj_query_sweep_table,
    kernel_exhaustion_table,
    matched_contest_table,
    multiclass_confusion_counts,
    objective_regime_table,
    quantum_bandwidth_table,
    quantum_diagnostics_table,
    ray_crossings_table,
    regime_table,
    relabel_control_table,
    robustness_summary_table,
    surrogate_control_table,
    three_class_confusion_table,
)
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml

log = get_logger(__name__)
RUNS = REPO_ROOT / "results" / "runs"

# Anchored on the stamp suffix: a bare `*surrogate_control_binary_q8` also matches the newer
# `<stamp>_group_surrogate_control_binary_q8` runs, whose results.json has no `summary`.
SURROGATE_CONTROL_GLOB = "*Z_surrogate_control_binary_q8"
# The newer tabular models (decision log 2026-09-01: V2-E1 TabICLv2, V2-E3 FT-Transformer at
# literature defaults, V2-E4 TabM, N-CAP capacity-matched LightGBM) and the gas accuracy verdict
# under the shared pipeline (R4-12) and with pressure log-scaled (R5-G1 option B).
TABULAR_RUNS = {
    "tabicl": "20260901T112903Z_tabicl_arm_binary",
    "ftt": "20260901T114347Z_deep_tabular_ftt_default_binary",
    "tabm": "20260901T131706Z_deep_tabular_tabm_binary",
    "capacity": "20260901T180502Z_capacity_matched_control_binary",
}
# The cap-cost probe behind the bridge chart of RQ1 (experiments/run_kernel_cap_cost.py): the
# run with the fixed-hyperparameter path, at the archived run's thread count (decision log
# 2026-11-20, probe CC2).
CAP_COST_RUN = "20261120T005230Z_probe_kernel_cap_cost_binary"
GAS_ACCURACY_RUNS = {
    "shared": "20261116T180611Z_gas_accuracy_verdict_gas_binary",
    "slog": "20261119T113544Z_gas_accuracy_verdict_gas_binary_slog",
}
# Every exhibit with a VQC in it reads the VQC at Table I's configuration (depth 6). Those
# runs come from the R4-23/R4-24 regeneration (decision log 2026-11-18T16:29Z, run map
# tmp/r4_24/run_map.json) and carry name suffixes that the `latest` globs below exclude, so
# they are pinned here; a missing pin fails the build.
VQC_TABLE1 = {
    "adversarial": "20261117T075906Z_adversarial_binary_q8_vqc_table1",
    "random": "20261118T145335Z_adversarial_binary_q8_random_vqc_table1",
    "surrogate": "20261116T122057Z_vqc_config_check_binary_q8",
    "p7b8": "20261116T201704Z_whitebox_qsvm_binary_q8_vqc_table1_p7b",
    "p7b12": "20261117T062637Z_whitebox_qsvm_binary_q12_vqc_table1",
    "data_efficiency": "20261118T160923Z_ablations_binary_q8_vqc_table1",
}


# E-M4b (decision log 2026-11-16T05:40Z, branch weighted_becomes_baseline): the MLP row of the
# PCA-8 accuracy table is the class-weighted MLP (arm W1), and Table III adds it as its own row.
WEIGHTED_MLP_RUN = "20261116T051946Z_weighted_mlp_baseline_binary_pca8"


def weighted_mlp_run() -> Path:
    path = RUNS / WEIGHTED_MLP_RUN
    if not (path / "results.json").exists():
        raise FileNotFoundError(f"pinned weighted-MLP run missing: {path}")
    return path


# Logistic regression's own white-box attack (experiments/run_whitebox_logreg.py; decision log
# 2026-11-21, experiment logreg_whitebox): the shared protocol reaches it by transfer only.
LOGREG_WHITEBOX_RUN = "20261121T125627Z_whitebox_logreg_binary_q8_clean_tree"


def logreg_whitebox_run() -> Path:
    path = RUNS / LOGREG_WHITEBOX_RUN
    if not (path / "results.json").exists():
        raise FileNotFoundError(f"pinned logistic-regression white-box run missing: {path}")
    return path


# Recall per class under both splits (experiments/run_level_per_class.py, design "file"; decision
# log 2026-11-19T08:39Z, NS13). Panel (c) of the split figure reads it; the figure fails unless
# this run reproduced the pinned group-split run.
LEVEL_PER_CLASS_RUN = "20261119T082822Z_level_per_class_file_binary"


def level_per_class_run() -> Path:
    path = RUNS / LEVEL_PER_CLASS_RUN
    if not (path / "results.json").exists():
        raise FileNotFoundError(f"pinned per-class recall run missing: {path}")
    return path


# HopSkipJump restarts: one archive-only run pools the restart runs each model's black-box cell
# quotes (experiments/pool_hsj_restarts.py reads experiments/hsj_restart_pool.yaml, which names
# the Table I VQC's run too; operator ruling (i), decision log 2026-11-20T09:19Z). A newer
# restart run enters the exhibits only through the pool file and a new pin here.
HSJ_POOLED_RUN = "20261120T094143Z_hsj_restarts_pooled_binary_q8"

# The score-based black-box attacks on the main dataset (three seeds, about 600 queries per
# point, radii 0.1, 0.3 and 0.5 sigma): the robustness table reads each at its worst tested
# radius for every model's lowest attacked score. The Square search of 2026-09-01 (V2-E2,
# decision log 2026-09-01T11:05Z) attacked the VQC's earlier configuration, so the VQC is read
# from the Square run on its Table I configuration (finding T2-a, 2026-11-18). The evolution-
# strategies run attacked the trees and the MLP. The table checks, seed by seed, that each run
# attacked the fit it prints.
SQUARE_RUN = "20260901T105256Z_square_blackbox_binary_q8"
SQUARE_MODELS = ("logreg", "rbf_svm", "rf", "xgb", "lgbm", "mlp", "qsvm")
SQUARE_VQC_RUN = "20261118T152316Z_square_blackbox_binary_q8_vqc_table1"
EVOLUTION_RUN = "20260822T173155Z_score_based_blackbox_binary_q8"
EVOLUTION_MODELS = ("rf", "xgb", "lgbm", "mlp")


def hsj_pooled_run() -> Path:
    path = RUNS / HSJ_POOLED_RUN
    if not (path / "results.json").exists():
        raise FileNotFoundError(f"pinned pooled HopSkipJump run missing: {path}")
    # the pin is only as current as the pool file it was built from
    built_from = yaml.safe_load((path / "params.yaml").read_text(encoding="utf-8"))["config"]
    pool_file = built_from.pop("pool_file")
    if built_from != load_yaml(REPO_ROOT / pool_file):
        raise ValueError(
            f"{pool_file} changed since the pinned pooled HopSkipJump run {HSJ_POOLED_RUN} was "
            "built: run `python -m experiments.pool_hsj_restarts` and pin the new run"
        )
    return path


# The HopSkipJump query-budget sweep: one archive-only run reads the restart runs named in
# experiments/hsj_query_sweep.yaml against the registered rules, then by direction and row by
# row against the noise control (experiments/read_hsj_query_sweep.py; decision log 2026-11-22,
# 14:40Z and 15:34Z). The complete build's sweep table reads it.
HSJ_SWEEP_VERDICT_RUN = "20261122T153709Z_hsj_query_sweep_verdict_binary_q8"
HSJ_SWEEP_FILE = "experiments/hsj_query_sweep.yaml"


def hsj_sweep_verdict_run() -> Path:
    path = RUNS / HSJ_SWEEP_VERDICT_RUN
    if not (path / "results.json").exists():
        raise FileNotFoundError(f"pinned query-sweep verdict run missing: {path}")
    # the pin is only as current as the sweep file it was built from
    built_from = yaml.safe_load((path / "params.yaml").read_text(encoding="utf-8"))["config"]
    verdict = json.loads((path / "results.json").read_text(encoding="utf-8"))
    spec = load_yaml(REPO_ROOT / HSJ_SWEEP_FILE)
    current = {name: budget["families"] for name, budget in spec["budgets"].items()}
    noise = verdict["after_the_result"]["noise_control"]
    same_noise = all(noise[k] == v for k, v in spec["noise_control"].items())
    same_rules = all(spec[k] == v for k, v in verdict["rules"].items())
    if built_from["sources"] != current or not same_noise or not same_rules:
        raise ValueError(
            f"{HSJ_SWEEP_FILE} changed since the pinned verdict run {HSJ_SWEEP_VERDICT_RUN} was "
            "built: run `python -m experiments.read_hsj_query_sweep` and pin the new run"
        )
    return path


# White-box cells: one archive-only run reads every stored white-box curve at the worst grid
# radius inside the budget (experiments/read_worst_radius.py reads
# experiments/worst_radius_sources.yaml; operator ruling (h), decision log 2026-11-20T10:17Z).
# The robustness table, panels 3 and 7 and replication rows 4, 8 and 10 read it.
WORST_RADIUS_RUN = "20261121T215512Z_whitebox_worst_radius"


def worst_radius_run() -> Path:
    path = RUNS / WORST_RADIUS_RUN
    if not (path / "results.json").exists():
        raise FileNotFoundError(f"pinned worst-radius run missing: {path}")
    built_from = yaml.safe_load((path / "params.yaml").read_text(encoding="utf-8"))["config"]
    sources_file = built_from.pop("sources_file")
    if built_from != load_yaml(REPO_ROOT / sources_file):
        raise ValueError(
            f"{sources_file} changed since the pinned worst-radius run {WORST_RADIUS_RUN} was "
            "built: run `python -m experiments.read_worst_radius` and pin the new run"
        )
    return path


# The split-protocol control behind the split table, the split figure and the tabular-model
# figure: pinned, so a later run of the same name cannot enter them unseen (red-team RT-n8).
GROUP_SPLIT_RUN = "20260816T023620Z_group_split_control_binary"


def group_split_run() -> Path:
    path = RUNS / GROUP_SPLIT_RUN
    if not (path / "results.json").exists():
        raise FileNotFoundError(f"pinned group-split run missing: {path}")
    return path


def vqc_table1(key: str) -> Path:
    path = RUNS / VQC_TABLE1[key]
    if not (path / "results.json").exists() and key != "data_efficiency":
        raise FileNotFoundError(f"pinned Table-I-VQC run missing: {path}")
    return path


def latest(pattern: str, key: str | None = None, cfg: dict | None = None) -> Path | None:
    """Newest run dir matching a glob that actually carries results.

    `key` restricts the search to runs whose results.json contains that top-level
    key. Multi-stage runners (run_whitebox_qsvm.py: p5a / p6a / p7b) are commonly
    executed one stage at a time, so "newest matching run" is not the same as
    "newest run carrying the stage I need".

    `cfg` additionally requires the run's resolved config to match every given field.
    The same stage is legitimately run at several configurations (P8b/P8c sweep the
    exhaustion stage over encodings and qubit counts), and those land in run dirs whose
    NAMES differ only by qubit count -- so without this the newest angle_ry run would
    silently supply the exhibit that documents the ZZ one.
    """
    import yaml

    hits = sorted(d for d in RUNS.glob(pattern) if (d / "results.json").exists())
    if key is not None:
        hits = [d for d in hits if key in json.loads((d / "results.json").read_text())]
    if cfg:
        keep = []
        for d in hits:
            p = d / "params.yaml"
            if not p.exists():
                continue
            got = (yaml.safe_load(p.read_text()) or {}).get("config", {})
            if all(got.get(k) == v for k, v in cfg.items()):
                keep.append(d)
        hits = keep
    return hits[-1] if hits else None


def merged_regime(pattern: str, label: str) -> Path | None:
    """Merge every run matching a regime glob into one results.json.

    Model zoos were executed as several concurrent jobs (fast models / SVM+MLP),
    so a regime's table is spread across run dirs. Later runs win per model, which
    makes re-runs supersede earlier ones. Provenance points back at the sources.
    """
    dirs = sorted(d for d in RUNS.glob(pattern) if (d / "results.json").exists())
    if not dirs:
        return None
    merged: dict = {}
    for d in dirs:
        merged.update(json.loads((d / "results.json").read_text()))
    out = RUNS / f"merged_{label}"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(merged, indent=2))
    (out / "sources.txt").write_text("\n".join(str(d) for d in dirs) + "\n")
    return out


def headline_stats(results_path: Path, reference: str | None = None) -> dict:
    """Paired Wilcoxon vs the best model, Holm-corrected across the family."""
    res = json.loads(results_path.read_text())
    per_seed = {
        k: np.array([s["macro_f1"] for s in v["per_seed"]])
        for k, v in res.items()
        if "per_seed" in v
    }
    if not per_seed:
        return {}
    ref = reference or max(per_seed, key=lambda k: per_seed[k].mean())
    raw, detail = {}, {}
    for name, arr in per_seed.items():
        if name == ref or len(arr) != len(per_seed[ref]):
            continue
        st = paired_wilcoxon(per_seed[ref], arr)
        raw[name] = st["p_value"]
        detail[name] = st
    corrected = holm_bonferroni(raw) if raw else {}
    return {
        "reference": ref,
        "reference_mean": float(per_seed[ref].mean()),
        "n_seeds": int(len(per_seed[ref])),
        "comparisons": {
            k: {
                "mean": float(per_seed[k].mean()),
                "mean_diff_vs_ref": detail[k]["mean_diff"],
                "cohens_d": detail[k]["cohens_d"],
                "rank_biserial": detail[k]["rank_biserial"],
                "p_raw": raw[k],
                "p_holm": corrected[k],
                "significant_holm_0.05": bool(corrected[k] < 0.05),
            }
            for k in raw
        },
    }


_QUANTUM_PREFIXES = ("qsvm_", "vqc")


def _is_quantum(model: str) -> bool:
    return model.startswith(_QUANTUM_PREFIXES)


def rq1_regime(n_dims: int) -> Path | None:
    """Merge classical + quantum results at ONE matched dimensionality.

    This is the paper's headline comparison and the only fair one: every model
    sees the same PCA-{n_dims} view, the same splits, and the same tuning budget.
    Quantum entries are namespaced (`qsvm_zz`, `vqc`) so a reader can never mistake
    which family a row belongs to. Reported against the full-feature regime, which
    is a *different* question (see the PCA-cost note in the decision log).
    """
    classical = merged_regime(f"*_classical_binary_pca{n_dims}*", f"accuracy_binary_pca{n_dims}")
    if classical is None:
        return None
    merged = json.loads((classical / "results.json").read_text())
    sources = [str(classical)]

    qk = latest(f"*quantum_qkernel_binary_q{n_dims}*")
    if qk:
        for enc, res in json.loads((qk / "results.json").read_text()).items():
            merged[f"qsvm_{enc}"] = res
        sources.append(str(qk))
    vqc = latest(f"*quantum_vqc_binary_q{n_dims}*")
    if vqc:
        merged.update(json.loads((vqc / "results.json").read_text()))
        sources.append(str(vqc))

    # trivial reference rows: without them a reader cannot tell a working model
    # from one that is at or below chance (stratified-random scores 0.5005 here)
    from qgridbench.data.preprocess import load_processed

    _, y, sp = load_processed("binary")
    merged.update(trivial_baselines(y[sp["train"]], y[sp["test"]]))

    if n_dims == 8:
        # The MLP row is the class-weighted fit (E-M4b arm W1, same splits and seeds). W1 did
        # not record timings, so the cost table keeps the unweighted fit's train time and
        # latency (same architecture; sample weights add no measurable cost).
        w1 = json.loads((weighted_mlp_run() / "results.json").read_text())["W1"]
        timing = {k: merged["mlp"]["aggregate"][k] for k in ("train_time_s", "infer_latency_s")}
        merged["mlp"] = {
            "best_params": w1["params"],
            "per_seed": w1["per_seed"],
            "aggregate": {**w1["aggregate"], **timing},
            "best_val_macro_f1": w1["best_val_macro_f1"],
        }
        sources.append(f"{weighted_mlp_run()} (mlp row: W1; timings from the unweighted fit)")

    out = RUNS / f"merged_rq1_binary_pca{n_dims}"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(merged, indent=2))
    (out / "sources.txt").write_text("\n".join(sources) + "\n")
    return out


def blackbox_summary(bb_dir: Path) -> dict:
    """Rank families by the perturbation a gradient-free attacker needs."""
    bb = json.loads((bb_dir / "results.json").read_text())["per_seed"][0]["blackbox"]
    ranked = sorted(bb.items(), key=lambda kv: -kv[1]["median_linf_distortion"])
    return {
        "n_families": len(bb),
        "queries_per_sample_mean": float(np.mean([r["queries_per_sample"] for r in bb.values()])),
        "ranking_by_median_linf": [
            {
                "model": k,
                "median_linf_sigma": v["median_linf_distortion"],
                "clean_f1": v["clean_f1"],
                "adv_f1": v["adv_f1"],
            }
            for k, v in ranked
        ],
    }


def main() -> None:
    with RunDir("report", config={"generator": "experiments.make_report"}) as run:
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        summary: dict = {}

        # ---- accuracy tables (each regime merged across its concurrent jobs) --
        regimes = [
            ("accuracy_binary_full", "*_classical_binary_full*"),
            ("accuracy_binary_pca8", "*_classical_binary_pca8*"),
            ("accuracy_binary_pca12", "*_classical_binary_pca12*"),
            ("accuracy_binary_pca16", "*_classical_binary_pca16*"),
            ("accuracy_triple_full", "*_classical_triple_full*"),
            ("accuracy_triple_pca8", "*_classical_triple_pca8*"),
        ]
        for label, pattern in regimes:
            path = merged_regime(pattern, label)
            if path is None:
                log.warning("skip %s (no run found)", label)
                continue
            accuracy_table(path, label)
            st = headline_stats(path / "results.json")
            summary[label] = st
            if st:
                log.info(
                    "%s: best=%s (%.4f), %d Holm-corrected comparisons",
                    label,
                    st["reference"],
                    st["reference_mean"],
                    len(st["comparisons"]),
                )

        # ---- RQ1 headline: classical vs quantum at MATCHED dimensionality -----
        for q in (8, 12, 16):
            path = rq1_regime(q)
            if path is None:
                log.warning("skip rq1_binary_pca%d (no classical regime)", q)
                continue
            accuracy_table(path, f"rq1_binary_pca{q}")
            cost_table(path, f"cost_binary_pca{q}")  # required: train time + latency
            st = headline_stats(path / "results.json")
            summary[f"rq1_binary_pca{q}"] = st
            if st:
                res = json.loads((path / "results.json").read_text())
                quantum = {
                    k: v["aggregate"]["macro_f1"]["mean"] for k, v in res.items() if _is_quantum(k)
                }
                log.info(
                    "rq1 pca%d: best=%s (%.4f) | quantum: %s",
                    q,
                    st["reference"],
                    st["reference_mean"],
                    ", ".join(f"{k}={v:.4f}" for k, v in sorted(quantum.items())),
                )

        # the matched contest at every dimensionality in one compact table (the ten-page
        # build prints it in place of the six-metric table); the class-weighted MLP fit
        # exists at eight dimensions only (rq1_regime)
        rq1_dirs = {
            q: RUNS / f"merged_rq1_binary_pca{q}"
            for q in (8, 12, 16)
            if f"rq1_binary_pca{q}" in summary
        }
        if rq1_dirs:
            matched_contest_table(rq1_dirs, "matched_contest_binary")

        # ---- quantum diagnostics --------------------------------------------
        diag_dirs: dict[int, Path] = {}
        for q in (8, 12, 16):
            d = latest(f"*quantum_qkernel_binary_q{q}*")
            if d:
                quantum_diagnostics_table(d, f"quantum_diagnostics_q{q}")
                diag_dirs[q] = d
        if diag_dirs:
            diagnostics_combined_table(diag_dirs, "diagnostics_all")

        # ---- robustness ------------------------------------------------------
        adv = vqc_table1("adversarial")
        if adv:
            summary["robustness"] = robustness_stats(adv)
            from qgridbench.eval.figures import (
                poisoning_sensor_figure,
                robustness_curves,
                transfer_heatmap,
            )

            robustness_curves(adv, "robustness_pgd", attack="pgd")
            robustness_curves(adv, "robustness_fgsm", attack="fgsm")
            transfer_heatmap(adv, "transfer_matrix")
            poisoning_sensor_figure(adv, "poisoning_sensor")
            log.info("robustness figures + stats from %s", adv.name)

        # ---- black-box control (families run as separate jobs, merged on disk) --
        bb = latest("*adversarial_binary_q8_blackbox")
        if bb:
            blackbox_table(bb, "blackbox_binary_q8")
            summary["blackbox"] = blackbox_summary(bb)
            log.info("black-box table from %s", bb.name)

        # ---- paper robustness exhibit: PGD + random control + HSJ in one table --
        rnd = vqc_table1("random")
        # the white-box column: every model's own attack at the worst radius inside the budget
        wbx = worst_radius_run()
        # HSJ restarts: the pinned pooled run covers all eight families at five or ten restarts,
        # so the HSJ column reports mean ± std instead of one unseeded draw.
        hsj_probes = [hsj_pooled_run()]
        if adv and rnd and bb:
            robustness_summary_table(
                adv,
                rnd,
                bb,
                "robustness_binary_q8",
                worst_radius_run_dir=wbx,
                hsj_restart_run_dirs=hsj_probes or None,
                weighted_mlp_run_dir=weighted_mlp_run(),
                logreg_whitebox_run_dir=logreg_whitebox_run(),
                # the score-based black-box attacks, for each model's lowest attacked score;
                # the VQC is read from the Square run on its Table I configuration
                score_attack_runs=[
                    {"attack": "S", "run": RUNS / SQUARE_RUN, "models": SQUARE_MODELS},
                    {"attack": "S", "run": RUNS / SQUARE_VQC_RUN, "models": ("vqc",)},
                    {"attack": "E", "run": RUNS / EVOLUTION_RUN, "models": EVOLUTION_MODELS},
                ],
            )
            log.info(
                "robustness summary table (PGD + white-box%s + random + HSJ%s)",
                f" at the worst radius from {wbx.name}",
                f" over restarts from {len(hsj_probes)} probes" if hsj_probes else " single-draw",
            )

        # ---- the HopSkipJump query-budget sweep, by direction and row by row --
        sweep = hsj_sweep_verdict_run()
        hsj_query_sweep_table(sweep, "hsj_query_sweep_binary_q8")
        log.info("query-sweep table from %s", sweep.name)

        # ---- P17e: interleaving measured directly, with its matched classical control --
        rc_q = latest("*probe_ray_crossings_binary_q8")
        rc_c = latest("*probe_ray_crossings_classical_binary_q8")
        if rc_q and rc_c:
            ray_crossings_table(rc_q, rc_c, "ray_crossings_binary_q8")
            log.info("ray-crossings table from %s + %s", rc_q.name, rc_c.name)

        # ---- C1 surrogate control: three arms, ordered by surrogate competence --
        sur = vqc_table1("surrogate")
        if sur:
            surrogate_control_table(sur, "surrogate_binary_q8", arm="depth6")
            log.info("surrogate control table from %s", sur.name)

        # ---- P12a split-protocol control: paired row-level vs whole-file exhibit --
        gsc = group_split_run()
        if gsc:
            ladder = latest("*split_protocol_ladder_binary")  # P25; column added when present
            group_split_table(gsc, "group_split_binary", ladder_dir=ladder)
            group_split_partitions_table(gsc, "group_split_partitions")
            log.info(
                "group-split control tables from %s (ladder: %s)",
                gsc.name,
                ladder.name if ladder else "none",
            )

        # ---- P7b mechanism exhibit: exhaustible vs non-exhaustible kernel ------
        # ZZ/q8 is the paper's reference configuration; pin it explicitly, because the
        # P8b/P8c generality cells write the same stage key into sibling run dirs.
        zz8 = {"encoding": "zz", "qubits": 8}
        exh = latest("*whitebox_qsvm_binary_q8", key="p7b_kernel_exhaustion", cfg=zz8)
        # The table has a VQC column, so it reads the pinned depth-6 rerun of P7b; its QSVM
        # and RBF cells equal the original run's at every radius (claim-drift review
        # 2026-11-19, finding 2).
        kernel_exhaustion_table(vqc_table1("p7b8"), "kernel_exhaustion_binary_q8")
        log.info("kernel-exhaustion table from %s", vqc_table1("p7b8").name)

        # ---- P8 mechanism exhibit: length scale, not family, selects the end state
        ls = latest("*whitebox_qsvm_binary_q8", key="p8a_lengthscale_control")
        ry8 = latest(
            "*whitebox_qsvm_binary_q8",
            key="p7b_kernel_exhaustion",
            cfg={"encoding": "angle_ry", "qubits": 8},
        )
        zz12 = latest(
            "*whitebox_qsvm_binary_q12",
            key="p7b_kernel_exhaustion",
            cfg={"encoding": "zz", "qubits": 12},
        )
        if ls and exh and ry8 and zz12:
            regime_table(
                ls,
                [
                    ("QSVM ZZ, 8 qubits", exh),
                    ("QSVM RY, 8 qubits", ry8),
                    ("QSVM ZZ, 12 qubits", zz12),
                    ("VQC, 8 qubits", vqc_table1("p7b8")),
                    ("VQC, 12 qubits", vqc_table1("p7b12")),
                ],
                "regime_binary_q8",
            )
            summary["regime_table_sources"] = [d.name for d in (ls, exh, ry8, zz12)]
            log.info("regime table from %s + 3 exhaustion cells", ls.name)

        # ---- P9a: the SYMMETRIC control -- the quantum knob swept like the classical one
        qbw = latest("*whitebox_qsvm_binary_q8", key="p9a_quantum_bandwidth_regime", cfg=zz8)
        if qbw:
            quantum_bandwidth_table(qbw, "quantum_bandwidth_binary_q8")
            summary["quantum_bandwidth_source"] = qbw.name
            log.info("quantum-bandwidth table from %s", qbw.name)

        # ---- P11c/P11d: the end state is objective x regularization, not the encoding
        obj = latest("*whitebox_qsvm_binary_q8", key="p11c_objective_and_regularization", cfg=zz8)
        if obj:
            objective_regime_table(obj, "objective_regime_binary_q8")
            summary["objective_regime_source"] = obj.name
            log.info("objective/regularization table from %s", obj.name)

        # ---- three-class confusion (study protocol §8.8 appendix item) ---------------
        conf = sorted(
            d
            for d in RUNS.glob("*ablations_triple*")
            if (d / "confusion_triple_full.json").exists()
            and (d / "confusion_triple_pca8.json").exists()
        )
        if conf:
            three_class_confusion_table(conf[-1], "confusion_three_class")
            summary["confusion_three_class_source"] = conf[-1].name
            log.info("three-class confusion table from %s", conf[-1].name)

        # ---- 37-class confusion (R4-20; arXiv appendix only) --------------------------
        mc = sorted(
            d
            for d in RUNS.glob("*Z_multiclass_confusion_multiclass")
            if all((d / f).exists() for f in MULTICLASS_CONFUSION_FILES.values())
        )
        if mc:
            from qgridbench.eval.figures import (
                multiclass_confusion_figure,
                multiclass_summary_figure,
            )

            multiclass_confusion_counts(mc[-1], "confusion_multiclass")
            for regime in MULTICLASS_CONFUSION_FILES:
                multiclass_confusion_figure(mc[-1], regime, f"confusion_multiclass_{regime}")
            # the appendix exhibit: scores per model and regime over three of the matrices
            multiclass_summary_figure(mc[-1], "multiclass_summary")
            summary["confusion_multiclass_source"] = mc[-1].name
            log.info("multiclass confusion counts and figures from %s", mc[-1].name)

        # ---- data efficiency --------------------------------------------------
        de = vqc_table1("data_efficiency")
        de_file = "data_efficiency_vqc_table1.json"
        if not (de / de_file).exists():
            raise FileNotFoundError(f"pinned data-efficiency file missing: {de / de_file}")
        from qgridbench.eval.figures import data_efficiency_curve

        data_efficiency_curve(de, "data_efficiency", filename=de_file)
        summary["data_efficiency_source"] = f"{de.name}/{de_file}"

        # ---- circuit diagrams (paper methodology figure, drawn from the SSOT) --
        from qgridbench.eval.figures import bandwidth_diagnostics_figure, circuit_diagrams

        circuit_diagrams()

        # ---- bandwidth sweep diagnostics (degeneracy exhibit) ------------------
        bw = sorted(
            d for d in RUNS.glob("*ablations_binary*") if (d / "bandwidth_curve.json").exists()
        )
        if bw:
            bandwidth_diagnostics_figure(bw[-1] / "bandwidth_curve.json", "bandwidth_diagnostics")
            summary["bandwidth_figure_source"] = bw[-1].name

        # ---- bandwidth fragility (no-safe-operating-point exhibit) -------------
        frag = sorted(
            d for d in RUNS.glob("*ablations_binary*") if (d / "bandwidth_fragility.json").exists()
        )
        if frag:
            from qgridbench.eval.figures import bandwidth_fragility_figure

            bandwidth_fragility_figure(frag[-1] / "bandwidth_fragility.json", "bandwidth_fragility")
            summary["bandwidth_fragility_source"] = frag[-1].name

        # ---- trainability: barren-plateau scan + depth sweep -------------------
        bp = sorted(
            d for d in RUNS.glob("*ablations_binary*") if (d / "barren_plateau_scan.json").exists()
        )
        ds = sorted(
            d for d in RUNS.glob("*ablations_binary*") if (d / "vqc_depth_sweep.json").exists()
        )
        if bp and ds:
            from qgridbench.eval.figures import barren_plateau_figure

            barren_plateau_figure(
                bp[-1] / "barren_plateau_scan.json",
                ds[-1] / "vqc_depth_sweep.json",
                "barren_plateau",
            )
            summary["barren_plateau_sources"] = [bp[-1].name, ds[-1].name]

        # ---- positive control: relabel cells, binary + triple ------------------
        relabel = [
            dirs[-1]
            for pattern in ("*ablations_binary*", "*ablations_triple*")
            if (
                dirs := sorted(
                    d for d in RUNS.glob(pattern) if (d / "relabel_control.json").exists()
                )
            )
        ]
        if relabel:
            relabel_control_table(relabel, "relabel_control")
            summary["relabel_table_sources"] = [d.name for d in relabel]

        # ---- draft v3 exhibits: Figs. A-D --------------------------------------
        # A and C come from the curated cell and panel files (experiments/
        # replication_matrix_cells.csv, instance_panels.csv) through their builder, which reads
        # every value from its artifact; B and D read the newest finished runs.
        from experiments.build_replication_matrix import main as build_v3_tables
        from qgridbench.eval.figures import (
            BRIDGE_DIMS,
            accuracy_bridge_figure,
            checklist_figure,
            console_figure,
            controls_figure,
            fdia_budget_figure,
            gas_pressure_figure,
            radius_curve_figure,
            replication_matrix_figure,
            split_leak_figure,
            tabular_split_figure,
            width_sweep_figure,
            write_design_colors,
        )

        build_v3_tables()
        # the box diagrams' measured numbers read instance_panels.csv, so they follow it
        from experiments.build_paper_numbers import main as build_paper_numbers

        build_paper_numbers()
        # the arXiv appendix table of further controls, each number read from its run artifact
        from experiments.build_further_controls import main as build_further_controls

        build_further_controls()
        # the meaning colors for the TikZ diagrams and table rows (one palette with the figures)
        write_design_colors()
        tables_dir = REPO_ROOT / "paper" / "tables"
        replication_matrix_figure(tables_dir / "replication_matrix.csv", "replication_matrix")
        checklist_figure(tables_dir / "instance_panels.csv", "checklist")
        v3 = {
            "group": group_split_run(),
            "decay": latest("*Z_leak_decay_curve_binary"),
            "fdia_split": latest("*Z_fdia_split_gap"),
            "sq14": latest("*Z_square_blackbox_fdia14_binary_q8"),
            "sq118": latest("*Z_square_blackbox_fdia118_binary_q8"),
            "h3_14": latest("*Z_fdia_consistent_attack_fdia14_binary"),
            "h3_118": latest("*Z_fdia_consistent_attack_fdia118_binary"),
            "console": latest("*Z_probe_operating_point_full_binary"),
            "catch": latest("*Z_evasion_square_per_class_binary_q8"),
            "width": latest("*whitebox_qsvm_binary_q8", key="p8a_lengthscale_control"),
        }
        missing = [k for k, v in v3.items() if v is None]
        if missing:
            raise FileNotFoundError(f"draft-v3 figure sources missing: {missing}")
        split_leak_figure(
            v3["group"], v3["decay"], v3["fdia_split"], "split_leak", level_per_class_run()
        )
        fdia_budget_figure(
            {"(a) IEEE 14-bus": v3["sq14"], "(b) IEEE 118-bus": v3["sq118"]},
            {"IEEE 14-bus": v3["h3_14"], "IEEE 118-bus": v3["h3_118"]},
            "fdia_budget",
        )
        console_figure(v3["console"], "console")
        # the robustness table as a picture: its own CSV (written above) and the per-class
        # Square run; the TSG build prints it in place of the table
        controls_figure(tables_dir / "robustness_binary_q8.csv", v3["catch"], "controls")
        # the ten-page build prints panels (a) to (c); panel (d) stays in the complete build
        controls_figure(
            tables_dir / "robustness_binary_q8.csv", v3["catch"], "controls_abc", with_catch=False
        )
        width_sweep_figure(v3["width"], "width_sweep")
        # the kernel machines' white-box curves over the tested radii (checklist item 5)
        radius_curve_figure(
            worst_radius_run(), tables_dir / "robustness_binary_q8.csv", "radius_curve"
        )
        # newer tabular models under both splits, and the gas pressure feature before and after:
        # pinned runs, the ones the decision log names as the results (several attempts exist)
        tabular_split_figure(
            {"lgbm": v3["group"], **{k: RUNS / name for k, name in TABULAR_RUNS.items()}},
            "tabular_split",
        )
        gas_pressure_figure(
            RUNS / GAS_ACCURACY_RUNS["shared"], RUNS / GAS_ACCURACY_RUNS["slog"], "gas_pressure"
        )
        # where the accuracy goes between the full-feature setting and the matched contest
        accuracy_bridge_figure(
            tables_dir / f"rq1_binary_pca{BRIDGE_DIMS}.csv", RUNS / CAP_COST_RUN, "accuracy_bridge"
        )
        summary["draft_v3_figure_sources"] = {
            **{k: v.name for k, v in v3.items()},
            "accuracy_bridge": CAP_COST_RUN,
        }

        # ---- refresh the model card's generated blocks from those artifacts ---
        from qgridbench.eval.model_card import refresh

        log.info("model card blocks refreshed: %s", ", ".join(refresh(robustness_run=adv)))

        run.write_json("report_summary.json", summary)
        print(
            json.dumps(
                {k: (v if not isinstance(v, dict) else "…") for k, v in summary.items()}, indent=2
            )
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
