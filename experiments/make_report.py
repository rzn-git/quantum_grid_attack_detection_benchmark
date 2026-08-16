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

from qgridbench.eval.baselines import trivial_baselines
from qgridbench.eval.stats import holm_bonferroni, paired_wilcoxon, robustness_stats
from qgridbench.eval.tables import (
    TABLE_DIR,
    accuracy_table,
    blackbox_table,
    cost_table,
    diagnostics_combined_table,
    group_split_partitions_table,
    group_split_table,
    kernel_exhaustion_table,
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
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger

log = get_logger(__name__)
RUNS = REPO_ROOT / "results" / "runs"


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
        adv = latest("*adversarial_binary_q8")
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
        rnd = latest("*adversarial_binary_q8_random")
        wbx = latest("*whitebox_qsvm_binary_q8", key="p5a_whitebox_qsvm")
        if adv and rnd and bb:
            robustness_summary_table(adv, rnd, bb, "robustness_binary_q8", whitebox_run_dir=wbx)
            log.info(
                "robustness summary table (PGD + white-box%s + random + HSJ)",
                f" from {wbx.name}" if wbx else " unavailable",
            )

        # ---- P17e: interleaving measured directly, with its matched classical control --
        rc_q = latest("*probe_ray_crossings_binary_q8")
        rc_c = latest("*probe_ray_crossings_classical_binary_q8")
        if rc_q and rc_c:
            ray_crossings_table(rc_q, rc_c, "ray_crossings_binary_q8")
            log.info("ray-crossings table from %s + %s", rc_q.name, rc_c.name)

        # ---- C1 surrogate control: three arms, ordered by surrogate competence --
        sur = latest("*surrogate_control_binary_q8")
        if sur:
            surrogate_control_table(sur, "surrogate_binary_q8")
            log.info("surrogate control table from %s", sur.name)

        # ---- P12a split-protocol control: paired row-level vs whole-file exhibit --
        gsc = latest("*group_split_control_binary")
        if gsc:
            group_split_table(gsc, "group_split_binary")
            group_split_partitions_table(gsc, "group_split_partitions")
            log.info("group-split control tables from %s", gsc.name)

        # ---- P7b mechanism exhibit: exhaustible vs non-exhaustible kernel ------
        # ZZ/q8 is the paper's reference configuration; pin it explicitly, because the
        # P8b/P8c generality cells write the same stage key into sibling run dirs.
        zz8 = {"encoding": "zz", "qubits": 8}
        exh = latest("*whitebox_qsvm_binary_q8", key="p7b_kernel_exhaustion", cfg=zz8)
        if exh:
            kernel_exhaustion_table(exh, "kernel_exhaustion_binary_q8")
            log.info("kernel-exhaustion table from %s", exh.name)

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
                    ("VQC, 8 qubits", exh),
                    ("VQC, 12 qubits", zz12),
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

        # ---- three-class confusion (CLAUDE.md 8.8 appendix item) ---------------
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

        # ---- data efficiency --------------------------------------------------
        de = sorted(
            d for d in RUNS.glob("*ablations_binary*") if (d / "data_efficiency.json").exists()
        )
        if de:
            from qgridbench.eval.figures import data_efficiency_curve

            data_efficiency_curve(de[-1], "data_efficiency")
            summary["data_efficiency_source"] = de[-1].name

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

        # ---- refresh the model card's generated blocks from those artifacts ---
        from qgridbench.eval.model_card import refresh

        log.info("model card blocks refreshed: %s", ", ".join(refresh()))

        run.write_json("report_summary.json", summary)
        print(
            json.dumps(
                {k: (v if not isinstance(v, dict) else "…") for k, v in summary.items()}, indent=2
            )
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
