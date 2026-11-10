"""Generated tables and figures follow the IEEE PES author-kit rules (Part 4a).

Footnote marks are letters, tables print words rather than code identifiers, circuit
labels print at 8 pt or more at column width, and axis labels are words with units in
parentheses. Offline: tiny fixtures, no run directories.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Arc, FancyArrow, FancyBboxPatch
from matplotlib.text import Text

from qgridbench.eval import design_tokens, figures, tables

SRC = Path(__file__).resolve().parents[1] / "src" / "qgridbench" / "eval"
CODE_NAMES = (
    "logreg",
    "rbf_svm",
    "xgb",
    "lgbm",
    "mlp",
    "qsvm_angle_ry",
    "qsvm_zz",
    "baseline_stratified_random",
    "macro_f1",
    "balanced_accuracy",
    "roc_auc",
)


@pytest.fixture
def table_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "TABLE_DIR", tmp_path / "tables")
    return tmp_path / "tables"


def _results(names_to_f1: dict[str, float], extra: dict | None = None) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    for name, level in names_to_f1.items():
        per_seed = [
            {m: float(level + 0.01 * rng.standard_normal()) for m in tables.HEADLINE_METRICS}
            for _ in range(10)
        ]
        out[name] = {
            "per_seed": per_seed,
            "aggregate": {
                m: {"mean": float(np.mean([s[m] for s in per_seed])), "std": 0.01}
                for m in tables.HEADLINE_METRICS
            },
            **(extra or {}),
        }
    return out


def test_accuracy_marks_are_letters_with_a_first_and_b_second(tmp_path, table_dir):
    run = tmp_path / "run"
    run.mkdir()
    # one best model and two clearly below it (significant after Holm correction)
    results = _results({"xgb": 0.90, "logreg": 0.50, "rf": 0.50})
    (run / "results.json").write_text(json.dumps(results))
    tex = tables.accuracy_table(run, "probe").read_text(encoding="utf-8")
    rows = {line.split("&")[0].strip(): line for line in tex.splitlines() if "&" in line}

    assert rows["XGBoost$^{a}$"]
    assert rows["Logistic regression$^{b}$"]
    assert rows["Random forest$^{b}$"]
    assert not re.search(r"\\(dd?agger|ast)|\$\^\{\*\}\$", tex)
    assert "% a = best model" in tex and "% b = significantly below" in tex


def test_cost_table_marks_simulated_models_with_a(tmp_path, table_dir):
    run = tmp_path / "run"
    run.mkdir()
    results = {
        name: {"aggregate": {"train_time_s": {"mean": 1.0}, "infer_latency_s": {"mean": 1e-5}}}
        for name in ("xgb", "vqc", "qsvm_zz")
    }
    (run / "results.json").write_text(json.dumps(results))
    tex = tables.cost_table(run, "probe").read_text(encoding="utf-8")

    assert "VQC$^{a}$" in tex and "QSVM (ZZ)$^{a}$" in tex
    assert "XGBoost &" in tex
    assert "dagger" not in tex
    assert "% a = simulated quantum model" in tex


def test_accuracy_table_prints_display_names_not_code_names(tmp_path, table_dir):
    run = tmp_path / "run"
    run.mkdir()
    names = list(tables.MODEL_DISPLAY)[:11]
    (run / "results.json").write_text(json.dumps(_results(dict.fromkeys(names, 0.5))))
    tex = tables.accuracy_table(run, "probe").read_text(encoding="utf-8")
    header = next(line for line in tex.splitlines() if line.startswith("Model &"))

    assert header == ("Model & Macro-F1 & Balanced accuracy & AUPRC & ROC-AUC & Brier & ECE \\\\")
    for shown in (
        "Logistic regression",
        "RBF-SVM",
        "Random forest",
        "XGBoost",
        "LightGBM",
        "MLP",
        "QSVM (RY-angle)",
        "QSVM (ZZ)",
        "VQC",
        "Stratified random",
    ):
        assert re.search(rf"^{re.escape(shown)}(\$\^\{{[ab]\}}\$)? &", tex, re.M), shown
    for code in CODE_NAMES:
        assert code not in tex and code.replace("_", r"\_") not in tex, code
    # the CSV sidecar keeps the identifiers
    csv_text = (table_dir / "probe.csv").read_text(encoding="utf-8")
    assert "qsvm_angle_ry" in csv_text and "macro_f1" in csv_text


def test_display_mappings_fail_loud_on_unknown_identifiers():
    with pytest.raises(KeyError):
        tables.display_model("not_a_model")
    with pytest.raises(KeyError):
        tables.display_metric("not_a_metric")
    assert tables.display_cell("zz_q12_binary") == "ZZ, 12 qubits, binary"
    assert tables.display_cell("angle_ry_q8_triple") == "RY-angle, 8 qubits, three-class"


def test_no_symbol_footnote_marks_remain_in_the_generators():
    source = (SRC / "tables.py").read_text(encoding="utf-8")
    assert "dagger" not in source
    assert "$^{*}$" not in source


def _capture_closed_figures(monkeypatch):
    closed = []
    real_close = plt.close

    def close(fig=None):
        closed.append(fig)
        real_close(fig)

    monkeypatch.setattr(plt, "close", close)
    return closed


def _assert_labels_inside_boxes(fig):
    """Every gate label and the measurement glyph sit inside their box with 1 pt to spare."""
    canvas = FigureCanvasAgg(fig)  # the saved figure is closed, so draw it on a fresh canvas
    canvas.draw()
    renderer = canvas.get_renderer()
    ax = fig.axes[0]
    boxes = [p.get_window_extent(renderer) for p in ax.patches if isinstance(p, FancyBboxPatch)]
    assert boxes

    gap = fig.dpi / 72  # one point of clearance between a label and the box outline

    def inside(extent):
        return any(
            b.x0 + gap <= extent.x0
            and extent.x1 <= b.x1 - gap
            and b.y0 + gap <= extent.y0
            and extent.y1 <= b.y1 - gap
            for b in boxes
        )

    gate_labels = [t for t in ax.texts if t.get_text() and not t.get_text().isdigit()]
    assert gate_labels
    for t in gate_labels:
        assert inside(t.get_window_extent(renderer)), f"label {t.get_text()!r} overruns its box"
    glyphs = [p for p in ax.patches if isinstance(p, Arc | FancyArrow)]
    assert glyphs
    for g in glyphs:
        assert inside(g.get_window_extent(renderer)), "measurement glyph overflows its box"


def test_circuit_figures_are_one_column_wide_with_8pt_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path)
    sizes = []
    real_save = figures._save_circuit

    def save(fig, out):
        real_save.__call__(fig, out)  # rescales and saves; the figure is closed afterwards
        sizes.append(
            [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
        )
        _assert_labels_inside_boxes(fig)

    monkeypatch.setattr(figures, "_save_circuit", save)
    outs = figures.circuit_diagrams()

    assert [o.name for o in outs] == ["circuit_zz.pdf", "circuit_vqc.pdf"]
    assert len(sizes) == 2
    for out, label_sizes in zip(outs, sizes, strict=True):
        box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)", out.read_bytes())
        assert abs(float(box.group(1)) - figures.COLUMN_WIDTH_PT) < 1.0, out.name
        assert label_sizes and min(label_sizes) >= 8.0, (out.name, min(label_sizes))
        assert b"/Type3" not in out.read_bytes(), out.name


def test_axis_labels_are_words_with_units_in_parentheses(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path)
    closed = _capture_closed_figures(monkeypatch)

    de = tmp_path / "de"
    de.mkdir()
    series = {s: {"mean": 0.5, "std": 0.01} for s in ("250", "500")}
    (de / "data_efficiency.json").write_text(
        json.dumps({"xgb_full": series, "qkernel_angle_ry": series, "vqc": series})
    )
    figures.data_efficiency_curve(de, "de")

    frag = tmp_path / "frag.json"
    frag.write_text(
        json.dumps(
            [
                {
                    "bandwidth": bw,
                    "clean_macro_f1": 0.5,
                    "flip_rate": {"0.5": {"draws": [0.1, 0.2, 0.3]}},
                }
                for bw in (0.1, 1.0, 10.0)
            ]
        )
    )
    figures.bandwidth_fragility_figure(frag, "frag")

    scan, depth = tmp_path / "scan.json", tmp_path / "depth.json"
    scan.write_text(
        json.dumps({"by_qubits": {str(q): {"var_single_param": 2.0**-q} for q in (4, 8, 12)}})
    )
    depth.write_text(
        json.dumps({f"depth_{d}": {"grad_norm_var_mean": 0.01 * d} for d in (2, 4, 6)})
    )
    figures.barren_plateau_figure(scan, depth, "bp")

    labels = []
    for fig in closed:
        for ax in fig.axes:
            labels += [ax.get_xlabel(), ax.get_ylabel()]
    labels = [lab for lab in labels if lab]
    assert len(closed) == 3 and labels
    joined = " | ".join(labels)
    assert "Training-set size (samples)" in joined
    assert "Encoding bandwidth (scale factor)" in joined
    assert "Number of qubits" in joined and "Circuit depth (layers, 8 qubits)" in joined
    for lab in labels:
        assert lab[0].isupper(), lab
        assert not re.search(r"\$|\\|@", lab), f"symbol in axis label: {lab!r}"
    legend_texts = [
        t.get_text()
        for fig in closed
        for legend in [*fig.legends, *(ax.get_legend() for ax in fig.axes)]
        if legend
        for t in legend.get_texts()
    ]
    assert {"XGBoost (all features)", "QSVM (RY-angle)", "VQC"} <= set(legend_texts)


FAMILIES = ["logreg", "rbf_svm", "rf", "xgb", "lgbm", "mlp", "qsvm", "vqc"]


def _adversarial_run(tmp_path: Path) -> Path:
    eps = ["0.01", "0.1", "0.5"]
    per_seed = []
    for seed in range(2):
        per_seed.append(
            {
                "evasion": {
                    "clean": {f: 0.6 for f in FAMILIES},
                    "curves": {
                        f: {
                            "clean": 0.6,
                            "fgsm": {e: 0.5 - 0.01 * seed for e in eps},
                            "pgd": {e: 0.4 - 0.01 * seed for e in eps},
                        }
                        for f in FAMILIES
                    },
                    "transfer_matrix": {
                        s: {t: 0.4 + 0.01 * seed for t in FAMILIES}
                        for s in ("mlp", "vqc", "mlp_surrogate")
                    },
                },
                "sensor_noise": {f: {"30": 0.5, "20": 0.45, "10": 0.4} for f in FAMILIES},
            }
        )
    run = tmp_path / "adv"
    run.mkdir()
    poisoning = {r: dict.fromkeys(FAMILIES, 0.5) for r in ("0.05", "0.1", "0.2")}
    (run / "results.json").write_text(json.dumps({"per_seed": per_seed, "poisoning": poisoning}))
    return run


def _bandwidth_rows() -> list[dict]:
    return [
        {
            "cell": cell,
            "bandwidth": bw,
            "median_offdiag": 0.5,
            "top_eig_share": 0.4,
            "frac_below_0.01": 0.1,
            "val_macro_f1": 0.55,
            "tuned_bandwidth": 1.0,
            "g_vs_tuned_rbf": 10.0 * bw,
            "kta_grid_labels": 0.01 * bw,
        }
        for cell in ("angle_ry_q8_binary", "zz_q8_binary", "zz_q12_binary", "zz_q16_binary")
        for bw in (0.1, 1.0, 3.0)
    ]


def test_every_figure_prints_at_8pt_or_more_at_its_placed_width(tmp_path, monkeypatch):
    """Each figure is drawn at the width it is placed at, so its labels print as drawn.

    Axis labels, ticks, legends, panel titles and heat-map cell numbers are all 8 pt or more,
    and the native width is one column (252 pt) or the full text width (516 pt).
    """
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "out")
    closed = _capture_closed_figures(monkeypatch)
    adv = _adversarial_run(tmp_path)

    de = tmp_path / "de"
    de.mkdir()
    series = {s: {"mean": 0.5, "std": 0.01} for s in ("250", "500", "1000", "2000")}
    (de / "data_efficiency.json").write_text(
        json.dumps({"xgb_full": series, "qkernel_angle_ry": series, "vqc": series})
    )
    bw = tmp_path / "bw.json"
    bw.write_text(json.dumps(_bandwidth_rows()))
    frag = tmp_path / "frag.json"
    frag.write_text(
        json.dumps(
            [
                {
                    "bandwidth": b,
                    "clean_macro_f1": 0.5,
                    "flip_rate": {"0.5": {"draws": [0.1, 0.2, 0.3]}},
                }
                for b in (0.1, 1.0, 10.0)
            ]
        )
    )
    scan, depth = tmp_path / "scan.json", tmp_path / "depth.json"
    scan.write_text(
        json.dumps({"by_qubits": {str(q): {"var_single_param": 2.0**-q} for q in (8, 12, 16)}})
    )
    depth.write_text(json.dumps({f"depth_{d}": {"grad_norm_var_mean": 0.002} for d in (2, 4, 6)}))

    figures.robustness_curves(adv, "robustness_pgd", attack="pgd")
    figures.robustness_curves(adv, "robustness_fgsm", attack="fgsm")
    figures.transfer_heatmap(adv, "transfer_matrix")
    figures.poisoning_sensor_figure(adv, "poisoning_sensor")
    figures.data_efficiency_curve(de, "data_efficiency")
    figures.bandwidth_diagnostics_figure(bw, "bandwidth_diagnostics")
    figures.bandwidth_fragility_figure(frag, "bandwidth_fragility")
    figures.barren_plateau_figure(scan, depth, "barren_plateau")

    placed = {  # in the order drawn above
        "robustness_pgd": design_tokens.COLUMN_WIDTH_PT,
        "robustness_fgsm": design_tokens.COLUMN_WIDTH_PT,
        "transfer_matrix": design_tokens.TEXT_WIDTH_PT,
        "poisoning_sensor": design_tokens.TEXT_WIDTH_PT,
        "data_efficiency": design_tokens.COLUMN_WIDTH_PT,
        "bandwidth_diagnostics": design_tokens.TEXT_WIDTH_PT,
        "bandwidth_fragility": design_tokens.COLUMN_WIDTH_PT,
        "barren_plateau": design_tokens.TEXT_WIDTH_PT,
    }
    assert len(closed) == len(placed)
    for (name, width), fig in zip(placed.items(), closed, strict=True):
        pdf = tmp_path / "out" / f"{name}.pdf"
        box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)", pdf.read_bytes())
        assert abs(float(box.group(1)) - width) < 0.5, name
        assert abs(fig.get_figwidth() * 72 - width) < 0.5, name
        fig.canvas.draw()  # tick labels and offset texts exist only after a draw
        texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
        sizes = [t.get_fontsize() for t in texts]
        assert sizes and min(sizes) >= design_tokens.LABEL_PT, (name, min(sizes))
        # a mathtext superscript (10^-3, 2^-n) prints at 70 percent of its text object's size
        mathtext = [t.get_text() for t in texts if "$" in t.get_text() or "^" in t.get_text()]
        assert not mathtext, (name, mathtext)
        assert b"/Type3" not in pdf.read_bytes(), name


def _write_run(tmp_path: Path, name: str, payload: dict) -> Path:
    run = tmp_path / name
    run.mkdir()
    (run / "results.json").write_text(json.dumps(payload))
    return run


def _body_cells(tex: str) -> str:
    """Table body without comment lines and without `$...$` math, where a hyphen is a hyphen."""
    body = "\n".join(line for line in tex.splitlines() if not line.startswith("%"))
    return re.sub(r"\$[^$]*\$", "", body)


def _assert_ieee_numbers(name: str, tex: str) -> None:
    body = _body_cells(tex)
    assert not re.search(r"(?<![0-9A-Za-z}])-\d", body), f"{name}: hyphen before a digit"
    assert not re.search(r"\d,\d{3}(?!\d)", tex), f"{name}: comma inside a number"
    assert not re.search(r"(?<![\d.])\d\\,\d{3}(?!\d)", tex), f"{name}: thin space in 4 digits"
    for word in ("neighbour", "w/", "feat.", "2k ", "dmg", "atk", "fwd"):
        assert word not in tex, f"{name}: {word!r}"


def test_generated_tables_use_true_minus_and_ieee_number_format(tmp_path, table_dir):
    neg = {"kernel_mass_ratio": 0.5, "proba_std": 0.1, "f1_minus_floor": -0.123}
    pos = {**neg, "f1_minus_floor": 0.045}
    clean = {"macro_f1": 0.58}

    def curve(last):
        return {"0.0": {**clean, **pos}, "2.0": last}

    per_model = {m: curve(neg) for m in ("qsvm", "vqc", "rbf_svm")}
    exh = _write_run(tmp_path, "exh", {"p7b_kernel_exhaustion": {"per_model": per_model}})
    out = {}
    out["kernel_exhaustion"] = tables.kernel_exhaustion_table(exh, "kernel_exhaustion")
    lengthscale = {
        "constant_predictor_floor": 0.5,
        "per_width": {"0.05": {"gamma": 0.2, "curve": curve(neg)}},
    }
    out["regime"] = tables.regime_table(
        _write_run(tmp_path, "ls", {"p8a_lengthscale_control": lengthscale}),
        [("QSVM ZZ, 8 qubits", exh), ("VQC, 8 qubits", exh)],
        "regime",
    )
    bandwidth = {
        "constant_predictor_floor": 0.5,
        "tuned_bandwidth": 0.8,
        "per_bandwidth": {"0.1": {"curve": curve(neg)}, "0.8": {"curve": curve(pos)}},
    }
    out["bandwidth"] = tables.quantum_bandwidth_table(
        _write_run(tmp_path, "bw", {"p9a_quantum_bandwidth_regime": bandwidth}), "bandwidth"
    )
    arm = {
        "surrogate_clean": 0.46,
        "surrogate_clean_minus_floor": -0.031,
        "vqc_damage": 0.3,
        "classical_damage": 0.2,
        "classical_peer_damage": 0.2,
        "forward_over_reverse": 10.2,
    }
    summary = {
        "mlp": {**arm, "surrogate_positive_rate": 0.1},
        "mlp_balanced": {**arm, "surrogate_positive_rate": 0.2},
        "rbf_svm": arm,
        "floor": 0.5,
        "eps": 0.5,
        "positive_rate_eval": 0.5,
    }
    out["surrogate"] = tables.surrogate_control_table(
        _write_run(tmp_path, "sur", {"summary": summary, "per_seed": [{}]}), "surrogate"
    )
    row = {
        "level_full_lgbm": 0.9,
        "matched_classical": 0.6,
        "matched_quantum": 0.65,
        "matched_contrast": -0.05,
        "floor_macro_f1": 0.5,
        "nn_frac_within_0.1": 0.4,
    }
    rows = [
        {**row, "split": "row (shipped)", "test_groups": None},
        {**row, "split": "group", "partition": 1, "test_groups": [3, 7, 9]},
    ]
    gs = _write_run(tmp_path, "gs", {"rows": rows, "verdict": "x"})
    out["group_split"] = tables.group_split_table(gs, "group_split")
    out["partitions"] = tables.group_split_partitions_table(gs, "partitions")
    cost = {
        name: {"aggregate": {"train_time_s": {"mean": 1.0}, "infer_latency_s": {"mean": lat}}}
        for name, lat in (("xgb", 1e-6), ("vqc", 2.3e-3), ("qsvm_zz", 1.13e-1))
    }
    out["cost"] = tables.cost_table(_write_run(tmp_path, "cost", cost), "cost")

    for name, path in out.items():
        _assert_ieee_numbers(name, path.read_text(encoding="utf-8"))
    cost_tex = out["cost"].read_text(encoding="utf-8")
    assert r"2300$\times$" in cost_tex and r"113\,000$\times$" in cost_tex
    assert "$-$0.123" in out["kernel_exhaustion"].read_text(encoding="utf-8")
    gs_tex = out["group_split"].read_text(encoding="utf-8")
    assert "Full 128 features" in gs_tex and "2000-row cap" in gs_tex
    assert "train neighbor" in gs_tex and "Test rows with" in gs_tex


def test_word_subscripts_are_roman(tmp_path, table_dir):
    d = {"kta_quantum": 0.1, "kta_classical_rbf": 0.2, "geometric_difference": 3.0}
    f1 = {"macro_f1": {"mean": 0.6, "std": 0.01}}
    run = _write_run(tmp_path, "diag", {"zz": {"diagnostics": d, "aggregate": f1}})
    for tex_path in (
        tables.diagnostics_combined_table({8: run}, "diag"),
        tables.quantum_diagnostics_table(run, "diag_q"),
    ):
        tex = tex_path.read_text(encoding="utf-8")
        assert r"KTA$_{\mathrm{RBF}}$" in tex
        assert "_{RBF}" not in tex.replace(r"\mathrm{RBF}", "")


def test_surrogate_control_glob_skips_the_group_surrogate_runs(tmp_path, monkeypatch):
    """`make_report` died on the newer `group_surrogate_control` runs, which carry no `summary`.

    Both run families end in `surrogate_control_binary_q8`; the group runs are newer, so an
    unanchored glob returned one of them as the "latest" surrogate-control run.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from experiments import make_report

    runs = tmp_path / "runs"
    for name in (
        "20260815T195529Z_surrogate_control_binary_q8",
        "20260822T092627Z_group_surrogate_control_binary_q8",
    ):
        (runs / name).mkdir(parents=True)
        (runs / name / "results.json").write_text("{}")
    monkeypatch.setattr(make_report, "RUNS", runs)

    hit = make_report.latest(make_report.SURROGATE_CONTROL_GLOB)
    assert hit is not None and hit.name == "20260815T195529Z_surrogate_control_binary_q8"
