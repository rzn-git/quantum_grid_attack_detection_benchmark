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
        # a quantum or baseline row opens with its meaning tint (design_tokens, ruling 16:18Z)
        tint = r"(\\rowcolor\{qg[A-Za-z]+Tint\})?"
        assert re.search(rf"^{tint}{re.escape(shown)}(\$\^\{{[ab]\}}\$)? &", tex, re.M), shown
    assert "\\rowcolor{qgQuantumTint}VQC" in tex  # quantum rows tinted
    assert re.search(r"^XGBoost", tex, re.M)  # classical rows are not
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

    scales = []

    def save(fig, out, scale=None):
        # rescales and saves; the figure is closed afterwards
        used = real_save.__call__(fig, out, scale)
        scales.append(used)
        sizes.append(
            [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
        )
        _assert_labels_inside_boxes(fig)
        return used

    monkeypatch.setattr(figures, "_save_circuit", save)
    outs = figures.circuit_diagrams()

    assert [o.name for o in outs] == ["circuit_zz.pdf", "circuit_vqc.pdf"]
    assert len(sizes) == 2
    # nearly one gate size in both drawings: the VQC is saved at the ZZ drawing's scale times a
    # small factor that keeps its widest label inside its box
    assert scales[1] == pytest.approx(scales[0] * figures.CIRCUIT_VQC_RELATIVE_SCALE)
    assert 1.0 <= figures.CIRCUIT_VQC_RELATIVE_SCALE <= 1.1
    widths = []
    for out, label_sizes in zip(outs, sizes, strict=True):
        box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)", out.read_bytes())
        widths.append(float(box.group(1)))
        assert label_sizes and min(label_sizes) >= 8.0, (out.name, min(label_sizes))
        assert b"/Type3" not in out.read_bytes(), out.name
    assert abs(widths[0] - figures.COLUMN_WIDTH_PT) < 1.0
    assert widths[1] < figures.COLUMN_WIDTH_PT  # placed at its natural size, never scaled up


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
        "reverse_damage": 0.03,
        "forward_over_reverse": 10.2,
        "per_target_adv_f1": {"qsvm": 0.5},
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


def test_split_leak_figure_reads_the_four_runs(tmp_path, monkeypatch):
    """Draft-v3 Fig. B: renders from toy group-split, leak-decay, FDIA split-gap and per-class
    runs, and fails when the per-class run was anchored on another group-split run."""
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    g, d, f, c = tmp_path / "g", tmp_path / "d", tmp_path / "f", tmp_path / "c"
    for p in (g, d, f, c):
        p.mkdir()
    near = "nn_frac_within_0.1"
    (g / "results.json").write_text(
        json.dumps(
            {
                "verdict": {
                    "level_row": 0.905,
                    "level_group_mean": 0.594,
                    "level_group_std": 0.008,
                    "floor_group_mean": 0.499,
                },
                "rows": [
                    {"split": "row (shipped)", near: 0.458},
                    {"split": "group", near: 0.047},
                    {"split": "group", near: 0.053},
                ],
            }
        )
    )

    def per_class(natural, attack):
        return {"per_class": {"natural": {"recall": natural}, "attack": {"recall": attack}}}

    units = {
        "row": per_class(0.83, 0.96),
        "group0": per_class(0.37, 0.83),
        "group1": per_class(0.35, 0.81),
    }

    def write_per_class(anchor):
        (c / "results.json").write_text(
            json.dumps({"anchor": {"archived_run": anchor, "pass": True}, "units": units})
        )

    write_per_class(g.name)
    curve = [{"gap": k, "macro_f1": 0.61 + 0.26 * np.exp(-(k - 1) / 2.5)} for k in range(1, 13)]
    (d / "results.json").write_text(
        json.dumps(
            {
                "curves": {"A_lgbm_matched": curve},
                "fits": {
                    "A_lgbm": {
                        "asymptote_fit": 0.61,
                        "amplitude_fit": 0.26,
                        "tau_released_rows": 2.5,
                    }
                },
            }
        )
    )
    var = {
        "rows": [{"split": "row", "level_full_lgbm": 1.0}],
        "summary": {"day_level_full_lgbm": {"mean": 0.99, "std": 0.006}},
    }
    (f / "results.json").write_text(
        json.dumps({"variants": {"fdia14_binary": var, "fdia118_binary": var}})
    )
    out = figures.split_leak_figure(g, d, f, "split_leak", c)
    assert out.exists() and out.stat().st_size > 0
    write_per_class("another_group_split_run")
    with pytest.raises(ValueError, match="anchored on 'another_group_split_run'"):
        figures.split_leak_figure(g, d, f, "split_leak", c)


def test_fdia_budget_figure_reads_square_and_h3_runs(tmp_path, monkeypatch):
    """Draft-v3 Fig. D: renders from toy Square per_seed.csv and H3 bad-data flag rates."""
    import pandas as pd

    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    sq, h3 = tmp_path / "sq", tmp_path / "h3"
    sq.mkdir()
    h3.mkdir()
    rows = []
    for fam in ("rbf_svm", "rf", "xgb", "lgbm", "mlp", "qsvm", "vqc"):
        for e in (0.1, 0.3, 0.5):
            for kind, v in (("square", 0.3), ("random", 0.6)):
                rows.append(
                    {
                        "seed": 0,
                        "family": fam,
                        "condition": f"{kind}@{e:g}",
                        "metric": "macro_f1",
                        "value": v,
                    }
                )
    pd.DataFrame(rows).to_csv(sq / "per_seed.csv", index=False)
    rates = {
        f"{s}|{e:g}|mlp": {"flag_rate": 1.0 if s == "U" else 0.0}
        for s in "UC"
        for e in (0.1, 0.25, 0.5)
    }
    (h3 / "results.json").write_text(json.dumps({"summary": {"bdd_flag_rates": rates}}))
    out = figures.fdia_budget_figure({"(a) grid": sq}, {"grid": h3}, "fdia_budget")
    assert out.exists() and out.stat().st_size > 0


def test_surrogate_table_reads_a_config_check_arm(tmp_path, table_dir):
    """Draft v3: the surrogate table can read one arm of a run_vqc_config_check run (the VQC at
    Table I's configuration), and gives the same table as a plain surrogate run's summary."""
    arm = {
        "mlp": {
            "surrogate_clean_minus_floor": -0.031,
            "surrogate_positive_rate": 0.966,
            "vqc_damage": 0.268,
            "classical_damage": 0.208,
            "damage_ratio_vqc_over_classical": 1.29,
            "forward_over_reverse": 4.62,
        },
        "mlp_balanced": {
            "surrogate_clean_minus_floor": 0.035,
            "surrogate_positive_rate": 0.56,
            "vqc_damage": 0.211,
            "classical_damage": 0.204,
            "damage_ratio_vqc_over_classical": 1.04,
            "forward_over_reverse": 0.87,
        },
        "rbf_svm": {
            "surrogate_clean_minus_floor": 0.090,
            "surrogate_positive_rate": None,
            "vqc_damage": 0.001,
            "classical_damage": 0.124,
            "damage_ratio_vqc_over_classical": 0.008,
            "forward_over_reverse": 0.006,
        },
    }
    for a, clean, rev, q_adv in zip(
        arm.values(),
        (0.465, 0.531, 0.587),
        (0.058, 0.242, 0.172),
        (0.488, 0.4175, 0.396),
        strict=True,
    ):
        a["surrogate_clean"] = clean
        a["classical_peer_damage"] = a["classical_damage"]
        a["reverse_damage"] = rev
        a["per_target_adv_f1"] = {"qsvm": q_adv}
    arm.update({"floor": 0.497, "eps": 0.5, "positive_rate_eval": 0.71})
    plain, cc = tmp_path / "plain", tmp_path / "cc"
    plain.mkdir()
    cc.mkdir()
    (plain / "results.json").write_text(json.dumps({"summary": arm, "per_seed": [{}] * 10}))
    stats = {"per_family": {"qsvm": {"clean_mean": 0.5631}}}
    (cc / "results.json").write_text(
        json.dumps(
            {
                "arms": {
                    "depth6": {
                        "surrogate_summary": arm,
                        "table_iii_stats": stats,
                        "unit_wall_s": [1.0] * 10,
                    }
                }
            }
        )
    )
    a = tables.surrogate_control_table(plain, "s_plain").read_text()
    b = tables.surrogate_control_table(cc, "s_cc", arm="depth6").read_text()

    def body(t):  # every column but the last, which only an arm can fill
        return [ln.rsplit(" & ", 1)[0] for ln in t.splitlines()]

    assert body(a) == body(b) and "4.62" in a and "0.242" in a
    # QSVM retention = per-target adversarial F1 / the arm's QSVM clean score
    assert "& 0.867 \\\\" in b and "& 0.741 \\\\" in b and "& 0.703 \\\\" in b
    assert a.count("& --- \\\\") == 3
    # the arm also writes every family's transferred retention under each surrogate
    ret = (tables.TABLE_DIR / "s_cc_transfer_retention.csv").read_text().splitlines()
    assert ret[0] == "family,mlp,mlp_balanced,rbf_svm"
    assert ret[1].startswith("qsvm,0.8666,0.7414,0.7032")


def test_matched_contest_table_marks_best_significance_and_the_mlp_fit(tmp_path, table_dir):
    """One compact table over the dimensionalities, from each dimensionality's merged run."""
    levels = {
        8: {"xgb": 0.5968, "mlp": 0.5223, "vqc": 0.5141, "baseline_majority": 0.9},
        16: {"xgb": 0.6071, "mlp": 0.4593, "baseline_majority": 0.9},
    }
    run_dirs = {}
    for d, names in levels.items():
        run_dirs[d] = tmp_path / f"merged_{d}"
        run_dirs[d].mkdir()
        (run_dirs[d] / "results.json").write_text(json.dumps(_results(names)))
    # the merged run says which MLP fit it holds: the class-weighted one at eight dimensions only
    (run_dirs[8] / "sources.txt").write_text("runs/x_weighted_mlp_baseline_binary_pca8 (mlp row)\n")
    with pytest.raises(FileNotFoundError, match="which MLP fit"):
        tables.matched_contest_table(run_dirs, "matched")
    (run_dirs[16] / "sources.txt").write_text("runs/merged_accuracy_binary_pca16\n")
    tex = tables.matched_contest_table(run_dirs, "matched").read_text(encoding="utf-8")
    rows = {line.split(" & ")[0]: line for line in tex.splitlines() if " & " in line}
    # a chance baseline is never the best model, however high it scores
    assert (
        "$^{a}$" in rows["XGBoost"]
        and "$^{a}$" not in rows["\\rowcolor{qgReferenceTint}Majority class"]
    )
    # the MLP is the class-weighted fit at eight dimensions and the unweighted one elsewhere
    cells = rows["MLP"].split(" & ")
    assert "c" not in cells[1] and "c" in cells[2]
    assert rows["\\rowcolor{qgQuantumTint}VQC"].rstrip(" \\").endswith("--")
    out = (table_dir / "matched.csv").read_text(encoding="utf-8")
    assert out.splitlines()[0] == "model,macro_f1_pca8,macro_f1_pca16"
    # the marks compare models with seeds: a model without seeds that leads fails loud
    doc = json.loads((run_dirs[16] / "results.json").read_text())
    doc["qsvm_zz"] = {"aggregate": {"macro_f1": {"mean": 0.70}}}
    (run_dirs[16] / "results.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="leads at 16 dimensions without seeds"):
        tables.matched_contest_table(run_dirs, "matched")


def test_mlp_fit_is_read_from_the_line_that_marks_the_mlp_row(tmp_path):
    """Only the line marked as the MLP row's source decides, and an unweighted run name on it
    does not count as the class-weighted fit, whatever its case."""
    cases = {
        "runs/a_weighted_mlp_baseline_binary_pca8 (mlp row: W1)\n": True,
        "runs/a_Unweighted_mlp_baseline_binary_pca8 (mlp row)\n": False,
        "runs/a_weighted_mlp_baseline_binary_pca8 (timings)\nruns/merged (mlp row)\n": False,
        "runs/merged_accuracy_binary_pca16\n": False,
    }
    for k, (text, expected) in enumerate(cases.items()):
        run = tmp_path / f"m{k}"
        run.mkdir()
        (run / "sources.txt").write_text(text)
        assert tables._mlp_is_class_weighted(run) is expected
    twice = tmp_path / "twice"
    twice.mkdir()
    (twice / "sources.txt").write_text("a (mlp row)\nb (mlp row)\n")
    with pytest.raises(ValueError, match="claim the MLP row"):
        tables._mlp_is_class_weighted(twice)
