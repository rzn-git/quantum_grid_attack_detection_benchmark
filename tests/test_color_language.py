"""The paper's color language (operator ruling, decision log 2026-11-19T16:18Z): one hue per
meaning, shared by the figures, the TikZ diagrams and the table rows; the console figure."""

import json
import re

import matplotlib.pyplot as plt
import pytest
from matplotlib.text import Text

from qgridbench.eval import design_tokens, figures, tables


def test_family_maps_every_model_key():
    assert design_tokens.family("qsvm_zz") == "quantum"
    assert design_tokens.family("qkernel_angle_ry") == "quantum"
    assert design_tokens.family("vqc") == "quantum"
    assert design_tokens.family("baseline_majority") == "reference"
    assert design_tokens.family("lgbm") == "classical"
    assert design_tokens.family_color("vqc") == design_tokens.MEANING["quantum"]


def test_tint_mixes_into_white():
    assert design_tokens.tint("#000000", 0.5) == "#808080"
    assert design_tokens.tint("#FFFFFF", 0.3) == "#FFFFFF"
    assert design_tokens.tint("#0072B2", 1.0) == "#0072B2"


def test_latex_colors_cover_every_meaning(tmp_path):
    out = figures.write_design_colors(tmp_path / "design_colors.tex")
    text = out.read_text(encoding="utf-8")
    for key, name in design_tokens.LATEX_COLOR_NAMES.items():
        assert f"\\definecolor{{{name}}}{{HTML}}{{{design_tokens.MEANING[key][1:]}}}" in text
        assert f"\\definecolor{{{name}Tint}}{{HTML}}" in text
    assert set(design_tokens.LATEX_COLOR_NAMES) == set(design_tokens.MEANING)
    # the diagrams draw every box outline at the figures' weight
    assert f"\\def\\dline{{{design_tokens.OUTLINE_PT}pt}}" in text
    assert design_tokens.OUTLINE_PT <= design_tokens.FRAME_PT <= 0.5


def test_table_rows_wear_their_family_tint():
    assert tables._row_tint("vqc") == "\\rowcolor{qgQuantumTint}"
    assert tables._row_tint("baseline_stratified_random") == "\\rowcolor{qgReferenceTint}"
    assert tables._row_tint("xgb") == ""


def test_console_verdict_names_the_two_failures():
    assert figures.console_verdict(0.38, 0.92) == "flooded"
    assert figures.console_verdict(0.013, 0.005) == "silent"
    assert figures.console_verdict(0.60, 0.02) == ""


def _console_run(tmp_path, drop=None, no_threshold=None, blind=None):
    rates = {
        "lgbm": (0.652, 0.007, 0.378, 0.920),
        "xgb": (0.466, 0.002, 0.454, 0.940),
        "logreg": (0.101, 0.014, 0.254, 0.924),
        "rf": (0.663, 0.0, 0.013, 0.005),
    }
    summary = {
        m: {
            "tpr@0.01_clean": tc,
            "fpr@0.01_clean": fc,
            "tpr@0.01_adv": ta,
            "fpr@0.01_adv": fa,
        }
        for m, (tc, fc, ta, fa) in (
            (k, (0.0, 0.0, 0.0, 0.0) if k == blind else v) for k, v in rates.items()
        )
        if m != drop
    }
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    per_seed = [
        {
            "op": {
                m: {"0.01": {"clean": {"threshold": float("inf") if m == no_threshold else 0.9}}}
                for m in summary
            }
        }
        for _ in range(3)
    ]
    (run / "results.json").write_text(json.dumps({"summary": summary, "per_seed": per_seed}))
    return run


def test_console_figure_is_one_column_at_8pt(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "out")
    closed = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: (closed.append(fig), real_close(fig)))
    out = figures.console_figure(_console_run(tmp_path), "console")
    assert out.exists()
    fig = closed[-1]
    assert abs(fig.get_figwidth() * 72 - design_tokens.COLUMN_WIDTH_PT) < 0.5
    fig.canvas.draw()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    assert min(t.get_fontsize() for t in texts) >= design_tokens.LABEL_PT
    words = {t.get_text() for t in texts}
    assert {"flooded", "silent"} <= words and "blind" not in words
    # four detectors: the MLP, which has no 1% operating point, is not a row
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert len(labels) == 4 and not any("MLP" in label for label in labels)
    # an unmoved reading keeps both marks visible: the attacked dot sits inside the clean ring
    # (XGBoost's detection, the random forest's false alarms), and every other dot is full size
    inner = [
        [ln for ln in ax.lines if ln.get_markersize() == figures.CONSOLE_INNER_MS]
        for ax in fig.axes[:2]
    ]
    assert [len(group) for group in inner] == [1, 1]
    assert inner[0][0].get_xdata()[0] == 0.454 and inner[1][0].get_xdata()[0] == 0.005


def test_console_figure_fails_loud_on_a_missing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "out")
    with pytest.raises(KeyError, match="no models"):
        figures.console_figure(_console_run(tmp_path, drop="rf"), "console")


def test_console_figure_fails_loud_without_an_operating_point(tmp_path, monkeypatch):
    """A detector with no finite threshold at the budget is not drawn as a silent one."""
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "out")
    with pytest.raises(ValueError, match="no finite threshold"):
        figures.console_figure(_console_run(tmp_path, no_threshold="rf"), "console")
    with pytest.raises(ValueError, match="of attacks clean"):
        figures.console_figure(_console_run(tmp_path, blind="rf"), "console")


def test_checklist_figure_refuses_text_that_does_not_fit(tmp_path, monkeypatch):
    """Checklist tiles: an overlong label fails the build instead of printing over its value."""
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    rows = [
        "order,panel,unit,side,part,label,value,value_label,scale,suffix,run,file,key",
        "1,Split,F1,a,value,Row,0.9,0.905,1,,r,f,k",
        "1,Split,F1,b,value,Files,0.6,0.594,1,,r,f,k",
    ]
    t = tmp_path / "p.csv"
    t.write_text("\n".join(rows) + "\n")
    out = figures.checklist_figure(t, "chk")
    box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)", out.read_bytes())
    assert abs(float(box.group(1)) - design_tokens.COLUMN_WIDTH_PT) < 0.5
    long_label = "A reading label far too long for a half-column tile at eight points"
    t.write_text("\n".join([*rows[:2], rows[2].replace("Files", long_label)]) + "\n")
    with pytest.raises(ValueError, match="does not fit|overlaps"):
        figures.checklist_figure(t, "chk")


def _multiclass_run(tmp_path, models=("lgbm", "qsvm")):
    run = tmp_path / "mc"
    run.mkdir()
    lines = ["regime,model,seed,macro_f1"]
    for regime, model, score in (("full", "lgbm", 0.8), ("pca", "lgbm", 0.2), ("pca", "qsvm", 0.1)):
        lines += [f"{regime},{model},{seed},{score}" for seed in (0, 1)]
    (run / "per_seed.csv").write_text("\n".join(lines) + "\n")
    cm = [[5, 1, 0], [1, 5, 0], [0, 2, 4]]
    for fname in figures.MULTICLASS_CONFUSION_FILES.values():
        payload = {"labels": [1, 2, 3], "per_model": {m: {"confusion_counts": cm} for m in models}}
        (run / fname).write_text(json.dumps(payload))
    return run


def test_multiclass_summary_figure_reads_scores_and_matrices(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    out = figures.multiclass_summary_figure(_multiclass_run(tmp_path), "mc")
    box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)", out.read_bytes())
    assert abs(float(box.group(1)) - design_tokens.TEXT_WIDTH_PT) < 0.5
    assert b"/Type3" not in out.read_bytes()


def test_multiclass_summary_figure_fails_loud_on_a_missing_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    with pytest.raises(KeyError, match="no multiclass confusion"):
        figures.multiclass_summary_figure(_multiclass_run(tmp_path, models=("lgbm",)), "mc")


def _square_run(tmp_path, drop=None):
    run = tmp_path / "sq"
    run.mkdir(parents=True)
    cell = {
        "clean_attack_recall": 0.8,
        "square_attack_recall": 0.2,
        "clean_false_alarm": 0.5,
        "square_false_alarm": 1.0,
    }
    square = {m: cell for m in figures.CONTROLS_SQUARE_MODELS if m != drop}
    (run / "results.json").write_text(json.dumps({"summary": {"square": square}}))
    return run


def _robustness_csv(tmp_path, vqc_own="0.0604"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rob = tmp_path / "rob.csv"
    rob.write_text(
        "model,clean_f1,adv_f1,adv_f1_minus_floor,retention,whitebox_retention,"
        "random_flip_rate,hsj_median_linf_sigma,hsj_restart_std,hsj_found_share,"
        "lowest_f1,lowest_attack\n"
        # the distance to the floor at six decimals, as the table writer stores it: at four,
        # -0.085485 would reach the figure as -0.0855 and print -0.086 beside the table's -0.085
        "qsvm,0.5631,0.4991,-0.001215,0.8863,0.0641,0.4467,3.9196,1.0349,,0.0066,S\n"
        "rbf_svm,0.5871,0.4148,-0.085485,0.7065,0.6848,0.2400,0.0132,0.0007,0.2400,"
        "0.1429,W\n"
        "lgbm,0.6339,0.3734,-0.126901,0.5890,,0.2433,0.8818,0.0979,,0.1516,S\n"
        "mlp_weighted,0.5211,0.2594,-0.240899,0.4977,0.0764,,,,,0.0398,W\n"
        f"vqc,0.4756,0.0287,-0.471545,0.0604,{vqc_own},0.2467,0.4809,0.0671,,0.0287,W\n"
    )
    return rob


def test_controls_figure_is_text_width_and_names_the_table_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    seen = []
    monkeypatch.setattr(plt, "close", lambda fig: seen.append(fig))
    out = figures.controls_figure(_robustness_csv(tmp_path), _square_run(tmp_path), "controls")
    box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)", out.read_bytes())
    assert abs(float(box.group(1)) - design_tokens.TEXT_WIDTH_PT) < 0.5
    words = {t.get_text() for t in seen[0].findobj(Text)}
    # the table's row names, the two readings the text quotes, and the marker key
    assert {"QSVM", "Class-weighted MLP", "3.92", "45%", "24%"} <= words
    assert {"clean", "shared PGD", "own white-box attack", "Square search"} <= words
    # the retentions the text argues from are printed per row, a dash where no white-box
    # attack was run (the TSG build has no table to carry them)
    assert {"0.886", "0.064", "0.707", "0.685", "0.589", "–"} <= words
    # a row the black-box attack started on for only part of the points prints that share
    assert "on 24%" in words
    # every second row carries a band across the whole figure, so a row can be followed
    bands = [a for a in seen[0].artists if isinstance(a, plt.Rectangle)]
    rows = len(seen[0].axes[0].get_yticks())
    assert len(bands) == (rows + 1) // 2 and all(b.get_width() == 1.0 for b in bands)
    # and every panel lets the bands show through, the panels of (d) included
    assert len(seen[0].axes) >= 5
    assert all(ax.get_facecolor()[3] == 0.0 for ax in seen[0].axes)


def test_controls_figure_three_panel_form_titles_its_panels(tmp_path, monkeypatch):
    """The ten-page build's form: no panel (d), no Square marker in the key, titled (b) and (c)."""
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    seen = []
    monkeypatch.setattr(plt, "close", lambda fig: seen.append(fig))
    figures.controls_figure(
        _robustness_csv(tmp_path), _square_run(tmp_path), "controls_abc", with_catch=False
    )
    words = {t.get_text() for t in seen[0].findobj(Text)}
    assert {"(b) Black-box attack", "(c) Its noise control", "on 24% of points"} <= words
    # the third column of (a): each row's lowest macro-F1 over the attacks run on the model,
    # with the attack that reached it (P shared PGD, W own white-box, S Square search)
    assert {"Lowest F1", "(attack)", "0.007 S", "0.143 W", "0.029 W"} <= words
    # a partial start is a hollow square: the hollow circle is keyed as the clean score
    squares = [ln for ax in seen[0].axes for ln in ax.lines if ln.get_marker() == "s"]
    assert squares and all(ln.get_markerfacecolor() == "white" for ln in squares)
    # and only a partial start: the same panel still draws full starts as filled circles
    full = [
        ln
        for ln in squares[0].axes.lines
        if ln.get_marker() == "o" and ln.get_markerfacecolor() != "white"
    ]
    assert full
    assert "Square search" not in words and "(d) Share of rows flagged" not in words
    assert len(seen[0].axes) == 3


def test_controls_figure_fails_loud_on_bad_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    square = _square_run(tmp_path)
    # the VQC's shared attack is its white-box attack: a table that says otherwise is wrong
    with pytest.raises(ValueError, match="should be its white-box one"):
        figures.controls_figure(_robustness_csv(tmp_path / "a", vqc_own="0.5000"), square, "c")
    rob = _robustness_csv(tmp_path / "b")
    rob.write_text(rob.read_text().replace("-0.085485", "-0.200000"))
    with pytest.raises(ValueError, match="disagree on the floor"):
        figures.controls_figure(rob, square, "c")
    # the three-panel form prints each model's lowest attacked score: a table written
    # before that column existed, or with an attack letter it does not know, is refused
    old = _robustness_csv(tmp_path / "e")
    old.write_text(old.read_text().replace(",0.1429,W", ",,"))
    with pytest.raises(ValueError, match="no lowest_f1"):
        figures.controls_figure(old, square, "c", with_catch=False)
    odd = _robustness_csv(tmp_path / "f")
    odd.write_text(odd.read_text().replace(",0.1516,S", ",0.1516,X"))
    with pytest.raises(ValueError, match="no lowest_f1"):
        figures.controls_figure(odd, square, "c", with_catch=False)
    with pytest.raises(KeyError, match="no models"):
        figures.controls_figure(
            _robustness_csv(tmp_path / "c"), _square_run(tmp_path / "d", drop="qsvm"), "c"
        )


def _width_run(tmp_path, radii=("0.0", "2.0")):
    run = tmp_path / "width"
    run.mkdir(parents=True)
    widths = {}
    for mult, (clean, attacked, mass) in {
        "1.0": (0.606, 0.415, 0.0),
        "0.03": (0.607, 0.242, 0.158),
        "0.01": (0.544, 0.047, 0.399),
        "0.001": (0.533, 0.0, 0.708),
    }.items():
        last = {"macro_f1": attacked, "f1_minus_floor": attacked - 0.415, "kernel_mass_ratio": mass}
        curve = {radii[0]: {"macro_f1": clean}, radii[1]: last}
        if mult == "1.0" and len(radii) > 2:
            curve[radii[2]] = last
        widths[mult] = {"curve": curve}
    payload = {"constant_predictor_floor": 0.415, "per_width": widths}
    (run / "results.json").write_text(json.dumps({"p8a_lengthscale_control": payload}))
    return run


def test_width_sweep_figure_names_both_end_states_and_one_radius(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    seen = []
    monkeypatch.setattr(plt, "close", lambda fig: seen.append(fig))
    out = figures.width_sweep_figure(_width_run(tmp_path / "a"), "width")
    box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)", out.read_bytes())
    assert abs(float(box.group(1)) - design_tokens.COLUMN_WIDTH_PT) < 0.5
    # the words come from the regime thresholds: two inverted widths, one disabled
    assert {"inverted", "disabled"} <= {t.get_text() for t in seen[0].findobj(Text)}
    with pytest.raises(ValueError, match="different radii"):
        figures.width_sweep_figure(_width_run(tmp_path / "b", radii=("0.0", "2.0", "3.0")), "w")


def _tabular_runs(tmp_path, rung="cap6"):
    payloads = {
        "lgbm": {"verdict": {"level_row": 0.905, "level_group_mean": 0.594}},
        "tabm": {"verdict": {"row_mean": 0.771, "group_mean": 0.604}},
        "ftt": {"verdict": {"row_mean": 0.636, "group_mean": 0.596}},
        "tabicl": {
            "verdict": {
                "row_lgbm": 0.631,
                "group_lgbm_mean": 0.557,
                "row_tabicl": 0.629,
                "group_tabicl_mean": 0.590,
            }
        },
        "capacity": {
            "verdict": {"matched": {"rung": rung}},
            "rungs": [{"name": "cap6", "row_mean": 0.617, "group_mean": 0.588}],
        },
    }
    runs = {}
    for role, payload in payloads.items():
        runs[role] = tmp_path / role
        runs[role].mkdir(parents=True)
        (runs[role] / "results.json").write_text(json.dumps(payload))
    return runs


def test_tabular_split_figure_prints_each_gap_and_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    seen = []
    monkeypatch.setattr(plt, "close", lambda fig: seen.append(fig))
    figures.tabular_split_figure(_tabular_runs(tmp_path / "a"), "tab")
    words = {t.get_text() for t in seen[0].findobj(Text)}
    # the gap of every model, the regularized rung included, and the highest row score on top
    assert {"0.311", "0.167", "0.040", "0.074", "0.039", "0.029"} <= words
    labels = [t.get_text() for t in seen[0].axes[0].get_yticklabels()]
    assert labels[-1] == "LightGBM, tuned" and labels[0] == "LightGBM, regularized"
    with pytest.raises(KeyError, match="rungs named"):
        figures.tabular_split_figure(_tabular_runs(tmp_path / "b", rung="cap9"), "tab")
    runs = _tabular_runs(tmp_path / "c")
    del runs["tabm"]
    with pytest.raises(KeyError, match="no run for"):
        figures.tabular_split_figure(runs, "tab")
    # a score the fixed axis would clip is refused, not drawn at the edge
    runs = _tabular_runs(tmp_path / "d")
    (runs["ftt"] / "results.json").write_text(
        json.dumps({"verdict": {"row_mean": 0.636, "group_mean": 0.48}})
    )
    with pytest.raises(ValueError, match="outside the axis"):
        figures.tabular_split_figure(runs, "tab")


def _gas_runs(tmp_path, drop=None, logreg_before=0.537):
    def run(name, pca, full):
        d = tmp_path / name
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps({"pca8": {"mean": pca}, "full_mean": full}))
        return d

    before = {"rf": 0.588, "qsvm_zz": 0.556, "logreg": logreg_before}
    after = {"rf": 0.761, "qsvm_zz": 0.748, "logreg": 0.552}
    if drop:
        before.pop(drop)
    return (
        run("before", before, {"rf": 0.634, "lgbm": 0.640}),
        run("after", after, {"rf": 0.856, "lgbm": 0.850}),
    )


def test_gas_pressure_figure_orders_rows_by_the_score_after(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    seen = []
    monkeypatch.setattr(plt, "close", lambda fig: seen.append(fig))
    figures.gas_pressure_figure(*_gas_runs(tmp_path / "a"), "gas")
    labels = [t.get_text() for t in seen[0].axes[0].get_yticklabels()]
    assert labels == [
        "LightGBM, all features",
        "Random forest, all features",
        "Logistic regression",
        "QSVM (ZZ)",
        "Random forest",
    ]
    with pytest.raises(KeyError, match="no reading for"):
        figures.gas_pressure_figure(*_gas_runs(tmp_path / "b", drop="qsvm_zz"), "gas")
    with pytest.raises(ValueError, match="outside the axis"):
        figures.gas_pressure_figure(*_gas_runs(tmp_path / "c", logreg_before=0.45), "gas")


def _bridge_inputs(tmp_path, capped=0.6038, quantum=True, retuned=0.8641, floor="0.5005"):
    tmp_path.mkdir(parents=True)
    matched = tmp_path / "matched.csv"
    rows = ["model,macro_f1", "xgb,0.6071", "lgbm,0.6061", f"baseline_stratified_random,{floor}"]
    if quantum:
        rows += ["qsvm_angle_ry,0.5742", "qsvm_zz,0.5670"]
    matched.write_text("\n".join(rows) + "\n")
    run = tmp_path / "cap"
    run.mkdir()
    cell = {"full_train": 0.8010, "capped": capped}
    path = {
        "full_features": {"lgbm": {"all_features": 0.9058, "pca16": {"full_train": 0.8547}}},
        "matched_regime": {"lgbm": {"all_features": retuned, "pca16": cell}},
    }
    legacy = {**cell, "n_full": 47020, "n_capped": 2000}
    (run / "results.json").write_text(
        json.dumps({"pca16_lgbm": legacy, "fixed_hyperparameters": path})
    )
    return matched, run


def test_accuracy_bridge_figure_prints_each_step_and_its_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    seen = []
    monkeypatch.setattr(plt, "close", lambda fig: seen.append(fig))
    figures.accuracy_bridge_figure(*_bridge_inputs(tmp_path / "a"), "bridge")
    ax = seen[0].axes[0]
    printed = {t.get_text() for t in ax.texts}
    assert {"0.906", "0.864", "0.801", "0.604", "0.607", "0.574"} <= printed
    assert {"−0.042", "−0.063", "−0.197", "−0.033"} <= printed
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == [
        "LightGBM, all features",
        "Matched-contest tuning",
        "16 components",
        "2000-row cap",
        "XGBoost, matched",
        "QSVM (RY-angle), matched",
    ]
    # each marker sits at its score, on its own row
    markers = [ln for ln in ax.lines if ln.get_marker() == "o"]
    assert [(ln.get_xdata()[0], ln.get_ydata()[0]) for ln in markers] == [
        (0.9058, 0),
        (0.8641, 1),
        (0.8010, 2),
        (0.6038, 3),
        (0.6071, 4),
        (0.5742, 5),
    ]
    # each cost bar spans the step it names
    spans = [(round(b.get_x(), 4), round(b.get_x() + b.get_width(), 4)) for b in ax.patches]
    assert spans == [(0.8641, 0.9058), (0.8010, 0.8641), (0.6038, 0.8010), (0.5742, 0.6071)]
    assert seen[0].get_figwidth() == pytest.approx(design_tokens.COLUMN_WIDTH_PT / 72)


def test_accuracy_bridge_figure_fails_loud_on_bad_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    with pytest.raises(KeyError, match="both families"):
        figures.accuracy_bridge_figure(*_bridge_inputs(tmp_path / "a", quantum=False), "bridge")
    with pytest.raises(ValueError, match="should each cost"):
        figures.accuracy_bridge_figure(*_bridge_inputs(tmp_path / "b", retuned=0.95), "bridge")
    with pytest.raises(ValueError, match="outside the axis"):
        figures.accuracy_bridge_figure(*_bridge_inputs(tmp_path / "c", floor="0.40"), "bridge")
    matched, run = _bridge_inputs(tmp_path / "d")
    (run / "results.json").write_text(json.dumps({"pca16_lgbm": {}}))
    with pytest.raises(KeyError, match="no fixed-hyperparameter path"):
        figures.accuracy_bridge_figure(matched, run, "bridge")


def _worst_radius_run(tmp_path, drop: str | None = None):
    models = {}
    for m, clean, curve in (
        ("rbf_svm", 0.587, {"0.01": 0.33, "0.1": 0.14, "0.5": 0.40}),
        ("qsvm", 0.563, {"0.01": 0.27, "0.1": 0.02, "0.5": 0.04}),
    ):
        if m == drop:
            continue
        worst = min(curve, key=curve.get)
        models[m] = {
            "clean": clean,
            "radii": [0.01, 0.1, 0.5],
            "mean_curve": curve,
            "per_start": {k: [v - 0.01, v, v + 0.01] for k, v in curve.items()},
            "worst": {"radius": float(worst), "mean": curve[worst]},
        }
    run = tmp_path / "worst"
    run.mkdir(parents=True)
    node = {"constant_predictor_floor": {"floor": 0.415}, "models": models}
    (run / "results.json").write_text(json.dumps({"sources": {"msu": node}}))
    return run


def test_radius_curve_figure_names_both_curves_and_marks_the_worst_radius(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    seen = []
    monkeypatch.setattr(plt, "close", lambda fig: seen.append(fig))
    out = figures.radius_curve_figure(
        _worst_radius_run(tmp_path), _robustness_csv(tmp_path), "radius"
    )
    box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)", out.read_bytes())
    assert abs(float(box.group(1)) - design_tokens.COLUMN_WIDTH_PT) < 0.5
    ax = seen[0].axes[0]
    words = {t.get_text() for t in seen[0].findobj(Text)}
    assert {"RBF-SVM", "QSVM", "clean", "worst radius"} <= words
    # both floors are drawn and named, each in its own line style
    assert {"constant-predictor floor", "stratified-random floor"} <= words
    styles = sorted(ln.get_linestyle() for ln in ax.lines if len(set(ln.get_ydata())) == 1)
    assert styles[:2] == ["--", ":"]
    # one diamond per model, at that model's worst radius and score
    diamonds = sorted(
        (float(ln.get_xdata()[0]), float(ln.get_ydata()[0]))
        for ln in ax.lines
        if ln.get_marker() == "D"
    )
    assert diamonds == [(0.1, 0.02), (0.1, 0.14)]


def test_radius_curve_figure_fails_loud_on_a_missing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    rob = _robustness_csv(tmp_path)
    with pytest.raises(KeyError, match="no white-box curve"):
        figures.radius_curve_figure(_worst_radius_run(tmp_path, drop="qsvm"), rob, "radius")


def test_radius_curve_figure_fails_loud_on_a_score_outside_the_drawn_range(tmp_path, monkeypatch):
    """A source whose clean score sits above the axis (a grid at 0.98) is not clipped silently."""
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    run = _worst_radius_run(tmp_path)
    doc = json.loads((run / "results.json").read_text())
    doc["sources"]["msu"]["models"]["qsvm"]["clean"] = 0.98
    (run / "results.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="leaves the drawn range"):
        figures.radius_curve_figure(run, _robustness_csv(tmp_path), "radius")
