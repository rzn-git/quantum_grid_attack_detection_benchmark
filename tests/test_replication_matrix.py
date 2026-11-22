"""Draft-v3 replication matrix: the builder's key syntax and labels, and the figure (toy files)."""

import ast
import csv
import json
import sys

import pytest

from qgridbench.eval import figures
from qgridbench.utils.run_tracking import REPO_ROOT

if str(REPO_ROOT) not in sys.path:  # experiments/ is not installed
    sys.path.insert(0, str(REPO_ROOT))
from experiments import build_replication_matrix as b  # noqa: E402


def test_json_csv_and_difference_keys(tmp_path):
    j = tmp_path / "r.json"
    j.write_text(json.dumps({"a": {"x": 0.9, "y": 0.1}, "l": [{"v": 3}]}))
    assert b._json_value(j, "a/x") == 0.9 and b._json_value(j, "l/0/v") == 3
    with pytest.raises(KeyError, match="no key 'z'"):
        b._json_value(j, "a/z")
    c = tmp_path / "t.csv"
    c.write_text('id,figure,depth6\n1,"forward/reverse, as tuned",4.6\n2,other,1.0\n')
    assert b._csv_value(c, "id=1;figure=forward/reverse| as tuned>depth6") == 4.6
    run = tmp_path / "results" / "runs" / "R"
    run.mkdir(parents=True)
    (run / "results.json").write_text(json.dumps({"p": {"t": 0.917, "w": 0.064}}))
    b.REPO_ROOT = tmp_path
    try:
        v, _ = b.resolve({"run": "R", "file": "results.json", "key": "p/t - p/w"})
    finally:
        b.REPO_ROOT = REPO_ROOT
    assert v == pytest.approx(0.853)


def test_labels_turn_signs_and_skip_words():
    assert b.label(-0.0078, "neg:+.3f") == "+0.008"
    assert b.label(0.311, ".3f") == "0.311"
    assert b.label(True, "") == "" and b.label("PHYSICS", "") == ""


def test_fold_label_reads_the_same_in_both_directions():
    # row 2 of the replication figure: how many times the refit moves the transfer ratio
    assert b.label(4.6166 / 0.8729, "fold:.2f") == "5.29×"
    assert b.label(0.5, "fold:.2f") == b.label(2.0, "fold:.2f") == "2.00×"
    assert b.label(0.9918, "fold:.2f") == "1.01×"
    for bad in (0.0, -2.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive ratio"):
            b.label(bad, "fold:.2f")


def test_curated_rows_reject_a_row_with_an_unquoted_comma(tmp_path):
    # 2026-11-21: a note with a bare comma gave line 23 of the cells file nine fields, and the
    # generated note lost its tail without an error
    p = tmp_path / "cells.csv"
    p.write_text("claim,note\nrow_a,one field\nrow_b,two, fields\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        b.curated_rows(p)
    p.write_text('claim,note\nrow_b,"two, fields"\n', encoding="utf-8")
    assert b.curated_rows(p) == [{"claim": "row_b", "note": "two, fields"}]
    p.write_text("claim,note\nrow_c\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        b.curated_rows(p)
    for shipped in (b.CELLS, b.PANELS):
        assert b.curated_rows(shipped)


def test_shipped_cells_print_the_statistic_their_threshold_tests():
    # row 2: the factor by which the refit moves the transfer ratio, in all four columns;
    # row 6: the QSVM's spread over restarts where the row was read
    with b.CELLS.open(encoding="utf-8") as f:
        cells = {(c["claim"], c["dataset"]): c for c in csv.DictReader(f)}
    for dataset in ("msu", "gas", "fdia14", "fdia118"):
        cell = cells[("surrogate_specification", dataset)]
        assert cell["fmt"] == "fold:.2f" and " over " in cell["key"], dataset
    for dataset in ("msu", "gas"):
        cell = cells[("starting_point", dataset)]
        assert cell["fmt"] == ".2f" and "cv" in cell["key"].split("/"), dataset


def test_ratio_of_two_csv_rows(tmp_path):
    run = tmp_path / "results" / "runs" / "R"
    run.mkdir(parents=True)
    (run / "t.csv").write_text("id,depth6\n1,4.6\n4,0.92\n")
    b.REPO_ROOT = tmp_path
    try:
        v, _ = b.resolve({"run": "R", "file": "t.csv", "key": "id=1>depth6 over id=4>depth6"})
    finally:
        b.REPO_ROOT = REPO_ROOT
    assert v == pytest.approx(5.0)


def test_each_printed_statistic_has_a_threshold_or_a_rule():
    # the marks run *, †, ‡, § down the rows; only row D is read by a rule with no single number
    stats = {tag: stat for tag, _, stat in figures.REPLICATION_CLAIMS.values()}
    marks = {tag: s[-1] for tag, s in stats.items() if s and s[-1] in "*†‡§"}
    assert marks == {"3": "*", "4": "†", "6": "‡", "B": "§"}
    assert [tag for tag, s in stats.items() if not s] == ["D"]
    assert "≥ 2×" in stats["2"] and "0.15" in stats["6"]


def test_cell_text_is_one_line():
    # regression (decision log 2026-11-19T14:21Z): a two-line cell ran into the row below
    assert figures.replication_cell_text("yes", "+0.031") == "yes +0.031"
    assert figures.replication_cell_text("yes", "") == "yes"
    assert "\n" not in figures.replication_cell_text("partly", "0.042")
    assert figures.replication_cell_text("no", "-0.10") == "no −0.10"
    # the study's own column prints the statistic alone, or the word when the row has none
    assert figures.replication_cell_text("found", "-0.22", origin=True) == "−0.22"
    assert figures.replication_cell_text("found", "", origin=True) == "found"


def test_rows_carry_the_checklist_numbers_then_letters():
    tags = [tag for tag, _, _ in figures.REPLICATION_CLAIMS.values()]
    assert tags == ["1", "2", "3", "4", "5", "6", "7", "A", "B", "C", "D"]
    assert "regularization" not in figures.REPLICATION_CLAIMS


def test_checklist_rows_open_with_the_checklist_figures_names():
    """A checklist row has one name in both figures: rows 1 to 7 of the replication figure open
    with the panel name the checklist figure prints (experiments/instance_panels.csv)."""
    with (REPO_ROOT / "experiments" / "instance_panels.csv").open(encoding="utf-8") as f:
        names = {r["order"]: r["panel"] for r in csv.DictReader(f)}
    for tag, finding, _ in figures.REPLICATION_CLAIMS.values():
        if tag.isdigit():
            assert finding.startswith(names[tag]), (tag, finding, names[tag])


def test_key_names_every_word_a_cell_can_print():
    """The key under the rows explains each reading a cell can carry. The study's own column is
    aside: it has no fill, so it cannot be taken for a column of no (fix-check of 2026-11-21,
    F10)."""
    key = dict(figures.REPLICATION_KEY)
    assert set(key) == set(figures.REPLICATION_STYLE) - {"origin"}
    assert key["yes"] == "it reproduces" and key["no"] == "it does not"
    assert key["partial"].startswith("in part") and "no registered" in key["not_tested"]
    fills = {reading: fill for reading, (fill, _) in figures.REPLICATION_STYLE.items()}
    assert fills["origin"] == "#FFFFFF" and fills["no"] != fills["origin"]


def _constants(script: str, *names: str) -> list[float]:
    """Module-level numeric constants of a runner, read from its source: importing the two
    split runners loads the whole model zoo."""
    tree = ast.parse((REPO_ROOT / "experiments" / script).read_text(encoding="utf-8"))
    found: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if isinstance(target, ast.Name):
            pairs = [(target, value)]
        elif isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
            pairs = list(zip(target.elts, value.elts, strict=True))
        else:
            continue
        for name, val in pairs:
            if isinstance(name, ast.Name) and name.id in names:
                found[name.id] = ast.literal_eval(val)
    return [found[n] for n in names]


def test_key_and_marks_print_the_runners_thresholds():
    """The band of a partly and the gas thresholds are typed in the figure; they must equal the
    constants the verdict runners applied (fix-check of 2026-11-21, F5: a wrong band, a wrong
    gas threshold and swapped mark texts all passed the earlier test)."""
    a_min, b_min, alpha = _constants("run_gas_robustness_verdict.py", "A_MIN", "B_MIN", "ALPHA")
    port, not_ = _constants("run_gas_split_ladder.py", "LADDER_PORT", "LADDER_NOT")
    assert _constants("run_fdia_split_gap.py", "G_HYP", "G_KILL") == [port, not_]
    key = dict(figures.REPLICATION_KEY)
    assert f"row 1: a gap of {not_:.2f} to {port:.2f}" in key["partial"]
    note = figures.REPLICATION_NOTE
    assert note.startswith("On the gas data:")
    marks = "*†‡§"
    at = [note.index(m) for m in marks]
    assert at == sorted(at)
    said = {
        m: note[i + 1 : j].strip() for m, i, j in zip(marks, at, [*at[1:], len(note)], strict=True)
    }
    assert said["*"] == f"{a_min:.2f}."
    assert said["†"] == "A rule on black-box rankings."
    assert said["‡"] == "Or a change of model order."
    assert said["§"] == f"−{b_min:.2f} with p < {alpha:.2f}."
    # the thresholds the rows print are the main study's and the grids'; each is held to the
    # constant its verdict runner applies (fix-check of the round of 07:15Z, F6: five of six
    # changed thresholds passed)
    factor, cv, ratio = _constants(
        "run_gas_robustness_verdict.py", "ITEM2_FACTOR", "ITEM5_CV", "ITEM6_RATIO"
    )
    f_max, grid_a_min = _constants("run_fdia_verdicts.py", "F_MAX", "A_MIN")
    stats = {tag: stat for tag, _, stat in figures.REPLICATION_CLAIMS.values()}
    assert stats["1"] == f"split gap ≥ {port:.2f}"
    assert stats["2"] == f"fit moves it ≥ {factor:g}×"
    assert stats["3"] == f"gap ≥ {grid_a_min:.2f}*"
    assert stats["6"] == f"spread ≥ {cv:.2f}‡"
    assert stats["7"] == f"flip ratio ≥ {ratio:g}"
    assert stats["B"] == f"gap ≤ −{abs(f_max):.2f}§"
    # rows 5 and C have no named constant in a runner; their printed thresholds are pinned
    assert stats["5"] == "gap ≥ 0.20" and stats["C"] == "Spearman ρ < 0.9"


def test_rows_3_and_7_say_what_their_cells_measure():
    """Row 3 is one model's retention gap and row 7 a random-noise flip ratio; their earlier
    names claimed a ranking and an attack (fix-check of 2026-11-21, F2). The cell notes say
    what is measured."""
    names = {tag: finding for tag, finding, _ in figures.REPLICATION_CLAIMS.values()}
    assert names["3"] == "Attack availability moves QSVM retention"
    assert names["7"] == "Measurement geometry: noise flips the QSVM"
    with b.CELLS.open(encoding="utf-8") as f:
        notes = {(c["claim"], c["dataset"]): c["note"] for c in csv.DictReader(f)}
    assert "retention" in notes[("attack_availability", "msu")]
    assert "random-noise flip" in notes[("measurement_geometry", "msu")]


def test_sum_key_reads_two_values_of_one_file(tmp_path):
    run = tmp_path / "results" / "runs" / "R"
    run.mkdir(parents=True)
    (run / "results.json").write_text(json.dumps({"floor": 0.415, "p": {"d": -0.402}}))
    b.REPO_ROOT = tmp_path
    try:
        v, _ = b.resolve({"run": "R", "file": "results.json", "key": "p/d + floor"})
        ratio, _ = b.resolve({"run": "R", "file": "results.json", "key": "floor over p/d"})
    finally:
        b.REPO_ROOT = REPO_ROOT
    assert v == pytest.approx(0.013)
    assert ratio == pytest.approx(0.415 / -0.402)


def test_checklist_reading_fails_loud_on_a_value_past_its_scale():
    row = {"part": "value", "value": "1.2", "value_label": "1.20", "scale": "1", "suffix": ""}
    with pytest.raises(ValueError, match="leaves its scale"):
        figures.checklist_reading({"value": row})


def test_figure_renders_and_rejects_unknown_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    rows = ["claim,dataset,reproduces,label,value,run,file,key,note"]
    for c in figures.REPLICATION_CLAIMS:
        for d in figures.REPLICATION_DATASETS:
            rows.append(f"{c},{d},{'origin' if d == 'msu' else 'yes'},0.1,0.1,r,f,k,")
    t = tmp_path / "m.csv"
    t.write_text("\n".join(rows) + "\n")
    drawn = []
    monkeypatch.setattr(
        figures.plt, "close", lambda fig: drawn.extend(x.get_text() for x in fig.axes[0].texts)
    )
    assert figures.replication_matrix_figure(t, "m").exists()
    # the key and the marks are drawn, not only defined
    for reading, meaning in figures.REPLICATION_KEY:
        assert meaning in drawn and figures.REPLICATION_STYLE[reading][1] in drawn
    for line in figures.REPLICATION_NOTE_LINES:
        assert line in drawn
    t.write_text("\n".join(rows + ["novel,msu,yes,,,,,,"]) + "\n")
    with pytest.raises(ValueError, match="no display name"):
        figures.replication_matrix_figure(t, "m")


def test_checklist_figure_needs_both_readings(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "fig")
    ok = [
        "order,panel,unit,side,part,label,value,value_label,scale,suffix,run,file,key",
        "1,Split,F1,a,value,Row,0.9,0.905,1,,r,f,k",
        "1,Split,F1,b,value,Files,0.6,0.594,1,,r,f,k",
        "2,Start,sigma,a,lo,One run,0.57,0.57,5,,r,f,k",
        "2,Start,sigma,a,hi,One run,4.99,4.99,5,,r,f,k",
        "2,Start,sigma,b,mean,Restarts,3.41,3.41,5,,r,f,k",
        "2,Start,sigma,b,std,Restarts,1.39,1.39,5,,r,f,k",
        "2,Start,sigma,ab,mark,a level,1.0,1.0,5,,r,f,k",
    ]
    t = tmp_path / "p.csv"
    t.write_text("\n".join(ok) + "\n", encoding="utf-8")
    assert figures.checklist_figure(t, "chk").exists()
    t.write_text("\n".join(ok[:2]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="a reading 'a' and a reading 'b'"):
        figures.checklist_figure(t, "chk")
    bad = [*ok[:3], ok[3].replace(",lo,", ",std,"), *ok[4:]]
    t.write_text("\n".join(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="are not a reading"):
        figures.checklist_figure(t, "chk")
    # a reference mark past its scale fails loud instead of landing outside the tile
    off = [*ok[:7], ok[7].replace(",1.0,1.0,5,", ",6.0,6.0,5,")]
    t.write_text("\n".join(off) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mark 'a level' leaves its scale"):
        figures.checklist_figure(t, "chk")


def test_checklist_reading_prints_values_ranges_and_spreads():
    row = lambda part, v, lab, suffix="": {  # noqa: E731
        "part": part,
        "value": v,
        "value_label": lab,
        "scale": "5",
        "suffix": suffix,
    }
    text, span, tick = figures.checklist_reading({"value": row("value", "4.62", "4.62", "×")})
    assert text == "4.62×" and span == pytest.approx((0.0, 0.924)) and tick is None
    text, span, tick = figures.checklist_reading(
        {"lo": row("lo", "0.57", "0.57"), "hi": row("hi", "4.99", "4.99")}
    )
    assert text == "0.57 to 4.99" and span == pytest.approx((0.114, 0.998))
    text, span, tick = figures.checklist_reading(
        {"mean": row("mean", "3.41", "3.41"), "std": row("std", "1.39", "1.39")}
    )
    assert text == "3.41 ± 1.39" and tick == pytest.approx(0.682)
    assert span == pytest.approx((0.404, 0.96))
