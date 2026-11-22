"""The further-controls table: key syntax, number formats, loud failures, and the shipped rows."""

import json
import sys

import pytest
import yaml

from qgridbench.utils.run_tracking import REPO_ROOT

if str(REPO_ROOT) not in sys.path:  # experiments/ is not installed
    sys.path.insert(0, str(REPO_ROOT))
from experiments import build_further_controls as fc  # noqa: E402


def _run(tmp_path, payload):
    run = tmp_path / "R"
    run.mkdir(exist_ok=True)
    (run / "results.json").write_text(json.dumps(payload))
    return tmp_path


def test_keys_plain_difference_and_aggregates(tmp_path):
    runs = _run(
        tmp_path,
        {
            "a": {"x": 0.9, "y": 0.1},
            "shares": {"rf": {"c": 1.2}, "xgb": {"c": 1.0}, "lgbm": {"c": 0.95}},
            "sweep": {"a": {"val": 0.576, "test": 0.559}, "b": {"val": 0.572, "test": 0.571}},
            "cells": [
                {"axis": "rbf", "value": 1.0, "r": 0.85, "source": "qsvm"},
                {"axis": "fidelity", "value": 0.793, "r": 0.05, "source": "random"},
            ],
        },
    )
    spec = {"run": "R", "file": "results.json"}
    assert fc.resolve_value({**spec, "key": "a/x"}, runs)[0] == 0.9
    assert fc.resolve_value({**spec, "key": "a/x - a/y"}, runs)[0] == pytest.approx(0.8)
    assert fc.resolve_value({**spec, "key": "shares/*/c", "agg": "min"}, runs)[0] == 0.95
    assert fc.resolve_value({**spec, "key": "shares/*/c", "agg": "max"}, runs)[0] == 1.2
    picked = {**spec, "key": "cells/*/r", "agg": "max", "where": {"axis": "rbf", "value": 1}}
    assert fc.resolve_value(picked, runs)[0] == 0.85
    kept = {**spec, "key": "cells/*/r", "agg": "min", "where_not": {"source": "random"}}
    assert fc.resolve_value(kept, runs)[0] == 0.85
    one = {**spec, "key": "cells/*/r", "agg": "one", "where": {"axis": "fidelity"}}
    assert fc.resolve_value(one, runs)[0] == 0.05
    named = {**spec, "key": "shares/*/c", "agg": "max", "keys": ["xgb", "lgbm"]}
    assert fc.resolve_value(named, runs)[0] == 1.0  # rf, the largest, is not named
    # the entry validation picks, read on test: not the best test score
    by_val = {**spec, "key": "sweep/*/test", "pick_by": "val"}
    assert fc.resolve_value(by_val, runs)[0] == 0.559


def test_keys_fail_loud(tmp_path):
    cells = [{"axis": "rbf", "r": 1.0}, {"axis": "rbf", "r": 0.9}]
    runs = _run(tmp_path, {"a": {"x": 0.9}, "cells": cells, "d": {"p": {"v": 1}, "q": {"v": 1}}})
    spec = {"run": "R", "file": "results.json"}
    with pytest.raises(ValueError, match="selects 2 entries"):
        fc.resolve_value({**spec, "key": "cells/*/r", "agg": "one"}, runs)
    with pytest.raises(KeyError, match="no entries"):
        fc.resolve_value({**spec, "key": "d/*/v", "agg": "max", "keys": ["p", "z"]}, runs)
    with pytest.raises(ValueError, match="`keys` needs a dict"):
        fc.resolve_value({**spec, "key": "cells/*/r", "agg": "max", "keys": ["p"]}, runs)
    with pytest.raises(ValueError, match="ties at its maximum"):
        fc.resolve_value({**spec, "key": "d/*/v", "pick_by": "v"}, runs)
    with pytest.raises(ValueError, match="pick_by or agg"):
        fc.resolve_value({**spec, "key": "d/*/v", "pick_by": "v", "agg": "max"}, runs)
    # a selection score or a difference operand that is not a number must not be coerced
    odd = _run(tmp_path, {"s": {"p": {"ok": True, "v": 1}, "q": {"ok": False, "v": 2}}, "t": "3"})
    with pytest.raises(ValueError, match="not a finite number"):
        fc.resolve_value({**spec, "key": "s/*/v", "pick_by": "ok"}, odd)
    with pytest.raises(ValueError, match="not a finite number"):
        fc.resolve_value({**spec, "key": "s/p/v - t"}, odd)
    with pytest.raises(ValueError, match="not a finite number"):
        fc.resolve_value({**spec, "key": "s/p/v - s/p/ok"}, odd)
    runs = _run(tmp_path, {"a": {"x": 0.9}, "cells": cells, "d": {"p": {"v": 1}, "q": {"v": 1}}})
    with pytest.raises(KeyError, match="no key 'z'"):
        fc.resolve_value({**spec, "key": "a/z"}, runs)
    with pytest.raises(FileNotFoundError):
        fc.resolve_value({"run": "missing", "file": "results.json", "key": "a/x"}, runs)
    with pytest.raises(ValueError, match="needs agg"):
        fc.resolve_value({**spec, "key": "cells/*/r"}, runs)
    with pytest.raises(ValueError, match="need a '\\*' step"):
        fc.resolve_value({**spec, "key": "a/x", "agg": "min"}, runs)
    with pytest.raises(KeyError, match="no entry"):
        fc.resolve_value(
            {**spec, "key": "cells/*/r", "agg": "min", "where": {"axis": "fidelity"}}, runs
        )


def test_numbers_print_in_ieee_form():
    assert fc.tex_number(78368, "count") == "78\\,368"
    assert fc.tex_number(0.00376, ".1%") == "0.4\\%"
    assert fc.tex_number(0.0163, "+.3f") == "$+0.016$"
    assert fc.tex_number(-0.065, ".3f") == "$-0.065$"  # math mode: a true minus
    # a value that rounds to zero carries no minus sign
    assert fc.tex_number(-0.0004, ".3f") == "$0.000$"
    assert fc.tex_number(-0.0004, "+.3f") == "$+0.000$"
    assert fc.plain_number(-0.0004, ".1%") == "0.0%"
    with pytest.raises(ValueError, match="whole number"):
        fc.plain_number(3.5, "count")


def test_values_must_be_finite_numbers_of_the_stated_sign(tmp_path):
    path = tmp_path / "results.json"
    spec = {"run": "R", "key": "a/x"}
    for bad in (True, float("nan"), float("inf"), "0.5", None):
        with pytest.raises(ValueError, match="not a finite number"):
            fc.shown_value(bad, spec, path)
    assert fc.shown_value(-0.065, {**spec, "sign": "negative"}, path) == 0.065
    assert fc.shown_value(0.015, {**spec, "sign": "positive"}, path) == 0.015
    with pytest.raises(ValueError, match="not negative"):  # "loses 0.065" for a gain
        fc.shown_value(0.065, {**spec, "sign": "negative"}, path)
    with pytest.raises(ValueError, match="not positive"):
        fc.shown_value(0.0, {**spec, "sign": "positive"}, path)
    with pytest.raises(ValueError, match="sign is"):
        fc.shown_value(0.065, {**spec, "sign": "abs"}, path)


def test_fill_matches_placeholders_and_keeps_one_math_level():
    assert fc.fill("from <0> to <1>", ["$0.5$", "$0.2$"], "r") == "from $0.5$ to $0.2$"
    assert fc.fill("is $<0> \\pm <1>$", ["$+0.016$", "$0.017$"], "r") == "is $+0.016 \\pm 0.017$"
    with pytest.raises(ValueError, match="placeholders"):
        fc.fill("only <0>", ["$1$", "$2$"], "r")
    with pytest.raises(ValueError, match="placeholders"):
        fc.fill("<0> and <1>", ["$1$"], "r")


def _spec(tmp_path, rows, meaning="attack"):
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump([{"group": "G", "meaning": meaning, "rows": rows}]))
    return p


def test_build_writes_tinted_groups_and_rows(tmp_path):
    runs = _run(tmp_path, {"a": {"x": 0.513, "ok": True}})
    value = {"run": "R", "file": "results.json", "key": "a/x", "fmt": ".3f", "expect": "0.513"}
    need = {"run": "R", "file": "results.json", "key": "a/ok"}
    row = {
        "id": "E",
        "check": "C",
        "reading": "it is <0>",
        "bears_on": "S",
        "values": [value],
        "require": [need],
    }
    lines, resolved, sources = fc.build(_spec(tmp_path, [row]), runs)
    assert lines[0].startswith("\\rowcolor{qgAttackTint}[0pt][0pt]") and "\\textbf{G}" in lines[0]
    assert lines[1] == "C & it is $0.513$ & S \\\\\\addlinespace[2pt]"
    assert resolved[0]["value"] == 0.513 and len(sources) == 1


def test_build_fails_loud_on_bad_rows(tmp_path):
    runs = _run(tmp_path, {"a": {"x": 0.5, "ok": False}})
    value = {"run": "R", "file": "results.json", "key": "a/x", "fmt": ".3f", "expect": "0.500"}
    row = {"id": "E", "check": "C", "reading": "<0>", "bears_on": "S", "values": [value]}
    # every value must state the text it prints as
    bare = {**row, "values": [{k: v for k, v in value.items() if k != "expect"}]}
    with pytest.raises(KeyError, match="no `expect`"):
        fc.build(_spec(tmp_path, [bare]), runs)
    with pytest.raises(KeyError, match="mystery"):
        fc.build(_spec(tmp_path, [row], meaning="mystery"), runs)
    with pytest.raises(ValueError, match="duplicate row id"):
        fc.build(_spec(tmp_path, [row, row]), runs)
    failing = {**row, "require": [{"run": "R", "file": "results.json", "key": "a/ok"}]}
    with pytest.raises(ValueError, match="not true"):
        fc.build(_spec(tmp_path, [failing]), runs)
    # a value that does not print as the row expects: a wrong key or a changed run
    wrong = {**row, "values": [{**value, "expect": "0.513"}]}
    with pytest.raises(ValueError, match="prints '0.500', the row expects '0.513'"):
        fc.build(_spec(tmp_path, [wrong]), runs)
    boolean = {**row, "values": [{**value, "key": "a/ok"}]}
    with pytest.raises(ValueError, match="not a finite number"):
        fc.build(_spec(tmp_path, [boolean]), runs)


def test_build_prints_counts_percentages_and_magnitudes(tmp_path):
    runs = _run(tmp_path, {"n": 78368, "share": 0.9828, "loss": -0.065})
    base = {"run": "R", "file": "results.json"}
    values = [
        {**base, "key": "n", "fmt": "count", "expect": "78368"},
        {**base, "key": "share", "fmt": ".0%", "expect": "98%"},
        {**base, "key": "loss", "fmt": ".3f", "sign": "negative", "expect": "0.065"},
    ]
    row = {"id": "E", "check": "C", "reading": "<0>, <1>, <2>", "bears_on": "S", "values": values}
    lines, resolved, _ = fc.build(_spec(tmp_path, [row]), runs)
    assert lines[1].startswith("C & 78\\,368, 98\\%, $0.065$ & S")
    assert resolved[2]["value"] == -0.065  # the CSV keeps the signed value


def test_shipped_rows_are_well_formed():
    """Offline check of the curated file (the run artifacts are not in the repository)."""
    groups = yaml.safe_load(fc.SPEC.read_text(encoding="utf-8"))
    ids = [row["id"] for g in groups for row in g["rows"]]
    assert len(ids) == len(set(ids)) and len(ids) >= 20
    for g in groups:
        assert g["meaning"] in fc.LATEX_COLOR_NAMES
        for row in g["rows"]:
            used = {int(m) for m in fc.PLACEHOLDER.findall(row["reading"])}
            assert used == set(range(len(row["values"]))), row["id"]
            for v in row["values"]:
                # every number states what it must print as, and none drops a sign unchecked
                assert {"run", "file", "key", "fmt", "expect"} <= set(v), row["id"]
                assert isinstance(v["expect"], str) and "abs" not in v, row["id"]
                assert ("*" in v["key"]) == ("agg" in v or "pick_by" in v), row["id"]


def test_means_counts_and_two_spec_combinations(tmp_path):
    """The other-checks table reads a mean over seeds, a count under a limit, and a difference
    or a ratio of two specs that may sit in different runs."""
    runs = _run(
        tmp_path, {"per_seed": [{"clean": 0.6, "noisy": 0.5}, {"clean": 0.7, "noisy": 0.5}]}
    )
    other = tmp_path / "S"
    other.mkdir()
    (other / "results.json").write_text(json.dumps({"latency": 0.004, "scores": [0.41, 0.42, 0.5]}))
    spec = {"run": "R", "file": "results.json"}
    mean = {**spec, "key": "per_seed/*/clean", "agg": "mean"}
    assert fc.resolve_value(mean, runs)[0] == pytest.approx(0.65)
    cost = {**mean, "minus": {**spec, "key": "per_seed/*/noisy", "agg": "mean"}}
    assert fc.resolve_value(cost, runs)[0] == pytest.approx(0.15)
    ratio = {**mean, "over": {"run": "S", "file": "results.json", "key": "latency"}}
    assert fc.resolve_value(ratio, runs)[0] == pytest.approx(162.5)
    # both files count as sources of a combined value
    assert [p.parent.name for p in fc.spec_paths(ratio, runs)] == ["R", "S"]
    near = {"run": "S", "file": "results.json", "key": "scores/*", "agg": "count_at_most"}
    assert fc.resolve_value({**near, "limit": 0.465}, runs)[0] == 2
    with pytest.raises(ValueError, match="needs a limit"):
        fc.resolve_value(near, runs)
    assert fc.plain_number(31326.7, "count2") == "31000"
    assert fc.tex_number(75080.4, "count2") == "75\\,000"
    assert fc.plain_number(1917.9, "count2") == "1900"


def test_other_checks_table_resolves_every_value_and_puts_the_setting_second():
    run = "20261117T075906Z_adversarial_binary_q8_vqc_table1"
    if not (REPO_ROOT / "results" / "runs" / run).is_dir():
        pytest.skip("the research run archive (results/runs) is not on this machine")
    lines, resolved, sources = fc.build(fc.OTHER_SPEC, last="setting", order="last-reading")
    rows = [ln for ln in lines if not ln.startswith("\\rowcolor")]
    assert len(rows) == 13 and len(resolved) == 48
    # "the other three models lose nothing" under sensor noise is words in the spec, so
    # the run is read here: clean minus noisy is not positive for any of the three
    for model in ("logreg", "mlp", "vqc"):
        noisy = {
            "run": run,
            "file": "results.json",
            "key": f"per_seed/*/sensor_noise/{model}/10",
            "agg": "mean",
        }
        clean = {**noisy, "key": f"per_seed/*/evasion/clean/{model}", "minus": noisy}
        cost, _ = fc.resolve_value(clean)
        assert cost <= 0, model
    # the depth row gives the accuracy at each depth and says the settings are fixed
    depth = next(r for r in rows if r.startswith("Depth"))
    assert "fixed training settings" in depth
    assert all(f"${n}$" in depth for n in ("0.45", "0.47", "0.44"))
    assert rows[0].startswith("Shots & tuned ZZ QSVM") and "$0.058$" in rows[0]
    # the training-rows row gives the lead over both quantum models
    lead = next(r for r in rows if r.startswith("Training rows"))
    assert all(f"${n}$" in lead for n in ("0.125", "0.118", "0.116", "0.121"))
    # the defenses band is grey: green is kept for a control that checks an evaluation choice
    groups = [ln for ln in lines if ln.startswith("\\rowcolor")]
    assert len(groups) == 4 and "qgControlTint" not in groups[2]
    # the physical-consistency rows state three things in words; the runs are read here
    grids = {
        "h3": (
            "20261118T195101Z_fdia_consistent_attack_fdia14_binary",
            "20261119T013227Z_fdia_consistent_attack_fdia118_binary",
        ),
        "square": (
            "20261118T161408Z_square_blackbox_fdia14_binary_q8",
            "20261119T044113Z_square_blackbox_fdia118_binary_q8",
        ),
    }
    offsets, square, noise = [], [], []
    for run in grids["h3"]:
        summary = json.loads((fc.RUNS / run / "results.json").read_text())["summary"]
        # "the unperturbed rate": a consistent attack is flagged exactly as often as clean rows
        for key, cell in summary["bdd_flag_rates"].items():
            if key.startswith("C|"):
                assert cell["flag_rate"] == cell["clean_flag_rate"], (run, key)
            else:
                assert cell["flag_rate"] == 1.0, (run, key)
        offsets += [
            r["random_f1"]
            for r in summary["control5"]
            if r["attacker"] == "C" and r["eps"] == 0.1 and r["source"] in ("qsvm", "mlp")
        ]
    for run in grids["square"]:
        seeds = json.loads((fc.RUNS / run / "results.json").read_text())["per_seed"]
        models = seeds[0]["clean"]
        accurate = [m for m in models if sum(s["clean"][m] for s in seeds) / len(seeds) >= 0.98]
        assert sorted(accurate) == ["lgbm", "mlp", "qsvm", "rbf_svm", "rf", "xgb"], run
        square += [sum(s["square"]["0.1"][m] for s in seeds) / len(seeds) for m in accurate]
        noise += [sum(s["random"]["0.1"][m] for s in seeds) / len(seeds) for m in accurate]
    by_id = {}
    for r in resolved:
        by_id.setdefault(r["row"], []).append(r["value"])
    # each printed range is the range over the six accurate models on both grids
    assert (min(square), max(square)) == pytest.approx(tuple(by_id["square_grids"][:2]))
    assert (min(noise), max(noise)) == pytest.approx(tuple(by_id["square_grids"][2:]))
    assert max(square) < min(noise)  # the prose: Square leaves less than noise of that size
    assert (min(offsets), max(offsets)) == pytest.approx(tuple(by_id["consistent_whitebox"][4:]))
    with pytest.raises(ValueError, match="order is"):
        fc.build(fc.OTHER_SPEC, last="setting", order="setting-first")


def test_new_aggregates_and_formats_fail_loud(tmp_path):
    """A NaN, a string or a bool among iterated entries fails; so do an empty iteration, a limit
    without its aggregate, minus with over, and a small value under count2."""
    runs = _run(
        tmp_path, {"s": [0.41, float("nan"), 0.5], "t": ["0.4", 0.5], "u": [True, 0.5], "e": []}
    )
    spec = {"run": "R", "file": "results.json"}
    for key in ("s/*", "t/*", "u/*"):
        with pytest.raises(ValueError, match="not a finite number"):
            fc.resolve_value({**spec, "key": key, "agg": "count_at_most", "limit": 0.465}, runs)
    with pytest.raises(KeyError, match="no entry of"):
        fc.resolve_value({**spec, "key": "e/*", "agg": "mean"}, runs)
    with pytest.raises(ValueError, match="limit belongs to"):
        fc.resolve_value({**spec, "key": "s/*", "agg": "mean", "limit": 0.5}, runs)
    both = {**spec, "key": "u/1", "minus": {**spec, "key": "u/1"}, "over": {**spec, "key": "u/1"}}
    with pytest.raises(ValueError, match="minus or over, not both"):
        fc.resolve_value(both, runs)
    with pytest.raises(ValueError, match="ratios of 100 or more"):
        fc.plain_number(9.6, "count2")


def test_a_value_can_come_from_a_run_params_yaml(tmp_path):
    run = tmp_path / "R"
    run.mkdir()
    (run / "params.yaml").write_text("config:\n  defense_bar: 0.05\n")
    spec = {"run": "R", "file": "params.yaml", "key": "config/defense_bar"}
    assert fc.resolve_value(spec, tmp_path)[0] == 0.05
