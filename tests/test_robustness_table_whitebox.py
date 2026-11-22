"""The white-box column of the robustness table is read from the worst-radius run and fails loud
(src/qgridbench/eval/tables.py, whitebox_cells; review of ruling (h), finding m10)."""

import json

import pytest

from qgridbench.eval.tables import (
    logreg_whitebox_cell,
    lowest_attacked,
    score_attack_cells,
    whitebox_cells,
)


def _run(tmp_path, budget=0.5, mlp_at_budget=0.62):
    models = {
        "qsvm": {"worst": {"retention": 0.02, "radius": 0.3}, "at_budget": {"retention": 0.06}},
        "rbf_svm": {"worst": {"retention": 0.24, "radius": 0.1}, "at_budget": {"retention": 0.68}},
        "mlp": {
            "worst": {"retention": mlp_at_budget, "radius": 0.5},
            "at_budget": {"retention": mlp_at_budget},
        },
        "vqc": {"worst": {"retention": 0.06, "radius": 0.5}, "at_budget": {"retention": 0.06}},
    }
    (tmp_path / "results.json").write_text(
        json.dumps({"budget": budget, "sources": {"msu": {"models": models}}})
    )
    return tmp_path


PER_FAMILY = {"mlp": {"retention": 0.62}, "vqc": {"retention": 0.06}}


def test_cells_carry_the_radius_read_and_the_reading_at_the_budget(tmp_path):
    wb = whitebox_cells(_run(tmp_path), 0.5, PER_FAMILY, tmp_path)
    assert wb["qsvm"] == {
        "retention": 0.02,
        "radius": 0.3,
        "at_budget": 0.06,
        "single_radius": False,
    }
    assert wb["rbf_svm"]["radius"] == 0.1 and set(wb) == {"qsvm", "rbf_svm", "mlp", "vqc"}


def test_a_run_read_inside_another_budget_fails(tmp_path):
    with pytest.raises(ValueError, match="read inside 0.3 sigma, the table is at 0.5"):
        whitebox_cells(_run(tmp_path, budget=0.3), 0.5, PER_FAMILY, tmp_path)


def test_a_run_built_from_another_adversarial_run_fails(tmp_path):
    with pytest.raises(ValueError, match="built from another adversarial run"):
        whitebox_cells(_run(tmp_path, mlp_at_budget=0.50), 0.5, PER_FAMILY, tmp_path)


def _logreg_run(tmp_path, budget=0.5, clean=0.5141, model="logreg"):
    out = {
        "model": model,
        "budget": budget,
        "clean_f1": clean,
        "worst": {"radius": 0.5, "retention": 0.21},
        "at_budget": {"radius": 0.5, "retention": 0.21},
    }
    (tmp_path / "results.json").write_text(json.dumps(out))
    return tmp_path


def test_logistic_regression_cell_is_read_from_its_own_run(tmp_path):
    cell = logreg_whitebox_cell(_logreg_run(tmp_path), 0.5, 0.51409)
    assert cell == {"retention": 0.21, "radius": 0.5, "at_budget": 0.21, "single_radius": False}


@pytest.mark.parametrize("kwargs", [{"budget": 0.3}, {"clean": 0.52}])
def test_a_logistic_regression_run_on_another_budget_or_fit_fails(tmp_path, kwargs):
    with pytest.raises(ValueError, match="the run attacked another fit"):
        logreg_whitebox_cell(_logreg_run(tmp_path, **kwargs), 0.5, 0.51409)


def test_a_run_of_another_model_is_not_read_as_logistic_regression(tmp_path):
    with pytest.raises(ValueError, match="not of logistic regression"):
        logreg_whitebox_cell(_logreg_run(tmp_path, model="rbf_svm"), 0.5, 0.51409)


def test_the_csv_carries_the_floor_gap_unrounded_for_the_figure(tmp_path, monkeypatch):
    """The robustness figure prints the floor gap at three decimals from this CSV, as the table
    does from the raw value: a CSV rounded to four decimals first prints 0.086 where the table
    prints 0.085 (fix check of 2026-11-21, finding F12)."""
    import csv

    import numpy as np

    from qgridbench.data import preprocess
    from qgridbench.eval import tables

    per_family = {
        "mlp": {"clean_mean": 0.55, "adv_mean": 0.341, "retention": 0.62},
        "vqc": {"clean_mean": 0.49, "adv_mean": 0.0294, "retention": 0.06},
    }
    stats = {"epsilon": 0.5, "per_family": per_family, "variant": "binary", "n_eval": 20}
    y = np.tile([0, 1], 50)
    split = {"train": np.arange(60), "test": np.arange(60, 100)}
    monkeypatch.setattr(tables, "TABLE_DIR", tmp_path / "tables")
    monkeypatch.setattr(tables, "robustness_stats", lambda run: stats)
    monkeypatch.setattr(preprocess, "load_processed", lambda variant: (None, y, split))
    for name, key in (("random", "random_control"), ("blackbox", "blackbox")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "results.json").write_text(json.dumps({"per_seed": [{key: {}}]}))
    (tmp_path / "worst").mkdir()

    tex = tables.robustness_summary_table(
        tmp_path,
        tmp_path / "random",
        tmp_path / "blackbox",
        "robustness",
        worst_radius_run_dir=_run(tmp_path / "worst"),
    )

    with (tmp_path / "tables" / "robustness.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["model"] for r in rows] == ["mlp", "vqc"]
    body = tex.read_text(encoding="utf-8")
    for r in rows:
        gap = r["adv_f1_minus_floor"]
        assert len(gap.partition(".")[2]) == 6
        assert f"& {tables._num(float(gap), '+.3f')} &" in body


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        # the Square search is the lowest, at the budget
        ({"P": (0.4991, 0.5), "W": (0.0128, 0.3), "S": (0.0066, 0.5)}, (0.0066, "S", 0.5)),
        # the model's own attack is, at its worst radius
        ({"P": (0.4148, 0.5), "W": (0.1429, 0.1), "S": (0.1511, 0.1)}, (0.1429, "W", 0.1)),
        # a tree: no attack of its own; evolution strategies at 0.3 sigma beats the Square search
        (
            {"P": (0.3673, 0.5), "W": None, "S": (0.2161, 0.1), "E": (0.1948, 0.3)},
            (0.1948, "E", 0.3),
        ),
        # a tie goes to the model's own attack, then the Square search, then evolution strategies
        ({"P": (0.4310, 0.5), "W": (0.1101, 0.5), "S": (0.1101, 0.5)}, (0.1101, "W", 0.5)),
        ({"P": (0.3, 0.5), "S": (0.1000, 0.3), "E": (0.1002, 0.3)}, (0.1000, "S", 0.3)),
        # the value and radius are the named attack's own, never a mix of two attacks
        ({"P": (0.3, 0.5), "S": (0.1004, 0.1), "E": (0.1000, 0.3)}, (0.1004, "S", 0.1)),
        # the shared PGD is the model's own attack
        ({"P": (0.0287, 0.5), "W": (0.0287, 0.5)}, (0.0287, "W", 0.5)),
        # nothing but the shared attack was run
        ({"P": (0.3000, 0.5), "W": None, "S": None, "E": None}, (0.3000, "P", 0.5)),
    ],
)
def test_the_lowest_attacked_score_names_its_attack_and_radius(candidates, expected):
    assert lowest_attacked(candidates) == expected


def test_no_attack_at_all_is_an_error():
    with pytest.raises(ValueError, match="no attack"):
        lowest_attacked({"P": None})


def _score_run(tmp_path, name, key, clean, scores):
    """A score-based run: `clean` per seed, `scores` {radius: per-seed list}, for one model."""
    run = tmp_path / name
    run.mkdir()
    per_seed = [
        {
            "seed": s,
            "clean": {"xgb": clean[s]},
            key: {r: {"xgb": v[s]} for r, v in scores.items()},
        }
        for s in range(len(clean))
    ]
    (run / "results.json").write_text(json.dumps({"per_seed": per_seed}))
    return run


SHARED = {"xgb": [0.61, 0.62, 0.63], "mlp": [0.47, 0.45, 0.49]}
SCORES = {
    "0.1": [0.09, 0.10, 0.08],
    "0.3": [0.08, 0.09, 0.07],
    "0.5": [0.12, 0.13, 0.11],
    "1.0": [0.0, 0.0, 0.0],
}


def test_a_score_attack_is_read_at_its_worst_radius_inside_the_budget(tmp_path):
    run = _score_run(tmp_path, "sq", "square", [0.615, 0.63, 0.62], SCORES)
    cells = score_attack_cells([{"attack": "S", "run": run, "models": ("xgb",)}], 0.5, SHARED)
    f1, radius = cells["xgb"]["S"]
    # 0.3 sigma is the lowest mean of the radii at or below the budget; 1.0 sigma is outside it
    assert radius == 0.3 and f1 == pytest.approx(0.08)


def test_a_score_run_on_another_fit_is_refused(tmp_path):
    # XGBoost's fit moves with the thread count (tolerance 0.02 per seed); 0.05 is another model
    run = _score_run(tmp_path, "sq", "square", [0.66, 0.62, 0.63], SCORES)
    with pytest.raises(ValueError, match="another fit"):
        score_attack_cells([{"attack": "S", "run": run, "models": ("xgb",)}], 0.5, SHARED)


def test_every_other_model_must_match_the_shared_run_exactly(tmp_path):
    run = tmp_path / "es"
    run.mkdir()
    per_seed = [
        {"seed": s, "clean": {"mlp": c}, "nes": {"0.5": {"mlp": 0.28}}}
        for s, c in enumerate([0.47, 0.451, 0.49])
    ]
    (run / "results.json").write_text(json.dumps({"per_seed": per_seed}))
    with pytest.raises(ValueError, match="another fit"):
        score_attack_cells([{"attack": "E", "run": run, "models": ("mlp",)}], 0.5, SHARED)


def test_one_attack_from_two_runs_is_refused(tmp_path):
    a = _score_run(tmp_path, "a", "square", [0.61, 0.62, 0.63], SCORES)
    b = _score_run(tmp_path, "b", "square", [0.61, 0.62, 0.63], SCORES)
    spec = [{"attack": "S", "run": r, "models": ("xgb",)} for r in (a, b)]
    with pytest.raises(ValueError, match="two runs"):
        score_attack_cells(spec, 0.5, SHARED)


def test_the_table_prints_the_lowest_score_with_its_attack(tmp_path, monkeypatch):
    import csv

    import numpy as np

    from qgridbench.data import preprocess
    from qgridbench.eval import tables

    per_family = {
        "mlp": {"clean_mean": 0.55, "adv_mean": 0.341, "retention": 0.62},
        "vqc": {"clean_mean": 0.49, "adv_mean": 0.0294, "retention": 0.06},
    }
    stats = {"epsilon": 0.5, "per_family": per_family, "variant": "binary", "n_eval": 20}
    y = np.tile([0, 1], 50)
    split = {"train": np.arange(60), "test": np.arange(60, 100)}
    monkeypatch.setattr(tables, "TABLE_DIR", tmp_path / "tables")
    monkeypatch.setattr(tables, "robustness_stats", lambda run: stats)
    monkeypatch.setattr(preprocess, "load_processed", lambda variant: (None, y, split))
    for name, key in (("random", "random_control"), ("blackbox", "blackbox")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "results.json").write_text(json.dumps({"per_seed": [{key: {}}]}))
    (tmp_path / "worst").mkdir()
    # the shared run's clean scores per seed, as the table reads them for the fit check
    shared = [{"evasion": {"clean": {"mlp": 0.55, "vqc": 0.49}}} for _ in range(3)]
    (tmp_path / "results.json").write_text(json.dumps({"per_seed": shared}))
    run = tmp_path / "sq"
    run.mkdir()
    per_seed = [
        {
            "seed": s,
            "clean": {"mlp": 0.55},
            "square": {"0.1": {"mlp": 0.40}, "0.3": {"mlp": 0.25}, "0.5": {"mlp": 0.30}},
        }
        for s in range(3)
    ]
    (run / "results.json").write_text(json.dumps({"per_seed": per_seed}))

    tex = tables.robustness_summary_table(
        tmp_path,
        tmp_path / "random",
        tmp_path / "blackbox",
        "robustness",
        worst_radius_run_dir=_run(tmp_path / "worst"),
        score_attack_runs=[{"attack": "S", "run": run, "models": ("mlp",)}],
    )
    with (tmp_path / "tables" / "robustness.csv").open(encoding="utf-8", newline="") as fh:
        rows = {r["model"]: r for r in csv.DictReader(fh)}
    mlp = rows["mlp"]
    assert (mlp["square_f1"], mlp["square_radius_sigma"]) == ("0.250000", "0.3")
    assert (mlp["lowest_f1"], mlp["lowest_attack"], mlp["lowest_radius_sigma"]) == (
        "0.250000",
        "S",
        "0.3",
    )
    # a model no score-based run names takes its lowest over its other attacks
    assert rows["vqc"]["square_f1"] == "" and rows["vqc"]["lowest_attack"] == "W"
    assert "0.250 (S)" in tex.read_text(encoding="utf-8")
