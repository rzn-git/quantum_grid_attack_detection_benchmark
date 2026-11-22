"""N-DECAY instrument, R4-10 additions: per-class curve, keyed fit, anchor, row classes."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

# the runner imports helpers from experiments/, which is not an installed package
sys.path.insert(0, str(REPO_ROOT))
_before = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "LOKY_MAX_CPU_COUNT")}
_spec = importlib.util.spec_from_file_location(
    "run_leak_decay_curve", REPO_ROOT / "experiments" / "run_leak_decay_curve.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)
# read straight after the import: other test files import runners that pin threads at import,
# so the environment at test time says nothing about this import
_after = {v: os.environ.get(v) for v in _before}


def test_import_leaves_thread_caps_alone():
    """Only `python -m` pins threads; an importer (the gas runner) keeps its own."""
    assert _after == _before


def _curve(a, b, tau, key, gaps=range(1, 21), n=300):
    return [{"gap": g, "n": n, key: a + b * np.exp(-(g - 1) / tau)} for g in gaps]


def test_fit_decay_keyed_recovers_tau_and_matches_default_key():
    fit = mod.fit_decay(_curve(0.6, 0.27, 2.5, "accuracy"), key="accuracy")
    assert fit["tau_released_rows"] == pytest.approx(2.5, rel=1e-6)
    assert fit["asymptote_fit"] == pytest.approx(0.6, abs=1e-6)
    assert fit == mod.fit_decay(_curve(0.6, 0.27, 2.5, "macro_f1"))


def test_per_gap_class_curve_is_class_accuracy_with_population_rule():
    gap = np.array([1, 1, 1, 2, 2, 2, 0], dtype=float)
    y = np.array([1, 1, 1, 1, 1, 1, 1])
    pred = np.array([1, 1, 0, 1, 0, 0, 1])
    test = gap > 0
    assert mod.per_gap_class_curve(pred, y, gap, test) == []  # 3 rows < MIN_ROWS
    pts = mod.per_gap_class_curve(pred, y, gap, test, min_rows=1)
    assert pts == [
        {"gap": 1, "n": 3, "accuracy": pytest.approx(2 / 3)},
        {"gap": 2, "n": 3, "accuracy": pytest.approx(1 / 3)},
    ]
    assert mod.per_gap_class_curve(pred, y, gap, test, max_gap=1, min_rows=1)[-1]["gap"] == 1


def _fake_anchor(tmp_path, monkeypatch, tau_lgbm, tau_xgb):
    root = tmp_path / "results" / "runs" / mod.ANCHOR_RUN
    root.mkdir(parents=True)
    curves = {"A_lgbm_matched": [{"gap": 1, "macro_f1": 0.8}], "A_xgb_matched": []}
    fits = {"A_lgbm": {"tau_released_rows": tau_lgbm}, "A_xgb": {"tau_released_rows": tau_xgb}}
    (root / "results.json").write_text(json.dumps({"curves": curves, "fits": fits}))
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    return fits, curves


def test_anchor_exact_passes_and_xgb_off_thread_count_is_reported_not_raised(tmp_path, monkeypatch):
    fits, curves = _fake_anchor(tmp_path, monkeypatch, 2.524354121017558, 2.29)
    monkeypatch.setattr(mod, "THREADS", 4)
    run_fits = {"A_lgbm": dict(fits["A_lgbm"]), "A_xgb": {"tau_released_rows": 2.55}}
    out = mod.anchor_check(run_fits, curves, ["lgbm", "xgb"], {})
    assert out["lgbm"]["exact"] and out["lgbm"]["blocking"]
    assert not out["xgb"]["exact"] and not out["xgb"]["blocking"]
    assert out["xgb"]["threads"] == 4


def test_anchor_fails_loud_on_any_lgbm_difference_and_on_xgb_at_archive_threads(
    tmp_path, monkeypatch
):
    fits, curves = _fake_anchor(tmp_path, monkeypatch, 2.524354121017558, 2.29)
    monkeypatch.setattr(mod, "THREADS", 4)
    with pytest.raises(SystemExit, match="ANCHOR FAILED \\(lgbm\\)"):
        mod.anchor_check({"A_lgbm": {"tau_released_rows": 2.524354121017559}}, curves, ["lgbm"], {})
    # XGBoost at the archive's count through --family-threads blocks, whatever --threads is
    with pytest.raises(SystemExit, match="ANCHOR FAILED \\(xgb\\)"):
        mod.anchor_check(
            {"A_xgb": {"tau_released_rows": 2.55}}, curves, ["xgb"], {"xgb": mod.ANCHOR_THREADS}
        )
    out = mod.anchor_check(
        {"A_xgb": {"tau_released_rows": 2.29}}, curves, ["xgb"], {"xgb": mod.ANCHOR_THREADS}
    )
    assert out["xgb"]["exact"] and out["xgb"]["blocking"] and out["xgb"]["threads"] == 24


def test_row_classes_checks_against_variant_labels(monkeypatch):
    monkeypatch.setattr(
        mod, "scenario_class_map", lambda: {"41": "NoEvents", "1": "Natural", "7": "Attack"}
    )
    scen = np.array(["41", "1", "7", "7"])
    assert list(mod.row_classes("binary", np.array([0, 0, 1, 1]), scen)) == [
        "NoEvents",
        "Natural",
        "Attack",
        "Attack",
    ]
    assert list(mod.row_classes("triple", np.array([0, 1, 2, 2]), scen))[0] == "NoEvents"
    with pytest.raises(SystemExit, match="disagree"):
        mod.row_classes("binary", np.array([0, 1, 1, 1]), scen)


def test_run_fits_resumes_finished_units_from_cache(tmp_path, monkeypatch):
    calls = []

    def fake_fit(unit):
        calls.append(unit[0])
        return unit[0], np.full(4, len(calls)), 1.5, 0.1

    monkeypatch.setattr(mod, "_fit_unit", fake_fit)
    monkeypatch.setattr(mod, "_init_worker", lambda *a: None)
    y = np.array([0, 1, 0, 1])
    params = {"lgbm": {"n_estimators": 3}}
    mask = np.array([True, True, False, False])
    units = [(("A", "lgbm"), "lgbm", mask), (("B", "lgbm"), "lgbm", ~mask)]
    first = mod.run_fits(units, None, y, params, 1, tmp_path, {})
    assert calls == [("A", "lgbm"), ("B", "lgbm")]
    again = mod.run_fits(units, None, y, params, 1, tmp_path, {})
    assert calls == [("A", "lgbm"), ("B", "lgbm")]  # nothing refitted
    for k in first:
        assert np.array_equal(first[k][0], again[k][0]) and again[k][1:] == (1.5, 0.1)
    other = {"lgbm": {"n_estimators": 4}}  # new parameters name a new file: refit
    mod.run_fits(units[:1], None, y, other, 1, tmp_path, {})
    assert calls[-1] == ("A", "lgbm") and len(calls) == 3


def test_unit_cache_names_the_thread_count():
    y = np.array([0, 1])
    u = (("A", "xgb"), "xgb", np.array([True, False]))
    params = {"xgb": {"n_estimators": 3}}
    assert mod.unit_cache_path(Path("c"), u, y, params, 24) != mod.unit_cache_path(
        Path("c"), u, y, params, 4
    )
