"""R4-21 runner: the N3 tie-set rule, the knob count and the S2 labelling, on tiny fixtures."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture(scope="module", autouse=True)
def _mod_on_path():
    """The runner imports `experiments.*` lazily (the S2 label reuses R4-03's `end_state`), so the
    repo root is on sys.path for this module's tests and removed afterwards."""
    global _mod
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        _mod = importlib.import_module("experiments.run_n3_selection_assembly")
        yield


_mod = None

ALL_INV = {"loss": "inverted", "margin": "inverted", "mass": "inverted"}


def _pt(value, clean, s1, s2=None):
    return {"value": value, "clean_val": clean, "s1_test": s1, "s2": s2 or dict(ALL_INV)}


def test_s1_spread_inside_tie_set_holds():
    pts = [_pt(1.0, 0.60, 0.01), _pt(3.0, 0.585, 0.40), _pt(0.1, 0.50, 0.90)]
    v = _mod.knob_verdict(pts, "K1")
    assert v["tie_set"] == [1.0, 3.0]  # 0.1 is outside the 0.02 band and its 0.90 is ignored
    assert v["s1_spread"] == pytest.approx(0.39)
    assert v["holds"] and v["s1_holds"] and not v["s2_holds"]
    assert v["rider_accuracy_pick_less_secure"]
    assert v["rider_clean_cost_of_s1_best"] == pytest.approx(0.015)
    assert not v["rider_s1_best_inside_search_box"]  # 3.0 > the 2.0 bandwidth box


def test_point_outside_tie_band_does_not_count():
    pts = [_pt(1.0, 0.60, 0.01), _pt(3.0, 0.579, 0.40, {**ALL_INV, "mass": "disabled"})]
    v = _mod.knob_verdict(pts, "K1")
    assert v["tie_set"] == [1.0]
    assert not v["holds"]


def test_s2_both_ends_under_one_objective_holds():
    pts = [
        _pt(835.0, 0.59, 0.02, {**ALL_INV}),
        _pt(10.0, 0.58, 0.03, {**ALL_INV, "mass": "disabled"}),
    ]
    v = _mod.knob_verdict(pts, "K2")
    assert v["s2_both_ends_by_objective"] == {"loss": False, "margin": False, "mass": True}
    assert v["holds"] and not v["s1_holds"]


def test_at_floor_counts_as_the_floor_end_and_partial_does_not():
    at_floor = [_pt(1, 0.6, 0.0), _pt(2, 0.6, 0.0, {**ALL_INV, "loss": "at_floor"})]
    partial = [_pt(1, 0.6, 0.0), _pt(2, 0.6, 0.0, {**ALL_INV, "loss": "partial"})]
    assert _mod.knob_verdict(at_floor, "K3")["holds"]
    assert not _mod.knob_verdict(partial, "K3")["holds"]


def test_missing_reading_in_tie_set_fails_loud():
    with pytest.raises(ValueError, match="without S1"):
        _mod.knob_verdict([_pt(1, 0.6, 0.0), _pt(2, 0.59, None)], "K1")
    with pytest.raises(ValueError, match="S2"):
        _mod.knob_verdict([_pt(1, 0.6, 0.0, {"loss": "inverted"})], "K1")


@pytest.mark.parametrize(("holds", "verdict"), [((1, 1, 0), "N3 HOLDS"), ((1, 0, 0), "KILL")])
def test_n3_needs_two_of_three_knobs(holds, verdict):
    knobs = {k: {"holds": bool(h)} for k, h in zip(("K1", "K2", "K3"), holds)}
    assert _mod.n3_verdict(knobs)["verdict"] == verdict


def test_s2_label_reads_seed_means():
    cells = [
        {"mass_ratio": 0.02, "f1_minus_floor": 0.01},
        {"mass_ratio": 0.03, "f1_minus_floor": 0.0},
    ]
    assert _mod.s2_label(cells)["label"] == "disabled"
    inv = [{"mass_ratio": 0.5, "f1_minus_floor": -0.3}]
    assert _mod.s2_label(inv)["label"] == "inverted"
    floor = [{"mass_ratio": 0.6, "f1_minus_floor": 0.03}]
    assert _mod.s2_label(floor)["label"] == "at_floor"
