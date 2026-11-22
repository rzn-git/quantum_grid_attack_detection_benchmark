"""R4-03 runner: end-state labels, scored-call mapping, per-class cells and the branch, on tiny
fixtures (no model fit, no attack)."""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")


@pytest.fixture(scope="module")
def m():
    """The runner (and run_evasion_per_class, which it imports) read --threads from argv at
    import and set the thread variables; both are restored after the module's tests."""
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        mp.setattr(sys, "argv", ["pytest", "--threads", "1"])
        for v in THREAD_VARS:
            if v in os.environ:
                mp.setenv(v, os.environ[v])
            else:
                mp.delenv(v, raising=False)
        yield importlib.import_module("experiments.run_length_scale_per_class")


def _onehot(pred):
    pred = np.asarray(pred)
    return np.column_stack([1 - pred, pred]).astype(float)


def test_end_state_names_partial_at_floor(m):
    assert m.end_state(0.0, 0.0) == "disabled"  # P8a tuned gamma at 2 sigma
    assert m.end_state(0.7, -0.415) == "inverted"
    assert m.end_state(0.596, 0.039) == "at_floor"  # P9a bandwidth 3.0
    assert m.end_state(0.164, -0.1994) == "partial"  # P9a tuned bandwidth: just above -0.2
    assert m.end_state(0.158, -0.173) == "partial"  # P8a gamma x0.03


def test_split_scored_calls_maps_floor_clean_and_seeds_in_order(m):
    calls = list(range(2 + 2 * 3))
    out = m.split_scored_calls(calls, ["0.0", "0.5", "2.0"], 3)
    assert out == {"floor": 0, "0.0": [1], "0.5": [2, 3, 4], "2.0": [5, 6, 7]}
    with pytest.raises(ValueError, match="scored calls"):
        m.split_scored_calls(calls[:-1], ["0.0", "0.5", "2.0"], 3)


def test_read_cells_evasion_and_false_alarm_rise(m):
    y = np.array([1, 1, 1, 1, 0, 0])
    clean = _onehot([1, 1, 1, 0, 0, 0])  # attack recall 0.75, false alarm 0
    s0 = _onehot([0, 0, 1, 0, 1, 0])  # recall 0.25, false alarm 0.5
    s1 = _onehot([0, 0, 0, 0, 1, 1])  # recall 0.0, false alarm 1.0
    const = _onehot([1] * 6)
    f1 = {
        k: m.per_class(y, p.argmax(1))["macro_f1"] for k, p in [("c", clean), ("0", s0), ("1", s1)]
    }
    split = {"floor": (y, const), "0.0": [(y, clean)], "2.0": [(y, s0), (y, s1)]}
    curve = {"0.0": {"macro_f1": f1["c"]}, "2.0": {"macro_f1": float(np.mean([f1["0"], f1["1"]]))}}
    floor = m.per_class(y, const.argmax(1))["macro_f1"]
    cells = m.read_cells(split, curve, floor, eps_keys=("2.0",))
    assert cells["2.0"]["evasion"]["per_seed"] == pytest.approx([0.5, 0.75])
    assert cells["2.0"]["evasion"]["mean"] == pytest.approx(0.625)
    assert cells["2.0"]["false_alarm_rise"]["mean"] == pytest.approx(0.75)
    # a mis-mapped call no longer matches the function's own macro-F1 and fails loud
    bad = {**split, "2.0": [(y, clean), (y, s1)]}
    with pytest.raises(AssertionError, match="wrong cell"):
        m.read_cells(bad, curve, floor, eps_keys=("2.0",))


def _cell(value, label, evasion):
    return {"value": value, "end_state": label, "cells": {"2.0": {"evasion": {"mean": evasion}}}}


def test_axis_statistic_means_each_end_and_reports_endpoints(m):
    cells = [
        _cell(1.0, "disabled", -0.2),
        _cell(0.3, "disabled", -0.1),
        _cell(0.05, "partial", 0.3),
        _cell(0.01, "inverted", 0.5),
        _cell(0.001, "inverted", 0.6),
    ]
    st = m.axis_statistic(cells)
    assert st["difference"] == pytest.approx(0.55 - (-0.15))
    assert st["excluded_partial_values"] == [0.05]
    assert st["sensitivity_endpoints"] == {
        "inverted_value": 0.001,
        "floor_value": 1.0,
        "difference": pytest.approx(0.8),
    }
    with pytest.raises(ValueError, match="interleave"):
        m.axis_statistic(cells + [_cell(2.0, "inverted", 0.5)])
    with pytest.raises(ValueError, match="lacks an end"):
        m.axis_statistic([c for c in cells if c["end_state"] != "disabled"])


def test_branch_thresholds(m):
    assert m.branch({"rbf": 0.30, "fidelity": 0.5}) == "HYPOTHESIS"
    assert m.branch({"rbf": 0.29, "fidelity": -0.4}) == "KILL"
    assert m.branch({"rbf": 0.6, "fidelity": 0.1}) == "SPLIT"
