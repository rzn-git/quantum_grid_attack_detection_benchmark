"""R4-19 runner: run segmentation, duplicate counts, the cross/within instrument and the verdict
lines, on tiny synthetic fixtures."""

import importlib.util

import numpy as np
import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

_spec = importlib.util.spec_from_file_location(
    "run_file_event_identity", REPO_ROOT / "experiments" / "run_file_event_identity.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_runs_segments_contiguous_labels():
    assert _mod.runs(np.array([5, 5, 3, 3, 3, 5])) == [(5, 0, 2), (3, 2, 5), (5, 5, 6)]


def test_coverage_flags_split_scenario_and_modal_order():
    labels = [np.array([2, 2, 1, 1]), np.array([2, 1, 1]), np.array([2, 1, 2])]
    per_file, order = _mod.coverage(["a", "b", "c"], labels)
    assert per_file["a"]["scenarios_in_more_than_one_run"] == []
    assert per_file["c"]["scenarios_in_more_than_one_run"] == [2]
    assert order["n_files_with_modal_sequence"] == 2


def test_duplicates_counts_within_and_cross_file():
    a = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, np.nan]])
    b = np.array([[3.0, np.nan], [9.0, 9.0]])
    d = _mod.duplicates([a, b])
    assert d["within_file_duplicate_rows"] == 1
    assert d["distinct_vectors_in_more_than_one_file"] == 1  # the NaN row is shared
    assert d["cross_file_share_of_rows"] == pytest.approx(1 / 5)


def _walks(n_files: int, n: int, offset: float, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One scenario; each file a smooth random walk started `offset` apart."""
    Z = np.vstack(
        [f * offset + np.cumsum(0.01 * rng.standard_normal((n, 3)), axis=0) for f in range(n_files)]
    )
    return Z, np.zeros(len(Z), dtype=int), np.repeat(np.arange(n_files), n)


def test_block_stats_separates_distinct_and_shared_recordings():
    rng = np.random.default_rng(0)
    Z, scen, g = _walks(3, 200, offset=5.0, rng=rng)
    distinct = _mod.block_stats(Z, scen, g)
    assert distinct["R_cross_over_within"] > 10
    assert distinct["same_group_nn_fraction"] > 0.99
    # interleave one walk into two groups: a shared recording by construction
    pos = np.arange(len(Z))
    shared = _mod.block_stats(Z, scen, g * 2 + pos % 2)
    assert shared["R_cross_over_within"] <= 1.2
    assert shared["same_group_nn_fraction"] < 0.5


def _files(n_scen, split=False):
    f = {
        "n_scenarios": n_scen,
        "scenarios_in_more_than_one_run": [3] if split else [],
    }
    return {"a": f, "b": dict(f)}


@pytest.mark.parametrize(
    ("per_file", "dup", "R", "same", "expected"),
    [
        (_files(37), 0.0, 9.8, 0.95, "DISTINCT SESSIONS"),
        (_files(37), 0.0, 9.8, 0.857, "INCONCLUSIVE"),  # the probe's reading
        (_files(37, split=True), 0.0, 9.8, 0.95, "INCONCLUSIVE"),
        (_files(37), 0.006, 9.8, 0.95, "SHARED RECORDING"),
        (_files(37), 0.0, 1.1, 0.5, "SHARED RECORDING"),
        (_files(37), 0.0, 3.0, 0.15, "SHARED RECORDING"),
    ],
)
def test_verdict_lines(per_file, dup, R, same, expected):
    v = _mod.verdict(per_file, dup, R, same)
    assert v["session_verdict"] == expected
    assert v["file_is_one_event"] is False


def test_verdict_file_is_one_event_when_every_file_has_one_scenario():
    assert _mod.verdict(_files(1), 0.0, 9.8, 0.95)["file_is_one_event"] is True
