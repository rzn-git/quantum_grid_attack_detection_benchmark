"""Per-family thread groups: parsing, the environment guard, and a spawned off-count group."""

import os
import sys

import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

# experiments/ is not installed; imported by name so a spawned child can unpickle its functions
sys.path.insert(0, str(REPO_ROOT))
from experiments import family_threads as ft  # noqa: E402


def test_parse_family_threads():
    assert ft.parse_family_threads(["xgb=24", "rf=2"]) == {"xgb": 24, "rf": 2}
    assert ft.parse_family_threads([]) == {}
    for bad in ("xgb", "xgb=0", "=4", "xgb=four"):
        with pytest.raises(SystemExit, match="FAMILY=N"):
            ft.parse_family_threads([bad])


def test_thread_env_sets_and_restores():
    before = {v: os.environ.get(v) for v in ft.THREAD_VARS}
    with ft.thread_env(7):
        ft.check_thread_env(7)
        with pytest.raises(AssertionError, match="expected 3"):
            ft.check_thread_env(3)
    assert {v: os.environ.get(v) for v in ft.THREAD_VARS} == before


def test_run_grouped_runs_default_in_process_and_off_count_in_a_spawned_process():
    """Each unit is a thread variable's name and `fn` is os.getenv, so a unit's result is the
    count its process actually ran under."""
    seen = {}
    units = ["LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS"]  # default-count unit listed first
    counts = {"OMP_NUM_THREADS": 3, "LOKY_MAX_CPU_COUNT": 2}
    with ft.thread_env(2):
        ft.run_grouped(
            units,
            counts.get,
            os.getenv,
            ft.check_thread_env,
            (),
            1,
            2,
            lambda u, r: seen.__setitem__(u, r),
        )
    assert seen == {"OMP_NUM_THREADS": "3", "LOKY_MAX_CPU_COUNT": "2"}
    assert list(seen) == ["OMP_NUM_THREADS", "LOKY_MAX_CPU_COUNT"]  # off-count group first
