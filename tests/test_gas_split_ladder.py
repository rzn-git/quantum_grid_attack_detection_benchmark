"""Branch logic of the R4-12 gas ladder runner: registered thresholds, controls, unit lists."""

import importlib.util
import os
import sys

import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

# the runner imports a W3 helper from experiments/, which is not an installed package
sys.path.insert(0, str(REPO_ROOT))
_saved = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                         "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")}  # fmt: skip
spec = importlib.util.spec_from_file_location(
    "run_gas_split_ladder", REPO_ROOT / "experiments" / "run_gas_split_ladder.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for _k, _v in _saved.items():  # the runner pins threads at import; restore the test session's
    if _v is None:
        os.environ.pop(_k, None)
    else:
        os.environ[_k] = _v


def _u(**result):
    return {"result": result, "peak_rss_gb": 0.1}


@pytest.mark.parametrize(
    ("deltas", "branch"),
    [
        ([0.005, -0.01], "not_load_bearing"),
        ([0.015, 0.0], "report_beside_numbers"),
        ([0.0, -0.021], "STOP"),
    ],
)
def test_missingness_branches(deltas, branch):
    units = {f"M_R_{i}": _u(delta=d) for i, d in enumerate(deltas)}
    assert mod.verdict_missingness(units)["branch"] == branch


def _ladder_units(r, t, s_shift=0.0, y=0.5, raw=0.9):
    units = {"L_R": _u(f1=r, neighbour_share=0.7), "L_Rraw": _u(f1=raw)}
    for p in range(5):
        units[f"L_T{p}"] = _u(f1=t, neighbour_share=0.7)
        units[f"L_Y{p}"] = _u(f1=y, neighbour_share=0.7)
        units[f"L_S{p}"] = _u(f1=r + s_shift, neighbour_share=0.7)
    return units


@pytest.mark.parametrize(
    ("r", "t", "branch"),
    [(0.70, 0.64, "PORTS"), (0.70, 0.69, "DOES_NOT_PORT"), (0.70, 0.67, "PARTIAL")],
)
def test_ladder_branches(r, t, branch):
    v = mod.verdict_ladder(_ladder_units(r, t), 5, phase1_seed0=r)
    assert v["branch_H_L1"] == branch and v["anchor"]["pass"]


def test_ladder_void_when_shuffled_control_fails():
    v = mod.verdict_ladder(_ladder_units(0.70, 0.60, s_shift=-0.03), 5, phase1_seed0=0.70)
    assert v["branch_H_L1"].startswith("VOID")


def test_ladder_anchor_fails_beyond_tolerance():
    v = mod.verdict_ladder(_ladder_units(0.70, 0.69), 5, phase1_seed0=0.70 + 2e-6)
    assert not v["anchor"]["pass"]


def test_framing_x3_kill_is_one_sided_above_005():
    def units(delta):
        fo = {"target_f1": 0.6 + delta, "source_matched_f1": 0.6, "source_full_f1": 0.62}
        return {"F_T0": _u(folds=[fo], negative_domain_auc=0.51)}

    assert mod.verdict_framing(units(0.051), "T")["x3_kill"]
    assert not mod.verdict_framing(units(0.049), "T")["x3_kill"]
    v = mod.verdict_framing(units(-0.06), "T")
    assert not v["x3_kill"] and not v["w3_two_sided_within_0.05"]


def test_unit_lists():
    assert len(mod.specs_for("missingness", 5)) == 8
    assert len(mod.specs_for("ladder", 5)) == 2 + 15
    assert [s["key"] for s in mod.specs_for("framing", 2)] == ["F_T0", "F_T1", "F_Y0", "F_Y1"]
