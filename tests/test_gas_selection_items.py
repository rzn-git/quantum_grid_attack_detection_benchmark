"""Checklist items 7 and 8 (R4-12) on synthetic P8a / P11c artifacts."""

import importlib.util

from qgridbench.utils.run_tracking import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "run_gas_selection_items", REPO_ROOT / "experiments" / "run_gas_selection_items.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _width(clean, mass, f1mf):
    return {
        "gamma": 1.0,
        "curve": {
            "0.0": {"macro_f1": clean},
            "2.0": {"kernel_mass_ratio": mass, "f1_minus_floor": f1mf},
        },
    }


def test_item7_needs_disabled_and_inverted_at_similar_clean_f1():
    p8a = {
        "constant_predictor_floor": 0.4,
        "per_width": {"1.0": _width(0.60, 0.01, 0.0), "0.01": _width(0.58, 0.5, -0.3)},
    }
    assert mod.item7(p8a)["reproduces"]
    p8a["per_width"]["0.01"] = _width(0.45, 0.5, -0.3)  # clean F1 moved 0.15
    assert not mod.item7(p8a)["reproduces"]


def test_item8_reads_non_degenerate_cells_only():
    part_b = {
        "1.0": {"disabled": True, "degenerate": False},
        "100.0": {"disabled": False, "degenerate": False},
        "0.01": {"disabled": False, "degenerate": True},
    }
    assert mod.item8({"part_b_regularization": part_b})["reproduces"]
    part_b["100.0"]["disabled"] = True
    assert not mod.item8({"part_b_regularization": part_b})["reproduces"]
