"""R4-25 branch logic and anchor on synthetic cells (no data, no fits)."""

import importlib.util

import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "run_gas_rbf_length_scale", REPO_ROOT / "experiments" / "run_gas_rbf_length_scale.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

ART = {
    "p5a_whitebox_qsvm": {
        "clean_f1": 0.56,
        "whitebox_retention_at_half": 0.11,
        "rbf_control": {"clean_f1": 0.55, "per_seed_pgd": {"0.5": [0.06] * 10}},
    }
}


def _cells(ret, readable=None):
    readable = readable or {}
    return {
        m: {
            "clean_f1": 0.55,
            "attacked_f1_per_seed": [0.06] * 10,
            "retention": r,
            "readable": readable.get(m, True),
        }
        for m, r in ret.items()
    }


@pytest.mark.parametrize(
    ("ret", "readable", "branch"),
    [
        ({0.1: 0.1, 1.0: 0.11, 10.0: 0.3, 100.0: 0.6}, None, "LENGTH-SCALE"),
        ({0.1: 0.1, 1.0: 0.11, 10.0: 0.15, 100.0: 0.05}, None, "NOT LENGTH-SCALE"),
        ({0.1: 0.1, 1.0: 0.11, 10.0: 0.3, 100.0: 0.6}, {100.0: False}, "INTERMEDIATE"),
    ],
)
def test_branches(ret, readable, branch):
    assert mod.verdict(_cells(ret, readable), ART)["branch"] == branch


def test_anchor_reads_clean_and_every_seed():
    cells = _cells({1.0: 0.11})
    assert mod.anchor(cells, ART)["pass"]
    cells[1.0]["attacked_f1_per_seed"][3] += 2e-6
    assert not mod.anchor(cells, ART)["pass"]
