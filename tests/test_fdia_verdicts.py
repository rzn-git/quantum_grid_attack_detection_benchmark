"""R4-22 verdict runner: the H1 branches and the H2 component lines on toy inputs (no runs)."""

import sys

import numpy as np
import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

if str(REPO_ROOT) not in sys.path:  # experiments/ is not installed
    sys.path.insert(0, str(REPO_ROOT))
from experiments import run_fdia_verdicts as v  # noqa: E402

SEEDS = 10


def _f1(best_c=0.99, vqc=0.80, qsvm=0.985):
    rng = np.random.default_rng(0)
    f1 = {m: list(0.9 + 0.001 * rng.standard_normal(SEEDS)) for m in v.CLASSICAL}
    f1["xgb"] = list(best_c + 0.001 * rng.standard_normal(SEEDS))
    f1["vqc"] = list(vqc + 0.001 * rng.standard_normal(SEEDS))
    f1["qsvm_zz"] = [qsvm] * SEEDS
    f1["qsvm_angle_ry"] = [qsvm - 0.01] * SEEDS
    return f1


def test_h1_holds_when_quantum_is_below_classical():
    out = v.h1_verdict(_f1())
    assert out["best_classical"] == "xgb" and out["best_quantum"] == "qsvm_zz"
    assert out["d_quantum_minus_classical"] == pytest.approx(0.985 - out["mean"]["xgb"])
    assert out["branch_H1"] == "HOLDS"


def test_h1_kill_needs_both_size_and_significance():
    killed = v.h1_verdict(_f1(best_c=0.90, qsvm=0.95))
    assert killed["branch_H1"] == "KILL"
    # a VQC 0.03 ahead with seed noise far above the margin is not significant
    rng = np.random.default_rng(1)
    f1 = _f1(best_c=0.90, qsvm=0.85)
    f1["vqc"] = list(0.93 + 0.2 * rng.standard_normal(SEEDS))
    noisy = v.h1_verdict(f1)
    assert noisy["best_quantum"] == "vqc" and noisy["branch_H1"] in ("INTERMEDIATE", "HOLDS")
    assert noisy["branch_H1"] != "KILL"


def test_h1_intermediate_between_the_lines():
    out = v.h1_verdict(_f1(best_c=0.90, qsvm=0.915))
    assert 0.01 < out["d_quantum_minus_classical"] < 0.02
    assert out["branch_H1"] == "INTERMEDIATE"


def _wb(q, r):
    seeds = [q * 0.9] * SEEDS
    return {
        "p5a_whitebox_qsvm": {
            "whitebox_retention_at_half": q,
            "clean_f1": 0.9,
            "per_seed_pgd": {"0.5": seeds},
            "rbf_control": {
                "whitebox_retention_at_half": r,
                "clean_f1": 0.9,
                "per_seed_pgd": {"0.5": [r * 0.9] * SEEDS},
            },
        }
    }


def test_h2_fragility_and_availability_lines():
    assert v.fragility(_wb(0.20, 0.31))["branch"] == "REPLICATES"
    assert v.fragility(_wb(0.235, 0.334))["branch"] == "NOT"  # F = -0.099, just inside
    adv = {"per_family": {"qsvm": {"retention": 0.45}}}
    assert v.availability(adv, _wb(0.25, 0.3))["branch"] == "REPLICATES"
    assert v.availability(adv, _wb(0.26, 0.3))["branch"] == "NOT"


def test_h2_asymmetry_b_matches_the_ratio_band():
    assert v.asymmetry_b(2.0) == pytest.approx(1 / 3)
    assert v.asymmetry_b(0.5) == pytest.approx(-1 / 3)
    sur = {
        "summary": {
            "mlp_balanced": {"forward_over_reverse": 1.9},
            "mlp": {"forward_over_reverse": 4.0},
        }
    }
    out = v.asymmetry(sur)
    assert out["branch"] == "REPLICATES" and out["mlp"]["within"] is False
