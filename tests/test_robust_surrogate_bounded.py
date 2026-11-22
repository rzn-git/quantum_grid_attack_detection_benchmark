"""R4-18 bounded-statistic helpers (run_robust_surrogate_transfer --rng-order p19d)."""

import copy
import importlib.util
import sys

import numpy as np
import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))  # the runner imports experiments.run_adversarial
_spec = importlib.util.spec_from_file_location(
    "run_robust_surrogate_transfer", REPO_ROOT / "experiments" / "run_robust_surrogate_transfer.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_bounded_b_bounds_sign_and_ratio_map():
    assert _mod.bounded_b(0.3, 0.2) == pytest.approx(0.2)  # ratio 1.5x -> B = 0.2
    assert _mod.bounded_b(0.2, 0.3) == pytest.approx(-0.2)
    assert _mod.bounded_b(0.3, -0.05) == pytest.approx(1.0)  # a negative reverse drop saturates
    assert _mod.bounded_b(0.0, 0.0) == 0.0
    rng = np.random.default_rng(0)
    for f, r in rng.normal(size=(200, 2)):
        assert -1.0 <= _mod.bounded_b(f, r) <= 1.0


def test_pooled_b_uses_seed_means_not_mean_of_ratios():
    F, R = [0.29, 0.20], [0.03, 0.24]
    assert _mod.pooled_b(F, R) == pytest.approx((0.245 - 0.135) / (0.245 + 0.135))


def test_paired_delta_ci_on_constant_seeds_is_degenerate_at_delta():
    Fs, Rs, Fa, Ra = [0.2] * 8, [0.2] * 8, [0.3] * 8, [0.1] * 8
    out = _mod.paired_delta_ci(Fs, Rs, Fa, Ra, 500)
    assert out["delta"] == pytest.approx(0.5)
    assert out["ci95"] == pytest.approx([0.5, 0.5])


def test_branches():
    b = _mod.r4_18_branch
    assert b(0.25, [0.05, 0.45]) == "HYPOTHESIS"
    assert b(0.25, [-0.01, 0.45]) == "INCONCLUSIVE"
    assert b(-0.30, [-0.50, -0.10]) == "REVERSED"
    assert b(0.05, [-0.15, 0.19]) == "KILL"
    assert b(0.05, [-0.25, 0.19]) == "INCONCLUSIVE"


def test_restored_generator_state_pairs_the_random_start():
    rng = np.random.default_rng(3)
    rng.uniform(size=5)  # earlier attacks advance the stream
    state = copy.deepcopy(rng.bit_generator.state)
    a = rng.uniform(-1, 1, (4, 3))
    rng.bit_generator.state = copy.deepcopy(state)
    b = rng.uniform(-1, 1, (4, 3))
    assert np.array_equal(a, b)


def _seed(s, own_std, own_at, dropped=()):
    arm = {"F": 0.2, "R": 0.2, "B": 0.0, "surrogate_clean_f1": 0.55, "mean_target_damage": 0.1}
    return {
        "seed": s,
        "clean": {"vqc": 0.5},
        "dropped_targets": [{"target": t, "reason": "constant"} for t in dropped],
        "std": {**arm, "surrogate_own_adv_f1": own_std},
        "at": {**arm, "F": 0.1, "surrogate_own_adv_f1": own_at},
    }


def _need_reference_run() -> None:
    """summarize_r4_18 reads the archived reference run, which the public release lacks."""
    if not (REPO_ROOT / _mod.P19D_RUN / "results.json").is_file():
        pytest.skip("the research run archive (results/runs) is not on this machine")


def test_validity_failure_keeps_b_unread_and_lists_dropped_targets():
    _need_reference_run()
    per = [_seed(20, 0.30, 0.28, dropped=("mlp",)), _seed(21, 0.30, 0.27)]
    out = _mod.summarize_r4_18(per)
    assert out["branch"] == "CONTROL_FAILED" and not out["controls"]["at_is_robust"]
    assert "delta" not in out and "delta" in out["not_read"]
    assert out["dropped_targets_per_seed"] == {"20": ["mlp"], "21": []}


def test_validity_pass_reads_b():
    _need_reference_run()
    out = _mod.summarize_r4_18([_seed(20, 0.20, 0.40), _seed(21, 0.20, 0.40)])
    assert out["controls"]["validity_pass"] and "delta" in out and "not_read" not in out
