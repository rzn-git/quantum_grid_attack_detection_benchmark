"""The registered reading of experiments/run_training_draws.py (pure parts; no model is fitted)."""

import numpy as np

from experiments.run_training_draws import ENCODINGS, PAIR, anchors, lead_statistic


def test_a_lead_on_every_draw_above_the_margin_is_real():
    out = lead_statistic(
        np.array([0.03, 0.04, 0.025, 0.035, 0.03, 0.05, 0.028, 0.031, 0.04, 0.033])
    )
    assert out["branch"] == "REAL" and out["n_positive"] == 10
    assert out["wilcoxon_p_two_sided"] < 0.05


def test_a_lead_that_changes_sign_is_not_established():
    out = lead_statistic(np.array([0.03, -0.02, 0.04, -0.03, 0.02, 0.01, -0.01, 0.03, -0.02, 0.02]))
    assert out["branch"] == "NOT_ESTABLISHED"


def test_a_consistent_lead_under_the_margin_is_not_established():
    out = lead_statistic(np.full(10, 0.01) + np.linspace(0, 0.002, 10))
    assert out["n_positive"] == 10 and out["mean"] < 0.02
    assert out["branch"] == "NOT_ESTABLISHED"


def test_the_mirror_image_is_reversed():
    out = lead_statistic(
        -np.array([0.03, 0.04, 0.025, 0.035, 0.03, 0.05, 0.028, 0.031, 0.04, 0.033])
    )
    assert out["branch"] == "REVERSED" and out["n_negative"] == 10


def test_one_draw_of_ten_may_differ_in_sign():
    d = np.array([0.03, 0.04, 0.025, 0.035, 0.03, 0.05, 0.028, 0.031, 0.04, -0.001])
    assert lead_statistic(d)["branch"] == "REAL"
    d[1] = -0.002
    assert lead_statistic(d)["branch"] == "NOT_ESTABLISHED"


def test_all_zero_differences_do_not_raise():
    out = lead_statistic(np.zeros(10))
    assert out["branch"] == "NOT_ESTABLISHED" and out["wilcoxon_p_two_sided"] == 1.0


def test_anchors_compare_draw_zero_with_the_published_run():
    published = {
        "qsvm_zz": {"aggregate": {"macro_f1": {"mean": 0.5643}}},
        "qsvm_angle_ry": {"aggregate": {"macro_f1": {"mean": 0.5265}}},
        "xgb": {"per_seed": [{"macro_f1": 0.596}, {"macro_f1": 0.598}, {"macro_f1": 0.597}]},
    }
    draw0 = {
        "scores": {
            "qsvm_zz": 0.5643,
            "qsvm_angle_ry": 0.5270,
            "xgb": {"0": 0.597, "1": 0.599, "2": 0.598},
        }
    }
    out = anchors(draw0, published, [0, 1, 2])
    assert out["qsvm_zz"]["pass"] and not out["qsvm_angle_ry"]["pass"]
    assert out["xgb"]["pass"] and abs(out["xgb"]["abs_diff"] - 0.001) < 1e-12


def test_every_pair_names_a_model_the_runner_scores():
    for classical, quantum in PAIR.values():
        assert classical == "xgb" and quantum in {f"qsvm_{e}" for e in ENCODINGS}
