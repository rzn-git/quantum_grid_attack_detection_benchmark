"""A HopSkipJump row the attack could not start on is not a zero-distortion attack.

Regression for the defect of decision log 2026-11-20T02:03Z: the library returns the input
unchanged when its random starting draws all keep the model's own prediction, and the runner
read those rows as distortion 0.000 (the RBF-SVM and the random forest in the robustness table).
"""

import logging

import numpy as np

from qgridbench.attacks.blackbox import found_mask, hopskipjump_attack, other_class_starts


def _threshold_proba(z):
    """Class 1 where the first feature is positive."""
    p1 = (np.asarray(z)[:, 0] > 0).astype(float)
    return np.column_stack([1 - p1, p1])


def test_found_mask_needs_a_move_and_a_flip():
    X = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    X_adv = np.array(
        [
            [1.0, 0.0],  # returned unchanged: the attack never started
            [-0.5, 0.0],  # moved across the boundary
            [0.5, 0.0],  # moved, prediction kept
            [0.25, 0.0],  # moved across the boundary
        ]
    )
    assert found_mask(_threshold_proba, X, X_adv).tolist() == [False, True, False, True]


def test_attack_that_cannot_start_reports_no_found_rows(caplog):
    def one_class(z):
        return np.column_stack([np.zeros(len(z)), np.ones(len(z))])

    X = np.random.default_rng(0).normal(size=(4, 3))
    with caplog.at_level(logging.WARNING):
        X_adv, budget = hopskipjump_attack(one_class, X, n_classes=2, max_iter=1, max_eval=20)
    # the library hands the inputs back; that is a failure to start, never distortion zero
    assert np.array_equal(X_adv, X.astype(np.float32).astype(np.float64))
    assert budget["found_share"] == 0.0
    assert not found_mask(one_class, X, X_adv).any()
    assert "found no adversarial point for 4 of 4 inputs" in caplog.text


def test_attack_that_starts_reports_every_row_found():
    X = np.random.default_rng(1).normal(size=(4, 3))
    X_adv, budget = hopskipjump_attack(_threshold_proba, X, n_classes=2, max_iter=2, max_eval=30)
    assert budget["found_share"] == 1.0
    assert found_mask(_threshold_proba, X, X_adv).all()


def _small_ball_proba(z):
    """Class 1 only within 0.01 of the origin: uniform draws in the box all but never land there."""
    p1 = (np.linalg.norm(np.asarray(z), axis=1) < 0.01).astype(float)
    return np.column_stack([1 - p1, p1])


def test_supplied_starts_are_pool_rows_of_another_predicted_class():
    X = np.array([[1.0, 0.2], [-1.0, 0.1], [0.8, -0.3]])
    pool = np.array([[0.9, 0.0], [-0.7, 0.0], [5.0, 0.0], [-0.4, 0.2]])
    starts, supplied, n_queries = other_class_starts(
        _threshold_proba, X, pool, np.random.default_rng(0)
    )
    assert supplied.all()
    assert n_queries == 3  # the row at 5.0 lies outside the box the library clips to
    assert (_threshold_proba(starts).argmax(1) != _threshold_proba(X).argmax(1)).all()
    inside = pool[:, 0] < 5.0
    for row in starts:
        assert any(np.allclose(row, p) for p in pool[inside])


def test_a_pool_without_another_class_supplies_nothing():
    def one_class(z):
        return np.column_stack([np.zeros(len(z)), np.ones(len(z))])

    X = np.random.default_rng(2).normal(size=(3, 2))
    starts, supplied, _ = other_class_starts(one_class, X, X[::-1], np.random.default_rng(0))
    assert not supplied.any()
    assert np.array_equal(starts, X.astype(np.float32).astype(np.float64))


def test_supplied_start_reaches_rows_the_random_start_cannot():
    X = np.random.default_rng(3).normal(size=(4, 3))
    assert _small_ball_proba(X).argmax(1).sum() == 0
    _, budget = hopskipjump_attack(_small_ball_proba, X, n_classes=2, max_iter=1, max_eval=20)
    assert budget["found_share"] == 0.0

    starts, supplied, _ = other_class_starts(
        _small_ball_proba, X, np.zeros((1, 3)), np.random.default_rng(0)
    )
    assert supplied.all()
    X_adv, budget = hopskipjump_attack(
        _small_ball_proba, X, n_classes=2, max_iter=1, max_eval=20, x_adv_init=starts
    )
    assert budget["found_share"] == 1.0
    # the attack ends no farther from the input than the start it was given
    assert (np.abs(X_adv - X).max(axis=1) <= np.abs(starts - X).max(axis=1) + 1e-6).all()
