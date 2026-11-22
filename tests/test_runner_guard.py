"""Runner guard: a fitted predictor that is constant or non-finite on clean rows raises, while an
attacked input that collapses the same model to one class does not (tiny fixtures, no data)."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from qgridbench.attacks.evasion import pgd


@pytest.fixture(scope="module")
def mods():
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        yield (
            importlib.import_module("experiments.runner_guard"),
            importlib.import_module("experiments.run_adversarial"),
        )


class _Sign:
    """Predicts attack (class 1) iff feature 0 is positive."""

    def predict_proba(self, X):
        p1 = (X[:, 0] > 0).astype(float)
        return np.column_stack([1 - p1, p1])


class _Constant:
    def predict_proba(self, X):
        return np.tile([0.2, 0.8], (len(X), 1))


class _NaN:
    def predict_proba(self, X):
        p = _Sign().predict_proba(X)
        p[3, :] = np.nan
        return p


def _zoo(run_adversarial, model):
    zoo = run_adversarial.FittedZoo.__new__(run_adversarial.FittedZoo)
    zoo.models = {"stub": model}
    return zoo


@pytest.fixture
def clean():
    X = np.random.default_rng(0).uniform(-1, 1, (40, 3))
    return X, (X[:, 0] > 0).astype(int)


def test_constant_classifier_raises(mods, clean):
    guard, run_adversarial = mods
    X, y = clean
    with pytest.raises(guard.DegeneratePredictorError, match=r"'stub'.*predicted counts \{1: 40\}"):
        _zoo(run_adversarial, _Constant()).clean_proba("stub", X, y)


def test_nan_classifier_raises(mods, clean):
    guard, run_adversarial = mods
    X, y = clean
    with pytest.raises(guard.DegeneratePredictorError, match="2 non-finite of 80 outputs"):
        _zoo(run_adversarial, _NaN()).clean_proba("stub", X, y)


def test_healthy_classifier_passes_and_returns_predict_proba(mods, clean):
    _, run_adversarial = mods
    X, y = clean
    zoo = _zoo(run_adversarial, _Sign())
    np.testing.assert_array_equal(zoo.clean_proba("stub", X, y), zoo.predict_proba("stub", X))


def test_constant_on_attacked_rows_does_not_raise(mods, clean):
    """PGD pushes every row across the boundary; the attack path reports the collapse."""
    guard, run_adversarial = mods
    X, y = clean
    zoo = _zoo(run_adversarial, _Sign())

    def push_to_natural(Xb, yb):
        g = np.zeros_like(Xb)
        g[:, 0] = -1.0
        return g

    Xa = pgd(push_to_natural, X, y, 3.0, np.ones(3), n_steps=10, step_frac=0.5)
    pred = zoo.predict_proba("stub", Xa).argmax(1)
    assert np.unique(pred).tolist() == [0]
    # the same rows on the clean call site would fail: the call site is the distinction
    with pytest.raises(guard.DegeneratePredictorError):
        zoo.clean_proba("stub", Xa, y)


def test_single_class_truth_is_not_constant_failure(mods):
    """A per-class subset (attack rows only) predicted all-attack is correct, not degenerate."""
    guard, _ = mods
    guard.assert_clean_predictor("stub", np.ones(5), np.ones(5), np.ones(5, dtype=int))


def test_decision_function_and_predictor_values(mods):
    """P9b's sign-of-decision path and D2e's per-file predictor path."""
    guard, _ = mods
    y = np.array([0, 1, 0, 1])
    guard.assert_clean_predictor("svm", y, np.array([-1.0, 2.0, -0.5, 0.3]), np.array([0, 1, 0, 1]))
    dec = np.array([-1.0, -2.0, -0.5, -0.3])
    with pytest.raises(guard.DegeneratePredictorError, match="predicted counts"):
        guard.assert_clean_predictor("svm", y, dec, dec > 0)
    excess = [0.1, 0.3, 0.2, 0.05]
    with pytest.raises(guard.DegeneratePredictorError, match="'attack_scenario_entropy'"):
        guard.assert_clean_predictor("attack_scenario_entropy", excess, [0.0] * 4, [0.0] * 4)
