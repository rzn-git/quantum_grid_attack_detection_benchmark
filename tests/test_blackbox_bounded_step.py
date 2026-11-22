"""The HopSkipJump step-size search ends when it can no longer succeed.

Regression for the incident of decision log 2026-11-21T19:34Z and 20:40Z: ART 1.20.1 halves the
step until a candidate is adversarial, with no other exit, and repeated one float32 query
forever on the gas data's quantum-kernel SVM. `BoundedStepHopSkipJump` ends that search, keeps
the current point and counts the event; where no search stalls it is the library's attack.
"""

import hashlib
import inspect

import numpy as np
import pytest
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import BlackBoxClassifierNeuralNetwork

from qgridbench.attacks.blackbox import BoundedStepHopSkipJump, found_mask, hopskipjump_attack

# sha256 of the source of HopSkipJump._attack in ART 1.20.1. The bounded class relies on that
# method's order of calls (the update, then the step-size search through
# _adversarial_satisfactory, then the next binary search); a library upgrade that changes the
# method must be read before this value is changed.
ATTACK_SOURCE_SHA256 = "c751cf912a13a1cf5296fe904d0d2dcda51f7032480a2f2085fe2996a9c0a2cd"


def _threshold_proba(z):
    """Class 1 where the first feature is above one."""
    p1 = (np.asarray(z)[:, 0] > 1.0).astype(float)
    return np.column_stack([1 - p1, p1])


def _limited(predict, limit=20000):
    """The predictor with a query limit: a search that does not end fails the test, not hangs it."""
    calls = {"n": 0}

    def limited(z):
        calls["n"] += len(z)
        if calls["n"] > limit:
            raise AssertionError("the step-size search did not end")
        return predict(z)

    return limited


def _classifier(predict, n_features):
    return BlackBoxClassifierNeuralNetwork(
        _limited(predict), input_shape=(n_features,), nb_classes=2, clip_values=None
    )


def test_the_library_method_the_bound_relies_on_is_unchanged():
    source = inspect.getsource(HopSkipJump._attack)
    assert "while not success:" in source
    assert hashlib.sha256(source.encode()).hexdigest() == ATTACK_SOURCE_SHA256


def test_without_a_stall_the_bounded_attack_is_the_library_attack():
    rng = np.random.default_rng(0)
    # rows on both sides of the boundary at one, so the box the library clips to (the smallest
    # and largest value of the batch) holds both classes
    X = rng.uniform(-1.0, 3.0, size=(8, 3)).astype(np.float32)
    X[0], X[1] = -1.0, 3.0
    starts = X.copy()
    # a start of the other class per row, so the library makes no unseeded random draw
    starts[:, 0] = np.where(X[:, 0] > 1.0, 0.0, 2.0)
    out = {}
    for cls in (HopSkipJump, BoundedStepHopSkipJump):
        attack = cls(
            classifier=_classifier(_threshold_proba, 3),
            targeted=False,
            norm=np.inf,
            max_iter=3,
            max_eval=40,
            init_eval=10,
            verbose=False,
        )
        np.random.seed(7)
        out[cls] = attack.generate(x=X, x_adv_init=starts)
        if cls is BoundedStepHopSkipJump:
            assert attack.stalled_steps == 0
    assert np.array_equal(out[HopSkipJump], out[BoundedStepHopSkipJump])
    assert found_mask(_threshold_proba, X, out[BoundedStepHopSkipJump]).all()


def test_a_step_search_that_cannot_succeed_ends_and_is_counted():
    """The stall, built from its two measured ingredients: a binary search that hands back a
    point on the clean side, and an update that points further into the clean class. The
    library's loop never returns from this; the predictor's query limit fails the test instead
    of hanging it if the bound does not engage."""
    clean_point = np.array([0.5, 0.2, 0.2], dtype=np.float32)

    class Stuck(BoundedStepHopSkipJump):
        def _binary_search(self, *args, **kwargs):
            super()._binary_search(*args, **kwargs)
            return clean_point.copy()

        def _compute_update(self, *args, **kwargs):
            super()._compute_update(*args, **kwargs)
            return np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    X = np.array([[0.0, 0.0, 0.0], [0.1, 0.3, 3.0]], dtype=np.float32)
    starts = np.array([[2.0, 0.0, 0.0], [2.5, 0.3, 3.0]], dtype=np.float32)
    attack = Stuck(
        classifier=_classifier(_threshold_proba, 3),
        targeted=False,
        norm=np.inf,
        max_iter=2,
        max_eval=20,
        init_eval=10,
        verbose=False,
    )
    np.random.seed(0)
    X_adv = attack.generate(x=X, x_adv_init=starts)
    # one stalled search per iteration and row; the attack kept the point it was handed
    assert attack.stalled_steps == 2 * len(X)
    assert np.array_equal(X_adv, np.tile(clean_point, (len(X), 1)))
    # a moved row that is not adversarial is not a found row
    assert not found_mask(_threshold_proba, X, X_adv).any()


def test_a_candidate_that_differs_from_the_current_point_keeps_the_search_open():
    attack = BoundedStepHopSkipJump(
        classifier=_classifier(_threshold_proba, 3), targeted=False, norm=np.inf, verbose=False
    )
    current = np.array([0.5, 0.2, 0.2], dtype=np.float32)
    attack._step_from = current
    far = (current + np.array([0.1, 0.0, 0.0], dtype=np.float32))[None]
    assert not attack._adversarial_satisfactory(far, target=0, clip_min=0.0, clip_max=3.0)[0]
    assert attack.stalled_steps == 0 and attack._step_from is not None
    # below half a float32 unit the model is asked about the current point itself
    near = (current.astype(np.float64) + 1e-12)[None]
    assert attack._adversarial_satisfactory(near, target=0, clip_min=0.0, clip_max=3.0)[0]
    assert attack.stalled_steps == 1 and attack._step_from is None


def test_the_wrapper_reports_stalled_steps():
    X = np.random.default_rng(1).normal(size=(4, 3))
    _, budget = hopskipjump_attack(
        lambda z: _threshold_proba(np.asarray(z) + 1.0), X, n_classes=2, max_iter=2, max_eval=30
    )
    assert budget["stalled_steps"] == 0


def test_the_cap_fails_loud_when_the_search_neither_succeeds_nor_reaches_the_point():
    attack = BoundedStepHopSkipJump(
        classifier=_classifier(_threshold_proba, 3), targeted=False, norm=np.inf, verbose=False
    )
    attack._step_from = np.array([0.5, 0.2, 0.2], dtype=np.float32)
    attack._halvings = attack.STEP_CAP
    far = np.array([[0.6, 0.2, 0.2]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="halvings without an adversarial candidate"):
        attack._adversarial_satisfactory(far, target=0, clip_min=0.0, clip_max=3.0)
