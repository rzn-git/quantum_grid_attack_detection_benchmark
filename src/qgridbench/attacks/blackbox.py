"""Query-based, gradient-free black-box attack (HopSkipJump via ART).

Run identically against EVERY model family (closes the objection that
surrogate-transfer under-attacks tree/kernel models). Query budget per sample
is capped and reported (study protocol §7).

NOT REPRODUCIBLE FROM ITS SEED — read this before quoting any number it returns.
ART builds HopSkipJump's initial adversarial point from `np.random.RandomState()` with no
seed (`art/attacks/evasion/hop_skip_jump.py:295`, ART 1.20.1), which the `np.random.seed`
call below cannot reach; only the later gradient-estimation noise uses the global RNG, and
that is what made this path look seeded for the whole of Phase 4. Measured: three runs at
one seed give median distortions 0.661 / 0.636 / 0.703, and across five restarts the
fidelity-kernel SVM moves between 2.808 and 4.917 sigma (CV 0.264) while the MLP moves by
1.7% (decision log 2026-08-15T11:05Z / 11:35Z). Any figure from this function is ONE DRAW:
report mean +/- std over restarts, never a point estimate. The `seed` argument is kept
because it still fixes the noise draws and labels the restart, not because it makes the
attack deterministic.
"""

from __future__ import annotations

import numpy as np
from art.attacks.evasion import HopSkipJump
from art.config import ART_NUMPY_DTYPE
from art.estimators.classification import BlackBoxClassifierNeuralNetwork

from qgridbench.utils.run_tracking import get_logger

log = get_logger(__name__)


class BoundedStepHopSkipJump(HopSkipJump):
    """ART's HopSkipJump with a step-size search that ends when it can no longer succeed.

    ART 1.20.1 halves the step until a candidate is adversarial and has no other exit
    (`hop_skip_jump.py:441-449`, `while not success`). The library casts every query to
    float32, so once the step is below half a float32 unit the model is asked about the current
    point itself. The binary search returns a point it did not test, one float32 rounding from
    the last point it did; where the model's decision flips at that scale (the complex64
    fidelity-kernel path with a large dual mass) that point can be on the clean side, and with
    a gradient estimate that also points into the clean class the library repeats one query
    forever (decision log 2026-11-21T20:40Z: reproduced on the fitted model, four stalls in
    2120 redraws).

    Here the search ends at the first candidate that is not adversarial and equals the current
    point in the precision the model is queried at: rounding is monotone, so every later
    candidate is that same query. The attack keeps its current point and goes on to its next
    iteration, and the event is counted in `stalled_steps`. Everywhere else the class returns
    the library's own answers, so a run without a stall is the library's run.
    """

    # the step is exactly zero after about 1080 halvings of a float64 of order one
    STEP_CAP = 1100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stalled_steps = 0
        self._step_from: np.ndarray | None = None
        self._halvings = 0

    def _binary_search(self, *args, **kwargs):
        self._step_from = None
        return super()._binary_search(*args, **kwargs)

    def _compute_update(self, current_sample, num_eval, delta, target, mask, clip_min, clip_max):
        self._step_from = None
        update = super()._compute_update(
            current_sample=current_sample,
            num_eval=num_eval,
            delta=delta,
            target=target,
            mask=mask,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        # the library's step-size search follows this call and asks the model only through
        # _adversarial_satisfactory until it ends (tests pin the library method's source)
        self._step_from = current_sample
        self._halvings = 0
        return update

    def _adversarial_satisfactory(self, samples, target, clip_min, clip_max):
        result = super()._adversarial_satisfactory(samples, target, clip_min, clip_max)
        if self._step_from is None:
            return result
        if result[0]:
            self._step_from = None
            return result
        self._halvings += 1
        asked = np.clip(samples[0], clip_min, clip_max).astype(ART_NUMPY_DTYPE)
        current = np.clip(self._step_from, clip_min, clip_max).astype(ART_NUMPY_DTYPE)
        if np.array_equal(asked, current):
            self.stalled_steps += 1
            self._step_from = None
            return np.ones_like(result)
        if self._halvings > self.STEP_CAP:
            raise RuntimeError(
                f"HopSkipJump step-size search: {self._halvings} halvings without an "
                "adversarial candidate and without reaching the current point"
            )
        return result


def hopskipjump_attack(
    predict_proba,
    X: np.ndarray,
    n_classes: int,
    max_iter: int = 10,
    max_eval: int = 500,
    init_eval: int = 10,
    seed: int = 0,
    x_adv_init: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """L-inf HopSkipJump against an arbitrary predict-proba function.

    Returns (X_adv, budget_info). Total query budget per sample is bounded by
    roughly max_iter * max_eval + init size; recorded in budget_info.

    `x_adv_init` supplies one starting point per input (`other_class_starts`). The library uses
    it where the model labels it differently from the input and falls back to its own random
    draws elsewhere. Absent, the call is the library default, as every earlier run used.
    """
    n_queries = {"count": 0}

    def counted_predict(x):
        n_queries["count"] += len(x)
        return predict_proba(np.asarray(x, dtype=np.float64))

    clf = BlackBoxClassifierNeuralNetwork(
        counted_predict,
        input_shape=(X.shape[1],),
        nb_classes=n_classes,
        clip_values=None,
    )
    attack = BoundedStepHopSkipJump(
        classifier=clf,
        targeted=False,
        norm=np.inf,
        max_iter=max_iter,
        max_eval=max_eval,
        init_eval=init_eval,
        verbose=False,
    )
    np.random.seed(seed)  # ART uses global numpy state
    if x_adv_init is None:
        X_adv = attack.generate(x=X.astype(np.float32))
    else:
        if x_adv_init.shape != X.shape:
            raise ValueError(f"x_adv_init has shape {x_adv_init.shape}, the inputs {X.shape}")
        X_adv = attack.generate(x=X.astype(np.float32), x_adv_init=x_adv_init.astype(np.float32))
    X_adv = np.asarray(X_adv, dtype=np.float64)
    found = found_mask(predict_proba, X, X_adv)
    budget = {
        "total_queries": int(n_queries["count"]),
        "queries_per_sample": float(n_queries["count"] / max(len(X), 1)),
        "max_iter": max_iter,
        "max_eval": max_eval,
        "init_eval": init_eval,
        "found_share": float(found.mean()) if len(found) else float("nan"),
        "stalled_steps": int(attack.stalled_steps),
    }
    log.info("HopSkipJump: %s", budget)
    if attack.stalled_steps:
        log.warning(
            "HopSkipJump: %d step-size searches ended on a current point that was not "
            "adversarial (BoundedStepHopSkipJump); the library would not have returned",
            attack.stalled_steps,
        )
    if not found.all():
        log.warning(
            "HopSkipJump found no adversarial point for %d of %d inputs and returned them "
            "unchanged: read distortions over the found points only (found_mask)",
            int((~found).sum()),
            len(found),
        )
    return X_adv, budget


def other_class_starts(
    predict_proba, X: np.ndarray, pool: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, int]:
    """One HopSkipJump starting point per input: a pool row the model labels differently.

    The library's default start is a uniform draw in the batch's bounding box, which never
    leaves the predicted class for a model that labels almost the whole box one way
    (`found_mask`). A supplied start is the attack's own alternative: a row of real data that
    the model assigns to another class, drawn at random per input, as the original attack starts
    its targeted mode from a sample of the target class. Pool rows outside the box the library
    clips to (the smallest and largest value of the attacked batch) are left out, so no start is
    moved by the clip.

    Returns (starts, supplied, n_queries). `starts[i]` is the input itself where no pool row has
    another predicted class; the library then falls back to its random draws for that input.
    `n_queries` is the number of pool rows whose label was asked for.
    """
    X32 = np.asarray(X, dtype=np.float32)
    pool32 = np.asarray(pool, dtype=np.float32)
    inside = ((pool32 >= X32.min()) & (pool32 <= X32.max())).all(axis=1)
    pool32 = pool32[inside]
    starts = X32.copy()
    supplied = np.zeros(len(X32), dtype=bool)
    if len(pool32):
        pool_label = np.asarray(predict_proba(pool32.astype(np.float64))).argmax(axis=1)
        clean = np.asarray(predict_proba(X32.astype(np.float64))).argmax(axis=1)
        for i, label in enumerate(clean):
            candidates = np.flatnonzero(pool_label != label)
            if len(candidates):
                starts[i] = pool32[rng.choice(candidates)]
                supplied[i] = True
    return starts.astype(np.float64), supplied, int(len(pool32))


def found_mask(predict_proba, X: np.ndarray, X_adv: np.ndarray) -> np.ndarray:
    """True where the attack returned an adversarial point for the input.

    ART's HopSkipJump starts each input from a uniform random draw in the batch's bounding box
    and gives up after `init_size` (100) draws that all keep the model's own prediction; it then
    returns the INPUT ITSELF (`hop_skip_jump.py:259-262`, ART 1.20.1). Such a row has distortion
    exactly zero and is not an attack: for a model that labels almost the whole box with one
    class, most rows of that class come back this way (decision log 2026-11-20T02:03Z: 76% of the
    RBF-SVM's rows). A row counts as found when it moved and its predicted label differs from
    the clean prediction. Any distortion statistic is taken over the found rows, with the found
    share reported beside it; a not-found row is never a zero.
    """
    X32 = np.asarray(X, dtype=np.float32).astype(np.float64)
    moved = np.any(np.asarray(X_adv, dtype=np.float64) != X32, axis=1)
    clean = np.asarray(predict_proba(X32)).argmax(axis=1)
    attacked = np.asarray(predict_proba(np.asarray(X_adv, dtype=np.float64))).argmax(axis=1)
    return moved & (attacked != clean)
