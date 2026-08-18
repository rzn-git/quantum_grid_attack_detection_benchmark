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
from art.estimators.classification import BlackBoxClassifierNeuralNetwork

from qgridbench.utils.run_tracking import get_logger

log = get_logger(__name__)


def hopskipjump_attack(
    predict_proba,
    X: np.ndarray,
    n_classes: int,
    max_iter: int = 10,
    max_eval: int = 500,
    init_eval: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """L-inf HopSkipJump against an arbitrary predict-proba function.

    Returns (X_adv, budget_info). Total query budget per sample is bounded by
    roughly max_iter * max_eval + init size; recorded in budget_info.
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
    attack = HopSkipJump(
        classifier=clf,
        targeted=False,
        norm=np.inf,
        max_iter=max_iter,
        max_eval=max_eval,
        init_eval=init_eval,
        verbose=False,
    )
    np.random.seed(seed)  # ART uses global numpy state
    X_adv = attack.generate(x=X.astype(np.float32))
    budget = {
        "total_queries": int(n_queries["count"]),
        "queries_per_sample": float(n_queries["count"] / max(len(X), 1)),
        "max_iter": max_iter,
        "max_eval": max_eval,
        "init_eval": init_eval,
    }
    log.info("HopSkipJump: %s", budget)
    return np.asarray(X_adv, dtype=np.float64), budget
