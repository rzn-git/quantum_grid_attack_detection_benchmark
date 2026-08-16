"""Trivial reference predictors — the floor every reported model must clear.

Without these rows a reader cannot tell a working model from a broken one. On the
binary task a stratified-random guesser already scores 0.5005 macro-F1, so several
models in the matched-dimensionality (PCA) comparison sit within noise of chance,
and one is below it. Any robustness or accuracy statement about a near-chance model
is close to meaningless, and the baseline row is what makes that visible instead of
leaving it to be discovered by a reviewer.

These are references, not competitors: they carry an `aggregate` shaped like a real
model so the table renders them, but no `per_seed`, so they are excluded from the
paired significance tests by construction.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from qgridbench.eval.metrics import brier_score, expected_calibration_error

N_DRAWS = 20  # random-baseline averaging; deterministic given the seed
SEED = 0


def _agg(value: float, std: float = 0.0) -> dict:
    return {"mean": float(value), "std": float(std)}


def trivial_baselines(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Majority-class and prior-matched random baselines, in results.json shape.

    The class prior is taken from TRAIN (the only split a real predictor may see);
    the metrics are computed on TEST, exactly as for every model.
    """
    classes, counts = np.unique(y_train, return_counts=True)
    prior = counts / counts.sum()
    n_classes = len(classes)
    out: dict = {}

    # --- majority class: the constant predictor -------------------------------
    maj = classes[counts.argmax()]
    pred = np.full(len(y_test), maj)
    proba = np.zeros((len(y_test), n_classes))
    proba[:, int(np.searchsorted(classes, maj))] = 1.0
    out["baseline_majority"] = {
        "aggregate": {
            "macro_f1": _agg(f1_score(y_test, pred, average="macro")),
            "balanced_accuracy": _agg(balanced_accuracy_score(y_test, pred)),
            "auprc": _agg(float(np.mean(y_test == maj)) if n_classes == 2 else float("nan")),
            "roc_auc": _agg(0.5),
            "brier": _agg(brier_score(y_test, proba)),
            "ece": _agg(expected_calibration_error(y_test, proba)),
        },
        "reference_only": True,
    }

    # --- stratified random: matches the train prior ---------------------------
    rng = np.random.default_rng(SEED)
    f1s, bas, aucs = [], [], []
    for _ in range(N_DRAWS):
        p = rng.choice(classes, size=len(y_test), p=prior)
        f1s.append(f1_score(y_test, p, average="macro"))
        bas.append(balanced_accuracy_score(y_test, p))
        if n_classes == 2:
            aucs.append(roc_auc_score(y_test, p))
    prior_proba = np.tile(prior, (len(y_test), 1))
    out["baseline_stratified_random"] = {
        "aggregate": {
            "macro_f1": _agg(np.mean(f1s), np.std(f1s, ddof=1)),
            "balanced_accuracy": _agg(np.mean(bas), np.std(bas, ddof=1)),
            "auprc": _agg(float(prior[classes == 1][0]) if n_classes == 2 else float("nan")),
            "roc_auc": _agg(np.mean(aucs) if aucs else float("nan")),
            "brier": _agg(brier_score(y_test, prior_proba)),
            "ece": _agg(expected_calibration_error(y_test, prior_proba)),
        },
        "reference_only": True,
    }
    return out
