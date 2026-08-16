"""Metric suite: AUPRC, macro-F1, balanced accuracy, ROC-AUC, Brier, ECE.

All metrics operate on predicted probabilities (n_samples, n_classes) so every
model family is scored identically. Binary AUPRC/AUC use the attack class
(label 1) as positive; multiclass uses macro one-vs-rest.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass Brier: mean squared distance to the one-hot target."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def compute_all(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n_classes = proba.shape[1]
    pred = proba.argmax(axis=1)
    out = {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier": brier_score(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
    }
    if n_classes == 2:
        out["auprc"] = float(average_precision_score(y_true, proba[:, 1]))
        out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
    else:
        out["auprc"] = float(
            np.mean(
                [
                    average_precision_score((y_true == c).astype(int), proba[:, c])
                    for c in range(n_classes)
                ]
            )
        )
        out["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    return out
