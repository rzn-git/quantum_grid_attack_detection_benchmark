"""Classical baseline zoo: logreg, RBF-SVM, RF, XGBoost, LightGBM, 2-layer MLP.

Equal-budget fairness: every model gets the same Optuna trial budget (see
configs/classical.yaml). Class imbalance is handled with class weights, never
synthetic oversampling (decision log).

KNOWN, REPORTED DEVIATION — read `CLASS_BALANCE_ROUTE` and `fit_model` before changing
anything here. `MLPClassifier` has no `class_weight` parameter, so the study's published
MLP is fitted with NO imbalance handling on a 71%-positive task. That is a defect, not a
design choice: it puts the MLP below the stratified-random floor and, because the MLP is
also the transfer surrogate, it manufactures the transfer asymmetry the paper originally
reported (decision log 2026-08-15T10:15Z / 11:20Z / 12:20Z). The default is kept so every
published artifact reproduces; `fit_model(..., balance_via_sample_weight=True)` is the
corrected fit. An earlier version of this docstring claimed sklearn supports neither
`class_weight` nor `sample_weight` for this estimator — the `sample_weight` half is wrong
on scikit-learn >= 1.9, and that error is what delayed finding the defect.
"""

from __future__ import annotations

import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

MODEL_NAMES = ["logreg", "rbf_svm", "rf", "xgb", "lgbm", "mlp"]


def suggest_params(name: str, trial) -> dict:
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-4, 1e2, log=True)}
    if name == "rbf_svm":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
        }
    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
        }
    if name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    if name == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    if name == "mlp":
        return {
            "h1": trial.suggest_categorical("h1", [64, 128, 256, 512]),
            "h2": trial.suggest_categorical("h2", [32, 64, 128, 256]),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }
    raise ValueError(f"unknown model '{name}'")


def build(name: str, params: dict, seed: int):
    if name == "logreg":
        return LogisticRegression(
            C=params["C"], max_iter=3000, class_weight="balanced", random_state=seed
        )
    if name == "rbf_svm":
        # probability=True forces internal 5-fold CV on every fit (intractable across
        # the 50-trial x multi-seed sweep). Fit fast; probabilities are derived from the
        # decision function at predict time (see predict_proba_timed).
        return SVC(
            C=params["C"],
            gamma=params["gamma"],
            probability=False,
            class_weight="balanced",
            cache_size=1000,
            random_state=seed,
        )
    if name == "rf":
        return RandomForestClassifier(
            **params, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        )
    if name == "xgb":
        return XGBClassifier(
            **params,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
        )
    if name == "lgbm":
        return LGBMClassifier(
            **params, class_weight="balanced", n_jobs=-1, random_state=seed, verbosity=-1
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(params["h1"], params["h2"]),
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            batch_size=params["batch_size"],
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=seed,
        )
    raise ValueError(f"unknown model '{name}'")


# How each estimator receives the class-imbalance policy. This map is the SSOT for the
# policy and is asserted by tests/test_boundary.py, because the policy used to be implicit
# in `build`'s keyword arguments and silently skipped the one estimator that takes none.
CLASS_BALANCE_ROUTE = {
    "logreg": "class_weight",
    "rbf_svm": "class_weight",
    "rf": "class_weight",
    "lgbm": "class_weight",
    "xgb": "sample_weight",  # no class_weight parameter
    "mlp": "none",  # no class_weight parameter; see the note below
}


def fit_model(
    name: str, est, X: np.ndarray, y: np.ndarray, balance_via_sample_weight: bool = False
) -> float:
    """Fit with the class-imbalance policy applied; returns wall-clock seconds.

    PUBLISHED BEHAVIOUR IS THE DEFAULT, deliberately. Most estimators receive the policy
    through `class_weight=` in `build`; XGBoost has no such parameter and always gets
    balanced `sample_weight`. `MLPClassifier` ALSO has no `class_weight`, and by default
    receives no weighting at all — that is the configuration every published table and
    figure was produced with, and it is why the MLP rows sit below the stratified-random
    floor (decision log 2026-08-15T10:15Z). It is kept as the default so those artifacts
    reproduce exactly; `CLASS_BALANCE_ROUTE` makes the gap explicit rather than implicit.

    `balance_via_sample_weight=True` selects the corrected fit. `MLPClassifier.fit` does
    accept `sample_weight` on scikit-learn >= 1.9 (the module docstring's earlier claim
    that it accepts neither was wrong — corrected 2026-08-15T11:20Z), and passing it lifts
    the MLP from 0.465 to 0.532 macro-F1 at PCA-8 and collapses the transfer asymmetry the
    unweighted fit manufactures. Use it for the corrected-surrogate control; switching the
    default is a re-run of Phases 2 and 4, tracked in BACKLOG.md.
    """
    t0 = time.perf_counter()
    if name == "xgb" or balance_via_sample_weight:
        est.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        est.fit(X, y)
    return time.perf_counter() - t0


def _decision_to_proba(est, X: np.ndarray) -> np.ndarray:
    """Softmax/sigmoid over the SVM decision function (no internal CV needed).

    Binary: sigmoid of the margin -> [P(natural), P(attack)]. Multiclass (OvR
    decision_function shape n x n_classes): softmax over per-class margins.
    """
    d = est.decision_function(X)
    classes = est.classes_
    if len(classes) == 2:
        p1 = 1.0 / (1.0 + np.exp(-d))
        return np.column_stack([1 - p1, p1])
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict_proba_timed(est, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Predict probabilities; returns (proba, per-sample latency seconds).

    Falls back to a decision-function sigmoid/softmax for estimators fit without
    probability calibration (the RBF-SVM, for speed)."""
    t0 = time.perf_counter()
    if hasattr(est, "predict_proba") and getattr(est, "probability", True):
        proba = est.predict_proba(X)
    elif hasattr(est, "decision_function"):
        proba = _decision_to_proba(est, X)
    else:
        proba = est.predict_proba(X)
    return proba, (time.perf_counter() - t0) / max(len(X), 1)


def stratified_cap(X, y, cap: int, seed: int) -> np.ndarray:
    """Deterministic stratified subset indices of size <= cap."""
    if len(y) <= cap:
        return np.arange(len(y))
    from sklearn.model_selection import train_test_split

    idx, _ = train_test_split(np.arange(len(y)), train_size=cap, stratify=y, random_state=seed)
    return np.sort(idx)
