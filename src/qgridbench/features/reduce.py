"""Fitted feature transforms: imputation, standardization, PCA, angle scaling.

LEAKAGE RULE (non-negotiable, enforced by tests/test_leakage.py): every fitted
transform is fit on the TRAIN split only and applied frozen to val/test.

Two regimes (CLAUDE.md section 5):
  - "full": median-impute + standardize all 128 features
  - "pca":  median-impute + standardize + PCA to n_components in {8, 12, 16}
            (the apples-to-apples comparison set for quantum models)

For quantum encodings, PCA outputs are additionally rescaled to [-pi, pi] by a
MinMaxScaler fit on train only (CLAUDE.md section 6 encoding-scaling rule).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ANGLE_RANGE = (-np.pi, np.pi)


def build_preprocessor(regime: str, n_components: int | None = None, seed: int = 0) -> Pipeline:
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if regime == "pca":
        if not n_components:
            raise ValueError("pca regime requires n_components")
        steps.append(("pca", PCA(n_components=n_components, random_state=seed)))
    elif regime != "full":
        raise ValueError(f"unknown regime '{regime}'")
    return Pipeline(steps)


def fit_transform_splits(
    pipe: Pipeline, X: np.ndarray, splits: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Fit on train only; transform all splits. Returns {'train','val','test'} arrays."""
    Z_train = pipe.fit_transform(X[splits["train"]])
    return {
        "train": Z_train,
        "val": pipe.transform(X[splits["val"]]),
        "test": pipe.transform(X[splits["test"]]),
    }


class AngleScaler:
    """Train-fitted MinMax scaling of standardized/PCA features into [-pi, pi].

    The quantum-kernel bandwidth hyperparameter multiplies these angles AFTER
    scaling; the encoding-range pytest asserts the pre-bandwidth range.
    """

    def __init__(self):
        self._mm = MinMaxScaler(feature_range=ANGLE_RANGE)

    def fit(self, Z_train: np.ndarray) -> AngleScaler:
        self._mm.fit(Z_train)
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        # clip: val/test values outside the train min/max would otherwise exceed
        # [-pi, pi] and wrap the rotation angles (documented, deliberate)
        return np.clip(self._mm.transform(Z), ANGLE_RANGE[0], ANGLE_RANGE[1])
