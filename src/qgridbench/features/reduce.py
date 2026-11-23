"""Fitted feature transforms: imputation, standardization, PCA, angle scaling.

LEAKAGE RULE (non-negotiable, enforced by tests/test_leakage.py): every fitted
transform is fit on the TRAIN split only and applied frozen to val/test.

Two regimes (study protocol §5):
  - "full": median-impute + standardize all 128 features
  - "pca":  median-impute + standardize + PCA to n_components in {8, 12, 16}
            (the apples-to-apples comparison set for quantum models)

For quantum encodings, PCA outputs are additionally rescaled to [-pi, pi] by a
MinMaxScaler fit on train only (study protocol §6 encoding-scaling rule).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ANGLE_RANGE = (-np.pi, np.pi)


def build_preprocessor(
    regime: str, n_components: int | None = None, seed: int = 0, add_indicator: bool = False
) -> Pipeline:
    """`add_indicator=True` appends SimpleImputer's missing-value indicator columns (the
    gas-pipeline missingness control, R4-12); the default is the shipped transform."""
    steps = [
        ("impute", SimpleImputer(strategy="median", add_indicator=add_indicator)),
        ("scale", StandardScaler()),
    ]
    if regime == "pca":
        if not n_components:
            raise ValueError("pca regime requires n_components")
        # The solver is pinned (decision log 2026-11-18, R4-22 incident): under "auto", a wide
        # input (fdia118_binary, 3456 x 726) gets the randomized solver, whose fit_transform
        # returns the approximate U*S, so the train features differed from transform(train) by
        # up to 0.27 while val and test went through transform. covariance_eigh is exact, its
        # fit_transform equals transform, and "auto" already chose it for every other variant
        # (bit-identical there).
        steps.append(
            ("pca", PCA(n_components=n_components, svd_solver="covariance_eigh", random_state=seed))
        )
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


class ClippedAngleScaler:
    """Each component clipped to two train percentiles, then MinMax into [-pi, pi].

    An arm of the angle-scaling control (experiment `angle_scaling`): the shipped
    `AngleScaler` gives the angle range to the extreme train values, so a heavy-tailed
    component leaves its bulk in a few percent of the range. No shipped model uses this.
    """

    def __init__(self, lo_pct: float = 1.0, hi_pct: float = 99.0):
        self.lo_pct, self.hi_pct = lo_pct, hi_pct
        self._mm = MinMaxScaler(feature_range=ANGLE_RANGE)

    def fit(self, Z_train: np.ndarray) -> ClippedAngleScaler:
        self.lo_, self.hi_ = np.percentile(Z_train, [self.lo_pct, self.hi_pct], axis=0)
        self._mm.fit(np.clip(Z_train, self.lo_, self.hi_))
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        scaled = self._mm.transform(np.clip(Z, self.lo_, self.hi_))
        return np.clip(scaled, ANGLE_RANGE[0], ANGLE_RANGE[1])


class QuantileAngleScaler:
    """Each component mapped through its train quantiles to a uniform law on [-pi, pi].

    The second arm of the angle-scaling control. No shipped model uses this.
    """

    def __init__(self, n_quantiles: int = 1000, seed: int = 0):
        from sklearn.preprocessing import QuantileTransformer

        self._qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="uniform",
            subsample=None,
            random_state=seed,
        )

    def fit(self, Z_train: np.ndarray) -> QuantileAngleScaler:
        self._qt.fit(Z_train)
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        lo, hi = ANGLE_RANGE
        return np.clip(lo + (hi - lo) * self._qt.transform(Z), lo, hi)
