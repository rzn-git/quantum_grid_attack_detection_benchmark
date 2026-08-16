"""Shared experimental protocol: feature regimes, per-seed subsets, evaluation.

Single source of truth used by every experiment script so that classical and
quantum families see IDENTICAL splits, transforms, and training subsets.

Protocol facts (decision log has rationale):
  - Transforms (imputer/scaler/PCA/angle-scaler) are fit on the FULL train split
    only (leakage rule), identically for every family.
  - The kernel-cap subset (<= 2000 stratified train points) is shared between the
    quantum kernels and the classical PCA-regime comparison so the contrast is fair.
  - The test split is materialized once per final config; all selection is on val.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qgridbench.data.preprocess import load_processed
from qgridbench.features.reduce import AngleScaler, build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import stratified_cap


@dataclass
class Regime:
    """A prepared feature view of one variant under one regime, all splits."""

    name: str
    n_components: int | None
    X: dict[str, np.ndarray]  # 'train'/'val'/'test' feature arrays
    y: dict[str, np.ndarray]
    angles: dict[str, np.ndarray] | None  # [-pi,pi]-scaled view (pca regime only)
    angle_scaler: AngleScaler | None = None  # train-fitted; attacks transform through it

    @property
    def feature_scale(self) -> np.ndarray:
        """Per-feature TRAIN std of this view — the unit of the attack budget.

        SSOT for the adversarial scale contract (attacks/evasion.py): PCA outputs
        carry eigenvalue-sized variance, so an epsilon expressed in raw PCA units
        is not the standardized budget the threat model specifies. Fit on train
        only, like every other transform (leakage rule).
        """
        return self.X["train"].std(axis=0, ddof=0)


def prepare_regime(
    variant: str, regime: str, n_components: int | None = None, seed: int = 0
) -> Regime:
    """Load a variant and produce a leakage-safe feature view for all splits."""
    X, y, splits = load_processed(variant)
    pipe = build_preprocessor(regime, n_components=n_components, seed=seed)
    Z = fit_transform_splits(pipe, X, splits)
    ys = {s: y[splits[s]] for s in ("train", "val", "test")}
    angles, scaler = None, None
    if regime == "pca":
        scaler = AngleScaler().fit(Z["train"])
        angles = {s: scaler.transform(Z[s]) for s in ("train", "val", "test")}
    return Regime(
        name=regime, n_components=n_components, X=Z, y=ys, angles=angles, angle_scaler=scaler
    )


def kernel_subset(reg: Regime, cap: int, seed: int) -> np.ndarray:
    """Shared stratified train-subset indices for O(N^2) kernel work."""
    return stratified_cap(reg.X["train"], reg.y["train"], cap, seed)


def aggregate_seeds(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean/std over a list of per-seed metric dicts -> {metric: {mean,std}}."""
    keys = per_seed[0].keys()
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in per_seed], dtype=float)
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[k] = {"mean": float(vals.mean()), "std": std}
    return out
