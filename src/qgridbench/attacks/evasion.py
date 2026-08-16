"""Evasion attacks: FGSM and PGD (hand-rolled, L-inf, feature-space).

COMPARABILITY CONTRACT (CLAUDE.md section 7): attacks operate in the
standardized PRE-ENCODING feature space shared by every model family in the
comparison set. Epsilon-boundedness is asserted by tests/test_attacks.py.
No clipping to a data box is applied: the space is standardized (unbounded),
and the physical-plausibility caveat is a stated limitation.

SCALE CONTRACT (why `scale` is a required argument, not an optional one):
the comparison space is PCA output, whose per-feature variance is the component
EIGENVALUE, not 1 -- measured std runs 2.0-5.6 on this dataset even though the
pre-PCA features are standardized. Treating raw PCA coordinates as if they were
standardized silently shrinks the budget by ~3.2x (eps=0.3 is 0.09 sigma), which
is the near-degenerate regime where smooth models appear robust for free. Every
attack therefore takes the TRAIN per-feature std explicitly and bounds feature j
by eps * scale[j]: an L-inf ball in standardized units, expressed in PCA
coordinates. It is required rather than defaulted so the mistake cannot recur
silently -- passing np.ones(d) is the deliberate way to ask for raw units.
"""

from __future__ import annotations

import numpy as np


def _as_scale(scale: np.ndarray, n_features: int) -> np.ndarray:
    s = np.asarray(scale, dtype=float)
    if s.shape != (n_features,):
        raise ValueError(f"scale must have shape ({n_features},), got {s.shape}")
    if not np.all(s > 0):
        raise ValueError("scale must be strictly positive")
    return s


def fgsm(grad_fn, X: np.ndarray, y: np.ndarray, eps: float, scale: np.ndarray) -> np.ndarray:
    """Fast Gradient Sign Method: single step of size eps, in units of `scale`."""
    s = _as_scale(scale, X.shape[1])
    g = grad_fn(X, y)
    return X + eps * s * np.sign(g)


def pgd(
    grad_fn,
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    scale: np.ndarray,
    n_steps: int = 10,
    step_frac: float = 0.25,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Projected Gradient Descent in the L-inf ball of radius eps * scale.

    Random start inside the ball, step size = step_frac * eps * scale, projection
    after every step. Deterministic given rng.
    """
    s = _as_scale(scale, X.shape[1])
    rng = rng or np.random.default_rng(0)
    alpha = step_frac * eps * s
    lo, hi = X - eps * s, X + eps * s
    X_adv = X + rng.uniform(-1.0, 1.0, size=X.shape) * eps * s
    for _ in range(n_steps):
        g = grad_fn(X_adv, y)
        X_adv = np.clip(X_adv + alpha * np.sign(g), lo, hi)  # L-inf projection
    return X_adv


def gaussian_sensor_noise(
    X: np.ndarray, snr_db: float, rng: np.random.Generator, scale: np.ndarray
) -> np.ndarray:
    """Benign additive measurement noise at a target per-feature SNR.

    Signal power per feature is scale[j]**2 (see SCALE CONTRACT above), so hitting
    a target SNR needs noise_std[j] = scale[j] * 10**(-snr_db/20). Assuming unit
    signal power on PCA outputs would mislabel every SNR level by ~10 dB.
    """
    s = _as_scale(scale, X.shape[1])
    return X + rng.normal(0.0, 1.0, size=X.shape) * s * 10.0 ** (-snr_db / 20.0)
