"""Label-noise poisoning: flip a fraction of TRAINING labels, retrain, measure.

Flips are uniform-random over the training set (seeded); for binary the label
is inverted, for multiclass a different class is drawn uniformly.
"""

from __future__ import annotations

import numpy as np


def flip_labels(y: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of y with `rate` fraction of labels flipped."""
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"rate must be in [0,1): {rate}")
    y_out = np.asarray(y).copy()
    n_flip = int(round(rate * len(y_out)))
    if n_flip == 0:
        return y_out
    idx = rng.choice(len(y_out), size=n_flip, replace=False)
    classes = np.unique(y_out)
    if len(classes) == 2:
        y_out[idx] = classes.sum() - y_out[idx]  # invert binary labels
    else:
        for i in idx:
            others = classes[classes != y_out[i]]
            y_out[i] = rng.choice(others)
    return y_out
