"""Central seed handling: one config seed drives python, numpy, and model libs.

Every experiment entrypoint calls set_all_seeds(seed) exactly once per run and
threads the returned Generator (or the seed itself) into every stochastic
component (sklearn/xgboost/lightgbm random_state, PennyLane init, attack RNG).
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEEDS = list(range(10))  # final-evaluation protocol: seeds {0..9}
TUNING_SEEDS = [0, 1, 2]  # hyperparameter search protocol (see CLAUDE.md section 5)


def set_all_seeds(seed: int) -> np.random.Generator:
    """Seed python, numpy legacy, and hashing; return a dedicated Generator."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


def spawn_rng(seed: int, stream: str) -> np.random.Generator:
    """Independent, reproducible RNG stream derived from (seed, stream-name).

    Use for per-worker / per-component randomness so parallel execution stays
    order-independent and reproducible.
    """
    ss = np.random.SeedSequence([seed, int.from_bytes(stream.encode(), "little") % (2**63)])
    return np.random.default_rng(ss)
