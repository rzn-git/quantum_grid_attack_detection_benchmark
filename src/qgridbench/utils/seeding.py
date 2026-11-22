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
TUNING_SEEDS = [0, 1, 2]  # hyperparameter search protocol (see study protocol §5)


def set_all_seeds(seed: int) -> np.random.Generator:
    """Seed python, numpy legacy, torch, and hashing; return a dedicated Generator.

    torch is imported inside the call, not at module scope: this module is imported by every
    entrypoint while torch costs ~1.5 s to load, and the cost belongs to callers that actually
    seed. The import is unconditional — no try/except — so a missing torch fails loud rather
    than silently leaving an RNG unseeded, which is the defect this function exists to prevent.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # covers CUDA devices too (torch delegates to cuda.manual_seed_all)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


def spawn_rng(seed: int, stream: str) -> np.random.Generator:
    """Reproducible RNG stream derived from (seed, stream-name).

    Use for per-worker / per-component randomness so parallel execution stays
    order-independent and reproducible.

    KNOWN LIMIT, kept because every stored run depends on these streams: only the first eight
    ASCII characters of the name enter the seed (the modulo keeps the low 63 bits of a
    little-endian integer). Two names that share their first eight characters are ONE stream:
    "poison_0.05" and "poison_0.2" draw the same numbers. A new stream name must differ from
    every other name within its first eight characters. Do not change this function: that would
    move every seeded result (tests/test_seeding_streams.py fails if it is changed).
    """
    ss = np.random.SeedSequence([seed, int.from_bytes(stream.encode(), "little") % (2**63)])
    return np.random.default_rng(ss)
