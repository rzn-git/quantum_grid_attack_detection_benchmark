"""Shared fixtures + thread pinning for reproducibility tests."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # bit-stability for reproducibility test

import numpy as np
import pytest


@pytest.fixture
def toy_binary():
    rng = np.random.default_rng(0)
    n, d = 400, 16
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = (X @ w + 0.3 * rng.standard_normal(n) > 0).astype(int)
    return X, y


@pytest.fixture
def toy_splits():
    idx = np.arange(400)
    return {"train": idx[:240], "val": idx[240:320], "test": idx[320:]}
