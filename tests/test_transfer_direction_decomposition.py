"""E-NS8(b) runner: the projectors, the smoothed-FD estimator, the null and the branch reading
on tiny synthetic fixtures (no data, no models)."""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")


@pytest.fixture(scope="module")
def ns8():
    """The runner reads --threads from argv at import and sets the thread variables; both are
    restored after the module's tests."""
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        mp.setattr(sys, "argv", ["pytest", "--threads", "1"])
        for v in THREAD_VARS:
            if v in os.environ:
                mp.setenv(v, os.environ[v])
            else:
                mp.delenv(v, raising=False)
        yield importlib.import_module("experiments.run_transfer_direction_decomposition")


def _tangent(rng, n, k):
    """M = [I; J] per row, as the runner builds it (controlled rows first)."""
    return np.concatenate([np.repeat(np.eye(k)[None], n, 0), rng.normal(size=(n, k, k))], 1)


def test_projectors_span_range_and_complement(ns8):
    M = _tangent(np.random.default_rng(0), 3, 4)
    q_on, q_off, ctl = ns8.projectors(M)
    assert (ctl["dim_on"], ctl["dim_off"]) == (4, 4)
    P = q_on @ np.swapaxes(q_on, 1, 2)
    np.testing.assert_allclose(P @ M, M, atol=1e-12)
    np.testing.assert_allclose(
        P + q_off @ np.swapaxes(q_off, 1, 2), np.eye(8)[None].repeat(3, 0), atol=1e-12
    )


def test_projectors_raise_on_rank_deficient_tangent(ns8):
    M = _tangent(np.random.default_rng(1), 2, 3)
    M[:, :, 2] = M[:, :, 1]
    with pytest.raises(AssertionError, match="rank-deficient"):
        ns8.projectors(M)


def test_subspace_cos_matches_projection_and_flags_zero(ns8):
    rng = np.random.default_rng(2)
    q_on, _, _ = ns8.projectors(_tangent(rng, 4, 3))
    a, b = rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
    b[3] = 0.0
    got = ns8.subspace_cos(q_on, a, b)
    P = q_on @ np.swapaxes(q_on, 1, 2)
    pa, pb = np.einsum("nij,nj->ni", P[:3], a[:3]), np.einsum("nij,nj->ni", P[:3], b[:3])
    want = (pa * pb).sum(1) / np.linalg.norm(pa, axis=1) / np.linalg.norm(pb, axis=1)
    np.testing.assert_allclose(got[:3], want, atol=1e-12)
    assert np.isnan(got[3])


def test_smoothed_estimator_recovers_a_linear_gradient(ns8):
    """f(Z) = c . Z with Z = Z0 + A u: the true gradient in u is A^T c, in row(A)."""
    rng = np.random.default_rng(3)
    A, c = rng.normal(size=(3, 20)), rng.normal(size=3)
    PA = ns8.row_space_projector(A)
    np.testing.assert_allclose(PA @ PA, PA, atol=1e-12)
    U = rng.standard_normal((2, 2048, 20))
    out = ns8.smoothed_grads(lambda z: z @ c, rng.normal(size=(2, 3)), U, A, PA, 0.3, [256, 2048])
    true = np.repeat((A.T @ c)[None], 2, 0)
    proj, plain = out[2048]
    assert ns8.row_cos(proj, true).min() > 0.99
    assert ns8.row_cos(proj, true).min() > ns8.row_cos(plain, true).min()
    assert ns8.row_cos(out[256][0], true).min() > 0.9


def test_null_p95_matches_the_dimension(ns8):
    """Random pairs in a k-dimensional subspace: cosine sd 1/sqrt(k), p95 about 1.645/sqrt(k)."""
    q_on, q_off, _ = ns8.projectors(_tangent(np.random.default_rng(4), 8, 56))
    nul = ns8.null_cosines(q_on, q_off, np.random.default_rng(5), 500)
    for k in ("on", "off"):
        assert abs(np.percentile(nul[k], 95) - 1.645 / np.sqrt(56)) < 0.02


def _stats(ns8, on_off):
    s = {t: {"c_on": a, "c_off": b} for t, (a, b) in zip(ns8.TREES, on_off, strict=True)}
    s["logreg"] = {"c_on": 0.5, "c_off": 0.1}
    return s


@pytest.mark.parametrize(
    "on_off,verdict",
    [
        ([(0.1, 0.4), (0.1, 0.3), (0.5, 0.1)], "HYPOTHESIS"),
        ([(0.1, 0.4), (0.1, 0.12), (0.5, 0.1)], "INTERMEDIATE"),  # one tree under the margin
        ([(0.1, 0.4), (0.0, 0.2), (0.5, 0.1)], "INTERMEDIATE"),  # c_off under the null
        ([(0.4, 0.1), (0.3, 0.3), (0.5, 0.6)], "KILL"),  # ties count as c_on >= c_off
    ],
)
def test_branch_reading(ns8, on_off, verdict):
    b = ns8.branch(_stats(ns8, on_off), {"on": 0.22, "off": 0.22})
    assert b["verdict"] == verdict
    assert b["logreg_triangulation"]["pass"]
