"""R4-05 runner: unbiased-HSIC CKA and the RBF Gram, on tiny fixtures."""

import importlib.util
import itertools
import sys

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from qgridbench.utils.run_tracking import REPO_ROOT

# experiments/ is not an installed package; load the runner by path, with the repo root on
# sys.path for its `from experiments.run_length_scale_collapse import ...`
sys.path.insert(0, str(REPO_ROOT))
_spec = importlib.util.spec_from_file_location(
    "run_kernel_cka", REPO_ROOT / "experiments" / "run_kernel_cka.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
cka, hsic_unbiased, rbf_gram = _mod.cka, _mod.hsic_unbiased, _mod.rbf_gram


def _rows(n=40, d=3, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d))


def test_hsic_unbiased_is_the_u_statistic():
    # Song et al. (2012), Theorem 3: the estimator is the U-statistic over distinct
    # (i, j, q, r) of k_ij l_ij + k_ij l_qr - 2 k_ij l_iq
    K = rbf_gram(pdist(_rows(7, 2, 4), "sqeuclidean"), 0.4)
    L = _rows(7, 3, 5) @ _rows(7, 3, 5).T
    terms = [
        K[i, j] * L[i, j] + K[i, j] * L[q, r] - 2 * K[i, j] * L[i, q]
        for i, j, q, r in itertools.permutations(range(7), 4)
    ]
    assert hsic_unbiased(K, L) == pytest.approx(np.mean(terms), rel=1e-12)


def test_hsic_unbiased_ignores_the_diagonal():
    K = rbf_gram(pdist(_rows(), "sqeuclidean"), 0.5)
    L = rbf_gram(pdist(_rows(seed=1), "sqeuclidean"), 0.2)
    K2 = K + np.diag(np.arange(len(K), dtype=float))
    assert hsic_unbiased(K2, L) == pytest.approx(hsic_unbiased(K, L), rel=1e-12)


def test_rbf_gram_is_the_kernel_with_unit_diagonal():
    X = _rows(5)
    L = rbf_gram(pdist(X, "sqeuclidean"), 0.3)
    assert np.allclose(np.diag(L), 1.0)
    assert L[1, 3] == pytest.approx(np.exp(-0.3 * ((X[1] - X[3]) ** 2).sum()))


def test_cka_self_is_one_and_scale_invariant():
    L = rbf_gram(pdist(_rows(), "sqeuclidean"), 0.5)
    assert abs(cka(L, L) - 1) < 1e-12
    assert cka(L, 3.0 * L) == pytest.approx(1.0, abs=1e-12)


def test_shuffled_rows_drop_cka_for_a_smooth_kernel():
    X = _rows(400, 2, 3)
    L = rbf_gram(pdist(X, "sqeuclidean"), 0.1)
    p = np.random.default_rng(0).permutation(len(X))
    assert cka(L, L[p][:, p]) < 0.05


def test_shuffled_rows_drop_cka_for_a_nearly_diagonal_kernel():
    # the case that broke the full-Gram form: a row permutation keeps the shared diagonal
    X = _rows(400, 8, 6)
    L = rbf_gram(pdist(X, "sqeuclidean"), 0.5)
    p = np.random.default_rng(0).permutation(len(X))
    assert abs(cka(L, L[p][:, p])) < 0.05
