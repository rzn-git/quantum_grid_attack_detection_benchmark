"""E-NS2 runner: gamma_eff fit, P10a's interpolant and leave-one-gamma-out, on tiny fixtures."""

import importlib.util

import numpy as np
import pytest
from scipy.optimize import nnls
from scipy.spatial.distance import pdist

from qgridbench.utils.run_tracking import REPO_ROOT

# experiments/ is not an installed package; load the runner by path (it imports only qgridbench)
_spec = importlib.util.spec_from_file_location(
    "run_length_scale_collapse", REPO_ROOT / "experiments" / "run_length_scale_collapse.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fit_gamma_eff, interpolant, logo, predict = (
    _mod.fit_gamma_eff,
    _mod.interpolant,
    _mod.logo,
    _mod.predict,
)


def test_fit_gamma_eff_recovers_rbf_gamma():
    X = np.random.default_rng(0).standard_normal((30, 4))
    d2 = pdist(X, "sqeuclidean")
    fit = fit_gamma_eff(np.exp(-0.37 * d2), d2)
    assert fit["gamma_eff"] == pytest.approx(0.37, rel=1e-12)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["n_pairs"] == 30 * 29 // 2


def test_fit_gamma_eff_fails_loud_on_zero_kernel():
    with pytest.raises(ValueError, match="k <= 0"):
        fit_gamma_eff(np.array([0.5, 0.0]), np.array([1.0, 2.0]))


def test_fit_gamma_loc_uses_near_pairs_only():
    X = np.random.default_rng(1).standard_normal((40, 3))
    d2 = pdist(X, "sqeuclidean")
    k = np.exp(-0.37 * d2)
    # a floor that sets the all-pairs slope but leaves every k >= 0.5 pair untouched
    k_floored = np.where(k < 0.5, np.maximum(k, 0.2), k)
    fit = _mod.fit_gamma_loc(k_floored, d2)
    assert fit["gamma_eff"] == pytest.approx(0.37, rel=1e-12)
    assert fit["n_near_pairs"] == int((k >= 0.5).sum()) == fit["n_pairs"]
    assert fit["few_near_pairs"]  # 40 rows give far fewer than 1,000 near pairs
    assert fit_gamma_eff(k_floored, d2)["gamma_eff"] < 0.37


def test_fit_gamma_loc_fails_loud_without_near_pairs():
    with pytest.raises(ValueError, match="near-pair slope is undefined"):
        _mod.fit_gamma_loc(np.array([0.4, 0.6, 0.1]), np.array([1.0, 0.5, 3.0]))


def test_rbf_self_control_near_pairs_skips_no_underflowing_gamma():
    rng = np.random.default_rng(2)
    base = rng.standard_normal((30, 3)) * 10.0
    X = np.vstack([base, base + 0.01 * rng.standard_normal((30, 3))])  # 30 near twins
    out = _mod.rbf_self_control(X, {"wide": 0.001, "narrow": 50.0}, "near-pairs")
    assert out["n_tested"] == 2 and out["passed"]
    assert _mod.rbf_self_control(X, {"narrow": 50.0})["rows"]["narrow"].get("skipped")


def _ard_fixture(gam, seed=3, n=50):
    X = np.random.default_rng(seed).standard_normal((n, len(gam)))
    D = (X[:, None, :] - X[None, :, :]) ** 2
    return X, D, np.exp(-(D * np.asarray(gam)).sum(-1))


def test_fit_gamma_ard_recovers_a_per_dimension_gamma():
    gam = [0.05, 0.3, 0.0, 0.8]
    X, _, K = _ard_fixture(gam)
    sigma = np.array([1.0, 2.0, 3.0, 0.5])
    fit = _mod.fit_gamma_ard(X, K, sigma)
    assert fit["gamma_j"] == pytest.approx(gam, abs=1e-10)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["gamma_eff"] == pytest.approx((np.array(gam) * sigma**2).sum() / (sigma**2).sum())
    iu = np.triu_indices(len(X), 1)
    assert fit["n_near_pairs"] == int((K[iu] >= 0.5).sum())


def test_fit_gamma_ard_equals_nnls_on_the_full_design():
    X, D, K = _ard_fixture([0.2, 0.2, 0.2], seed=4, n=60)
    K = K ** (1 + 0.3 * np.random.default_rng(5).random(K.shape))  # off-model, still in (0, 1]
    K = np.triu(K, 1) + np.triu(K, 1).T + np.eye(len(K))
    iu = np.triu_indices(len(X), 1)
    near = K[iu] >= 0.5
    A, y = D[iu][near], -np.log(K[iu][near])
    want, _ = nnls(A, y)
    fit = _mod.fit_gamma_ard(X, K, np.ones(3))
    assert fit["gamma_j"] == pytest.approx(want.tolist(), rel=1e-8, abs=1e-12)
    resid = y - A @ want
    assert fit["r2"] == pytest.approx(1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum())


def test_fit_gamma_ard_fails_loud_with_fewer_near_pairs_than_dimensions():
    X = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 0.0]])
    K = np.full((3, 3), 0.1) + 0.9 * np.eye(3)
    with pytest.raises(ValueError, match="undetermined"):
        _mod.fit_gamma_ard(X, K, np.ones(2))


def test_rbf_self_control_ard_returns_gamma_in_every_dimension():
    rng = np.random.default_rng(2)
    base = rng.standard_normal((30, 3)) * 3.0
    X = np.vstack([base, base + 0.05 * rng.standard_normal((30, 3))])
    out = _mod.rbf_self_control(X, {"a": 0.01, "b": 2.0, "c": 1e4}, "ard", np.ones(3))
    assert out["passed"] and out["per_dim_passed"] and out["n_tested"] == 2
    assert out["rows"]["b"]["rel_error_per_dim_max"] < 1e-8
    assert "near pairs < 3 gamma_j" in out["rows"]["c"]["skipped"]


def test_interpolant_averages_duplicate_x():
    xu, yu = interpolant(np.array([1.0, 0.0, 1.0]), np.array([2.0, 5.0, 4.0]))
    assert xu.tolist() == [0.0, 1.0]
    assert yu.tolist() == [5.0, 3.0]


def _collapsed_cells(gammas, radii, f):
    """Cells whose targets depend on (gamma, eps) only through u = gamma * eps^2."""
    return [
        {"gamma": g, "eps": e, "f1_minus_floor": f(g * e**2), "kernel_mass_ratio": f(g * e**2)}
        for g in gammas
        for e in radii
    ]


def test_logo_is_near_zero_on_a_perfect_collapse():
    cells = _collapsed_cells([0.1, 0.2, 0.4, 0.8], [0.5, 1.0, 2.0], lambda u: np.exp(-u))
    out = logo(cells, 1.0, "kernel_mass_ratio")
    assert out["n"] == 12
    assert out["median_abs_error"] < 0.03


def test_predict_hits_the_curve_and_flags_the_hull():
    rbf = _collapsed_cells([0.1, 1.0], [0.5, 1.0, 2.0], lambda u: -u)
    fid = [
        {"bandwidth": 1.0, "eps": 1.0, "f1_minus_floor": -1.0, "kernel_mass_ratio": -1.0},
        {"bandwidth": 1.0, "eps": 10.0, "f1_minus_floor": -100.0, "kernel_mass_ratio": -100.0},
    ]
    out = predict(rbf, 1.0, fid, {1.0: 1.0})
    inside, outside = out["cells"]
    assert inside["f1_minus_floor_abs_error"] == pytest.approx(0.0)
    assert not inside["outside_hull"]
    assert outside["outside_hull"] and outside["f1_minus_floor_predicted"] == pytest.approx(-4.0)
    assert out["frac_outside_hull"] == 0.5
