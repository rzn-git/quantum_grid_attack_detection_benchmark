"""Data schema + split determinism + leakage-rule tests."""

import numpy as np

from qgridbench.features.reduce import (
    ANGLE_RANGE,
    AngleScaler,
    build_preprocessor,
    fit_transform_splits,
)


def test_split_determinism_and_disjointness():
    """Fixed-seed stratified split is identical across calls and non-overlapping."""
    from sklearn.model_selection import train_test_split

    y = np.repeat([0, 1], [300, 100])
    idx = np.arange(len(y))

    def split():
        tr, tmp = train_test_split(idx, train_size=0.6, stratify=y, random_state=1337)
        va, te = train_test_split(tmp, train_size=0.5, stratify=y[tmp], random_state=1337)
        return set(tr), set(va), set(te)

    a = split()
    b = split()
    assert a == b, "split not deterministic under fixed seed"
    tr, va, te = a
    assert tr & va == set() and tr & te == set() and va & te == set()
    assert len(tr) + len(va) + len(te) == len(y)


def test_leakage_pca_fit_on_train_only(toy_binary, toy_splits):
    """PCA/scaler components must depend only on train rows (leakage rule)."""
    X, _ = toy_binary
    pipe = build_preprocessor("pca", n_components=8, seed=0)
    _ = fit_transform_splits(pipe, X, toy_splits)
    comps_full = pipe.named_steps["pca"].components_.copy()

    # refit on train rows alone; components must match exactly
    pipe2 = build_preprocessor("pca", n_components=8, seed=0)
    pipe2.fit(X[toy_splits["train"]])
    assert np.allclose(comps_full, pipe2.named_steps["pca"].components_)

    # perturbing test rows must not change the fitted transform
    X_pert = X.copy()
    X_pert[toy_splits["test"]] += 100.0
    pipe3 = build_preprocessor("pca", n_components=8, seed=0)
    pipe3.fit(X_pert[toy_splits["train"]])
    assert np.allclose(comps_full, pipe3.named_steps["pca"].components_)


def test_angle_encoding_range(toy_binary, toy_splits):
    """Angle-scaled inputs must lie within [-pi, pi] (encoding-range rule)."""
    X, _ = toy_binary
    pipe = build_preprocessor("pca", n_components=8, seed=0)
    Z = fit_transform_splits(pipe, X, toy_splits)
    scaler = AngleScaler().fit(Z["train"])
    for split in ("train", "val", "test"):
        A = scaler.transform(Z[split])
        assert A.min() >= ANGLE_RANGE[0] - 1e-9
        assert A.max() <= ANGLE_RANGE[1] + 1e-9


def test_missing_indicator_is_opt_in_and_fit_on_train(toy_splits):
    """Default transform is unchanged; add_indicator appends one column per feature that is
    missing somewhere in TRAIN, fitted on train rows only (R4-12 missingness control)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 4))
    X[toy_splits["train"][::3], 1] = np.nan  # feature 1 missing in train
    X[toy_splits["test"][::2], 2] = np.nan  # feature 2 missing in test only
    base = fit_transform_splits(build_preprocessor("full"), X, toy_splits)
    ind = fit_transform_splits(build_preprocessor("full", add_indicator=True), X, toy_splits)
    assert base["train"].shape[1] == 4 and ind["train"].shape[1] == 5
    np.testing.assert_array_equal(base["test"], ind["test"][:, :4])
    np.testing.assert_array_equal(
        ind["train"][:, 4] > ind["train"][:, 4].mean(), np.isnan(X[toy_splits["train"], 1])
    )


def test_pca_train_features_equal_transform_of_train():
    """Regression (decision log 2026-11-18, R4-22 incident): on a wide input PCA's "auto"
    solver is randomized, whose fit_transform returns the approximate U*S, so the train
    features differed from transform(train) (0.27 on fdia118_binary) while val and test went
    through transform. The shipped transform pins covariance_eigh: one frozen map on every
    split, bit for bit."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    X = (rng.standard_normal((600, 120)) * np.linspace(3.0, 0.1, 120)).astype(np.float32)
    auto = PCA(n_components=8, random_state=0)
    gap = np.abs(auto.fit_transform(X) - auto.transform(X)).max()
    assert auto._fit_svd_solver == "randomized" and gap > 1e-4  # the hazard is present here
    pipe = build_preprocessor("pca", n_components=8, seed=0)
    Z = pipe.fit_transform(X)
    assert pipe.named_steps["pca"]._fit_svd_solver == "covariance_eigh"
    np.testing.assert_array_equal(Z, pipe.transform(X))
