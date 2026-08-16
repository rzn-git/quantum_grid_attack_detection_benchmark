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
