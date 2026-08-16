"""Seed reproducibility: identical seed -> metrics within tolerance (not bit-exact,
multi-threaded libs; OMP_NUM_THREADS pinned in conftest)."""

import numpy as np

from qgridbench.eval.metrics import compute_all
from qgridbench.models.classical.zoo import build, fit_model
from qgridbench.models.quantum.vqc import VQClassifier
from qgridbench.utils.seeding import set_all_seeds


def _run_xgb(seed, toy_binary, toy_splits):
    set_all_seeds(seed)
    X, y = toy_binary
    tr, te = toy_splits["train"], toy_splits["test"]
    est = build(
        "xgb",
        {
            "n_estimators": 80,
            "learning_rate": 0.1,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        },
        seed,
    )
    fit_model("xgb", est, X[tr], y[tr])
    return compute_all(y[te], est.predict_proba(X[te]))


def test_xgb_reproducible(toy_binary, toy_splits):
    a = _run_xgb(3, toy_binary, toy_splits)
    b = _run_xgb(3, toy_binary, toy_splits)
    for k in a:
        assert abs(a[k] - b[k]) <= 1e-6 * (abs(a[k]) + 1e-9) + 1e-9


def test_vqc_reproducible_and_trains(toy_binary, toy_splits):
    X, y = toy_binary
    tr, va = toy_splits["train"], toy_splits["val"]

    def run():
        set_all_seeds(0)
        m = VQClassifier(
            n_qubits=X.shape[1], depth=2, n_classes=2, max_epochs=4, batch_size=64, seed=0
        )
        m.fit(X[tr], y[tr], X[va], y[va])
        return m

    m1, m2 = run(), run()
    w1 = m1.weights.detach().numpy()
    w2 = m2.weights.detach().numpy()
    assert np.allclose(w1, w2, atol=1e-6)
    # gradient-norm history recorded (trainability logging)
    assert len(m1.history["grad_norm_var"]) >= 1
    assert all(np.isfinite(m1.history["grad_norm_mean"]))
