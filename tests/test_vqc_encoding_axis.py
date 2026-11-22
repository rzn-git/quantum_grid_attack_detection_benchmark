"""R4-13: the VQC encoding arms carry the deployed circuit at c = 1 and exact input gradients,
and the branch rule reads as registered (the `feature_map` tests are in tests/test_quantum.py)."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from qgridbench.models.quantum.vqc import VQClassifier


@pytest.fixture(scope="module")
def m():
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        yield importlib.import_module("experiments.run_vqc_encoding_axis")


def _xy(n=64, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, (n, d))
    return X, (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)


def test_angle_arm_at_c1_is_the_deployed_circuit(m):
    X, y = _xy()
    kw = dict(depth=2, n_classes=2, lr=0.05, batch_size=16, max_epochs=1, patience=5, seed=7)
    dep = VQClassifier(4, **kw).fit(X, y)
    arm = m.encoded_vqc_class()(4, encoding="angle_ry", bandwidth=1.0, reupload=False, **kw).fit(
        X, y
    )
    assert np.array_equal(dep.predict_proba(X), arm.predict_proba(X))


@pytest.mark.parametrize(("encoding", "reupload"), [("zz", False), ("angle_ry", True)])
def test_backprop_input_gradient_equals_parameter_shift(m, encoding, reupload):
    X, y = _xy(n=2)
    kw = dict(depth=2, n_classes=2, lr=0.05, batch_size=2, max_epochs=1, patience=1, seed=1)
    bp = m.encoded_vqc_class()(4, encoding=encoding, bandwidth=0.7, reupload=reupload, **kw)
    ps = m.encoded_vqc_class()(
        4, encoding=encoding, bandwidth=0.7, reupload=reupload, grad_method="parameter-shift", **kw
    )
    assert np.abs(bp.loss_input_grad(X, y) - ps.loss_input_grad(X, y)).max() < 1e-6


def _summary(clean, adv, loss_label, floor=0.415):
    return {
        "readable": clean - floor > 0.05,
        "retention_0.5": adv / clean,
        "end_state_2sigma": {"loss": {"label": loss_label}},
    }


def test_branch_rules(m):
    general = {"A0": _summary(0.50, 0.03, "inverted"), "A2": _summary(0.55, 0.10, "partial")}
    assert m.branch(general)["verdict"] == "ENCODING-GENERAL"
    by_retention = {**general, "A3": _summary(0.55, 0.20, "inverted")}
    assert m.branch(by_retention)["verdict"] == "ENCODING-SPECIFIC"
    by_disabled = {**general, "A1": _summary(0.55, 0.02, "disabled")}
    assert m.branch(by_disabled)["arms_driving_specific"] == ["A1"]
    unreadable = {**general, "A1": _summary(0.44, 0.40, "disabled")}
    b = m.branch(unreadable)
    assert b["verdict"] == "ENCODING-GENERAL" and b["unreadable_arms"] == ["A1"]


def test_negative_control_blocks_pgd_only(m):
    r2 = {"pgd": {"0.1": 0.30, "0.5": 0.20}, "fgsm": {"0.1": 0.30, "0.5": 0.55}}
    nc = m.negative_control(0.52, r2)
    assert nc == {"pgd_above_clean": [], "fgsm_overshoot_above_clean": ["0.5"]}
    r2["pgd"]["0.5"] = 0.53
    assert m.negative_control(0.52, r2)["pgd_above_clean"] == ["0.5"]
