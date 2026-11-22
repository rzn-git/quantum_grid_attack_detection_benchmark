"""R4-11b runner: the deadband band rule, the band-zeroed sequence residual and the per-relay
impedance regression, on tiny synthetic phasor fixtures (no data, no models)."""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
A = np.exp(2j * np.pi / 3)


@pytest.fixture(scope="module")
def mod():
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
        yield importlib.import_module("experiments.run_consistency_defense_sequence")


def _relay_map(mod):
    s2 = mod.s2
    return s2.RelayMap(
        relay=1,
        ctrl_names=[f"c{i}" for i in range(14)],
        der_names=[
            f"R1-{n}"
            for n in [f"PM{k}" for k in range(7, 13)]
            + [f"PA{k}" for k in range(7, 13)]
            + ["PA:Z", "PA:ZH"]
        ],
        ctrl_idx=np.arange(14),
        der_idx=np.arange(14, 28),
        assign={7: "pos", 8: "neg", 9: "zero", 10: "pos", 11: "neg", 12: "zero"},
        deadband={7: 1.0, 8: 1000.0, 9: 1000.0, 10: 3.0, 11: 3.0, 12: 3.0},
        log_ratio=0.0,
        zh_offset=0.0,
        z_impute=1.0,
    )


def _rows(mod, rm, v1, i1, i2):
    """Phase phasors from sequence components (I0 = 0, balanced voltages), derived channels
    from the map itself, so every row is exactly consistent before any edit."""
    n = len(v1)
    V = np.stack([v1, A * A * v1, A * v1], axis=1)
    Iph = np.stack([i1 + i2, A * A * i1 + A * i2, A * i1 + A * A * i2], axis=1)
    P = np.concatenate([V, Iph], axis=1)
    xc = np.column_stack([np.abs(P), np.rad2deg(np.angle(P)), np.full(n, 60.0), np.zeros(n)])
    X = np.zeros((n, 28))
    X[:, :14] = xc
    X[:, 14:] = mod.s2.derived_np(xc, rm)
    return X


def test_band_rule_marks_near_threshold_mismatch_only(mod):
    rm = _relay_map(mod)
    X = _rows(
        mod,
        rm,
        v1=np.array([100.0, 100.0, 100.0]),
        i1=np.array([50.0, 50.0, 50.0]),
        i2=np.array([3.03, 5.0, 4.0]),  # |I2| 1% over, 67% over, 33% over t = 3
    )
    j = 4  # PM11 (k = 11), the negative-sequence current
    X[:2, 14 + j] = 0.0  # rows 0 and 1: the relay reports 0 though the map says on
    dis, band, gross = mod.band_state(X, rm)
    assert dis[:, j].tolist() == [True, True, False]
    assert dis[:, mod.N_SEQ + j].tolist() == [True, True, False]  # the angle partner
    assert band[:, j].tolist() == [True, False, False]
    assert gross[:, j].tolist() == [False, True, False]
    assert not dis[:, [0, 1, 2, 3, 5]].any()


def test_seq_residual_zeroes_band_mismatch_and_keeps_the_rest(mod, monkeypatch):
    rm = _relay_map(mod)
    X = _rows(
        mod,
        rm,
        v1=np.array([100.0, 100.0, 100.0]),
        i1=np.array([50.0, 50.0, 50.0]),
        i2=np.array([3.03, 5.0, 4.0]),
    )
    j = 4
    X[:2, 14 + j] = 0.0
    monkeypatch.setattr(mod, "RELAYS", [1])
    res = mod.seq_residuals(X, {1: rm}, {1: np.ones(14)})
    assert res.shape == (3, 1, 12)
    assert res[0, 0, j] == 0.0
    assert res[1, 0, j] == pytest.approx(5.0, rel=1e-12)
    assert res[2].max() < 1e-9  # an untouched consistent row scores 0
    assert mod.cd.row_score(res).tolist()[1] == pytest.approx(5.0, rel=1e-12)


def test_impedance_regression_recovers_log_linear_coefficients(mod):
    rm = _relay_map(mod)
    rng = np.random.default_rng(0)
    v1 = rng.uniform(80.0, 120.0, 40)
    i1 = rng.uniform(20.0, 80.0, 40)
    X = _rows(mod, rm, v1=v1 + 0j, i1=i1 + 0j, i2=np.zeros(40) + 0j)
    X[:, 14 + mod.cd.Z_COL] = np.exp(0.3 + 2.0 * np.log(v1) - 0.5 * np.log(i1))
    out = mod.impedance_regression(X[:30], X[30:], rm)
    assert np.allclose(out["beta_1_logV1_logI1"], [0.3, 2.0, -0.5], atol=1e-9)
    assert out["M1_regression"]["val_r2"] == pytest.approx(1.0, abs=1e-9)
