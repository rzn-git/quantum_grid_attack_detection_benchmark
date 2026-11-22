"""R4-06 magnitude unit of the consistent relay attack runner: the physical unit's row rule,
the unit written into the attack box, the scale-equal identity and the registered branch, on
tiny synthetic fixtures (no data, no models)."""

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")


@pytest.fixture(scope="module")
def cra():
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
        yield importlib.import_module("experiments.run_consistent_relay_attack")


def _relay_map(cra, deadband7=50.0):
    """One relay on 28 columns: controlled 0..13 (PM1..PM6, PA1..PA6, F, DF), derived 14..27."""
    return cra.RelayMap(
        relay=1,
        ctrl_names=[f"c{i}" for i in range(14)],
        der_names=[f"d{i}" for i in range(14)],
        ctrl_idx=np.arange(14),
        der_idx=np.arange(14, 28),
        assign={7: "pos", 8: "neg", 9: "zero", 10: "pos", 11: "neg", 12: "zero"},
        deadband={7: deadband7, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0},
        log_ratio=0.0,
        zh_offset=0.0,
        z_impute=1.0,
    )


def _train_voltages():
    v = np.array(
        [
            [100.0, 110.0, 120.0],  # in
            [102.0, 108.0, 125.0],  # in
            [0.0, 0.0, 0.0],  # outage: out
            [100.0, 0.0, 120.0],  # one phase out: out
            [np.nan, 100.0, 100.0],  # missing: out
            [104.0, 112.0, 121.0],  # in
            [50.0, 200.0, 200.0],  # at the deadband, not over it: out
        ]
    )
    X = np.ones((len(v), 28))
    X[:, :3] = v
    return X


def test_physical_unit_is_std_over_rows_with_every_voltage_over_the_deadband(cra):
    rm, X = _relay_map(cra), _train_voltages()
    s = np.arange(1.0, 29.0) * 1000.0
    unit, diag = cra.fit_magnitude_unit(X, rm, s, "physical")
    expect = X[[0, 1, 5], :3].std(axis=0)
    np.testing.assert_array_equal(unit, expect)
    assert diag["n_rows_used"] == 3
    assert diag["n_rows_any_voltage_nan"] == 1
    assert diag["n_rows_all_voltages_at_or_under_deadband"] == 1
    assert diag["n_rows_some_voltages_at_or_under_deadband"] == 2
    assert diag["scaler_scale_V"] == [1000.0, 2000.0, 3000.0]
    assert diag["pm7_deadband_V"] == 50.0


def test_scale_equal_returns_scaler_scale_and_rejects_scale(cra):
    rm, X = _relay_map(cra), _train_voltages()
    s = np.arange(1.0, 29.0) * 1000.0
    unit, diag = cra.fit_magnitude_unit(X, rm, s, "scale-equal")
    np.testing.assert_array_equal(unit, s[:3])
    np.testing.assert_array_equal(diag["physical_unit_V"], X[[0, 1, 5], :3].std(axis=0))
    with pytest.raises(ValueError, match="no fitted unit"):
        cra.fit_magnitude_unit(X, rm, s, "scale")


def test_physical_unit_fails_loud_without_operating_rows(cra):
    rm, X = _relay_map(cra, deadband7=1e6), _train_voltages()
    with pytest.raises(AssertionError, match="have every voltage over"):
        cra.fit_magnitude_unit(X, rm, np.ones(28), "physical")


def _space(cra, mag_unit):
    rng = np.random.default_rng(0)
    X0 = np.concatenate([rng.uniform(50.0, 150.0, (5, 6)), rng.uniform(-90, 90, (5, 6))], 1)
    X0 = np.concatenate([X0, rng.uniform(59.0, 61.0, (5, 2)), rng.uniform(1.0, 9.0, (5, 14))], 1)
    sh = SimpleNamespace(
        X0raw=X0, X0imp=X0, s=np.arange(1.0, 29.0) * 10.0, angle_unit="marginal", ctx=None
    )
    rm = _relay_map(cra, deadband7=1.0)
    rm.mag_unit = mag_unit
    return cra.AttackSpace("consistent", rm, sh, "cpu"), sh


def test_physical_unit_replaces_only_the_voltage_magnitudes_in_the_box(cra):
    unit = np.array([3.0, 4.0, 5.0])
    sp, sh = _space(cra, unit)
    np.testing.assert_array_equal(sp.sc[:3], unit)
    np.testing.assert_array_equal(sp.sc[3:], sh.s[3:14])
    np.testing.assert_array_equal(sp.floor_v[:, :3], -sp.x0c[:, :3] / unit)
    np.testing.assert_array_equal(sp.floor_v[:, 3:], -sp.x0c[:, 3:6] / sh.s[3:6])


def test_scale_equal_unit_is_bit_identical_to_the_default_path(cra):
    base, sh = _space(cra, None)
    eq, _ = _space(cra, sh.s[:3].copy())
    np.testing.assert_array_equal(eq.sc, base.sc)
    np.testing.assert_array_equal(eq.floor_v, base.floor_v)
    for a, b in zip(eq.box(0.5), base.box(0.5), strict=True):
        np.testing.assert_array_equal(a, b)


def test_magnitude_verdict_is_strict_and_needs_every_eps(cra):
    assert cra.magnitude_verdict({0.1: 0.19, 0.25: 0.21, 0.5: 0.41}) == "SURVIVES"
    assert cra.magnitude_verdict({0.1: 0.19, 0.25: 0.21, 0.5: 0.415}) == "BUDGET-BOUND"
    assert cra.magnitude_verdict({0.1: 0.50, 0.25: 0.21, 0.5: 0.29}) == "BUDGET-BOUND"
    assert cra.MAG_UNIT_FLOOR == 0.415
    with pytest.raises(ValueError, match="no QSVM"):
        cra.magnitude_verdict({})


def test_crossing_rows_sums_relays_and_seeds_and_skips_projection(cra):
    def chk(eps, src, n):
        rel = {"deadband_crossing": {"n_rows_any": n, "n_attackable_rows": 10}}
        return {"mode": "joint", "eps": eps, "src": src, "by_relay": {"R1": rel, "R2": rel}}

    checks = {
        "s0": [chk(0.0, "projection_only", 9), chk(0.5, "qsvm", 3), chk(0.5, "random", 1)],
        "s1": [chk(0.5, "qsvm", 2)],
    }
    assert cra.crossing_rows(checks) == {"0.5|qsvm": [10, 40], "0.5|random": [2, 20]}
