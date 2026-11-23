"""The two scalers of the angle-scaling control and the registered reading of its scores."""

import numpy as np
import pytest

from experiments.run_angle_scaling import THRESHOLD, central_share, lead_reading, verdict
from qgridbench.features.reduce import (
    ANGLE_RANGE,
    AngleScaler,
    ClippedAngleScaler,
    QuantileAngleScaler,
)


@pytest.fixture
def heavy_tailed():
    rng = np.random.default_rng(0)
    train = rng.standard_t(df=2, size=(4000, 3))
    other = rng.standard_t(df=2, size=(1000, 3)) * 3.0  # reaches beyond the train range
    return train, other


@pytest.mark.parametrize("scaler_cls", [ClippedAngleScaler, QuantileAngleScaler])
def test_angles_stay_in_range_on_unseen_rows(scaler_cls, heavy_tailed):
    train, other = heavy_tailed
    scaler = scaler_cls().fit(train)
    for Z in (train, other):
        a = scaler.transform(Z)
        assert a.min() >= ANGLE_RANGE[0] and a.max() <= ANGLE_RANGE[1]


@pytest.mark.parametrize("scaler_cls", [ClippedAngleScaler, QuantileAngleScaler])
def test_fit_uses_the_training_rows_only(scaler_cls, heavy_tailed):
    train, other = heavy_tailed
    alone = scaler_cls().fit(train).transform(train)
    scaler = scaler_cls().fit(train)
    scaler.transform(other)  # transforming other rows must not change the fitted map
    assert np.array_equal(alone, scaler.transform(train))


@pytest.mark.parametrize("scaler_cls", [ClippedAngleScaler, QuantileAngleScaler])
def test_map_is_monotone_per_component(scaler_cls, heavy_tailed):
    train, _ = heavy_tailed
    scaler = scaler_cls().fit(train)
    grid = np.tile(np.linspace(-50.0, 50.0, 2001)[:, None], (1, train.shape[1]))
    assert np.all(np.diff(scaler.transform(grid), axis=0) >= 0)


@pytest.mark.parametrize("scaler_cls", [ClippedAngleScaler, QuantileAngleScaler])
def test_bulk_fills_more_of_the_range_than_under_minmax(scaler_cls, heavy_tailed):
    train, _ = heavy_tailed
    shipped = central_share(AngleScaler().fit(train).transform(train))
    spread = central_share(scaler_cls().fit(train).transform(train))
    assert np.all(spread > shipped)


def _scores(zz_clip_test, zz_clip_val):
    base = {"test": 0.56, "val": 0.57}
    flat = {"test": 0.56, "val": 0.57}
    return {
        "zz": {
            "minmax": base,
            "clip": {"test": zz_clip_test, "val": zz_clip_val},
            "quantile": flat,
        },
        "angle_ry": {"minmax": base, "clip": flat, "quantile": flat},
    }


def test_verdict_needs_test_gain_over_threshold_and_validation_gain():
    assert verdict(_scores(0.56 + THRESHOLD + 0.001, 0.58))["branch"] == "SCALING_MATTERS"
    # not counted: a test gain under the threshold, one without a validation gain, a loss
    assert verdict(_scores(0.56 + THRESHOLD - 0.001, 0.58))["branch"] == "NULL_SURVIVES"
    assert verdict(_scores(0.60, 0.57))["branch"] == "NULL_SURVIVES"
    assert verdict(_scores(0.50, 0.50))["branch"] == "NULL_SURVIVES"


def test_verdict_reports_every_difference():
    out = verdict(_scores(0.50, 0.50))
    assert len(out["differences"]) == 4
    assert out["smallest_d_test"] == pytest.approx(-0.06)
    assert out["largest_d_test"] == pytest.approx(0.0)


def test_verdict_reads_the_arms_that_ran():
    scores = _scores(0.60, 0.60)
    for by_arm in scores.values():
        del by_arm["clip"]  # 12 and 16 dimensions run the control and the quantile arm only
    out = verdict(scores)
    assert [r["arm"] for r in out["differences"]] == ["quantile", "quantile"]
    assert out["branch"] == "NULL_SURVIVES"


def test_lead_reading_uses_the_margin_on_both_sides():
    assert lead_reading(0.607, 0.574)["branch"] == "LEAD_HOLDS"
    assert lead_reading(0.597, 0.580)["branch"] == "PARITY"
    assert lead_reading(0.607, 0.640)["branch"] == "QUANTUM_AHEAD"
    assert lead_reading(0.597, 0.580)["lead"] == pytest.approx(0.017)
