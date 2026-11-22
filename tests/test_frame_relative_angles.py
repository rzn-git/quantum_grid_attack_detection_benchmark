"""R4-09 runner: the frame-relative transform, the permuted control's derangement and the branch."""

import importlib.util

import numpy as np
import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

# experiments/ is not an installed package; load the runner by path (it exits only as __main__)
_spec = importlib.util.spec_from_file_location(
    "run_frame_relative_angles", REPO_ROOT / "experiments" / "run_frame_relative_angles.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _names():
    names = []
    for r in (1, 2, 3, 4):
        for k in range(1, 13):
            names += [f"R{r}-PA{k}:{'VH' if k in (1, 2, 3, 7, 8, 9) else 'IH'}", f"R{r}-PM{k}:V"]
        names += [f"R{r}:F", f"R{r}-PA:Z", f"R{r}-PA:ZH"]
    return names


def test_wrap_deg_half_open_interval():
    d = np.array([0.0, 180.0, -180.0, 190.0, -190.0, 359.0, 540.0, -360.0])
    np.testing.assert_allclose(
        _mod.wrap_deg(d), [0.0, 180.0, 180.0, -170.0, 170.0, -1.0, 180.0, 0.0]
    )


def test_frame_relative_toy_case():
    names = _names()
    ch = _mod.angle_channels(names)
    assert len(ch) == 4 and all(len(c) == 12 for c in ch.values())
    rng = np.random.default_rng(0)
    X = rng.uniform(-180, 180, size=(5, len(names))).astype(np.float32)
    X[0, ch[1][7]], X[0, ch[1][1]] = 170.0, -170.0  # crosses the cut: -340 wraps to +20
    out = _mod.frame_relative(X, ch)
    assert out.dtype == X.dtype
    assert out[0, ch[1][1]] == pytest.approx(20.0)
    for c in ch.values():
        assert np.all(out[:, c[7]] == 0.0)
        for j in c.values():
            want = _mod.wrap_deg(X[:, j].astype(np.float64) - X[:, c[7]].astype(np.float64))
            np.testing.assert_allclose(out[:, j], want.astype(np.float32))
    moved = [j for c in ch.values() for j in c.values()]
    keep = np.setdiff1d(np.arange(len(names)), moved)
    assert np.array_equal(out[:, keep], X[:, keep])  # PM, F, PA:Z and PA:ZH untouched
    assert "R1-PA:ZH" in [names[j] for j in keep]
    assert np.all((out[:, moved] > -180) & (out[:, moved] <= 180))


def test_frame_relative_float32_cut_stays_half_open():
    names = _names()
    ch = _mod.angle_channels(names)
    X = np.zeros((1, len(names)), dtype=np.float32)
    X[0, ch[1][7]] = np.float32(89.999995)
    X[0, ch[1][1]] = np.float32(-90.0)
    wrapped = _mod.wrap_deg(np.float64(X[0, ch[1][1]]) - np.float64(X[0, ch[1][7]]))
    assert wrapped > -180.0 and np.float32(wrapped) == -180.0  # the cast lands on the cut
    out = _mod.frame_relative(X, ch)
    assert out[0, ch[1][1]] == 180.0


def test_frame_relative_with_reference_rows():
    names = _names()
    ch = _mod.angle_channels(names)
    X = np.random.default_rng(1).uniform(-180, 180, size=(4, len(names)))
    ref = np.array([1, 2, 3, 0])
    out = _mod.frame_relative(X, ch, ref)
    j, r7 = ch[2][4], ch[2][7]
    np.testing.assert_allclose(out[:, j], _mod.wrap_deg(X[:, j] - X[ref, r7]))


def test_angle_channels_fails_loud_on_gap():
    names = [n for n in _names() if n != "R3-PA5:IH"]
    with pytest.raises(ValueError, match="R3"):
        _mod.angle_channels(names)


def test_derangement_stays_in_its_part_without_fixed_points():
    parts = {"train": np.arange(0, 10), "val": np.arange(10, 13), "test": np.arange(13, 20)}
    ref = _mod.derangement_within(parts, 20, np.random.default_rng(0))
    assert not np.any(ref == np.arange(20))
    for idx in parts.values():
        assert set(ref[idx]) == set(idx)  # a permutation of its own part


def test_gains_and_branch():
    seeds = range(3)
    cells = {}
    for fam, (g_row, g_wf) in {"logreg": (0.05, 0.03), "qsvm": (0.01, 0.0)}.items():
        cells[("absolute", "row", fam)] = dict.fromkeys(seeds, 0.5)
        cells[("relative", "row", fam)] = dict.fromkeys(seeds, 0.5 + g_row)
        for p in _mod.SPLITS[1:]:
            cells[("absolute", p, fam)] = dict.fromkeys(seeds, 0.5)
            cells[("relative", p, fam)] = dict.fromkeys(seeds, 0.5 + g_wf)
    g = _mod.gains(cells, "relative")
    assert g["row"]["logreg"]["gain"]["mean"] == pytest.approx(0.05)
    assert g["whole_file"]["logreg"]["gain"]["n"] == 15
    assert g["row_mean_over_families"] == pytest.approx(0.03)
    assert g["whole_file_mean_over_families"] == pytest.approx(0.015)
    assert _mod.branch(g) == "ROW_ONLY"
    assert _mod.branch({"row_mean_over_families": 0.0, "whole_file_mean_over_families": 0.02}) == (
        "HYPOTHESIS"
    )
    assert _mod.branch({"row_mean_over_families": 0.01, "whole_file_mean_over_families": 0.0}) == (
        "KILL"
    )
