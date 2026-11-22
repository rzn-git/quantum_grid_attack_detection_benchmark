"""R4-17(a) co-peak classifier (run_ablations --only copeak), on synthetic flip profiles."""

import importlib.util

import numpy as np
import pytest
from scipy.signal import find_peaks, peak_prominences

from qgridbench.utils.run_tracking import REPO_ROOT

_spec = importlib.util.spec_from_file_location(
    "run_ablations", REPO_ROOT / "experiments" / "run_ablations.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
peak_prominence, classify_copeak, known_answers = (
    _mod.peak_prominence,
    _mod.classify_copeak,
    _mod._copeak_known_answers,
)
TAU = 0.065


def test_prominence_matches_scipy_on_interior_peaks():
    rng = np.random.default_rng(0)
    for _ in range(200):
        f = rng.uniform(0, 1, 7)
        peaks, _ = find_peaks(f)
        if len(peaks):
            want = peak_prominences(f, peaks)[0]
            got = [peak_prominence(f, int(p)) for p in peaks]
            assert got == pytest.approx(want, abs=1e-12)


def test_non_peak_and_border_score_zero():
    f = [0.1, 0.3, 0.2, 0.5, 0.4]
    assert peak_prominence(f, 2) == 0.0  # higher neighbour
    assert peak_prominence(f, 0) == 0.0  # border
    assert peak_prominence(f, 4) == 0.0  # border, even when not the max


def test_known_answers_pass():
    out = known_answers(TAU)
    assert {k: v["class"] for k, v in out.items()} == {k: v["expected"] for k, v in out.items()}


def test_deployed_profile_reads_as_logged_hand_arithmetic():
    # deployed ZZ profile at eps 0.5 (archive 20260814T002646Z), b* = 0.8 (index 2)
    f = [0.19, 0.3267, 0.42, 0.40, 0.2933]
    full = classify_copeak(f, 2, TAU)
    assert full["class"] == "CO-PEAK" and full["prominence"] == pytest.approx(0.42 - 0.2933)
    healthy = classify_copeak(f[1:4], 1, TAU)
    assert healthy["class"] == "TIED-TOP" and healthy["prominence"] == pytest.approx(0.02)


def test_classes_are_exclusive_in_order():
    assert classify_copeak([0.40, 0.41, 0.40], 1, TAU)["class"] == "FLAT"
    assert classify_copeak([0.1, 0.5, 0.1], 0, TAU)["class"] == "MISALIGNED"
