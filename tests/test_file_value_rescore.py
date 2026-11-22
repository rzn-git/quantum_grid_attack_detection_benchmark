"""R4-16 re-score helpers (run_file_quality_predictors --rescore-from), synthetic fixtures."""

import importlib.util
import sys

import numpy as np
import pytest
from scipy.stats import spearmanr

from qgridbench.utils.run_tracking import REPO_ROOT

# the runner imports `experiments.run_group_split_control`, so the repo root goes on sys.path
sys.path.insert(0, str(REPO_ROOT))
_spec = importlib.util.spec_from_file_location(
    "run_file_quality_predictors", REPO_ROOT / "experiments" / "run_file_quality_predictors.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
perm_rhos, rescore_verdict, per_file_means = (
    _mod.perm_rhos,
    _mod.rescore_verdict,
    _mod.per_file_means,
)
NAMES = ("a", "b", "c", "d")


def test_perm_rhos_equals_spearman_of_each_permutation():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(4, 15))
    y = rng.normal(size=15)
    got = perm_rhos(P, y, 50, np.random.default_rng(7))
    ry_rng = np.random.default_rng(7)
    from scipy.stats import rankdata

    ry = rankdata(y)
    for t in range(50):
        yp = ry_rng.permutation(ry)
        for j in range(4):
            assert got[t, j] == pytest.approx(abs(spearmanr(P[j], yp).statistic), abs=1e-12)


def test_familywise_bar_is_at_least_each_own_bar():
    rng = np.random.default_rng(2)
    P = rng.normal(size=(4, 15))
    y = rng.normal(size=15)
    null = perm_rhos(P, y, 2000, rng)
    rho = {k: float(spearmanr(P[i], y).statistic) for i, k in enumerate(NAMES)}
    v = rescore_verdict(rho, null, NAMES, 0.5)
    assert all(v["familywise_p95"] >= p["own_p95"] for p in v["per_predictor"].values())
    assert all(p["fw_p"] >= p["own_p"] for p in v["per_predictor"].values())


def test_monotone_predictor_is_predictable_and_noise_is_not():
    rng = np.random.default_rng(3)
    y = rng.normal(size=15)
    P = np.vstack([y + 0.0, rng.normal(size=(3, 15))])
    rho = {k: float(spearmanr(P[i], y).statistic) for i, k in enumerate(NAMES)}
    v = rescore_verdict(rho, perm_rhos(P, y, 2000, rng), NAMES, 0.5)
    assert v["per_predictor"]["a"]["rho"] == pytest.approx(1.0)
    assert v["branch"] == "PREDICTABLE"
    noise = rng.normal(size=(4, 15))
    rho_n = {k: float(spearmanr(noise[i], y).statistic) for i, k in enumerate(NAMES)}
    v_n = rescore_verdict(rho_n, perm_rhos(noise, y, 2000, rng), NAMES, 0.5)
    assert v_n["branch"] in {"NOT_PREDICTABLE", "INTERMEDIATE"}


def test_per_file_means_averages_over_partitions():
    cells = [
        {"partition": 0, "file": 2, "excess": 0.1},
        {"partition": 1, "file": 2, "excess": 0.3},
        {"partition": 0, "file": 5, "excess": 0.4},
    ]
    assert per_file_means(cells, "excess") == {2: pytest.approx(0.2), 5: pytest.approx(0.4)}
