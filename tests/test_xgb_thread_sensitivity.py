"""R4-01 runner: the per-seed fit is run_classical's, and the anchor and branch read as registered
(tiny fixtures, no data)."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mods():
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        yield (
            importlib.import_module("experiments.run_xgb_thread_sensitivity"),
            importlib.import_module("experiments.run_classical"),
        )


XGB_PARAMS = {
    "n_estimators": 20,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
}
RF_PARAMS = {"n_estimators": 20, "max_depth": 5, "min_samples_leaf": 1, "max_features": "sqrt"}


@pytest.mark.parametrize(("model", "params"), [("xgb", XGB_PARAMS), ("rf", RF_PARAMS)])
def test_fit_score_matches_run_classical_eval_model(
    mods, toy_binary, toy_splits, model, params, monkeypatch
):
    # one joblib thread: RF's predict_proba sums trees in thread-arrival order otherwise
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    run, classical = mods
    X, y = toy_binary
    reg = SimpleNamespace(
        X={k: X[v] for k, v in toy_splits.items()}, y={k: y[v] for k, v in toy_splits.items()}
    )
    published = classical.eval_model(model, params, reg, [0, 1], None)["per_seed"]
    ours = [
        run.fit_score(
            model, params, reg.X["train"], reg.y["train"], reg.X["test"], reg.y["test"], s
        )
        for s in (0, 1)
    ]
    assert [o["metrics"] for o in ours] == published


def _cells(means_by_t, spread=0.001):
    return {
        t: {"per_seed": [m - spread, m + spread], "mean": m, "std": spread * np.sqrt(2)}
        for t, m in means_by_t.items()
    }


def test_branch_documents_at_the_seed_std_and_discloses_above(mods):
    run, _ = mods
    cells = _cells({1: 0.500, 4: 0.502, 10: 0.501, 24: 0.503})
    assert run.branch(cells, (1, 4, 10, 24), 0.0031)["branch"] == "DOCUMENT"
    out = run.branch(cells, (1, 4, 10, 24), 0.0029)
    assert out["branch"] == "DISCLOSE"
    assert out["D"] == pytest.approx(0.003)
    assert (out["argmax_threads"], out["argmin_threads"]) == (24, 1)


def test_anchor_uses_the_published_seeds_and_tolerance(mods):
    run, _ = mods
    pub = [0.1 * i for i in range(10)]
    cells = {6: {"per_seed": [pub[2], pub[5] + 5e-7]}}
    out = run.anchor(cells, pub, [2, 5], 6)
    assert out["passed"]
    assert out["mean_published"] == pytest.approx(0.35)
    cells[6]["per_seed"][1] = pub[5] + 3e-6
    assert not run.anchor(cells, pub, [2, 5], 6)["passed"]


def test_cell_table_fails_loud_on_a_missing_seed(mods):
    run, _ = mods
    units = [
        {"regime": "pca8", "model": "xgb", "threads": 4, "seed": s, "metrics": {"macro_f1": 0.5}}
        for s in (0, 1)
    ]
    assert run.cell_table(units, [0, 1])["pca8"]["xgb"][4]["mean"] == 0.5
    with pytest.raises(AssertionError, match="seeds \\[2\\] missing"):
        run.cell_table(units, [0, 1, 2])
