"""run_classical and run_quantum exports (decision log 2026-11-17T02:18Z): write-only recording.

On toy regimes: the evaluation's results are identical with and without an Exports object, and
the recorded per-seed metrics and predictions equal what the evaluation computed.
"""

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from qgridbench.features.reduce import AngleScaler
from qgridbench.protocol import Regime
from qgridbench.utils.run_tracking import REPO_ROOT

if str(REPO_ROOT) not in sys.path:  # the runners import experiments.exports
    sys.path.insert(0, str(REPO_ROOT))
from experiments.exports import Exports  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [name]  # no --threads: nothing is pinned
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


rc = _load("run_classical")
rq = _load("run_quantum")


def _toy_regime(n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X, y = {}, {}
    for s, n in (("train", 80), ("val", 40), ("test", 30)):
        y[s] = np.tile([0, 1], n // 2)
        X[s] = rng.normal(size=(n, n_features)) + y[s][:, None] * 1.5
    sc = AngleScaler().fit(X["train"])
    ang = {s: sc.transform(X[s]) for s in X}
    return Regime(name="pca", n_components=n_features, X=X, y=y, angles=ang, angle_scaler=sc)


def _per_seed_from_exports(exp, family):
    out = {}
    for r in exp.metrics:
        if r["family"] == family and r["metric"] not in ("train_time_s", "infer_latency_s"):
            out.setdefault(r["seed"], {})[r["metric"]] = r["value"]
    return [out[s] for s in sorted(out)]


@pytest.mark.parametrize("name", ["logreg", "rf"])
def test_classical_eval_is_unchanged_and_recorded(name):
    reg = _toy_regime()
    params = {"logreg": {"C": 1.0}, "rf": {"n_estimators": 20, "max_depth": 4,
              "min_samples_leaf": 1, "max_features": "sqrt"}}[name]  # fmt: skip
    plain = rc.eval_model(name, params, reg, [0, 1], None)
    exp = Exports("toy", None)
    rec = rc.eval_model(name, params, reg, [0, 1], None, exp)
    if name == "logreg":  # deterministic: bit-identical with and without the recorder
        assert rec["per_seed"] == plain["per_seed"]
    else:  # a threaded forest averages its trees in arrival order (last-bit noise, measured)
        for r, q in zip(rec["per_seed"], plain["per_seed"], strict=True):
            assert r.keys() == q.keys()
            assert all(r[k] == pytest.approx(q[k], rel=1e-12, abs=1e-15) for k in r)
    assert _per_seed_from_exports(exp, name) == rec["per_seed"]
    assert set(exp.preds) == {(0, "clean", name), (1, "clean", name)}
    p = exp.preds[(0, "clean", name)]
    assert p["proba"].shape == (30, 2) and np.array_equal(p["y"], reg.y["test"])
    assert np.array_equal(p["test_pos"], np.arange(30))


class _Run:
    def write_json(self, *a, **k):
        pass


def test_vqc_eval_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(rq, "mlflow_log", lambda *a, **k: None)
    reg = _toy_regime(n_features=2)
    cfg_q = {
        "qkernel": {"subset_cap": 40, "tuning_val_subset": 20},
        "vqc": {"optuna": {"n_trials": 1, "lr": {"low": 0.01, "high": 0.02, "log": True},
                           "batch_size": [16]}, "depths": [1], "tuning_max_epochs": 1,
                "max_epochs": 1, "patience": 1},
    }  # fmt: skip
    storage = f"sqlite:///{(tmp_path / 'vqc.db').as_posix()}"
    exp = Exports("toy", None)
    res = rq.run_vqc(reg, 2, [0, 1], cfg_q, storage, "toy", _Run(), exp)
    assert _per_seed_from_exports(exp, "vqc") == res["per_seed"]
    assert set(exp.preds) == {(0, "clean", "vqc"), (1, "clean", "vqc")}


def test_qkernel_eval_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(rq, "mlflow_log", lambda *a, **k: None)
    monkeypatch.setattr(rq, "cached_kernel", lambda key: None)
    monkeypatch.setattr(rq, "store_kernel", lambda *a, **k: None)
    reg = _toy_regime(n_features=2)
    cfg_q = {
        "qkernel": {"subset_cap": 40, "tuning_val_subset": 20,
                    "bandwidth": {"low": 0.5, "high": 1.0, "log": True},
                    "svm_C": {"low": 1.0, "high": 10.0, "log": True}},
        "vqc": {"optuna": {"n_trials": 2}},
    }  # fmt: skip
    storage = f"sqlite:///{(tmp_path / 'qk.db').as_posix()}"
    exp = Exports("toy", None)
    res = rq.run_qkernel(reg, "zz", 2, [0, 1], cfg_q, storage, "toy", _Run(), exp)
    assert _per_seed_from_exports(exp, "qsvm_zz") == res["per_seed"]
    assert np.array_equal(exp.preds[(0, "clean", "qsvm_zz")]["proba"],
                          exp.preds[(1, "clean", "qsvm_zz")]["proba"])  # fmt: skip


def test_exports_write_shapes(tmp_path, monkeypatch):
    import pandas as pd

    import experiments.exports as ex

    monkeypatch.setattr(
        "qgridbench.data.preprocess.load_processed",
        lambda v: (None, None, {"test": np.arange(100, 130)}),
    )
    reg = _toy_regime()
    exp = Exports("toy", None)
    exp.extra = {"threads": None}
    rc.eval_model("logreg", {"C": 1.0}, reg, [0, 1], None, exp)
    exp.write(tmp_path)
    pred = pd.read_parquet(tmp_path / "predictions.parquet")
    assert len(pred) == 60 and pred["row_id"].min() == 100
    ps = pd.read_csv(tmp_path / "per_seed.csv")
    assert set(ps["seed"]) == {0, 1} and "macro_f1" in set(ps["metric"])
    assert (tmp_path / "exports_provenance.json").exists() and ex.Exports is Exports
    _ = SimpleNamespace
