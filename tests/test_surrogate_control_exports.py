"""run_surrogate_control exports (operator rule 2026-11-17): the recorder is write-only.

seed_record runs with fakes for fitting and PGD; the same macro-F1 values come back with and
without an Exports object, and the recorder holds every prediction, metric and crafted set."""

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np

from qgridbench.utils.run_tracking import REPO_ROOT

if str(REPO_ROOT) not in sys.path:  # the runner imports experiments.exports
    sys.path.insert(0, str(REPO_ROOT))
from experiments.exports import Exports  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sc = _load("run_surrogate_control")


def _proba(X, w):
    z = 1 / (1 + np.exp(-(X @ w)))
    return np.column_stack([1 - z, z])


def _fakes(monkeypatch):
    monkeypatch.setattr(sc, "spawn_rng", lambda s, name: np.random.default_rng(s))
    monkeypatch.setattr(sc, "set_all_seeds", lambda s: None)
    monkeypatch.setattr(
        sc,
        "build",
        lambda name, params, s: SimpleNamespace(
            predict_proba=lambda X: _proba(X, -np.ones(X.shape[1]))
        ),
    )
    monkeypatch.setattr(sc, "fit_model", lambda *a, **k: None)
    monkeypatch.setattr(sc, "GradProvider", lambda model, name: name)

    def fake_pgd(gp, X, y, eps, scale, n_steps, step_frac, rng):
        return X + eps * rng.standard_normal(X.shape)

    monkeypatch.setattr(sc, "pgd", fake_pgd)
    weights = {m: np.linspace(-1, 1, 3) * (i + 1) for i, m in enumerate(sc.FAMILIES)}
    return SimpleNamespace(
        grad_provider=lambda name: name,
        predict_proba=lambda name, X: _proba(X, weights[name]),
        models={"rbf_svm": "svm"},
        Xtr=None,
        ytr=None,
    )


def test_recorder_is_write_only(monkeypatch):
    zoo = _fakes(monkeypatch)
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 3)), np.tile([0, 1], 20)
    pos = np.arange(100, 140)
    cfg = {"n_steps": 3, "step_frac": 0.25}
    plain = sc.seed_record(zoo, {"mlp": {}}, X, y, 0.3, 1.0, cfg, 2)
    ex = Exports("binary", None)
    recorded = sc.seed_record(zoo, {"mlp": {}}, X, y, 0.3, 1.0, cfg, 2, ex, pos)
    assert recorded == plain
    n_sets, n_models = 5, len(sc.FAMILIES) + 1  # clean + four PGD sources; + mlp_balanced
    assert len(ex.preds) == n_sets * n_models == len(ex.metrics)
    assert sorted(ex.crafted) == sorted(
        f"2|pgd_{s}|{s}" for s in ("mlp", "mlp_balanced", "vqc", "rbf_svm")
    )
    m = next(r for r in ex.metrics if r["condition"] == "pgd_vqc" and r["family"] == "rf")
    assert m["value"] == plain["f1"]["pgd_vqc"]["rf"]
    assert (ex.preds[(2, "clean", "vqc")]["test_pos"] == pos).all()
