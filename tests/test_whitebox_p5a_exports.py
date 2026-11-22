"""run_whitebox_qsvm p5a exports (operator rule 2026-11-17): the recorder is write-only.

p5a runs with fakes for the zoo, the gate and the attacks; the same result comes back with and
without an Exports object, and the recorder holds every prediction, metric and attacked set."""

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


wb = _load("run_whitebox_qsvm")
N, D = 30, 3


def _proba(X, w):
    z = 1 / (1 + np.exp(-(X @ w)))
    return np.column_stack([1 - z, z])


def _fakes(monkeypatch):
    w = {"qsvm": np.array([1.0, -0.5, 0.2]), "rbf_svm": np.array([-0.3, 0.8, 0.1])}

    class Zoo:
        def __init__(self, *a, **k):
            self.models = {"qsvm": "q", "rbf_svm": "r"}
            self._states_tr, self._angle = None, None

        def predict_proba(self, name, X):
            return _proba(X, w[name])

        def grad_provider(self, name):
            return name

    monkeypatch.setattr(wb, "FittedZoo", Zoo)
    monkeypatch.setattr(wb, "QSVMWhiteBox", lambda *a: "wb")
    gate = {"forward_rel_err": 0.0, "deployed_float32_rel_err": 0.0, "grad_rel_err": 0.0}
    monkeypatch.setattr(wb, "assert_matches_reference", lambda *a: gate)
    monkeypatch.setattr(wb, "stratified_cap", lambda X, y, n, seed: np.arange(min(n, len(y))))
    monkeypatch.setattr(wb, "GradProvider", lambda m, name: name)
    monkeypatch.setattr(wb, "spawn_rng", lambda s, name: np.random.default_rng(s))

    def fake_pgd(gp, X, y, eps, scale, n_steps, step_frac, rng):
        return X + eps * rng.standard_normal(X.shape)

    monkeypatch.setattr(wb, "pgd", fake_pgd)
    monkeypatch.setattr(wb, "fgsm", lambda g, X, y, e, s: X + e)


def test_p5a_recorder_is_write_only(monkeypatch):
    _fakes(monkeypatch)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, D))
    reg = SimpleNamespace(
        X={"test": X}, y={"test": np.tile([0, 1], N // 2)}, feature_scale=np.ones(D)
    )
    args = SimpleNamespace(encoding="zz", qubits=D)
    cfg = {
        "eval_subset_adv": N,
        "evasion": {"epsilons": [0.1, 0.5], "pgd": {"n_steps": 2, "step_frac": 0.25}},
    }
    plain = wb.p5a_whitebox_qsvm(reg, args, cfg, {}, 10, attack_seeds=(0, 1))
    ex = Exports("binary", None)
    recorded = wb.p5a_whitebox_qsvm(reg, args, cfg, {}, 10, attack_seeds=(0, 1), ex=ex)
    assert recorded == plain
    # clean x2 models; per attack seed: 2 eps x (qsvm + rbf) + 1 transfer; FGSM 2 eps
    assert len(ex.preds) == 2 + 2 * (2 * 2 + 1) + 2 == len(ex.metrics)
    assert len(ex.crafted) == 2 * (2 * 2 + 1) + 2
    assert "fgsm|fgsm@0.5|qsvm" in ex.crafted and "1|pgd_rbf@0.1|rbf_svm" in ex.crafted


def test_p5a_exact_eval_scores_the_exact_path_and_stores_the_deployed_distance(monkeypatch):
    """--exact-eval (gate incident of 2026-11-21): the QSVM is scored on the exact Gram, the gate
    runs without the deployed bound, and the distance to the deployed path is stored per set."""
    _fakes(monkeypatch)
    w_exact = np.array([1.0, -0.5, 0.2])
    w_deployed = np.array([1.0, -0.5, 0.9])  # a different model, so the two paths disagree

    class Zoo:
        n_classes = 2

        def __init__(self, *a, **k):
            self.models = {"qsvm": "q", "rbf_svm": "r"}
            self._states_tr = "train-states"
            self._angle = SimpleNamespace(transform=lambda X: X)

        def predict_proba(self, name, X):
            return _proba(X, w_deployed if name == "qsvm" else np.array([-0.3, 0.8, 0.1]))

        def grad_provider(self, name):
            return name

    gate_kwargs = []

    def gate(*a, **k):
        gate_kwargs.append(k)
        return {"forward_rel_err": 0.0, "deployed_float32_rel_err": 0.0, "grad_rel_err": 0.0}

    exact_flags = []

    def kernel(states, states_tr, exact=False):
        exact_flags.append(exact)
        return states  # the fake "Gram" is the input itself

    monkeypatch.setattr(wb, "FittedZoo", Zoo)
    monkeypatch.setattr(wb, "assert_matches_reference", gate)
    monkeypatch.setattr(wb, "compute_states", lambda A, encoding, bandwidth: A)
    monkeypatch.setattr(wb, "fidelity_kernel", kernel)
    monkeypatch.setattr(wb, "_svc_proba", lambda svc, K, n_classes: _proba(K, w_exact))

    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, D))
    y = (X @ w_exact > 0).astype(int)  # the exact path's own labels: it scores 1.0 on clean rows
    reg = SimpleNamespace(X={"test": X}, y={"test": y}, feature_scale=np.ones(D))
    cfg = {
        "eval_subset_adv": N,
        "evasion": {"epsilons": [0.1, 0.5], "pgd": {"n_steps": 2, "step_frac": 0.25}},
    }
    args = SimpleNamespace(encoding="zz", qubits=D, exact_eval=True)
    out = wb.p5a_whitebox_qsvm(reg, args, cfg, {}, 10, attack_seeds=(0, 1))

    assert gate_kwargs == [{"deployed_tol": None}]
    assert out["qsvm_evaluation_path"] == "exact complex128"
    check = out["deployed_path_check"]
    # clean; per attack seed 2 eps + 1 transfer; FGSM at 2 eps
    assert check["n_sets"] == 1 + 2 * (2 + 1) + 2 == len(exact_flags)
    assert all(exact_flags)
    assert check["n_rows_per_set"] == N
    assert check["max_predictions_differ"] > 0  # the fake deployed model is another model
    # the reported clean score is the exact path's
    assert out["clean_f1"] == 1.0 > wb._f1(y, _proba(X, w_deployed))

    # the default path is untouched: no keyword to the gate, no exact kernel, no extra keys
    gate_kwargs.clear()
    exact_flags.clear()
    plain = wb.p5a_whitebox_qsvm(
        reg, SimpleNamespace(encoding="zz", qubits=D), cfg, {}, 10, attack_seeds=(0, 1)
    )
    assert gate_kwargs == [{}] and exact_flags == []
    assert "qsvm_evaluation_path" not in plain and "deployed_path_check" not in plain
    assert plain["clean_f1"] == wb._f1(y, _proba(X, w_deployed))
