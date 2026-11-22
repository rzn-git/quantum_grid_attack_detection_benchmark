"""R4-22 H3 runner: branch rules and the state-space chain rule on a case14 toy context."""

import copy
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np

from qgridbench.data.fdia_synth import Case, fdia_config
from qgridbench.utils.run_tracking import REPO_ROOT

if str(REPO_ROOT) not in sys.path:  # the runner imports experiments.* modules
    sys.path.insert(0, str(REPO_ROOT))
_argv = sys.argv
sys.argv = ["run_fdia_consistent_attack", "--threads", "1"]  # thread caps are read at import
try:
    _spec = importlib.util.spec_from_file_location(
        "run_fdia_consistent_attack", REPO_ROOT / "experiments" / "run_fdia_consistent_attack.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _mod  # its dataclass resolves annotations through sys.modules
    _spec.loader.exec_module(_mod)
finally:
    sys.argv = _argv

T = ("rf", "xgb", "lgbm")


def _e(a, b, c):
    return dict(zip(T, (a, b, c), strict=True))


def test_h3_branch():
    assert _mod.h3_branch(_e(0.2, 0.2, 0.2), _e(0.0, 0.01, 0.2)) == "PHYSICS"
    assert _mod.h3_branch(_e(0.2, 0.2, 0.2), _e(0.06, 0.07, 0.0)) == "SURVIVES"
    assert _mod.h3_branch(_e(0.01, 0.02, 0.2), _e(0.0, 0.0, 0.0)) == "NOT_TESTABLE"
    assert _mod.h3_branch(_e(0.2, 0.2, 0.01), _e(0.0, 0.06, 0.0)) == "MIXED"


def test_tree_branch():
    assert _mod.tree_branch(0.05, 0.0) == "TREE_TRANSFER_SURVIVES"
    assert _mod.tree_branch(0.049, 0.049) == "PHYSICS_REMOVES_IT_FOR_TREES_TOO"


def test_state_space_chain_rule_matches_finite_differences():
    case = Case("case14", copy.deepcopy(fdia_config()))
    rng = np.random.default_rng(0)
    x0 = case.pack(case.vm0, case.va0)
    xhat = x0 + rng.normal(0, 0.01, size=(4, x0.size))
    h0 = case.h_rows(xhat)
    s = np.full(case.m, 0.1)
    W = rng.normal(size=(8, case.m))
    w = rng.normal(size=8)
    ctx = SimpleNamespace(U0=h0 / s, Z0=np.zeros((4, 8)), W=W, s=s, xhat=xhat)

    def grad_z(Z, y):  # loss = Z @ w per row
        return np.tile(w, (len(Z), 1))

    def loss(D):
        return ((case.h_rows(xhat + D) - h0) / s) @ W.T @ w

    D = rng.normal(0, 0.02, size=xhat.shape)
    g = _mod.c_space_grad(grad_z, ctx, case, h0)(D, np.zeros(4))
    eps = 1e-6
    for j in range(0, x0.size, 5):
        e = np.zeros_like(D)
        e[:, j] = eps
        fd = (loss(D + e) - loss(D - e)) / (2 * eps)
        assert np.allclose(fd, g[:, j], rtol=1e-5, atol=1e-6)
    assert np.all(_mod.state_map(case, xhat, h0, np.zeros_like(xhat)) == 0.0)


def test_norms():
    n = _mod.norms(np.array([[3.0, -4.0], [0.0, 1.0]]))
    assert n["du_linf_median"] == 2.5 and n["du_l2_median"] == 3.0


def test_write_exports_from_unit_pickles(tmp_path, monkeypatch):
    import pickle

    import pandas as pd

    monkeypatch.setattr(
        "qgridbench.data.preprocess.load_processed",
        lambda v: (None, None, {"test": np.arange(1000, 1010)}),
    )
    y = np.array([0, 1, 0, 1])
    test_pos = np.array([1, 3, 5, 7])
    p = np.full((4, 2), 0.5)
    ue = {
        "preds": [("clean", "rf", p), ("U@0.5:transfer_mlp", "rf", p)],
        "crafted": [
            ("U@0.5", "transfer_mlp", np.ones((4, 3))),
            ("C_state_offset@0.5", "transfer_mlp", np.zeros((4, 2))),
        ],
    }  # fmt: skip
    with open(tmp_path / "unit_exports_seed0.pkl", "wb") as fh:
        pickle.dump(ue, fh)
    rec = {"attacker": "U", "eps": 0.5, "kind": "transfer_mlp", "family": "rf", "clean_f1": 0.9,
           "attacked_f1": 0.4, "du_linf_median": 0.5, "du_l2_median": 1.0}  # fmt: skip
    bdd = {"attacker": "U", "eps": 0.5, "source": "mlp", "flag_rate": 1.0, "clean_flag_rate": 0.0,
           "agreement": 0.0}  # fmt: skip
    units = [{"seed": 0, "records": [rec], "bdd": [bdd]}]
    ctx = SimpleNamespace(y=y, test_pos=test_pos, eps_grid=[0.5], threads=1, tuned_file=None)
    args = SimpleNamespace(variant="toy", workers=1)
    _mod.write_exports(tmp_path, args, ctx, units)
    pred = pd.read_parquet(tmp_path / "predictions.parquet")
    assert set(pred["condition"]) == {"clean", "U@0.5:transfer_mlp"}
    assert pred["row_id"].tolist()[:4] == [1001, 1003, 1005, 1007]
    cr = np.load(tmp_path / "crafted_inputs.npz")
    assert set(cr.files) == {"0|U@0.5|transfer_mlp", "0|C_state_offset@0.5|transfer_mlp"}
    ps = pd.read_csv(tmp_path / "per_seed.csv")
    assert {"attacked_f1", "flag_rate"} <= set(ps["metric"])
    assert not (tmp_path / "unit_exports_seed0.pkl").exists()
    import json

    prov = json.loads((tmp_path / "exports_provenance.json").read_text())
    assert "0:fdia_U_mlp_0.5" in prov["random_streams"]
