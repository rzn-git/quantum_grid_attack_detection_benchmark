"""Round-4 exports (decision log 2026-11-17T02:18Z) of the R4-12 runners: each hook leaves the
reported numbers unchanged and records predictions, crafted inputs and per-seed metrics
(tiny synthetic data, fake splits; no repository data)."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qgridbench.features.reduce import AngleScaler
from qgridbench.utils.run_tracking import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))  # experiments/ is not an installed package
from experiments import exports as ex_mod  # noqa: E402

N_TEST = 40


@pytest.fixture
def ex(monkeypatch, tmp_path):
    import qgridbench.data.preprocess as pre

    ids = np.arange(100, 100 + N_TEST)
    monkeypatch.setattr(pre, "load_processed", lambda v: (None, None, {"test": ids}))
    (tmp_path / "tuned.json").write_text("{}")
    monkeypatch.setattr(ex_mod, "REPO_ROOT", tmp_path)
    return ex_mod.Exports("gas_binary", "tuned.json")


def _toy_reg(d=2, n_tr=60, seed=0):
    rng = np.random.default_rng(seed)
    X = {"train": rng.standard_normal((n_tr, d)), "test": rng.standard_normal((N_TEST, d))}
    y = {k: (v[:, 0] > 0).astype(int) for k, v in X.items()}
    reg = SimpleNamespace(X=X, y=y, feature_scale=X["train"].std(axis=0))
    reg.angle_scaler = AngleScaler().fit(X["train"])
    return reg


CFG_ADV = {
    "evasion": {"epsilons": [0.5], "pgd": {"n_steps": 2, "step_frac": 0.25}},
    "blackbox": {"eval_subset": 10},
    "eval_subset_adv": 20,
}


class _FakeZoo:
    def clean_proba(self, f, X, y):
        return self.predict_proba(f, X)

    def predict_proba(self, f, X):
        p1 = 1.0 / (1.0 + np.exp(-X[:, 0] * (2.0 if f == "qsvm" else 1.0)))
        return np.column_stack([1 - p1, p1])


def test_random_control_hook_records_without_changing_numbers(ex, tmp_path):
    from experiments.run_adversarial import random_perturbation_control

    X = np.random.default_rng(1).standard_normal((N_TEST, 3))
    y = (X[:, 0] > 0).astype(int)
    pos = np.arange(N_TEST)
    cfg = {"evasion": {"epsilons": [0.1, 0.5]}}
    base = random_perturbation_control(_FakeZoo(), X, y, cfg, np.random.default_rng(3), 1.0)
    hooked = random_perturbation_control(
        _FakeZoo(), X, y, cfg, np.random.default_rng(3), 1.0, ex=ex, seed=0, pos=pos
    )
    assert hooked == base
    ex.write(tmp_path)
    p = pd.read_parquet(tmp_path / "predictions.parquet")
    assert set(p.condition) >= {"random_base", "random@0.5|draw2"}
    c = np.load(tmp_path / "crafted_inputs.npz")
    assert "0|random@0.1|draw0|random" in c.files
    m = pd.read_csv(tmp_path / "per_seed.csv")
    assert set(m.metric) == {"macro_f1", "flip_rate"}


def test_p8a_hook_records_without_changing_numbers(ex, tmp_path):
    from experiments.run_whitebox_qsvm import p8a_lengthscale_control

    reg = _toy_reg()
    bp = {"rbf_svm": {"C": 10.0, "gamma": 0.5}}
    kw = {"seed": 0, "gamma_mults": (1.0, 0.1), "attack_seeds": (0, 1)}
    base = p8a_lengthscale_control(reg, CFG_ADV, bp, 50, **kw)
    hooked = p8a_lengthscale_control(reg, CFG_ADV, bp, 50, ex=ex, **kw)
    assert json.dumps(hooked, sort_keys=True) == json.dumps(base, sort_keys=True)
    ex.write(tmp_path)
    p = pd.read_parquet(tmp_path / "predictions.parquet")
    assert {"gamma_x1.0|clean", "gamma_x0.1|pgd_self@2.0"} <= set(p.condition)
    assert set(p.row_id) <= set(range(100, 100 + N_TEST))
    c = np.load(tmp_path / "crafted_inputs.npz")
    assert "1|gamma_x0.1|pgd_self@0.5|rbf_svm" in c.files


def test_per_class_writer(monkeypatch, ex, tmp_path):
    import experiments.run_evasion_per_class as pc

    monkeypatch.setattr(pc, "Exports", lambda v, f: ex)
    monkeypatch.setattr(pc, "best_params_path", lambda v, q: REPO_ROOT / "x.json")
    ev, y = np.arange(4), np.array([0, 1, 1, 0])
    pr = np.full((4, 2), 0.5)
    rec = {"macro_f1": 0.5, "attack_recall": 0.5, "false_alarm": 0.5}
    rows = [{"seed": 3, "families": {"mlp": {"clean": rec, "shared": rec, "whitebox": rec}}}]
    arrays = {
        3: {
            "ev": ev,
            "y": y,
            "proba": {("clean", "mlp"): pr, ("pgd@0.5|shared", "mlp"): pr},
            "crafted": {("pgd@0.5|shared", "mlp"): np.zeros((4, 2))},
        }
    }
    pc.write_exports(tmp_path, "gas_binary", rows, arrays)
    m = pd.read_csv(tmp_path / "per_seed.csv")
    assert len(m) == 9 and set(m.condition) == {"clean", "pgd@0.5|shared", "pgd@0.5|whitebox"}
    assert np.load(tmp_path / "crafted_inputs.npz").files == ["3|pgd@0.5|shared|mlp"]


def test_r4_25_unit_arrays_round_trip_and_writer(monkeypatch, ex, tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "r425", REPO_ROOT / "experiments" / "run_gas_rbf_length_scale.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = {
        "ev": np.arange(3),
        "y": np.array([0, 1, 1]),
        "clean": np.full((3, 2), 0.5),
        "proba": {s: np.full((3, 2), 0.5) for s in mod.ATTACK_SEEDS},
        "crafted": {s: np.full((3, 2), float(s)) for s in mod.ATTACK_SEEDS},
    }
    mod.save_unit_arrays(tmp_path / "u.npz", a)
    b = mod.load_unit_arrays(tmp_path / "u.npz")
    assert set(b["proba"]) == set(mod.ATTACK_SEEDS)
    np.testing.assert_array_equal(b["crafted"][7], a["crafted"][7])
    monkeypatch.setattr(mod, "Exports", lambda v, f: ex)
    monkeypatch.setattr(mod, "best_params_path", lambda v, q: REPO_ROOT / "x.json")
    cells = {1.0: {"clean_f1": 0.5, "attacked_f1_per_seed": [0.1] * len(mod.ATTACK_SEEDS)}}
    mod.write_exports(tmp_path, cells, {1.0: b})
    m = pd.read_csv(tmp_path / "per_seed.csv")
    assert len(m) == 1 + len(mod.ATTACK_SEEDS)
    assert len(np.load(tmp_path / "crafted_inputs.npz").files) == len(mod.ATTACK_SEEDS)


def test_p11c_hook_records_without_changing_numbers(ex, tmp_path):
    from experiments.run_whitebox_qsvm import p11c_objective_and_regularization

    reg = _toy_reg(d=2, n_tr=30)
    args = SimpleNamespace(encoding="angle_ry")
    cfg_q = {"ablations": {"bandwidth_curve": {"bandwidths": [1.0]}}}
    bp = {"qk_C": 10.0, "qk_bw": 1.0, "rbf_svm": {"C": 10.0, "gamma": 0.5}}
    kw = {"seed": 0, "c_grid": (1.0,)}
    base = p11c_objective_and_regularization(reg, args, CFG_ADV, cfg_q, bp, 30, **kw)
    hooked = p11c_objective_and_regularization(reg, args, CFG_ADV, cfg_q, bp, 30, ex=ex, **kw)
    assert json.dumps(hooked, sort_keys=True, default=str) == json.dumps(
        base, sort_keys=True, default=str
    )
    ex.write(tmp_path)
    c = np.load(tmp_path / "crafted_inputs.npz")
    assert "1|partA_bw1.0|mass_objective@2.0|qsvm" in c.files
    assert "2|partB_C1.0|mass_descent@2.0|qsvm" in c.files
    p = pd.read_parquet(tmp_path / "predictions.parquet")
    assert {"partB_C1.0|clean", "partB_C1.0|mass_descent@2.0"} <= set(p.condition)


def test_p11e_curves_per_objective_and_the_per_input_reading(ex, tmp_path):
    from experiments.run_whitebox_qsvm import p11e_regularization_radii

    reg = _toy_reg(d=2, n_tr=30)
    args = SimpleNamespace(encoding="angle_ry")
    bp = {"qk_C": 10.0, "qk_bw": 1.0}
    kw = {"seed": 0, "c_grid": (1.0,), "attack_seeds": (0, 1)}
    base = p11e_regularization_radii(reg, args, CFG_ADV, bp, 30, **kw)
    hooked = p11e_regularization_radii(reg, args, CFG_ADV, bp, 30, ex=ex, **kw)
    assert json.dumps(hooked, sort_keys=True, default=str) == json.dumps(
        base, sort_keys=True, default=str
    )
    assert base["c_grid"] == [1.0, 10.0] and set(base["per_c"]) == {"1.0", "10.0"}
    for kind in ("loss", "mass", "random"):
        curve = base[f"per_c_{kind}"]["10.0"]["curve"]
        assert list(curve) == ["0.5", "1.0", "2.0"]
        assert [r["seed"] for r in curve["2.0"]["per_seed"]] == [0, 1]
        assert curve["2.0"]["f1_minus_floor"] == pytest.approx(
            curve["2.0"]["macro_f1"] - base["constant_predictor_floor"]
        )
    # an attacker who picks the radius per input does at least as well as any one radius
    for C, v in base["per_c"].items():
        u = v["per_input_union"]
        for kind in ("loss", "mass"):
            lowest = min(c["macro_f1"] for c in base[f"per_c_{kind}"][C]["curve"].values())
            assert u[kind]["mean"] <= lowest + 1e-12
            assert u["either"]["mean"] <= u[kind]["mean"] + 1e-12
    # one value of the grid alone equals that value in the full run
    one = p11e_regularization_radii(reg, args, CFG_ADV, bp, 30, c_index=1, **kw)
    assert set(one["per_c"]) == {"10.0"}
    assert json.dumps(one["per_c_loss"]["10.0"], sort_keys=True) == json.dumps(
        base["per_c_loss"]["10.0"], sort_keys=True
    )
    ex.write(tmp_path)
    c = np.load(tmp_path / "crafted_inputs.npz")
    assert {"1|C10.0|mass@2.0|qsvm", "0|C1.0|loss@0.5|qsvm", "1|C1.0|random@1.0|random"} <= set(
        c.files
    )
    p = pd.read_parquet(tmp_path / "predictions.parquet")
    assert {"C1.0|clean", "C10.0|loss@2.0", "C10.0|mass@0.5"} <= set(p.condition)
