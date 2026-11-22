"""Synthetic FDIA generator (R4-22): schema, determinism, physics, generator control, leakage.

Fixture: case14, two days of three 4-snapshot episodes (one each of normal, stealthy and
non-stealthy), written to tmp_path. The load tables live in the config, so nothing downloads.
"""

import copy

import numpy as np
import pytest

from qgridbench.data import fdia_synth as fs
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits

DAYS = np.array([10, 200])


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(fs.fdia_config())
    c["days"]["steps_per_day"] = 12
    c["episodes"].update(
        length=4, per_day=3, composition={"normal": 1, "stealthy": 1, "nonstealthy": 1}
    )
    return c


@pytest.fixture(scope="module")
def generated(cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("fdia_w1")
    meta = fs.generate_variant("fdia14_binary", cfg, out_dir=out, workers=1, days=DAYS)
    return out, meta


def _load(out, name):
    d = np.load(out / name)
    return {k: d[k] for k in d.files}


def test_config_tables_shape():
    lf = fs.fdia_config()["load_factors"]
    assert len(lf["weekly"]) == 52 and len(lf["daily"]) == 7
    assert all(len(v) == 24 for v in lf["hourly"].values()) and len(lf["hourly"]) == 6
    f = fs.step_factors(fs.fdia_config())
    assert f.shape == (364, 96) and 0.2 < f.min() and f.max() <= 1.0


def test_sample_days_stratified():
    d = fs.sample_days(fs.fdia_config())
    assert d.size == 90 and np.unique(d).size == 90
    counts = np.bincount(d // 28, minlength=13)
    assert set(counts) <= {6, 7}


def test_schema(generated):
    out, meta = generated
    v = _load(out, "fdia14_binary.npz")
    aux = _load(out, "fdia14_binary_aux.npz")
    assert v["X"].dtype == np.float32 and v["X"].shape == (16, 82)
    assert set(np.unique(v["y"])) == {0, 1} and np.bincount(v["y"]).tolist() == [8, 8]
    for k in ("day", "episode", "cls", "scale", "J_clean", "J_row", "x_hat", "x_true"):
        assert len(aux[k]) == 16
    assert aux["x_hat"].shape[1] == 27 and meta["chi2_dof"] == 55
    assert meta["class_counts_generated"] == {0: 8, 1: 8, 2: 8}


def test_determinism_and_worker_invariance(cfg, generated, tmp_path):
    out1, _ = generated
    fs.generate_variant("fdia14_binary", cfg, out_dir=tmp_path, workers=2, days=DAYS)
    a, b = _load(out1, "fdia14_binary.npz"), _load(tmp_path, "fdia14_binary.npz")
    assert np.array_equal(a["X"], b["X"]) and np.array_equal(a["y"], b["y"])
    c2 = copy.deepcopy(cfg)
    c2["seed"] = 1
    other = fs.generate_day("case14", 10, fs.step_factors(c2)[10, :12], c2)
    same = fs.generate_day("case14", 10, fs.step_factors(cfg)[10, :12], cfg)
    assert not np.allclose(other["X"], same["X"])


def test_physics_controls(cfg):
    case = fs.Case("case14", cfg)
    assert case.control_h() < 1e-6
    x0 = case.pack(case.vm0, case.va0)
    H = case.jac_x(x0)
    eps = 1e-6
    fd = np.empty_like(H)
    for j in range(x0.size):
        e = np.zeros_like(x0)
        e[j] = eps
        fd[:, j] = (case.h_x(x0 + e) - case.h_x(x0 - e)) / (2 * eps)
    assert np.max(np.abs(fd - H)) / np.max(np.abs(H)) < 1e-6
    xs = x0 + np.random.default_rng(0).normal(0, 0.01, size=(5, x0.size))
    assert np.allclose(case.h_rows(xs), np.stack([case.h_x(x) for x in xs]), rtol=0, atol=1e-12)


def test_generator_control(generated, cfg):
    _, meta = generated
    row = meta["bdd"][str(cfg["estimator"]["alpha"])]
    assert row["agree_stealthy"] >= cfg["controls"]["paired_decision_agreement_min"]
    assert row["pass_nonstealthy"] <= 0.05
    fs.check_controls(meta, cfg)


def test_check_controls_raises(generated, cfg):
    _, meta = generated
    bad = copy.deepcopy(meta)
    bad["controls"]["paired_decision_agreement_stealthy"] = 0.9
    with pytest.raises(RuntimeError):
        fs.check_controls(bad, cfg)


def test_day_folds_disjoint():
    day = np.repeat(np.arange(10), 4)
    seen = []
    for k in range(5):
        s = fs.day_fold_splits(day, k, 5, 0)
        sets = [set(day[s[x]]) for x in fs.SIDES]
        assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
        assert sum(len(s[x]) for x in fs.SIDES) == day.size
        seen += sorted(sets[2])
    assert sorted(seen) == list(range(10))


def test_leakage_transforms_fit_on_train_only(generated):
    out, _ = generated
    v = _load(out, "fdia14_binary.npz")
    sp = _load(out, "splits_fdia14_binary.npz")
    splits = {k: sp[k] for k in fs.SIDES}
    X = v["X"].astype(np.float64)
    pipe = build_preprocessor("pca", n_components=4)
    fit_transform_splits(pipe, X, splits)
    X2 = X.copy()
    X2[splits["test"]] += 100.0
    pipe2 = build_preprocessor("pca", n_components=4)
    fit_transform_splits(pipe2, X2, splits)
    assert np.allclose(pipe["scale"].mean_, pipe2["scale"].mean_)
    assert np.allclose(np.abs(pipe["pca"].components_), np.abs(pipe2["pca"].components_))
