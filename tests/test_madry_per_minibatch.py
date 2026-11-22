"""R4-08 per-minibatch Madry runner: the eps-0 shortcut, the MLP inner loop, the early-stopping
replay, the model swap and E12b's rule arithmetic, on tiny synthetic fixtures (no data)."""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
MLP_PARAMS = {"h1": 8, "h2": 4, "alpha": 1e-4, "learning_rate_init": 1e-2, "batch_size": 16}


@pytest.fixture(scope="module")
def rm():
    """The runner reads --threads from argv at import and sets the thread variables; both are
    restored after the module's tests."""
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        mp.setattr(sys, "argv", ["pytest", "--threads", "1"])
        for v in THREAD_VARS:
            if v in os.environ:
                mp.setenv(v, os.environ[v])
            else:
                mp.delenv(v, raising=False)
        yield importlib.import_module("experiments.run_madry_per_minibatch")


def _blobs(n=120, d=4, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.6).astype(int)
    X = rng.standard_normal((n, d)) + 1.5 * y[:, None]
    return X, y


def test_pgd_at_zero_radius_returns_the_batch_exactly(rm):
    """The eps-0 arm skips the inner loop; that is output-identical because pgd's random start
    and projection box both collapse to X at eps 0."""
    X, y = _blobs()

    def grad(Z, yy):
        return np.ones_like(Z)

    out = rm.pgd(grad, X, y, 0.0, np.ones(X.shape[1]), n_steps=10, rng=np.random.default_rng(1))
    assert np.array_equal(out, X)


class _Recorder(MLPClassifier):
    calls: list = []

    def partial_fit(self, X, y, sample_weight=None, classes=None):
        type(self).calls.append(np.array(X, copy=True))
        return super().partial_fit(X, y, classes=classes)


def _recording_build(name, params, seed):
    return _Recorder(
        hidden_layer_sizes=(params["h1"], params["h2"]),
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        batch_size=params["batch_size"],
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=seed,
    )


def test_mlp_inner_loop_replaces_every_batch_but_the_first(rm, monkeypatch):
    X, y = _blobs()
    scale = np.full(X.shape[1], 0.7)
    monkeypatch.setattr(rm, "build", _recording_build)
    _Recorder.calls = []
    _, trace = rm.train_mlp_madry(MLP_PARAMS, 0, X, y, 0.5, scale, 10, 0.25, max_epochs=2)
    adv = _Recorder.calls
    _Recorder.calls = []
    rm.train_mlp_madry(MLP_PARAMS, 0, X, y, 0.0, scale, 10, 0.25, max_epochs=2)
    clean = _Recorder.calls
    n_fit = len(y) - int(np.ceil(0.1 * len(y)))  # sklearn's 10% early-stopping hold-out
    assert len(adv) == len(clean) == 2 * int(np.ceil(n_fit / 16))
    assert np.array_equal(adv[0], clean[0])  # no parameters exist before the first step
    for a, c in zip(adv[1:], clean[1:]):
        d = np.abs(a - c)
        assert d.max() <= 0.5 * 0.7 + 1e-12  # inside the eps box
        assert d.max() > 0.1  # and actually moved
    assert trace["clean_steps"] == 1 and trace["epochs"] == 2


@pytest.mark.parametrize("seed", [0, 1, 3])
def test_eps0_replay_is_mlpclassifier_fit(rm, seed):
    """Pre-run amendment 2026-11-16T18:55Z: with sklearn's draw order (init, split, one shuffle per
    epoch) and its stop rule, the eps-0 partial_fit loop is the tuned recipe's own fit."""
    from qgridbench.models.classical.zoo import build, fit_model

    X, y = _blobs(n=300, d=8, seed=seed)
    sk = build("mlp", MLP_PARAMS, seed)
    fit_model("mlp", sk, X, y)
    mlp, trace = rm.train_mlp_madry(MLP_PARAMS, seed, X, y, 0.0, np.ones(8), 10, 0.25)
    assert trace["epochs"] == sk.n_iter_
    assert trace["val_accuracy"] == pytest.approx(sk.validation_scores_, abs=0)
    assert trace["best_val_accuracy"] == sk.best_validation_score_
    for a, b in zip(mlp.coefs_ + mlp.intercepts_, sk.coefs_ + sk.intercepts_, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    assert np.array_equal(mlp.predict(X), sk.predict(X))


def test_with_models_swaps_only_the_named_family(rm):
    class Z:
        pass

    z = Z()
    z.models = {"mlp": "m0", "vqc": "v0", "xgb": "x0"}
    z2 = rm.with_models(z, vqc="v1")
    assert z2.models == {"mlp": "m0", "vqc": "v1", "xgb": "x0"}
    assert z.models["vqc"] == "v0"


def _cell(seed, und, dfd, cls_clean=0.8, cls_att=0.6, collapsed=(), std_collapsed=()):
    fam = {
        f: {
            "undefended": {
                "clean": und[f][0],
                "0.5": und[f][1],
                "constant_on_clean": f in std_collapsed,
            },
            "defended": {
                "clean": dfd[f][0],
                "0.5": dfd[f][1],
                "constant_on_clean": f in collapsed,
            },
        }
        for f in ("mlp", "vqc")
    }
    return {
        "seed": seed,
        "undefended": {f: fam[f]["undefended"] for f in fam},
        "defended": {f: fam[f]["defended"] for f in fam},
        "classical": {
            c: {"clean": cls_clean, "0.5": cls_att}
            for c in ("logreg", "rbf_svm", "rf", "xgb", "lgbm")
        },
    }


def test_read_rules_is_e12b_arithmetic(rm):
    cells = [
        _cell(0, {"mlp": (0.5, 0.3), "vqc": (0.5, 0.1)}, {"mlp": (0.5, 0.36), "vqc": (0.5, 0.1)}),
        _cell(1, {"mlp": (0.5, 0.3), "vqc": (0.5, 0.1)}, {"mlp": (0.4, 0.34), "vqc": (0.5, 0.2)}),
    ]
    r = rm.read_rules(cells, "0.5")
    assert r["families"]["mlp"]["robustness_gain"]["mean"] == pytest.approx(0.05)
    assert r["families"]["mlp"]["robustness_gain"]["per_seed"] == pytest.approx([0.06, 0.04])
    assert r["q1_defense_is_real"]
    assert r["mean_classical_damage"] == pytest.approx(0.2)
    assert r["normalized_vqc_damage_undefended"] == pytest.approx(0.4 / 0.2)
    assert r["normalized_vqc_damage_defended"] == pytest.approx(0.35 / 0.2)
    assert r["q2_picture_changed"]  # |1.75 - 2.0| >= 0.10
    assert r["family_verdicts"]["mlp"].startswith("defense")
    same = {"mlp": (0.5, 0.3), "vqc": (0.5, 0.1)}
    null = rm.read_rules([_cell(0, same, same)], "0.5")
    assert not null["q1_defense_is_real"] and null["q2_picture_changed"] is None
    assert null["branch"] == "per_minibatch_defense_is_a_NULL"


def test_collapsed_seeds_leave_rule_one_and_a_majority_reads_collapsed(rm):
    """Ruling 2 (2026-11-16T13:05Z): a constant defended model sits at the floor clean and
    attacked alike, so its 'gain' must not make rule (1) fire."""
    und = {"mlp": (0.47, 0.28), "vqc": (0.49, 0.03)}
    floor = {"mlp": (0.415, 0.415), "vqc": (0.49, 0.03)}  # the MLP collapsed: +0.135 'gain'
    honest = {"mlp": (0.45, 0.29), "vqc": (0.49, 0.03)}
    cells = [
        _cell(0, und, floor, collapsed=("mlp",)),
        _cell(1, und, floor, collapsed=("mlp",)),
        _cell(2, und, honest),
    ]
    r = rm.read_rules(cells, "0.5")
    m = r["families"]["mlp"]
    assert m["robustness_gain"]["mean"] > 0.05  # the registered all-seed figure would fire
    assert m["collapsed_seeds"] == [0, 1] and m["collapsed_majority"]
    assert m["robustness_gain_non_collapsed"]["per_seed"] == pytest.approx([0.01])
    assert r["family_verdicts"]["mlp"] == "COLLAPSED, not a defense"
    assert not r["q1_defense_is_real"] and r["q2_picture_changed"] is None
    assert r["branch"] == "per_minibatch_defense_is_a_NULL (mlp COLLAPSED)"
    # a minority collapse leaves the family read on its other seeds
    r2 = rm.read_rules([cells[0], cells[2], _cell(3, und, honest)], "0.5")
    assert not r2["families"]["mlp"]["collapsed_majority"]
    assert r2["family_verdicts"]["mlp"].startswith("no defense")


def test_a_constant_standard_fit_collapses_the_seed_in_both_arms(rm):
    """Amendment 2026-11-16T18:55Z: a seed where sklearn's own fit is constant leaves rule (1)."""
    und = {"mlp": (0.47, 0.28), "vqc": (0.49, 0.03)}
    honest = {"mlp": (0.45, 0.29), "vqc": (0.49, 0.03)}
    cells = [
        _cell(0, und, honest),
        _cell(1, {"mlp": (0.415, 0.415), "vqc": (0.49, 0.03)}, honest, std_collapsed=("mlp",)),
        _cell(2, und, honest),
    ]
    m = rm.read_rules(cells, "0.5")["families"]["mlp"]
    assert m["collapsed_standard_fit_seeds"] == [1] and m["collapsed_defended_seeds"] == []
    assert m["collapsed_seeds"] == [1] and not m["collapsed_majority"]
    assert m["robustness_gain_non_collapsed"]["per_seed"] == pytest.approx([0.01, 0.01])


def test_vqc_anchor_follows_the_vqc_configuration(rm, tmp_path, monkeypatch):
    """Ruling 2 of 2026-11-16T13:33Z: depth 4 anchors on E12b, Table I's depth 6 on R4-23."""
    import json

    d4, d6 = tmp_path / "d4.json", tmp_path / "d6.json"
    d4.write_text(json.dumps({"vqc_depth": 4, "vqc_lr": 0.0397}))
    d6.write_text(json.dumps({"vqc_depth": 6, "vqc_lr": 0.0175}))
    monkeypatch.setattr(rm, "best_params_path", lambda v, q: d4)
    monkeypatch.setattr(rm, "TABLE1_PARAMS", d6)
    assert rm.vqc_anchor({"vqc_depth": 4, "vqc_lr": 0.0397, "vqc_epochs": 2}) == "e12b"
    assert rm.vqc_anchor({"vqc_depth": 6, "vqc_lr": 0.0175}) == "r4_23"
    with pytest.raises(ValueError, match="no archived undefended VQC"):
        rm.vqc_anchor({"vqc_depth": 6, "vqc_lr": 0.0397})


def test_undefended_references_take_the_vqc_from_r4_23_depth6_only(rm):
    e12b = {
        "per_seed": [
            {"seed": 0, "undefended": {f: {"clean": 0.4, "0.5": 0.2} for f in ("mlp", "vqc")}}
        ]
    }
    r423 = {
        "units": [
            {
                "arm": a,
                "seed": 0,
                "evasion": {"clean": {"vqc": c}, "curves": {"vqc": {"pgd": {"0.5": p}}}},
            }
            for a, c, p in (("depth4", 0.47, 0.03), ("depth6", 0.48, 0.04))
        ]
    }
    refs = rm.undefended_references(e12b, r423, "r4_23", "0.5")
    assert refs["mlp"] == {0: {"clean": 0.4, "0.5": 0.2}}
    assert refs["vqc"] == {0: {"clean": 0.48, "0.5": 0.04}}
    assert rm.undefended_references(e12b, None, "e12b", "0.5")["vqc"] == {
        0: {"clean": 0.4, "0.5": 0.2}
    }
