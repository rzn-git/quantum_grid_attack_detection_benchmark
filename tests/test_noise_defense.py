"""R4-14 runner: the M-draw predictor, the EOT gradient, sigma selection under the clean-cost cap
and the step-1 verdicts, on tiny synthetic fixtures (no data, no fitted models)."""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")


@pytest.fixture(scope="module")
def nd():
    # the runner pins these at import. They are put back by hand: MonkeyPatch.delenv records
    # nothing for a variable that is not set, so it left QGB_VQC_CHUNK at 256 for later files
    saved = {v: os.environ.get(v) for v in (*THREAD_VARS, "QGB_VQC_CHUNK")}
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.syspath_prepend(str(Path(__file__).resolve().parents[1]))
            mp.setattr(sys, "argv", ["pytest", "--threads", "1"])
            yield importlib.import_module("experiments.run_noise_defense")
    finally:
        # also when the import fails part-way: the runner pins before its other imports
        for v, value in saved.items():
            if value is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = value


class StubZoo:
    """predict_proba = logistic of the row sum; the family name is ignored."""

    def predict_proba(self, fam, X):
        p1 = 1.0 / (1.0 + np.exp(-X.sum(axis=1)))
        return np.column_stack([1.0 - p1, p1])


def test_defended_sigma_zero_is_the_undefended_model(nd):
    X = np.random.default_rng(0).normal(size=(5, 3))
    D = nd.Defended(StubZoo(), "mlp", 0.0, np.ones(3), None)
    assert np.array_equal(D.proba(X), StubZoo().predict_proba("mlp", X))


def test_defended_is_the_mean_over_fixed_draws(nd):
    rng = np.random.default_rng(1)
    X, scale = rng.normal(size=(4, 3)), np.array([1.0, 2.0, 0.5])
    E = rng.standard_normal((6, 4, 3))
    D = nd.Defended(StubZoo(), "mlp", 0.3, scale, E)
    want = np.mean([StubZoo().predict_proba("mlp", X + 0.3 * scale * E[m]) for m in range(6)], 0)
    assert np.allclose(D.proba(X), want, atol=1e-15)
    assert np.array_equal(D.proba(X), D.proba(X))  # deterministic on its rows
    with pytest.raises(AssertionError):
        D.proba(X[:2])


def test_eot_averages_k_fresh_draws_and_is_identity_at_sigma_zero(nd):
    def grad(X, y):
        return 2.0 * X + y[:, None]

    X, y, scale = np.ones((3, 2)), np.array([0.0, 1.0, 0.0]), np.array([1.0, 3.0])
    assert nd.eot(grad, 0.0, scale, 8, np.random.default_rng(0)) is grad
    g = nd.eot(grad, 0.5, scale, 8, np.random.default_rng(7))(X, y)
    eta = np.random.default_rng(7).standard_normal((8, 3, 2))
    want = np.mean([grad(X + 0.5 * scale * eta[k], y) for k in range(8)], axis=0)
    assert np.allclose(g, want, atol=1e-15)


def _unit(seed, sigma, clean, attacked, ok=True):
    fams = {
        f: {"clean": clean, "attacked": attacked, "clean_ok": ok}
        for f in ["logreg", "rbf_svm", "rf", "xgb", "lgbm", "mlp", "qsvm", "vqc"]
    }
    return {"seed": seed, "sigma": sigma, "families": fams}


def test_select_respects_the_cost_cap_and_breaks_ties_to_the_smaller_sigma(nd):
    units = []
    for s in (0, 1):
        units += [
            _unit(s, 0.0, 0.80, 0.10),
            _unit(s, 0.5, 0.76, 0.40),  # inside the cap: eligible
            _unit(s, 1.0, 0.76, 0.40),  # ties 0.5: the smaller sigma wins
            _unit(s, 2.0, 0.70, 0.60),  # outside the cap (0.10 drop)
            _unit(s, 4.0, 0.79, 0.90, ok=False),  # a constant predictor: never eligible
        ]
    sel = nd.select(units, (0.0, 0.5, 1.0, 2.0, 4.0))
    assert sel["mlp"]["selected_sigma"] == 0.5
    assert sel["mlp"]["eligible"] == [0.0, 0.5, 1.0]


def _merged(nd, gain_eot, gain_single):
    m = {}
    for f in nd.FAMILIES:
        m[f] = {
            "undefended": {"whitebox_0.5": 0.1},
            "defended": {
                "whitebox_0.5": 0.1 + gain_eot.get(f, 0.0),
                "whitebox_single_draw_0.5": 0.1 + gain_single.get(f, 0.0),
            },
        }
    return m


def test_step1_verdicts(nd):
    assert nd.step1(_merged(nd, {"qsvm": 0.06}, {}))["verdict"] == "REAL"
    assert nd.step1(_merged(nd, {"mlp": 0.01}, {"mlp": 0.30}))["verdict"] == "OBFUSCATED"
    assert nd.step1(_merged(nd, {"vqc": 0.049}, {"vqc": 0.049}))["verdict"] == "NULL"


def test_defended_mean_is_unchanged_by_draw_blocks(nd):
    rng = np.random.default_rng(3)
    X, scale = rng.normal(size=(4, 3)), np.ones(3)
    E = rng.standard_normal((nd.DRAW_BLOCK * 2 + 3, 4, 3))  # not a multiple of the block
    D = nd.Defended(StubZoo(), "mlp", 0.4, scale, E)
    want = np.mean([StubZoo().predict_proba("mlp", X + 0.4 * scale * e) for e in E], axis=0)
    assert np.allclose(D.proba(X), want, atol=1e-14)


def test_ruling_6_switches_every_cell_to_the_adequacy_draws(nd):
    within = {"mc_check": {"vqc": {"abs_diff": 0.004}, "qsvm": {"abs_diff": 0.005}}}
    over = {"mc_check": {"vqc": {"abs_diff": 0.004}, "qsvm": {"abs_diff": 0.0051}}}
    assert nd.prediction_draws(within, 16, 64) | {"max_abs_diff": 0} == {
        "primary": 16,
        "beside": None,
        "max_abs_diff": 0,
        "tol": 0.005,
        "switched": False,
    }
    r = nd.prediction_draws(over, 16, 64)
    assert (r["primary"], r["beside"], r["switched"]) == (64, 16, True)


def test_beside_units_swap_in_the_m_draw_values(nd):
    u = {
        "seed": 0,
        "defended": {
            "vqc": {"sigma": 0.5, "clean": 0.7, "beside": {"prediction_draws": 16, "clean": 0.6}},
            "mlp": {"sigma": 0.0, "undefended": True},
        },
    }
    b = nd.beside_units([u])[0]["defended"]
    assert b["vqc"] == {"prediction_draws": 16, "clean": 0.6, "sigma": 0.5}
    assert b["mlp"] == {"sigma": 0.0, "undefended": True}
    assert u["defended"]["vqc"]["clean"] == 0.7  # the input is not modified


def _write(path, obj):
    import json

    path.write_text(json.dumps(obj))
    return path


def test_tuned_params_take_table1_vqc_keys_and_check_r4_23(nd, tmp_path):
    t3 = _write(
        tmp_path / "t3.json",
        {
            "mlp": {"a": 1},
            "vqc_depth": 4,
            "vqc_lr": 0.04,
            "vqc_batch": 128,
            "vqc_epochs": 40,
            "qk_bw": 1.0,
        },
    )
    t1 = _write(
        tmp_path / "t1.json", {"mlp": {"a": 9}, "vqc_depth": 6, "vqc_lr": 0.0175, "vqc_batch": 128}
    )
    d6 = {"vqc_depth": 6, "vqc_lr": 0.0175, "vqc_batch": 128, "vqc_epochs": 40}
    r = _write(
        tmp_path / "r.json",
        {
            "units": [
                {"arm": "depth4", "vqc_config": {**d6, "vqc_depth": 4}},
                {"arm": "depth6", "vqc_config": d6},
                {"arm": "depth6", "vqc_config": d6},
            ]
        },
    )
    bp = nd.tuned_params(t3, t1, r)
    assert bp["mlp"] == {"a": 1} and bp["qk_bw"] == 1.0  # only the VQC keys move
    assert {k: bp[k] for k in nd.VQC_KEYS} == d6
    bad = _write(
        tmp_path / "bad.json", {"units": [{"arm": "depth6", "vqc_config": {**d6, "vqc_lr": 0.02}}]}
    )
    with pytest.raises(AssertionError):
        nd.tuned_params(t3, t1, bad)


def test_vqc_chunk_is_pinned_before_the_vqc_is_built(nd):
    assert os.environ["QGB_VQC_CHUNK"] == str(nd.VQC_CHUNK) == "256"
    from qgridbench.models.quantum.vqc import resolve_chunk

    assert resolve_chunk(8, 6) == nd.VQC_CHUNK


@pytest.mark.parametrize("kind", ["rbf_svm", "mlp"])
def test_eot_row_blocks_are_bit_identical_to_the_whole_stack(nd, monkeypatch, kind):
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    from qgridbench.attacks.gradients import GradProvider

    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(300, 8))
    ytr = (Xtr[:, 0] + 0.5 * Xtr[:, 1] > 0).astype(int)
    if kind == "rbf_svm":
        est = SVC(kernel="rbf", gamma=0.3, C=1.0).fit(Xtr, ytr)
    else:
        est = MLPClassifier(hidden_layer_sizes=(16,), max_iter=300, random_state=0).fit(Xtr, ytr)
    gp = GradProvider(est, kind)
    X, y, scale = rng.normal(size=(50, 8)), rng.integers(0, 2, 50), np.full(8, 0.7)
    monkeypatch.setattr(nd, "EOT_BLOCK", 64)  # 4 x 50 = 200 rows: four blocks, the last partial
    blocked = nd.eot(gp, 0.5, scale, 4, np.random.default_rng(11))(X, y)
    eta = np.random.default_rng(11).standard_normal((4, 50, 8))
    whole = gp((X[None] + 0.5 * scale * eta).reshape(200, 8), np.tile(y, 4))
    assert np.array_equal(blocked, whole.reshape(4, 50, 8).mean(axis=0))


class StubAttackZoo(StubZoo):
    """A zoo stand-in for the Table III replay: a per-row gradient and a two-class predictor."""

    def clean_proba(self, fam, X, y):
        return self.predict_proba(fam, X)

    def grad_provider(self, fam):
        return lambda X, y: X - y[:, None]


def test_replay_consumes_table_iii_stream_exactly(nd):
    """The replay's sensor-noise cells equal run_adversarial's own evasion + noise stages on the
    same stream: every random start in evasion_and_transfer (its transfer-matrix row draws one
    per family) is consumed before the noise draws."""
    from experiments.run_adversarial import evasion_and_transfer, sensor_noise
    from qgridbench.utils.run_tracking import REPO_ROOT, load_yaml
    from qgridbench.utils.seeding import spawn_rng

    cfg = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    rng = np.random.default_rng(5)
    X, y, scale = rng.normal(size=(30, 8)), rng.integers(0, 2, 30), np.full(8, 0.9)
    c = {"cfg_adv": cfg, "Xte": X, "yte": y, "scale": scale, "n_steps": 2, "step_frac": 0.25}
    zoo = StubAttackZoo()
    ours = nd.replay_table_iii(c, zoo, seed=3)["noise"]
    cfg2 = {**cfg, "evasion": {**cfg["evasion"], "pgd": {**cfg["evasion"]["pgd"], "n_steps": 2}}}
    r = spawn_rng(3, "adv")
    evasion_and_transfer(zoo, X, y, cfg2, r, scale)
    ref = sensor_noise(zoo, X, y, cfg2, r, scale)
    assert {snr: {f: ref[f][int(snr)] for f in nd.FAMILIES} for snr in ours} == ours


def test_anchor_only_unit_returns_the_replay_cells(nd, monkeypatch):
    from qgridbench.utils.run_tracking import REPO_ROOT, load_yaml

    cfg = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    rng = np.random.default_rng(8)
    X, y, scale = rng.normal(size=(30, 8)), rng.integers(0, 2, 30), np.full(8, 0.9)
    c = {"cfg_adv": cfg, "Xte": X, "yte": y, "scale": scale, "n_steps": 2, "step_frac": 0.25}
    monkeypatch.setattr(nd, "ctx", lambda smoke: c)
    monkeypatch.setattr(nd, "fit_zoo", lambda c_, sigma, seed: StubAttackZoo())
    u = nd.unit_anchor({"seed": 4})
    want = nd.replay_table_iii(c, StubAttackZoo(), 4)
    assert u["stage"] == "anchor" and u["seed"] == 4
    assert u["anchor_values"]["noise"] == want["noise"]
    assert u["anchor_values"]["published_pgd_0.5"] == want["published_pgd_0.5"]
    assert set(u["anchor_values"]["clean"]) == set(nd.FAMILIES)
    assert u["peak_private_gb"] > 0


def test_pgd_restarts_keep_each_rows_highest_loss_end_point(nd, monkeypatch):
    monkeypatch.setattr(nd, "STRENGTH_STEPS", 3)
    monkeypatch.setattr(nd, "STRENGTH_RESTARTS", 4)
    zoo = StubZoo()

    def predict(Z):
        return zoo.predict_proba("mlp", Z)

    rng = np.random.default_rng(2)
    X, y, scale = rng.normal(size=(40, 8)), rng.integers(0, 2, 40), np.full(8, 1.0)

    def grad(X_, y_):
        return X_ - y_[:, None]

    best = nd.pgd_restarts(lambda r: grad, predict, X, y, 0.5, scale, 7, "t")
    ends = [
        nd.pgd(
            grad,
            X,
            y,
            0.5,
            scale,
            n_steps=3,
            step_frac=nd.STRENGTH_STEP_FRAC,
            rng=nd.spawn_rng(7, f"t_start{r}"),
        )
        for r in range(4)
    ]
    losses = np.stack([nd.ce_loss(predict(e), y) for e in ends])
    want = np.stack(ends)[losses.argmax(axis=0), np.arange(40)]
    assert np.array_equal(best, want)


def _strength_unit(seed, und, dfd):
    keys = ("clean", "pgd20", "pgd100x5", "square")
    return {
        "seed": seed,
        "undefended": dict(zip(keys, und, strict=True)),
        "defended": dict(zip(keys, dfd, strict=True)),
    }


def _run_b(clean_u, pgd_u, clean_d, pgd_d):
    return {
        "undefended": {"vqc": {"clean": clean_u, "whitebox_0.5": pgd_u}},
        "defended": {"vqc": {"clean": clean_d, "whitebox": {"0.5": pgd_d}}},
    }


def test_strength_reading_takes_each_arms_strongest_attack(nd):
    units = [_strength_unit(s, (0.5, 0.10, 0.05, 0.08), (0.5, 0.30, 0.12, 0.20)) for s in (0, 1)]
    run_b = {s: _run_b(0.5, 0.10, 0.5, 0.30) for s in (0, 1)}
    r = nd.strength_reading(units, run_b)
    assert r["strongest_attack"] == {"undefended": "pgd100x5", "defended": "pgd100x5"}
    assert r["gain_under_strongest"] == pytest.approx(0.07)
    assert r["verdict"] == "HOLDS" and r["reproduction_pass"] and r["no_attack_above_clean"]
    assert not r["defended_above_floor"]  # 0.12 against the 0.415 floor
    for u in units:  # Square becomes the strongest defended attack: 0.06 - 0.05 < 0.05
        u["defended"]["square"] = 0.06
    r = nd.strength_reading(units, run_b)
    assert r["strongest_attack"]["defended"] == "square" and r["verdict"] == "COLLAPSES"
    units[1]["undefended"]["square"] = 0.9  # an attack above clean is flagged
    assert not nd.strength_reading(units, run_b)["no_attack_above_clean"]
    run_b[0] = _run_b(0.5, 0.11, 0.5, 0.30)  # a PGD-20 cell that does not reproduce
    assert not nd.strength_reading(units, run_b)["reproduction_pass"]


def test_blackbox_axis_reads_nd_from_the_square_cells(nd):
    def cell(clean, sq):
        return {"clean": clean, "square": sq}

    axis = []
    for s in (0, 1):
        cells = {
            c: {"undefended": cell(0.6, 0.4), "defended": cell(0.6, 0.4)}
            for c in nd.CLASSICAL_TARGETS
        }
        cells["qsvm"] = {"undefended": cell(0.6, 0.2), "defended": cell(0.6, 0.2)}
        axis.append({"seed": s, "cells": cells})
    strength = [
        {"seed": s, "undefended": cell(0.5, 0.1), "defended": cell(0.5, 0.4)} for s in (0, 1)
    ]
    out = nd.blackbox_axis(strength, axis)
    assert out["classical_damage"] == {
        "undefended": pytest.approx(0.2),
        "defended": pytest.approx(0.2),
    }
    assert out["quantum"]["vqc"]["nd"] == {
        "undefended": pytest.approx(2.0),
        "defended": pytest.approx(0.5),
    }
    assert out["quantum"]["vqc"]["moves"] and not out["quantum"]["qsvm"]["moves"]


def test_write_exports_per_seed_csv_predictions_and_crafted_inputs(nd, tmp_path):
    import pandas as pd

    n = 6
    rng = np.random.default_rng(1)
    y, row_ids = rng.integers(0, 2, n), np.arange(100, 100 + n)
    (tmp_path / "unit_arrays").mkdir()

    def p():
        a = rng.random(n)
        return np.column_stack([1 - a, a])

    arrays = {}
    for arm in ("undefended", "defended"):
        arrays[f"vqc|{arm}|clean|proba"] = p()
        for cond in nd.ATTACKS:
            arrays[f"vqc|{arm}|{cond}|proba"] = p()
            arrays[f"vqc|{arm}|{cond}|X"] = rng.normal(size=(n, 8))
    np.savez_compressed(tmp_path / "unit_arrays" / "strength_s0.npz", **arrays)
    np.savez_compressed(
        tmp_path / "unit_arrays" / "axis_s0.npz",
        **{
            "rf|undefended|clean|proba": p(),
            "rf|undefended|square|proba": p(),
            "rf|undefended|square|X": rng.normal(size=(n, 8)),
        },
    )
    rec = {"clean": 0.6, "pgd20": 0.1, "pgd100x5": 0.05, "square": 0.2, "square_queries": 602.0}
    strength = [{"seed": 0, "undefended": rec, "defended": rec}]
    cell = {"clean": 0.6, "square": 0.4, "square_queries": 602.0, "sigma": 0.0}
    axis = [{"seed": 0, "cells": {"rf": {"undefended": cell, "defended": cell}}}]
    out = nd.write_exports(tmp_path, strength, axis, row_ids, y)
    csv = pd.read_csv(tmp_path / "per_seed.csv")
    assert len(csv) == out["per_seed_rows"] == 2 * 5 + 2 * 3
    pq = pd.read_parquet(tmp_path / "predictions.parquet")
    assert len(pq) == out["prediction_rows"] == (8 + 2) * n
    sel = pq[(pq.family == "vqc") & (pq.arm == "defended") & (pq.condition == "square")]
    assert np.array_equal(sel.row_id, row_ids) and np.array_equal(sel.y_true, y)
    assert np.allclose(sel.prob_attack, arrays["vqc|defended|square|proba"][:, 1])
    with np.load(tmp_path / "crafted_inputs.npz") as z:
        assert sorted(z.files) == sorted(
            [f"0|vqc|{a}|{c}" for a in ("undefended", "defended") for c in nd.ATTACKS]
            + ["0|rf|undefended|square"]
        )
        assert np.array_equal(
            z["0|rf|undefended|square"],
            np.load(tmp_path / "unit_arrays" / "axis_s0.npz")["rf|undefended|square|X"],
        )
