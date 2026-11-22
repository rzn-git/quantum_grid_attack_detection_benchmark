"""R4-17(b) crossing-count models (run_ray_crossings --count-model), on synthetic counts."""

import importlib.util
import sys

import numpy as np
import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))  # the runner imports experiments.run_ablations
_spec = importlib.util.spec_from_file_location(
    "run_ray_crossings", REPO_ROOT / "experiments" / "run_ray_crossings.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_nb_parity_formula_matches_simulation():
    rng = np.random.default_rng(0)
    for mu, r in [(3.84, 0.4), (3.84, 4.0), (1.0, 1.0), (0.5, 50.0)]:
        sim = rng.negative_binomial(r, r / (r + mu), 400_000)
        assert _mod.nb_p_odd(mu, r) == pytest.approx(np.mean(sim % 2 == 1), abs=3e-3)
    assert _mod.nb_p_odd(2.0, float("inf")) == pytest.approx((1 - np.exp(-4.0)) / 2)
    assert _mod.nb_p_odd(0.0, 1.0) == 0.0


def test_fit_nb_recovers_parameters():
    rng = np.random.default_rng(1)
    c = rng.negative_binomial(2.0, 2.0 / (2.0 + 3.0), 20_000)
    fit = _mod.fit_nb(c)
    assert fit["mu"] == pytest.approx(3.0, rel=0.03)
    assert fit["r"] == pytest.approx(2.0, rel=0.08)


def test_pair_model_recovers_q_and_beats_nb_on_pair_counts():
    rng = np.random.default_rng(2)
    b = rng.random(5000) < 0.34
    c = b.astype(np.int64) + 2 * rng.poisson(1.75, 5000)
    fit = _mod.fit_models(c)
    assert fit["pair"]["q"] == pytest.approx(0.34, abs=0.02)
    assert fit["nb"]["aic"] - fit["pair"]["aic"] > 4


def test_score_cell_flags_pairing_and_passes_nb():
    rng = np.random.default_rng(3)
    pair = (rng.random(100) < 0.34).astype(np.int64) + 2 * rng.poisson(1.75, 100)
    s = _mod.score_cell(pair, 300, seed=0)
    assert s["pairing_cell"] and s["d_par"] <= -0.10
    nb = rng.negative_binomial(1.0, 1.0 / 4.84, 100)
    s_nb = _mod.score_cell(nb, 300, seed=0)
    assert s_nb["in_band"] and not s_nb["pairing_cell"]


def test_ray_stats_counts_and_parity_identity():
    t = np.linspace(0, 1, 200)[:, None]
    F = np.hstack([np.sin(2 * np.pi * 2.25 * t + 0.1), np.cos(2 * np.pi * 0.0 * t), t - 0.5])
    st = _mod.ray_stats(F)
    assert st["counts"].tolist() == [4, 0, 1]
    assert ((st["counts"] % 2 == 1) == st["flip"]).all()  # the H2 identity, by construction


def test_rays_roundtrip(tmp_path):
    st = _mod.ray_stats(np.random.default_rng(4).standard_normal((50, 7)))
    _mod.save_rays(tmp_path / "rays.npz", {"0.8": st})
    back = _mod.load_rays(tmp_path / "rays.npz")[0.8]
    assert back["counts"].tolist() == st["counts"].tolist()
    assert all(
        (a == b).all() for a, b in zip(back["crossing_index"], st["crossing_index"], strict=True)
    )


def test_gap_check_uniform_reference():
    st = {"crossing_index": [np.array([10, 12]), np.array([5]), np.array([0, 100, 198])]}
    g = _mod.gap_check(st, 200)
    assert g["n_gaps"] == 3 and g["obs_short_share"] == pytest.approx(1 / 3)
    want = (1 * (1 - 0.95**2) + 2 * (1 - 0.95**3)) / 3
    assert g["uniform_short_share"] == pytest.approx(want)


def _toy_reg(n=600, d=8):
    from types import SimpleNamespace

    rng = np.random.default_rng(5)
    y = (rng.random(n) < 0.71).astype(np.int64)
    return SimpleNamespace(
        X={"test": rng.standard_normal((n, d))}, y={"test": y}, feature_scale=np.full(d, 2.0)
    )


def test_ray_set_keeps_the_archived_block_and_adds_disjoint_rays():
    from qgridbench.models.classical.zoo import stratified_cap

    reg = _toy_reg()
    ev = stratified_cap(reg.X["test"], reg.y["test"], 100, seed=0)
    want_delta = np.random.default_rng(0).uniform(-1.0, 1.0, (100, 8)) * 0.5 * reg.feature_scale
    one = _mod.ray_set(reg, 100, 100, 0.5)
    assert len(one) == 1
    assert np.array_equal(one[0][0], reg.X["test"][ev])
    assert np.array_equal(one[0][2], want_delta)  # the archived P17e construction, unchanged
    three = _mod.ray_set(reg, 100, 300, 0.5)
    assert np.array_equal(three[0][2], want_delta) and len(three[1][0]) == 200
    base = {tuple(r) for r in three[0][0]}
    assert not base & {tuple(r) for r in three[1][0]}
    with pytest.raises(ValueError):
        _mod.ray_set(reg, 100, 50, 0.5)


def test_control_replicate_uses_the_requested_ray_count():
    out = _mod._control_replicate(("neg", 0, 200))
    assert set(out) == {"kind", "d_par", "in_band", "pairing"}
    assert _mod.CTRL_NEG["min_pass"] == 0.95 and _mod.CTRL_POS["min_pass"] == 0.80
