"""R4-07 runner: Square in a projected box, part (b)'s statistic, branch and E-M1c identities,
and part (a)'s branch, on tiny synthetic fixtures (no data, no models)."""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
FAMILIES = ["logreg", "rbf_svm", "rf", "xgb", "lgbm", "mlp", "qsvm", "vqc"]


@pytest.fixture(scope="module")
def tst():
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
        yield importlib.import_module("experiments.run_tree_surrogate_transfer")


def _square_on_box(proba_fn, lo, hi, eps, y, rng, max_iter, p_at):
    """Independent path: the Square schedule written directly on the box [lo, hi] from v = 0."""
    n, d = lo.shape

    def margin(V):
        p = np.clip(proba_fn(V), 1e-12, 1.0)
        return p[np.arange(n), y] - p[np.arange(n), 1 - y]

    X = np.zeros((n, d))
    best = margin(X)
    init = np.clip(eps * rng.choice([-1.0, 1.0], size=(n, d)), lo, hi)
    m0 = margin(init)
    Xa = np.where((m0 < best)[:, None], init, X)
    best = np.minimum(best, m0)
    for i in range(max_iter):
        k = max(1, int(round(p_at(i, max_iter) * d)))
        cols = np.argsort(rng.random((n, d)), axis=1)[:, :k]
        signs = rng.choice([-1.0, 1.0], size=(n, k))
        prop = Xa.copy()
        np.put_along_axis(prop, cols, eps * signs, axis=1)
        prop = np.clip(prop, lo, hi)
        m = margin(prop)
        better = m < best
        Xa[better], best[better] = prop[better], m[better]
    return Xa


def _toy_proba(w):
    def f(V):
        p1 = 1.0 / (1.0 + np.exp(-(V @ w)))
        return np.column_stack([1 - p1, p1])

    return f


def test_square_in_box_equals_square_on_the_box(tst):
    from experiments.run_square_blackbox import _p_at

    rng = np.random.default_rng(3)
    n, d, eps = 12, 7, 0.5
    w = rng.standard_normal(d)
    y = rng.integers(0, 2, n)
    lo, hi = np.full((n, d), -eps), np.full((n, d), eps)
    lo[:, :2] = -rng.uniform(0, eps, (n, 2))  # a floor above -eps
    lo[:3, 4:], hi[:3, 4:] = 0.0, 0.0  # an unattackable block
    seen = []

    def oracle(V):
        seen.append(V.copy())
        return _toy_proba(w)(V)

    v, q = tst.square_in_box(oracle, lo, hi, eps, y, np.random.default_rng(11), 40)
    ref = _square_on_box(_toy_proba(w), lo, hi, eps, y, np.random.default_rng(11), 40, _p_at)
    np.testing.assert_array_equal(v, ref)
    assert q == 42.0  # clean, init, 40 iterations
    assert all(np.all(V >= lo) and np.all(V <= hi) for V in seen)
    assert np.all(v[:3, 4:] == 0.0)


def test_square_in_box_rejects_a_box_outside_eps(tst):
    lo, hi = np.full((2, 3), -0.6), np.full((2, 3), 0.5)
    with pytest.raises(ValueError, match="inside"):
        tst.square_in_box(_toy_proba(np.ones(3)), lo, hi, 0.5, np.array([0, 1]), None, 1)


def _units_and_em1c(e_rf, e_lgbm, xgb_square=0.30, eps_grid=(0.1, 0.25, 0.5)):
    """Per seed: random F1 0.5 everywhere; transfer F1 0.5 - E for rf / lgbm at every eps."""
    units, em1c = [], {"records": [], "transfer_matrix": []}
    for s in (0, 1, 2):
        recs = []
        for f in FAMILIES:
            recs.append(
                {
                    "seed": s,
                    "eps": 0.0,
                    "family": f,
                    "kind": "projection_only",
                    "clean_f1": 0.6,
                    "attacked_f1": 0.59,
                }
            )
            for e in eps_grid:
                t = {"rf": 0.5 - e_rf, "lgbm": 0.5 - e_lgbm, "xgb": xgb_square}.get(f, 0.49)
                recs.append(
                    {
                        "seed": s,
                        "eps": e,
                        "family": f,
                        "kind": "transfer_xgb",
                        "clean_f1": 0.6,
                        "attacked_f1": t,
                    }
                )
                recs.append(
                    {
                        "seed": s,
                        "eps": e,
                        "family": f,
                        "kind": "random",
                        "clean_f1": 0.6,
                        "attacked_f1": 0.5,
                    }
                )
        units.append({"seed": s, "records": recs})
        em1c["records"] += [
            {**r, "mode": "joint"} for r in recs if r["kind"] in ("projection_only", "random")
        ]
    for f in FAMILIES:
        for e in eps_grid:
            em1c["transfer_matrix"].append(
                {
                    "source": "mlp",
                    "target": f,
                    "eps": e,
                    "transfer_minus_random_damage": {"mean": 0.01},
                }
            )
    return units, em1c, list(eps_grid)


def test_part_b_physics_branch_and_identities(tst):
    units, em1c, grid = _units_and_em1c(0.049, 0.02)
    out = tst.read_consistent(units, grid, em1c)
    assert out["statistic"]["E_xgb_to"] == pytest.approx({"rf": 0.049, "lgbm": 0.02})
    assert out["branch"] == "PHYSICS_REMOVES_IT_FOR_TREES_TOO"
    assert out["complete"]
    ctl = out["controls"]
    assert ctl["positive_pass"]
    assert ctl["clean_equals_em1c"]["pass"] and ctl["random_equals_em1c"]["pass"]
    assert out["statistic"]["em1c_mlp_source_E_same_cells"]["rf|0.5"] == 0.01


def test_part_b_one_tree_over_the_margin_survives(tst):
    units, em1c, grid = _units_and_em1c(0.0, 0.06)
    assert tst.read_consistent(units, grid, em1c)["branch"] == "TREE_TRANSFER_SURVIVES"


def test_part_b_controls_fail_loud_in_the_reading(tst):
    units, em1c, grid = _units_and_em1c(0.0, 0.0, xgb_square=0.5)  # Square no better than random
    em1c["records"][1]["attacked_f1"] += 1e-9  # one replayed random cell off E-M1c
    ctl = tst.read_consistent(units, grid, em1c)["controls"]
    assert not ctl["positive_pass"]
    assert not ctl["random_equals_em1c"]["pass"]
    assert ctl["random_equals_em1c"]["max_abs_diff"] == pytest.approx(1e-9)


def _ret(**over):
    base = {f: 0.6 for f in FAMILIES}
    base["qsvm"] = 0.88
    return {**base, **over}


def test_part_a_branches(tst):
    ref = _ret()
    assert tst.branch_pca(_ret(rf=0.64), ref)["verdict"] == "SURROGATE_ROBUST"
    assert tst.branch_pca(_ret(rf=0.71), ref)["verdict"] == "SURROGATE_DEPENDENT"
    assert tst.branch_pca(_ret(rf=0.68), ref)["verdict"] == "INTERMEDIATE"
    close = _ret(qsvm=0.65, logreg=0.62)
    lost = tst.branch_pca(_ret(qsvm=0.62, logreg=0.64), close)  # changes under 0.05, lead lost
    assert lost["qsvm_lost_lead"] and lost["verdict"] == "SURROGATE_DEPENDENT"
    assert tst.branch_pca(_ret(vqc=0.48), ref)["families_moved_at_least_0.10"] == ["vqc"]


def test_table1_params_are_the_grid_file_plus_the_depth6_overrides(tst):
    grid = {"xgb": {"max_depth": 4}, "vqc_depth": 4, "vqc_lr": 0.04, "vqc_batch": 128}
    over = {"vqc_depth": 6, "vqc_lr": 0.0175, "vqc_batch": 128}
    tst.check_table1_params(grid, {**grid, **over}, over)
    with pytest.raises(AssertionError, match="xgb"):
        tst.check_table1_params(grid, {**grid, **over, "xgb": {"max_depth": 5}}, over)
    with pytest.raises(AssertionError, match="vqc_lr"):
        tst.check_table1_params(grid, {**grid, "vqc_depth": 6}, over)
    with pytest.raises(AssertionError, match="keys differ"):
        tst.check_table1_params(grid, {**grid, **over, "extra": 1}, over)


def test_one_worker_runs_units_in_process_and_skips_resumed(tst, tmp_path, monkeypatch):
    import torch

    t = torch.get_num_threads()
    for v in THREAD_VARS:
        monkeypatch.setenv(v, str(t))
    resume = tmp_path / "partial.json"
    resume.write_text('{"units": [{"unit": "s0", "seed": 0}]}')
    written = []
    run = type("Run", (), {"write_json": lambda self, name, obj: written.append(obj)})()
    specs = [{"unit": f"s{s}", "seed": s} for s in (0, 1, 2)]
    pid = os.getpid()
    done = tst._run(
        lambda sp: {"unit": sp["unit"], "seed": sp["seed"], "pid": os.getpid()},
        specs,
        type("Sh", (), {"threads": t})(),
        1,
        run,
        resume,
    )
    assert [u["unit"] for u in done] == ["s0", "s1", "s2"]
    assert all(u["pid"] == pid for u in done[1:])
    assert len(written) == 2
