"""R4-13 attack-strength control: stream offset, restart selection, strongest attack, the branch
lines and the anchors, on tiny fixtures (no model, no torch)."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def m():
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        yield importlib.import_module("experiments.run_vqc_encoding_attack_control")


def test_stream_offset_matches_the_published_order(m):
    grid = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    assert m.stream_draws_before(grid, 0.5) == 11  # the arm's 0.5-sigma set is the 12th draw
    assert m.stream_draws_before(grid, 0.01) == 1  # after the MLP's first set


def test_stream_offset_reproduces_a_replayed_draw(m):
    from qgridbench.utils.seeding import spawn_rng

    grid, shape = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5], (3, 2)
    full = spawn_rng(4, "adv")
    draws = [full.uniform(-1.0, 1.0, size=shape) for _ in range(12)]
    rng = spawn_rng(4, "adv")
    for _ in range(m.stream_draws_before(grid, 0.5)):
        rng.uniform(-1.0, 1.0, size=shape)
    assert np.array_equal(rng.uniform(-1.0, 1.0, size=shape), draws[11])


def test_worst_over_restarts_picks_the_highest_loss_per_row(m):
    losses = np.array([[0.1, 0.9, 0.2], [0.5, 0.3, 0.2], [0.4, 0.1, 0.8]])
    assert m.worst_over_restarts(losses).tolist() == [1, 0, 2]


def _cells(pgd20, pgd100, sq):
    return {
        "pgd20": {"adv_mean": pgd20},
        "pgd100x5": {"adv_mean": pgd100},
        "square": {"adv_mean": sq},
    }


def test_strongest_is_the_lowest_mean(m):
    assert m.strongest(_cells(0.17, 0.12, 0.30)) == "pgd100x5"


@pytest.mark.parametrize(
    ("a2", "a1", "branch"),
    [
        (0.33, 0.06, "HOLDS"),
        (0.10, 0.06, "COLLAPSES"),
        (0.25, 0.06, "INTERMEDIATE"),
        (0.33, 0.25, "INTERMEDIATE"),
    ],
)
def test_verdict_lines(m, a2, a1, branch):
    ret = {"A2_zz": {"pgd100x5": a2}, "A1_angle": {"pgd100x5": a1}}
    strong = {"A2_zz": "pgd100x5", "A1_angle": "pgd100x5"}
    assert m.verdict(ret, strong)["branch"] == branch


def _unit(arm, seed, clean, pgd20, pgd100, sq):
    return {
        "arm": arm,
        "seed": seed,
        "clean": clean,
        "adv": {"pgd20": pgd20, "pgd100x5": pgd100, "square": sq},
        "worst_of_three": min(pgd20, pgd100, sq),
    }


def test_summarize_reads_retention_under_each_arms_strongest_attack(m):
    units = [_unit("A2_zz", s, 0.5, 0.17, 0.16, 0.30) for s in range(3)]
    units += [_unit("A1_angle", s, 0.5, 0.03, 0.02, 0.20) for s in range(3)]
    s = m.summarize(units)
    assert s["strongest"] == {"A1_angle": "pgd100x5", "A2_zz": "pgd100x5"}
    assert s["verdict"]["a2_retention_strongest"] == pytest.approx(0.32)
    assert s["verdict"]["branch"] == "HOLDS"


def test_anchors_compare_clean_and_pgd20_with_the_main_run(m):
    main = {"A2_zz_s0": {"clean": {"eval500": 0.5}, "r2": {"pgd": {"0.5": 0.17}}}}
    assert m.anchors([_unit("A2_zz", 0, 0.5, 0.17, 0.1, 0.3)], main)["pass"]
    assert not m.anchors([_unit("A2_zz", 0, 0.5, 0.171, 0.1, 0.3)], main)["pass"]
