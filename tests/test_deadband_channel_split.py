"""R4-02 (NS9) runner: the channel split, the Stage 1 stream replay, the deadband count, the
per-class split, the statistic and the branch, on tiny synthetic fixtures (no data, no models)."""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")


@pytest.fixture(scope="module")
def ns9():
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
        yield importlib.import_module("experiments.run_deadband_channel_split")


def _columns():
    """Four relays in the dataset's naming, each with its status flag, plus one log channel."""
    cols = []
    for r in (1, 2, 3, 4):
        cols += [f"R{r}-PM{k}:{'V' if k <= 3 or 7 <= k <= 9 else 'I'}" for k in range(1, 13)]
        cols += [f"R{r}-PA{k}:{'V' if k <= 3 or 7 <= k <= 9 else 'I'}H" for k in range(1, 13)]
        cols += [f"R{r}-PA:Z", f"R{r}-PA:ZH", f"R{r}:F", f"R{r}:DF", f"R{r}:S"]
    return cols + ["snort_log1"]


def _a1(cols):
    return np.array([not (c.endswith(":S") or c.startswith("snort")) for c in cols])


def test_part_masks_split_a1_into_56_controlled_and_56_derived(ns9):
    cols = _columns()
    masks, seq = ns9.part_masks(cols, _a1(cols))
    ctrl = [c for c, m in zip(cols, masks["controlled"], strict=True) if m]
    der = [c for c, m in zip(cols, masks["derived"], strict=True) if m]
    assert len(ctrl) == len(der) == 56
    assert "R3-PM4:I" in ctrl and "R3:DF" in ctrl and "R3-PA6:IH" in ctrl
    assert "R3-PM10:I" in der and "R3-PA:ZH" in der and "R3-PA9:VH" in der
    assert not set(ctrl) & set(der)
    assert len(seq["pm"]) == len(seq["pa"]) == 24
    assert [cols[i] for i in seq["pm"][:6]] == [
        f"R1-PM{k}:{'V' if k <= 9 else 'I'}" for k in range(7, 13)
    ]
    assert [cols[i] for i in seq["pa"][:6]] == [
        f"R1-PA{k}:{'V' if k <= 9 else 'I'}H" for k in range(7, 13)
    ]
    assert seq["names"][:6] == ["PM7", "PM8", "PM9", "PM10", "PM11", "PM12"]


def test_part_masks_raise_when_parts_are_not_a1(ns9):
    cols = _columns()
    a1 = _a1(cols)
    a1[cols.index("R2:S")] = True
    with pytest.raises(AssertionError, match="not A1"):
        ns9.part_masks(cols, a1)


def test_stage1_skips_follow_stage1_loop_order(ns9):
    # arms A0, A1, ...; eps 0.1, 0.25, 0.5; sources mlp, qsvm, rbf_svm, vqc
    assert [ns9.stage1_pgd_skip(e) for e in range(3)] == [12, 16, 20]
    assert ns9.stage1_pgd_skip(0, arm="A0", source="vqc") == 3
    assert [ns9.stage1_random_skip(e) for e in range(3)] == [3, 4, 5]


def test_stage1_draw_replays_a_sequential_stream(ns9):
    shape = (5, 7)

    def unif(g):
        return g.uniform(-1.0, 1.0, size=shape)

    seq = ns9.spawn_rng(2, "m1a")
    draws = [unif(seq) for _ in range(21)]  # Stage 1's calls up to and including A1 eps 0.5 mlp
    replay = ns9.stage1_draw(2, "m1a", ns9.stage1_pgd_skip(2), unif)
    np.testing.assert_array_equal(unif(replay), draws[20])


def test_deadband_moves_counts_only_zero_channels_that_move(ns9):
    seq = {"pm": np.array([0, 1]), "pa": np.array([2, 3]), "names": ["PM7", "PM8"]}
    x0 = np.array(
        [
            [0.0, 5.0, 0.0, 9.0],  # PM7 in deadband
            [0.0, 0.0, 0.0, 0.0],  # both in deadband
            [3.0, 4.0, 1.0, 2.0],  # none
        ]
    )
    d = np.array(
        [
            [0.0, 0.0, 0.2, 0.0],  # PM7 moved through its angle
            [0.0, 0.0, 0.0, 0.0],  # nothing moved
            [0.5, 0.5, 0.5, 0.5],  # moved, but nothing was in deadband
        ]
    )
    out = ns9.deadband_moves(x0, d, seq)
    assert out["n_rows_with_deadband_zero"] == 2
    assert out["n_rows_moved_off_zero"] == 1
    assert out["by_channel"] == {"PM7": 1, "PM8": 0}


def test_class_split_and_share(ns9):
    clean = {"attack_recall": 0.8, "false_alarm": 0.1}
    att = {"attack_recall": 0.6, "false_alarm": 0.3, "n_attack": 300, "n_natural": 100}
    out = ns9.class_split(clean, att)
    assert out["evasion"] == pytest.approx(0.2)
    assert out["false_alarm_rise"] == pytest.approx(0.2)
    assert out["evasion_share_rows"] == pytest.approx(60 / 80)
    att_neg = att | {"false_alarm": 0.05}
    assert ns9.class_split(clean, att_neg)["evasion_share_rows"] is None


def test_excess_is_random_minus_transfer_seed_mean(ns9):
    f1 = {
        (0, 0.5, "rf", "derived", "random"): 0.5,
        (0, 0.5, "rf", "derived", "transfer_mlp"): 0.3,
        (1, 0.5, "rf", "derived", "random"): 0.6,
        (1, 0.5, "rf", "derived", "transfer_mlp"): 0.5,
    }
    out = ns9.excess(f1, [0, 1], "rf", 0.5, "derived")
    assert out["mean"] == pytest.approx(0.15)
    assert out["per_seed"] == pytest.approx([0.2, 0.1])


@pytest.mark.parametrize(
    ("shares", "verdict"),
    [
        ({"rf": (0.1, 0.8), "xgb": (0.3, 0.5), "lgbm": (0.9, 0.1)}, "HYPOTHESIS"),
        ({"rf": (0.6, 0.4), "xgb": (0.5, 0.9), "lgbm": (0.1, 0.9)}, "KILL"),
        ({"rf": (0.4, 0.6), "xgb": (0.31, 0.9), "lgbm": (0.6, 0.2)}, "INTERMEDIATE"),
        ({"rf": (0.1, 0.49), "xgb": (0.2, 0.8), "lgbm": (0.2, 0.3)}, "INTERMEDIATE"),
    ],
)
def test_branch_reads_the_registered_thresholds(ns9, shares, verdict):
    s = {t: {"controlled": c, "derived": d} for t, (c, d) in shares.items()}
    assert ns9.branch(s)["verdict"] == verdict


def test_anchor_is_exact(ns9):
    rec = {
        "seed": 0,
        "eps": 0.5,
        "family": "rf",
        "kind": "random",
        "part": "full",
        "clean_f1": 0.6,
        "attacked_f1": 0.5,
        "radius_mean": 1.0,
        "radius_max": 2.0,
    }
    stage1 = {(0, 0.5, "rf", "random"): dict(rec)}
    assert ns9.anchor([rec, rec | {"part": "derived", "attacked_f1": 0.1}], stage1)["passed"]
    off = rec | {"attacked_f1": 0.5 + 1e-15}
    out = ns9.anchor([off], stage1)
    assert not out["passed"] and out["n_differ"] == 1
    assert not ns9.anchor([rec | {"seed": 1}], stage1)["passed"]
