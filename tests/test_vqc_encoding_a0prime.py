"""R4-13 A0': the Table I parameter guard, the R4-23 anchor, the reference deltas and the paired
tests, on tiny fixtures (no model, no torch)."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def m():
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        yield importlib.import_module("experiments.run_vqc_encoding_a0prime")


GRID = {"vqc_depth": 4, "vqc_lr": 0.0397, "vqc_batch": 128, "vqc_epochs": 40, "qk_bw": 0.79}


def test_table1_params_accepts_only_depth_and_lr_changes(m):
    t1 = {**GRID, "vqc_depth": 6, "vqc_lr": 0.017467604901736582}
    assert m.table1_params(GRID, t1) == {"lr": 0.017467604901736582, "depth": 6, "batch_size": 128}
    with pytest.raises(AssertionError, match="differs"):
        m.table1_params(GRID, {**t1, "qk_bw": 1.0})
    with pytest.raises(AssertionError, match="expected"):
        m.table1_params(GRID, {**t1, "vqc_depth": 5})


def _unit(seed, clean, pgd, tr, arm="A0p_table1"):
    return {
        "arm": arm,
        "seed": seed,
        "clean": {"eval500": clean},
        "r2": {"pgd": {"0.5": pgd}},
        "r4": {"mlp_to_arm": tr},
    }


def _ref(seed, clean, pgd, tr, arm="depth6"):
    ev = {
        "clean": {"vqc": clean},
        "curves": {"vqc": {"pgd": {"0.5": pgd}}},
        "transfer_matrix": {"mlp_surrogate": {"vqc": tr}},
    }
    return {"arm": arm, "seed": seed, "evasion": ev}


def test_anchor_blocks_on_clean_and_pgd_and_reports_transfer(m):
    ref = [_ref(0, 0.48, 0.03, 0.12), _ref(0, 0.50, 0.04, 0.2, arm="depth4")]
    ok = m.anchors([_unit(0, 0.48, 0.03, 0.13)], ref)
    assert ok["pass"] and not ok["reported_transfer_all_exact"]
    assert not m.anchors([_unit(0, 0.48, 0.0301, 0.12)], ref)["pass"]


def _summary(clean, pgd, loss="inverted"):
    return {
        "clean500": {"mean": clean},
        "pgd_0.5": {"mean": pgd},
        "retention_0.5": pgd / clean,
        "transfer_mlp_to_arm_0.5": {"mean": 0.3},
        "transfer_arm_to_mlp_0.5": {"mean": 0.4},
        "end_state_2sigma": {"loss": {"label": loss}, "distance": {"label": "partial"}},
    }


def test_deltas_against_both_references(m):
    s = {
        "A0_deployed": _summary(0.50, 0.036),
        "A0p_table1": _summary(0.48, 0.029),
        "A1_angle": _summary(0.50, 0.010),
        "A2_zz": _summary(0.53, 0.237, loss="partial"),
        "A3_reupload": _summary(0.48, 0.014),
    }
    d = m.deltas(s)
    assert d["A2_zz"]["minus_A0_deployed"]["pgd_0.5"] == pytest.approx(0.201)
    assert d["A2_zz"]["minus_A0p_table1"]["clean500"] == pytest.approx(0.05)
    assert d["A2_zz"]["end_state_vs_A0p_table1"]["loss"] == {"arm": "partial", "ref": "inverted"}


def test_wilcoxon_pairs_by_seed_and_applies_holm(m):
    rng = np.random.default_rng(0)
    units = []
    for arm, shift in (
        ("A0_deployed", 0),
        ("A0p_table1", 0),
        ("A1_angle", 0),
        ("A2_zz", 0.2),
        ("A3_reupload", 0),
    ):
        for s in range(10):
            units.append(_unit(s, 0.5, 0.03 + shift + 0.01 * rng.standard_normal(), 0.1, arm=arm))
    w = m.wilcoxon_tests(units)
    assert w["n_seeds"] == 10 and len(w["vs_references"]) == 6
    assert w["vs_references"]["A2_zz-A0_deployed"]["p_value"] < 0.01
    assert (
        w["design_note_vs_A1"]["A2_zz-A1_angle"]["p_holm"]
        >= w["design_note_vs_A1"]["A2_zz-A1_angle"]["p_value"]
    )
    with pytest.raises(AssertionError, match="seeds"):
        m.wilcoxon_tests(units[:-1])
