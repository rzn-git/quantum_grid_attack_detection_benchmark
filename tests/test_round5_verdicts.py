"""Registered branch rules of the round-5 runners (decision log 2026-11-19T08:25Z)."""

import math

from experiments.run_gas_duplicate_rescore import branch as ns15_branch
from experiments.run_level_per_class import verdict_family, verdict_file


def _unit(macro, nat_r, att_r, nat_f1, att_f1, floor=0.5, held=None):
    u = {
        "macro_f1": macro,
        "floor_macro_f1": floor,
        "per_class": {
            "natural": {"recall": nat_r, "f1": nat_f1},
            "attack": {"recall": att_r, "f1": att_f1},
        },
    }
    if held is not None:
        u["held_attack_recall"] = held
    return u


def test_ns13_holds_when_natural_carries_the_drop():
    m = {"row": _unit(0.9, 0.8, 0.95, 0.85, 0.95)}
    for p in range(5):
        m[f"group{p}"] = _unit(0.6, 0.35, 0.8, 0.4, 0.8)
    v = verdict_file(m)
    assert v["branch"] == "HOLDS"
    assert math.isclose(v["natural_share_of_macro_f1_drop"], 0.45 / 0.6)


def test_ns13_kill_when_attack_recall_drops_more():
    m = {"row": _unit(0.9, 0.8, 0.95, 0.85, 0.95)}
    for p in range(5):
        m[f"group{p}"] = _unit(0.6, 0.7, 0.5, 0.7, 0.6)
    assert verdict_file(m)["branch"] == "KILL"


def _family(f_scores, s_scores, r=0.9):
    m = {"R": _unit(r, 0.8, 0.95, 0.85, 0.95)}
    for k, f, s in zip(
        ("data_injection", "remote_tripping", "relay_setting_change"), f_scores, s_scores
    ):
        m[f"F_{k}"] = _unit(f, 0.9, 0.5, 0.8, 0.5, held=0.5)
        m[f"S_{k}"] = _unit(s, 0.9, 0.9, 0.8, 0.9, held=0.9)
    return m


def test_ns14_branches():
    assert verdict_family(_family((0.6, 0.7, 0.6), (0.9, 0.9, 0.9)))["branch"] == "KILL"
    assert verdict_family(_family((0.4, 0.45, 0.6), (0.9, 0.9, 0.9)))["branch"] == "REPLICATES"
    assert verdict_family(_family((0.4, 0.7, 0.6), (0.9, 0.9, 0.9)))["branch"] == "PARTIAL"
    assert verdict_family(_family((0.6, 0.7, 0.6), (0.9, 0.9, 0.7)))["branch"] == "VOID"


def _ns15(rh, ds, rows=26):
    hsj = {t: {"R_h": rh, "readable": rows >= 20} for t in ("rf", "xgb", "lgbm")}
    sq = {t: {"0.1": {"dS": ds, "readable": True}} for t in ("rf", "xgb", "lgbm")}
    return ns15_branch(hsj, sq)["verdict"]


def test_ns15_branches():
    assert _ns15(5000.0, 0.6) == "CONFIRMED"
    assert _ns15(1.5, 0.01) == "KILL"
    assert _ns15(5.0, 0.07) == "PARTIAL"
    assert _ns15(5000.0, 0.6, rows=10) == "PARTIAL"  # unreadable non-duplicate set
