"""The verdict reader of the regularization sweep at the grid radii (synthetic blocks only)."""

import sys

import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))  # experiments/ is not an installed package
from experiments import read_regularization_radii as mod  # noqa: E402
from experiments.read_worst_radius import read_sweep  # noqa: E402

FLOOR, RADII, SEEDS = 0.4, ("0.5", "1.0", "2.0"), tuple(range(10))


def _curve(f1_by_radius):
    return {
        "curve": {
            r: {
                "macro_f1": f1,
                "f1_minus_floor": f1 - FLOOR,
                "per_seed": [
                    {
                        "seed": s,
                        "macro_f1": f1,
                        "dec_spread_ratio": 0.5,
                        "f1_minus_floor": f1 - FLOOR,
                    }
                    for s in SEEDS
                ],
            }
            for r, f1 in zip(RADII, f1_by_radius, strict=True)
        }
    }


def _block(loss, mass, grid=(1.0, 10.0, 100.0)):
    """`loss` and `mass` map a C to its mean attacked macro-F1 at the three radii."""
    shared = {
        "constant_predictor_floor": FLOOR,
        "n_eval": 200,
        "budget_sigma": 2.0,
        "radii": [float(r) for r in RADII],
        "n_steps": 20,
        "tuned_bandwidth": 0.8,
        "tuned_C": 100.0,
        "c_grid": list(grid),
        "attack_seeds": list(SEEDS),
    }
    union = {
        k: {"mean": 0.0, "std": 0.0, "per_start": [0.0] * len(SEEDS)}
        for k in ("loss", "mass", "either")
    }
    return shared | {
        "per_c": {
            str(C): {
                "clean_macro_f1": 0.6,
                "degenerate": False,
                "n_support": 10,
                "numeric_gate": {},
                "per_input_union": union,
            }
            for C in loss
        },
        "per_c_loss": {str(C): _curve(v) for C, v in loss.items()},
        "per_c_mass": {str(C): _curve(v) for C, v in mass.items()},
        "per_c_random": {str(C): _curve((0.6, 0.6, 0.6)) for C in loss},
    }


def _verdict(block):
    sweeps = {
        k: read_sweep(
            {"constant_predictor_floor": FLOOR, "per_c": block[f"per_c_{k}"]}, "per_c", 2.0, 3
        )
        for k in ("loss", "mass")
    }
    readings = {C: mod.read_c(block, C, sweeps) for C in block["per_c"]}
    return readings, mod.verdict(readings, str(block["tuned_C"]))


def test_falls_when_every_c_is_inverted_at_three_radii():
    inverted, at_floor = (0.1, 0.05, 0.15), (0.5, 0.45, 0.4)
    block = _block(
        {C: inverted for C in (1.0, 10.0, 100.0)}, {C: at_floor for C in (1.0, 10.0, 100.0)}
    )
    readings, v = _verdict(block)
    assert v["branch"] == "FALLS UNDER R" and v["small_c"] == ["1.0", "10.0"]
    w = readings["1.0"]["worst"]
    assert (w["objective"], w["radius"], w["end_state"]) == ("loss", 1.0, "inverted")
    assert w["random_control_macro_f1"] == 0.6


def test_stands_when_small_c_is_not_inverted_and_the_tuned_c_is():
    block = _block(
        {1.0: (0.5, 0.45, 0.4), 10.0: (0.5, 0.42, 0.4), 100.0: (0.1, 0.05, 0.15)},
        {C: (0.5, 0.45, 0.4) for C in (1.0, 10.0, 100.0)},
    )
    _, v = _verdict(block)
    assert v["branch"] == "STANDS UNDER R"
    assert v["small_c_minus_tuned_at_worst"]["10.0"]["mean"] == pytest.approx(0.35)


def test_mixed_when_one_small_c_is_inverted_or_the_label_holds_at_one_radius():
    block = _block(
        {1.0: (0.5, 0.45, 0.4), 10.0: (0.1, 0.05, 0.15), 100.0: (0.1, 0.05, 0.15)},
        {C: (0.5, 0.45, 0.4) for C in (1.0, 10.0, 100.0)},
    )
    assert _verdict(block)[1]["branch"] == "MIXED"
    one_radius = (0.5, 0.05, 0.45)  # inverted at one radius only: not a persistent label
    block = _block(
        {C: one_radius for C in (1.0, 10.0, 100.0)},
        {C: (0.5, 0.45, 0.4) for C in (1.0, 10.0, 100.0)},
    )
    _, v = _verdict(block)
    assert v["branch"] == "MIXED" and not any(v["inverted_at_min_radii_or_more"].values())


def test_the_worst_reading_takes_the_lower_objective():
    block = _block(
        {C: (0.5, 0.45, 0.4) for C in (1.0, 10.0, 100.0)},
        {C: (0.5, 0.3, 0.1) for C in (1.0, 10.0, 100.0)},
    )
    readings, _ = _verdict(block)
    assert (readings["10.0"]["worst"]["objective"], readings["10.0"]["worst"]["radius"]) == (
        "mass",
        2.0,
    )


def test_merge_needs_every_c_once_and_agreeing_runs():
    full = _block(
        {C: (0.5, 0.45, 0.4) for C in (1.0, 10.0, 100.0)},
        {C: (0.5, 0.45, 0.4) for C in (1.0, 10.0, 100.0)},
    )

    def only(C):
        return {k: ({str(C): v[str(C)]} if k in mod.AXES else v) for k, v in full.items()}

    merged = mod.merge([only(1.0), only(10.0), only(100.0)])
    assert sorted(merged["per_c_loss"]) == ["1.0", "10.0", "100.0"]
    with pytest.raises(ValueError, match="two runs"):
        mod.merge([only(1.0), only(1.0), only(100.0)])
    with pytest.raises(ValueError, match="the grid is"):
        mod.merge([only(1.0), only(100.0)])
    with pytest.raises(ValueError, match="disagree"):
        mod.merge([only(1.0), only(10.0) | {"n_steps": 5}, only(100.0)])


def test_anchor_fails_when_the_budget_reading_leaves_the_stored_run():
    block = _block(
        {C: (0.5, 0.45, 0.4) for C in (1.0, 10.0, 100.0)},
        {C: (0.5, 0.45, 0.4) for C in (1.0, 10.0, 100.0)},
    )

    def part_b(f1_minus_floor):
        return {
            C: {
                "clean_macro_f1": 0.6,
                "per_seed": [{"seed": 1}, {"seed": 2}],
                "mean": {
                    "dec_spread_ratio": {"mean": 0.5},
                    "f1_minus_floor": {"mean": f1_minus_floor},
                },
            }
            for C in block["per_c"]
        }

    p9a = {
        "per_bandwidth": {
            "0.8": {
                "curve": {r: {"macro_f1": f} for r, f in zip(RADII, (0.5, 0.45, 0.4), strict=True)}
            }
        }
    }
    anchors: list = []
    mod.check_anchors(block, part_b(0.0), p9a, [0, 1], anchors)
    assert len(anchors) == 3 * 3 + 3 and all(a["ok"] for a in anchors)
    with pytest.raises(ValueError, match="anchor failed"):
        mod.check_anchors(block, part_b(0.05), p9a, [0, 1], [])
