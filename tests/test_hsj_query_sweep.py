"""The query-budget arguments of the restart runner and the sweep reader's rules (no attack)."""

import json

import numpy as np
import pytest

from experiments.read_hsj_query_sweep import (
    branch,
    check_queries,
    noise_row_counts,
    pool,
    pool_direction,
)
from experiments.run_hopskipjump_restarts import budget_settings
from qgridbench.eval import tables

PUBLISHED = {
    "attack": "hopskipjump",
    "max_iter": 10,
    "max_eval": 200,
    "init_eval": 20,
    "eval_subset": 100,
}


def test_no_argument_keeps_the_published_settings():
    none = {"max_iter": None, "init_eval": None, "max_eval": None}
    assert budget_settings(PUBLISHED, none) == PUBLISHED
    assert budget_settings(PUBLISHED, {}) == PUBLISHED


def test_an_argument_replaces_its_setting_only():
    out = budget_settings(PUBLISHED, {"max_iter": 40, "init_eval": 60, "max_eval": None})
    assert (out["max_iter"], out["init_eval"], out["max_eval"]) == (40, 60, 200)
    assert out["eval_subset"] == 100 and PUBLISHED["max_iter"] == 10


def test_a_setting_below_one_is_an_error():
    with pytest.raises(ValueError):
        budget_settings(PUBLISHED, {"max_iter": 0})


def _rec(restart, found_median, p30=None, queries=620.0, found=1.0, stalled=0, all_median=None):
    rec = {
        "restart": restart,
        # the median over all points differs from the one over found rows when the attack
        # left a row on the clean side; the reader must take the found one
        "median_linf_sigma": found_median if all_median is None else all_median,
        "median_linf_sigma_found": found_median,
        "queries_per_sample": queries,
        "found_share": found,
        "stalled_steps": stalled,
    }
    if p30 is not None:
        rec["p30_found"] = p30
    return rec


def test_pool_reads_the_spread_over_restarts():
    out = pool([_rec(1, 3.0, 1.0), _rec(0, 1.0, 0.4), _rec(2, 2.0, 0.7)])
    assert out["restarts"] == [0, 1, 2] and out["medians_found"] == [1.0, 3.0, 2.0]
    assert out["mean"] == pytest.approx(2.0) and out["std"] == pytest.approx(1.0)
    assert out["cv"] == pytest.approx(0.5) and out["p30_mean"] == pytest.approx(0.7)
    assert out["standard_error"] == pytest.approx(1.0 / 3**0.5)


def test_pool_reads_found_rows_not_all_points():
    """A row the attack could not flip comes back on the clean side: its distance is not a
    distance to the boundary (root-cause review of 2026-11-22, finding 2)."""
    out = pool([_rec(0, 2.0, found=0.98, all_median=2.4), _rec(1, 3.0, found=1.0)])
    assert out["mean"] == pytest.approx(2.5) and out["found_share_min"] == 0.98


def test_pool_refuses_a_restart_given_twice():
    with pytest.raises(ValueError):
        pool([_rec(0, 1.0), _rec(0, 2.0)])


def test_pool_without_percentiles_has_no_p30():
    assert "p30_mean" not in pool([_rec(0, 1.0), _rec(1, 2.0)])


def test_measured_queries_outside_the_sweep_files_range_are_refused():
    pooled = pool([_rec(0, 1.0, queries=2700.0), _rec(1, 2.0, queries=2850.0)])
    check_queries("mid", "qsvm", pooled, [2400, 3200])
    with pytest.raises(ValueError, match="queries per point"):
        check_queries("mid", "qsvm", pooled, [500, 700])


@pytest.mark.parametrize(
    ("mean", "p30", "classical", "expected"),
    [
        (0.60, 0.20, 0.50, "BUDGET_ARTIFACT"),  # under the kill line, whatever else holds
        (2.50, 0.90, 0.80, "HOLDS"),
        (2.50, 0.40, 0.80, "WEAKENED"),  # the lower tail is inside the noise radius
        (0.95, 0.90, 1.10, "WEAKENED"),  # a classical model is further at the same budget
        (0.88, 0.90, 0.80, "HOLDS"),  # the kill line itself is not under it
    ],
)
def test_branches_cover_every_outcome(mean, p30, classical, expected):
    assert branch({"mean": mean, "p30_mean": p30}, classical, 0.88, 0.5) == expected


def _directed(restart, natural, attack, rows=(3, 7), found=None):
    found = rows if found is None else found
    return {
        "restart": restart,
        "by_clean_prediction": {
            "0": {"n": rows[0], "found": found[0], "median_linf_sigma_found": natural},
            "1": {"n": rows[1], "found": found[1], "median_linf_sigma_found": attack},
        },
    }


def test_pool_direction_reads_each_group_on_its_own():
    """The two groups sit sigmas apart; a median over both describes neither."""
    records = [_directed(1, 0.03, 6.0), _directed(0, 0.01, 4.0), _directed(2, 0.02, 5.0)]
    attack = pool_direction(records, "1")
    assert attack["rows"] == 7 and attack["medians_found"] == [4.0, 6.0, 5.0]
    assert attack["mean"] == pytest.approx(5.0) and attack["std"] == pytest.approx(1.0)
    assert attack["cv"] == pytest.approx(0.2) and (attack["min"], attack["max"]) == (4.0, 6.0)
    assert pool_direction(records, "0")["mean"] == pytest.approx(0.02)


def test_pool_direction_keeps_the_found_rows_of_each_restart():
    records = [_directed(0, 0.01, 4.0, found=(2, 7)), _directed(1, 0.03, 6.0)]
    assert pool_direction(records, "0")["found"] == [2, 3]


def test_pool_direction_refuses_restarts_on_different_rows():
    with pytest.raises(ValueError, match="disagree"):
        pool_direction([_directed(0, 0.01, 4.0), _directed(1, 0.01, 4.0, rows=(4, 6))], "1")


def test_pool_direction_refuses_a_group_the_attack_never_flipped():
    with pytest.raises(ValueError, match="found no row"):
        pool_direction([_directed(0, 0.01, None, found=(3, 0))], "1")


def test_pool_direction_of_an_empty_group_has_no_distance():
    out = pool_direction([_directed(0, None, 4.0, rows=(0, 10))], "0")
    assert out == {"rows": 0, "found": [0]}


def test_noise_row_counts_sets_the_attack_against_the_noise_row_by_row():
    #                 row:  0     1     2     3     4     5
    dist = np.array([4.0, 0.2, 3.0, 0.4, 0.01, 2.0])
    found = np.array([True, True, True, True, True, False])
    attack_labelled = np.array([True, True, True, True, False, True])
    noise_flips = np.array(
        [
            [True, True, False, False, True, True],
            [True, True, True, False, True, True],
            [False, True, False, False, True, True],
        ]
    )
    out = noise_row_counts(dist, found, attack_labelled, noise_flips, radius=0.5, at_least=2)
    # rows 0 to 3 are labelled attack and found; row 4 is labelled natural, row 5 was not found
    assert out["rows"] == 4
    # the noise flips rows 0 and 1 in two draws or more; the attack puts row 0 at 4 sigma
    assert (out["noise_flips"], out["noise_flips_attack_beyond"]) == (2, 1)
    # one draw also flips row 2, which the attack puts at 3 sigma
    assert (out["noise_flips_any_draw"], out["noise_flips_any_draw_attack_beyond"]) == (3, 2)
    assert out["attack_within_share"] == pytest.approx(0.5)


def test_noise_row_counts_a_distance_at_the_radius_is_not_beyond_it():
    out = noise_row_counts(
        np.array([0.5, 0.5000001]),
        np.array([True, True]),
        np.array([True, True]),
        np.array([[True, True]]),
        radius=0.5,
        at_least=1,
    )
    assert (out["noise_flips"], out["noise_flips_attack_beyond"]) == (2, 1)


def test_noise_row_counts_refuses_an_empty_group():
    with pytest.raises(ValueError, match="no row"):
        noise_row_counts(
            np.array([1.0]),
            np.array([True]),
            np.array([False]),
            np.array([[True]]),
            radius=0.5,
            at_least=1,
        )


def _sweep_verdict():
    """A two-budget, two-model verdict in the reader's layout."""

    def pooled(mean, std, lo, hi):
        return {"mean": mean, "std": std, "queries_per_sample": {"min": lo, "max": hi}}

    def group(rows, mean, std):
        return {"rows": rows, "mean": mean, "std": std}

    def beyond(two, any_draw):
        return {
            "mean": {
                "noise_flips_attack_beyond": two,
                "noise_flips_any_draw_attack_beyond": any_draw,
            }
        }

    return {
        "per_budget": {
            "low": {
                "settings": {},
                "qsvm": pooled(2.7026, 1.1311, 619.83, 626.07),
                "mlp": pooled(0.8702, 0.0172, 594.2, 597.1),
            },
            "high": {
                "settings": {},
                "qsvm": pooled(0.4249, 0.1607, 10865.27, 10879.51),
                "mlp": pooled(0.7170, 0.0021, 10777.0, 10780.4),
            },
        },
        "after_the_result": {
            "by_direction": {
                "low": {
                    "qsvm": {
                        "labelled_attack": group(67, 6.448, 0.853),
                        "labelled_natural": group(33, 0.0204, 0.0031),
                    },
                    "mlp": {
                        "labelled_attack": group(94, 0.893, 0.014),
                        "labelled_natural": group(6, 0.1448, 0.01),
                    },
                },
                "high": {
                    "qsvm": {
                        "labelled_attack": group(67, 4.086, 0.296),
                        "labelled_natural": group(33, 0.0123, 0.0006),
                    },
                    "mlp": {
                        "labelled_attack": group(94, 0.737, 0.002),
                        "labelled_natural": group(6, 0.1222, 0.01),
                    },
                },
            },
            "noise_rows": {
                "low": {"qsvm": beyond(24.6, 43.0), "mlp": beyond(0.0, 0.0)},
                "high": {"qsvm": beyond(20.4, 35.0), "mlp": beyond(0.0, 0.0)},
            },
            "noise_control": {
                "radius_sigma": 0.5,
                "flipped_in_at_least": 2,
                "rows_labelled_attack_noise_flips": {"qsvm": 28, "mlp": 0},
                "rows_labelled_attack_noise_flips_any_draw": {"qsvm": 51, "mlp": 2},
            },
        },
    }


def test_sweep_table_prints_each_reading_per_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "TABLE_DIR", tmp_path / "tables")
    (tmp_path / "results.json").write_text(json.dumps(_sweep_verdict()))
    tex = tables.hsj_query_sweep_table(tmp_path, "sweep").read_text()
    pm = "\\,$\\pm$\\,"
    end = " \\\\"
    # the measured query range over the models, lowest to highest budget
    assert "Queries per point & 594--626 & 10\\,777--10\\,880" + end in tex
    assert "Median over both groups (registered)" in tex
    assert f"QSVM & 2.70{pm}1.13 & 0.42{pm}0.16" + end in tex
    # a spread that is not zero never prints as zero
    assert f"MLP & 0.87{pm}0.02 & 0.72{pm}$<$\\,0.01" + end in tex
    assert f"QSVM (67 rows) & 6.45{pm}0.85 & 4.09{pm}0.30" + end in tex
    # a block with a distortion under 0.1 sigma prints three decimals for every model
    assert f"QSVM (33 rows) & 0.020{pm}0.003 & 0.012{pm}0.001" + end in tex
    assert f"MLP (6 rows) & 0.145{pm}0.010 & 0.122{pm}0.010" + end in tex
    # one block per rule for "the noise flips the row", the stricter rule first
    two, any_draw = tex.split("flips in at least two of its draws")[1].split("flips in any draw")
    assert "QSVM (28 rows) & 24.6 & 20.4" + end in two
    assert "QSVM (51 rows) & 43.0 & 35.0" + end in any_draw
    # a model none of whose rows the noise flips has nothing to place beyond the radius
    assert "MLP (0 rows) & -- & --" + end in two
    assert "MLP (2 rows) & 0.0 & 0.0" + end in any_draw
    assert "places beyond $0.5\\sigma$" in tex
    assert "\\begin{tabular}{lrr}" in tex and "\\bottomrule" in tex
    rows = (tmp_path / "tables" / "sweep.csv").read_text().splitlines()
    assert rows[0] == "block,model,rows,low,high"
    assert "labelled_attack,qsvm,67,6.4480,4.0860" in rows


def test_sweep_table_refuses_groups_that_change_between_budgets(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "TABLE_DIR", tmp_path / "tables")
    verdict = _sweep_verdict()
    verdict["after_the_result"]["by_direction"]["high"]["qsvm"]["labelled_attack"]["rows"] = 66
    (tmp_path / "results.json").write_text(json.dumps(verdict))
    with pytest.raises(ValueError, match="disagree"):
        tables.hsj_query_sweep_table(tmp_path, "sweep")
