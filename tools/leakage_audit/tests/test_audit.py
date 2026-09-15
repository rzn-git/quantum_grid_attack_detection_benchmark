"""Unit tests for the leakage audit on synthetic data with a known answer.

Not collected by the project's default pytest run (testpaths = tests/); run explicitly:
    pytest tools/leakage_audit/tests
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tools.leakage_audit.audit import (
    _assert_group_disjoint,
    _group_partition,
    neighbour_audit,
    paired_protocol,
)


def _grouped_near_duplicate_data(n_groups=10, rows_per_group=60, seed=0):
    """Groups of near-duplicate rows; the label depends on a group-level latent.

    A row-level split then sees near-copies of its test rows in train (leakage), while a
    group split does not — the construction the tool exists to detect.
    """
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for gi in range(n_groups):
        center = rng.standard_normal(8) * 3.0
        X.append(center + 0.01 * rng.standard_normal((rows_per_group, 8)))
        label = int(center[0] > 0)
        y.append(np.full(rows_per_group, label))
        g.append(np.full(rows_per_group, gi))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def test_neighbour_audit_separates_the_two_regimes():
    X, y, g = _grouped_near_duplicate_data()
    rng = np.random.default_rng(0)
    row_te = rng.choice(len(y), size=len(y) // 5, replace=False)
    row_tr = np.setdiff1d(np.arange(len(y)), row_te)
    row = neighbour_audit(X[row_tr], X[row_te], threshold=0.1)

    grp_te = np.isin(g, [0, 1])
    grp = neighbour_audit(X[~grp_te], X[grp_te], threshold=0.1)

    assert row["frac_within_threshold"] > 0.9, "row split must show near-duplicate neighbours"
    assert grp["frac_within_threshold"] < 0.1, "group split must not"


def test_neighbour_audit_threshold_is_monotone():
    X, y, g = _grouped_near_duplicate_data()
    tr, te = np.arange(0, 500), np.arange(500, 600)
    fracs = [
        neighbour_audit(X[tr], X[te], threshold=t)["frac_within_threshold"]
        for t in (0.01, 0.1, 1.0, 100.0)
    ]
    assert fracs == sorted(fracs)
    assert fracs[-1] == 1.0


def test_paired_protocol_reports_the_gap_and_disjoint_groups():
    X, y, g = _grouped_near_duplicate_data(n_groups=12)
    res = paired_protocol(
        LogisticRegression(max_iter=1000),
        X,
        y,
        g,
        n_partitions=3,
        seed=0,
    )
    assert res["row_level"]["score"] > 0.95, "near-duplicate leakage should make the row arm easy"
    assert res["row_level"]["nn_frac_within_threshold"] > 0.9
    assert res["whole_group"]["nn_frac_within_threshold_mean"] < 0.1
    assert res["delta_row_minus_group"] == pytest.approx(
        res["row_level"]["score"] - res["whole_group"]["score_mean"]
    )
    for p in res["whole_group"]["partitions"]:
        assert p["held_out_groups"], "each partition must name its held-out groups"


def test_group_partition_asserts_disjointness():
    g = np.repeat(np.arange(5), 10)
    splits, held = _group_partition(g, np.random.default_rng(0), val_size=0.2, test_size=0.2)
    _assert_group_disjoint(splits, g)  # must not raise on a genuine group split
    # a corrupted split must raise
    bad = dict(splits)
    bad["test"] = np.concatenate([splits["test"], splits["train"][:1]])
    with pytest.raises(AssertionError, match="not group-disjoint"):
        _assert_group_disjoint(bad, g)


def test_length_mismatch_fails_loud():
    X, y, g = _grouped_near_duplicate_data(n_groups=4)
    with pytest.raises(ValueError, match="length mismatch"):
        paired_protocol(LogisticRegression(), X, y[:-1], g)
