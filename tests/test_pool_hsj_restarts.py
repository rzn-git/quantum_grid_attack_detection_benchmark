"""Pooling HopSkipJump restart runs: one sample per model, read the way the exhibits read it."""

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments import pool_hsj_restarts as pool
from experiments.pool_hsj_restarts import pool_family, statistic


def _rec(restart, median, share=None, found=None):
    rec = {"restart": restart, "median_linf_sigma": median}
    if share is not None:
        rec["found_share"] = share
        rec["median_linf_sigma_found"] = median if found is None else found
    return rec


def test_two_runs_pool_into_one_sample():
    old = [_rec(i, m) for i, m in enumerate([4.917, 2.808, 2.831, 4.792, 4.249])]
    new = [
        _rec(i, m, share=s, found=f)
        for i, (m, s, f) in enumerate(
            [
                (4.990, 0.99, 5.263),
                (3.626, 0.99, 3.819),
                (0.573, 1.0, 0.573),
                (2.799, 1.0, 2.799),
                (2.548, 0.98, 2.697),
            ]
        )
    ]
    records, s = pool_family({"august": old, "november": new}, min_found_share=0.95)
    values = np.array([4.917, 2.808, 2.831, 4.792, 4.249, 4.990, 3.626, 0.573, 2.799, 2.548])
    assert s["n_restarts"] == 10 and s["source_runs"] == ["august", "november"]
    assert s["mean"] == pytest.approx(values.mean())
    assert s["std"] == pytest.approx(values.std(ddof=1))
    assert (s["min"], s["max"]) == (0.573, 4.990)
    assert s["cv"] == pytest.approx(values.std(ddof=1) / values.mean())
    # half the restarts predate the found share: read over all points, the share is disclosed
    assert "found" not in s and "found_share" not in s
    assert s["found_share_recorded"] == {
        "n_restarts": 5,
        "mean": pytest.approx(0.992),
        "min": 0.98,
    }
    assert [r["source_run"] for r in records] == ["august"] * 5 + ["november"] * 5
    assert statistic(s, records[5]) == 4.990


def test_found_share_on_every_restart_reads_found_rows():
    recs = [_rec(i, 0.0, share=0.24, found=f) for i, f in enumerate([0.0115, 0.0150, 0.0134])]
    records, s = pool_family({"run": recs}, min_found_share=0.95)
    assert s["mean"] == 0.0 and np.isnan(s["cv"])
    assert s["found_share"] == {"mean": pytest.approx(0.24), "min": 0.24, "max": 0.24}
    assert s["found"]["mean"] == pytest.approx(np.mean([0.0115, 0.0150, 0.0134]))
    assert statistic(s, records[1]) == 0.0150


def test_zero_median_without_found_share_cannot_be_pooled():
    with pytest.raises(ValueError, match="zero median"):
        pool_family({"august": [_rec(0, 0.0), _rec(1, 0.0)]}, min_found_share=0.95)


def test_low_recorded_share_cannot_be_read_over_all_points():
    old = [_rec(0, 0.4), _rec(1, 0.5)]
    new = [_rec(0, 0.3, share=0.60, found=0.5)]
    with pytest.raises(ValueError, match="under 0.95"):
        pool_family({"august": old, "november": new}, min_found_share=0.95)


def test_restart_that_found_no_row_cannot_be_summarised():
    recs = [_rec(0, 0.0, share=0.24, found=0.01), _rec(1, 0.0, share=0.0, found=float("nan"))]
    with pytest.raises(ValueError, match="found no row"):
        pool_family({"run": recs}, min_found_share=0.95)


SPEC = {
    "variant": "binary",
    "qubits": 8,
    "must_match": ["max_iter", "eval_subset"],
    "min_found_share_all_points": 0.95,
}
BUDGET = {"max_iter": 10, "eval_subset": 100}


def _run(root, name, config, row_ids=None):
    d = root / name
    d.mkdir()
    (d / "params.yaml").write_text(yaml.safe_dump({"config": config}), encoding="utf-8")
    if row_ids is not None:
        pd.DataFrame({"condition": "clean", "row_id": row_ids}).to_parquet(
            d / "predictions.parquet"
        )
    return name


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(pool, "RUNS", tmp_path)
    return tmp_path


def test_run_without_variant_or_tuned_file_is_not_assumed(runs):
    old = _run(runs, "a_binary_q8", dict(BUDGET))
    spec = {**SPEC, "families": {"qsvm": [old]}}
    with pytest.raises(ValueError, match="state them"):
        pool.check_sources(spec)
    spec["stated"] = {old: {"variant": "binary", "best_params": "tuned.json"}}
    assert pool.check_sources(spec)[old]["best_params"] == "tuned.json"


def test_stated_value_cannot_contradict_the_run(runs):
    new = _run(runs, "b_binary_q8", {**BUDGET, "variant": "binary", "best_params": "t1.json"})
    spec = {**SPEC, "families": {"qsvm": [new]}, "stated": {new: {"best_params": "t2.json"}}}
    with pytest.raises(ValueError, match="differently"):
        pool.check_sources(spec)


def test_pooled_runs_must_share_tuned_file_budget_and_name(runs):
    full = {**BUDGET, "variant": "binary"}
    a = _run(runs, "a_binary_q8", {**full, "best_params": "t1.json"})
    b = _run(runs, "b_binary_q8", {**full, "best_params": "t2.json"})
    with pytest.raises(ValueError, match="different tuned files"):
        pool.check_sources({**SPEC, "families": {"qsvm": [a, b]}})
    c = _run(runs, "c_binary_q8", {**full, "best_params": "t1.json", "max_iter": 50})
    with pytest.raises(ValueError, match="disagree on max_iter"):
        pool.check_sources({**SPEC, "families": {"qsvm": [a], "mlp": [c]}})
    d = _run(runs, "d_triple_q8", {**full, "best_params": "t1.json"})
    with pytest.raises(ValueError, match="does not carry"):
        pool.check_sources({**SPEC, "families": {"qsvm": [d]}})


def test_runs_with_stored_rows_must_attack_the_same_rows(runs):
    a = _run(runs, "a_binary_q8", {}, row_ids=[1, 2, 3])
    b = _run(runs, "b_binary_q8", {}, row_ids=[1, 2, 3])
    old = _run(runs, "c_binary_q8", {})
    assert pool.check_points([a, b, old]) == {a: True, b: True, old: False}
    e = _run(runs, "e_binary_q8", {}, row_ids=[1, 2, 4])
    with pytest.raises(ValueError, match="different test rows"):
        pool.check_points([a, e])
