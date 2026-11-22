"""The diagram-number macros: read from their artifacts, IEEE-formatted, and loud on bad input."""

import json

import pytest

from experiments import build_paper_numbers as pn


def _meta(tmp_path, counts, n):
    p = tmp_path / "meta.json"
    p.write_text(json.dumps({"n_rows": n, "class_counts": counts, "split_sizes": {"train": 47020}}))
    return p


def test_data_numbers_format_and_shares(tmp_path, monkeypatch):
    monkeypatch.setattr(pn, "META_BINARY", _meta(tmp_path, {"0": 22706, "1": 55662}, 78368))
    monkeypatch.setattr(pn, "REPO_ROOT", tmp_path)
    got = {n["macro"]: n["value"] for n in pn.data_numbers()}
    assert got["NumRowsBinary"] == "78\\,368" and got["NumTrainRows"] == "47\\,020"
    assert got["NumAttackShare"] == "71" and got["NumNaturalShare"] == "29"


def test_data_numbers_reject_counts_that_do_not_sum(tmp_path, monkeypatch):
    monkeypatch.setattr(pn, "META_BINARY", _meta(tmp_path, {"0": 1, "1": 1}, 3))
    with pytest.raises(ValueError, match="do not sum"):
        pn.data_numbers()


def _replication_tree(tmp_path, monkeypatch, large_rows=5760):
    """A fake processed folder, cells file and three verdict runs for replication_numbers."""
    proc, runs = tmp_path / "data" / "processed", tmp_path / "results" / "runs"
    proc.mkdir(parents=True)
    (proc / "meta_gas_binary.json").write_text(
        json.dumps(
            {
                "n_rows": 38054,
                "n_raw_rows": 274628,
                "n_features": 16,
                "class_counts": {"0": 17489, "1": 20565},
            }
        )
    )
    for name, feats, n in (("fdia14", 82, 5760), ("fdia118", 726, large_rows)):
        (proc / f"meta_{name}_binary.json").write_text(
            json.dumps(
                {
                    "n_rows": n,
                    "n_rows_generated": 8640,
                    "n_features": feats,
                    "class_counts": {"0": n // 2, "1": n - n // 2},
                }
            )
        )
    verdicts = {
        pn.GAS_VERDICT_LOG_SCALED: ("rf", "qsvm_zz", {"rf": 0.76057, "qsvm_zz": 0.74834}),
        "grid_small_run": ("rbf_svm", "qsvm_zz", {"rbf_svm": 0.99566, "qsvm_zz": 0.98785}),
        "grid_large_run": ("rbf_svm", "qsvm_zz", {"rbf_svm": 0.9974, "qsvm_zz": 0.99306}),
    }
    for run, (best_c, best_q, mean) in verdicts.items():
        (runs / run).mkdir(parents=True)
        (runs / run / "results.json").write_text(
            json.dumps({"pca8": {"best_classical": best_c, "best_quantum": best_q, "mean": mean}})
        )
    cells = tmp_path / "cells.csv"
    cells.write_text(
        "claim,dataset,run\naccuracy,gas,ignored\naccuracy,fdia14,grid_small_run\n"
        "accuracy,fdia118,grid_large_run\nsplit,fdia14,another_run\n"
    )
    for attr, value in (("PROCESSED", proc), ("RUNS", runs), ("CELLS", cells)):
        monkeypatch.setattr(pn, attr, value)
    monkeypatch.setattr(pn, "REPO_ROOT", tmp_path)


def test_replication_numbers_read_rows_shares_and_best_scores(tmp_path, monkeypatch):
    _replication_tree(tmp_path, monkeypatch)
    got = {n["macro"]: n for n in pn.replication_numbers()}
    values = {k: v["value"] for k, v in got.items()}
    assert values["NumGasRows"] == r"38\,054" and values["NumGasRawRows"] == r"274\,628"
    assert values["NumGasFeatures"] == "16" and values["NumGasAttackShare"] == "54"
    assert values["NumGridRows"] == "5760" and values["NumGridRowsGenerated"] == "8640"
    assert values["NumGridSmallFeatures"] == "82" and values["NumGridLargeFeatures"] == "726"
    assert values["NumGridAttackShare"] == "50"
    assert values["NumGasBestClassical"] == "0.761" and values["NumGasBestQuantum"] == "0.748"
    assert values["NumGridSmallBestClassical"] == "0.996"
    assert values["NumGridSmallBestQuantum"] == "0.988"
    assert values["NumGridLargeBestClassical"] == "0.997"
    assert values["NumGridLargeBestQuantum"] == "0.993"
    # every score names the model it read, and every macro name is a LaTeX command name
    assert got["NumGasBestClassical"]["key"] == "pca8/mean/rf"
    assert all(name.isalpha() for name in got)


def test_replication_numbers_reject_grids_that_differ_in_rows(tmp_path, monkeypatch):
    _replication_tree(tmp_path, monkeypatch, large_rows=5000)
    with pytest.raises(ValueError, match="disagree on n_rows"):
        pn.replication_numbers()


def test_the_gas_verdict_is_the_run_the_report_pins():
    import experiments.make_report as mr

    assert pn.GAS_VERDICT_LOG_SCALED == mr.GAS_ACCURACY_RUNS["slog"]
