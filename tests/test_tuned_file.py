"""R4-24 flags: the default tuned file is the variant's own (so default runs stay bit-identical),
an explicit file is read as given, the run tag only suffixes, and the VQC-key override copies
exactly the four VQC keys and fails loud on a missing one (tiny files, no data, no models)."""

import argparse
import json
import sys

import pytest

import qgridbench.protocol as protocol
from qgridbench.utils.run_tracking import REPO_ROOT

# experiments/ is not installed
sys.path.insert(0, str(REPO_ROOT))
from experiments import tuned_file as tf  # noqa: E402

GRID = {"mlp": {"h1": 8}, "vqc_depth": 4, "vqc_lr": 0.04, "vqc_batch": 128, "vqc_epochs": 40}
TABLE1 = {**GRID, "vqc_depth": 6, "vqc_lr": 0.0175}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "adv_best_params_binary_q8.json").write_text(json.dumps(GRID))
    (tmp_path / "results" / "table1.json").write_text(json.dumps(TABLE1))
    monkeypatch.setattr(protocol, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tf, "REPO_ROOT", tmp_path)
    return tmp_path


def _parse(argv):
    ap = argparse.ArgumentParser()
    tf.add_tuned_file_args(ap)
    return ap.parse_args(argv)


def test_default_is_the_variant_file(repo):
    args = _parse([])
    assert args.best_params is None and args.run_tag is None
    rel, params = tf.load_tuned_file(args.best_params, "binary", 8)
    assert rel == "results/adv_best_params_binary_q8.json"
    assert params == GRID


def test_explicit_file_is_read_as_given(repo):
    args = _parse(["--best-params", "results/table1.json", "--run-tag", "table1"])
    rel, params = tf.load_tuned_file(args.best_params, "binary", 8)
    assert rel == "results/table1.json"
    assert params["vqc_depth"] == 6
    assert tf.run_name("probe_x_binary_q8", args.run_tag) == "probe_x_binary_q8_table1"


def test_missing_variant_file_fails_loud(repo):
    with pytest.raises(FileNotFoundError, match="gas_binary"):
        tf.load_tuned_file(None, "gas_binary", 8)


def test_run_name_without_tag_is_unchanged():
    assert tf.run_name("adversarial_binary_q8", None) == "adversarial_binary_q8"


def test_with_vqc_from_copies_only_the_vqc_keys():
    other = {"mlp": {"h1": 99}, "vqc_depth": 2, "vqc_lr": 0.1, "vqc_batch": 32, "vqc_epochs": 40}
    got = tf.with_vqc_from(GRID, other)
    assert got["mlp"] == {"h1": 8}
    assert (got["vqc_depth"], got["vqc_lr"], got["vqc_batch"]) == (2, 0.1, 32)
    assert GRID["vqc_depth"] == 4  # the input file's dict is never edited


def test_with_vqc_from_fails_loud_on_a_missing_key():
    with pytest.raises(KeyError, match="vqc_epochs"):
        tf.with_vqc_from(GRID, {"vqc_depth": 6, "vqc_lr": 0.01, "vqc_batch": 128})
