"""Shared exports helper: a toy write and read-back of all four files, and the fail-loud duplicate
keys (tiny arrays and a fake splits file; no data, no models)."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

# experiments/ is not installed
sys.path.insert(0, str(REPO_ROOT))
from experiments import exports as ex_mod  # noqa: E402

TEST_IDS = np.array([10, 11, 12, 13, 14])


@pytest.fixture
def ex(monkeypatch, tmp_path):
    import qgridbench.data.preprocess as pre

    monkeypatch.setattr(pre, "load_processed", lambda v: (None, None, {"test": TEST_IDS}))
    (tmp_path / "tuned.json").write_text("{}")
    monkeypatch.setattr(ex_mod, "REPO_ROOT", tmp_path)
    return ex_mod.Exports("binary", "tuned.json")


def test_write_and_read_back(ex, tmp_path):
    y = np.array([0, 1, 1])
    pos = np.array([0, 2, 4])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    X = np.arange(6.0).reshape(3, 2)
    out = ex.score(
        0,
        {"clean": X, "pgd@0.5": X + 1},
        ["mlp", "vqc"],
        lambda m, Z: proba,
        y,
        pos,
        lambda yy, p: float((p.argmax(1) == yy).mean()),
        {"pgd@0.5": "mlp"},
    )
    assert out["clean"]["vqc"] == pytest.approx(2 / 3)
    ex.stream(ex.stream("adv"))
    ex.extra["note"] = "x"
    run = tmp_path / "run"
    run.mkdir()
    ex.write(run)

    m = pd.read_csv(run / "per_seed.csv")
    assert len(m) == 4 and set(m["condition"]) == {"clean", "pgd@0.5"}
    p = pd.read_parquet(run / "predictions.parquet")
    assert len(p) == 12
    clean_vqc = p[(p.condition == "clean") & (p.model == "vqc")]
    assert clean_vqc["row_id"].tolist() == [10, 12, 14]
    assert clean_vqc["label"].tolist() == [0, 1, 0]
    assert clean_vqc["proba_1"].tolist() == pytest.approx([0.1, 0.8, 0.4])
    c = np.load(run / "crafted_inputs.npz")
    assert list(c.files) == ["0|pgd@0.5|mlp"]
    np.testing.assert_array_equal(c["0|pgd@0.5|mlp"], X + 1)
    prov = json.loads((run / "exports_provenance.json").read_text())
    assert prov["random_streams"] == ["adv"] and prov["note"] == "x"
    assert prov["tuned_file"]["path"] == "tuned.json" and len(prov["tuned_file"]["sha256"]) == 64
    assert "QGB_VQC_CHUNK" in prov["env"]


def test_mixed_seed_labels_write_as_text(ex, tmp_path):
    """Regression (decision log 2026-11-18, R4-24 data-efficiency anchor): a run that keys a
    deterministic model by the label "deterministic" beside integer seeds crashed in
    to_parquet; such a run now writes every seed as text, and integer-only runs keep ints."""
    y, pos, p = np.array([0, 1]), np.array([0, 1]), np.array([[0.9, 0.1], [0.3, 0.7]])
    ex.predictions(0, "size250", "vqc", p, y, pos)
    ex.predictions("deterministic", "size250", "qkernel_angle_ry", p, y, pos)
    run = tmp_path / "mixed"
    run.mkdir()
    ex.write(run)
    got = pd.read_parquet(run / "predictions.parquet")
    assert sorted(set(got["seed"])) == ["0", "deterministic"]


def test_duplicate_keys_fail_loud(ex):
    y, pos, p = np.array([0]), np.array([0]), np.array([[0.5, 0.5]])
    ex.predictions(0, "clean", "vqc", p, y, pos)
    with pytest.raises(KeyError, match="recorded twice"):
        ex.predictions(0, "clean", "vqc", p, y, pos)
    ex.crafted_input(0, "pgd", "mlp", np.zeros((1, 2)))
    with pytest.raises(KeyError, match="recorded twice"):
        ex.crafted_input(0, "pgd", "mlp", np.zeros((1, 2)))
    with pytest.raises(ValueError, match="length mismatch"):
        ex.predictions(1, "clean", "vqc", p, np.array([0, 1]), pos)


def test_run_adversarial_record_hook_is_plain_f1_without_exports(ex):
    from experiments.run_adversarial import _f1, _rec

    y, pos = np.array([0, 1, 1]), np.array([0, 1, 2])
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.7, 0.3]])
    assert _rec(None, 0, pos, y, "clean", "vqc", proba) == _f1(y, proba)
    X = np.ones((3, 2))
    assert _rec(ex, 0, pos, y, "pgd@0.5", "vqc", proba, X, "mlp") == _f1(y, proba)
    assert ("0", "pgd@0.5", "vqc") not in ex.preds and (0, "pgd@0.5", "vqc") in ex.preds
    assert "0|pgd@0.5|mlp" in ex.crafted
