"""R4-20 multiclass appendix: QSVM cell, merge order, counts CSV, figure, and the anchor table."""

import json
import sys

import numpy as np
import pytest

from qgridbench.eval import figures, tables
from qgridbench.protocol import Regime
from qgridbench.utils.run_tracking import REPO_ROOT

# experiments/ is not installed; imported by name so its spawn-safe functions pickle
sys.path.insert(0, str(REPO_ROOT))
from experiments import run_multiclass_confusion as mod  # noqa: E402


def _toy_regime():
    rng = np.random.default_rng(0)
    y = np.repeat([1, 5, 9], 12)  # three scenario markers, not 0..k-1
    centers = {1: (-2.0, -2.0), 5: (0.0, 2.0), 9: (2.0, -1.0)}
    a = np.array([centers[c] for c in y]) + 0.2 * rng.standard_normal((len(y), 2))
    return Regime("pca", 2, X={}, y={"train": y, "test": y}, angles={"train": a, "test": a})


def test_qsvm_cell_is_one_deterministic_fit_counted_per_seed(monkeypatch):
    reg = _toy_regime()
    monkeypatch.setattr(mod, "prepare_regime", lambda *a, **k: reg)
    monkeypatch.setattr(mod, "kernel_subset", lambda r, cap, seed: np.arange(len(r.y["train"])))
    rec = mod.SeedRecorder("pca", np.array([1, 5, 9]), np.arange(100, 136))
    out = mod.qsvm_confusion("toy", [0, 1, 2], 2000, {"bandwidth": 0.5, "C": 10.0}, rec)
    cm = np.array(out["per_model"]["qsvm"]["confusion_counts"])
    assert out["labels"] == [1, 5, 9]
    assert cm.shape == (3, 3) and (cm % 3 == 0).all()  # three identical seed copies
    assert cm.sum(axis=1).tolist() == [36, 36, 36]
    assert np.trace(cm) == cm.sum()  # separable toy clusters
    pc = out["per_model"]["qsvm"]["per_class"]
    assert pc["5"]["f1"] == {"mean": 1.0, "std": 0.0}
    # exports: one metrics row and 36 prediction rows per seed, markers not codes
    assert [r["seed"] for r in rec.per_seed] == [0, 1, 2]
    assert rec.per_seed[0]["macro_f1"] == 1.0 == rec.per_seed[0]["macro_f1_predicted"]
    pred = rec.predictions()
    assert len(pred) == 3 * 36 and set(pred["y_true"]) == {1, 5, 9}
    assert (pred["y_true"] == pred["y_pred"]).all() and pred["row"].iloc[0] == 100
    probs = pred[["p_1", "p_5", "p_9"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_recorder_writes_markers_and_fails_loud_on_a_misaligned_probability_matrix():
    rec = mod.SeedRecorder("full", np.array([2, 7]), np.array([10, 11, 12]))
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    rec.add("rf", 4, np.array([0, 1, 1]), np.array([0, 1, 0]), proba)
    row = rec.per_seed[0]
    assert (row["regime"], row["model"], row["seed"]) == ("full", "rf", 4)
    assert row["macro_f1_predicted"] == pytest.approx(2 / 3)
    frame = rec.predictions()
    assert frame["y_true"].tolist() == [2, 7, 7] and frame["y_pred"].tolist() == [2, 7, 2]
    assert frame["p_7"].tolist() == pytest.approx([0.1, 0.8, 0.4])
    with pytest.raises(SystemExit, match="proba"):
        rec.add("rf", 5, np.array([0, 1, 1]), np.array([0, 1, 0]), proba[:, :1])


def test_write_exports_copies_one_part_per_unit_and_the_per_seed_rows(tmp_path):
    units = [("v", "full", "rf", None), ("v", "pca", "qsvm", None)]
    parts, results = {}, {}
    for u in units:
        rec = mod.SeedRecorder(u[1], np.array([2, 7]), np.array([0, 1]))
        rec.add(u[2], 0, np.array([0, 1]), np.array([0, 1]), np.array([[1.0, 0.0], [0.0, 1.0]]))
        parts[u] = tmp_path / f"{u[2]}.part.parquet"
        rec.predictions().to_parquet(parts[u], index=False)
        results[u] = {"per_seed": rec.per_seed}
    run = tmp_path / "run"
    run.mkdir()
    assert mod.write_exports(run, units, results, parts) == {
        "per_seed_rows": 2,
        "prediction_parts": 2,
    }
    import pandas as pd

    per_seed = pd.read_csv(run / "per_seed.csv")
    assert per_seed[["regime", "model"]].values.tolist() == [["full", "rf"], ["pca", "qsvm"]]
    both = pd.read_parquet(run / "predictions.parquet")
    assert len(both) == 4 and set(both["model"]) == {"rf", "qsvm"}


def test_merge_keeps_unit_order_and_fails_loud_on_label_mismatch():
    cell = {"labels": [1, 2], "class_support": {"1": 3, "2": 4}}
    units = [("v", "pca", "rf", None), ("v", "pca", "logreg", None), ("v", "full", "rf", None)]
    res = {
        units[0]: {**cell, "per_model": {"rf": {"x": 1}}},
        units[1]: {**cell, "per_model": {"logreg": {"x": 2}}},
        units[2]: {**cell, "per_model": {"rf": {"x": 3}}},
    }
    m = mod.merge(res, units, "pca", "v", [0, 1])
    assert list(m["per_model"]) == ["rf", "logreg"] and m["n_components"] == mod.QUBITS
    assert m["notes"] == mod.NOTES and m["tuned_run"] == mod.TUNED_RUNS["pca"]
    res[units[1]] = {**cell, "class_support": {"1": 3, "2": 5}, "per_model": {}}
    with pytest.raises(SystemExit, match="disagree"):
        mod.merge(res, units, "pca", "v", [0, 1])


def _artifact(models):
    labels = [1, 5, 9]
    cm = [[8, 2, 0], [0, 10, 0], [1, 0, 9]]
    pc = {str(c): {"f1": {"mean": 0.9, "std": 0.01}} for c in labels}
    return {
        "labels": labels,
        "class_support": {"1": 10, "5": 10, "9": 10},
        "seeds": [0, 1],
        "tuned_run": "results/runs/merged_accuracy_triple_pca8",
        "n_components": 8,
        "notes": "toy",
        "excluded_units": [],
        "per_model": {m: {"confusion_counts": cm, "per_class": pc} for m in models},
    }


def test_counts_csv_and_figure_from_a_toy_artifact(tmp_path, monkeypatch):
    for regime, fname in tables.MULTICLASS_CONFUSION_FILES.items():
        models = ["rf", "qsvm"] if regime == "pca" else ["rf"]
        (tmp_path / fname).write_text(json.dumps(_artifact(models)))
    out = tables.multiclass_confusion_counts(tmp_path, "cm", out_dir=tmp_path / "t")
    rows = out.read_text().splitlines()
    assert rows[0] == "regime,model,true_scenario,predicted_scenario,count"
    assert len(rows) == 1 + 3 * 9  # (full rf, pca rf, pca qsvm) x 3 x 3 cells, zeros kept
    assert "pca,qsvm,9,1,1" in rows
    f1 = (tmp_path / "t" / "cm_per_class_f1.csv").read_text().splitlines()
    assert f1[1] == "full,rf,1,10,0.9000,0.0100"
    monkeypatch.setattr(figures, "FIG_DIR", tmp_path / "f")
    pdf = figures.multiclass_confusion_figure(tmp_path, "pca", "cm_pca")
    assert pdf.exists() and pdf.stat().st_size > 0


ARCHIVE = REPO_ROOT / mod.ANCHOR_RUN


@pytest.mark.skipif(not ARCHIVE.exists(), reason="DVC-tracked archive run not pulled")
def test_anchor_table_path_reproduces_the_shipped_csv_from_the_archive(tmp_path):
    """The anchor's comparison is meaningful only if the table writer, pointed elsewhere,
    rebuilds the shipped CSV from the archived artifacts byte for byte."""
    tables.three_class_confusion_table(ARCHIVE, "confusion_three_class", out_dir=tmp_path)
    assert (tmp_path / "confusion_three_class.csv").read_bytes() == mod.ANCHOR_CSV.read_bytes()


def test_exclude_drops_named_units_and_fails_loud_on_unknown_names():
    units = [("v", "full", "rf", None), ("v", "full", "logreg", None), ("v", "pca", "rf", None)]
    assert mod.exclude_units(units, ["full/rf"]) == units[1:]
    assert mod.exclude_units(units, []) == units
    with pytest.raises(SystemExit, match="names no planned unit"):
        mod.exclude_units(units, ["full/vqc"])


def test_encode_and_decode_round_trip_scenario_markers():
    markers = np.array([1, 5, 9, 41])
    reg = Regime(
        "full",
        None,
        X={},
        y={"train": np.array([41, 1, 5, 9]), "test": np.array([9, 1])},
        angles=None,
    )
    enc = mod.encode_regime(reg, markers)
    assert enc.y["train"].tolist() == [3, 0, 1, 2] and enc.y["test"].tolist() == [2, 0]
    out = {
        "labels": [0, 1, 2, 3],
        "class_support": {"0": 4, "1": 5, "2": 6, "3": 7},
        "per_model": {
            "xgb": {
                "confusion_counts": [[1]],
                "per_class": {str(i): {"f1": i} for i in range(4)},
                "natural_called_attack": 0.1,
                "attack_called_natural": 0.2,
            }
        },
    }
    dec = mod.decode_cell(out, markers)
    assert dec["labels"] == [1, 5, 9, 41] and dec["class_support"]["41"] == 7
    assert dec["per_model"]["xgb"]["per_class"]["9"] == {"f1": 2}
    assert "natural_called_attack" not in dec["per_model"]["xgb"]
    with pytest.raises(SystemExit, match="outside the release"):
        mod.encode_regime(
            Regime("full", None, X={}, y={"train": np.array([2])}, angles=None), markers
        )
    with pytest.raises(SystemExit, match="are not the codes"):
        mod.decode_cell({"labels": [0, 2]}, markers)


def test_merge_discloses_excluded_units():
    cell = {"labels": [1], "class_support": {"1": 3}, "per_model": {"logreg": {}}}
    units = [("v", "full", "logreg", None)]
    m = mod.merge({units[0]: cell}, units, "full", "v", [0], ["full/rf"])
    assert m["excluded_units"] == ["full/rf"] and "full/rf" in m["notes"]
    assert mod.merge({units[0]: cell}, units, "full", "v", [0])["notes"] == mod.NOTES


def test_anchor_compare_reports_a_missing_family(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    archive.mkdir()
    cm = [[1, 0], [0, 1]]
    both = {"rf": {"confusion_counts": cm}, "xgb": {"confusion_counts": cm}}
    for fname in mod.ANCHOR_FILES.values():
        (archive / fname).write_text(json.dumps({"per_model": both}))
        (tmp_path / fname).write_text(json.dumps({"per_model": {"xgb": both["xgb"]}}))
    shipped = tmp_path / "shipped.csv"
    shipped.write_text("x")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "ANCHOR_RUN", "archive")
    monkeypatch.setattr(mod, "ANCHOR_CSV", shipped)
    monkeypatch.setattr(
        mod,
        "three_class_confusion_table",
        lambda run, name, out_dir: (out_dir / f"{name}.csv").write_text("y"),
    )
    out = mod.anchor_compare(tmp_path)
    assert out["csv_exact"] is False
    assert out["counts_exact"]["full"] == {"rf": "missing", "xgb": True}


def test_anchor_passes_byte_for_byte_or_per_family_under_exclusions():
    exact = {"full": {"rf": True, "xgb": True}, "pca": {"rf": True}}
    assert mod.anchor_passes({"csv_exact": True, "counts_exact": exact}, [])
    assert not mod.anchor_passes({"csv_exact": False, "counts_exact": exact}, [])
    amended = {"full": {"rf": "missing", "xgb": True}, "pca": {"rf": True}}
    assert mod.anchor_passes({"csv_exact": False, "counts_exact": amended}, ["full/rf"])
    off = {"full": {"rf": "missing", "xgb": False}, "pca": {"rf": True}}
    assert not mod.anchor_passes({"csv_exact": False, "counts_exact": off}, ["full/rf"])
    gone = {"full": {"rf": "missing", "xgb": "missing"}, "pca": {"rf": True}}
    assert not mod.anchor_passes({"csv_exact": False, "counts_exact": gone}, ["full/rf"])
