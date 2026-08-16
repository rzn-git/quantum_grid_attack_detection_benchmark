"""Metric correctness on toy fixtures."""

import numpy as np

from qgridbench.eval.metrics import brier_score, compute_all, expected_calibration_error
from qgridbench.eval.stats import holm_bonferroni, paired_wilcoxon


def test_perfect_classifier_metrics():
    y = np.array([0, 1, 0, 1, 1])
    proba = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [0, 1]], dtype=float)
    m = compute_all(y, proba)
    assert m["macro_f1"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["auprc"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["brier"] == 0.0


def test_brier_and_ece_known_values():
    y = np.array([0, 1])
    proba = np.array([[0.7, 0.3], [0.4, 0.6]])
    # brier = mean( (0.7-1)^2+(0.3-0)^2 , (0.4-0)^2+(0.6-1)^2 ) = mean(0.18, 0.32)
    assert abs(brier_score(y, proba) - 0.25) < 1e-12
    # both predictions correct; ece is |acc - conf| weighted
    ece = expected_calibration_error(y, proba, n_bins=15)
    assert 0.0 <= ece <= 1.0


def test_holm_bonferroni_monotone_and_bounded():
    corr = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.5})
    assert corr["a"] <= corr["b"] <= corr["c"]
    assert all(0 <= v <= 1 for v in corr.values())
    # smallest p (0.01) times m=3
    assert abs(corr["a"] - 0.03) < 1e-12


def test_paired_wilcoxon_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0.9, 0.01, 10)
    b = a - 0.05  # a strictly better
    res = paired_wilcoxon(a, b)
    assert res["mean_diff"] > 0
    assert res["rank_biserial"] > 0.9
    assert res["p_value"] < 0.05


def test_generated_tex_escapes_underscores(tmp_path):
    """Generated .tex must compile: a bare `_` outside math mode aborts pdflatex.

    Model identifiers carry underscores (`rbf_svm`, `qsvm_angle_ry`), so this is a
    build-breaking class of defect that no numeric assertion would catch.
    """
    import json
    import re

    from qgridbench.eval import tables

    run = tmp_path / "run"
    run.mkdir()
    metrics = dict.fromkeys(tables.HEADLINE_METRICS, 0.5)
    (run / "results.json").write_text(
        json.dumps(
            {
                name: {
                    "aggregate": {m: {"mean": 0.5, "std": 0.01} for m in tables.HEADLINE_METRICS},
                    "per_seed": [metrics, metrics, metrics],
                }
                for name in ("rbf_svm", "qsvm_angle_ry", "xgb")
            }
        )
    )
    tables.TABLE_DIR = tmp_path / "tables"
    tex = tables.accuracy_table(run, "probe").read_text(encoding="utf-8")

    assert r"rbf\_svm" in tex and r"qsvm\_angle\_ry" in tex
    assert not re.search(r"(?<!\\)_", tex), "unescaped underscore would break pdflatex"


def test_every_figure_savefig_suppresses_the_pdf_timestamp():
    """Figures must be byte-reproducible, so no savefig may omit PDF_METADATA.

    Matplotlib stamps a wall-clock /CreationDate into each PDF; without
    suppression a reviewer re-running make_report gets artifacts that differ
    byte-for-byte from the released ones for no substantive reason. Static check
    because rendering every figure in the unit suite would be slow and would need
    real run dirs.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "qgridbench" / "eval" / "figures.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "savefig"
    ]
    assert calls, "no savefig calls found — did figures.py move?"
    for call in calls:
        kwargs = {k.arg for k in call.keywords}
        assert "metadata" in kwargs, (
            f"savefig at line {call.lineno} omits metadata=PDF_METADATA; "
            "the PDF timestamp would break byte-reproducibility"
        )
