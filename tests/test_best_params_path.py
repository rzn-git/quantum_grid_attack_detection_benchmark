"""The tuned-parameter file is resolved per variant and fails loud when it is absent.

Regression guard for the R4-12 hazard: robustness runners used to read the grid's
`adv_best_params_binary_q8.json` by literal or default, so a run on another variant would
have fitted its models with the grid's hyperparameters without any error.
"""

from pathlib import Path

import pytest

import qgridbench.protocol as protocol
from qgridbench.utils.run_tracking import REPO_ROOT

RUNNERS = [
    "run_adversarial",
    "run_whitebox_qsvm",
    "run_surrogate_control",
    "run_square_blackbox",
    "run_evasion_per_class",
    "run_hopskipjump_restarts",
    "run_review_claims",
    "run_constant_shift",
    "run_vqc_objective_axis",
    "run_adversarial_augmentation",
    "run_reuploading_fragility",
    "run_physical_realizability",
    "run_sklearn_sample_weight",
]


def test_resolves_the_variant_file(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "REPO_ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    target = tmp_path / "results" / "adv_best_params_binary_q8.json"
    target.write_text("{}")
    assert protocol.best_params_path("binary", 8) == target


def test_missing_variant_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "REPO_ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "adv_best_params_binary_q8.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="gas_binary"):
        protocol.best_params_path("gas_binary", 8)


@pytest.mark.parametrize("runner", RUNNERS)
def test_runners_read_through_the_helper(runner):
    src = Path(REPO_ROOT / "experiments" / f"{runner}.py").read_text(encoding="utf-8")
    assert "adv_best_params_binary_q8" not in src
    # directly, or through experiments/tuned_file.py, which resolves with best_params_path
    assert "best_params_path(" in src or "load_tuned_file(" in src
