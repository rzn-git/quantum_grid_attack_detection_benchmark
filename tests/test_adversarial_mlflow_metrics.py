"""Regression (decision log 2026-11-17T05:31Z): a random-control-only run_adversarial run has no
evasion block, and its MLflow metrics line raised KeyError after results.json was written."""

import importlib.util
import sys

from qgridbench.utils.run_tracking import REPO_ROOT

if str(REPO_ROOT) not in sys.path:  # the runner imports experiments.exports
    sys.path.insert(0, str(REPO_ROOT))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [name]  # no --threads: nothing is pinned
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


ra = _load("run_adversarial")


def test_random_control_only_run_logs_no_clean_metrics():
    assert ra.clean_metrics_for_mlflow([{"random_control": {"mlp": {}}}]) == {}


def test_evasion_run_logs_first_seed_clean_metrics():
    per_seed = [
        {"evasion": {"clean": {"macro_f1": 0.9, "auprc": 0.8}}},
        {"evasion": {"clean": {"macro_f1": 0.1}}},
    ]
    assert ra.clean_metrics_for_mlflow(per_seed) == {"clean_macro_f1": 0.9, "clean_auprc": 0.8}
