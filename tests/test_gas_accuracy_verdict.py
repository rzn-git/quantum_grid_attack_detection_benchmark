"""H-C2 branch logic of the R4-12 accuracy verdict on synthetic per-seed scores."""

import importlib.util

import numpy as np
import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "run_gas_accuracy_verdict", REPO_ROOT / "experiments" / "run_gas_accuracy_verdict.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _f1(best, vqc, qsvm, noise=0.002):
    rng = np.random.default_rng(0)
    out = {m: list(0.50 + noise * rng.standard_normal(10)) for m in mod.CLASSICAL}
    out["xgb"] = list(best + noise * rng.standard_normal(10))
    out["vqc"] = list(vqc + noise * rng.standard_normal(10))
    out["qsvm_zz"] = [qsvm] * 10
    out["qsvm_angle_ry"] = [qsvm - 0.01] * 10
    return out


@pytest.mark.parametrize(
    ("best", "vqc", "qsvm", "branch"),
    [
        (0.60, 0.52, 0.55, "REPLICATES"),
        (0.60, 0.52, 0.595, "PARITY"),
        (0.60, 0.52, 0.61, "KILL"),
        (0.60, 0.52, 0.575, "INTERMEDIATE"),  # delta 0.025 but margin < 5 classical stds
    ],
)
def test_branches(best, vqc, qsvm, branch):
    v = mod.verdict(_f1(best, vqc, qsvm, noise=0.002 if branch != "INTERMEDIATE" else 0.01))
    assert v["best_classical"] == "xgb"
    assert v["branch_H_C2"] == branch


def _run(root, name, res):
    d = root / name
    d.mkdir()
    (d / "metrics.json").write_text('{"status": "ok"}')
    (d / "results.json").write_text(__import__("json").dumps(res))
    return name


def test_collect_reads_suffixed_run_dirs_and_keeps_variants_apart(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    seeds = {"per_seed": [{"macro_f1": 0.7}, {"macro_f1": 0.8}]}
    runs = [
        _run(tmp_path, "20261119T105911Z_classical_gas_binary_slog_full_2", {"mlp": seeds}),
        _run(tmp_path, "20261119T105911Z_classical_gas_binary_slog_pca8", {"rf": seeds}),
        _run(tmp_path, "20261119T105916Z_quantum_qkernel_gas_binary_slog_q8_2", {"zz": seeds}),
        _run(tmp_path, "20261119T105916Z_quantum_vqc_gas_binary_slog_q8", {"vqc": seeds}),
    ]
    f1 = mod.collect(runs, "gas_binary_slog")
    assert f1["full"] == {"mlp": [0.7, 0.8]}
    assert set(f1["pca8"]) == {"rf", "qsvm_zz", "vqc"}
    with pytest.raises(ValueError, match="not a Phase-1 run of gas_binary"):
        mod.collect(runs[:1], "gas_binary")
