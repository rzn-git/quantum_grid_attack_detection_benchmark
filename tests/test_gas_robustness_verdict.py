"""Branch logic of the R4-12 robustness verdict (H-C1a, H-C1b, H-C1c) on synthetic artifacts."""

import importlib.util

import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "run_gas_robustness_verdict", REPO_ROOT / "experiments" / "run_gas_robustness_verdict.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _wb(q_seeds, r_seeds, clean=0.5):
    return {
        "p5a_whitebox_qsvm": {
            "clean_f1": clean,
            "per_seed_pgd": {"0.5": q_seeds},
            "whitebox_retention_at_half": sum(q_seeds) / len(q_seeds) / clean,
            "rbf_control": {
                "clean_f1": clean,
                "per_seed_pgd": {"0.5": r_seeds},
                "whitebox_retention_at_half": sum(r_seeds) / len(r_seeds) / clean,
            },
        }
    }


@pytest.mark.parametrize(
    ("transfer", "q", "r", "branch"),
    [
        (0.90, 0.05, 0.35, "REPLICATES"),
        (0.90, 0.30, 0.20, "KILL"),
        (0.40, 0.05, 0.35, "INTERMEDIATE"),  # A = 0.30 - 0.1 retention gap too small
    ],
)
def test_h_c1a(transfer, q, r, branch):
    stats = {"per_family": {"qsvm": {"retention": transfer}}}
    qs = [q + 0.001 * i for i in range(10)]
    rs = [r + 0.001 * i for i in range(10)]
    assert mod.h_c1a(stats, _wb(qs, rs))["branch"] == branch


def _pairs(diffs):
    return {
        "quantum_vs_classical": {
            f"p{i}": {"mean_diff": d, "significant_holm_0.05": s} for i, (d, s) in enumerate(diffs)
        }
    }


def test_h_c1b():
    both = [(0.1, True)] * 6 + [(-0.1, True)] * 6
    one = [(0.1, True)] * 10 + [(0.1, False)] * 2
    few = [(0.1, True)] * 4 + [(0.1, False)] * 8
    assert mod.h_c1b(_pairs(both))["branch"] == "REPLICATES"
    assert mod.h_c1b(_pairs(one))["branch"] == "KILL"
    assert mod.h_c1b(_pairs(few))["branch"] == "INTERMEDIATE"


@pytest.mark.parametrize(
    ("ratio", "branch"),
    [(0.8, "REPLICATES"), (2.0, "REPLICATES"), (2.5, "INTERMEDIATE"), (3.5, "NEW_FINDING_escalate"),
     (0.3, "NEW_FINDING_escalate")],
)  # fmt: skip
def test_asymmetry_branches(ratio, branch):
    assert mod._asym_branch(ratio) == branch


def test_item2_factor_reads_both_statistics():
    s = {
        "mlp": {"forward_over_reverse": 10.0, "damage_ratio_vqc_over_classical": 1.6},
        "mlp_balanced": {"forward_over_reverse": 0.8, "damage_ratio_vqc_over_classical": 1.0},
    }
    out = mod.h_c1c({"summary": s})
    assert out["registered"] == "forward_over_reverse" and out["branch"] == "REPLICATES"
    assert out["forward_over_reverse"]["item2_reproduces"]
    assert not out["damage_ratio_vqc_over_classical"]["item2_reproduces"]


def test_assert_variant_rejects_a_run_of_another_pipeline(tmp_path, monkeypatch):
    """The verdict of one gas pipeline must not read a run of the other (experiment
    gas_slog_robustness, 2026-11-21)."""
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    for name, variant in (("run_main", "gas_binary"), ("run_slog", "gas_binary_slog")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "params.yaml").write_text(f"config:\n  variant: {variant}\n")
    mod.assert_variant("run_slog", "gas_binary_slog")
    with pytest.raises(ValueError, match="variant 'gas_binary', not 'gas_binary_slog'"):
        mod.assert_variant("run_main", "gas_binary_slog")
