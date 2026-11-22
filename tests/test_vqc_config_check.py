"""R4-23 runner: the arm parameters, the anchors, the registered figures and the branch, and the
surrogate-control summary it reuses, on tiny synthetic records (no data, no models)."""

import importlib
import os
import sys
from pathlib import Path

import pytest

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
FAMS = ["logreg", "rbf_svm", "rf", "xgb", "lgbm", "mlp", "qsvm", "vqc"]


@pytest.fixture(scope="module")
def m():
    """The runner reads --threads from argv at import and sets the thread variables; both are
    restored after the module's tests."""
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(Path(__file__).resolve().parents[1]))
        mp.setattr(sys, "argv", ["pytest", "--threads", "1"])
        for v in THREAD_VARS:
            if v in os.environ:
                mp.setenv(v, os.environ[v])
            else:
                mp.delenv(v, raising=False)
        yield importlib.import_module("experiments.run_vqc_config_check")


@pytest.fixture(scope="module")
def sc(m):
    return importlib.import_module("experiments.run_surrogate_control")


def test_arm_params_copies_and_replaces_only_existing_keys(m):
    base = {"mlp": {"h1": 8}, "vqc_depth": 4, "vqc_lr": 0.04, "vqc_batch": 128}
    got = m.arm_params(base, {"vqc_depth": 6, "vqc_lr": 0.0175})
    assert got == {"mlp": {"h1": 8}, "vqc_depth": 6, "vqc_lr": 0.0175, "vqc_batch": 128}
    assert base["vqc_depth"] == 4  # the grid file's dict is never edited
    with pytest.raises(KeyError, match="vqc_layers"):
        m.arm_params(base, {"vqc_layers": 6})


def _evasion(vqc_clean=0.5, vqc_pgd=0.03, mlp_to_vqc=0.2, rev=0.4, d=0.0):
    clean = {f: 0.5 + d for f in FAMS}
    clean["vqc"] = vqc_clean
    curve = {"clean": 0.5, "fgsm": {"0.1": 0.4, "0.5": 0.01 + d}, "pgd": {"0.1": 0.4, "0.5": 0.3}}
    curves = {f: dict(curve) for f in FAMS}
    curves["vqc"] = {**curve, "pgd": {"0.1": 0.4, "0.5": vqc_pgd}}
    row = {f: rev for f in FAMS}
    return {
        "clean": clean,
        "curves": curves,
        "transfer_matrix": {
            "mlp": {**row, "vqc": mlp_to_vqc},
            "vqc": {**row, "vqc": vqc_pgd},
            "mlp_surrogate": {**row, "vqc": mlp_to_vqc},
        },
    }


def test_leaf_diffs_reads_every_leaf_and_fails_on_a_missing_key(m):
    a, b = _evasion(), _evasion(d=1e-7)
    diffs = m.leaf_diffs(b, a)
    assert len(diffs) == 8 + 8 * 5 + 3 * 8
    assert max(diffs.values()) == pytest.approx(1e-7, abs=1e-12)
    del b["transfer_matrix"]["vqc"]["rf"]
    with pytest.raises(KeyError, match="transfer_matrix/vqc"):
        m.leaf_diffs(b, a)


def test_anchor_reads_depth4_units_only_and_fails_past_tolerance(m):
    ref = {0: _evasion(), 1: _evasion()}
    units = [
        {"arm": "depth4", "seed": 0, "evasion": _evasion(d=5e-7)},
        {"arm": "depth4", "seed": 1, "evasion": _evasion()},
        {"arm": "depth6", "seed": 0, "evasion": _evasion(d=0.3)},
    ]
    ok = m.anchor(units, ref, "evasion")
    assert ok["passed"] and ok["max_abs_diff"] == pytest.approx(5e-7, abs=1e-12)
    units[1]["evasion"] = _evasion(d=2e-6)
    bad = m.anchor(units, ref, "evasion")
    assert not bad["passed"] and bad["worst"].startswith("seed1/")
    with pytest.raises(ValueError, match="no depth-4 units"):
        m.anchor(units[2:], ref, "evasion")


def _sur(fwd, rev):
    return {"mlp": {"vqc_damage": fwd, "reverse_damage": rev, "forward_over_reverse": fwd / rev}}


def test_figures_are_seed_means_and_retention_is_over_the_target_clean_mean(m):
    units = [{"evasion": _evasion(vqc_clean=0.4)}, {"evasion": _evasion(vqc_clean=0.6)}]
    fig = m.figures(units, _sur(0.34, 0.034))
    assert fig["vqc_clean_f1"] == pytest.approx(0.5)
    assert fig["vqc_whitebox_pgd_f1@0.5"] == pytest.approx(0.03)
    assert fig["vqc_whitebox_fgsm_f1@0.5"] == pytest.approx(0.01)
    assert fig["retention_mlp->vqc"] == pytest.approx(0.2 / 0.5)
    assert fig["retention_mlp_surrogate->vqc"] == pytest.approx(0.2 / 0.5)
    assert fig["retention_vqc->rf"] == pytest.approx(0.4 / 0.5)
    assert fig["retention_vqc->vqc"] == pytest.approx(0.03 / 0.5)
    assert fig["asymmetry_difference"] == pytest.approx(0.306)
    assert fig["asymmetry_ratio"] == pytest.approx(10.0)
    assert sum(k.startswith("retention_vqc->") for k in fig) == len(FAMS)


def test_branch_consistent_material_on_a_move_and_material_on_the_ratio(m):
    units = [{"evasion": _evasion()}]
    f4 = m.figures(units, _sur(0.34, 0.034))
    # the ratio may move a lot inside its band without counting as a 0.05 move
    f6 = m.figures([{"evasion": _evasion(vqc_clean=0.5 + 0.049)}], _sur(0.34, 0.06))
    v = m.compare(f4, f6, 10.16)
    assert v["branch"] == "CONSISTENT", v
    assert v["ratio_band"] == pytest.approx([5.08, 20.32])
    f6 = m.figures([{"evasion": _evasion(vqc_pgd=0.03 + 0.05)}], _sur(0.34, 0.034))
    v = m.compare(f4, f6, 10.16)
    assert v["branch"] == "MATERIAL" and "vqc_whitebox_pgd_f1@0.5" in v["figures_at_or_over_tol"]
    f6 = m.figures(units, _sur(0.34, 0.07))  # ratio 4.86, under 0.5 x 10.16
    v = m.compare(f4, f6, 10.16)
    assert v["branch"] == "MATERIAL" and not v["figures_at_or_over_tol"]


def test_per_class_counts_fail_when_units_saw_different_rows(m):
    rec = {"n_attack": 355, "n_natural": 145, "attack_recall": 0.9, "false_alarm": 0.2}
    units = [{"clean_per_class": {"vqc": rec}}, {"clean_per_class": {"vqc": dict(rec)}}]
    assert m.per_class_counts(units)["n_attack_n_natural"] == [355, 145]
    units[1]["clean_per_class"]["vqc"]["n_attack"] = 354
    with pytest.raises(AssertionError, match="eval rows differ"):
        m.per_class_counts(units)


def test_surrogate_summary_forward_reverse_and_ratio(sc):
    models = FAMS + ["mlp_balanced"]

    def seed(vqc_fwd, mlp_rev):
        f1 = {"clean": {t: 0.5 for t in models}}
        for k in ("pgd_mlp", "pgd_mlp_balanced", "pgd_vqc", "pgd_rbf_svm"):
            f1[k] = {t: 0.4 for t in models}  # every classical target loses 0.1
        f1["pgd_mlp"]["vqc"] = 0.5 - vqc_fwd
        f1["pgd_vqc"]["mlp"] = 0.5 - mlp_rev
        return {"f1": f1, "surrogate_positive_rate": {"mlp": 0.9, "mlp_balanced": 0.5}}

    s = sc.summarize([seed(0.3, 0.02), seed(0.4, 0.05)], 0.49, 0.5, 0.71)
    assert s["mlp"]["vqc_damage"] == pytest.approx(0.35)
    assert s["mlp"]["reverse_damage"] == pytest.approx(0.035)
    assert s["mlp"]["forward_over_reverse"] == pytest.approx(10.0)
    assert s["mlp"]["damage_ratio_vqc_over_classical"] == pytest.approx(3.5)
    assert s["mlp_balanced"]["vqc_damage"] == pytest.approx(0.1)


def test_anchor_reports_an_excluded_model_without_gating_on_it(m):
    ref = {0: {"f1": {"clean": {"vqc": 0.5, "xgb": 0.6}}}}
    units = [
        {"arm": "depth4", "seed": 0, "surrogate": {"f1": {"clean": {"vqc": 0.5, "xgb": 0.62}}}}
    ]
    assert not m.anchor(units, ref, "surrogate")["passed"]
    a = m.anchor(units, ref, "surrogate", ("xgb",))
    assert a["passed"] and a["n_leaves"] == 1
    assert a["excluded_n_leaves"] == 1 and a["excluded_max_abs_diff"] == pytest.approx(0.02)


def test_asymmetry_anchor(m):
    ref = _sur(0.3442, 0.0339)
    assert m.asymmetry_anchor(_sur(0.3442, 0.0339), ref)["passed"]
    assert not m.asymmetry_anchor(_sur(0.3442, 0.0340), ref)["passed"]
