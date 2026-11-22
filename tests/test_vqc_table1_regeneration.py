"""R4-24 analysis: the registered branch per figure test, the VQC-path rule the anchors split on,
the anchor comparison, and two derived figures, on tiny synthetic records (no data, no models)."""

import math
import sys

import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

# experiments/ is not installed
sys.path.insert(0, str(REPO_ROOT))
from experiments import run_vqc_table1_regeneration as m  # noqa: E402


def test_abs_branch_keeps_under_005_and_flags_a_sign_flip():
    assert m.branch("abs", 0.344, 0.300)["branch"] == "KEEP"
    assert m.branch("abs", 0.344, 0.290)["branch"] == "REWRITE"
    b = m.branch("abs", -0.012, 0.010)
    assert b["branch"] == "KEEP" and b["verdict_change"] is True
    assert m.branch("abs", 0.0, 0.01)["verdict_change"] is False  # zero has no sign


def test_ratio_branch_uses_the_band_only():
    assert m.branch("ratio", 10.16, 5.08)["branch"] == "KEEP"
    b = m.branch("ratio", 10.16, 4.62)
    assert b["branch"] == "REWRITE" and b["ratio_check"] is False
    assert m.branch("ratio", 0.887, 1.6)["branch"] == "KEEP"  # change 0.71, ratio 1.8


def test_claim_branch_is_a_verdict_change_when_it_flips():
    assert m.branch("claim", True, True) == {
        "abs_change": None,
        "ratio": None,
        "ratio_check": None,
        "branch": "KEEP",
        "verdict_change": False,
    }
    b = m.branch("claim", True, False)
    assert b["branch"] == "REWRITE" and b["verdict_change"] is True


@pytest.mark.parametrize(
    "path,vqc",
    [
        ("/0/sensor_noise/vqc/30", True),
        ("/0/f1/pgd_vqc/logreg", True),
        ("/0/ensembles/clean/cls3+vqc", True),
        ("/0/deployed_d6_lr0.0175/clean", True),
        ("/0/f1/pgd_mlp/logreg", False),
        ("/all/per_model/qsvm/0.1/macro_f1", False),
    ],
)
def test_vqc_path_rule(path, vqc):
    assert m.is_vqc_path(path) is vqc


def test_compare_gates_shared_leaves_and_reports_excluded():
    ref = {"0": {"logreg": 0.5, "xgb": 0.6, "vqc": 0.4, "only_ref": 1.0, "seconds": 3.0}}
    ours = {"0": {"logreg": 0.5, "xgb": 0.61, "vqc": 0.2, "seconds": 9.0}}
    a = m.compare(ours, ref, ("xgb",), only_non_vqc=True)
    assert a["passed"] is True and a["n_leaves"] == 1
    assert a["excluded_n_leaves"] == 1 and a["excluded_max_abs_diff"] == pytest.approx(0.01)
    full = m.compare(ours, ref, ("xgb",))
    assert full["passed"] is False and full["worst"] == "/0/vqc"
    nan = m.compare({"a": math.nan}, {"a": math.nan})
    assert nan["passed"] is True


def test_compare_without_non_vqc_leaves_is_not_applicable():
    out = m.compare({"vqc": 0.1}, {"vqc": 0.2}, only_non_vqc=True)
    assert out["passed"] is None and out["n_leaves"] == 0
    with pytest.raises(ValueError, match="no shared leaves"):
        m.compare({"a": 1.0}, {"b": 1.0})


def _seed(pgd, fgsm):
    return {
        "evasion": {
            "curves": {
                "vqc": {"pgd": {"0.5": pgd}, "fgsm": {"0.5": fgsm}},
                "mlp": {"pgd": {"0.5": 0.3}, "fgsm": {"0.5": 0.3}},
            }
        }
    }


def test_fgsm_pgd_agreement_on_seed_means():
    out = m.fgsm_pgd([_seed(0.10, 0.20), _seed(0.30, 0.20)])
    assert out["mean"] == pytest.approx(0.0) and out["max"] == pytest.approx(0.0)
    out = m.fgsm_pgd([_seed(0.10, 0.30)])
    assert out["mean"] == pytest.approx(0.1) and out["max"] == pytest.approx(0.2)


def test_sw_artifact_is_sample_weight_minus_as_tuned():
    targets = m.CLS5 + ["qsvm", "vqc"]
    clean = {t: 0.5 + 0.01 * i for i, t in enumerate(targets)}
    as_tuned = {t: 0.4 for t in targets}
    sw = {t: 0.4 + 0.02 * i for i, t in enumerate(targets)}
    res = {
        "summary": {"floor": 0.5},
        "per_seed": [{"f1": {"clean": clean, "pgd_mlp": as_tuned, "pgd_mlp_sample_weight": sw}}],
    }
    out = m.sw_artifact(res)
    assert out["vqc"] == pytest.approx(0.12)
    assert out["rho"] == pytest.approx(1.0)


def test_restart_range_anchor_passes_inside_and_fails_outside():
    """HopSkipJump anchors by range (ruling, decision log 2026-11-18T15:40Z): ART's unseeded
    start makes a 1e-6 anchor impossible, so each anchor restart must sit inside the archive's
    restart range per family and metric."""

    def rows(meds):
        return [
            {
                "restart": i,
                "median_linf_sigma": v,
                "mean_linf_sigma": v + 0.1,
                "queries_per_sample": 580.0 + i,
            }
            for i, v in enumerate(meds)
        ]

    archive = {"per_family": {"vqc": rows([0.37, 0.53, 0.44, 0.46, 0.49])}}
    inside = {"per_family": {"vqc": [{**rows([0.43])[0], "queries_per_sample": 581.0}]}}
    out = m.anchor_within_restart_range(inside, archive)
    assert out["passed"] and out["n_leaves"] == 1 and out["n_outside"] == 0
    outside = {"per_family": {"vqc": rows([0.60])}}
    bad = m.anchor_within_restart_range(outside, archive)
    assert not bad["passed"] and "median_linf_sigma" in bad["worst"]
