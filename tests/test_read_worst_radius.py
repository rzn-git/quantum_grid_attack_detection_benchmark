"""The worst-radius reading of a stored white-box curve (experiments/read_worst_radius.py)."""

import numpy as np
import pandas as pd
import pytest

from experiments.read_worst_radius import (
    constant_predictor_floor,
    end_state,
    read_model,
    reading_radius_verdict,
    registered_verdicts,
    worst_radius,
)

# a curve that dips inside the ball and recovers at the budget, three attack starts
DIP = {
    "0.01": [0.50, 0.52, 0.51],
    "0.1": [0.10, 0.12, 0.11],
    "0.3": [0.20, 0.22, 0.21],
    "0.5": [0.40, 0.42, 0.41],
    "2.0": [0.01, 0.01, 0.01],
}


def test_worst_radius_is_the_lowest_mean_at_or_below_the_budget():
    assert worst_radius(DIP, 0.5) == "0.1"
    # a radius above the budget is never read, however low its score
    assert worst_radius(DIP, 2.0) == "2.0"
    assert worst_radius(DIP, 0.01) == "0.01"


def test_a_tie_goes_to_the_smaller_radius():
    assert worst_radius({"0.1": [0.2, 0.2], "0.3": [0.2, 0.2], "0.5": [0.3, 0.3]}, 0.5) == "0.1"


def test_no_radius_inside_the_budget_fails():
    with pytest.raises(ValueError, match="no stored radius"):
        worst_radius({"0.5": [0.1]}, 0.1)


def test_read_model_gives_both_readings():
    m = read_model(0.6, DIP, 0.5, floor=0.415)
    assert m["at_budget"]["radius"] == 0.5 and m["worst"]["radius"] == 0.1
    assert m["at_budget"]["retention"] == pytest.approx(0.41 / 0.6)
    assert m["worst"]["retention"] == pytest.approx(0.11 / 0.6)
    assert m["d_retention"] == pytest.approx(0.30 / 0.6)
    assert m["worst"]["std"] == pytest.approx(np.std([0.10, 0.12, 0.11], ddof=1))
    assert m["worst"]["starts_at_or_below_the_budget_reading"] == 3
    assert m["at_budget"]["end_state"] == "disabled" and m["worst"]["end_state"] == "inverted"
    assert not m["monotone_inside_budget"]


def test_a_monotone_curve_is_read_at_the_budget():
    mono = {"0.1": [0.5, 0.5], "0.3": [0.3, 0.3], "0.5": [0.1, 0.1]}
    m = read_model(0.6, mono, 0.5, floor=None)
    assert m["worst"]["radius"] == 0.5 and m["d_retention"] == 0.0
    assert m["monotone_inside_budget"] and "end_state" not in m["worst"]


def test_read_model_fails_on_a_budget_that_was_not_run():
    with pytest.raises(ValueError, match="not a stored radius"):
        read_model(0.6, DIP, 0.4, floor=None)


def test_read_model_fails_on_unequal_start_counts():
    with pytest.raises(ValueError, match="different numbers of attack starts"):
        read_model(0.6, {"0.1": [0.1, 0.2], "0.5": [0.3]}, 0.5, floor=None)


def test_end_state_labels():
    assert end_state(0.42, 0.415) == "disabled"
    assert end_state(0.20, 0.415) == "inverted"
    assert end_state(0.30, 0.415) == "partial"


def test_reading_radius_bars():
    assert reading_radius_verdict(0.44, 0.20, 0.05) == "yes"
    assert reading_radius_verdict(0.20, 0.20, 0.05) == "yes"
    assert reading_radius_verdict(0.07, 0.20, 0.05) == "partial"
    assert reading_radius_verdict(0.02, 0.20, 0.05) == "no"


def test_registered_verdicts_follow_each_source_rule():
    gas = {"rule": "gas", "a_min": 0.30, "b_min": 0.20, "alpha": 0.05}
    assert registered_verdicts(gas, 0.75, 0.001, 0.5) == {
        "availability": "REPLICATES",
        "contrast": "KILL",
    }
    assert registered_verdicts(gas, 0.75, -0.25, 0.002)["contrast"] == "REPLICATES"
    assert registered_verdicts(gas, 0.75, -0.25, 0.20)["contrast"] == "INTERMEDIATE"
    assert registered_verdicts(gas, 0.10, -0.25, 0.002)["availability"] == "NOT"
    grid = {"rule": "grid", "a_min": 0.20, "f_max": -0.10}
    assert registered_verdicts(grid, 0.08, -0.09, 0.002) == {
        "availability": "NOT",
        "contrast": "NOT",
    }
    assert registered_verdicts(grid, 0.20, -0.10, 0.002) == {
        "availability": "REPLICATES",
        "contrast": "REPLICATES",
    }
    with pytest.raises(ValueError, match="unknown registered rule"):
        registered_verdicts({"rule": "other"}, 0.1, 0.1, 0.1)


def _labels(tmp_path, y, pred):
    rows = [
        {"seed": s, "condition": c, "model": "qsvm", "row_id": i, "y_true": int(t), "label": int(p)}
        for s in (0, 1)
        for c in ("clean", "pgd")
        for i, (t, p) in enumerate(zip(y, pred))
    ]
    path = tmp_path / "predictions.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_floor_is_the_majority_class_score_on_the_stored_labels(tmp_path):
    y = np.array([1] * 7 + [0] * 3)
    pred = np.array([1] * 6 + [0] + [0] * 2 + [1])
    path = _labels(tmp_path, y, pred)
    from sklearn.metrics import f1_score

    clean = f1_score(y, pred, average="macro")
    out = constant_predictor_floor(path, 10, clean)
    # predicting "attack" everywhere: F1 of the attack class 14/17, of the other class 0
    assert out["floor"] == pytest.approx(0.5 * 14 / 17)
    assert out["majority_share"] == pytest.approx(0.7) and out["n_rows"] == 10


def test_floor_fails_on_rows_that_are_not_the_kernel_runs(tmp_path):
    y = np.array([1] * 7 + [0] * 3)
    path = _labels(tmp_path, y, y)
    with pytest.raises(ValueError, match="clean QSVM rows"):
        constant_predictor_floor(path, 500, 1.0)
    with pytest.raises(ValueError, match="not the kernel run's rows or model"):
        constant_predictor_floor(path, 10, 0.9)


def _curve(**points):
    return {"curve": {r.replace("_", "."): {"f1_minus_floor": v} for r, v in points.items()}}


def test_read_sweep_reads_the_budget_and_the_worst_radius():
    from experiments.read_worst_radius import read_sweep

    block = {
        "constant_predictor_floor": 0.415,
        "per_width": {
            "1.0": _curve(r0_0=0.19, r0_1=-0.27, r0_5=-0.02, r2_0=0.0),
            "0.03": _curve(r0_0=0.19, r0_5=-0.39, r2_0=-0.17),
        },
    }
    block["per_width"] = {
        k: {"curve": {r[1:]: c for r, c in v["curve"].items()}}
        for k, v in block["per_width"].items()
    }
    out = read_sweep(block, "per_width", 2.0, 2)
    tuned, wide = out["settings"]["1.0"], out["settings"]["0.03"]
    assert tuned["at_budget"]["end_state"] == "disabled"
    assert tuned["worst"]["radius"] == 0.1 and tuned["worst"]["end_state"] == "inverted"
    assert wide["at_budget"]["end_state"] == "partial" and wide["worst"]["radius"] == 0.5
    # the clean score at radius 0 is never the worst attack reading
    assert out["worst_highest"] == pytest.approx(-0.27)
    assert out["worst_lowest"] == pytest.approx(-0.39)
    assert out["n_inverted_at_budget"] == 0 and out["n_inverted_at_worst"] == 2
    # the worst-radius label is one radius; a label that holds needs a run of radii
    assert tuned["end_state_by_radius"] == {"0.1": "inverted", "0.5": "disabled", "2.0": "disabled"}
    assert tuned["n_radii_inverted"] == 1 and wide["n_radii_inverted"] == 1
    assert out["n_inverted_over_min_radii"] == 0
    assert out["settings_inverted_at_fewer_radii"] == ["0.03", "1.0"]
    assert read_sweep(block, "per_width", 2.0, 1)["n_inverted_over_min_radii"] == 2
    with pytest.raises(ValueError, match="not a stored radius"):
        read_sweep(block, "per_width", 1.0, 2)


def test_whole_file_branch():
    from experiments.read_worst_radius import whole_file_branch

    assert whole_file_branch([True, True]) == "REPLICATES UNDER R"
    assert whole_file_branch([True, False]) == "MIXED"
    assert whole_file_branch([False, False]) == "DOES NOT"


def _whole_file_runs(tmp_path, monkeypatch, rbf_inside):
    import json

    import experiments.read_worst_radius as rwr

    monkeypatch.setattr(rwr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rwr, "RUNS", tmp_path)
    q = {"0.1": [0.03, 0.04], "0.5": [0.08, 0.10]}
    if not isinstance(rbf_inside, list):
        rbf_inside = [rbf_inside, rbf_inside + 0.01]
    r = {"0.1": rbf_inside, "0.5": [0.39, 0.41]}
    part = {
        "partition": 0,
        "held_out_files": [1, 8],
        "floor": 0.5,
        "clean_qsvm": 0.53,
        "clean_rbf": 0.47,
        "curve": {"qsvm": q, "rbf_svm": r},
    }
    first = {"partition": 0, "adv_qsvm_mean": 0.09, "adv_rbf_mean": 0.40}
    for name, payload in (("new", part), ("first", first)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "results.json").write_text(json.dumps({"partitions": [payload]}))
    # stored predictions: every row evaded at every radius, so the per-input reading is 0
    y = [0, 0, 1, 1]
    pd.DataFrame(
        [
            {
                "partition": 0,
                "model": m,
                "condition": c,
                "attack_seed": s,
                "row_id": i,
                "y_true": t,
                "label": 1 - t,
            }
            for m in ("qsvm", "rbf_svm")
            for c in ("pgd@0.1", "pgd@0.5")
            for s in (0, 1)
            for i, t in enumerate(y)
        ]
    ).to_parquet(tmp_path / "new" / "predictions.parquet")
    return {
        "runs": ["new"],
        "anchor_run": "first",
        "anchor_tol": 0.004,
        "gap_min": 0.05,
        "seeds_min": 2,
    }


def test_read_whole_file_passes_on_a_gap_at_the_worst_radii(tmp_path, monkeypatch):
    from experiments.read_worst_radius import read_whole_file

    spec = _whole_file_runs(tmp_path, monkeypatch, rbf_inside=0.14)
    anchors: list = []
    out = read_whole_file(spec, 0.5, {}, anchors)
    p0 = out["partitions"]["0"]
    assert p0["models"]["rbf_svm"]["worst"]["radius"] == 0.1
    assert p0["gap_rbf_minus_qsvm_worst"] == pytest.approx(0.145 - 0.035)
    assert p0["gap_rbf_minus_qsvm_at_budget"] == pytest.approx(0.40 - 0.09)
    assert p0["models"]["qsvm"]["worst"]["f1_minus_stratified_floor"] == pytest.approx(-0.465)
    assert p0["seeds_qsvm_below_rbf_worst"] == 2 and p0["passes"]
    assert p0["seeds_gap_at_least_gap_min"] == 2
    assert p0["smallest_per_seed_gap"] == pytest.approx(0.11)
    assert p0["models"]["qsvm"]["per_input_union"]["mean"] == 0.0
    assert out["branch"] == "REPLICATES UNDER R" and all(a["ok"] for a in anchors)


def test_read_whole_file_fails_the_bar_when_the_gap_closes_inside_the_ball(tmp_path, monkeypatch):
    import json

    from experiments.read_worst_radius import read_whole_file

    spec = _whole_file_runs(tmp_path, monkeypatch, rbf_inside=0.05)
    out = read_whole_file(spec, 0.5, {}, [])
    assert not out["partitions"]["0"]["passes"] and out["branch"] == "DOES NOT"
    # a budget reading that does not reproduce the first run is a defect, not a result
    first = {"partition": 0, "adv_qsvm_mean": 0.10, "adv_rbf_mean": 0.40}
    (tmp_path / "first" / "results.json").write_text(json.dumps({"partitions": [first]}))
    with pytest.raises(ValueError, match="anchor failed"):
        read_whole_file(spec, 0.5, {}, [])


def test_read_whole_file_counts_seeds_that_clear_the_gap_not_seeds_merely_lower(
    tmp_path, monkeypatch
):
    from experiments.read_worst_radius import read_whole_file

    # mean gap 0.095 clears 0.05 and the QSVM is lower in both seeds, but one seed's gap is 0.02
    spec = _whole_file_runs(tmp_path, monkeypatch, rbf_inside=[0.20, 0.06])
    p0 = read_whole_file(spec, 0.5, {}, [])["partitions"]["0"]
    assert p0["gap_rbf_minus_qsvm_worst"] == pytest.approx(0.095)
    assert p0["seeds_qsvm_below_rbf_worst"] == 2 and p0["seeds_gap_at_least_gap_min"] == 1
    assert not p0["passes"]


def test_read_whole_file_fails_on_predictions_that_are_not_the_runs(tmp_path, monkeypatch):
    from experiments.read_worst_radius import read_whole_file

    spec = _whole_file_runs(tmp_path, monkeypatch, rbf_inside=0.14)
    p = pd.read_parquet(tmp_path / "new" / "predictions.parquet")
    p["label"] = p["y_true"]  # nothing evaded: a per-input reading above the worst radius
    p.to_parquet(tmp_path / "new" / "predictions.parquet")
    with pytest.raises(ValueError, match="above the worst radius"):
        read_whole_file(spec, 0.5, {}, [])


def test_per_input_union_counts_a_row_evaded_at_any_radius():
    from sklearn.metrics import f1_score

    from experiments.read_worst_radius import per_input_union

    y = [0, 0, 1, 1]
    wrong = {"pgd@0.1": [1, 0, 0, 0], "pgd@0.5": [0, 0, 1, 0], "pgd@2.0": [0, 1, 0, 0]}
    rows = [
        {
            "model": "qsvm",
            "condition": c,
            "attack_seed": 0,
            "row_id": i,
            "y_true": t,
            "label": 1 - t if w else t,
        }
        for c, ws in wrong.items()
        for i, (t, w) in enumerate(zip(y, ws, strict=True))
    ]
    rows.append({**rows[0], "condition": "clean", "label": 0})
    out = per_input_union(pd.DataFrame(rows), "qsvm", 0.5)
    # rows 0 and 2 are evaded at one of the two radii inside the budget; 2 sigma is outside it
    assert out["radii"] == [0.1, 0.5]
    assert out["mean"] == pytest.approx(f1_score(y, [1, 0, 0, 1], average="macro"))
    with pytest.raises(ValueError, match="no attacked predictions"):
        per_input_union(pd.DataFrame(rows), "rbf_svm", 0.5)


def test_read_model_fails_on_a_clean_score_of_zero():
    with pytest.raises(ValueError, match="must be positive"):
        read_model(0.0, DIP, 0.5, None)


def _source_runs(tmp_path, monkeypatch, eps_half=0.5, stored_qsvm=None):
    """A kernel run, its label file, a verdict run and a shared-protocol run for one source."""
    import json

    from sklearn.metrics import f1_score

    import experiments.read_worst_radius as rwr

    monkeypatch.setattr(rwr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rwr, "RUNS", tmp_path)
    y = np.array([1] * 7 + [0] * 3)
    pred = np.array([1] * 6 + [0] + [0] * 2 + [1])
    clean = float(f1_score(y, pred, average="macro"))
    labels = _labels(tmp_path, y, pred)
    q = {"0.2": [0.10, 0.12], "0.5": [0.20, 0.22]}  # dips inside the ball
    r = {"0.2": [0.30, 0.32], "0.5": [0.30, 0.32]}
    kernel = {
        "p5a_whitebox_qsvm": {
            "eps_half": eps_half,
            "n_eval": 10,
            "clean_f1": clean,
            "per_seed_pgd": q,
            "whitebox_retention_at_half": 0.21 / clean if stored_qsvm is None else stored_qsvm,
            "rbf_control": {
                "clean_f1": 0.6,
                "per_seed_pgd": r,
                "whitebox_retention_at_half": 0.31 / 0.6,
            },
        }
    }
    transfer_curve = {"0.2": 0.45, "0.5": 0.54}  # the transferred attack also dips inside
    shared = {
        "per_seed": [
            {
                "evasion": {
                    "clean": {"qsvm": 0.6},
                    "curves": {"qsvm": {"pgd": {k: v + d for k, v in transfer_curve.items()}}},
                }
            }
            for d in (-0.01, 0.01)
        ]
    }
    verdict = {
        "availability": {"A": 0.9 - 0.21 / clean, "qsvm_transfer_retention": 0.9},
        "fragility": {"F": 0.21 / clean - 0.31 / 0.6},
    }
    for name, payload in (("kernel", kernel), ("shared", shared), ("verdict", verdict)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "results.json").write_text(json.dumps(payload))
    spec = {
        "kernel_run": "kernel",
        "labels_file": labels.name,
        "verdict_run": "verdict",
        "shared_run": "shared",
        "registered": {"rule": "grid", "a_min": 0.20, "f_max": -0.10},
    }
    cfg = {"budget": 0.5, "reading_radius": {"yes_at": 0.20, "partial_at": 0.05}}
    return spec, cfg, clean


def test_read_source_reads_both_radii_and_the_transfer_at_its_own_worst_radius(
    tmp_path, monkeypatch
):
    from experiments.read_worst_radius import read_source

    spec, cfg, clean = _source_runs(tmp_path, monkeypatch)
    anchors: list = []
    out = read_source("toy", spec, cfg, {}, anchors)
    assert out["models"]["qsvm"]["worst"]["radius"] == 0.2
    assert out["reading_radius"]["D_qsvm"] == pytest.approx(0.10 / clean)
    assert out["reading_radius"]["D_rbf"] == 0.0 and out["reading_radius"]["reading"] == "no"
    av = out["availability"]
    assert av["A_at_budget"] == pytest.approx(0.9 - 0.21 / clean)
    assert av["A_worst"] == pytest.approx(0.9 - 0.11 / clean)
    both = av["transfer_also_at_its_worst_radius"]
    assert both["qsvm_transfer_worst_radius"] == 0.2
    assert both["A"] == pytest.approx(0.45 / 0.6 - 0.11 / clean)
    assert len(anchors) == 5 and all(a["ok"] for a in anchors)


def test_read_source_fails_when_the_kernel_run_read_another_radius(tmp_path, monkeypatch):
    from experiments.read_worst_radius import read_source

    spec, cfg, _ = _source_runs(tmp_path, monkeypatch, eps_half=0.3)
    with pytest.raises(ValueError, match="the budget is 0.5"):
        read_source("toy", spec, cfg, {}, [])


def test_read_source_fails_when_the_stored_retention_is_not_reproduced(tmp_path, monkeypatch):
    from experiments.read_worst_radius import read_source

    spec, cfg, _ = _source_runs(tmp_path, monkeypatch, stored_qsvm=0.9)
    with pytest.raises(ValueError, match="anchor failed: toy QSVM white-box retention"):
        read_source("toy", spec, cfg, {}, [])
