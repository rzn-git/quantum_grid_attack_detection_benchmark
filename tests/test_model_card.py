"""The model card's robustness and black-box blocks quote the paper's robustness table.

Regression for the 2026-11-20 mismatch: the robustness block recomputed its rows from the newest
adversarial run and so printed an earlier VQC (0.502, 0.036, 0.072) than the paper's table
(0.476, 0.029, 0.060), and the restart figures were typed into the card's prose.
"""

import pytest

from qgridbench.eval import model_card

ROBUSTNESS_CSV = (
    "model,clean_f1,adv_f1,adv_f1_minus_floor,retention,whitebox_retention,random_flip_rate,"
    "hsj_median_linf_sigma,hsj_restart_std,hsj_found_share,hsj_n_restarts,"
    "whitebox_radius_sigma,whitebox_retention_at_budget,whitebox_single_radius\n"
    "qsvm,0.5631,0.4991,-0.0012,0.886296,0.022810,0.4467,3.4134,1.3861,,10,0.3,0.064082,0\n"
    "rbf_svm,0.5871,0.4148,-0.0855,0.706506,0.243400,0.2400,0.0134,0.0013,0.2400,5,0.1,"
    "0.684769,0\n"
    "rf,0.6160,0.3673,-0.1329,0.596275,,0.1833,0.0623,0.0420,0.2760,5,,,\n"
    "mlp_weighted,0.5211,0.2594,-0.2409,0.497740,0.076374,,,,,,0.5,0.076374,1\n"
    "vqc,0.4756,0.0287,-0.4715,0.060412,0.060412,0.2467,0.4809,0.0671,,5,0.5,0.060412,0\n"
)
BLACKBOX_CSV = (
    "model,clean_f1,adv_f1,median_linf_sigma,queries_per_sample\n"
    "qsvm,0.5559,0.3790,2.8185,624\n"
    "vqc,0.4658,0.4400,0.4519,590\n"
    "rbf_svm,0.5764,0.4152,0.0000,216\n"
)
# what the run behind the table would give if the block recomputed its rows: an earlier VQC
STALE_STATS = {
    "epsilon": 0.5,
    "n_seeds": 10,
    "n_significant_holm": 12,
    "n_comparisons": 12,
    "per_family": {"vqc": {"clean_mean": 0.502, "adv_mean": 0.036, "retention": 0.072}},
}


@pytest.fixture
def tables(tmp_path, monkeypatch):
    (tmp_path / "robustness_binary_q8.csv").write_text(ROBUSTNESS_CSV, encoding="utf-8")
    (tmp_path / "blackbox_binary_q8.csv").write_text(BLACKBOX_CSV, encoding="utf-8")
    monkeypatch.setattr(model_card, "TABLE_DIR", tmp_path)
    monkeypatch.setattr(model_card, "robustness_stats", lambda run: STALE_STATS)
    return tmp_path


def test_robustness_rows_are_the_paper_table(tables):
    block = model_card._robustness_block(tables)
    assert "| vqc | 0.476 | 0.029 | 0.060 | 0.060 | 25% |" in block
    assert "0.502" not in block and "0.072" not in block
    assert "| mlp (class-weighted) | 0.521 | 0.259 | 0.498 | 0.076 at 0.5σ only | — |" in block
    assert "| rf | 0.616 | 0.367 | 0.596 | — | 18% |" in block
    assert "At ε = 0.5σ, 10 seeds." in block and "**12 of 12**" in block


def test_robustness_block_needs_the_run_behind_the_table(tables):
    with pytest.raises(ValueError, match="robustness_run"):
        model_card._robustness_block(None)


def test_blackbox_restart_column_is_the_paper_table(tables):
    block = model_card._blackbox_block()
    assert "| qsvm | 0.556 | 0.379 | 2.818 | 624 | 3.413 ± 1.386 (n = 10) |" in block
    assert "| vqc | 0.466 | 0.440 | 0.452 | 590 | 0.481 ± 0.067 (n = 5) |" in block
    assert (
        "| rbf_svm | 0.576 | 0.415 | 0.000 † | 216 | 0.013 ± 0.001 on 24% of points (n = 5) |"
        in block
    )


def test_blackbox_model_missing_from_the_robustness_table_fails(tables):
    kept = [ln for ln in ROBUSTNESS_CSV.splitlines() if not ln.startswith("vqc,")]
    (tables / "robustness_binary_q8.csv").write_text("\n".join(kept) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'vqc' has no row"):
        model_card._blackbox_block()


def test_blackbox_without_a_robustness_table_drops_the_restart_column(tables):
    (tables / "robustness_binary_q8.csv").unlink()
    block = model_card._blackbox_block()
    assert "over restarts" not in block.split("\n")[0]
    assert "| qsvm | 0.556 | 0.379 | 2.818 | 624 |" in block
    assert "restart column is not shown" in block


def test_whitebox_cell_prints_the_radius_when_it_is_inside_the_ball():
    # read at the worst radius inside the budget: the radius is printed
    row = {
        "whitebox_retention": "0.022810",
        "whitebox_radius_sigma": "0.3",
        "whitebox_single_radius": "0",
    }
    assert model_card._whitebox_cell(row, 0.5) == "0.023 at 0.3σ"
    # read at the budget itself: the figure only
    assert model_card._whitebox_cell({**row, "whitebox_radius_sigma": "0.5"}, 0.5) == "0.023"
    # an attack that ran one radius only says so: its cell is an upper bound
    single = {**row, "whitebox_radius_sigma": "0.5", "whitebox_single_radius": "1"}
    assert model_card._whitebox_cell(single, 0.5) == "0.023 at 0.5σ only"
    assert model_card._whitebox_cell({"whitebox_retention": ""}, 0.5) == "—"


def test_whitebox_cell_fails_on_a_retention_without_its_radius():
    # a table that predates the radius columns must be rebuilt, never printed as a bare figure
    with pytest.raises(ValueError, match="without the radius"):
        model_card._whitebox_cell({"model": "qsvm", "whitebox_retention": "0.076374"}, 0.5)
    with pytest.raises(ValueError, match="without the radius"):
        model_card._whitebox_cell(
            {"model": "qsvm", "whitebox_retention": "0.07", "whitebox_radius_sigma": "0.5"}, 0.5
        )
