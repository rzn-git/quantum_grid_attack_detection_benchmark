"""assemble_best_params maps each study's best trial onto the robustness runners' keys."""

import importlib.util

import optuna
import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

optuna.logging.set_verbosity(optuna.logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "assemble_best_params", REPO_ROOT / "experiments" / "assemble_best_params.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _study(root, db_stem, name, params_seq, values):
    storage = f"sqlite:///{(root / 'results' / f'optuna_{db_stem}.db').as_posix()}"
    st = optuna.create_study(direction="maximize", study_name=name, storage=storage)
    choices = {k: sorted({p[k] for p in params_seq}) for k in params_seq[0]}
    dists = {k: optuna.distributions.CategoricalDistribution(c) for k, c in choices.items()}
    for p, v in zip(params_seq, values):
        st.add_trial(optuna.trial.create_trial(params=p, distributions=dists, value=v))


@pytest.fixture
def studies(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    for m in mod.MODEL_NAMES:
        n = f"cls_{m}_toy_pca8"
        _study(tmp_path, n, n, [{"a": 1}, {"a": 2}], [0.1, 0.2])
    _study(tmp_path, "qk_zz_toy_q8", "qk_zz_toy_q8", [{"bandwidth": 0.5, "C": 3.0}] * 2, [0.3, 0.1])
    _study(
        tmp_path,
        "vqc_toy_q8",
        "vqc_toy_q8_q8",
        [{"lr": 0.01, "depth": 2, "batch_size": 32}, {"lr": 0.02, "depth": 4, "batch_size": 64}],
        [0.1, 0.4],
    )
    return tmp_path


def test_maps_best_trials_to_runner_keys(studies):
    out, prov = mod.assemble("toy", 8, "zz", 40, n_trials=2)
    assert out["xgb"] == {"a": 2} and out["mlp"] == {"a": 2}
    assert (out["qk_bw"], out["qk_C"]) == (0.5, 3.0)
    assert (out["vqc_depth"], out["vqc_lr"], out["vqc_batch"]) == (4, 0.02, 64)
    assert out["vqc_epochs"] == 40
    assert prov["vqc"]["study"] == "vqc_toy_q8_q8"


def test_short_study_fails_loud(studies):
    with pytest.raises(RuntimeError, match="budget is 3"):
        mod.assemble("toy", 8, "zz", 40, n_trials=3)


def test_diff_names_every_differing_key():
    lines = mod.diff({"a": 1, "m": {"x": 1}}, {"a": 2, "m": {"x": 1}, "b": 0})
    assert lines == ["a: assembled 1 != file 2", "b: assembled '<absent>' != file 0"]


def test_out_writes_the_named_file_and_the_default_path_is_unchanged(studies, monkeypatch):
    argv = ["assemble_best_params", "--variant", "toy", "--vqc-epochs", "40", "--n-trials", "2"]
    monkeypatch.setattr(mod.sys, "argv", [*argv, "--out", "results/alt.json"])
    mod.main()
    assert (studies / "results" / "alt.json").exists()
    assert (studies / "results" / "alt_provenance.json").exists()
    assert not (studies / "results" / "adv_best_params_toy_q8.json").exists()
    monkeypatch.setattr(mod.sys, "argv", argv)
    mod.main()
    default = studies / "results" / "adv_best_params_toy_q8.json"
    assert default.read_text() == (studies / "results" / "alt.json").read_text()


def test_study_suffix_reads_the_fresh_search_only(studies):
    """A re-search after a regime change (decision log 2026-11-18, R4-22 incident) runs under
    suffixed names: assembly with the suffix reads those studies, never the old ones."""
    for m in mod.MODEL_NAMES:
        n = f"cls_{m}_toy_pca8_eigh"
        _study(studies, n, n, [{"a": 7}, {"a": 9}], [0.5, 0.1])
    _study(
        studies,
        "qk_zz_toy_q8_eigh",
        "qk_zz_toy_q8_eigh",
        [{"bandwidth": 0.9, "C": 1.0}] * 2,
        [0.3, 0.1],
    )
    _study(
        studies,
        "vqc_toy_q8_eigh",
        "vqc_toy_q8_eigh_q8",
        [{"lr": 0.05, "depth": 6, "batch_size": 16}, {"lr": 0.02, "depth": 4, "batch_size": 64}],
        [0.9, 0.4],
    )
    out, prov = mod.assemble("toy", 8, "zz", 40, n_trials=2, suffix="eigh")
    assert out["xgb"] == {"a": 7} and out["qk_bw"] == 0.9 and out["vqc_depth"] == 6
    assert (
        prov["xgb"]["study"] == "cls_xgb_toy_pca8_eigh"
        and prov["vqc"]["study"] == "vqc_toy_q8_eigh_q8"
    )
    assert mod.assemble("toy", 8, "zz", 40, n_trials=2)[0]["xgb"] == {"a": 2}
