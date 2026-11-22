"""R4-22 H4 runner: the branch rule, episode ids and the same-episode nearest-neighbour share."""

import importlib.util
import sys

import numpy as np

from qgridbench.utils.run_tracking import REPO_ROOT

if str(REPO_ROOT) not in sys.path:  # the runner imports experiments.run_group_split_control
    sys.path.insert(0, str(REPO_ROOT))
_argv = sys.argv
sys.argv = ["run_fdia_split_gap", "--threads", "1"]  # the runner pins threads at import
try:
    _spec = importlib.util.spec_from_file_location(
        "run_fdia_split_gap", REPO_ROOT / "experiments" / "run_fdia_split_gap.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
finally:
    sys.argv = _argv


def test_branch_rule():
    assert _mod.branch([0.05, 0.31]) == "HYPOTHESIS"
    assert _mod.branch([0.0, 0.019]) == "KILL"
    assert _mod.branch([0.06, 0.01]) == "INTERMEDIATE"
    assert _mod.branch([0.03, 0.04]) == "INTERMEDIATE"


def test_episode_ids_unique_per_day():
    aux = {"day": np.array([3, 3, 4, 4]), "episode": np.array([0, 1, 0, 1])}
    assert _mod.episode_ids(aux, 12).tolist() == [36, 37, 48, 49]


def test_same_episode_share_known_answer():
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 10, size=(6, 3))
    ep = np.repeat(np.arange(6), 10)
    X = centres[ep] + rng.normal(0, 0.01, size=(60, 3))
    idx = np.arange(60)
    test = idx[idx % 10 == 0]
    train = np.setdiff1d(idx, test)
    splits = {"train": train, "val": train[:5], "test": test}
    assert _mod.same_episode_share(splits, X, ep) == 1.0
    held = {"train": idx[ep < 4], "val": idx[ep == 4], "test": idx[ep == 5]}
    assert _mod.same_episode_share(held, X, ep) == 0.0


def test_refit_probabilities_match_table_i_functions():
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 90)
    X = rng.normal(size=(180, 12)) + y[:, None] * 0.6
    idx = rng.permutation(180)
    sp = {"train": np.sort(idx[:100]), "val": np.sort(idx[100:140]), "test": np.sort(idx[140:])}
    lgbm = {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 5,
            "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0}  # fmt: skip
    best = {"xgb": {"n_estimators": 30, "learning_rate": 0.1, "max_depth": 3, "subsample": 1.0,
                    "colsample_bytree": 1.0, "min_child_weight": 1, "reg_lambda": 1.0},
            "qk_bw": 0.5, "qk_C": 1.0}  # fmt: skip
    row = {
        "level_full_lgbm": _mod.score_level(lgbm, sp, X, y),
        **{f"matched_{k}": v for k, v in _mod.score_matched(best, sp, X, y).items()},
    }
    probs = _mod.arm_probabilities(lgbm, best, sp, X, y)
    assert set(probs) == {"level_full_lgbm", "matched_classical", "matched_quantum"}
    assert all(d == 0.0 for d in _mod.check_against_table_i(row, probs, y[sp["test"]]).values())
    bad = dict(row, matched_quantum=row["matched_quantum"] + 0.01)
    try:
        _mod.check_against_table_i(bad, probs, y[sp["test"]])
    except AssertionError:
        pass
    else:
        raise AssertionError("a changed F1 must raise")
