"""experiments/index_runs.py: one row per run folder; labels override inference; read-only."""

import json
import sys

import pytest

from qgridbench.utils.run_tracking import REPO_ROOT

if str(REPO_ROOT) not in sys.path:  # experiments/ is a folder of runners, not an installed package
    sys.path.insert(0, str(REPO_ROOT))
from experiments import index_runs as ix  # noqa: E402


def _run(root, name, metrics=None, files=()):
    d = root / name
    d.mkdir()
    if metrics is not None:
        (d / "metrics.json").write_text(json.dumps(metrics))
    for f in files:
        (d / f).write_text("x")
    return d


def test_inference_and_labels(tmp_path):
    _run(
        tmp_path,
        "20261117T010101Z_classical_x",
        {"status": "ok", "metrics": {"a": 1}},
        ["results.json", "per_seed.csv"],
    )
    _run(tmp_path, "20261117T020202Z_thing_smoke")
    _run(tmp_path, "20261117T030303Z_anchor_triple", {"status": "error"})
    labels = {
        "20261117T030303Z_anchor_triple": {
            "experiment": "R4-20",
            "run_role": "anchor",
            "status": "void",
            "reason": "killed",
            "supersedes": "",
        }
    }
    before = sorted(p.name for p in tmp_path.rglob("*"))
    df = ix.build(tmp_path, labels).set_index("run_id")
    assert sorted(p.name for p in tmp_path.rglob("*")) == before  # read-only
    a = df.loc["20261117T010101Z_classical_x"]
    assert (a.run_role, a.status, a.has_per_seed, a.has_predictions) == ("full", "ok", True, False)
    assert df.loc["20261117T020202Z_thing_smoke"][["run_role", "status"]].tolist() == [
        "smoke",
        "incomplete",
    ]
    c = df.loc["20261117T030303Z_anchor_triple"]
    assert (c.experiment, c.status, c.reason, c.labelled) == ("R4-20", "void", "killed", True)


def test_label_for_missing_run_fails_loud(tmp_path):
    _run(tmp_path, "20261117T010101Z_x", {"status": "ok"})
    with pytest.raises(ValueError, match="do not exist"):
        ix.build(tmp_path, {"20990101T000000Z_gone": dict.fromkeys(ix.LABEL_FIELDS, "")})
