"""A run from a dirty tree records which paths differed from its sha
(src/qgridbench/utils/run_tracking.py; review of ruling (h), finding m11)."""

from qgridbench.utils import run_tracking


def test_a_dirty_tree_writes_its_paths_and_a_clean_tree_writes_none(tmp_path, monkeypatch):
    monkeypatch.setattr(run_tracking, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_tracking, "git_describe", lambda: "abc1234-dirty")
    monkeypatch.setattr(run_tracking, "git_dirty_paths", lambda: " M paper/main.tex\n?? notes\n")
    with run_tracking.RunDir("dirty", config={}) as run:
        pass
    assert (run.path / "git_dirty_paths.txt").read_text() == " M paper/main.tex\n?? notes\n"
    assert (run.path / "git_sha.txt").read_text() == "abc1234-dirty\n"

    monkeypatch.setattr(run_tracking, "git_describe", lambda: "abc1234")
    monkeypatch.setattr(run_tracking, "git_dirty_paths", lambda: "")
    with run_tracking.RunDir("clean", config={}, seeds=[0, 1]) as run:
        pass
    assert not (run.path / "git_dirty_paths.txt").exists()
    assert "seeds:\n- 0\n- 1" in (run.path / "params.yaml").read_text()
