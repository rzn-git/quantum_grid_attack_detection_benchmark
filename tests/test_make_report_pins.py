"""make_report builds every exhibit that carries a VQC column from the pinned Table I runs.

The kernel-exhaustion table has a VQC column and was built from a `latest()` glob that picks
the depth-4 run (claim-drift review 2026-11-19, finding 2). This reads make_report's source
and checks that each VQC-bearing builder's run argument comes from `vqc_table1(...)`.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "experiments" / "make_report.py"

# builder -> pin key its run argument must resolve to
VQC_BUILDERS = {
    "kernel_exhaustion_table": "p7b8",
    "surrogate_control_table": "surrogate",
    "data_efficiency_curve": "data_efficiency",
    "robustness_curves": "adversarial",
    "transfer_heatmap": "adversarial",
    "robustness_summary_table": "adversarial",
}


def _assignments(tree: ast.AST) -> dict[str, ast.expr]:
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name):
                out[t.id] = node.value
    return out


def _pin_key(expr: ast.expr, names: dict[str, ast.expr]) -> str | None:
    if isinstance(expr, ast.Name) and expr.id in names:
        expr = names[expr.id]
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "vqc_table1"
        and expr.args
        and isinstance(expr.args[0], ast.Constant)
    ):
        return expr.args[0].value
    return None


def test_vqc_bearing_exhibits_read_pinned_table1_runs():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    names = _assignments(tree)
    seen = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in VQC_BUILDERS
        ):
            seen.setdefault(node.func.id, []).append(_pin_key(node.args[0], names))
    assert set(seen) == set(VQC_BUILDERS), f"builders not called: {set(VQC_BUILDERS) - set(seen)}"
    for builder, keys in seen.items():
        assert all(k == VQC_BUILDERS[builder] for k in keys), (builder, keys)


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
    ]


def test_white_box_cells_and_the_split_exhibits_read_pinned_runs():
    """The white-box column reads the worst-radius run by pin, and no `latest()` glob picks the
    group-split run (red-team RT-n8) or the kernel white-box run for the robustness table."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    names = _assignments(tree)
    (table,) = _calls(tree, "robustness_summary_table")
    source = {k.arg: k.value for k in table.keywords}["worst_radius_run_dir"]
    if isinstance(source, ast.Name):
        source = names[source.id]
    assert isinstance(source, ast.Call) and source.func.id == "worst_radius_run"
    globs = [
        c.args[0].value
        for c in _calls(tree, "latest")
        if c.args and isinstance(c.args[0], ast.Constant)
    ]
    assert not [g for g in globs if "group_split_control" in g], globs
    assert len(_calls(tree, "group_split_run")) >= 2


def _pinned_worst_radius_run(tmp_path, monkeypatch, sources_now):
    """A pinned worst-radius run built from {"budget": 0.5}, and the sources file as it is now."""
    import yaml

    import experiments.make_report as mr

    monkeypatch.setattr(mr, "RUNS", tmp_path)
    monkeypatch.setattr(mr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mr, "WORST_RADIUS_RUN", "pinned")
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump(sources_now), encoding="utf-8")
    run = tmp_path / "pinned"
    run.mkdir()
    (run / "params.yaml").write_text(
        yaml.safe_dump({"config": {"sources_file": "sources.yaml", "budget": 0.5}}),
        encoding="utf-8",
    )
    return mr, run


def test_worst_radius_run_fails_when_the_pinned_run_is_missing(tmp_path, monkeypatch):
    import pytest

    mr, _ = _pinned_worst_radius_run(tmp_path, monkeypatch, {"budget": 0.5})
    with pytest.raises(FileNotFoundError, match="pinned worst-radius run missing"):
        mr.worst_radius_run()


def test_worst_radius_run_fails_when_the_sources_file_changed_since_the_pin(tmp_path, monkeypatch):
    import pytest

    mr, run = _pinned_worst_radius_run(tmp_path, monkeypatch, {"budget": 0.5})
    (run / "results.json").write_text("{}", encoding="utf-8")
    assert mr.worst_radius_run() == run
    (tmp_path / "sources.yaml").write_text("budget: 0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since the pinned worst-radius run"):
        mr.worst_radius_run()
