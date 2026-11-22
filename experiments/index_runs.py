"""Run index: one row per run folder under results/runs/, an MLflow-style runs table kept as a file.

    uv run python -m experiments.index_runs

Writes results/runs_index.parquet and results/runs_index.csv. Both are regenerated from the run
folders and from the curated labels in experiments/run_labels.csv, and never edited by hand.
Nothing is deleted or moved (operator rule 2026-11-17: experiments are never thrown away). A label
row sets a run's experiment, role, status, reason and the run it supersedes; a run without a
label gets its role and status inferred from its folder name and its metrics.json.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd

from qgridbench.utils.run_tracking import REPO_ROOT

RUNS = REPO_ROOT / "results" / "runs"
LABELS = REPO_ROOT / "experiments" / "run_labels.csv"
OUT = REPO_ROOT / "results" / "runs_index"
LABEL_FIELDS = ("experiment", "run_role", "status", "reason", "supersedes")
STAMP = re.compile(r"^(\d{8}T\d{6}Z)_(.+)$")
ROLE_MARKS = (("smoke", "smoke"), ("anchor", "anchor"), ("probe", "probe"), ("merged_", "merged"))
EXPORTS = ("per_seed.csv", "predictions.parquet", "crafted_inputs.npz")


def read_labels(path: Path = LABELS) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    labels = {}
    for r in rows:
        rid = r["run_id"].strip()
        if rid in labels:
            raise ValueError(f"run_labels.csv lists {rid} twice")
        labels[rid] = {k: (r.get(k) or "").strip() for k in LABEL_FIELDS}
    return labels


def infer_role(name: str) -> str:
    for mark, role in ROLE_MARKS:
        if mark in name:
            return role
    return "full"


def index_row(run: Path, label: dict | None) -> dict:
    m = STAMP.match(run.name)
    stamp, name = (m.group(1), m.group(2)) if m else ("", run.name)
    metrics = {}
    if (run / "metrics.json").exists():
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    sha = (run / "git_sha.txt").read_text().strip() if (run / "git_sha.txt").exists() else ""
    if metrics.get("status"):
        status = metrics["status"]
    elif (run / "results.json").exists():
        status = "ok_no_metrics"
    else:
        status = "incomplete"
    label = label or {}
    headline = metrics.get("metrics")
    return {
        "run_id": run.name,
        "stamp": stamp,
        "name": name,
        "experiment": label.get("experiment") or name,
        "run_role": label.get("run_role") or infer_role(run.name),
        "status": label.get("status") or status,
        "reason": label.get("reason", ""),
        "supersedes": label.get("supersedes", ""),
        "labelled": bool(label),
        "git_sha": sha.removesuffix("-dirty"),
        "dirty": sha.endswith("-dirty"),
        "wall_time_s": metrics.get("wall_time_s"),
        "has_results": (run / "results.json").exists(),
        **{f"has_{e.split('.')[0]}": (run / e).exists() for e in EXPORTS},
        "headline": json.dumps(headline, sort_keys=True)[:500] if headline else "",
        "path": (run.relative_to(REPO_ROOT) if run.is_relative_to(REPO_ROOT) else run).as_posix(),
    }


def build(runs: Path = RUNS, labels: dict[str, dict] | None = None) -> pd.DataFrame:
    labels = read_labels() if labels is None else labels
    folders = sorted(p for p in runs.iterdir() if p.is_dir())
    unknown = sorted(set(labels) - {p.name for p in folders})
    if unknown:
        raise ValueError(f"run_labels.csv names runs that do not exist: {unknown}")
    return pd.DataFrame([index_row(p, labels.get(p.name)) for p in folders])


def main() -> None:
    df = build()
    df.to_parquet(OUT.with_suffix(".parquet"), index=False)
    df.to_csv(OUT.with_suffix(".csv"), index=False)
    print(f"{len(df)} runs -> {OUT.with_suffix('.parquet').relative_to(REPO_ROOT)}")
    print(df.groupby(["run_role", "status"]).size().to_string())


if __name__ == "__main__":
    main()
