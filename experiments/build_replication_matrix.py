"""Build two draft-v3 exhibit tables from the run artifacts that hold each statistic: the
replication matrix (Fig. C; which MSU/ORNL findings reproduce on the gas-pipeline data and the two
synthetic FDIA grids) and the instance panels (Fig. A; each evaluation choice, two readings).

The cells, their source runs and keys, and each cell's reading (origin / yes / partial / no /
not_tested) are curated in `experiments/replication_matrix_cells.csv`; the readings come from the
registered verdicts in the decision log (R4-12 report 2026-11-18T16:18Z, R4-22 report
2026-11-19T04:05Z). This script reads every value from its artifact, fails loud on a missing run,
file or key, and writes `paper/tables/replication_matrix.csv` with a provenance file beside it.

Key syntax: a JSON file takes a "/"-separated path (list indices as integers); a CSV file takes
"col=val;col=val>target" ("|" inside a value stands for a comma).

Run:  uv run python -m experiments.build_replication_matrix
"""

from __future__ import annotations

import csv
import hashlib
import json
import math

from qgridbench.utils.run_tracking import REPO_ROOT, git_describe, utc_stamp

CELLS = REPO_ROOT / "experiments" / "replication_matrix_cells.csv"
OUT = REPO_ROOT / "paper" / "tables" / "replication_matrix.csv"
READINGS = {"origin", "yes", "partial", "no", "not_tested"}
PANELS = REPO_ROOT / "experiments" / "instance_panels.csv"
PANELS_OUT = REPO_ROOT / "paper" / "tables" / "instance_panels.csv"


def curated_rows(path) -> list[dict]:
    """The rows of a curated CSV; a row with more or fewer fields than the header is an error
    (csv.DictReader would file the extra field of an unquoted comma under None and go on)."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for n, r in enumerate(rows, start=2):
        if None in r or None in r.values():
            raise ValueError(f"{path}: line {n} does not have the header's number of fields")
    return rows


def _json_value(path, key: str):
    v = json.loads(path.read_text())
    for part in key.split("/"):
        if isinstance(v, list):
            v = v[int(part)]
        elif part in v:
            v = v[part]
        else:
            raise KeyError(f"{path}: no key {part!r} on the path {key!r}")
    return v


def _csv_value(path, key: str):
    filt, target = key.split(">")
    conds = [c.split("=", 1) for c in filt.split(";")]
    with path.open(encoding="utf-8") as f:
        hits = [
            r for r in csv.DictReader(f) if all(r[c] == val.replace("|", ",") for c, val in conds)
        ]
    if len(hits) < 1:
        raise KeyError(f"{path}: no row matches {filt!r}")
    values = {h[target] for h in hits}
    if len(values) != 1:
        raise ValueError(f"{path}: rows matching {filt!r} disagree on {target}: {values}")
    return float(values.pop())


def resolve(cell: dict):
    path = REPO_ROOT / "results" / "runs" / cell["run"] / cell["file"]
    if not path.exists():
        raise FileNotFoundError(path)
    if cell["file"].endswith(".csv"):
        if " over " in cell["key"]:  # the ratio of two rows' values
            a, b = cell["key"].split(" over ")
            return _csv_value(path, a) / _csv_value(path, b), path
        return _csv_value(path, cell["key"]), path
    for op, sign in ((" - ", -1.0), (" + ", 1.0)):  # two values in the same file
        if op in cell["key"]:
            a, b = cell["key"].split(op)
            return float(_json_value(path, a)) + sign * float(_json_value(path, b)), path
    if " over " in cell["key"]:  # their ratio
        a, b = cell["key"].split(" over ")
        return float(_json_value(path, a)) / float(_json_value(path, b)), path
    return _json_value(path, cell["key"]), path


def label(value, fmt: str) -> str:
    """The cell's printed number: empty when the cell prints its reading only (no fmt), the
    value with the sign turned for "neg:" (so every accuracy cell reads classical minus quantum),
    or for "fold:" how many times one reading is the other, whichever is the larger (a ratio of
    0.5 and a ratio of 2 both print 2.00×)."""
    if not fmt or isinstance(value, bool) or isinstance(value, str):
        return ""
    if fmt.startswith("neg:"):
        return format(-float(value), fmt[4:])
    if fmt.startswith("fold:"):
        ratio = float(value)
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"a fold change needs a finite positive ratio, got {ratio}")
        return format(max(ratio, 1.0 / ratio), fmt[5:]) + "×"
    return format(float(value), fmt)


PANEL_SIDES = {"a", "b", "ab"}  # the uncontrolled choice, its control, or a mark on both
PANEL_PARTS = {"value", "lo", "hi", "mean", "std", "mark"}


def build_panels() -> dict:
    """The checklist figure's table: each evaluation choice as the reading under the uncontrolled
    choice (side a) and under its control (side b) at fixed models, every value read from its
    artifact (experiments/instance_panels.csv names run, file and key). A reading is one value,
    a range (lo, hi) or a mean with its std; a mark is a reference level drawn on both readings."""
    panels = curated_rows(PANELS)
    sources = {}
    for r in panels:
        if r["side"] not in PANEL_SIDES or r["part"] not in PANEL_PARTS:
            raise ValueError(
                f"{PANELS}: order {r['order']}: side {r['side']!r}, part {r['part']!r}"
            )
        value, path = resolve(r)
        r["value"] = float(value)
        r["value_label"] = label(value, r["fmt"])
        sources[path.relative_to(REPO_ROOT).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    with PANELS_OUT.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "order",
            "panel",
            "unit",
            "side",
            "part",
            "label",
            "value",
            "value_label",
            "scale",
            "suffix",
            "run",
            "file",
            "key",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in panels:
            w.writerow({k: r[k] for k in fields})
    return sources


def main() -> None:
    cells = curated_rows(CELLS)
    rows, sources = [], {}
    for c in cells:
        if c["reproduces"] not in READINGS:
            raise ValueError(f"unknown reading {c['reproduces']!r} for {c['claim']}/{c['dataset']}")
        value, text = None, ""
        if c["run"]:
            value, path = resolve(c)
            text = label(value, c["fmt"])
            rel = path.relative_to(REPO_ROOT).as_posix()
            sources[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif c["reproduces"] != "not_tested":
            raise ValueError(f"{c['claim']}/{c['dataset']}: a tested cell needs a source run")
        rows.append({**c, "value": value, "label": text})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "claim",
                "dataset",
                "reproduces",
                "label",
                "value",
                "run",
                "file",
                "key",
                "note",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    sources.update(build_panels())
    OUT.with_name("replication_matrix_meta.json").write_text(
        json.dumps(
            {
                "written_at": utc_stamp(),
                "git": git_describe(),
                "cells": str(CELLS.name),
                "sources_sha256": sources,
            },
            indent=2,
        )
    )
    print(f"-> {OUT} and {PANELS_OUT} ({len(rows)} cells, {len(sources)} source files)")


if __name__ == "__main__":
    main()
