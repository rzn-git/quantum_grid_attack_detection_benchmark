"""Build the two curated tables of checks, every number read from the run artifact that holds
it: the arXiv appendix table "Further controls" (checks the body does not have room for) and the
body table "RQ3 and the other checks" (`experiments/other_checks.yaml`, written to
`paper/tables/other_checks.tex`; its rows carry a `setting` where the appendix rows carry
`bears_on`).

The rows, their wording and the source of each number (run, file, key, format, and the text the
number must print as) are curated in `experiments/further_controls.yaml` and
`experiments/other_checks.yaml`. This script resolves every key, fails loud on a missing run, file
or key, on a value that is not a finite number, on a value that does not print as its row expects,
or on a failed `require`, and writes, per table, the `.tex` file under `paper/tables/`, a CSV of
the resolved values (each with the full spec it was read by) and a provenance file (git state and
the sha256 of every source file). A source file is JSON, or YAML (a run's `params.yaml`).

Run:  uv run python -m experiments.build_further_controls
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import yaml

from experiments.build_replication_matrix import _json_value
from qgridbench.eval.design_tokens import LATEX_COLOR_NAMES
from qgridbench.eval.tables import _count
from qgridbench.utils.run_tracking import REPO_ROOT, git_describe, utc_stamp

SPEC = REPO_ROOT / "experiments" / "further_controls.yaml"
OUT = REPO_ROOT / "paper" / "tables" / "further_controls.tex"
OTHER_SPEC = REPO_ROOT / "experiments" / "other_checks.yaml"
OTHER_OUT = REPO_ROOT / "paper" / "tables" / "other_checks.tex"
RUNS = REPO_ROOT / "results" / "runs"
PLACEHOLDER = re.compile(r"<(\d+)>")
AGGREGATES = {"min": min, "max": max, "mean": statistics.fmean}
ITERATION_FIELDS = ("agg", "where", "where_not", "keys", "pick_by", "limit")
COMBINE = {"minus": lambda a, b: a - b, "over": lambda a, b: a / b}  # with a second value spec
SIGNS = {"negative": -1, "positive": 1}


def _same(a, b) -> bool:
    """Field equality for `where`: numbers by value, everything else as text."""
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def _walk(node, steps: list[str], path: Path):
    for step in steps:
        if isinstance(node, list):
            node = node[int(step)]
        elif step in node:
            node = node[step]
        else:
            raise KeyError(f"{path}: no key {step!r} on the path {'/'.join(steps)!r}")
    return node


def _entries(node, spec: dict, head: str, path: Path) -> list:
    """The entries a `*` step iterates, after `keys` (a dict's named entries only) and the
    `where` / `where_not` filters (lists of dicts)."""
    if "keys" in spec:
        if not isinstance(node, dict):
            raise ValueError(f"{path}: `keys` needs a dict at {head!r}")
        missing = [k for k in spec["keys"] if k not in node]
        if missing:
            raise KeyError(f"{path}: no entries {missing} at {head!r}")
        entries = [node[k] for k in spec["keys"]]
    else:
        entries = list(node.values()) if isinstance(node, dict) else list(node)
    for field, want in spec.get("where", {}).items():
        entries = [e for e in entries if _same(e[field], want)]
    for field, drop in spec.get("where_not", {}).items():
        entries = [e for e in entries if not _same(e[field], drop)]
    if not entries:
        raise KeyError(f"{path}: no entry of {head!r} matches the where clause of {spec}")
    return entries


def spec_paths(spec: dict, runs: Path = RUNS) -> list[Path]:
    """Every file a value spec reads: its own and those of the specs it combines with."""
    own = [runs / spec["run"] / spec["file"]]
    return own + [p for op in COMBINE if op in spec for p in spec_paths(spec[op], runs)]


def resolve_value(spec: dict, runs: Path = RUNS) -> tuple[float, Path]:
    """The value a spec names, and the file it came from. With `minus` or `over` (a second value
    spec, in any run) it is the difference or the ratio of the two."""
    if all(op in spec for op in COMBINE):
        raise ValueError(f"{spec['run']}: a value takes minus or over, not both")
    value, path = _resolve_one(spec, runs)
    for op, combine in COMBINE.items():
        if op in spec:
            other, other_path = resolve_value(spec[op], runs)
            value = combine(
                _number(value, path, spec["key"]), _number(other, other_path, spec[op]["key"])
            )
    return value, path


def _resolve_one(spec: dict, runs: Path) -> tuple[float, Path]:
    path = runs / spec["run"] / spec["file"]
    if not path.exists():
        raise FileNotFoundError(path)
    key = spec["key"]
    if "limit" in spec and spec.get("agg") != "count_at_most":
        raise ValueError(f"{spec['run']}: limit belongs to agg count_at_most")
    if "*" not in key:
        if any(k in spec for k in ITERATION_FIELDS):
            raise ValueError(f"{spec['run']}: agg and where need a '*' step in the key {key!r}")
        if path.suffix in (".yaml", ".yml"):  # a run's params.yaml: one plain key
            steps = [s for s in key.split("/") if s]
            return _walk(yaml.safe_load(path.read_text(encoding="utf-8")), steps, path), path
        if " - " in key:
            a, b = (_number(_json_value(path, part), path, part) for part in key.split(" - "))
            return a - b, path
        return _json_value(path, key), path
    head, tail = key.split("*", 1)
    node = _walk(json.loads(path.read_text()), [s for s in head.split("/") if s], path)
    entries = _entries(node, spec, head, path)
    steps = [s for s in tail.split("/") if s]
    if "pick_by" in spec:
        # the entry a selection rule picks (the highest validation score), read at `tail`
        if "agg" in spec:
            raise ValueError(f"{spec['run']}: {key!r} takes pick_by or agg, not both")
        by = [s for s in spec["pick_by"].split("/") if s]
        scores = [_number(_walk(e, by, path), path, spec["pick_by"]) for e in entries]
        if scores.count(max(scores)) != 1:
            raise ValueError(f"{path}: pick_by {spec['pick_by']!r} ties at its maximum")
        return _walk(entries[scores.index(max(scores))], steps, path), path
    if spec.get("agg") == "one":
        if len(entries) != 1:
            raise ValueError(f"{path}: {key!r} selects {len(entries)} entries, not one")
        return _walk(entries[0], steps, path), path
    # every iterated entry is a finite number: a NaN, a string or a bool fails here
    values = [_number(_walk(e, steps, path), path, key) for e in entries]
    if spec.get("agg") == "count_at_most":
        if "limit" not in spec:
            raise ValueError(f"{spec['run']}: agg count_at_most needs a limit")
        return sum(v <= _number(spec["limit"], path, "limit") for v in values), path
    if spec.get("agg") not in AGGREGATES:
        raise ValueError(
            f"{spec['run']}: the key {key!r} iterates and needs agg: min, max, mean, one or "
            "count_at_most"
        )
    return AGGREGATES[spec["agg"]](values), path


def _number(value, path: Path, key: str) -> float:
    """`value` if it is a finite number; a bool, a string, None, NaN or an infinity fails (a bool
    would otherwise count as 1 and a numeric string would be coerced)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{path}: {key} is {value!r}, not a finite number")
    return value


def shown_value(value, spec: dict, path: Path) -> float:
    """The number a value spec prints: finite and numeric, and with `sign` its magnitude, after
    checking that it has the sign the wording assumes ("loses 0.065")."""
    _number(value, path, spec["key"])
    if "sign" not in spec:
        return value
    if spec["sign"] not in SIGNS:
        raise ValueError(f"{spec['run']}: sign is 'negative' or 'positive', not {spec['sign']!r}")
    if value * SIGNS[spec["sign"]] <= 0:
        raise ValueError(f"{path}: {spec['key']} is {value!r}, not {spec['sign']}")
    return abs(value)


def plain_number(value, fmt: str) -> str:
    """A value as plain text at its format. A value that rounds to zero carries no minus."""
    if fmt == "count":
        if float(value) != int(value):
            raise ValueError(f"a count must be a whole number, got {value!r}")
        return str(int(value))
    if fmt == "count2":  # a large ratio at two significant figures
        if abs(float(value)) < 100:
            raise ValueError(f"count2 is for ratios of 100 or more, got {value!r}")
        return str(int(float(f"{float(value):.2g}")))
    text = format(float(value), fmt)
    return format(0.0, fmt) if float(text.rstrip("%")) == 0 else text


def tex_number(value, fmt: str) -> str:
    """A value as LaTeX: a count with IEEE thousands spacing, a percentage, or a number in math
    mode (so a minus is a true minus and a `+` spec shows its sign)."""
    if fmt == "count":
        return _count(value)
    if fmt == "count2":
        return _count(int(plain_number(value, fmt)))
    text = plain_number(value, fmt)
    if text.endswith("%"):
        return text[:-1] + "\\%"
    return f"${text}$"


def fill(reading: str, numbers: list[str], row_id: str) -> str:
    """The reading with each `<n>` replaced by number n; every number must be used once or more
    and every placeholder must have a number."""
    used = {int(m) for m in PLACEHOLDER.findall(reading)}
    if used != set(range(len(numbers))):
        raise ValueError(f"{row_id}: placeholders {sorted(used)} against {len(numbers)} values")

    def put(match: re.Match) -> str:
        text = numbers[int(match.group(1))]
        # inside an author-written math span the number stands bare (no nested math)
        return text.strip("$") if _in_math(reading, match.start()) else text

    return PLACEHOLDER.sub(put, reading)


def _in_math(text: str, pos: int) -> bool:
    return text[:pos].replace("\\$", "").count("$") % 2 == 1


def build(
    spec_path: Path = SPEC,
    runs: Path = RUNS,
    last: str = "bears_on",
    order: str = "reading-last",
    gap: str = "2pt",
) -> tuple[list[str], list[dict], dict]:
    """The table's lines. A row prints its check, its reading (the numbers filled in) and the
    row field `last` (where the reading applies, or the setting it was measured at); with
    order "last-reading" that field stands between the check and the reading."""
    if order not in ("reading-last", "last-reading"):
        raise ValueError(f"order is 'reading-last' or 'last-reading', not {order!r}")
    groups = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    lines, resolved, sources = [], [], {}
    seen: set[str] = set()
    for group in groups:
        tint = LATEX_COLOR_NAMES[group["meaning"]] + "Tint"  # an unknown meaning fails loud
        title = f"\\textbf{{{group['group']}}}"
        # no overhang: the tint stops at the table's rules instead of running into the margins
        lines.append(
            f"\\rowcolor{{{tint}}}[0pt][0pt]\\multicolumn{{3}}{{@{{}}l@{{}}}}{{{title}}} \\\\"
        )
        for row in group["rows"]:
            if row["id"] in seen:
                raise ValueError(f"duplicate row id {row['id']!r}")
            seen.add(row["id"])
            for need in row.get("require", []):
                value, path = resolve_value(need, runs)
                if value is not True:
                    raise ValueError(f"{row['id']}: {path} {need['key']} is {value!r}, not true")
                sources[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            numbers = []
            for k, spec in enumerate(row["values"]):
                value, path = resolve_value(spec, runs)
                shown = shown_value(value, spec, path)
                plain = plain_number(shown, spec["fmt"])
                if "expect" not in spec:
                    raise KeyError(f"{row['id']} value {k}: no `expect` (the text it must print)")
                if str(spec["expect"]) != plain:
                    raise ValueError(
                        f"{row['id']} value {k}: {path} {spec['key']} prints {plain!r}, "
                        f"the row expects {spec['expect']!r}"
                    )
                numbers.append(tex_number(shown, spec["fmt"]))
                for read in spec_paths(spec, runs):
                    sources[read] = hashlib.sha256(read.read_bytes()).hexdigest()
                resolved.append(
                    {
                        "row": row["id"],
                        "slot": k,
                        "value": value,
                        "printed": numbers[-1],
                        "run": spec["run"],
                        "file": spec["file"],
                        "key": spec["key"],
                        # everything the value was read by, so the CSV alone re-derives it
                        "spec": json.dumps(
                            {f: spec[f] for f in spec if f not in ("fmt", "expect", "sign")}
                        ),
                    }
                )
            reading = fill(" ".join(row["reading"].split()), numbers, row["id"])
            cells = [row["check"], reading, row[last]]
            if order == "last-reading":
                cells = [row["check"], row[last], reading]
            # a little air between rows: most wrap to two or three lines
            lines.append(" & ".join(cells) + f" \\\\\\addlinespace[{gap}]")
    return lines, resolved, sources


RAGGED = ">{\\raggedright\\arraybackslash}"
TABLES = (  # spec, output, column spec, head, the row field of the third cell, cell order
    (
        SPEC,
        OUT,
        f"\\begin{{tabularx}}{{\\textwidth}}{{@{{}}{RAGGED}p{{0.27\\textwidth}}Y"
        f"{RAGGED}p{{0.2\\textwidth}}@{{}}}}",
        "Check & Reading & Where it applies \\\\",
        "bears_on",
        "reading-last",
    ),
    (
        OTHER_SPEC,
        OTHER_OUT,
        f"\\begin{{tabularx}}{{\\textwidth}}{{@{{}}{RAGGED}p{{0.19\\textwidth}}"
        f"{RAGGED}p{{0.33\\textwidth}}Y@{{}}}}",
        "Check & Setting & Reading (macro-F1 unless stated) \\\\",
        "setting",
        "last-reading",
    ),
)
ROW_GAP = {OUT: "2pt", OTHER_OUT: "0.5pt"}  # the body table's rows are one or two lines


def write_table(spec: Path, out: Path, begin: str, head: str, last: str, order: str) -> None:
    lines, resolved, sources = build(spec, last=last, order=order, gap=ROW_GAP[out])
    body = [
        "% auto-generated by experiments/build_further_controls.py -- do not edit by hand",
        begin,
        "\\toprule",
        head,
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabularx}",
    ]
    out.write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resolved[0]))
        w.writeheader()
        w.writerows(resolved)
    out.with_name(f"{out.stem}_provenance.json").write_text(
        json.dumps(
            {
                "written_at": utc_stamp(),
                "git": git_describe(),
                "spec": spec.relative_to(REPO_ROOT).as_posix(),
                "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
                "sources_sha256": {
                    p.relative_to(REPO_ROOT).as_posix(): h for p, h in sorted(sources.items())
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"-> {out} ({len(resolved)} values in {len(lines)} lines)")


def main() -> None:
    for table in TABLES:
        write_table(*table)


if __name__ == "__main__":
    main()
