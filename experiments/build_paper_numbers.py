"""Write the LaTeX macros that carry every measured number shown in the paper's box diagrams and
in its hand-laid replication-data table.

The diagrams (paper/sections/fig_*.tex, visual pass, operator ruling 2026-11-19T15:42Z) and the
replication-data table (paper/sections/methodology.tex) are laid out by hand, so their measured
numbers come in through macros, never typed: this script reads each value from its artifact,
fails loud on a missing file or key, and writes `paper/tables/paper_numbers.tex` with
`paper_numbers_provenance.json` beside it (value, source file, key and the source's sha256).
Protocol constants that the runners take as settings (budgets, grids, seed sets) are not
measured values and stay in the diagram text, checked against the runner code as the prose was.

Sources: data/processed/meta_binary.json (rows, split sizes, class counts); for the replication
data, data/processed/meta_gas_binary.json and meta_fdia{14,118}_binary.json (rows kept and
generated, features, class counts) and the accuracy verdict of each source (best classical and
best quantum macro-F1 at eight dimensions). The checklist's readings are drawn by
figures.checklist_figure from paper/tables/instance_panels.csv and need no macros (the box
checklist and the ring that used them were retired on 2026-11-20).

Run:  uv run python -m experiments.build_paper_numbers
"""

from __future__ import annotations

import csv
import hashlib
import json

from qgridbench.eval.tables import _count
from qgridbench.utils.run_tracking import REPO_ROOT, git_describe, utc_stamp

OUT = REPO_ROOT / "paper" / "tables" / "paper_numbers.tex"
PROCESSED = REPO_ROOT / "data" / "processed"
META_BINARY = PROCESSED / "meta_binary.json"
RUNS = REPO_ROOT / "results" / "runs"
CELLS = REPO_ROOT / "experiments" / "replication_matrix_cells.csv"
# The gas accuracy verdict with pressure log-scaled: the run that
# experiments/make_report.py pins as GAS_ACCURACY_RUNS["slog"] (tests/test_paper_numbers.py
# checks that the two agree).
GAS_VERDICT_LOG_SCALED = "20261119T113544Z_gas_accuracy_verdict_gas_binary_slog"
# macro stem -> (metadata file, dataset name in the replication-matrix cells)
GRIDS = {
    "GridSmall": ("meta_fdia14_binary.json", "fdia14"),
    "GridLarge": ("meta_fdia118_binary.json", "fdia118"),
}


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _share(counts: dict, n: int, path) -> int:
    """Attack share in percent; the two class counts must sum to the row count."""
    if int(counts["0"]) + int(counts["1"]) != n:
        raise ValueError(f"{path}: class counts {counts} do not sum to n_rows {n}")
    return round(100 * int(counts["1"]) / n)


def data_numbers() -> list[dict]:
    meta = json.loads(META_BINARY.read_text())
    n = meta["n_rows"]
    attack = _share(meta["class_counts"], n, META_BINARY)
    src = _rel(META_BINARY)
    return [
        {"macro": "NumRowsBinary", "value": _count(n), "source": src, "key": "n_rows"},
        {
            "macro": "NumTrainRows",
            "value": _count(meta["split_sizes"]["train"]),
            "source": src,
            "key": "split_sizes/train",
        },
        {"macro": "NumAttackShare", "value": str(attack), "source": src, "key": "class_counts"},
        {
            "macro": "NumNaturalShare",
            "value": str(100 - attack),
            "source": src,
            "key": "class_counts",
        },
    ]


def _grid_verdict_runs() -> dict[str, str]:
    """The accuracy verdict run of each simulated grid, as the replication matrix reads it."""
    with CELLS.open(encoding="utf-8") as f:
        runs = {r["dataset"]: r["run"] for r in csv.DictReader(f) if r["claim"] == "accuracy"}
    missing = [d for _, d in GRIDS.values() if not runs.get(d)]
    if missing:
        raise KeyError(f"{CELLS}: no accuracy cell with a run for {missing}")
    return runs


def _best_pair(run: str, stem: str) -> list[dict]:
    """Best classical and best quantum mean macro-F1 of an accuracy verdict, at eight dimensions."""
    path = RUNS / run / "results.json"
    pca8 = json.loads(path.read_text())["pca8"]
    out = []
    for group in ("classical", "quantum"):
        model = pca8[f"best_{group}"]
        out.append(
            {
                "macro": f"Num{stem}Best{group.capitalize()}",
                "value": f"{pca8['mean'][model]:.3f}",
                "source": _rel(path),
                "key": f"pca8/mean/{model}",
            }
        )
    return out


def replication_numbers() -> list[dict]:
    """The replication-data table: rows, features and attack share of the gas data and the two
    simulated grids, and each source's best matched scores."""
    gas_path = PROCESSED / "meta_gas_binary.json"
    gas = json.loads(gas_path.read_text())
    src = _rel(gas_path)
    out = [
        {"macro": "NumGasRows", "value": _count(gas["n_rows"]), "source": src, "key": "n_rows"},
        {
            "macro": "NumGasRawRows",
            "value": _count(gas["n_raw_rows"]),
            "source": src,
            "key": "n_raw_rows",
        },
        {
            "macro": "NumGasFeatures",
            "value": str(gas["n_features"]),
            "source": src,
            "key": "n_features",
        },
        {
            "macro": "NumGasAttackShare",
            "value": str(_share(gas["class_counts"], gas["n_rows"], gas_path)),
            "source": src,
            "key": "class_counts",
        },
    ]
    out += _best_pair(GAS_VERDICT_LOG_SCALED, "Gas")
    grid_runs = _grid_verdict_runs()
    shared: dict[str, set] = {"n_rows": set(), "n_rows_generated": set(), "share": set()}
    for stem, (name, dataset) in GRIDS.items():
        path = PROCESSED / name
        meta = json.loads(path.read_text())
        shared["n_rows"].add(meta["n_rows"])
        shared["n_rows_generated"].add(meta["n_rows_generated"])
        shared["share"].add(_share(meta["class_counts"], meta["n_rows"], path))
        out.append(
            {
                "macro": f"Num{stem}Features",
                "value": str(meta["n_features"]),
                "source": _rel(path),
                "key": "n_features",
            }
        )
        out += _best_pair(grid_runs[dataset], stem)
    # the table prints one row count and one attack share for both grids
    for key, values in shared.items():
        if len(values) != 1:
            raise ValueError(f"the two grids disagree on {key}: {sorted(values)}")
    both = " and ".join(_rel(PROCESSED / name) for name, _ in GRIDS.values())
    out += [
        {
            "macro": "NumGridRows",
            "value": _count(shared["n_rows"].pop()),
            "source": both,
            "key": "n_rows",
        },
        {
            "macro": "NumGridRowsGenerated",
            "value": _count(shared["n_rows_generated"].pop()),
            "source": both,
            "key": "n_rows_generated",
        },
        {
            "macro": "NumGridAttackShare",
            "value": str(shared["share"].pop()),
            "source": both,
            "key": "class_counts",
        },
    ]
    return out


def main() -> None:
    numbers = data_numbers() + replication_numbers()
    names = [n["macro"] for n in numbers]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate macro names: {sorted(names)}")
    lines = [
        "% Generated by experiments/build_paper_numbers.py; do not edit. Provenance:",
        "% paper/tables/paper_numbers_provenance.json",
    ]
    lines += [f"\\newcommand{{\\{n['macro']}}}{{{n['value']}}}" for n in numbers]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = {REPO_ROOT / part for n in numbers for part in n["source"].split(" and ")}
    sources = {_rel(p): _sha(p) for p in sorted(files)}
    OUT.with_name("paper_numbers_provenance.json").write_text(
        json.dumps(
            {
                "written_at": utc_stamp(),
                "git": git_describe(),
                "numbers": numbers,
                "sources_sha256": sources,
            },
            indent=2,
        )
    )
    print(f"-> {OUT} ({len(numbers)} macros)")


if __name__ == "__main__":
    main()
