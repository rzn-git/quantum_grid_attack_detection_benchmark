"""Pool HopSkipJump restart runs into one summary per model (archive-only: attacks nothing).

The paper's black-box cells quote restarts that sit in more than one run: for four models the
restarts of 2026-08-15 and of 2026-11-20 are one sample of ten (operator ruling (i), decision
log 2026-11-20T09:19Z). This reads the restart runs named in experiments/hsj_restart_pool.yaml,
concatenates each model's per-restart records, and writes one run whose `summary` has the
schema the exhibit builders already read (`run_hopskipjump_restarts.py` writes the same keys),
so the robustness table, the instances figure and the replication figure read one run.

A model is read over the rows the attack found when every pooled restart records the found
share; otherwise over all points, which is allowed only while no restart has a zero median and
every recorded share is at least `min_found_share_all_points`.

Every source run must record its variant, tuned file and attack budget, or have them stated in
the pool file; runs that stored per-point rows must have attacked the same test rows.

Run: python -m experiments.pool_hsj_restarts
     python -m experiments.pool_hsj_restarts --pool-file experiments/hsj_supplied_start_pool.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml

log = get_logger(__name__)

RUNS = REPO_ROOT / "results" / "runs"
DEFAULT_POOL_FILE = "experiments/hsj_restart_pool.yaml"
REQUIRED = ("variant", "best_params")


def _spread(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def pool_family(records_by_run: dict[str, list[dict]], min_found_share: float) -> tuple:
    """Concatenate one model's restart records and summarise them.

    Returns (records, summary). `records` carry their `source_run`; `summary` has mean, std
    (ddof 1), min, max and cv of the all-points medians, plus `found_share` and `found` when
    every record carries the found share (the keys the table builder reads).
    """
    records = [
        {**rec, "source_run": run, "pooled_index": i}
        for i, (run, rec) in enumerate(
            (run, rec) for run, recs in records_by_run.items() for rec in recs
        )
    ]
    if not records:
        raise ValueError("no restart records to pool")
    medians = np.array([r["median_linf_sigma"] for r in records], dtype=float)
    summary = _spread(medians)
    summary["cv"] = summary["std"] / summary["mean"] if summary["mean"] else float("nan")
    summary["n_restarts"] = len(records)
    summary["source_runs"] = list(records_by_run)

    has_share = ["found_share" in r for r in records]
    if all(has_share):
        share = np.array([r["found_share"] for r in records], dtype=float)
        found = np.array([r["median_linf_sigma_found"] for r in records], dtype=float)
        if np.isnan(found).any():
            raise ValueError(
                "a restart found no row at all, so it has no found-rows median: the model has "
                "no distortion to report for that restart and cannot be summarised over them"
            )
        summary["found_share"] = {
            "mean": float(share.mean()),
            "min": float(share.min()),
            "max": float(share.max()),
        }
        summary["found"] = _spread(found)
        return records, summary

    # some restarts predate the found share: the cell is read over all points
    if (medians == 0).any():
        raise ValueError(
            "a restart without a recorded found share has a zero median: rows the attack "
            "could not start on are not a zero distortion, so this run cannot be pooled"
        )
    if any(has_share):
        share = np.array([r["found_share"] for r in records if "found_share" in r], dtype=float)
        if share.min() < min_found_share:
            raise ValueError(
                f"recorded found share {share.min():.2f} is under {min_found_share}: the model "
                "must be read over found rows, which needs the share on every pooled restart"
            )
        summary["found_share_recorded"] = {
            "n_restarts": int(len(share)),
            "mean": float(share.mean()),
            "min": float(share.min()),
        }
    return records, summary


def statistic(summary: dict, record: dict) -> float:
    """The restart value the exhibits read for this model: found rows when recorded throughout."""
    key = "median_linf_sigma_found" if "found" in summary else "median_linf_sigma"
    return float(record[key])


def source_config(run: str, spec: dict) -> dict:
    """A source run's config, completed from the pool file's `stated` block; nothing is assumed."""
    cfg = yaml.safe_load((RUNS / run / "params.yaml").read_text(encoding="utf-8"))["config"]
    stated = (spec.get("stated") or {}).get(run, {})
    clash = sorted(k for k in stated if k in cfg and cfg[k] != stated[k])
    if clash:
        raise ValueError(f"{run}: the pool file states {clash} differently from its params.yaml")
    cfg = {**stated, **cfg}
    missing = [k for k in (*REQUIRED, *spec["must_match"]) if k not in cfg]
    if missing:
        raise ValueError(
            f"{run}: params.yaml does not record {missing}; state them with their evidence "
            "under `stated` in the pool file"
        )
    return cfg


def check_sources(spec: dict) -> dict[str, dict]:
    """Every source run is of the pool's variant and width and agrees on the attack budget;
    the runs pooled under one model used one tuned file. Returns each run's config."""
    tag = f"_{spec['variant']}_q{spec['qubits']}"
    configs = {}
    for run in sorted({r for runs in spec["families"].values() for r in runs}):
        if tag not in run:
            raise ValueError(f"{run}: the run name does not carry {tag!r}")
        cfg = source_config(run, spec)
        if cfg["variant"] != spec["variant"]:
            raise ValueError(f"{run}: variant {cfg['variant']!r}, expected {spec['variant']!r}")
        configs[run] = cfg
    for key in spec["must_match"]:
        seen = {run: cfg[key] for run, cfg in configs.items()}
        if len(set(seen.values())) != 1:
            raise ValueError(f"source runs disagree on {key}: {seen}")
    for fam, runs in spec["families"].items():
        tuned = {configs[r]["best_params"] for r in runs}
        if len(tuned) > 1:
            raise ValueError(f"{fam}: pooled runs used different tuned files: {sorted(tuned)}")
    return configs


def check_points(runs: list[str]) -> dict[str, bool]:
    """Runs that stored per-point rows attacked the same test rows. Returns, per run, whether
    it could be checked; a run without stored rows is reported, not assumed."""
    ids = {}
    for run in runs:
        path = RUNS / run / "predictions.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["condition", "row_id"])
            ids[run] = frozenset(df.loc[df["condition"] == "clean", "row_id"])
    if len(set(ids.values())) > 1:
        raise ValueError(
            f"source runs attacked different test rows: { {r: len(v) for r, v in ids.items()} }"
        )
    unchecked = [r for r in runs if r not in ids]
    if unchecked:
        log.warning("attacked rows not stored, so not compared, for: %s", ", ".join(unchecked))
    return {run: run in ids for run in runs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool-file", default=DEFAULT_POOL_FILE)
    args = ap.parse_args()
    spec = load_yaml(REPO_ROOT / args.pool_file)
    configs = check_sources(spec)
    points_checked = check_points(sorted(configs))

    sources, per_family, summary = {}, {}, {}
    for fam, runs in spec["families"].items():
        by_run = {}
        for run in runs:
            path = RUNS / run / "results.json"
            sources[run] = hashlib.sha256(path.read_bytes()).hexdigest()
            by_run[run] = json.loads(path.read_text(encoding="utf-8"))["per_family"][fam]
        per_family[fam], summary[fam] = pool_family(by_run, spec["min_found_share_all_points"])

    out: dict = {"per_family": per_family, "summary": summary}
    if "qsvm" in summary:
        others = {
            f: max(statistic(summary[f], r) for r in per_family[f]) for f in summary if f != "qsvm"
        }
        top = max(others, key=others.get)
        above = [statistic(summary["qsvm"], r) > others[top] for r in per_family["qsvm"]]
        out["qsvm_above_every_other_restart"] = {
            "count": int(sum(above)),
            "of": len(above),
            "highest_other": others[top],
            "highest_other_family": top,
        }
        out["qsvm_highest_every_restart"] = bool(all(above))

    # an optional `tag` in the pool file names another pooled set (the supplied-start runs),
    # so its run folder is never taken for the one the exhibits pin
    name = f"hsj_restarts_pooled_{spec['variant']}_q{spec['qubits']}"
    if spec.get("tag"):
        name += f"_{spec['tag']}"
    with RunDir(name, config={"pool_file": args.pool_file, **spec}, seeds=[]) as run:
        out["sources"] = {
            r: {
                "results_sha256": sources[r],
                "config": configs[r],
                "points_checked": points_checked[r],
            }
            for r in sorted(sources)
        }
        rows = ["family,source_run,restart,median_linf_sigma,found_share,median_linf_sigma_found"]
        for fam, recs in per_family.items():
            for r in recs:
                rows.append(
                    f"{fam},{r['source_run']},{r['restart']},{r['median_linf_sigma']!r},"
                    f"{r.get('found_share', '')},{r.get('median_linf_sigma_found', '')}"
                )
        Path(run.path, "per_restart.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        run.write_json("results.json", out)
        for fam, s in summary.items():
            log.info(
                "%s: %d restarts from %d run(s), %.4f +- %.4f",
                fam,
                s["n_restarts"],
                len(s["source_runs"]),
                s["mean"],
                s["std"],
            )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
