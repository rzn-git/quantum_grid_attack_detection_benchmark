"""R4-01: refits the published XGBoost accuracy rows at several thread counts and reports the
10-seed mean and std of test macro-F1 per thread count, with LightGBM and RF as extra columns.

Full run (the cell at the largest thread count runs last and alone):
    uv run python -m experiments.run_xgb_thread_sensitivity --workers 1
Smoke:
    uv run python -m experiments.run_xgb_thread_sensitivity --workers 1 --seeds 0 1 --smoke
Each process peaks at 1.22 GB (measured in the smoke), so `--workers W` needs W x 1.25 GB free;
one worker runs the cells one after another in about 90 minutes.

XGBoost's `tree_method="hist"` sums gradients in thread-arrival order, so its fitted model
depends on the thread count at a fixed seed (decision log 2026-08-15T20:09Z). A thread count
cannot change inside a process once OpenMP is loaded, so every thread count runs in its own
subprocess (`--worker`), whose thread variables are set before numpy is imported.

Rows refitted: the binary task, the shipped split, the tuned parameters of the published
classical runs, and their code path (`build`, `fit_model`, `predict_proba_timed`,
`compute_all`, exactly as `experiments.run_classical.eval_model` calls them):
    full  - 128 features, `20260813T064447Z_classical_binary_full` (Table VI, arXiv appendix)
    pca8  - PCA-8, 2,000-row cap, `20260813T064540Z_classical_binary_pca8` (Table II)
Neither run recorded its thread count. The published thread count of each is the one whose cell
reproduces its seed-0 XGBoost macro-F1 exactly (6 for full, 4 for pca8), and the anchor checks
it on every seed.

Pre-registration R4-01 (decision log 2026-11-16T11:07Z):
    statistic: D = the largest difference between the XGBoost 10-seed means over 1, 4, 10, 24
        threads, against the seed std (the published row's std)
    DOCUMENT if D <= seed std, DISCLOSE otherwise
    anchor (blocking): the cell at the published thread count reproduces the published
        XGBoost mean within 1e-6
"""

import argparse
import os
import sys

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--worker", action="store_true")
_pre.add_argument("--threads", type=int, default=None)
_PRE = _pre.parse_known_args(sys.argv[1:])[0]
if _PRE.worker:
    if _PRE.threads is None:
        raise SystemExit("--worker needs --threads")
    for _var in THREAD_VARS:
        os.environ[_var] = str(_PRE.threads)

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from experiments.runner_guard import assert_clean_predictor  # noqa: E402
from qgridbench.eval.metrics import compute_all  # noqa: E402
from qgridbench.models.classical.zoo import (  # noqa: E402
    build,
    fit_model,
    predict_proba_timed,
    stratified_cap,
)
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import set_all_seeds  # noqa: E402

log = get_logger(__name__)

REGISTERED_GRID = (1, 4, 10, 24)
ANCHOR_TOL = 1e-6
MEM_FRACTION = 0.8
MODELS = ("xgb", "lgbm", "rf")
REGIMES = {
    "full": {
        "run": "20260813T064447Z_classical_binary_full",
        "regime": "full",
        "pca": None,
        "subset_cap": None,
        "published_threads": 6,
        "paper_table": "VI (tab:binary_full, arXiv appendix)",
    },
    "pca8": {
        "run": "20260813T064540Z_classical_binary_pca8",
        "regime": "pca",
        "pca": 8,
        "subset_cap": load_yaml(REPO_ROOT / "configs" / "classical.yaml")["subset_cap"],
        "published_threads": 4,
        "paper_table": "II (tab:rq1)",
    },
}


def published(regime: str) -> dict:
    path = REPO_ROOT / "results" / "runs" / REGIMES[regime]["run"] / "results.json"
    return json.loads(path.read_text())


def _peak_rss_gb() -> float:
    mi = psutil.Process().memory_info()
    if sys.platform == "win32":
        return mi.peak_wset / 2**30
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def fit_score(name, params, Xtr, ytr, Xte, yte, seed) -> dict:
    """One seed of `run_classical.eval_model`: the same calls in the same order."""
    set_all_seeds(seed)
    est = build(name, params, seed)
    train_s = fit_model(name, est, Xtr, ytr)
    proba, latency = predict_proba_timed(est, Xte)
    assert_clean_predictor(name, yte, proba, proba.argmax(1))
    return {"metrics": compute_all(yte, proba), "train_time_s": train_s, "latency_s": latency}


def unit_name(threads: int, regime: str, model: str, seed: int) -> str:
    return f"t{threads:02d}_{regime}_{model}_s{seed}.json"


def thread_report() -> dict:
    """What the libraries in this process actually see; recorded with every unit."""
    import joblib
    import torch
    from threadpoolctl import threadpool_info

    return {
        "env": {v: os.environ.get(v) for v in THREAD_VARS},
        "joblib_cpu_count": joblib.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "threadpools": [
            {k: p.get(k) for k in ("user_api", "internal_api", "num_threads", "filepath")}
            for p in threadpool_info()
        ],
    }


def run_worker(args) -> None:
    """Fit every unit of this (thread count, shard) that has no file in units/ yet."""
    from qgridbench.protocol import prepare_regime

    want = str(args.threads)
    got = {v: os.environ.get(v) for v in THREAD_VARS}
    if any(x != want for x in got.values()):
        raise AssertionError(f"worker {os.getpid()}: thread caps {got}, expected {want}")
    units_dir = Path(args.run_dir) / "units"
    seeds = [s for s in args.seeds if s % args.n_shards == args.shard]
    for regime in args.regimes:
        spec = REGIMES[regime]
        pub = published(regime)
        todo = [
            (m, s)
            for m in args.models
            for s in seeds
            if not (units_dir / unit_name(args.threads, regime, m, s)).exists()
        ]
        if not todo:
            continue
        reg = prepare_regime("binary", spec["regime"], n_components=spec["pca"], seed=0)
        Xtr, ytr = reg.X["train"], reg.y["train"]
        cap = spec["subset_cap"]
        idx = stratified_cap(Xtr, ytr, cap, seed=0) if cap else np.arange(len(ytr))
        for m, s in todo:
            t0 = time.perf_counter()
            out = fit_score(
                m, pub[m]["best_params"], Xtr[idx], ytr[idx], reg.X["test"], reg.y["test"], s
            )
            out.update(
                {
                    "threads": args.threads,
                    "regime": regime,
                    "model": m,
                    "seed": s,
                    "n_train": len(idx),
                    "unit_wall_s": time.perf_counter() - t0,
                    "peak_rss_gb": _peak_rss_gb(),
                    "thread_report": thread_report(),
                }
            )
            (units_dir / unit_name(args.threads, regime, m, s)).write_text(json.dumps(out))
            log.info(
                "T=%d %s %s s%d macro-F1 %.6f (%.1f s)",
                args.threads,
                regime,
                m,
                s,
                out["metrics"]["macro_f1"],
                out["unit_wall_s"],
            )


def memory_guard(workers: int, rss_gb: float, budget_gb: float, what: str) -> dict:
    avail = psutil.virtual_memory().available / 2**30
    need = workers * rss_gb
    cap = min(budget_gb, MEM_FRACTION * avail)
    out = {"workers": workers, "worker_rss_gb": rss_gb, "need_gb": need, "cap_gb": cap}
    log.info("memory (%s): %d x %.2f GB = %.2f GB vs cap %.2f GB", what, *out.values())
    if need > cap:
        raise MemoryError(f"{what}: {need:.2f} GB > cap {cap:.2f} GB")
    return out


def _launch(job: dict, run_dir: Path, args) -> subprocess.Popen:
    t, shard, n = job["threads"], job["shard"], job["n_shards"]
    env = {**os.environ, **{v: str(t) for v in THREAD_VARS}}
    cmd = [
        sys.executable,
        "-m",
        "experiments.run_xgb_thread_sensitivity",
        "--worker",
        "--threads",
        str(t),
        "--shard",
        str(shard),
        "--n-shards",
        str(n),
        "--run-dir",
        str(run_dir),
        "--seeds",
        *map(str, args.seeds),
        "--regimes",
        *args.regimes,
        "--models",
        *args.models,
    ]
    logf = open(run_dir / "logs" / f"worker_t{t:02d}_shard{shard}of{n}.log", "a")  # noqa: SIM115
    return subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)


def run_jobs(jobs: list[dict], workers: int, run_dir: Path, args) -> None:
    """At most `workers` subprocesses at once; any non-zero exit fails after the queue drains."""
    queue, running, failed = list(jobs), [], []
    while queue or running:
        while queue and len(running) < workers:
            job = queue.pop(0)
            running.append((job, _launch(job, run_dir, args)))
        time.sleep(2)
        for item in list(running):
            job, proc = item
            if proc.poll() is not None:
                running.remove(item)
                if proc.returncode != 0:
                    log.error("job %s exited %d", job, proc.returncode)
                    failed.append(job)
    if failed:
        raise RuntimeError(f"{len(failed)} worker job(s) failed (see {run_dir / 'logs'}): {failed}")


def _ms(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def cell_table(units: list[dict], seeds: list[int]) -> dict:
    """{regime: {model: {threads: {per_seed, mean, std}}}} from unit records."""
    table: dict = {}
    for u in units:
        table.setdefault(u["regime"], {}).setdefault(u["model"], {}).setdefault(u["threads"], {})[
            u["seed"]
        ] = u["metrics"]["macro_f1"]
    out: dict = {}
    for regime, by_model in table.items():
        for model, by_t in by_model.items():
            for t, by_seed in by_t.items():
                missing = sorted(set(seeds) - set(by_seed))
                if missing:
                    raise AssertionError(f"{regime}/{model}/T={t}: seeds {missing} missing")
                per_seed = [by_seed[s] for s in seeds]
                out.setdefault(regime, {}).setdefault(model, {})[t] = {
                    "per_seed": per_seed,
                    **_ms(per_seed),
                }
    return out


def anchor(cells: dict, pub_per_seed: list[float], seeds: list[int], threads: int) -> dict:
    """The published thread count's cell against the published row on the same seeds."""
    ours = cells[threads]["per_seed"]
    theirs = [pub_per_seed[s] for s in seeds]
    per_seed = [a - b for a, b in zip(ours, theirs, strict=True)]
    mean_diff = float(np.mean(ours) - np.mean(theirs))
    return {
        "threads": threads,
        "mean_ours": float(np.mean(ours)),
        "mean_published": float(np.mean(theirs)),
        "mean_abs_diff": abs(mean_diff),
        "max_abs_seed_diff": float(np.max(np.abs(per_seed))),
        "tol": ANCHOR_TOL,
        "passed": abs(mean_diff) <= ANCHOR_TOL,
    }


def branch(cells: dict, grid, seed_std: float) -> dict:
    """D = the largest across-thread difference in means over `grid`; DOCUMENT iff D <= std."""
    means = {t: cells[t]["mean"] for t in grid}
    d = max(means.values()) - min(means.values())
    return {
        "grid": list(grid),
        "means": means,
        "D": d,
        "seed_std": seed_std,
        "D_over_std": d / seed_std,
        "branch": "DOCUMENT" if d <= seed_std else "DISCLOSE",
        "argmax_threads": max(means, key=means.get),
        "argmin_threads": min(means, key=means.get),
    }


def stability(cells: dict, ref_threads: int) -> dict:
    """Largest per-seed change of a model's macro-F1 against the reference thread count."""
    ref = np.asarray(cells[ref_threads]["per_seed"])
    diffs = {t: float(np.max(np.abs(np.asarray(c["per_seed"]) - ref))) for t, c in cells.items()}
    return {"ref_threads": ref_threads, "max_abs_seed_diff": diffs, "max": max(diffs.values())}


def summarize(units: list[dict], seeds: list[int], regimes: list[str], grid) -> dict:
    cells = cell_table(units, seeds)
    out = {}
    for regime in regimes:
        spec, pub = REGIMES[regime], published(regime)
        t_pub = spec["published_threads"]
        xgb = cells[regime]["xgb"]
        reg_grid = [t for t in grid if t in REGISTERED_GRID]
        all_t = sorted(xgb)
        std_pub = pub["xgb"]["aggregate"]["macro_f1"]["std"]
        cell_stds = [xgb[t]["std"] for t in reg_grid]
        rec = {
            "paper_table": spec["paper_table"],
            "published_run": spec["run"],
            "published_threads": t_pub,
            "anchor": anchor(xgb, [r["macro_f1"] for r in pub["xgb"]["per_seed"]], seeds, t_pub),
            "xgb_cells": xgb,
            "statistic": branch(xgb, reg_grid, std_pub),
            "sensitivity": {
                "with_published_threads_cell": branch(xgb, all_t, std_pub),
                "against_min_cell_std": branch(xgb, reg_grid, min(cell_stds)),
                "against_max_cell_std": branch(xgb, reg_grid, max(cell_stds)),
            },
            "other_models": {},
        }
        for m in [m for m in cells[regime] if m != "xgb"]:
            pub_m = [r["macro_f1"] for r in pub[m]["per_seed"]]
            rec["other_models"][m] = {
                "cells": {
                    t: {"mean": c["mean"], "std": c["std"]} for t, c in cells[regime][m].items()
                },
                "across_threads": stability(cells[regime][m], t_pub),
                "anchor_at_published_threads": anchor(cells[regime][m], pub_m, seeds, t_pub),
            }
        out[regime] = rec
    branches = {r: out[r]["statistic"]["branch"] for r in regimes}
    return {
        "regimes": out,
        "branch_by_regime": branches,
        "branch": "DISCLOSE" if "DISCLOSE" in branches.values() else "DOCUMENT",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help="internal: one thread count's fits")
    ap.add_argument("--threads", type=int, default=None, help="internal: the worker's count")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--grid", type=int, nargs="+", default=list(REGISTERED_GRID))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--workers", type=int, default=1, help="concurrent cells before the last")
    ap.add_argument(
        "--shards",
        nargs="*",
        default=[],
        help="T:k splits the T-thread cell's seeds over k processes (e.g. 1:2)",
    )
    ap.add_argument("--worker-rss-gb", type=float, default=1.25, help="measured peak per process")
    ap.add_argument("--mem-budget-gb", type=float, default=1.5)
    ap.add_argument("--resume", type=Path, default=None, help="run dir of an interrupted run")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.worker:
        run_worker(args)
        return

    pub_t = sorted({REGIMES[r]["published_threads"] for r in args.regimes})
    cells = sorted(set(args.grid) | set(pub_t))
    shards = {int(k): int(v) for k, v in (s.split(":") for s in args.shards)}
    last = max(cells)
    if shards.get(last, 1) != 1:
        raise ValueError(f"the {last}-thread cell runs alone as one process")
    early = [
        {"threads": t, "shard": i, "n_shards": shards.get(t, 1)}
        for t in cells[:-1]
        for i in range(shards.get(t, 1))
    ]
    late = [{"threads": last, "shard": 0, "n_shards": 1}]
    name = "xgb_thread_sensitivity_binary" + ("_smoke" if args.smoke else "")
    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "cells": cells,
        "registered_grid": list(REGISTERED_GRID),
        "regime_specs": REGIMES,
        "anchor_tol": ANCHOR_TOL,
    }
    with RunDir(name, config=config, seeds=args.seeds) as run:
        (run.path / "units").mkdir()
        (run.path / "logs").mkdir()
        if args.resume:
            for f in (args.resume / "units").glob("*.json"):
                shutil.copy2(f, run.path / "units" / f.name)
        workers = min(args.workers, len(early))
        guard = [
            memory_guard(workers, args.worker_rss_gb, args.mem_budget_gb, "cells below the last"),
            memory_guard(1, args.worker_rss_gb, args.mem_budget_gb, f"{last}-thread cell"),
        ]
        t0 = time.perf_counter()
        run_jobs(early, workers, run.path, args)
        t_early = time.perf_counter() - t0
        run_jobs(late, 1, run.path, args)
        t_late = time.perf_counter() - t0 - t_early

        units = [json.loads(f.read_text()) for f in sorted((run.path / "units").glob("*.json"))]
        summary = summarize(units, args.seeds, args.regimes, cells)
        peak = max(u["peak_rss_gb"] for u in units)
        cost = {
            "peak_rss_gb_max": peak,
            "wall_s_cells_below_last": t_early,
            "wall_s_last_cell": t_late,
            "unit_wall_s": {
                f"T{t}_{r}_{m}": _ms(
                    [
                        u["unit_wall_s"]
                        for u in units
                        if (u["threads"], u["regime"], u["model"]) == (t, r, m)
                    ]
                )["mean"]
                for t in cells
                for r in args.regimes
                for m in args.models
            },
            "memory_guard": guard,
        }
        run.write_json("results.json", {"summary": summary, "cost": cost, "units": units})
        run.metrics.update(
            {
                "branch": summary["branch"],
                **{f"{r}_branch": b for r, b in summary["branch_by_regime"].items()},
                **{
                    f"{r}_anchor_passed": summary["regimes"][r]["anchor"]["passed"]
                    for r in args.regimes
                },
                **{f"{r}_D": summary["regimes"][r]["statistic"]["D"] for r in args.regimes},
                "peak_rss_gb_max": peak,
            }
        )
        print(json.dumps(run.metrics, indent=2))
        print(f"-> {run.path}")
        failed = [r for r in args.regimes if not summary["regimes"][r]["anchor"]["passed"]]
        if failed:
            raise AssertionError(
                "anchor failed: "
                + "; ".join(f"{r} {summary['regimes'][r]['anchor']}" for r in failed)
            )
        if peak > args.worker_rss_gb:
            raise MemoryError(
                f"measured peak {peak:.2f} GB > the guard's {args.worker_rss_gb:.2f} GB; "
                "rerun with --worker-rss-gb set to the measured peak"
            )


if __name__ == "__main__":
    main()
