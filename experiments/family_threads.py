"""Per-family thread counts for runners whose anchors were fitted at different thread counts.

XGBoost's fitted model depends on its thread count, and `n_jobs` above OMP_NUM_THREADS is
silently capped (decision log 2026-11-16T13:05Z), so a family that must run at another count
runs in its own spawned process whose environment carries that count. Used by
`experiments/run_leak_decay_curve.py` and `experiments/run_multiclass_confusion.py`.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")


def parse_family_threads(items: list[str]) -> dict[str, int]:
    """`["xgb=24"]` -> `{"xgb": 24}`; fails loud on anything else."""
    out = {}
    for it in items:
        fam, sep, n = it.partition("=")
        if not sep or not fam or not n.isdigit() or int(n) < 1:
            raise SystemExit(f"--family-threads takes FAMILY=N with N >= 1, got {it!r}")
        out[fam] = int(n)
    return out


@contextmanager
def thread_env(n: int) -> Iterator[None]:
    """Every thread variable set to n for the block (children spawned inside inherit it)."""
    saved = {v: os.environ.get(v) for v in THREAD_VARS}
    for v in THREAD_VARS:
        os.environ[v] = str(n)
    try:
        yield
    finally:
        for v, x in saved.items():
            if x is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = x


def check_thread_env(n: int) -> None:
    got = {v: os.environ.get(v) for v in THREAD_VARS}
    if any(x != str(n) for x in got.values()):
        raise AssertionError(f"process {os.getpid()}: thread caps {got}, expected {n}")


def group_by_threads(units: list, threads_of: Callable[[object], int]) -> dict[int, list]:
    groups: dict[int, list] = {}
    for u in units:
        groups.setdefault(threads_of(u), []).append(u)
    return groups


def run_grouped(
    units: list,
    threads_of: Callable[[object], int],
    fn: Callable,
    init: Callable,
    initargs: tuple,
    workers: int,
    default_threads: int,
    on_done: Callable,
) -> None:
    """Run `fn(unit)` for every unit; `init(threads, *initargs)` prepares each process.

    Units at `default_threads` run in-process when `workers == 1`, else in a pool of
    `workers`. Units at any other count run one at a time in a spawned process whose
    environment carries that count. `on_done(unit, result)` sees each result as it lands.

    Off-count groups run FIRST, while this process is still light: run after the in-process
    units, the spawned child's memory adds to everything those units left resident here.
    """
    groups = group_by_threads(units, threads_of)
    for t in sorted(groups, key=lambda t: (t == default_threads, t)):
        group = groups[t]
        if t == default_threads and workers == 1:
            init(t, *initargs)
            for u in group:
                on_done(u, fn(u))
            continue
        with (
            thread_env(t),
            ProcessPoolExecutor(
                max_workers=workers if t == default_threads else 1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=init,
                initargs=(t, *initargs),
            ) as ex,
        ):
            futs = {ex.submit(fn, u): u for u in group}
            for f in as_completed(futs):
                on_done(futs[f], f.result())
