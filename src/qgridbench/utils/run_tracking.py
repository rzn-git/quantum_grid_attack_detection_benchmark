"""Run-directory + MLflow tracking helpers.

Every experiment produces BOTH (per study protocol §9):
  1. a plain-JSON run dir under results/runs/<UTC-timestamp>_<name>/ containing
     the resolved config, git SHA (+dirty flag), seed list, wall time, and the
     package lock hash — the artifact-traceable record every reported number
     must resolve to; and
  2. an MLflow run in the local file store (mlruns/, git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results" / "runs"

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    return logger


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def git_describe() -> str:
    """Git SHA with dirty flag, e.g. 'a1b2c3d' or 'a1b2c3d-dirty'."""
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def lock_hash() -> str:
    """SHA256 of requirements.lock.txt — pins the environment a run executed in."""
    lock = REPO_ROOT / "requirements.lock.txt"
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def compute_env() -> dict[str, Any]:
    """Worker/thread counts the fits depend on — provenance the sha and lock do not carry.

    XGBoost's `tree_method="hist"` sums gradients in thread-arrival order, so its fitted
    model is a function of the WORKER COUNT even at a fixed `random_state`: measured
    2-vs-10 threads moves its macro-F1 by 0.0144 at one seed, while a single-threaded
    estimator on the identical path is bit-identical (decision log 2026-08-15T20:09Z).
    Recording the counts is what makes a run reproducible rather than merely traceable;
    the runners deliberately use the whole machine, so the number is captured, not pinned.
    """
    return {
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


class RunDir:
    """Context manager: creates the run dir, writes provenance on entry and
    metrics + wall time on exit. Exceptions propagate (fail loud) but the
    partial run dir is kept with an 'error' status for post-mortem."""

    def __init__(self, name: str, config: dict[str, Any], seeds: list[int] | None = None):
        self.name = name
        self.config = config
        self.seeds = seeds or []
        self.path = RESULTS_DIR / f"{utc_stamp()}_{name}"
        self.metrics: dict[str, Any] = {}
        self._t0 = 0.0

    def __enter__(self) -> RunDir:
        # same-second launches of same-named runs collide on the UTC stamp;
        # suffix deterministically instead of dying (parallel-process safe)
        base = self.path
        for i in range(2, 100):
            try:
                self.path.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                self.path = base.with_name(f"{base.name}_{i}")
        else:
            raise FileExistsError(f"could not allocate run dir for {base}")
        (self.path / "params.yaml").write_text(
            yaml.safe_dump({"config": self.config, "seeds": self.seeds}, sort_keys=False)
        )
        (self.path / "git_sha.txt").write_text(git_describe() + "\n")
        (self.path / "lock_hash.txt").write_text(lock_hash() + "\n")
        (self.path / "compute_env.json").write_text(json.dumps(compute_env(), indent=2))
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        wall = time.perf_counter() - self._t0
        payload = {
            "status": "error" if exc_type else "ok",
            "wall_time_s": round(wall, 3),
            "metrics": self.metrics,
        }
        if exc_type:
            payload["error"] = repr(exc)
        (self.path / "metrics.json").write_text(json.dumps(payload, indent=2, default=_np_safe))
        # never swallow the exception
        return None

    def write_json(self, filename: str, obj: Any) -> Path:
        p = self.path / filename
        p.write_text(json.dumps(obj, indent=2, default=_np_safe))
        return p


def _np_safe(o: Any):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def mlflow_log(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
) -> None:
    """Log one finished run to the local MLflow store.

    MLflow 3.x deprecated the filesystem backend; a local sqlite DB inside the
    git-ignored mlruns/ dir is the supported equivalent (decision log 2026-08-13).
    """
    import mlflow

    (REPO_ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(REPO_ROOT / 'mlruns' / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("qgridbench")
    with mlflow.start_run(run_name=run_name):
        flat = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in params.items()}
        mlflow.log_params({k: str(v)[:490] for k, v in flat.items()})
        mlflow.log_metrics(
            {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v}
        )
        if tags:
            mlflow.set_tags(tags)


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
