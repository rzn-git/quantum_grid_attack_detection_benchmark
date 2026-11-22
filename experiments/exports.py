"""Per-run exports beside results.json (operator rule, decision log 2026-11-17T02:18Z).

A runner records what its figures are computed from, and `Exports.write` puts four files in
its run directory:
  per_seed.csv               every metric per seed x family x condition
  predictions.parquet        per evaluation row: its position in the test split, its row id in
                             the splits file, the true label, and each model's class-1
                             probability and predicted label, per condition and seed
  crafted_inputs.npz         every attacked input, keyed seed|condition|source
  exports_provenance.json    thread variables, QGB_VQC_CHUNK, the random-stream names, the
                             tuned file's path and sha256, and any extra inputs the runner names
Conditions are free-form names chosen by the runner ("clean", "pgd_mlp@0.5", ...); a key
recorded twice fails loud.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from qgridbench.utils.run_tracking import REPO_ROOT

ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
    "QGB_VQC_CHUNK",
)


class Exports:
    def __init__(self, variant: str, tuned_file: str | None):
        self.variant = variant
        self.tuned_file = tuned_file
        self.metrics: list[dict] = []
        self.preds: dict[tuple, dict] = {}
        self.crafted: dict[str, np.ndarray] = {}
        self.streams: set[str] = set()
        self.extra: dict = {}

    def stream(self, name: str) -> str:
        """Record a random-stream name (pass-through, so it wraps the spawn_rng argument)."""
        self.streams.add(name)
        return name

    def metric(self, seed, family: str, condition: str, metric: str, value: float) -> None:
        self.metrics.append(
            {
                "seed": seed,
                "family": family,
                "condition": condition,
                "metric": metric,
                "value": float(value),
            }
        )

    def predictions(self, seed, condition: str, model: str, proba, y, test_pos) -> None:
        key = (seed, condition, model)
        if key in self.preds:
            raise KeyError(f"predictions recorded twice: {key}")
        proba, y, test_pos = np.asarray(proba), np.asarray(y), np.asarray(test_pos)
        if not (len(proba) == len(y) == len(test_pos)):
            raise ValueError(f"length mismatch for {key}")
        self.preds[key] = {"proba": proba, "y": y, "test_pos": test_pos}

    def crafted_input(self, seed, condition: str, source: str, X) -> None:
        key = f"{seed}|{condition}|{source}"
        if key in self.crafted:
            raise KeyError(f"crafted input recorded twice: {key}")
        self.crafted[key] = np.asarray(X, dtype=np.float64)

    def score(self, seed, sets: dict, models, proba_fn, y, test_pos, metric_fn, sources=None):
        """{condition: {model: metric}} over named input sets, recording every prediction, the
        macro-F1 per seed x model x condition, and every non-clean set as a crafted input
        (`sources` maps a condition to the model or procedure that crafted it)."""
        out = {}
        for cond, X in sets.items():
            if cond != "clean":
                self.crafted_input(seed, cond, (sources or {}).get(cond, "unnamed"), X)
            out[cond] = {}
            for m in models:
                p = proba_fn(m, X)
                self.predictions(seed, cond, m, p, y, test_pos)
                out[cond][m] = metric_fn(y, p)
                self.metric(seed, m, cond, "macro_f1", out[cond][m])
        return out

    def write(self, run_dir: Path) -> None:
        import pandas as pd

        from qgridbench.data.preprocess import load_processed

        _, _, splits = load_processed(self.variant)
        test_ids = splits["test"]
        pd.DataFrame(
            self.metrics, columns=["seed", "family", "condition", "metric", "value"]
        ).to_csv(run_dir / "per_seed.csv", index=False)
        frames = []
        # A runner may key a deterministic model by a label instead of a seed number
        # (run_ablations data_eff: "deterministic"); parquet needs one type per column, so
        # such a run writes every seed as text. All-integer runs keep integer seeds.
        seed_as_text = any(not isinstance(k[0], (int, np.integer)) for k in self.preds)
        for (seed, cond, model), r in self.preds.items():
            p = r["proba"]
            frames.append(
                pd.DataFrame(
                    {
                        "seed": str(seed) if seed_as_text else seed,
                        "condition": cond,
                        "model": model,
                        "test_pos": r["test_pos"],
                        "row_id": test_ids[r["test_pos"]],
                        "y_true": r["y"],
                        "proba_1": p[:, 1],
                        "label": p.argmax(1),
                    }
                )
            )
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(run_dir / "predictions.parquet")
        np.savez_compressed(run_dir / "crafted_inputs.npz", **self.crafted)
        tuned = None
        if self.tuned_file:
            data = (REPO_ROOT / self.tuned_file).read_bytes()
            tuned = {"path": self.tuned_file, "sha256": hashlib.sha256(data).hexdigest()}
        (run_dir / "exports_provenance.json").write_text(
            json.dumps(
                {
                    "env": {k: os.environ.get(k) for k in ENV_KEYS},
                    "random_streams": sorted(self.streams),
                    "tuned_file": tuned,
                    "splits_file": f"splits_{self.variant}.npz, test split",
                    "n_metric_rows": len(self.metrics),
                    "n_prediction_blocks": len(self.preds),
                    "n_crafted": len(self.crafted),
                    **self.extra,
                },
                indent=2,
                default=str,
            )
        )
