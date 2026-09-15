"""Command-line entry point for the leakage audit.

Two subcommands:

  neighbours  — the neighbour audit on a prepared train/test pair (CSV or .npy/.npz).
  paired      — the paired-protocol runner on one CSV with a label column and a group
                column; the estimator is any importable scikit-learn-compatible class.

Both print a JSON report to stdout. Examples in the README.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tools.leakage_audit.audit import neighbour_audit, paired_protocol


def _load_matrix(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".npz":
        z = np.load(p)
        if "X" in z:
            return z["X"]
        return z[z.files[0]]
    df = pd.read_csv(p)
    return df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)


def _build_estimator(spec: str, params_json: str):
    mod_path, _, cls_name = spec.rpartition(".")
    if not mod_path:
        raise SystemExit(f"--model must be a full import path, got '{spec}'")
    cls = getattr(importlib.import_module(mod_path), cls_name)
    return cls(**json.loads(params_json))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="leakage_audit", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("neighbours", help="neighbour audit on a train/test pair")
    n.add_argument(
        "--train", required=True, help="train features: CSV (numeric cols), .npy or .npz"
    )
    n.add_argument("--test", required=True, help="test features: same formats")
    n.add_argument("--threshold", type=float, default=0.1)
    n.add_argument(
        "--no-standardize",
        action="store_true",
        help="inputs are already standardized; skip the train-fitted z-scoring",
    )

    p = sub.add_parser("paired", help="row-level vs whole-group protocol on one CSV")
    p.add_argument("--data", required=True, help="CSV with features + label + group columns")
    p.add_argument("--label-col", required=True)
    p.add_argument("--group-col", required=True)
    p.add_argument(
        "--model",
        default="sklearn.ensemble.RandomForestClassifier",
        help="import path of a scikit-learn-compatible classifier",
    )
    p.add_argument(
        "--model-params",
        default='{"random_state": 0}',
        help="constructor kwargs as JSON (pass a random_state for determinism)",
    )
    p.add_argument("--partitions", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args(argv)

    if args.cmd == "neighbours":
        report = neighbour_audit(
            _load_matrix(args.train),
            _load_matrix(args.test),
            threshold=args.threshold,
            standardize=not args.no_standardize,
        )
    else:
        df = pd.read_csv(args.data)
        for col in (args.label_col, args.group_col):
            if col not in df.columns:
                raise SystemExit(f"column '{col}' not in {args.data}")
        y = df[args.label_col].to_numpy()
        groups = df[args.group_col].to_numpy()
        X = (
            df.drop(columns=[args.label_col, args.group_col])
            .select_dtypes(include=[np.number])
            .to_numpy(dtype=np.float64)
        )
        report = paired_protocol(
            _build_estimator(args.model, args.model_params),
            X,
            y,
            groups,
            val_size=args.val_size,
            test_size=args.test_size,
            n_partitions=args.partitions,
            seed=args.seed,
            threshold=args.threshold,
        )

    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
