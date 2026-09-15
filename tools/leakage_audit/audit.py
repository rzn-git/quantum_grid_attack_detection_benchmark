"""Leakage audit for group-structured tabular data: neighbour audit + paired-protocol runner.

Two questions, two functions:

  * `neighbour_audit` — what fraction of test rows have a train neighbour closer than a
    threshold in standardized feature space? A high fraction means a row-level split has put
    near-duplicates on both sides of the train/test boundary, and the score measures
    memorization of those near-duplicates rather than generalization.
  * `paired_protocol` — fit and score ONE scikit-learn-compatible estimator under both the
    row-level and the whole-group protocols with identical transforms, and report both
    numbers, the delta, and the neighbour audit that explains it.

Depends only on numpy, pandas (CLI only) and scikit-learn — no project code — so the module
can be pointed at any dataset and any estimator. The worked example under `examples/`
reproduces the QuantumGridBench headline (row-level 0.905 vs whole-file 0.594 macro-F1 on the
MSU/ORNL power system attack data, arXiv:2608.15617) so the tool can be verified against a
published result before it is trusted on new data.

Protocol notes (they are the whole point):
  * Every fitted transform — the audit's standardizer and the model transform — is fit on the
    TRAIN fold only and applied frozen to held-out rows.
  * Group splits hold out WHOLE groups (source files, sessions, patients, captures...);
    disjointness of the group sets is asserted, never assumed.
  * A validation fold (default 20%) is carved out and EXCLUDED from training in both
    protocols, mirroring a tuning protocol's holdout; set val_size=0 to train on everything
    outside the test fold.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _standardizer() -> Pipeline:
    """Median-impute + standardize: the audit's reference feature space."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


def neighbour_audit(
    X_train: np.ndarray,
    X_test: np.ndarray,
    threshold: float = 0.1,
    standardize: bool = True,
) -> dict:
    """Fraction of test rows with a train neighbour closer than `threshold`.

    Distances are Euclidean in standardized feature space (median-impute + z-score, fit on
    train only). Pass standardize=False if the inputs are already standardized.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    if standardize:
        sc = _standardizer().fit(X_train)
        X_train, X_test = sc.transform(X_train), sc.transform(X_test)
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X_train)
    d = nn.kneighbors(X_test)[0].ravel()
    return {
        "threshold": float(threshold),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "frac_within_threshold": float((d < threshold).mean()),
        "nn_distance_median": float(np.median(d)),
        "nn_distance_p05": float(np.quantile(d, 0.05)),
        "nn_distance_p95": float(np.quantile(d, 0.95)),
    }


def _assert_group_disjoint(splits: dict[str, np.ndarray], groups: np.ndarray) -> None:
    """The rule is asserted in code, never assumed."""
    gs = {k: set(np.unique(groups[idx]).tolist()) for k, idx in splits.items() if len(idx)}
    keys = sorted(gs)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            overlap = gs[a] & gs[b]
            if overlap:
                raise AssertionError(
                    f"group split is not group-disjoint: '{a}' and '{b}' share {sorted(overlap)}"
                )


def _row_partition(y: np.ndarray, val_size: float, test_size: float, seed: int) -> dict:
    idx = np.arange(len(y))
    trval, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    if val_size > 0:
        tr, va = train_test_split(
            trval,
            test_size=val_size / (1.0 - test_size),
            stratify=y[trval],
            random_state=seed,
        )
    else:
        tr, va = trval, idx[:0]
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}


def _group_partition(
    groups: np.ndarray, rng: np.random.Generator, val_size: float, test_size: float
) -> tuple[dict, list]:
    """Hold out whole groups: shuffle the group ids, slice train/val/test by fraction."""
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n = len(uniq)
    cut1, cut2 = int((1.0 - val_size - test_size) * n), int((1.0 - test_size) * n)
    tr_g, va_g, te_g = uniq[:cut1], uniq[cut1:cut2], uniq[cut2:]
    idx = np.arange(len(groups))
    splits = {
        "train": idx[np.isin(groups, tr_g)],
        "val": idx[np.isin(groups, va_g)],
        "test": idx[np.isin(groups, te_g)],
    }
    _assert_group_disjoint(splits, groups)
    return splits, sorted(te_g.tolist())


def _fit_and_score(
    estimator,
    transform,
    X: np.ndarray,
    y: np.ndarray,
    splits: dict[str, np.ndarray],
    metric: Callable,
) -> float:
    tf = clone(transform)
    Z_train = tf.fit_transform(X[splits["train"]])  # fit on train only, frozen elsewhere
    est = clone(estimator)
    est.fit(Z_train, y[splits["train"]])
    return float(metric(y[splits["test"]], est.predict(tf.transform(X[splits["test"]]))))


def paired_protocol(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    transform=None,
    metric: Callable | None = None,
    val_size: float = 0.2,
    test_size: float = 0.2,
    n_partitions: int = 5,
    seed: int = 0,
    threshold: float = 0.1,
    row_split_indices: dict[str, np.ndarray] | None = None,
) -> dict:
    """Score `estimator` under the row-level and whole-group protocols; report the delta.

    Arguments:
        estimator: any scikit-learn-compatible classifier (cloned per fit).
        X, y, groups: features, labels, and one group id per row (the unit whose leakage is
            being tested — source file, session, patient...).
        transform: sklearn transformer fit on train only per arm (default: median-impute +
            standardize). Cloned per fit.
        metric: metric(y_true, y_pred) -> float; default macro-F1.
        val_size / test_size: fold fractions (of rows for the row arm, of groups for the
            group arm). The val fold is excluded from training.
        n_partitions: group-arm repetitions, each drawing a fresh group shuffle from one
            seeded stream; mean and std reported.
        seed: seeds the row split and the group-shuffle stream.
        threshold: neighbour-audit distance threshold.
        row_split_indices: optional predefined {'train','val','test'} row indices for the
            row arm (e.g. a project's shipped fixed split) — used instead of a fresh
            stratified draw, so a published number can be reproduced exactly.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if not (len(X) == len(y) == len(groups)):
        raise ValueError(f"length mismatch: X {len(X)}, y {len(y)}, groups {len(groups)}")
    transform = transform if transform is not None else _standardizer()
    metric = metric if metric is not None else lambda a, b: f1_score(a, b, average="macro")

    if row_split_indices is not None:
        row_splits = {k: np.asarray(v) for k, v in row_split_indices.items()}
    else:
        row_splits = _row_partition(y, val_size, test_size, seed)
    row_score = _fit_and_score(estimator, transform, X, y, row_splits, metric)
    row_nn = neighbour_audit(X[row_splits["train"]], X[row_splits["test"]], threshold)

    rng = np.random.default_rng(seed)
    partitions = []
    for p in range(n_partitions):
        splits, held = _group_partition(groups, rng, val_size, test_size)
        score = _fit_and_score(estimator, transform, X, y, splits, metric)
        nn = neighbour_audit(X[splits["train"]], X[splits["test"]], threshold)
        partitions.append(
            {
                "partition": p,
                "held_out_groups": held,
                "score": score,
                "nn_frac_within_threshold": nn["frac_within_threshold"],
            }
        )

    g_scores = np.array([p["score"] for p in partitions])
    g_nn = np.array([p["nn_frac_within_threshold"] for p in partitions])
    return {
        "row_level": {
            "score": row_score,
            "nn_frac_within_threshold": row_nn["frac_within_threshold"],
            "predefined_split": row_split_indices is not None,
        },
        "whole_group": {
            "score_mean": float(g_scores.mean()),
            "score_std": float(g_scores.std(ddof=1)) if len(g_scores) > 1 else 0.0,
            "nn_frac_within_threshold_mean": float(g_nn.mean()),
            "partitions": partitions,
        },
        "delta_row_minus_group": float(row_score - g_scores.mean()),
        "config": {
            "threshold": threshold,
            "val_size": val_size,
            "test_size": test_size,
            "n_partitions": n_partitions,
            "seed": seed,
        },
    }
