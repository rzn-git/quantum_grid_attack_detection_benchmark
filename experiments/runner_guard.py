"""Runner guard: a fitted predictor that is constant or non-finite on CLEAN data fails the run.

Imported by the runners; it has no command of its own and parses no arguments at import, so tests
import it directly. Test: `uv run pytest -q tests/test_runner_guard.py`.

Two void runs were caught only by a scipy warning and an impossible score (P9b 2026-08-23T21:04Z,
D2e 2026-08-24T01:01Z in the decision log): a predictor that returns one value for every row is a
defect, so it raises instead of warning.
"""

from __future__ import annotations

import sys

import numpy as np


class DegeneratePredictorError(RuntimeError):
    """A fitted predictor is constant or non-finite on clean rows."""


def _runner() -> str:
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    return spec.name if spec is not None else sys.argv[0]


def _counts(a: np.ndarray) -> dict:
    values, counts = np.unique(a, return_counts=True)
    return {v.item(): int(c) for v, c in zip(values, counts, strict=True)}


def assert_clean_predictor(model: str, y_true, outputs, predicted) -> None:
    """Raise if `model`'s outputs on CLEAN rows hold a NaN or inf, or if it predicts one value for
    every row while `y_true` holds more than one.

    `outputs` are the raw scores (predict_proba, a decision function, a per-unit predictor value);
    `predicted` is what is scored against `y_true` (the argmax class, the sign, the value itself).

    Call it ONLY at a clean-evaluation call site, on unperturbed rows. An attacked or perturbed
    input that collapses a model to one class is a measured end state the paper reports
    ("random-saturated", "disabled"), so the attack and perturbation paths never call this; the
    distinction is which call site runs it, not a threshold on the outputs.
    """
    outputs = np.asarray(outputs)
    predicted = np.asarray(predicted)
    y_true = np.asarray(y_true)
    n_nonfinite = int((~np.isfinite(outputs)).sum())
    constant = np.unique(predicted).size == 1 and np.unique(y_true).size > 1
    if n_nonfinite or constant:
        raise DegeneratePredictorError(
            f"runner {_runner()}: model {model!r} is degenerate on clean data: "
            f"{n_nonfinite} non-finite of {outputs.size} outputs; "
            f"predicted counts {_counts(predicted)} against true counts {_counts(y_true)}"
        )
