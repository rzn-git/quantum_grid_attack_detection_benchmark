"""The tuned-parameter file and run-name tag flags shared by the robustness runners.

R4-24 (decision log 2026-11-16T13:33Z) re-runs every paper figure that rests on the depth-4 VQC
at Table I's VQC configuration. Each runner it touches takes:
  --best-params FILE   tuned-parameter JSON relative to the repo root; default: the variant's own
                       file (`protocol.best_params_path`), so the default run is bit-identical
  --run-tag TAG        suffix for the run directory name; default: none (the published name).
                       `experiments/make_report.py` selects paper sources by run-name glob, so a
                       tagged run never replaces a published source by accident.
`with_vqc_from` takes the VQC keys from a second tuned file (the 12-qubit run keeps the q8
file's other families, as its archive did, and takes its VQC from the 12-qubit study).
"""

from __future__ import annotations

import argparse
import json

from qgridbench.protocol import best_params_path
from qgridbench.utils.run_tracking import REPO_ROOT

VQC_KEYS = ("vqc_depth", "vqc_lr", "vqc_batch", "vqc_epochs")


def add_tuned_file_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--best-params",
        default=None,
        help="tuned-parameter JSON relative to the repo root; default: the variant's own file "
        "(protocol.best_params_path)",
    )
    ap.add_argument(
        "--run-tag",
        default=None,
        help="suffix for the run directory name (default: the published name)",
    )


def load_tuned_file(arg: str | None, variant: str, qubits: int) -> tuple[str, dict]:
    """(path relative to the repo root, parsed file); the default is best_params_path's file."""
    rel = arg or best_params_path(variant, qubits).relative_to(REPO_ROOT).as_posix()
    return rel, json.loads((REPO_ROOT / rel).read_text())


def run_name(base: str, tag: str | None) -> str:
    return f"{base}_{tag}" if tag else base


def with_vqc_from(params: dict, vqc_source: dict) -> dict:
    """A copy of `params` whose VQC keys come from `vqc_source`; every key must exist in both."""
    missing = [k for k in VQC_KEYS if k not in params or k not in vqc_source]
    if missing:
        raise KeyError(f"VQC keys absent from a tuned file: {missing}")
    return {**params, **{k: vqc_source[k] for k in VQC_KEYS}}
