"""R4-12 checklist items 7 and 8 on the gas pipeline (the model-selection protocol).

Pre-registration: decision log 2026-11-16T11:43Z, "PRE-RUN AMENDMENT (R4-12", ruling 6 (H-K).
Reads `run_whitebox_qsvm --variant gas_binary` artifacts:
  item 7, kernel length scale (P8a): along the RBF-SVM gamma axis (C fixed), the 2-sigma end
          state of each width is `qgridbench.eval.tables._regime` (the paper's rule). The item
          REPRODUCES if one width ends disabled and another inverted while their clean
          macro-F1 differ by less than 0.08.
  item 8, regularization C (P11c part B): the QSVM's C sweep at its tuned bandwidth. The item
          REPRODUCES if the failure mode changes across the non-degenerate C cells (some
          disabled, some not).

Run:  uv run python -m experiments.run_gas_selection_items --p8a-run results/runs/<...> \
          --p11c-run results/runs/<...>
"""

from __future__ import annotations

import argparse
import itertools
import json

from qgridbench.eval.tables import _regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger

log = get_logger(__name__)

CLEAN_SPREAD = 0.08


def _load(rd: str) -> dict:
    path = REPO_ROOT / rd
    status = json.loads((path / "metrics.json").read_text())["status"]
    if status != "ok":
        raise RuntimeError(f"{rd} finished with status {status}")
    return json.loads((path / "results.json").read_text())


def item7(p8a: dict) -> dict:
    floor = p8a["constant_predictor_floor"]
    cells = {}
    for mult, w in p8a["per_width"].items():
        curve = w["curve"]
        top = curve[str(max(float(k) for k in curve))]
        cells[mult] = {
            "gamma": w["gamma"],
            "clean_f1": curve["0.0"]["macro_f1"],
            "end_state": _regime(top["kernel_mass_ratio"], top["f1_minus_floor"]),
            "mass_ratio": top["kernel_mass_ratio"],
            "f1_minus_floor": top["f1_minus_floor"],
        }
    pairs = [
        (a, b)
        for a, b in itertools.permutations(cells, 2)
        if cells[a]["end_state"] == "disabled"
        and cells[b]["end_state"] == "inverted"
        and abs(cells[a]["clean_f1"] - cells[b]["clean_f1"]) < CLEAN_SPREAD
    ]
    return {"floor": floor, "cells": cells, "qualifying_pairs": pairs, "reproduces": bool(pairs)}


def item8(p11c: dict) -> dict:
    part_b = p11c["part_b_regularization"]
    live = {c: v["disabled"] for c, v in part_b.items() if not v["degenerate"]}
    return {
        "disabled_by_C": live,
        "degenerate_C": [c for c, v in part_b.items() if v["degenerate"]],
        "reproduces": bool(len(set(live.values())) > 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--p8a-run", required=True)
    ap.add_argument("--p11c-run", required=True)
    args = ap.parse_args()
    i7 = item7(_load(args.p8a_run)["p8a_lengthscale_control"])
    i8 = item8(_load(args.p11c_run)["p11c_objective_and_regularization"])
    with RunDir("gas_selection_items_gas_binary", config=vars(args)) as run:
        run.write_json(
            "results.json", {"item7_kernel_length_scale": i7, "item8_regularization": i8}
        )
        run.metrics.update({"item7": i7["reproduces"], "item8": i8["reproduces"]})
        log.info("item 7 %s | item 8 %s", json.dumps(i7, default=str), json.dumps(i8))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
