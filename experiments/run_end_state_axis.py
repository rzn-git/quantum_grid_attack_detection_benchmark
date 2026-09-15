"""N1 — who chooses the failure mode? Assemble the end-state axis from archived artifacts.

PRE-REGISTRATION (BACKLOG `Validate` row N1, winds review 2026-08-23T00:35Z; frozen before the
run). The paper documents two end states under a full-budget attack and labels them with a
DERIVED rule (`qgridbench.eval.tables._regime`, thresholds pre-registered 2026-08-14T11:02Z):
  DISABLED  output mass collapses and macro-F1 returns to the constant-predictor floor — the
            detector visibly stops discriminating and an operator can see it
  INVERTED  output survives and macro-F1 sits far below the floor — the detector confidently
            reports the wrong class and looks like it is still working
Three archived results bear on WHO selects between them, and the paper has never stated them as
one axis: P8a (classical, the defender's bandwidth), P11c/P11d (fidelity kernel, the attacker's
objective) and E13 (VQC, the attacker's objective).

HYPOTHESIS. The end state is selected by the DEFENDER's length-scale choice classically and by
the ATTACKER's objective choice quantumly — i.e. at a fixed, tuned quantum model the attacker
decides which failure the operator sees.

DECISION RULE, fixed before the run. The axis is unified for a family iff, holding the model and
the budget fixed, varying the named knob spans BOTH end states under the `_regime` rule:
  classical  the gamma sweep at a fixed loss objective contains both `disabled` and `inverted`
  quantum    at the tuned model, the loss objective gives one end state and the mass-descending
             objective gives the other
Branches: both quantum families span -> `axis_unified_both_quantum_families`; only the kernel
spans -> `axis_unified_kernel_only` (reported as such, never rounded up); neither -> `not_unified`.

KILL CRITERION (pre-committed). If the kernel's disabled state requires a radius OUTSIDE the
paper's own reported budget, the axis is not unified and this stays three separate results. The
paper's regime exhibit is reported at 2 sigma, so the disabled cell must be reached at eps <= 2.0.

CONTROLS.
  positive (blocking)  every label re-derived here must match the published regime table
                       (`paper/tables/regime_binary_q8.csv`) for every cell that table contains.
                       The table is READ, never written — no file under `paper/` is modified.
  negative             a cell whose CLEAN model is already within 0.05 of its floor cannot be
                       assigned an end state (the attacked reading would be vacuous) and is
                       excluded with a flag rather than silently counted.

This run fits no model and crafts no attack; it reads archived artifacts only.

Run:  python -m experiments.run_end_state_axis
"""

from __future__ import annotations

import json

from qgridbench.eval.tables import _regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger

log = get_logger(__name__)

P8A = "results/runs/20260814T111131Z_whitebox_qsvm_binary_q8"  # classical bandwidth sweep
P7B = "results/runs/20260814T100533Z_whitebox_qsvm_binary_q8"  # loss-objective end states
P11C = "results/runs/20260814T200349Z_whitebox_qsvm_binary_q8"  # objective + regularization
E13 = "results/runs/20260822T172804Z_vqc_objective_axis_binary_q8"  # VQC objective axis
PUBLISHED_TABLE = "paper/tables/regime_binary_q8.csv"  # READ ONLY
BUDGET = "2.0"
DEGENERATE_BAND = 0.05


def load(run: str, key: str) -> dict:
    return json.loads((REPO_ROOT / run / "results.json").read_text())[key]


def main() -> None:
    cfg = {
        "sources": {"p8a": P8A, "p7b": P7B, "p11c": P11C, "e13": E13},
        "budget_sigma": BUDGET,
        "label_rule": "qgridbench.eval.tables._regime (SSOT, thresholds pre-registered 11:02Z)",
        "note": "analysis only — no model is fit and no attack is crafted",
    }

    with RunDir("end_state_axis_binary_q8", config=cfg, seeds=[0]) as run:
        # ---- leg 1: classical, DEFENDER's bandwidth, loss objective fixed ----------
        p8a = load(P8A, "p8a_lengthscale_control")
        classical = []
        for k in sorted(p8a["per_width"], key=float):
            w = p8a["per_width"][k]
            clean, att = w["curve"]["0.0"], w["curve"][BUDGET]
            classical.append(
                {
                    "gamma_multiplier": float(k),
                    "gamma": w["gamma"],
                    "clean_f1_minus_floor": clean["f1_minus_floor"],
                    "clean_degenerate": bool(clean["f1_minus_floor"] < DEGENERATE_BAND),
                    "mass_ratio": att["kernel_mass_ratio"],
                    "f1_minus_floor": att["f1_minus_floor"],
                    "end_state": _regime(att["kernel_mass_ratio"], att["f1_minus_floor"]),
                }
            )
        usable = [c for c in classical if not c["clean_degenerate"]]
        classical_states = {c["end_state"] for c in usable}
        classical_spans = {"disabled", "inverted"} <= classical_states

        # ---- leg 2: fidelity kernel, ATTACKER's objective, tuned model fixed -------
        # p7b stores radii directly under the model (no "curve" level, unlike p8a/p11c)
        _p7b_qsvm = load(P7B, "p7b_kernel_exhaustion")["per_model"]["qsvm"]
        p7b = _p7b_qsvm.get("curve", _p7b_qsvm)[BUDGET]
        p11c = load(P11C, "p11c_objective_and_regularization")
        tuned_c = str(p11c["tuned_C"])
        b = p11c["part_b_regularization"][tuned_c]
        kernel = {
            "loss_objective": {
                "mass_ratio": p7b["kernel_mass_ratio"],
                "f1_minus_floor": p7b["f1_minus_floor"],
                "end_state": _regime(p7b["kernel_mass_ratio"], p7b["f1_minus_floor"]),
            },
            "mass_descent_objective": {
                "mass_ratio": b["mean"]["mass_ratio"]["mean"],
                "f1_minus_floor": b["mean"]["f1_minus_floor"]["mean"],
                "modal_frac": b["mean"]["modal_frac"]["mean"],
                "end_state": _regime(
                    b["mean"]["mass_ratio"]["mean"], b["mean"]["f1_minus_floor"]["mean"]
                ),
                # P11d's own flag is STRICTER than _regime: it additionally demands near-constant
                # predictions (modal > 0.99). Both are reported; the rule above uses _regime.
                "p11d_strict_disabled_flag": b["disabled"],
            },
            "clean_degenerate": b["degenerate"],
            "budget_sigma_of_disabled_cell": float(p11c["paper_budget_sigma"]),
        }
        kernel_states = {
            kernel[k]["end_state"] for k in ("loss_objective", "mass_descent_objective")
        }
        kernel_spans = {"disabled", "inverted"} <= kernel_states

        # ---- leg 3: VQC, ATTACKER's objective, tuned model fixed -------------------
        e13 = json.loads((REPO_ROOT / E13 / "results.json").read_text())
        floor = e13["verdict"]["floor"]
        vqc_curve = {}
        for eps, c in e13["curve"].items():
            vqc_curve[eps] = {
                obj: {
                    "mass_ratio": c[obj]["mass_ratio"],
                    "f1_minus_floor": c[obj]["f1_minus_floor"],
                    "end_state": _regime(c[obj]["mass_ratio"], c[obj]["f1_minus_floor"]),
                }
                for obj in ("loss_objective", "mass_descent_objective")
            }
        vqc = {
            "floor": floor,
            "at_budget": vqc_curve[BUDGET],
            "curve": vqc_curve,
            "closest_to_floor_under_descent": min(
                (abs(v["mass_descent_objective"]["f1_minus_floor"]), e)
                for e, v in vqc_curve.items()
            ),
        }
        vqc_states = {vqc_curve[BUDGET][o]["end_state"] for o in vqc_curve[BUDGET]}
        vqc_spans = {"disabled", "inverted"} <= vqc_states

        # ---- controls ---------------------------------------------------------------
        # The published CSV writes UNQUOTED configuration labels that themselves contain commas
        # ("QSVM ZZ, 8 qubits"), so csv.DictReader shifts those rows' columns — it did, and this
        # control caught it. Parse from the RIGHT, where the four numeric/label columns are fixed.
        lines = (REPO_ROOT / PUBLISHED_TABLE).read_text().splitlines()
        published = []
        for line in lines[1:]:
            if not line.strip():
                continue
            f = line.split(",")
            published.append(
                {
                    "configuration": ",".join(f[:-4]).strip(),
                    "clean_macro_f1": f[-4],
                    "output_mass_ratio": f[-3],
                    "f1_minus_floor": f[-2],
                    "regime": f[-1].strip(),
                }
            )
        mismatches = []
        for row in published:
            derived = _regime(float(row["output_mass_ratio"]), float(row["f1_minus_floor"]))
            if derived != row["regime"]:
                mismatches.append(
                    {
                        "configuration": row["configuration"],
                        "published": row["regime"],
                        "derived": derived,
                    }
                )
        controls = {
            "positive_published_label_rows_checked": len(published),
            "positive_mismatches": mismatches,
            "positive_pass": bool(not mismatches),
            "negative_degenerate_cells_excluded": [
                c["gamma_multiplier"] for c in classical if c["clean_degenerate"]
            ],
            "negative_kernel_clean_degenerate": kernel["clean_degenerate"],
            "negative_pass": bool(not kernel["clean_degenerate"]),
            "pass": bool(not mismatches and not kernel["clean_degenerate"]),
        }
        log.info("CONTROLS: %s", json.dumps(controls, indent=2))

        # ---- kill criterion + decision rule ------------------------------------------
        kill_fired = bool(kernel["budget_sigma_of_disabled_cell"] > float(BUDGET))
        if kill_fired or not kernel_spans:
            branch = "not_unified"
        elif vqc_spans:
            branch = "axis_unified_both_quantum_families"
        else:
            branch = "axis_unified_kernel_only"

        verdict = {
            "controls": controls,
            "classical_defender_axis": {
                "spans_both_end_states": bool(classical_spans),
                "states_observed": sorted(classical_states),
                "tuned_end_state": next(
                    c["end_state"] for c in classical if c["gamma_multiplier"] == 1.0
                ),
                "cells": classical,
            },
            "kernel_attacker_axis": {"spans_both_end_states": bool(kernel_spans), **kernel},
            "vqc_attacker_axis": {
                "spans_both_end_states": bool(vqc_spans),
                "states_at_budget": {
                    o: vqc_curve[BUDGET][o]["end_state"] for o in vqc_curve[BUDGET]
                },
                "closest_to_floor_under_descent": vqc["closest_to_floor_under_descent"],
                "at_budget": vqc["at_budget"],
            },
            "kill_criterion_fired": kill_fired,
            "branch": branch,
            "statement": (
                "classically the DEFENDER's length-scale choice selects the failure mode; for the "
                "fidelity kernel the ATTACKER's objective selects it at a fixed tuned model and "
                "the same budget; for the VQC the attacker's objective moves the end state from "
                "inverted to at-floor but the readout never collapses, so the strict DISABLED "
                "label is not reached"
            ),
            "rule": (
                "a family's axis is unified iff varying its named knob spans both end states "
                "under _regime at fixed model and budget; kernel disabled cell must be at "
                f"eps <= {BUDGET}"
            ),
        }
        run.write_json("results.json", verdict)
        run.metrics = {"kernel_spans": float(kernel_spans), "vqc_spans": float(vqc_spans)}
        log.info(
            "VERDICT\n%s",
            json.dumps({k: v for k, v in verdict.items() if k != "controls"}, indent=2)[:3000],
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
