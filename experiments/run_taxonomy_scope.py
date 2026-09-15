"""N4 — is the end-state taxonomy well posed for models with an OUTPUT HEAD?

PRE-REGISTRATION (BACKLOG `Validate` row N4, winds review 2026-08-23T01:05Z; frozen before the
run). The paper's `disabled` label requires BOTH halves: output mass collapses AND macro-F1
returns to the constant-predictor floor. That criterion was built on kernels, where the decision
function IS a sum over kernel mass, so the two halves cannot come apart. For the VQC they do:
mass descent puts accuracy on the floor (|F1 - floor| = 0.0046 at 0.5 sigma) while |<Z>| stays at
0.325, because the VQC's head carries a trainable scale and bias and a reduced-but-nonzero
expectation already fails to separate the classes. The VQC's strict `partial` label may therefore
be an artifact of applying a kernel-shaped definition to a head-carrying model.

METHOD. Assemble every (mass, F1 - floor) cell this project has archived, label each twice, and
count where the two rules disagree:
  RULE A (published, `qgridbench.eval.tables._regime`) disabled iff mass <= 0.05 AND
          |F1-floor| <= 0.05; inverted iff F1-floor <= -0.2; else partial
  RULE B (accuracy half alone)                          disabled iff |F1-floor| <= 0.05;
          inverted iff F1-floor <= -0.2; else partial
Rule B is Rule A with the mass condition dropped — it is the label a reader would assign from
accuracy alone, and it is the only half that is meaningful for a model whose head can rescale a
surviving readout.

DECISION RULE, fixed now.
  every disagreement is a VQC cell -> the taxonomy is SOUND and needs only a scope sentence
                                     ("this classifies kernel-like models")
  any kernel or RBF cell disagrees -> the CRITERION itself needs revising, which is a larger
                                     edit than a scope sentence, and the paper must say which
                                     half it means where the two come apart

CONTROLS.
  positive (blocking)  Rule A re-derived here must reproduce the published regime table's label
                       for all 13 of its rows (the same control N1 used, and the same
                       right-hand parse — that CSV writes unquoted commas inside its labels)
  negative             a cell where the two rules agree by construction (|F1-floor| <= 0.05 AND
                       mass <= 0.05) cannot count as evidence either way and is reported
                       separately from the disagreements

Analysis only: no model is fit, no attack is crafted.

Run:  python -m experiments.run_taxonomy_scope
"""

from __future__ import annotations

import json
from collections import Counter

from qgridbench.eval.tables import _regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger

log = get_logger(__name__)

PUBLISHED_TABLE = "paper/tables/regime_binary_q8.csv"  # READ ONLY
SOURCES = {
    "p8a": ("results/runs/20260814T111131Z_whitebox_qsvm_binary_q8", "p8a_lengthscale_control"),
    "p7b": ("results/runs/20260814T100533Z_whitebox_qsvm_binary_q8", "p7b_kernel_exhaustion"),
    "p11c": (
        "results/runs/20260814T200349Z_whitebox_qsvm_binary_q8",
        "p11c_objective_and_regularization",
    ),
    "e13": ("results/runs/20260822T172804Z_vqc_objective_axis_binary_q8", None),
    "e15": ("results/runs/20260822T173334Z_kernel_width_exhaustion_binary", None),
}


def rule_b(f1_minus_floor: float) -> str:
    """The accuracy half alone — Rule A with the mass condition dropped."""
    if abs(f1_minus_floor) <= 0.05:
        return "disabled"
    if f1_minus_floor <= -0.2:
        return "inverted"
    return "partial"


def load(run: str, key: str | None):
    d = json.loads((REPO_ROOT / run / "results.json").read_text())
    return d[key] if key else d


def main() -> None:
    cfg = {
        "sources": {k: v[0] for k, v in SOURCES.items()},
        "rule_a": "qgridbench.eval.tables._regime (published, two-part)",
        "rule_b": "accuracy half alone (mass condition dropped)",
        "note": "analysis only — no model is fit and no attack is crafted",
    }

    with RunDir("taxonomy_scope_binary_q8", config=cfg, seeds=[0]) as run:
        cells: list[dict] = []

        def add(family, source, key, mass, f1):
            cells.append(
                {
                    "family": family,
                    "source": source,
                    "cell": key,
                    "mass_ratio": float(mass),
                    "f1_minus_floor": float(f1),
                    "rule_a": _regime(float(mass), float(f1)),
                    "rule_b": rule_b(float(f1)),
                }
            )

        # P8a — classical RBF, bandwidth sweep, every radius
        p8a = load(*SOURCES["p8a"])
        for w, v in p8a["per_width"].items():
            for eps, c in v["curve"].items():
                if eps == "0.0":
                    continue
                add("rbf", "p8a", f"gamma_x{w}@{eps}", c["kernel_mass_ratio"], c["f1_minus_floor"])

        # p7b — three families under the loss objective, every radius
        p7b = load(*SOURCES["p7b"])["per_model"]
        for m, v in p7b.items():
            radii = v.get("curve", v)
            for eps, c in radii.items():
                if eps == "0.0":
                    continue
                add(m, "p7b", f"{m}@{eps}", c["kernel_mass_ratio"], c["f1_minus_floor"])

        # P11c part B — the fidelity kernel under MASS DESCENT, per regularization
        for C, v in load(*SOURCES["p11c"])["part_b_regularization"].items():
            add(
                "qsvm",
                "p11c_mass_descent",
                f"C={float(C):g}",
                v["mean"]["mass_ratio"]["mean"],
                v["mean"]["f1_minus_floor"]["mean"],
            )

        # E13 — the VQC under both objectives, every radius
        for eps, c in load(*SOURCES["e13"])["curve"].items():
            for obj in ("loss_objective", "mass_descent_objective"):
                add(
                    "vqc",
                    f"e13_{obj}",
                    f"vqc@{eps}",
                    c[obj]["mass_ratio"],
                    c[obj]["f1_minus_floor"],
                )

        # E15 — the fidelity kernel across widths
        for n, c in load(*SOURCES["e15"])["cells"].items():
            for eps, x in c["curve"].items():
                add("qsvm", "e15", f"q{n}@{eps}", x["mass_ratio"], x["f1_minus_floor"])

        # ---- control: Rule A must reproduce the published table --------------------
        lines = (REPO_ROOT / PUBLISHED_TABLE).read_text().splitlines()[1:]
        mismatches = []
        for line in [x for x in lines if x.strip()]:
            f = line.split(",")
            derived = _regime(float(f[-3]), float(f[-2]))
            if derived != f[-1].strip():
                mismatches.append(
                    {
                        "row": ",".join(f[:-4]).strip(),
                        "published": f[-1].strip(),
                        "derived": derived,
                    }
                )
        controls = {
            "positive_rows_checked": len([x for x in lines if x.strip()]),
            "positive_mismatches": mismatches,
            "positive_pass": bool(not mismatches),
        }

        # ---- disagreement analysis ---------------------------------------------------
        disagree = [c for c in cells if c["rule_a"] != c["rule_b"]]
        agree_trivially = [c for c in cells if c["rule_a"] == c["rule_b"] == "disabled"]
        by_family = Counter(c["family"] for c in disagree)
        non_vqc = sorted({c["family"] for c in disagree} - {"vqc"})
        controls["negative_trivially_agreeing_cells"] = len(agree_trivially)
        controls["negative_note"] = (
            "cells where both rules say disabled agree by construction and are counted "
            "separately from the disagreements"
        )
        controls["pass"] = controls["positive_pass"]
        log.info("CONTROLS: %s", json.dumps(controls, indent=2))

        # ---- persistence: is an at-floor reading TRANSIT or a PARKED end state? -------
        # A model crossing from good to inverted passes THROUGH its floor, so a single
        # at-floor radius is not an end state. Count, per configuration series, how many of
        # its radii Rule B calls disabled — an isolated hit is transit, a run of them is an
        # end state the mass condition is rejecting.
        series: dict[tuple[str, str], list] = {}
        for c in cells:
            if "@" not in c["cell"]:
                continue
            cfgname, eps = c["cell"].rsplit("@", 1)
            series.setdefault((c["source"], cfgname), []).append((float(eps), c))
        persistence = {}
        for (src, cfgname), pts in series.items():
            pts.sort()
            hits = [e for e, c in pts if c["rule_b"] == "disabled" and c["rule_a"] != "disabled"]
            if hits:
                persistence[f"{src}|{cfgname}"] = {
                    "n_radii": len(pts),
                    "n_disagreeing": len(hits),
                    "radii": hits,
                    "fraction": len(hits) / len(pts),
                    "sustained": bool(len(hits) >= 3),
                    "family": pts[0][1]["family"],
                }
        sustained = {k: v for k, v in persistence.items() if v["sustained"]}
        transit = {k: v for k, v in persistence.items() if not v["sustained"]}

        verdict = {
            "controls": controls,
            "persistence": persistence,
            "sustained_series": sorted(sustained),
            "transit_series": sorted(transit),
            "families_with_sustained_disagreement": sorted(
                {v["family"] for v in sustained.values()}
            ),
            "n_cells": len(cells),
            "n_disagreements": len(disagree),
            "disagreements_by_family": dict(by_family),
            "non_vqc_families_disagreeing": non_vqc,
            "disagreement_cells": disagree,
            "branch": (
                "criterion_needs_revising" if non_vqc else "taxonomy_sound_needs_scope_sentence"
            ),
            # refined reading: the mass condition is doing NECESSARY work if the classical
            # disagreements are transit only, and over-rejecting if any family shows a
            # SUSTAINED at-floor plateau that mass alone vetoes
            "refined_branch": (
                "mass_guard_necessary_but_needs_persistence_qualifier"
                if sustained and all(v["family"] == "vqc" for v in sustained.values())
                else (
                    "mass_guard_over_rejects_multiple_families"
                    if sustained
                    else "mass_guard_necessary_all_disagreements_are_transit"
                )
            ),
            "rule": (
                "every disagreement is a VQC cell -> scope sentence; any kernel/RBF cell "
                "disagrees -> the criterion itself needs revising"
            ),
        }
        run.write_json("results.json", verdict)
        run.metrics = {"n_disagreements": len(disagree), "n_cells": len(cells)}
        log.info(
            "VERDICT %s | %d/%d cells disagree | by family %s",
            verdict["branch"],
            len(disagree),
            len(cells),
            dict(by_family),
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
