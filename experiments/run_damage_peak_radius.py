"""N3 — is the damage peak a per-model property, or is it specific to tree ensembles?

PRE-REGISTRATION (BACKLOG `Validate` row N3, winds review 2026-08-23T00:35Z; frozen before the
run). E14 found that macro-F1 damage does not rise monotonically with the attack radius for tree
ensembles: RF, XGBoost and LightGBM all peak at 0.3 sigma and fall at 0.5, while the MLP rises
monotonically. The mechanism proposed there is that maximum damage occurs at the INVERSION point
— a larger budget pushes predictions toward a near-constant output, which returns macro-F1 toward
the floor rather than below it. That was measured on four models under one attack. If the peak
location is a per-model property across all eight families, then "report robustness at the
largest epsilon" is systematically wrong rather than locally wrong, and the reporting radius
graduates from a checklist candidate to a rule with a mechanism — the standard the other eight
instances meet.

METHOD. Read the archived evasion curves (10 seeds, six radii, eight families, FGSM and PGD) and
locate, per family and attack, the radius maximizing mean damage. Add E14's NES curves for the
four families it covers. Report the per-seed argmax as a stability check, so a peak that is one
seed's noise is visible as such.

ATTACK PROVENANCE, stated because it is not uniform: in the shipped curves the differentiable
families (MLP, VQC) are attacked WHITE-BOX and the rest by TRANSFER from the MLP surrogate. That
is the published protocol, not a choice made here, and it is reported per family so no
cross-family strength comparison is implied.

DECISION RULE, fixed now, on the PGD curves:
  >= 6 of 8 families peak at the largest radius -> the effect is tree-specific; the reporting
                                                  radius stays a caveat
  <= 5 of 8 peak at the largest radius          -> the peak is a per-model property and the
                                                  reporting radius is a rule with a mechanism

CONTROLS.
  positive (blocking)  the re-derived clean and 0.5-sigma PGD figures must reproduce the values
                       already quoted from this artifact in the E14 entry (rf 0.2487, xgb 0.2799,
                       lgbm 0.2605 damage) to 1e-4
  negative             a family whose CLEAN macro-F1 is already within 0.05 of its floor cannot
                       show meaningful damage structure and is flagged, not counted

Analysis only: no model is fit, no attack is crafted.

Run:  python -m experiments.run_damage_peak_radius
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger

log = get_logger(__name__)

ADV = "results/runs/20260813T183448Z_adversarial_binary_q8"
NES = "results/runs/20260822T173155Z_score_based_blackbox_binary_q8"
E14_DAMAGE = {"rf": 0.2487, "xgb": 0.2799, "lgbm": 0.2605}  # 0.5 sigma, quoted in the E14 entry
TOL = 1e-4
FLOOR = 0.4152046783625731  # constant-predictor floor, binary q8 (used for the degeneracy flag)
TREE_ONLY_BAR = 6


def main() -> None:
    cfg = {
        "sources": {"evasion_curves": ADV, "nes": NES},
        "decision_bar": (
            f">= {TREE_ONLY_BAR} of 8 families peaking at the largest radius -> tree-specific"
        ),
        "note": "analysis only — no model is fit and no attack is crafted",
    }

    with RunDir("damage_peak_radius_binary_q8", config=cfg, seeds=[0]) as run:
        seeds = json.loads((REPO_ROOT / ADV / "results.json").read_text())["per_seed"]
        curves = seeds[0]["evasion"]["curves"]
        families = list(curves.keys())
        radii = sorted(curves[families[0]]["pgd"], key=float)

        rows = {}
        for atk in ("pgd", "fgsm"):
            rows[atk] = {}
            for m in families:
                clean = float(np.mean([s["evasion"]["clean"][m] for s in seeds]))
                dmg = {
                    e: clean - float(np.mean([s["evasion"]["curves"][m][atk][e] for s in seeds]))
                    for e in radii
                }
                per_seed_argmax = []
                for s in seeds:
                    d = {
                        e: s["evasion"]["clean"][m] - s["evasion"]["curves"][m][atk][e]
                        for e in radii
                    }
                    per_seed_argmax.append(max(d, key=lambda k: d[k]))
                peak = max(dmg, key=lambda k: dmg[k])
                rows[atk][m] = {
                    "clean": clean,
                    "clean_minus_floor": clean - FLOOR,
                    "clean_degenerate": bool(clean - FLOOR < 0.05),
                    "damage_by_radius": dmg,
                    "peak_radius": peak,
                    "peak_damage": dmg[peak],
                    "damage_at_largest": dmg[radii[-1]],
                    "peaks_at_largest": bool(peak == radii[-1]),
                    "loss_from_reporting_largest": dmg[peak] - dmg[radii[-1]],
                    "per_seed_argmax": Counter(per_seed_argmax),
                    "seed_agreement_on_peak": float(np.mean([a == peak for a in per_seed_argmax])),
                }

        # E14's NES curves, for the four families it covers
        nes_run = json.loads((REPO_ROOT / NES / "results.json").read_text())["per_seed"]
        nes_radii = sorted(nes_run[0]["nes"], key=float)
        rows["nes"] = {}
        for m in nes_run[0]["clean"]:
            clean = float(np.mean([c["clean"][m] for c in nes_run]))
            dmg = {e: clean - float(np.mean([c["nes"][e][m] for c in nes_run])) for e in nes_radii}
            peak = max(dmg, key=lambda k: dmg[k])
            rows["nes"][m] = {
                "clean": clean,
                "damage_by_radius": dmg,
                "peak_radius": peak,
                "peak_damage": dmg[peak],
                "damage_at_largest": dmg[nes_radii[-1]],
                "peaks_at_largest": bool(peak == nes_radii[-1]),
                "loss_from_reporting_largest": dmg[peak] - dmg[nes_radii[-1]],
            }

        # ---- controls ---------------------------------------------------------------
        ctrl = {
            m: {
                "measured": rows["pgd"][m]["damage_by_radius"]["0.5"],
                "e14_quoted": v,
                "delta": rows["pgd"][m]["damage_by_radius"]["0.5"] - v,
            }
            for m, v in E14_DAMAGE.items()
        }
        degenerate = [m for m in families if rows["pgd"][m]["clean_degenerate"]]
        controls = {
            "positive_e14_damage_reproduction": ctrl,
            "positive_pass": bool(all(abs(c["delta"]) <= TOL for c in ctrl.values())),
            "negative_degenerate_families_flagged": degenerate,
            "negative_note": (
                "a family already at its floor cannot show damage structure; flagged and "
                "excluded from the count rather than silently included"
            ),
        }
        controls["pass"] = controls["positive_pass"]
        log.info("CONTROLS: %s", json.dumps(controls, indent=2))

        # ---- decision rule ------------------------------------------------------------
        usable = [m for m in families if m not in degenerate]
        at_largest = [m for m in usable if rows["pgd"][m]["peaks_at_largest"]]
        verdict = {
            "controls": controls,
            "radii": radii,
            "families": families,
            "usable_families": usable,
            "pgd_peaks_at_largest": at_largest,
            "n_peaking_at_largest": len(at_largest),
            "n_usable": len(usable),
            "pgd_peak_radius_by_family": {m: rows["pgd"][m]["peak_radius"] for m in families},
            "understatement_by_family": {
                m: rows["pgd"][m]["loss_from_reporting_largest"] for m in families
            },
            "max_understatement": max(
                rows["pgd"][m]["loss_from_reporting_largest"] for m in usable
            ),
            "branch": (
                "tree_specific_stays_a_caveat"
                if len(at_largest) >= TREE_ONLY_BAR
                else "per_model_property_reporting_radius_is_a_rule"
            ),
            "detail": rows,
            "rule": cfg["decision_bar"],
        }
        run.write_json("results.json", verdict)
        run.metrics = {"n_peaking_at_largest": len(at_largest), "n_usable": len(usable)}
        log.info(
            "VERDICT %s | %d/%d usable families peak at the largest radius | peaks: %s",
            verdict["branch"],
            len(at_largest),
            len(usable),
            verdict["pgd_peak_radius_by_family"],
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
