"""E15 — does the fidelity kernel concentrate into the DISABLED regime at large width?

PRE-REGISTRATION (pass-4 workplan item E15; frozen before the run).
The fidelity kernel's retained mass under a full-budget white-box attack FALLS with qubit
count: 0.179 at q8, 0.103 at q12 (P8c, 2026-08-14). That is consistent with the ~2^-n fidelity
concentration already logged on the qubit axis, and the paper therefore states non-decay at 8
and 12 qubits and explicitly DECLINES to extrapolate — a sufficiently wide kernel may well
concentrate into the disabled end state, and nothing published here rules it out. A q16 cell
settles it.

SCOPE, stated rather than discovered later: this run computes the KERNEL arm only. p7b's cell
also fits a VQC, whose backprop cost scales with the 2^n statevector and would put a q16 fit in
the hours-to-a-day range for a model this question does not ask about. The VQC's width axis
stays untested and is not claimed either way.

DECISION RULE (the P8b kill criterion, unchanged). At the full 2-sigma budget:
  retained mass <= 0.05 AND macro-F1 within 0.05 of the constant-predictor floor
    -> the kernel CONCENTRATES into the disabled regime at q16; non-decay does not survive to
       width, and the paper's non-extrapolation clause becomes a measured boundary
  otherwise
    -> non-decay survives to 16 qubits; the clause stays, now with a third width behind it

CONTROLS (blocking, and they are the reason this runner recomputes q8 and q12 rather than
quoting them). The SAME code path must reproduce the published widths:
  q8   retained mass 0.179 within 0.02
  q12  retained mass 0.103 within 0.02
A new runner that cannot reproduce the two known widths cannot be trusted on the third.

Run:  python -m experiments.run_kernel_width_exhaustion --widths 8 12 16
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.svm import SVC

from experiments.run_adversarial import _f1
from qgridbench.attacks.evasion import pgd
from qgridbench.attacks.qsvm_whitebox import QSVMWhiteBox
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
from qgridbench.protocol import kernel_subset, prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)

VARIANT = "binary"
ENCODING = "zz"
PUBLISHED_MASS = {8: 0.179, 12: 0.103}  # P8c, decision log 2026-08-14
CONTROL_TOL = 0.02
DISABLED_MASS = 0.05
DISABLED_F1_BAND = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="*", default=[8, 12, 16])
    ap.add_argument("--radii", type=float, nargs="*", default=[0.5, 2.0])
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=32, help="statevector batch; 2^n bound")
    ap.add_argument(
        "--pgd-steps",
        type=int,
        default=None,
        help="override the config step count — the under-convergence control: a weaker attack "
        "at large width would look exactly like retained mass",
    )
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    pgd_cfg = dict(cfg_adv["evasion"]["pgd"])
    if args.pgd_steps:
        pgd_cfg["n_steps"] = args.pgd_steps
    bw = best["qk_bw"]  # held at the q8-tuned value across widths, exactly as P8c did

    cfg = {
        "variant": VARIANT,
        "encoding": ENCODING,
        "widths": args.widths,
        "radii": args.radii,
        "n_eval": args.n_eval,
        "pgd_steps": pgd_cfg["n_steps"],
        "bandwidth": bw,
        "qk_C": best["qk_C"],
        "published_mass": PUBLISHED_MASS,
        "scope": "kernel arm only; the VQC width axis is not run and is not claimed",
    }

    with RunDir(f"kernel_width_exhaustion_{VARIANT}", config=cfg, seeds=[0]) as run:
        cells = {}
        for n in args.widths:
            t0 = time.perf_counter()
            reg = prepare_regime(VARIANT, "pca", n_components=n, seed=0)
            sub = kernel_subset(reg, cfg_q["qkernel"]["subset_cap"], seed=0)
            Xtr, ytr = reg.X["train"][sub], reg.y["train"][sub]
            states_tr = compute_states(reg.angle_scaler.transform(Xtr), ENCODING, bandwidth=bw)
            set_all_seeds(0)
            svc = SVC(kernel="precomputed", C=best["qk_C"], class_weight="balanced")
            svc.fit(fidelity_kernel(states_tr, states_tr), ytr)
            wb = QSVMWhiteBox(svc, states_tr, reg.angle_scaler, ENCODING, bw, chunk=args.chunk)

            ev = stratified_cap(reg.X["test"], reg.y["test"], args.n_eval, seed=0)
            X, y = reg.X["test"][ev], reg.y["test"][ev]
            scale = reg.feature_scale
            maj = int(np.round(y.mean()))
            floor = _f1(
                y, np.column_stack([1 - np.full(len(y), maj), np.full(len(y), maj)]).astype(float)
            )

            def mass(Z, s=states_tr, m=svc, w=wb):
                st = compute_states(reg.angle_scaler.transform(Z), ENCODING, bandwidth=bw)
                return float(fidelity_kernel(st, s)[:, m.support_].mean())

            def f1_of(Z, w=wb):
                d = w.decision_function(Z)
                p = 1.0 / (1.0 + np.exp(-d))
                return _f1(y, np.column_stack([1 - p, p]))

            m0 = mass(X)
            cell = {
                "n_qubits": n,
                "clean_mass_abs": m0,
                "clean_f1": f1_of(X),
                "floor": floor,
                "n_support": int(len(svc.support_)),
                "curve": {},
            }
            for e in args.radii:
                Xa = pgd(
                    wb,
                    X,
                    y,
                    e,
                    scale,
                    n_steps=pgd_cfg["n_steps"],
                    step_frac=pgd_cfg["step_frac"],
                    rng=spawn_rng(0, f"p7b_qsvm_{e}"),  # p7b's stream, for the controls
                )
                f1 = f1_of(Xa)
                cell["curve"][str(e)] = {
                    "mass_ratio": mass(Xa) / m0,
                    "macro_f1": f1,
                    "f1_minus_floor": float(f1 - floor),
                }
                c = cell["curve"][str(e)]
                log.info(
                    "q%-3d eps %-4s | mass %.4f | F1 %.4f (floor %+0.4f)",
                    n,
                    e,
                    c["mass_ratio"],
                    c["macro_f1"],
                    c["f1_minus_floor"],
                )
            cell["wall_seconds"] = float(time.perf_counter() - t0)
            cells[str(n)] = cell
            run.write_json("results_partial.json", {"cells": cells})
            log.info("q%d done in %.1f min", n, cell["wall_seconds"] / 60)

        # ---- controls -------------------------------------------------------------
        full = str(max(args.radii))
        pos = {
            str(n): {
                "measured": cells[str(n)]["curve"][full]["mass_ratio"],
                "published": v,
                "delta": cells[str(n)]["curve"][full]["mass_ratio"] - v,
            }
            for n, v in PUBLISHED_MASS.items()
            if str(n) in cells
        }
        pos_ok = all(abs(c["delta"]) <= CONTROL_TOL for c in pos.values())
        controls = {
            "positive_published_width_reproduction": pos,
            "positive_pass": bool(pos_ok),
            "note": (
                "a runner that cannot reproduce the two known widths cannot be trusted on the "
                "third; the q8/q12 cells are recomputed here rather than quoted"
            ),
            "pass": bool(pos_ok),
        }
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        # ---- decision rule ----------------------------------------------------------
        widest = str(max(args.widths))
        w = cells[widest]["curve"][full]
        disabled = bool(
            w["mass_ratio"] <= DISABLED_MASS and abs(w["f1_minus_floor"]) <= DISABLED_F1_BAND
        )
        trend = {str(n): cells[str(n)]["curve"][full]["mass_ratio"] for n in args.widths}
        verdict = {
            "controls": controls,
            "mass_by_width_at_full_budget": trend,
            "widest_cell": w,
            "disabled_regime_reached": disabled,
            "branch": (
                "concentrates_into_disabled_regime" if disabled else "non_decay_survives_to_width"
            ),
            "monotone_decline_in_mass": bool(
                all(
                    trend[str(args.widths[i + 1])] <= trend[str(args.widths[i])] + 1e-12
                    for i in range(len(args.widths) - 1)
                )
            ),
            "rule": (
                f"mass <= {DISABLED_MASS} AND |F1 - floor| <= {DISABLED_F1_BAND} at the full "
                "budget -> concentration into the disabled regime"
            ),
        }
        run.write_json("results.json", {"cells": cells, "verdict": verdict})
        run.metrics = {"mass_widest": w["mass_ratio"]}
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
