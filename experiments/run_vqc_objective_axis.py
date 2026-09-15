"""E13 — VQC objective axis: does |<Z>| decay under a DISTANCE-type objective?

PRE-REGISTRATION (pass-4 workplan item E13; frozen before the run).
P11c/P11d (2026-08-15T07:05Z / 07:20Z) showed the fidelity kernel's "no decay regime" was an
artifact of the LOSS-MAXIMIZING objective: a mass-DESCENDING attack reaches the disabled end
state the loss attack never expressed, and the paper now says so. The VQC's own non-decay claim
— |<Z>| dips to 0.887 at eps = 0.1 and then RISES to 2.67 at 2 sigma — rests on exactly the
same loss-objective evidence and has never been tested against the descending objective. The
paper states that gap rather than closing it. This run closes it.

WHY |<Z>| IS THE RIGHT READOUT. The VQC reaches its decision through a capped head
(scale * <PauliZ> + bias, expectation bounded in [-1, 1]), so it cannot gain decision magnitude
the way the QSVM does. |<Z>| is its analogue of kernel mass: the quantity that would have to
collapse for the model to exhaust. |<Z>| -> 0 puts the head at a constant output, predictions
become constant, and macro-F1 returns to the constant-predictor floor — the DISABLED end state,
which is the monitorable failure an operator can see.

DECISION RULE (kill criterion, mirroring P8b's and fixed before the run). At any radius:
  mass ratio <= 0.05 AND macro-F1 within 0.05 of the floor
  -> the disabled regime IS reachable for the variational family under a distance-type
     objective, and the VQC's non-decay claim is OBJECTIVE-DEPENDENT exactly as the kernel's
     was. The paper's scoped sentence becomes a measured one.
  otherwise
  -> non-decay holds for the VQC across BOTH objectives, which is a stronger statement than
     the paper currently makes and licenses removing the scope hedge.

CONTROLS.
  positive (blocking)  the LOSS-objective arm re-run here must reproduce the published profile
                       within 0.02 absolute at every radius (0.9825 / 0.9248 / 0.8869 / 0.9805 /
                       1.2102 / 1.8585 / 2.7157 / 2.6684), artifact
                       `20260814T100533Z_whitebox_qsvm_binary_q8`
  negative             a RANDOM perturbation at the matched radius must not reduce mass MORE
                       than the descending objective at any radius. If noise beats the targeted
                       attack, the gradient is wrong and no number here means anything.

Run:  python -m experiments.run_vqc_objective_axis
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from experiments.run_adversarial import CLASSICAL, FittedZoo, _f1  # noqa: F401  (CLASSICAL: zoo)
from experiments.run_whitebox_qsvm import _mass_descent
from qgridbench.attacks.evasion import pgd
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import spawn_rng

log = get_logger(__name__)

VARIANT = "binary"
# Published loss-objective profile (p7b, artifact 20260814T100533Z_whitebox_qsvm_binary_q8).
PUBLISHED_LOSS_MASS = {
    "0.01": 0.9825,
    "0.05": 0.9248,
    "0.1": 0.8869,
    "0.2": 0.9805,
    "0.3": 1.2102,
    "0.5": 1.8585,
    "1.0": 2.7157,
    "2.0": 2.6684,
}
CONTROL_TOL = 0.02
DISABLED_MASS = 0.05
DISABLED_F1_BAND = 0.05


def expvals_np(vqc, angle_scaler, Z: np.ndarray) -> np.ndarray:
    """|<Z>| readout, the VQC's analogue of kernel mass (p7b's `k_rows` for the vqc arm)."""
    A = torch.as_tensor(angle_scaler.transform(Z), dtype=torch.float64)
    with torch.no_grad():
        return vqc._expvals(A, vqc.weights).numpy()


def vqc_mass_grad(vqc, angle_scaler, Z: np.ndarray, chunk: int) -> np.ndarray:
    """d/dx of mean |<Z>|, SUMMED over the batch so each row's gradient is its own.

    Mirrors `_qsvm_mass_grad` exactly, including the sum-over-batch convention: a batch MEAN
    silently rescales by 1/B, which sign() hides from the attack but not from a gradient check.
    The chain rule runs through the train-fitted angle scaler, so the attack stays in the shared
    pre-encoding feature space (comparability contract).
    """
    a_scale = torch.as_tensor(angle_scaler._mm.scale_, dtype=torch.float64)
    a_min = torch.as_tensor(angle_scaler._mm.min_, dtype=torch.float64)
    g = np.empty_like(Z)
    for s in range(0, len(Z), chunk):
        xb = torch.tensor(Z[s : s + chunk], dtype=torch.float64, requires_grad=True)
        # clamp, not bare affine: AngleScaler.transform clips to [-pi, pi], and
        # `_qsvm_mass_grad` clamps for the same reason — a gradient that flows where the
        # readout saturates is a different function from the one being attacked
        ang = torch.clamp(xb * a_scale + a_min, -np.pi, np.pi)
        m = vqc._expvals(ang, vqc.weights).abs().sum()
        (gr,) = torch.autograd.grad(m, xb)
        g[s : s + chunk] = gr.detach().numpy()
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="zz")  # routes the QSVM only; the VQC ansatz is fixed
    ap.add_argument("--chunk", type=int, default=128)
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    pgd_cfg = cfg_adv["evasion"]["pgd"]
    radii = sorted(set(cfg_adv["evasion"]["epsilons"]) | {1.0, 2.0})

    cfg = {
        "variant": args.variant,
        "qubits": args.qubits,
        "radii": radii,
        "pgd": pgd_cfg,
        "published_loss_mass": PUBLISHED_LOSS_MASS,
        "control_tol": CONTROL_TOL,
        "disabled_mass": DISABLED_MASS,
        "disabled_f1_band": DISABLED_F1_BAND,
    }

    with RunDir(f"vqc_objective_axis_{args.variant}_q{args.qubits}", config=cfg, seeds=[0]) as run:
        reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
        zoo = FittedZoo(reg, args.encoding, args.qubits, 0, cfg_q["qkernel"]["subset_cap"], best)
        vqc, sc = zoo.models["vqc"], zoo._angle
        gp_vqc = zoo.grad_provider("vqc")
        scale = reg.feature_scale

        # p7b's evaluation subset, verbatim, so the arms are comparable to the published cell
        n_eval = cfg_adv["blackbox"]["eval_subset"] * 2
        idx = stratified_cap(reg.X["test"], reg.y["test"], n_eval, seed=0)
        X, y = reg.X["test"][idx], reg.y["test"][idx]
        maj = int(np.round(y.mean()))
        floor = _f1(
            y, np.column_stack([1 - np.full(len(y), maj), np.full(len(y), maj)]).astype(float)
        )
        m0 = float(np.abs(expvals_np(vqc, sc, X)).mean())
        log.info("clean |<Z>| %.4f | floor %.4f | n_eval %d", m0, floor, len(y))

        def read(Xa: np.ndarray) -> dict:
            proba = zoo.predict_proba("vqc", Xa)
            f1 = _f1(y, proba)
            pred = proba.argmax(1)
            return {
                "mass_ratio": float(np.abs(expvals_np(vqc, sc, Xa)).mean() / m0),
                "macro_f1": f1,
                "f1_minus_floor": float(f1 - floor),
                "modal_frac": float(max(pred.mean(), 1.0 - pred.mean())),
            }

        curve: dict[str, dict] = {}
        for e in radii:
            loss_arm = pgd(
                gp_vqc,
                X,
                y,
                e,
                scale,
                n_steps=pgd_cfg["n_steps"],
                step_frac=pgd_cfg["step_frac"],
                rng=spawn_rng(0, f"p7b_vqc_{e}"),  # p7b's stream, so the control can be exact
            )
            desc_arm = _mass_descent(
                lambda Z: vqc_mass_grad(vqc, sc, Z, args.chunk),
                X,
                e,
                scale,
                np.random.default_rng(1),
                pgd_cfg["n_steps"],
                pgd_cfg["step_frac"],
            )
            rnd_arm = X + np.random.default_rng(2).uniform(-1, 1, X.shape) * e * scale
            curve[str(e)] = {
                "loss_objective": read(loss_arm),
                "mass_descent_objective": read(desc_arm),
                "random_control": read(rnd_arm),
            }
            c = curve[str(e)]
            log.info(
                "eps %-4s | loss mass %.4f f1 %.4f | DESCENT mass %.4f f1 %.4f | random mass %.4f",
                e,
                c["loss_objective"]["mass_ratio"],
                c["loss_objective"]["macro_f1"],
                c["mass_descent_objective"]["mass_ratio"],
                c["mass_descent_objective"]["macro_f1"],
                c["random_control"]["mass_ratio"],
            )
            run.write_json("results_partial.json", {"curve": curve, "clean_mass": m0})

        # ---- controls -------------------------------------------------------------
        pos = {
            e: {
                "measured": curve[e]["loss_objective"]["mass_ratio"],
                "published": v,
                "delta": curve[e]["loss_objective"]["mass_ratio"] - v,
            }
            for e, v in PUBLISHED_LOSS_MASS.items()
            if e in curve
        }
        pos_ok = all(abs(c["delta"]) <= CONTROL_TOL for c in pos.values())
        neg_violations = [
            e
            for e in curve
            if curve[e]["random_control"]["mass_ratio"]
            < curve[e]["mass_descent_objective"]["mass_ratio"] - 1e-12
        ]
        controls = {
            "positive_loss_profile_reproduction": pos,
            "positive_pass": bool(pos_ok),
            "negative_random_beats_descent_at": neg_violations,
            "negative_pass": bool(not neg_violations),
            "pass": bool(pos_ok and not neg_violations),
        }
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        # ---- decision rule ---------------------------------------------------------
        disabled_at = [
            e
            for e in curve
            if curve[e]["mass_descent_objective"]["mass_ratio"] <= DISABLED_MASS
            and abs(curve[e]["mass_descent_objective"]["f1_minus_floor"]) <= DISABLED_F1_BAND
        ]
        min_mass = min(curve[e]["mass_descent_objective"]["mass_ratio"] for e in curve)
        min_loss_mass = min(curve[e]["loss_objective"]["mass_ratio"] for e in curve)
        verdict = {
            "controls": controls,
            "clean_mass_abs": m0,
            "floor": floor,
            "min_mass_ratio_descent": float(min_mass),
            "min_mass_ratio_loss": float(min_loss_mass),
            "descent_over_loss_at_2sigma": float(
                curve["2.0"]["mass_descent_objective"]["mass_ratio"]
                / max(curve["2.0"]["loss_objective"]["mass_ratio"], 1e-12)
            ),
            "disabled_regime_reached_at": disabled_at,
            "branch": (
                "objective_dependent_like_the_kernel"
                if disabled_at
                else "non_decay_holds_across_both_objectives"
            ),
            "rule": (
                f"mass <= {DISABLED_MASS} AND |f1 - floor| <= {DISABLED_F1_BAND} at any radius "
                "-> the disabled regime is reachable and the claim is objective-dependent"
            ),
        }
        run.write_json("results.json", {"curve": curve, "verdict": verdict})
        run.metrics = {"min_mass_ratio_descent": float(min_mass)}
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
