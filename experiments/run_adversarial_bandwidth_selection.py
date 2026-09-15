"""P9b — does selecting the kernel length scale against an ADVERSARIAL objective, rather than clean
validation macro-F1, buy robustness? BOTH families.

PRE-REGISTRATION (BACKLOG `Validate` row "Adversarially-aware bandwidth selection, BOTH families",
opened by P9a 2026-08-15T00:15Z, re-checked and NOT superseded 2026-08-23T14:55Z; frozen before the
run). TIER A under the study protocol §11 gate: it answers a named reviewer objection —
hyperparameter selection under attack — and it feeds the checklist-instance decision already
queued as a v2 row. Run authorised by the operator to have the answer in hand before the
objection is raised.

WHY. P9a found the robustness-optimal quantum bandwidth (3.0) sits OUTSIDE the accuracy protocol's
tuning box (`tuning_box_high: 2.0` in `configs/quantum.yaml`), so accuracy tuning could never select
it; P8a found the same tension classically. N1 (2026-08-23T00:40Z) then established the mechanism:
classically it is the DEFENDER's length scale that picks the failure mode. Selecting bandwidth
against an adversarial objective is therefore choosing a failure mode deliberately instead of
inheriting whichever one clean-accuracy tuning lands on. E15 (2026-08-22T17:53Z) adds that the
fidelity kernel does not concentrate into the disabled regime at 16 qubits, so the quantum half of
the sweep has usable range.

THE QUESTION, and it is NOT "is there a tension" — P9a already showed there is. It is whether the
tension CONVERTS: does picking the robustness-optimal length scale actually deliver more robustness
on the held-out split, and what does it cost in clean accuracy?

DECISION RULE, fixed now:
  the robustness-optimal point differs from the accuracy-optimal point AND delivers >= 0.05 more
  adversarial macro-F1 ON TEST
      -> adversarially-aware selection is a REAL lever; hyperparameter selection under attack is a
         checklist instance and the paper says so
  otherwise
      -> the P9a/P8a tension does NOT convert into a usable defense; the paper reports it as a
         characterisation of the model class and makes no selection recommendation

Read per family. A split verdict (one family converts, the other does not) is itself the finding and
is reported as such rather than averaged away.

PROTOCOL DISCIPLINE. Selection happens on VAL for both objectives; TEST is touched once, at the two
selected points per family, exactly as the accuracy protocol does. C is re-tuned per bandwidth on
clean val, so the comparison is between length scales and not between regularisation strengths.

ATTACKS. White-box for both families, so neither is under-attacked:
  quantum   `QSVMWhiteBox` differentiates the fidelity decision function through the statevector
            simulator. The MARGIN objective is used, not BCE: P14a established that this model's
            duals pin |f| large, so BCE's dL/df underflows on confidently-correct points and
            understates the attack.
  classical the RBF-SVM decision function is analytically differentiable —
            f(x) = sum_i alpha_i exp(-gamma ||x - x_i||^2) + b — so the same margin objective is
            implemented directly rather than falling back to a surrogate transfer attack, which
            would be strictly weaker and would confound the cross-family comparison.

CONTROLS.
  positive (blocking)  the accuracy-optimal QUANTUM bandwidth must fall INSIDE the tuning box
                       (<= 2.0), reproducing P9a's premise that accuracy tuning cannot reach 3.0.
                       If it does not, the sweep is not measuring what P9a measured.
  positive (reported)  clean macro-F1 at bandwidth 3.0 against P9a's archived 0.601.
  negative             at eps = 0 the adversarial score must equal the clean score at every point,
                       for both families; an attack path that fires at zero radius is broken.
  negative (added      NO adversarial score may EXCEED its own clean score. An attack cannot make
  after the first run) a model more accurate; the first run returned adv = 1.0000 above clean
                       everywhere because the hand-rolled PGD descended the margin instead of
                       ascending it. This is now asserted rather than eyeballed.
  reported             the full clean-vs-adversarial curve per family, so a flat optimum — where the
                       "selection" is a coin flip between indistinguishable points — is visible
                       rather than reported as a choice.

Run:  python -m experiments.run_adversarial_bandwidth_selection --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from qgridbench.attacks.evasion import pgd as shipped_pgd
from qgridbench.attacks.qsvm_whitebox import QSVMWhiteBox
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
QUBITS = 8
ENCODING = "zz"
TRAIN_CAP = 2000
EVAL_CAP = 100  # ATTACK eval subset — matches the Phase-4 white-box control; attacks are costly
CLEAN_CAP = 2000  # CLEAN selection subset — `bandwidth_curve.val_subset` in configs/quantum.yaml
ATTACK_EPS = 0.5  # sigma units, the radius the robustness table headlines
PGD_STEPS = 40
STEP_FRAC = 0.1
CONVERT_BAR = 0.05
TUNING_BOX_HIGH = 2.0
P9A_BW3_CLEAN = 0.601


class RBFWhiteBox:
    """Analytic differentiable decision function of a fitted RBF SVC.

    f(x) = sum_i alpha_i exp(-gamma ||x - sv_i||^2) + b. Implemented here rather than in `src/`
    because it serves this experiment only; if a second caller ever appears it graduates.
    """

    def __init__(self, svc: SVC, gamma: float):
        if len(getattr(svc, "classes_", [])) != 2:
            raise ValueError("binary only")
        self.sv = torch.as_tensor(svc.support_vectors_, dtype=torch.float64)
        self.alpha = torch.as_tensor(svc.dual_coef_[0], dtype=torch.float64)
        self.b = float(svc.intercept_[0])
        self.gamma = float(gamma)

    def _decision_t(self, x: torch.Tensor) -> torch.Tensor:
        d2 = torch.cdist(x, self.sv) ** 2
        return torch.exp(-self.gamma * d2) @ self.alpha + self.b

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._decision_t(torch.as_tensor(X, dtype=torch.float64)).numpy()

    def margin_input_grad(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        sgn = -(2.0 * np.asarray(y, dtype=np.float64) - 1.0)
        xb = torch.tensor(np.asarray(X, dtype=np.float64), requires_grad=True)
        sb = torch.as_tensor(sgn, dtype=torch.float64)
        (g,) = torch.autograd.grad((sb * self._decision_t(xb)).sum(), xb)
        return g.detach().numpy()


def pgd_margin(model, X, y, eps, scale, rng_seed=0):
    """White-box PGD on the shared pre-encoding space, MARGIN objective.

    Delegates to the SHIPPED `attacks.evasion.pgd` rather than reimplementing the loop. The first
    version of this file rolled its own and inverted the sign — it DESCENDED the margin, which
    optimises the model's correctness instead of attacking it, and produced adversarial macro-F1 of
    1.0000 above every clean score. The shipped helper ascends (`X_adv + alpha * sign(g)`) and adds
    the random start inside the ball; passing `margin_input_grad` as its `grad_fn` gets the
    non-saturating objective P14a requires without owning the loop.
    """
    if eps == 0.0:
        return np.asarray(X, dtype=np.float64).copy()
    return shipped_pgd(
        model.margin_input_grad,
        np.asarray(X, dtype=np.float64),
        np.asarray(y),
        eps,
        scale,
        n_steps=PGD_STEPS,
        step_frac=STEP_FRAC,
        rng=np.random.default_rng(rng_seed),
    )


def _f1(y, dec):
    return float(f1_score(y, (np.asarray(dec) > 0).astype(int), average="macro"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--eps", type=float, default=ATTACK_EPS)
    args = ap.parse_args()

    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    bws = cfg_q["ablations"]["bandwidth_curve"]["bandwidths"]
    c_grid = cfg_q["ablations"]["bandwidth_curve"]["svm_C_grid"]
    gammas = list(np.logspace(-3, 1, 9))

    cfg = {
        "variant": VARIANT,
        "qubits": QUBITS,
        "encoding": ENCODING,
        "bandwidths": bws,
        "gammas": [float(g) for g in gammas],
        "svm_C_grid": c_grid,
        "attack": f"white-box PGD, MARGIN objective, eps={args.eps} sigma, {PGD_STEPS} steps",
        "train_cap": TRAIN_CAP,
        "eval_cap": EVAL_CAP,
        "convert_bar": CONVERT_BAR,
        "tuning_box_high": TUNING_BOX_HIGH,
        "seeds": args.seeds,
        "note": "selection on VAL for both objectives; TEST touched once per selected point",
    }

    with RunDir(
        f"adversarial_bandwidth_selection_{VARIANT}_q{QUBITS}", config=cfg, seeds=args.seeds
    ) as run:
        reg = prepare_regime(VARIANT, "pca", n_components=QUBITS, seed=0)
        scale = reg.feature_scale
        tr = stratified_cap(reg.X["train"], reg.y["train"], TRAIN_CAP, seed=0)
        Xtr, ytr = reg.X["train"][tr], reg.y["train"][tr]
        # Two val views, and the asymmetry is deliberate. CLEAN selection uses the protocol's own
        # 2,000-point subset; ADVERSARIAL selection uses 100 because a white-box attack through the
        # statevector simulator at 2,000 points is not affordable. The first run of this file used
        # 100 for BOTH and the clean curve was noise: its accuracy optimum sat at bandwidth 1.20
        # against 0.80 at the proper size, and bandwidth 3.0's clean penalty read 0.107 against
        # 0.019. The selection a protocol makes must be made on the sample that protocol uses.
        vc = stratified_cap(reg.X["val"], reg.y["val"], CLEAN_CAP, seed=0)
        Xvc, yvc = reg.X["val"][vc], reg.y["val"][vc]
        va = stratified_cap(reg.X["val"], reg.y["val"], EVAL_CAP, seed=0)
        Xva, yva = reg.X["val"][va], reg.y["val"][va]
        te = stratified_cap(reg.X["test"], reg.y["test"], EVAL_CAP, seed=0)
        Xte, yte = reg.X["test"][te], reg.y["test"][te]

        def best_C(K_tr, K_ev, y_ev):
            out = []
            for C in c_grid:
                set_all_seeds(0)
                m = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K_tr, ytr)
                out.append((_f1(y_ev, m.decision_function(K_ev)), C, m))
            return max(out, key=lambda t: t[0])

        # ---------------------------------------------------------------- quantum
        qcurve = []
        for bw in bws:
            s_tr = compute_states(reg.angle_scaler.transform(Xtr), ENCODING, bandwidth=bw)
            K_tr = fidelity_kernel(s_tr, s_tr)
            s_vc = compute_states(reg.angle_scaler.transform(Xvc), ENCODING, bandwidth=bw)
            clean_va, C, svc = best_C(K_tr, fidelity_kernel(s_vc, s_tr), yvc)
            wb = QSVMWhiteBox(svc, s_tr, reg.angle_scaler, ENCODING, bw)
            adv_va = _f1(yva, wb.decision_function(pgd_margin(wb, Xva, yva, args.eps, scale)))
            zero_va = _f1(yva, wb.decision_function(pgd_margin(wb, Xva, yva, 0.0, scale)))
            clean_on_attack_subset = _f1(yva, wb.decision_function(Xva))
            qcurve.append(
                {
                    "bandwidth": float(bw),
                    "C": C,
                    "clean_val": clean_va,
                    "clean_val_attack_subset": clean_on_attack_subset,
                    "adv_val": adv_va,
                    "zero_radius_val": zero_va,
                    "in_tuning_box": bool(bw <= TUNING_BOX_HIGH),
                }
            )
            log.info("q bw %.2f | C %-7g clean %.4f adv %.4f", bw, C, clean_va, adv_va)
            run.write_json("results_partial.json", {"quantum_curve": qcurve})

        # -------------------------------------------------------------- classical
        ccurve = []
        for g in gammas:
            outs = []
            for C in c_grid:
                set_all_seeds(0)
                m = SVC(C=C, gamma=g, class_weight="balanced").fit(Xtr, ytr)
                outs.append((_f1(yvc, m.decision_function(Xvc)), C, m))
            clean_va, C, svc = max(outs, key=lambda t: t[0])
            wb = RBFWhiteBox(svc, g)
            adv_va = _f1(yva, wb.decision_function(pgd_margin(wb, Xva, yva, args.eps, scale)))
            zero_va = _f1(yva, wb.decision_function(pgd_margin(wb, Xva, yva, 0.0, scale)))
            clean_on_attack_subset = _f1(yva, wb.decision_function(Xva))
            ccurve.append(
                {
                    "gamma": float(g),
                    "C": C,
                    "clean_val": clean_va,
                    "clean_val_attack_subset": clean_on_attack_subset,
                    "adv_val": adv_va,
                    "zero_radius_val": zero_va,
                }
            )
            log.info("c gamma %.4g | C %-7g clean %.4f adv %.4f", g, C, clean_va, adv_va)
            run.write_json("results_partial.json", {"quantum_curve": qcurve, "classical": ccurve})

        # ------------------------------------------------ selection, then TEST once
        def refit_and_test(fam, key, val):
            if fam == "quantum":
                s_tr = compute_states(reg.angle_scaler.transform(Xtr), ENCODING, bandwidth=val)
                K_tr = fidelity_kernel(s_tr, s_tr)
                s_te = compute_states(reg.angle_scaler.transform(Xte), ENCODING, bandwidth=val)
                # C comes from the VAL curve, never from test — the whole protocol point
                pt = [p for p in qcurve if p["bandwidth"] == val][0]
                set_all_seeds(0)
                svc = SVC(kernel="precomputed", C=pt["C"], class_weight="balanced").fit(K_tr, ytr)
                wb = QSVMWhiteBox(svc, s_tr, reg.angle_scaler, ENCODING, val)
                clean = _f1(yte, svc.decision_function(fidelity_kernel(s_te, s_tr)))
            else:
                pt = [p for p in ccurve if p["gamma"] == val][0]
                set_all_seeds(0)
                svc = SVC(C=pt["C"], gamma=val, class_weight="balanced").fit(Xtr, ytr)
                wb = RBFWhiteBox(svc, val)
                clean = _f1(yte, svc.decision_function(Xte))
            adv = _f1(yte, wb.decision_function(pgd_margin(wb, Xte, yte, args.eps, scale)))
            return {key: float(val), "C": pt["C"], "clean_test": clean, "adv_test": adv}

        results = {}
        for fam, curve, key in (
            ("quantum", qcurve, "bandwidth"),
            ("classical", ccurve, "gamma"),
        ):
            acc_pt = max(curve, key=lambda p: p["clean_val"])
            rob_pt = max(curve, key=lambda p: p["adv_val"])
            acc_t = refit_and_test(fam, key, acc_pt[key])
            rob_t = refit_and_test(fam, key, rob_pt[key])
            gain = rob_t["adv_test"] - acc_t["adv_test"]
            results[fam] = {
                "accuracy_selected": acc_t,
                "robustness_selected": rob_t,
                "selection_differs": bool(acc_pt[key] != rob_pt[key]),
                "adv_gain_on_test": float(gain),
                "clean_cost_on_test": float(acc_t["clean_test"] - rob_t["clean_test"]),
                "converts": bool(acc_pt[key] != rob_pt[key] and gain >= CONVERT_BAR),
            }
            log.info(
                "%s: acc-opt %s=%g adv %.4f | rob-opt %s=%g adv %.4f | gain %+0.4f converts=%s",
                fam,
                key,
                acc_pt[key],
                acc_t["adv_test"],
                key,
                rob_pt[key],
                rob_t["adv_test"],
                gain,
                results[fam]["converts"],
            )

        acc_bw = results["quantum"]["accuracy_selected"]["bandwidth"]
        bw3 = [p for p in qcurve if p["bandwidth"] == 3.0]
        controls = {
            "positive_accuracy_optimal_bw": acc_bw,
            "positive_inside_tuning_box": bool(acc_bw <= TUNING_BOX_HIGH),
            "positive_pass": bool(acc_bw <= TUNING_BOX_HIGH),
            "positive_note": "P9a's premise: accuracy tuning cannot reach bandwidth 3.0",
            "reported_bw3_clean_val": bw3[0]["clean_val"] if bw3 else None,
            "reported_p9a_bw3_clean": P9A_BW3_CLEAN,
            "negative_adv_never_exceeds_clean": {
                "quantum": bool(
                    all(p["adv_val"] <= p["clean_val_attack_subset"] + 1e-9 for p in qcurve)
                ),
                "classical": bool(
                    all(p["adv_val"] <= p["clean_val_attack_subset"] + 1e-9 for p in ccurve)
                ),
                "note": "an attack cannot make a model more accurate; the first run of this file "
                "returned adv = 1.0000 above clean everywhere because its hand-rolled PGD "
                "descended the margin instead of ascending it",
            },
            "negative_zero_radius_no_op": {
                "quantum": bool(
                    all(
                        abs(p["zero_radius_val"] - p["clean_val_attack_subset"]) < 1e-9
                        for p in qcurve
                    )
                ),
                "classical": bool(
                    all(
                        abs(p["zero_radius_val"] - p["clean_val_attack_subset"]) < 1e-9
                        for p in ccurve
                    )
                ),
            },
        }
        controls["negative_pass"] = bool(
            all(controls["negative_zero_radius_no_op"].values())
            and controls["negative_adv_never_exceeds_clean"]["quantum"]
            and controls["negative_adv_never_exceeds_clean"]["classical"]
        )
        controls["pass"] = bool(controls["positive_pass"] and controls["negative_pass"])
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        verdict = {
            "controls": controls,
            "quantum_curve": qcurve,
            "classical_curve": ccurve,
            "selection": results,
            "convert_bar": CONVERT_BAR,
            "branch": (
                "adversarial_selection_CONVERTS"
                if any(v["converts"] for v in results.values())
                else "tension_does_NOT_convert"
            ),
            "split_verdict": bool(
                results["quantum"]["converts"] != results["classical"]["converts"]
            ),
            "rule": f"selection differs AND >= {CONVERT_BAR} more adversarial macro-F1 on test",
        }
        run.write_json("results.json", verdict)
        run.metrics = {
            "quantum_adv_gain": results["quantum"]["adv_gain_on_test"],
            "classical_adv_gain": results["classical"]["adv_gain_on_test"],
        }
        log.info("VERDICT %s | split=%s", verdict["branch"], verdict["split_verdict"])
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
