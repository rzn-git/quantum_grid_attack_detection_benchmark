"""P5a + P6a: the two white-box anchors the robustness section was missing.

P5a — white-box PGD/FGSM through the simulated fidelity kernel. The robustness
table reports the QSVM's 0.886 PGD retention as a TRANSFERRED (strictly weaker)
attack and labels it an attack-limited upper bound, because no working targeted
attack against the fidelity kernel existed in the protocol. This runs the
strongest feasible one: differentiate f(x)=sum alpha_j k(x,x_j) through the
statevector simulator (qsvm_whitebox.py), numerically gated against the
lightning.qubit reference, then PGD in the shared PCA feature space at the same
epsilon grid every other family sees.

P6a — full-feature white-box anchor. One PGD run on a 128-feature MLP tuned on
the full regime, testing whether adversarial vulnerability persists at
deployment-grade accuracy (0.766 macro-F1) or is an artifact of the matched
low-dimensional regime.

Both consume the SAME shared protocol (prepare_regime, tuned params, stratified
subsets) as run_adversarial.py, so numbers are directly comparable. Outputs a
run dir under results/; make_report picks up the JSON to regenerate exhibits.

Run:  python -m experiments.run_whitebox_qsvm --variant binary --qubits 8
"""

from __future__ import annotations

import argparse
import json

import numpy as np

# reuse the exact zoo the robustness table was built from
from experiments.run_adversarial import FittedZoo, _f1, _svc_on_kernel, _svc_proba
from qgridbench.attacks.evasion import fgsm, pgd
from qgridbench.attacks.gradients import GradProvider, mlp_loss_input_grad
from qgridbench.attacks.qsvm_whitebox import QSVMWhiteBox, assert_matches_reference
from qgridbench.eval.stats import paired_wilcoxon
from qgridbench.models.classical.zoo import (
    build,
    fit_model,
    predict_proba_timed,
    stratified_cap,
)
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
from qgridbench.protocol import kernel_subset, prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)


def _macro_f1_curve(predict_proba, X, y, X_adv_by_eps):
    return {str(e): _f1(y, predict_proba(Xa)) for e, Xa in X_adv_by_eps.items()}


def p5a_whitebox_qsvm(reg, args, cfg_adv, best_params, subset_cap, seed=0, attack_seeds=(0,)):
    """Targeted white-box PGD/FGSM against the fidelity-kernel QSVM.

    The QSVM is deterministic given its fixed training subset (the paper's stated
    protocol), so the only stochastic element is PGD's random start. Spread is
    therefore reported over ATTACK seeds, with the model held fixed — stated as
    such wherever the number appears.
    """
    zoo = FittedZoo(reg, args.encoding, args.qubits, seed, subset_cap, best_params)
    svc = zoo.models["qsvm"]
    bw = best_params.get("qk_bw", 1.0)
    wb = QSVMWhiteBox(svc, zoo._states_tr, zoo._angle, args.encoding, bw)

    # NUMERIC GATE on real data before any reported number depends on this path
    gate_idx = stratified_cap(reg.X["test"], reg.y["test"], 40, seed=0)
    gate = assert_matches_reference(wb, svc, zoo._states_tr, zoo._angle, reg.X["test"][gate_idx])
    log.info(
        "P5a gate: forward(exact) %.2e, forward(deployed) %.2e, grad %.2e",
        gate["forward_rel_err"],
        gate["deployed_float32_rel_err"],
        gate["grad_rel_err"],
    )

    ev_idx = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    Xte, yte = reg.X["test"][ev_idx], reg.y["test"][ev_idx]
    scale = reg.feature_scale
    eps_grid = cfg_adv["evasion"]["epsilons"]
    pgd_cfg = cfg_adv["evasion"]["pgd"]

    def qsvm_proba(X):
        return zoo.predict_proba("qsvm", X)

    clean = _f1(yte, qsvm_proba(Xte))
    half = 0.5 if 0.5 in eps_grid else max(eps_grid)

    # per-attack-seed curves -> mean/std per epsilon
    per_seed_pgd = {str(e): [] for e in eps_grid}
    per_seed_fgsm = {str(e): [] for e in eps_grid}  # FGSM is deterministic; one entry
    per_seed_transfer = []
    # CLASSICAL-KERNEL CONTROL: the same targeted attack on the tuned RBF-SVM, so
    # "the fidelity kernel collapses" is measured against a kernel machine attacked
    # at identical strength, not against models the protocol could only transfer to.
    per_seed_rbf = {str(e): [] for e in eps_grid}
    gp_mlp = zoo.grad_provider("mlp")
    gp_rbf = GradProvider(zoo.models["rbf_svm"], "rbf_svm")

    def rbf_proba(X):
        return zoo.predict_proba("rbf_svm", X)

    clean_rbf = _f1(yte, rbf_proba(Xte))
    for a_seed in attack_seeds:
        rng = spawn_rng(a_seed, "p5a")
        for e in eps_grid:
            Xa = pgd(
                wb,
                Xte,
                yte,
                e,
                scale,
                n_steps=pgd_cfg["n_steps"],
                step_frac=pgd_cfg["step_frac"],
                rng=rng,
            )
            per_seed_pgd[str(e)].append(_f1(yte, qsvm_proba(Xa)))
            Xr = pgd(
                gp_rbf,
                Xte,
                yte,
                e,
                scale,
                n_steps=pgd_cfg["n_steps"],
                step_frac=pgd_cfg["step_frac"],
                rng=rng,
            )
            per_seed_rbf[str(e)].append(_f1(yte, rbf_proba(Xr)))
        X_tr = pgd(
            gp_mlp,
            Xte,
            yte,
            half,
            scale,
            n_steps=pgd_cfg["n_steps"],
            step_frac=pgd_cfg["step_frac"],
            rng=rng,
        )
        per_seed_transfer.append(_f1(yte, qsvm_proba(X_tr)))
    for e in eps_grid:  # FGSM has no random start
        per_seed_fgsm[str(e)].append(_f1(yte, qsvm_proba(fgsm(wb, Xte, yte, e, scale))))

    def agg(d):
        return {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
            for k, v in d.items()
        }

    pgd_curve, fgsm_curve, rbf_curve = agg(per_seed_pgd), agg(per_seed_fgsm), agg(per_seed_rbf)
    wb_half, rbf_half = pgd_curve[str(half)], rbf_curve[str(half)]
    tr_mean = float(np.mean(per_seed_transfer))
    log.info(
        "P5a QSVM white-box PGD: clean %.3f -> %.3f +/- %.3f at %.2f sigma "
        "(retention %.3f); transferred %.3f (retention %.3f); n_attack_seeds=%d",
        clean,
        wb_half["mean"],
        wb_half["std"],
        half,
        wb_half["mean"] / clean,
        tr_mean,
        tr_mean / clean,
        len(attack_seeds),
    )
    log.info(
        "P5a classical-kernel control (RBF-SVM, same attack): clean %.3f -> %.3f +/- %.3f "
        "at %.2f sigma (retention %.3f)",
        clean_rbf,
        rbf_half["mean"],
        rbf_half["std"],
        half,
        rbf_half["mean"] / clean_rbf,
    )
    return {
        "clean_f1": clean,
        "pgd_curve": pgd_curve,
        "fgsm_curve": fgsm_curve,
        "per_seed_pgd": per_seed_pgd,
        "eps_half": half,
        "whitebox_f1_at_half": wb_half,
        "whitebox_retention_at_half": wb_half["mean"] / clean if clean else float("nan"),
        "transferred_f1_at_half": tr_mean,
        "transferred_retention_at_half": tr_mean / clean if clean else float("nan"),
        "rbf_control": {
            "clean_f1": clean_rbf,
            "pgd_curve": rbf_curve,
            "per_seed_pgd": per_seed_rbf,
            "whitebox_f1_at_half": rbf_half,
            "whitebox_retention_at_half": (
                rbf_half["mean"] / clean_rbf if clean_rbf else float("nan")
            ),
        },
        "attack_seeds": list(attack_seeds),
        "model_is_deterministic": True,
        "numeric_gate": gate,
        "n_eval": int(len(ev_idx)),
    }


def p7b_kernel_exhaustion(reg, args, cfg_adv, best_params, subset_cap, seed=0):
    """Why the two kernels diverge: the classical one exhausts, the quantum one does not.

    Local sensitivity does NOT explain the gap (P7a, killed at its pre-registered line:
    the kernels are 2.4x apart and move almost identically under a single small step).
    The divergence is at large radius. Sweeping epsilon well past the attack grid and
    measuring the kernel itself separates two failure modes:

      DISABLED  - k(x,x_i) -> 0, the decision reverts to the intercept, predicted
                  probability becomes constant, macro-F1 returns to the constant-predictor
                  floor. The attacker owns an off-switch. (tuned RBF: exp decay, finite
                  length scale.)
      INVERTED  - k stays O(1) because the fidelity kernel is bounded in [0,1] and
                  PERIODIC in the encoding angles, so it never decays; |f| grows and
                  macro-F1 sits far BELOW the floor. The attacker keeps steering
                  authority at every radius.

    Scope: one feature map and one tuned classical kernel; a deliberately wide RBF would
    exhaust more slowly. Reported, not generalised.
    """
    zoo = FittedZoo(reg, args.encoding, args.qubits, seed, subset_cap, best_params)
    svc_q, svc_r = zoo.models["qsvm"], zoo.models["rbf_svm"]
    bw = best_params.get("qk_bw", 1.0)
    wb = QSVMWhiteBox(svc_q, zoo._states_tr, zoo._angle, args.encoding, bw)
    gp_r = GradProvider(svc_r, "rbf_svm")
    gamma = svc_r._gamma if hasattr(svc_r, "_gamma") else svc_r.gamma
    states_tr = np.asarray(zoo._states_tr)
    scale = reg.feature_scale
    pgd_cfg = cfg_adv["evasion"]["pgd"]

    n_eval = cfg_adv["blackbox"]["eval_subset"] * 2
    idx = stratified_cap(reg.X["test"], reg.y["test"], n_eval, seed=0)
    X, y = reg.X["test"][idx], reg.y["test"][idx]
    maj = int(np.round(y.mean()))
    floor = _f1(y, np.column_stack([1 - np.full(len(y), maj), np.full(len(y), maj)]).astype(float))

    def k_full_q(Z):
        states = compute_states(zoo._angle.transform(Z), args.encoding, bw)
        return fidelity_kernel(states, states_tr)

    # P7c: the VQC shares the bounded, periodic angle encoding but reaches its decision
    # through a capped head (scale * <PauliZ> + bias, expectation in [-1, 1]), so it
    # CANNOT gain decision magnitude the way the QSVM does. Its analogue of "kernel mass"
    # is |<PauliZ>|: the quantity that would have to collapse for the model to exhaust.
    vqc = zoo.models["vqc"]
    gp_vqc = zoo.grad_provider("vqc")

    def _expvals(Z):
        import torch

        A = torch.as_tensor(zoo._angle.transform(Z), dtype=torch.float64)
        with torch.no_grad():
            return vqc._expvals(A, vqc.weights).numpy()

    def k_rows(name, Z):
        if name == "qsvm":
            return k_full_q(Z)[:, svc_q.support_]
        if name == "vqc":
            return np.abs(_expvals(Z))
        d2 = ((Z[:, None, :] - svc_r.support_vectors_[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-gamma * d2)

    def decision(name, Z):
        if name == "qsvm":
            return svc_q.decision_function(k_full_q(Z))
        if name == "vqc":
            p = zoo.predict_proba("vqc", Z)[:, 1].clip(1e-9, 1 - 1e-9)
            return np.log(p / (1 - p))  # recover the logit the head produced
        return svc_r.decision_function(Z)

    # --- P7a, retained as the killed alternative hypothesis -------------------
    # sigma-weighted L-inf sensitivity S(x) = sum_j |df/dx_j| sigma_j, reported in
    # PROBABILITY space (bounded [0,1]) so the cross-model ranking is scale-free.
    # It separates the MLP from the kernels but NOT the two kernels from each other,
    # which is why the mechanism below is exhaustion rather than local sharpness.
    import torch

    def grad_f_qsvm(Z):
        xb = torch.tensor(Z, dtype=torch.float64, requires_grad=True)
        (g,) = torch.autograd.grad(wb._decision_t(xb).sum(), xb)
        return g.numpy()

    def grad_f_rbf(Z):
        d2 = ((Z[:, None, :] - svc_r.support_vectors_[None, :, :]) ** 2).sum(axis=2)
        w = (svc_r.dual_coef_[0][None, :] * np.exp(-gamma * d2)) * (-2.0 * gamma)
        return w.sum(axis=1)[:, None] * Z - w @ svc_r.support_vectors_

    def grad_f_mlp(Z):
        # dL/dx = (p - y) df/dx, so evaluating at y=0 and y=1 and differencing isolates df/dx
        m = zoo.models["mlp"]
        return mlp_loss_input_grad(m, Z, np.zeros(len(Z), dtype=int)) - mlp_loss_input_grad(
            m, Z, np.ones(len(Z), dtype=int)
        )

    sens = {}
    for nm, gfn in (("qsvm", grad_f_qsvm), ("rbf_svm", grad_f_rbf), ("mlp", grad_f_mlp)):
        p = zoo.predict_proba(nm, X)[:, 1]
        s_dec = (np.abs(gfn(X)) * scale).sum(axis=1)
        sens[nm] = {
            "median_sensitivity_decision": float(np.median(s_dec)),
            "median_sensitivity_probability": float(np.median(s_dec * p * (1 - p))),
        }
    small = min(cfg_adv["evasion"]["epsilons"])
    gp_mlp_s = GradProvider(zoo.models["mlp"], "mlp")
    for nm, gp in (("qsvm", wb), ("rbf_svm", gp_r), ("mlp", gp_mlp_s)):
        Xa = fgsm(gp, X, y, small, scale)
        b, a = zoo.predict_proba(nm, X)[:, 1], zoo.predict_proba(nm, Xa)[:, 1]
        sens[nm]["single_step_mean_abs_dproba"] = float(np.abs(a - b).mean())
        sens[nm]["single_step_flip_rate"] = float(np.mean((a > 0.5) != (b > 0.5)))
    ratio = (
        sens["qsvm"]["median_sensitivity_probability"]
        / sens["rbf_svm"]["median_sensitivity_probability"]
    )
    log.info(
        "P7a local sensitivity (probability space): qsvm %.3f, rbf %.3f, mlp %.3f "
        "-> qsvm/rbf = %.1fx; single-step |dp| %.3f vs %.3f at eps=%.2f",
        sens["qsvm"]["median_sensitivity_probability"],
        sens["rbf_svm"]["median_sensitivity_probability"],
        sens["mlp"]["median_sensitivity_probability"],
        ratio,
        sens["qsvm"]["single_step_mean_abs_dproba"],
        sens["rbf_svm"]["single_step_mean_abs_dproba"],
        small,
    )

    radii = sorted(set(cfg_adv["evasion"]["epsilons"]) | {1.0, 2.0})
    out = {
        "constant_predictor_floor": floor,
        "n_eval": int(len(idx)),
        "local_sensitivity": sens,
        "sensitivity_ratio_qsvm_over_rbf": float(ratio),
        "single_step_epsilon": float(small),
        "per_model": {},
    }
    for name, gp in (("qsvm", wb), ("rbf_svm", gp_r), ("vqc", gp_vqc)):
        K0, f0 = k_rows(name, X), np.abs(decision(name, X))
        rows = {}
        for e in [0.0] + radii:
            Xa = (
                X
                if e == 0.0
                else pgd(
                    gp,
                    X,
                    y,
                    e,
                    scale,
                    n_steps=pgd_cfg["n_steps"],
                    step_frac=pgd_cfg["step_frac"],
                    rng=spawn_rng(seed, f"p7b_{name}_{e}"),
                )
            )
            K, f = k_rows(name, Xa), np.abs(decision(name, Xa))
            p = zoo.predict_proba(name, Xa)[:, 1]
            f1 = _f1(y, zoo.predict_proba(name, Xa))
            rows[str(e)] = {
                "kernel_mass_ratio": float(K.mean() / K0.mean()),
                "decision_magnitude_ratio": float(f.mean() / f0.mean()),
                "proba_std": float(p.std()),
                "macro_f1": f1,
                "f1_minus_floor": float(f1 - floor),
            }
        out["per_model"][name] = rows
        last = rows[str(radii[-1])]
        log.info(
            "P7b %s at %.1f sigma: bounded-output mass %.3f of clean, |f| %.2fx, "
            "proba std %.4f, F1 %.3f (floor %+.3f)",
            name,
            radii[-1],
            last["kernel_mass_ratio"],
            last["decision_magnitude_ratio"],
            last["proba_std"],
            last["macro_f1"],
            last["f1_minus_floor"],
        )
    return out


def p8a_lengthscale_control(
    reg, cfg_adv, best_params, subset_cap, seed=0, gamma_mults=None, attack_seeds=(0,)
):
    """Is exhaustion a CLASSICAL property, or a length-scale property? (P8a)

    The mechanism claim -- "the classical kernel can be exhausted, the quantum models
    cannot" -- is vulnerable to one obvious reading: the tuned RBF exhausts because its
    length scale sits INSIDE the attack budget, not because it is classical. Widening the
    kernel is the control that separates those. Only gamma moves; C, the subset, the
    evaluation points, the radii, and the PGD budget are all held.

    Clean macro-F1 is reported per width so a cell that "survives" only because it never
    learned anything is visible rather than counted as evidence -- that is the failure
    mode this control would otherwise introduce.
    """
    sub = kernel_subset(reg, subset_cap, seed=0)
    Xtr, ytr = reg.X["train"][sub], reg.y["train"][sub]
    scale = reg.feature_scale
    pgd_cfg = cfg_adv["evasion"]["pgd"]
    radii = sorted(set(cfg_adv["evasion"]["epsilons"]) | {1.0, 2.0})

    n_eval = cfg_adv["blackbox"]["eval_subset"] * 2
    idx = stratified_cap(reg.X["test"], reg.y["test"], n_eval, seed=0)
    X, y = reg.X["test"][idx], reg.y["test"][idx]
    maj = int(np.round(y.mean()))
    floor = _f1(y, np.column_stack([1 - np.full(len(y), maj), np.full(len(y), maj)]).astype(float))

    tuned = best_params["rbf_svm"]
    out = {"constant_predictor_floor": floor, "n_eval": int(len(idx)), "per_width": {}}
    for mult in gamma_mults or (1.0, 0.1, 0.01, 0.001):
        set_all_seeds(seed)
        est = build("rbf_svm", {**tuned, "gamma": tuned["gamma"] * mult}, seed)
        fit_model("rbf_svm", est, Xtr, ytr)
        g = est._gamma if hasattr(est, "_gamma") else est.gamma
        gp = GradProvider(est, "rbf_svm")

        def k_rows(Z, sv=est.support_vectors_, gg=g):
            return np.exp(-gg * ((Z[:, None, :] - sv[None, :, :]) ** 2).sum(axis=2))

        K0 = k_rows(X)
        f0 = np.abs(est.decision_function(X))
        rows = {}
        for e in [0.0] + radii:
            # SVC is deterministic given the fixed subset, so the only stochastic element
            # is PGD's random start (same situation as P5a). Spread is over ATTACK seeds
            # with the model held fixed; at e=0 there is no attack, hence no spread, and
            # the clean row's std is 0 by construction rather than by measurement.
            reps = [X] if e == 0.0 else []
            for a_seed in attack_seeds:
                if e == 0.0:
                    break
                reps.append(
                    pgd(
                        gp,
                        X,
                        y,
                        e,
                        scale,
                        n_steps=pgd_cfg["n_steps"],
                        step_frac=pgd_cfg["step_frac"],
                        rng=spawn_rng(a_seed, f"p8a_{mult}_{e}"),
                    )
                )
            stats = {
                "kernel_mass_ratio": [],
                "decision_magnitude_ratio": [],
                "proba_std": [],
                "macro_f1": [],
            }
            for Xa in reps:
                proba, _ = predict_proba_timed(est, Xa)
                stats["kernel_mass_ratio"].append(float(k_rows(Xa).mean() / K0.mean()))
                stats["decision_magnitude_ratio"].append(
                    float(np.abs(est.decision_function(Xa)).mean() / f0.mean())
                )
                stats["proba_std"].append(float(proba[:, 1].std()))
                stats["macro_f1"].append(_f1(y, proba))
            rows[str(e)] = {k: float(np.mean(v)) for k, v in stats.items()}
            rows[str(e)]["macro_f1_std"] = (
                float(np.std(stats["macro_f1"], ddof=1)) if len(reps) > 1 else 0.0
            )
            rows[str(e)]["f1_minus_floor"] = float(rows[str(e)]["macro_f1"] - floor)
            rows[str(e)]["n_attack_seeds"] = len(reps) if e != 0.0 else 0
        out["per_width"][str(mult)] = {"gamma": float(g), "gamma_multiplier": mult, "curve": rows}
        last, clean = rows[str(radii[-1])], rows["0.0"]
        log.info(
            "P8a rbf gamma x%.3g (gamma=%.4f): clean F1 %.3f -> at %.1f sigma mass %.3f, "
            "proba std %.4f, F1 %.3f +/- %.3f (floor %+.3f)",
            mult,
            g,
            clean["macro_f1"],
            radii[-1],
            last["kernel_mass_ratio"],
            last["proba_std"],
            last["macro_f1"],
            last["macro_f1_std"],
            last["f1_minus_floor"],
        )
    return out


def p9a_quantum_bandwidth_regime(
    reg, args, cfg_adv, cfg_q, best_params, subset_cap, seed=0, attack_seeds=(0,)
):
    """The SYMMETRIC control to P8a: sweep the QUANTUM kernel's own tuning knob (P9a).

    P8a swept the classical kernel's bandwidth over three decades and found it spans both
    end states, which is why the paper's claim is now "length scale, not family". The
    surviving asymmetry -- that the defender has a tuning choice classically and none
    quantumly -- was asserted from fixed configurations and never tested along the quantum
    side's own knob. This runs that test.

    Bandwidth scales the encoding angles. Because the fidelity kernel is bounded in [0,1]
    and PERIODIC in those angles it has no decay regime, so the prediction is that no
    setting reaches DISABLED. The degeneracy confound is explicit: at large bandwidth the
    kernel fragments toward the ~2^-n random-state baseline and the model collapses to
    chance, which would satisfy "macro-F1 at the floor" for a reason unrelated to the
    mechanism. Clean macro-F1 is therefore recorded per bandwidth and the non-degeneracy
    precondition is applied when the end state is read, never silently.
    """
    sub = kernel_subset(reg, subset_cap, seed=0)
    Xtr, ytr = reg.X["train"][sub], reg.y["train"][sub]
    angle = reg.angle_scaler
    A_tr = angle.transform(Xtr)
    scale = reg.feature_scale
    pgd_cfg = cfg_adv["evasion"]["pgd"]
    radii = sorted(set(cfg_adv["evasion"]["epsilons"]) | {1.0, 2.0})
    C = best_params.get("qk_C", 1.0)

    n_eval = cfg_adv["blackbox"]["eval_subset"] * 2
    idx = stratified_cap(reg.X["test"], reg.y["test"], n_eval, seed=0)
    X, y = reg.X["test"][idx], reg.y["test"][idx]
    maj = int(np.round(y.mean()))
    floor = _f1(y, np.column_stack([1 - np.full(len(y), maj), np.full(len(y), maj)]).astype(float))
    gate_idx = stratified_cap(reg.X["test"], reg.y["test"], 40, seed=0)
    X_gate = reg.X["test"][gate_idx]

    grid = sorted(
        set(cfg_q["ablations"]["bandwidth_curve"]["bandwidths"])
        | {float(best_params.get("qk_bw", 1.0))}
    )
    out = {
        "constant_predictor_floor": floor,
        "n_eval": int(len(idx)),
        "tuned_bandwidth": float(best_params.get("qk_bw", 1.0)),
        "svm_C": float(C),
        "per_bandwidth": {},
    }
    for bw in grid:
        set_all_seeds(seed)
        states_tr = compute_states(A_tr, args.encoding, bandwidth=bw)
        svc = _svc_on_kernel(2, C).fit(fidelity_kernel(states_tr, states_tr), ytr)
        wb = QSVMWhiteBox(svc, states_tr, angle, args.encoding, bw)
        # Each bandwidth is a NEW decision function -> re-gate before trusting its numbers.
        # deployed_tol=None because this stage evaluates through the EXACT complex128 Gram
        # below, matching the attack path: at small bandwidth the near-degenerate Gram
        # amplifies complex64 accumulation into the decision value (1.9e-2 at bw 0.05), and
        # this stage reports decision-magnitude quantities. The strict exact-math check and
        # the prediction-agreement check both still block; the discrepancy is recorded.
        gate = assert_matches_reference(wb, svc, states_tr, angle, X_gate, deployed_tol=None)

        def _gram(Z, st=states_tr, b=bw):
            return fidelity_kernel(
                compute_states(angle.transform(Z), args.encoding, b), st, exact=True
            )

        def proba(Z, s=svc):
            return _svc_proba(s, _gram(Z), 2)

        def k_rows(Z, s=svc):
            return _gram(Z)[:, s.support_]

        K0 = k_rows(X)
        rows = {}
        for e in [0.0] + radii:
            reps = (
                [X]
                if e == 0.0
                else [
                    pgd(
                        wb,
                        X,
                        y,
                        e,
                        scale,
                        n_steps=pgd_cfg["n_steps"],
                        step_frac=pgd_cfg["step_frac"],
                        rng=spawn_rng(a, f"p9a_{bw}_{e}"),
                    )
                    for a in attack_seeds
                ]
            )
            mass, pstd, f1s = [], [], []
            for Xa in reps:
                p = proba(Xa)
                mass.append(float(k_rows(Xa).mean() / K0.mean()))
                pstd.append(float(p[:, 1].std()))
                f1s.append(_f1(y, p))
            rows[str(e)] = {
                "kernel_mass_ratio": float(np.mean(mass)),
                "proba_std": float(np.mean(pstd)),
                "macro_f1": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                "f1_minus_floor": float(np.mean(f1s) - floor),
            }
        clean, last = rows["0.0"], rows[str(radii[-1])]
        # a cell that never rose above the floor cannot be read as "the attacker disabled
        # it" -- pre-registered precondition, recorded per cell rather than applied silently
        degenerate = bool(clean["f1_minus_floor"] < 0.05)
        out["per_bandwidth"][str(bw)] = {
            "curve": rows,
            "clean_mean_kernel_mass": float(K0.mean()),
            "degenerate": degenerate,
            "numeric_gate": gate,
        }
        log.info(
            "P9a bw %.3g: clean F1 %.3f (floor %+.3f, degenerate=%s) -> at %.1f sigma "
            "mass %.3f, proba std %.4f, F1 %.3f +/- %.3f (floor %+.3f)",
            bw,
            clean["macro_f1"],
            clean["f1_minus_floor"],
            degenerate,
            radii[-1],
            last["kernel_mass_ratio"],
            last["proba_std"],
            last["macro_f1"],
            last["macro_f1_std"],
            last["f1_minus_floor"],
        )
    return out


def p6a_fullfeature_mlp(args, cfg_adv, full_mlp_params, seeds=(0,)):
    """White-box PGD on a full-feature MLP — deployment-grade robustness anchor.

    Model refit per seed (unlike the deterministic QSVM), so the spread covers
    both training and attack randomness, matching the table's seeded protocol.
    """
    reg = prepare_regime(args.variant, "full", seed=0)
    ev_idx = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    Xte, yte = reg.X["test"][ev_idx], reg.y["test"][ev_idx]
    scale = reg.feature_scale
    eps_grid = cfg_adv["evasion"]["epsilons"]
    pgd_cfg = cfg_adv["evasion"]["pgd"]

    per_seed_clean, per_seed_curve = [], {str(e): [] for e in eps_grid}
    for s in seeds:
        set_all_seeds(s)
        mlp = build("mlp", full_mlp_params, s)
        fit_model("mlp", mlp, reg.X["train"], reg.y["train"])
        gp = GradProvider(mlp, "mlp")
        rng = spawn_rng(s, "p6a")
        per_seed_clean.append(_f1(yte, mlp.predict_proba(Xte)))
        for e in eps_grid:
            Xa = pgd(
                gp,
                Xte,
                yte,
                e,
                scale,
                n_steps=pgd_cfg["n_steps"],
                step_frac=pgd_cfg["step_frac"],
                rng=rng,
            )
            per_seed_curve[str(e)].append(_f1(yte, mlp.predict_proba(Xa)))

    def agg(v):
        return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}

    clean = agg(per_seed_clean)
    curve = {k: agg(v) for k, v in per_seed_curve.items()}
    half = 0.5 if 0.5 in eps_grid else max(eps_grid)
    log.info(
        "P6a full-feature MLP: clean %.3f +/- %.3f -> %.3f +/- %.3f at %.2f sigma "
        "(retention %.3f, %d seeds)",
        clean["mean"],
        clean["std"],
        curve[str(half)]["mean"],
        curve[str(half)]["std"],
        half,
        curve[str(half)]["mean"] / clean["mean"],
        len(seeds),
    )
    return {
        "clean_f1": clean,
        "pgd_curve": curve,
        "per_seed_pgd": per_seed_curve,
        "eps_half": half,
        "retention_at_half": curve[str(half)]["mean"] / clean["mean"],
        "n_features": int(reg.X["train"].shape[1]),
        "seeds": list(seeds),
        "n_eval": int(len(ev_idx)),
    }


def _mass_descent(grad_fn, X, eps, scale, rng, n_steps, step_frac):
    """L-inf projected gradient DESCENT on kernel mass — evasion.pgd with the sign flipped.

    The paper's attacks all ASCEND classification loss. Classically that implies distance (the
    RBF decays), so the kernel row collapses as a side effect. Quantumly it does not: a bounded,
    periodic kernel lets the attacker cross into the opposite class while staying kernel-close.
    This is the objective the loss attack never expressed.
    """
    alpha, lo, hi = step_frac * eps * scale, X - eps * scale, X + eps * scale
    Xa = X + rng.uniform(-1.0, 1.0, size=X.shape) * eps * scale
    for _ in range(n_steps):
        Xa = np.clip(Xa - alpha * np.sign(grad_fn(Xa)), lo, hi)
    return Xa


def _qsvm_mass_grad(wb, Z):
    """d/dx of the mean fidelity to the support set. Mean over SVs, SUM over the batch, so each
    row's gradient is the true d(row mass)/dx and is batch-size independent (a batch MEAN here
    silently rescales by 1/B, which sign() hides from the attack but not from a gradient check)."""
    import torch

    g = np.empty_like(Z)
    for s in range(0, len(Z), wb.chunk):
        xb = torch.tensor(Z[s : s + wb.chunk], dtype=torch.float64, requires_grad=True)
        ang = torch.clamp(xb * wb._a_scale + wb._a_min, -np.pi, np.pi)
        psi = wb._qnode(wb.bandwidth * ang).reshape(-1, 2**wb.n_qubits).to(torch.complex128)
        m = ((psi.conj() @ wb._psi_sv.T).abs() ** 2).real.mean(axis=1).sum()
        (gr,) = torch.autograd.grad(m, xb)
        g[s : s + wb.chunk] = gr.detach().numpy()
    return g


def _end_state(dec_clean, dec_adv, y, floor):
    """Scale-free end-state read. proba = sigmoid(decision), so proba spread is NOT comparable
    across C; the decision-spread RATIO against the same model's own clean spread is, and the
    modal-prediction fraction is the paper's own "predictions constant" wording made numeric."""
    pred = (dec_adv > 0).astype(int)
    p1 = 1.0 / (1.0 + np.exp(-dec_adv))
    f1 = _f1(y, np.column_stack([1 - p1, p1]))
    return {
        "dec_spread_ratio": float(dec_adv.std() / max(dec_clean.std(), 1e-300)),
        "modal_frac": float(max(pred.mean(), 1.0 - pred.mean())),
        "macro_f1": float(f1),
        "f1_minus_floor": float(f1 - floor),
    }


def p11c_objective_and_regularization(
    reg, args, cfg_adv, cfg_q, best_params, subset_cap, seed=0, c_grid=(1.0, 10.0, 100.0)
):
    """Does the fidelity kernel's non-decay belong to the ENCODING, or to the attack objective
    and the fit? (P11c + P11d, pre-registered 2026-08-15T05:40Z and 06:20Z.)

    P9a established that no bandwidth drives the fidelity kernel to the disabled end state, and
    the paper attributed that to the encoding being bounded and periodic. Every attack behind
    that attribution ASCENDS classification loss. Part A re-runs the identical protocol with the
    objective as the only variable; part B holds the attacked mass fixed and sweeps the SVM's
    regularization instead. Together they relocate the mechanism from the encoding to
    objective x regularization -- the same reframing P8a forced on the classical side.

    Controls (all blocking or reported, never silent): the classical RBF under the identical
    mass-minimizing optimizer is the POSITIVE control (its baseline is reachable, so failure to
    reach it means the optimizer is broken); random perturbation at matched radius is the
    NEGATIVE control; the unrelated-pair baseline is pre-committed to column-permuted eval points
    with the uniform-angle baseline reported as sensitivity only.
    """
    pgd_cfg = cfg_adv["evasion"]["pgd"]
    n_steps, step_frac = pgd_cfg["n_steps"], pgd_cfg["step_frac"]
    radii = sorted(set(cfg_adv["evasion"]["epsilons"]) | {1.0, 2.0})
    paper_budget = max(radii)
    sub = kernel_subset(reg, subset_cap, seed=0)
    Xtr, ytr = reg.X["train"][sub], reg.y["train"][sub]
    angle, scale = reg.angle_scaler, np.asarray(reg.feature_scale, dtype=np.float64)
    A_tr = angle.transform(Xtr)
    idx = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["blackbox"]["eval_subset"] * 2, 0)
    X, y = reg.X["test"][idx], reg.y["test"][idx]
    maj = int(np.round(y.mean()))
    floor = _f1(y, np.column_stack([1 - np.full(len(y), maj), np.full(len(y), maj)]).astype(float))
    C_tuned, bw_tuned = float(best_params["qk_C"]), float(best_params["qk_bw"])

    rng = np.random.default_rng(seed)
    # PRE-COMMITTED baseline: per-feature marginals preserved, joint structure destroyed.
    X_perm = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])

    # ---- POSITIVE CONTROL (blocking): the classical RBF, whose baseline is reachable ----------
    g_rbf = best_params["rbf_svm"]["gamma"]
    est = build("rbf_svm", best_params["rbf_svm"], 0)
    fit_model("rbf_svm", est, Xtr, ytr)
    SV = est.support_vectors_

    def rbf_K(Z):
        return np.exp(-g_rbf * ((Z[:, None, :] - SV[None, :, :]) ** 2).sum(axis=2))

    def rbf_grad(Z):
        d = Z[:, None, :] - SV[None, :, :]
        return (rbf_K(Z)[:, :, None] * (-2.0 * g_rbf) * d).mean(axis=1)

    K0c = rbf_K(X)
    dec0c = K0c @ est.dual_coef_[0] + est.intercept_[0]
    Kc = rbf_K(
        _mass_descent(
            rbf_grad, X, paper_budget, scale, np.random.default_rng(1), n_steps, step_frac
        )
    )
    control = {
        "mass_ratio": float(Kc.mean() / K0c.mean()),
        "baseline_permuted": float(rbf_K(X_perm).mean() / K0c.mean()),
        **_end_state(dec0c, Kc @ est.dual_coef_[0] + est.intercept_[0], y, floor),
    }
    if control["mass_ratio"] > control["baseline_permuted"] + 0.05:
        raise RuntimeError(
            "POSITIVE CONTROL FAILED: the mass-minimizing optimizer did not drive the classical "
            f"RBF to its own baseline ({control['mass_ratio']:.4f} vs "
            f"{control['baseline_permuted']:.4f}). The optimizer is broken; no quantum number "
            "from this stage may be read."
        )
    log.info(
        "P11c positive control passed: RBF mass %.4f -> spread %.4f, modal %.3f, F1 %+.3f vs floor",
        control["mass_ratio"],
        control["dec_spread_ratio"],
        control["modal_frac"],
        control["f1_minus_floor"],
    )

    out = {
        "constant_predictor_floor": floor,
        "n_eval": int(len(idx)),
        "paper_budget_sigma": paper_budget,
        "n_steps": n_steps,
        "tuned_bandwidth": bw_tuned,
        "tuned_C": C_tuned,
        "classical_positive_control": control,
        "part_a_objective": {},
        "part_b_regularization": {},
    }

    # ---- PART A: objective is the ONLY variable, swept across bandwidth ----------------------
    grid = sorted(set(cfg_q["ablations"]["bandwidth_curve"]["bandwidths"]) | {bw_tuned})
    for bw in grid:
        set_all_seeds(seed)
        st = compute_states(A_tr, args.encoding, bandwidth=bw)
        svc = _svc_on_kernel(2, C_tuned).fit(fidelity_kernel(st, st), ytr)
        wb = QSVMWhiteBox(svc, st, angle, args.encoding, bw)
        # deployed_tol=None: this stage evaluates through the EXACT complex128 Gram, matching the
        # attack path (see P9a). The strict exact check and prediction agreement both still block.
        gate = assert_matches_reference(wb, svc, st, angle, X[:40], deployed_tol=None)
        sv_states = st[svc.support_]

        def q_mass(Z, s=sv_states, b=bw):
            psi = compute_states(angle.transform(Z), args.encoding, b)
            return float(fidelity_kernel(psi, s, exact=True).mean())

        m0 = q_mass(X)
        ua = rng.uniform(-np.pi, np.pi, size=(len(X), wb.n_qubits))
        cell = {
            "clean_mean_mass": m0,
            "baseline_permuted": q_mass(X_perm) / m0,
            "baseline_uniform_angles": float(
                fidelity_kernel(compute_states(ua, args.encoding, bw), sv_states, True).mean() / m0
            ),
            "random_state_floor": 2.0**-wb.n_qubits,
            "numeric_gate": gate,
            "curve": {},
        }
        for e in radii:
            r = np.random.default_rng(1)
            mass_obj = q_mass(
                _mass_descent(
                    lambda Z, w=wb: _qsvm_mass_grad(w, Z), X, e, scale, r, n_steps, step_frac
                )
            )
            loss_obj = q_mass(
                pgd(
                    wb,
                    X,
                    y,
                    e,
                    scale,
                    n_steps=n_steps,
                    step_frac=step_frac,
                    rng=np.random.default_rng(1),
                )
            )
            rnd = q_mass(X + np.random.default_rng(2).uniform(-1, 1, X.shape) * e * scale)
            cell["curve"][str(e)] = {
                "mass_objective": mass_obj / m0,
                "loss_objective": loss_obj / m0,
                "random_control": rnd / m0,
                "beats_random": bool(mass_obj < rnd),
            }
        c = cell["curve"][str(paper_budget)]
        log.info(
            "P11c bw %.3g: at %.1f sigma mass-objective %.4f vs loss-objective %.4f "
            "(baseline %.4f) -> %.1fx lower, beats random=%s",
            bw,
            paper_budget,
            c["mass_objective"],
            c["loss_objective"],
            cell["baseline_permuted"],
            c["loss_objective"] / max(c["mass_objective"], 1e-12),
            c["beats_random"],
        )
        out["part_a_objective"][str(bw)] = cell

    # ---- PART B: hold the attacked mass fixed, sweep the fit's regularization -----------------
    set_all_seeds(seed)
    st = compute_states(A_tr, args.encoding, bandwidth=bw_tuned)
    Ktr = fidelity_kernel(st, st)
    for C in sorted({*c_grid, C_tuned}):
        svc = _svc_on_kernel(2, C).fit(Ktr, ytr)
        wb = QSVMWhiteBox(svc, st, angle, args.encoding, bw_tuned)
        duals, sup = svc.dual_coef_[0], svc.support_
        bound = C * float(np.max(getattr(svc, "class_weight_", np.ones(2))))

        def full_gram(Z, b=bw_tuned):
            return fidelity_kernel(
                compute_states(angle.transform(Z), args.encoding, b), st, exact=True
            )

        K0 = full_gram(X)
        dec0 = svc.decision_function(K0)
        clean = _end_state(dec0, dec0, y, floor)
        per_seed = []
        for s in (1, 2, 3):  # the end-state read is seed-sensitive; 3 attack seeds, spread reported
            Ka = full_gram(
                _mass_descent(
                    lambda Z, w=wb: _qsvm_mass_grad(w, Z),
                    X,
                    paper_budget,
                    scale,
                    np.random.default_rng(s),
                    n_steps,
                    step_frac,
                )
            )
            per_seed.append(
                {
                    "seed": s,
                    "mass_ratio": float(Ka[:, sup].mean() / K0[:, sup].mean()),
                    **_end_state(dec0, svc.decision_function(Ka), y, floor),
                }
            )
        agg = {
            k: {
                "mean": float(np.mean([p[k] for p in per_seed])),
                "std": float(np.std([p[k] for p in per_seed], ddof=1)),
            }
            for k in ("mass_ratio", "dec_spread_ratio", "modal_frac", "f1_minus_floor")
        }
        out["part_b_regularization"][str(C)] = {
            "clean_macro_f1": clean["macro_f1"],
            "clean_f1_minus_floor": clean["f1_minus_floor"],
            # the clean model must NOT already be constant, or the attacked reading is vacuous
            "clean_modal_frac": clean["modal_frac"],
            "degenerate": bool(clean["f1_minus_floor"] < 0.05),
            "max_abs_dual": float(np.abs(duals).max()),
            "dual_bound": bound,
            "frac_duals_pinned": float(np.mean(np.abs(np.abs(duals) - bound) < 1e-6 * bound)),
            "n_support": int(len(duals)),
            "per_seed": per_seed,
            "mean": agg,
            "disabled": bool(
                agg["modal_frac"]["mean"] > 0.99 and abs(agg["f1_minus_floor"]["mean"]) < 0.05
            ),
        }
        log.info(
            "P11c/d C %.4g: clean F1 %+.3f vs floor (modal %.3f) | max|dual| %.4g of bound %.4g "
            "(%.0f%% pinned) -> mass %.4f, spread %.4f, modal %.3f, F1 %+.3f, disabled=%s",
            C,
            clean["f1_minus_floor"],
            clean["modal_frac"],
            np.abs(duals).max(),
            bound,
            100 * out["part_b_regularization"][str(C)]["frac_duals_pinned"],
            agg["mass_ratio"]["mean"],
            agg["dec_spread_ratio"]["mean"],
            agg["modal_frac"]["mean"],
            agg["f1_minus_floor"]["mean"],
            out["part_b_regularization"][str(C)]["disabled"],
        )
    return out


def p14a_objective_control(reg, args, cfg_adv, best_params, subset_cap, seed=0, attack_seeds=(0,)):
    """Is the published retention headline objective-limited? (P14a, pre-registered 10:40Z.)

    Every retention number in this benchmark comes from PGD ascending BCE. P11c showed that an
    end-state claim is a claim about the attacker's OBJECTIVE, so the same question points back
    at the headline. Three objectives at the headline budget, everything else held:

      bce     — the published objective.
      margin  — L = -(2y-1) f(x). Analytically the SAME attack under sign-PGD, because
                dL_BCE/df = -(2y-1)|sigmoid(f) - y| differs from it by a strictly positive
                per-point scalar that sign() discards. Included because that equivalence is
                the answer to the CW-style reviewer objection, and because it is only exact
                while the BCE factor does not underflow to zero -- which is measured, not
                assumed, by `saturation_fraction`.
      mass    — descend mean kernel mass instead. This is the genuinely DIFFERENT direction
                (P11c), and therefore the real test of whether BCE under-attacks.

    Kill (pre-committed): any objective driving either model's macro-F1 more than 0.05 below
    its BCE figure means BCE is not the strongest objective and that retention number is
    relabelled an upper bound.
    """
    pgd_cfg = cfg_adv["evasion"]["pgd"]
    n_steps, step_frac = pgd_cfg["n_steps"], pgd_cfg["step_frac"]
    eps = 0.5 if 0.5 in cfg_adv["evasion"]["epsilons"] else max(cfg_adv["evasion"]["epsilons"])
    sub = kernel_subset(reg, subset_cap, seed=0)
    Xtr, ytr = reg.X["train"][sub], reg.y["train"][sub]
    angle, scale = reg.angle_scaler, np.asarray(reg.feature_scale, dtype=np.float64)
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    X, y = reg.X["test"][ev], reg.y["test"][ev]

    set_all_seeds(seed)
    states_tr = compute_states(angle.transform(Xtr), args.encoding, float(best_params["qk_bw"]))
    svc = _svc_on_kernel(2, best_params["qk_C"]).fit(fidelity_kernel(states_tr, states_tr), ytr)
    wb = QSVMWhiteBox(svc, states_tr, angle, args.encoding, float(best_params["qk_bw"]))
    gate = assert_matches_reference(wb, svc, states_tr, angle, X[:40])
    est = build("rbf_svm", best_params["rbf_svm"], 0)
    fit_model("rbf_svm", est, Xtr, ytr)
    SV, g_rbf = est.support_vectors_, best_params["rbf_svm"]["gamma"]

    def rbf_mass_grad(Z):
        d = Z[:, None, :] - SV[None, :, :]
        K = np.exp(-g_rbf * (d**2).sum(axis=2))
        return (K[:, :, None] * (-2.0 * g_rbf) * d).mean(axis=1)

    def q_proba(Z):
        K = fidelity_kernel(
            compute_states(angle.transform(Z), args.encoding, float(best_params["qk_bw"])),
            states_tr,
        )
        return _svc_proba(svc, K, 2)

    models = {
        "qsvm": {
            "proba": q_proba,
            "bce": wb,
            "margin": lambda Z, yy: wb.margin_input_grad(Z, yy),
            "mass": lambda Z: _qsvm_mass_grad(wb, Z),
            # the mechanism P14a pre-registered: measured, so a null result is explained
            "saturation": wb.saturation_fraction(X, y),
            "max_abs_decision": float(np.abs(wb.decision_function(X)).max()),
        },
        "rbf_svm": {
            # the zoo builds SVC(probability=False); probabilities are sigmoid(decision),
            # the same derivation _svc_proba applies everywhere else in the benchmark
            "proba": lambda Z: _svc_proba(est, Z, 2),
            "bce": GradProvider(est, "rbf_svm", "bce"),
            "margin": GradProvider(est, "rbf_svm", "margin"),
            "mass": rbf_mass_grad,
            "saturation": float(
                np.mean(np.abs(1 / (1 + np.exp(-est.decision_function(X))) - y) < 1e-6)
            ),
            "max_abs_decision": float(np.abs(est.decision_function(X)).max()),
        },
    }

    out = {
        "epsilon": eps,
        "n_eval": int(len(ev)),
        "n_steps": n_steps,
        "attack_seeds": list(attack_seeds),
        "numeric_gate": gate,
        "per_model": {},
    }
    for name, m in models.items():
        clean = _f1(y, m["proba"](X))
        per_obj = {}
        for obj in ("bce", "margin", "mass"):
            f1s = []
            for s in attack_seeds:
                rng = spawn_rng(s, f"p14a_{name}_{obj}")
                Xa = (
                    _mass_descent(m["mass"], X, eps, scale, rng, n_steps, step_frac)
                    if obj == "mass"
                    else pgd(
                        m[obj], X, y, eps, scale, n_steps=n_steps, step_frac=step_frac, rng=rng
                    )
                )
                f1s.append(_f1(y, m["proba"](Xa)))
            per_obj[obj] = {
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                "retention": float(np.mean(f1s) / clean) if clean else float("nan"),
                "per_seed": [float(v) for v in f1s],
            }
        base = per_obj["bce"]["macro_f1_mean"]
        worst = min(per_obj, key=lambda k: per_obj[k]["macro_f1_mean"])
        out["per_model"][name] = {
            "clean_f1": clean,
            "bce_saturation_fraction": m["saturation"],
            "max_abs_decision": m["max_abs_decision"],
            "by_objective": per_obj,
            "strongest_objective": worst,
            "delta_vs_bce": float(per_obj[worst]["macro_f1_mean"] - base),
            # pre-registered kill line: >0.05 absolute below BCE means BCE was not strongest
            "bce_is_upper_bound": bool(base - per_obj[worst]["macro_f1_mean"] > 0.05),
        }
        log.info(
            "P14a %s: clean %.3f | BCE %.3f+/-%.3f, margin %.3f+/-%.3f, mass %.3f+/-%.3f "
            "| strongest=%s (delta %+.3f) | BCE saturation %.1f%%, max|f| %.3g",
            name,
            clean,
            per_obj["bce"]["macro_f1_mean"],
            per_obj["bce"]["macro_f1_std"],
            per_obj["margin"]["macro_f1_mean"],
            per_obj["margin"]["macro_f1_std"],
            per_obj["mass"]["macro_f1_mean"],
            per_obj["mass"]["macro_f1_std"],
            worst,
            out["per_model"][name]["delta_vs_bce"],
            100 * m["saturation"],
            m["max_abs_decision"],
        )
    # the comparison the headline rests on must survive whichever objective is strongest
    q, r = out["per_model"]["qsvm"], out["per_model"]["rbf_svm"]
    qb = q["by_objective"][q["strongest_objective"]]["per_seed"]
    rb = r["by_objective"][r["strongest_objective"]]["per_seed"]
    out["headline_under_strongest"] = {
        "qsvm_f1": float(np.mean(qb)),
        "rbf_f1": float(np.mean(rb)),
        "wilcoxon": paired_wilcoxon(np.array(rb), np.array(qb)),
        "ordering_holds": bool(np.mean(qb) < np.mean(rb)),
    }
    log.info(
        "P14a headline under each model's STRONGEST objective: qsvm %.3f vs rbf %.3f, "
        "ordering holds=%s, p=%.4g",
        out["headline_under_strongest"]["qsvm_f1"],
        out["headline_under_strongest"]["rbf_f1"],
        out["headline_under_strongest"]["ordering_holds"],
        out["headline_under_strongest"]["wilcoxon"]["p_value"],
    )
    return out


def p15a_transfer_alignment(reg, args, cfg_adv, best_params, subset_cap, seed=0):
    """A mechanism for C1: does perturbation/gradient alignment explain the transfer matrix?
    (P15a, pre-registered 2026-08-15T11:45Z.)

    C1 reports that MLP-crafted attacks break the VQC harder than any classical target while
    VQC-crafted attacks barely move classical families. That is a bare matrix. A transferred
    attack damages a target exactly to the extent its displacement has a component along that
    target's own loss-ascent direction, so alignment is the mechanism stated directly:

        a(S -> T) = mean_i cos( delta_S(x_i), grad_x L_T(x_i) )

    Asymmetric by construction, which matters: principal angles between equal-dimension
    subspaces are SYMMETRIC and therefore cannot explain an asymmetry at all. Cosine is also
    scale-free, so a family with larger gradients cannot score higher for that reason alone.
    """
    from scipy.stats import spearmanr

    zoo = FittedZoo(reg, args.encoding, args.qubits, seed, subset_cap, best_params)
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    X, y = reg.X["test"][ev], reg.y["test"][ev]
    scale = np.asarray(reg.feature_scale, dtype=np.float64)
    pgd_cfg = cfg_adv["evasion"]["pgd"]
    eps = max(cfg_adv["evasion"]["epsilons"])

    wb = QSVMWhiteBox(
        zoo.models["qsvm"], zoo._states_tr, zoo._angle, args.encoding, best_params["qk_bw"]
    )
    assert_matches_reference(wb, zoo.models["qsvm"], zoo._states_tr, zoo._angle, X[:40])
    # only these four expose input gradients; trees enter the damage side but cannot supply an
    # alignment value, and their exclusion from rho is reported rather than silently dropped
    grads = {
        "mlp": zoo.grad_provider("mlp"),
        "vqc": zoo.grad_provider("vqc"),
        "rbf_svm": GradProvider(zoo.models["rbf_svm"], "rbf_svm"),
        "qsvm": wb,
    }
    clean = {t: _f1(y, zoo.predict_proba(t, X)) for t in grads}
    G = {t: g(X, y) for t, g in grads.items()}  # gradients at the CLEAN points, where the
    # attacker computes its step
    rng_seed = 11

    def cos_rows(A, B):
        num = (A * B).sum(axis=1)
        den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
        ok = den > 0
        return float(np.mean(num[ok] / den[ok]))

    deltas = {
        s: pgd(
            grads[s],
            X,
            y,
            eps,
            scale,
            n_steps=pgd_cfg["n_steps"],
            step_frac=pgd_cfg["step_frac"],
            rng=spawn_rng(rng_seed, f"p15a_{s}"),
        )
        - X
        for s in ("mlp", "vqc")
    }
    # NEGATIVE CONTROL: matched per-point norm, no gradient information
    r = np.random.default_rng(rng_seed)
    rnd = r.normal(size=X.shape)
    rnd *= np.linalg.norm(deltas["mlp"], axis=1, keepdims=True) / np.maximum(
        np.linalg.norm(rnd, axis=1, keepdims=True), 1e-300
    )
    deltas["random_control"] = rnd

    cells, out = [], {"epsilon": eps, "n_eval": int(len(ev)), "clean_f1": clean, "cells": {}}
    for s, d in deltas.items():
        Xa = X + d
        out["cells"][s] = {}
        for t in grads:
            f1 = _f1(y, zoo.predict_proba(t, Xa))
            rec = {
                "alignment": cos_rows(d, G[t]),
                "adv_f1": f1,
                "drop": clean[t] - f1,
                # retention as well as raw drop: a target already near chance cannot
                # manufacture a correlation through having more to lose (rule 2)
                "retention": f1 / clean[t] if clean[t] else float("nan"),
                "self": bool(s == t),
            }
            out["cells"][s][t] = rec
            if s != "random_control" and s != t:
                cells.append((rec["alignment"], rec["drop"], 1.0 - rec["retention"]))
    align, drop, lostfrac = (np.array([c[i] for c in cells]) for i in range(3))
    rho_drop = spearmanr(align, drop)
    rho_ret = spearmanr(align, lostfrac)
    self_ok = all(
        out["cells"][t][t]["alignment"]
        >= max(out["cells"][s][t]["alignment"] for s in deltas if s != t)
        for t in ("mlp", "vqc")
    )
    rnd_align = {t: out["cells"]["random_control"][t]["alignment"] for t in grads}
    out["controls"] = {
        "self_transfer_is_max_alignment": bool(self_ok),
        "random_control_alignment": rnd_align,
        "excluded_from_rho": ["logreg", "rf", "xgb", "lgbm"],
    }
    out["verdict"] = {
        "spearman_alignment_vs_drop": {
            "rho": float(rho_drop.statistic),
            "p": float(rho_drop.pvalue),
        },
        "spearman_alignment_vs_lost_fraction": {
            "rho": float(rho_ret.statistic),
            "p": float(rho_ret.pvalue),
        },
        "n_cells": int(len(cells)),
        # pre-committed kill line
        "explains_matrix": bool(rho_drop.statistic >= 0.5),
        "asymmetry_mlp_to_vqc": out["cells"]["mlp"]["vqc"]["alignment"],
        "asymmetry_vqc_to_mlp": out["cells"]["vqc"]["mlp"]["alignment"],
    }
    for s in deltas:
        log.info(
            "P15a alignment from %-14s -> %s",
            s,
            {t: round(out["cells"][s][t]["alignment"], 3) for t in grads},
        )
    log.info(
        "P15a verdict: rho(alignment, drop) = %+.3f (p=%.4g, n=%d) -> %s | "
        "MLP->VQC %.3f vs VQC->MLP %.3f | self-max=%s | random control %s",
        rho_drop.statistic,
        rho_drop.pvalue,
        len(cells),
        "EXPLAINS" if out["verdict"]["explains_matrix"] else "KILLED (bar +0.5)",
        out["verdict"]["asymmetry_mlp_to_vqc"],
        out["verdict"]["asymmetry_vqc_to_mlp"],
        self_ok,
        {k: round(v, 3) for k, v in rnd_align.items()},
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="zz")  # the robustness-table QSVM config
    ap.add_argument("--best-params", default="results/adv_best_params_binary_q8.json")
    ap.add_argument(
        "--full-mlp-run",
        default="results/runs/20260813T081418Z_classical_binary_full",
        help="classical full-feature run dir supplying the tuned MLP params for P6a",
    )
    ap.add_argument("--only", nargs="*", default=["p5a", "p6a", "p7b"])
    ap.add_argument(
        "--gamma-mults",
        type=float,
        nargs="*",
        default=None,
        help="P8a: RBF gamma multipliers relative to the tuned value "
        "(default: three decades, 1 / 0.1 / 0.01 / 0.001).",
    )
    ap.add_argument(
        "--c-grid",
        type=float,
        nargs="*",
        default=[1.0, 10.0, 100.0],
        help="P11c part B: SVM regularization values swept at the tuned bandwidth "
        "(the tuned C is always appended).",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=list(range(10)),
        help="P5a: PGD random-start seeds (QSVM itself is deterministic). "
        "P6a: full model refit + attack seeds.",
    )
    args = ap.parse_args()
    do = set(args.only)

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    subset_cap = cfg_q["qkernel"]["subset_cap"]
    best_params = json.loads((REPO_ROOT / args.best_params).read_text())

    tag = f"{args.variant}_q{args.qubits}"
    with RunDir(f"whitebox_qsvm_{tag}", config=vars(args), seeds=args.seeds) as run:
        out = {}
        if "p5a" in do:
            reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
            out["p5a_whitebox_qsvm"] = p5a_whitebox_qsvm(
                reg, args, cfg_adv, best_params, subset_cap, seed=0, attack_seeds=args.seeds
            )
            run.write_json("results_partial.json", out)
        if "p7b" in do:
            reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
            out["p7b_kernel_exhaustion"] = p7b_kernel_exhaustion(
                reg, args, cfg_adv, best_params, subset_cap, seed=0
            )
            run.write_json("results_partial.json", out)
        if "p8a" in do:
            reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
            out["p8a_lengthscale_control"] = p8a_lengthscale_control(
                reg,
                cfg_adv,
                best_params,
                subset_cap,
                seed=0,
                gamma_mults=args.gamma_mults,
                attack_seeds=args.seeds,
            )
            run.write_json("results_partial.json", out)
        if "p9a" in do:
            reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
            out["p9a_quantum_bandwidth_regime"] = p9a_quantum_bandwidth_regime(
                reg,
                args,
                cfg_adv,
                cfg_q,
                best_params,
                subset_cap,
                seed=0,
                attack_seeds=args.seeds,
            )
            run.write_json("results_partial.json", out)
        if "p11c" in do:
            reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
            out["p11c_objective_and_regularization"] = p11c_objective_and_regularization(
                reg, args, cfg_adv, cfg_q, best_params, subset_cap, seed=0, c_grid=args.c_grid
            )
            run.write_json("results_partial.json", out)
        if "p14a" in do:
            reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
            out["p14a_objective_control"] = p14a_objective_control(
                reg, args, cfg_adv, best_params, subset_cap, seed=0, attack_seeds=args.seeds
            )
            run.write_json("results_partial.json", out)
        if "p15a" in do:
            reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
            out["p15a_transfer_alignment"] = p15a_transfer_alignment(
                reg, args, cfg_adv, best_params, subset_cap, seed=0
            )
            run.write_json("results_partial.json", out)
        if "p6a" in do:
            full = json.loads((REPO_ROOT / args.full_mlp_run / "results.json").read_text())
            full_mlp_params = full["mlp"]["best_params"]
            out["p6a_fullfeature_mlp"] = p6a_fullfeature_mlp(
                args, cfg_adv, full_mlp_params, seeds=args.seeds
            )
        run.write_json("results.json", out)
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
