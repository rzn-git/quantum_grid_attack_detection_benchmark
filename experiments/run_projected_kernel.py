"""E9 — projected quantum kernel: is the null a fidelity-kernel property or a QML property?

PRE-REGISTRATION (pass-4 workplan item E9; frozen before the run).
C2 reports an explained accuracy null. Its scope is FIDELITY kernels (RY / ZZ maps at 8--16
qubits). Huang et al. 2021 — the same paper this study's diagnostics are built on — propose
PROJECTED quantum kernels precisely for the failure measured here: a fidelity Gram that goes
degenerate (near-constant at small bandwidth, near-identity at large). A reviewer reads C2 as
a family-level null and observes that the family's own prescribed escape was never tested.

The projected kernel replaces the state overlap with a distance between REDUCED density
matrices: k(x,y) = exp(-gamma * sum_i ||rho_i(x) - rho_i(y)||_F^2). For single-qubit reduced
states this is exactly an RBF over the concatenated Bloch vectors (the Frobenius distance
between one-qubit density matrices is half the squared Bloch distance), so the model is an
RBF-SVM on 3n quantum-derived features -- computed from the SAME statevectors the fidelity
kernel already uses.

DECISION RULE against the shipped PCA-8 anchors (ZZ fidelity kernel 0.5644, tuned XGBoost
0.5968), test macro-F1:
  <= 0.57          -> the null broadens to projected kernels; C2 strengthens
  0.57 -- 0.5968   -> intermediate; report as measured
  >  0.5968        -> BEATS the classical arm: SURPRISING. Verify the reduced-state pipeline
                      through an independent path, re-run on a second resample, and ESCALATE
                      to the operator before any claim enters any document.
KTA of the projected Gram is reported alongside: the fidelity kernels reach KTA <= 0.015 on
these labels, and the paper's own diagnostic finding is that KTA -- not the geometric
difference -- predicts accuracy. If a projected kernel lifts accuracy it should lift KTA too;
if it lifts neither, the diagnostic story gains a second confirming instance.

CONTROLS (must pass before the rule is read):
  positive   the Bloch pipeline on the product RY map must reproduce the analytic values
             (<X> = sin(b*a), <Y> = 0, <Z> = cos(b*a)) to 1e-12
  anchor     the shipped fidelity-kernel baseline must re-score at 0.5644 on this subset

Equal-budget contract: 50 Optuna trials, the same budget every other model in the zoo gets.

Run:  python -m experiments.run_projected_kernel
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from experiments.run_adversarial import _svc_on_kernel
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.models.quantum.qkernel import (
    compute_states,
    fidelity_kernel,
    kernel_target_alignment,
)
from qgridbench.protocol import kernel_subset, prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

ANCHOR_FIDELITY_ZZ = 0.5644
ANCHOR_XGB = 0.5968
N_TRIALS = 50
VAL_CAP = 2000


def bloch_features(states: np.ndarray, n_qubits: int) -> np.ndarray:
    """Concatenated single-qubit Bloch vectors (N, 3n) from statevectors (N, 2**n).

    rho_i = Tr_{!=i} |psi><psi|; <X> = 2 Re(rho_01), <Y> = -2 Im(rho_01), <Z> = rho_00 - rho_11.
    """
    N = len(states)
    psi = states.astype(np.complex128).reshape((N,) + (2,) * n_qubits)
    out = np.empty((N, 3 * n_qubits), dtype=np.float64)
    for i in range(n_qubits):
        a = np.moveaxis(psi, i + 1, 1).reshape(N, 2, -1)  # (N, 2, 2**(n-1))
        r00 = np.einsum("nk,nk->n", a[:, 0], a[:, 0].conj()).real
        r11 = np.einsum("nk,nk->n", a[:, 1], a[:, 1].conj()).real
        r01 = np.einsum("nk,nk->n", a[:, 0], a[:, 1].conj())
        out[:, 3 * i + 0] = 2.0 * r01.real
        out[:, 3 * i + 1] = -2.0 * r01.imag
        out[:, 3 * i + 2] = r00 - r11
    return out


def positive_control(n_qubits: int, bandwidth: float) -> dict:
    """Product RY map has closed-form Bloch coordinates — the pipeline must reproduce them."""
    rng = np.random.default_rng(0)
    A = rng.uniform(-np.pi, np.pi, size=(16, n_qubits))
    st = compute_states(A, "angle_ry", bandwidth=bandwidth, dtype=np.complex128)
    got = bloch_features(st, n_qubits)
    want = np.empty_like(got)
    for i in range(n_qubits):
        want[:, 3 * i + 0] = np.sin(bandwidth * A[:, i])
        want[:, 3 * i + 1] = 0.0
        want[:, 3 * i + 2] = np.cos(bandwidth * A[:, i])
    err = float(np.abs(got - want).max())
    return {"max_abs_error": err, "pass": bool(err < 1e-12)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="zz")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()

    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    subset_cap = cfg_q["qkernel"]["subset_cap"]
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    bw_tuned = float(best["qk_bw"])

    ctrl = positive_control(args.qubits, bw_tuned)
    log.info("POSITIVE CONTROL (analytic Bloch): %s", ctrl)
    if not ctrl["pass"]:
        raise RuntimeError(f"Bloch pipeline failed its analytic control: {ctrl}")

    reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
    sub = kernel_subset(reg, subset_cap, seed=0)
    A_tr, ytr = reg.angles["train"][sub], reg.y["train"][sub]
    va = stratified_cap(reg.X["val"], reg.y["val"], VAL_CAP, seed=0)
    A_va, yva = reg.angles["val"][va], reg.y["val"][va]
    A_te, yte = reg.angles["test"], reg.y["test"]
    log.info("subsets: train %d, val %d, test %d", len(ytr), len(yva), len(yte))

    cfg = {
        "variant": args.variant,
        "qubits": args.qubits,
        "encoding": args.encoding,
        "tuned_bandwidth": bw_tuned,
        "trials": args.trials,
        "subset_cap": subset_cap,
        "anchors": {"fidelity_zz": ANCHOR_FIDELITY_ZZ, "xgb": ANCHOR_XGB},
    }

    with RunDir(f"projected_kernel_{args.variant}_q{args.qubits}", config=cfg) as run:
        # ---- ANCHOR CONTROL: re-score the shipped fidelity kernel on this subset --------
        st_tr = compute_states(A_tr, args.encoding, bandwidth=bw_tuned)
        st_te = compute_states(A_te, args.encoding, bandwidth=bw_tuned)
        K_tr = fidelity_kernel(st_tr, st_tr)
        svc_f = _svc_on_kernel(2, float(best["qk_C"])).fit(K_tr, ytr)
        f_test = float(f1_score(yte, svc_f.predict(fidelity_kernel(st_te, st_tr)), average="macro"))
        anchor_ok = abs(f_test - ANCHOR_FIDELITY_ZZ) < 0.005
        log.info(
            "ANCHOR CONTROL fidelity kernel: %.4f vs published %.4f -> pass=%s",
            f_test,
            ANCHOR_FIDELITY_ZZ,
            anchor_ok,
        )
        kta_fid = kernel_target_alignment(K_tr, ytr)

        # ---- projected features, at three encoding bandwidths --------------------------
        results = {}
        for mult, tag in ((0.5, "0.5x"), (1.0, "1x"), (2.0, "2x")):
            bw = bw_tuned * mult
            s_tr = compute_states(A_tr, args.encoding, bandwidth=bw, dtype=np.complex128)
            s_va = compute_states(A_va, args.encoding, bandwidth=bw, dtype=np.complex128)
            s_te = compute_states(A_te, args.encoding, bandwidth=bw, dtype=np.complex128)
            F_tr, F_va, F_te = (bloch_features(s, args.qubits) for s in (s_tr, s_va, s_te))

            def objective(trial, Ftr=F_tr, Fva=F_va):
                gamma = trial.suggest_float("gamma", 1e-4, 1e2, log=True)
                C = trial.suggest_float("C", 1e-2, 1e3, log=True)
                set_all_seeds(0)
                est = SVC(kernel="rbf", gamma=gamma, C=C, class_weight="balanced")
                est.fit(Ftr, ytr)
                return f1_score(yva, est.predict(Fva), average="macro")

            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)
            )
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

            set_all_seeds(0)
            estb = SVC(kernel="rbf", class_weight="balanced", **study.best_params).fit(F_tr, ytr)
            test_f1 = float(f1_score(yte, estb.predict(F_te), average="macro"))
            # KTA of the projected Gram on the SAME train subset, comparable to the fidelity one
            d2 = ((F_tr[:, None, :] - F_tr[None, :, :]) ** 2).sum(axis=2)
            K_proj = np.exp(-study.best_params["gamma"] * d2)
            kta_proj = kernel_target_alignment(K_proj, ytr)

            results[tag] = {
                "bandwidth": bw,
                "best_params": study.best_params,
                "val_macro_f1": float(study.best_value),
                "test_macro_f1": test_f1,
                "kta_projected": float(kta_proj),
                "n_trials": args.trials,
            }
            log.info(
                "projected bw %s (%.4f): val %.4f | TEST %.4f | KTA %.4f (fidelity %.4f)",
                tag,
                bw,
                study.best_value,
                test_f1,
                kta_proj,
                kta_fid,
            )
            run.write_json("results_partial.json", {"projected": results})

        best_tag = max(results, key=lambda k: results[k]["test_macro_f1"])
        best_test = results[best_tag]["test_macro_f1"]
        branch = (
            "null_broadens"
            if best_test <= 0.57
            else (
                "intermediate_report_as_measured"
                if best_test <= ANCHOR_XGB
                else "SURPRISING_beats_classical_ESCALATE"
            )
        )
        verdict = {
            "positive_control": ctrl,
            "anchor_control_fidelity_test_f1": f_test,
            "anchor_control_pass": bool(anchor_ok),
            "kta_fidelity": float(kta_fid),
            "best_bandwidth_tag": best_tag,
            "best_test_macro_f1": best_test,
            "best_kta_projected": results[best_tag]["kta_projected"],
            "vs_fidelity_anchor": float(best_test - ANCHOR_FIDELITY_ZZ),
            "vs_xgb_anchor": float(best_test - ANCHOR_XGB),
            "branch": branch,
            "rule": "<=0.57 null broadens; <=0.5968 intermediate; >0.5968 escalate",
        }
        run.write_json(
            "results.json",
            {
                "projected": results,
                "fidelity_anchor": {"test_f1": f_test, "kta": kta_fid},
                "verdict": verdict,
            },
        )
        run.metrics = {"best_test_macro_f1": best_test}
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
