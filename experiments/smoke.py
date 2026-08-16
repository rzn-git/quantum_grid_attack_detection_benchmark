"""MVP smoke pass (CLAUDE.md section 11): the ENTIRE pipeline end-to-end in
miniature — tiny data subset, 2 seeds, few Optuna trials, 4 qubits, one epsilon —
so integration breaks surface in minutes, not after a week of kernel compute.

This is NOT an experiment; it produces no reportable numbers. It asserts that
every stage runs and writes its artifact. Fail loud on any break.

Run:  python -m experiments.smoke
"""

from __future__ import annotations

import numpy as np

from qgridbench.attacks.evasion import fgsm, gaussian_sensor_noise, pgd
from qgridbench.attacks.gradients import GradProvider
from qgridbench.attacks.label_noise import flip_labels
from qgridbench.eval.metrics import compute_all
from qgridbench.eval.stats import holm_bonferroni, paired_wilcoxon
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.models.quantum.qkernel import (
    compute_states,
    fidelity_kernel,
    geometric_difference,
    kernel_target_alignment,
    shot_noise_kernel,
)
from qgridbench.models.quantum.vqc import VQClassifier
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)

N_QUBITS = 4
SEEDS = [0, 1]
CAP = 300
EPS = 0.1


def main():
    with RunDir("smoke", config={"qubits": N_QUBITS, "seeds": SEEDS, "cap": CAP}) as run:
        # ---- data + features (PCA regime, 4 dims, angle-scaled) --------------
        reg = prepare_regime("binary", "pca", n_components=N_QUBITS, seed=0)
        sub = stratified_cap(reg.X["train"], reg.y["train"], CAP, seed=0)
        n_te = 200
        Xtr, ytr = reg.X["train"][sub], reg.y["train"][sub]
        Xte, yte = reg.X["test"][:n_te], reg.y["test"][:n_te]
        Atr, Ate = reg.angles["train"][sub], reg.angles["test"][:n_te]
        assert Atr.min() >= -np.pi - 1e-6 and Atr.max() <= np.pi + 1e-6
        log.info("data ok | train %s test %s", Xtr.shape, Xte.shape)

        stage = {}

        # ---- classical (one model, 2 seeds) ----------------------------------
        cls_metrics = []
        for s in SEEDS:
            set_all_seeds(s)
            est = build(
                "xgb",
                {
                    "n_estimators": 60,
                    "learning_rate": 0.1,
                    "max_depth": 4,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "min_child_weight": 1,
                    "reg_lambda": 1.0,
                },
                s,
            )
            fit_model("xgb", est, Xtr, ytr)
            cls_metrics.append(compute_all(yte, est.predict_proba(Xte)))
        stage["classical_xgb_macro_f1"] = float(np.mean([m["macro_f1"] for m in cls_metrics]))
        mlp = build(
            "mlp",
            {"h1": 64, "h2": 32, "alpha": 1e-4, "learning_rate_init": 1e-3, "batch_size": 128},
            0,
        )
        fit_model("mlp", mlp, Xtr, ytr)

        # ---- quantum kernel + diagnostics + shot noise -----------------------
        states_tr = compute_states(Atr, "angle_ry")
        states_te = compute_states(Ate, "angle_ry")
        K_tr = fidelity_kernel(states_tr, states_tr)
        K_te = fidelity_kernel(states_te, states_tr)
        from sklearn.svm import SVC

        qk = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(K_tr, ytr)
        qk_f1 = float(compute_all(yte, _margin_proba(qk, K_te))["macro_f1"])
        stage["qkernel_macro_f1"] = qk_f1
        from sklearn.metrics.pairwise import rbf_kernel

        stage["kta_quantum"] = kernel_target_alignment(K_tr, ytr)
        stage["geometric_difference"] = geometric_difference(rbf_kernel(Xtr, gamma=0.1), K_tr)
        K_shot = shot_noise_kernel(K_tr, shots=1024, rng=spawn_rng(0, "shots"))
        stage["shot_noise_mae"] = float(np.abs(K_shot - K_tr).mean())

        # ---- VQC (tiny) + trainability logging -------------------------------
        set_all_seeds(0)
        vqc = VQClassifier(
            N_QUBITS, depth=2, n_classes=2, lr=0.05, batch_size=64, max_epochs=6, patience=4, seed=0
        )
        vqc.fit(Atr, ytr, reg.angles["val"][:200], reg.y["val"][:200])
        stage["vqc_macro_f1"] = float(compute_all(yte, vqc.predict_proba(Ate))["macro_f1"])
        stage["vqc_grad_var_final"] = vqc.history["grad_norm_var"][-1]
        # shot-based inference path
        _ = vqc.predict_proba(Ate[:10], shots=1024)

        # ---- adversarial: FGSM/PGD (MLP white-box), transfer, noise, poison --
        gp = GradProvider(mlp, "mlp")
        scale = reg.feature_scale  # attack budget unit (evasion.py SCALE CONTRACT)
        X_fgsm = fgsm(gp, Xte, yte, EPS, scale)
        X_pgd = pgd(gp, Xte, yte, EPS, scale, n_steps=5, rng=spawn_rng(0, "pgd"))
        assert np.max(np.abs(X_pgd - Xte) / scale) <= EPS + 1e-6
        stage["mlp_clean_f1"] = float(compute_all(yte, mlp.predict_proba(Xte))["macro_f1"])
        stage["mlp_pgd_f1"] = float(compute_all(yte, mlp.predict_proba(X_pgd))["macro_f1"])
        # transfer PGD(MLP) -> XGB
        stage["xgb_transfer_f1"] = float(compute_all(yte, est.predict_proba(X_fgsm))["macro_f1"])
        # sensor noise
        Xn = gaussian_sensor_noise(Xte, 20, spawn_rng(0, "noise"), scale)
        stage["xgb_noise20_f1"] = float(compute_all(yte, est.predict_proba(Xn))["macro_f1"])
        # poisoning
        yf = flip_labels(ytr, 0.2, spawn_rng(0, "poison"))
        set_all_seeds(0)
        est_p = build(
            "xgb",
            {
                "n_estimators": 60,
                "learning_rate": 0.1,
                "max_depth": 4,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 1,
                "reg_lambda": 1.0,
            },
            0,
        )
        fit_model("xgb", est_p, Xtr, yf)
        stage["xgb_poison20_f1"] = float(compute_all(yte, est_p.predict_proba(Xte))["macro_f1"])

        # ---- stats plumbing --------------------------------------------------
        a = np.array([m["macro_f1"] for m in cls_metrics] + [0.9])
        b = a - 0.02
        stage["wilcoxon_p"] = paired_wilcoxon(a, b)["p_value"]
        stage["holm"] = holm_bonferroni({"x": 0.01, "y": 0.2})

        run.write_json("smoke_results.json", stage)
        run.metrics = {k: v for k, v in stage.items() if isinstance(v, (int, float))}
        log.info("SMOKE PASS OK")
        for k, v in stage.items():
            if isinstance(v, float):
                log.info("  %-26s %.4f", k, v)
        print(f"-> {run.path}")


def _margin_proba(clf, K):
    d = clf.decision_function(K)
    p1 = 1.0 / (1.0 + np.exp(-d))
    return np.column_stack([1 - p1, p1])


if __name__ == "__main__":
    main()
