"""Phase 3: quantum models — QSVM (fidelity kernels) + VQC, with diagnostics.

QSVM: precompute train/test statevectors, build fidelity kernel, Optuna over
(bandwidth, C) on val, evaluate over seeds. Kernels cached to models/kernels.
Diagnostics (KTA, geometric difference vs tuned classical RBF) reported.
VQC: Optuna over (lr, depth, batch), early stopping, gradient-norm logging.

Run:  python -m experiments.run_quantum --variant binary --qubits 8 --model qkernel
      python -m experiments.run_quantum --variant binary --qubits 8 --model vqc
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from qgridbench.eval.metrics import compute_all
from qgridbench.models.quantum.qkernel import (
    cached_kernel,
    compute_states,
    fidelity_kernel,
    geometric_difference,
    kernel_cache_key,
    kernel_target_alignment,
    store_kernel,
)
from qgridbench.models.quantum.vqc import VQClassifier
from qgridbench.protocol import aggregate_seeds, kernel_subset, prepare_regime
from qgridbench.utils.run_tracking import (
    REPO_ROOT,
    RunDir,
    get_logger,
    load_yaml,
    mlflow_log,
)
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ----------------------------- QSVM ------------------------------------------


def _svc_on_kernel(n_classes, C):
    base = SVC(kernel="precomputed", C=C, class_weight="balanced")
    return OneVsRestClassifier(base) if n_classes > 2 else base


def run_qkernel(reg, encoding, n_qubits, seeds, cfg_q, storage, tag, run: RunDir):
    """Tune bandwidth+C, evaluate on the full test split, cache final kernels.

    Tuning evaluates on a stratified val SUBSAMPLE (config tuning_val_subset) —
    full-val statevector computation per trial is the wall-clock bottleneck at
    16 qubits (decision log 2026-08-13). The final config is evaluated once on
    the FULL test split, chunked to bound memory.
    """
    import time as _time

    from qgridbench.models.classical.zoo import stratified_cap

    cap = cfg_q["qkernel"]["subset_cap"]
    sub = kernel_subset(reg, cap, seed=0)
    A_tr = reg.angles["train"][sub]
    y_tr = reg.y["train"][sub]
    A_va, y_va = reg.angles["val"], reg.y["val"]
    A_te, y_te = reg.angles["test"], reg.y["test"]
    n_classes = len(np.unique(reg.y["train"]))

    va_sub = stratified_cap(A_va, y_va, cfg_q["qkernel"]["tuning_val_subset"], seed=0)
    A_va_s, y_va_s = A_va[va_sub], y_va[va_sub]

    bw_cfg, c_cfg = cfg_q["qkernel"]["bandwidth"], cfg_q["qkernel"]["svm_C"]

    def objective(trial):
        bw = trial.suggest_float("bandwidth", bw_cfg["low"], bw_cfg["high"], log=bw_cfg["log"])
        C = trial.suggest_float("C", c_cfg["low"], c_cfg["high"], log=c_cfg["log"])
        s_tr = compute_states(A_tr, encoding, bandwidth=bw)
        K_tr = fidelity_kernel(s_tr, s_tr)
        K_va = fidelity_kernel(compute_states(A_va_s, encoding, bandwidth=bw), s_tr)
        clf = _svc_on_kernel(n_classes, C).fit(K_tr, y_tr)
        return f1_score(y_va_s, clf.predict(K_va), average="macro")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"qk_{encoding}_{tag}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    n_done = len([t for t in study.trials if t.state.is_finished()])
    n_trials = cfg_q["vqc"]["optuna"]["n_trials"] - n_done
    if n_trials > 0:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    bw, C = best["bandwidth"], best["C"]

    # final kernels (cached): train Gram + chunked full-test block
    t0 = _time.perf_counter()
    states_tr = compute_states(A_tr, encoding, bandwidth=bw)
    K_tr = fidelity_kernel(states_tr, states_tr)
    train_kernel_s = _time.perf_counter() - t0
    key = kernel_cache_key(enc=encoding, q=n_qubits, bw=round(bw, 6), tag=tag, block="final")
    if cached_kernel(key) is None:
        store_kernel(
            key,
            K_tr,
            {
                "encoding": encoding,
                "qubits": n_qubits,
                "bandwidth": bw,
                "block": "final_train",
                "tag": tag,
            },
        )
    t0 = _time.perf_counter()
    chunk = 2000
    K_te = np.empty((len(A_te), len(A_tr)))
    for s in range(0, len(A_te), chunk):
        se = compute_states(A_te[s : s + chunk], encoding, bandwidth=bw)
        K_te[s : s + chunk] = fidelity_kernel(se, states_tr)
    test_kernel_s = _time.perf_counter() - t0
    latency_per_sample = test_kernel_s / len(A_te)  # simulator latency, labeled as such

    # diagnostics on the train subset (vs tuned classical RBF)
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.svm import SVC as _SVC

    gamma_grid = np.logspace(-3, 1, 12)
    best_g, best_s = gamma_grid[0], -1
    Xtr_pca = reg.X["train"][sub]
    for g in gamma_grid:
        s = f1_score(
            y_va,
            _SVC(C=C, gamma=g, class_weight="balanced").fit(Xtr_pca, y_tr).predict(reg.X["val"]),
            average="macro",
        )
        if s > best_s:
            best_s, best_g = s, g
    K_rbf = rbf_kernel(Xtr_pca, gamma=best_g)
    kta_q = kernel_target_alignment(K_tr, y_tr)
    kta_c = kernel_target_alignment(K_rbf, y_tr)
    geo = geometric_difference(K_rbf, K_tr)

    # analytic-kernel QSVM is deterministic: one fit/eval; seed variation appears
    # only in the shot-noise ablation (run_ablations). Reported std = 0 honestly.
    import time as _t

    t0 = _t.perf_counter()
    clf = _svc_on_kernel(n_classes, C).fit(K_tr, y_tr)
    svc_fit_s = _t.perf_counter() - t0
    proba_te = _svc_proba(clf, K_te, y_tr, n_classes)
    metrics = compute_all(y_te, proba_te)
    per_seed = [metrics for _ in seeds]
    agg = aggregate_seeds(per_seed)
    agg["train_time_s"] = {"mean": train_kernel_s + svc_fit_s, "std": 0.0}
    agg["infer_latency_s"] = {"mean": latency_per_sample, "std": 0.0}

    result = {
        "encoding": encoding,
        "qubits": n_qubits,
        "best_params": best,
        "rbf_gamma": float(best_g),
        "diagnostics": {
            "kta_quantum": kta_q,
            "kta_classical_rbf": kta_c,
            "geometric_difference": geo,
        },
        "aggregate": agg,
        # Persisted for table assembly, but note these rows are IDENTICAL: with the
        # training subset fixed, the fidelity kernel and the SVM solve are exactly
        # deterministic, so seed variance is genuinely zero rather than unmeasured.
        # A paired seed-wise test against a constant series is degenerate — this
        # model is compared by margin against the classical seed distribution.
        "per_seed": per_seed,
        "deterministic": True,
        "subset_size": int(len(sub)),
        "timing": {
            "train_kernel_s": train_kernel_s,
            "svc_fit_s": svc_fit_s,
            "test_kernel_s": test_kernel_s,
        },
    }
    run.write_json(f"qkernel_{encoding}_q{n_qubits}.json", result)
    log.info(
        "QK %s q%d | test macro-F1 %.4f | KTA_q=%.3f KTA_c=%.3f g=%.2f",
        encoding,
        n_qubits,
        agg["macro_f1"]["mean"],
        kta_q,
        kta_c,
        geo,
    )
    mlflow_log(
        f"qkernel_{encoding}_{tag}",
        {"encoding": encoding, "qubits": n_qubits, **best},
        {f"test_{k}": v["mean"] for k, v in agg.items()}
        | {"kta_quantum": kta_q, "geometric_difference": geo},
        tags={"phase": "quantum", "family": "qkernel"},
    )
    return result


def _svc_proba(clf, K, y_tr, n_classes):
    """Probability-like scores from a precomputed-kernel SVM (decision margins)."""
    if hasattr(clf, "decision_function"):
        d = clf.decision_function(K)
    else:
        d = clf.predict(K)
    if n_classes == 2:
        p1 = 1.0 / (1.0 + np.exp(-d))
        return np.column_stack([1 - p1, p1])
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ------------------------------ VQC ------------------------------------------


def run_vqc(reg, n_qubits, seeds, cfg_q, storage, tag, run: RunDir):
    from qgridbench.models.classical.zoo import stratified_cap

    A_tr, y_tr = reg.angles["train"], reg.y["train"]
    A_va, y_va = reg.angles["val"], reg.y["val"]
    A_te, y_te = reg.angles["test"], reg.y["test"]
    n_classes = len(np.unique(y_tr))
    cap = cfg_q["qkernel"]["subset_cap"]
    sub = kernel_subset(reg, cap, seed=0)  # cap VQC train size to match kernel fairness
    A_trc, y_trc = A_tr[sub], y_tr[sub]
    # early-stopping/objective val subsample: bounds GPU memory (full-val states
    # exceed VRAM at 16 qubits) and matches the qkernel tuning protocol
    va_sub = stratified_cap(A_va, y_va, cfg_q["qkernel"]["tuning_val_subset"], seed=0)
    A_vas, y_vas = A_va[va_sub], y_va[va_sub]

    o = cfg_q["vqc"]["optuna"]

    def objective(trial):
        lr = trial.suggest_float("lr", o["lr"]["low"], o["lr"]["high"], log=o["lr"]["log"])
        depth = trial.suggest_categorical("depth", cfg_q["vqc"]["depths"])
        bs = trial.suggest_categorical("batch_size", o["batch_size"])
        set_all_seeds(0)
        m = VQClassifier(
            n_qubits,
            depth=depth,
            n_classes=n_classes,
            lr=lr,
            batch_size=bs,
            max_epochs=cfg_q["vqc"]["tuning_max_epochs"],
            patience=cfg_q["vqc"]["patience"],
            seed=0,
        )
        m.fit(A_trc, y_trc, A_vas, y_vas)
        return m.best_val_f1_

    study = optuna.create_study(
        direction="maximize",
        study_name=f"vqc_{tag}_q{n_qubits}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    n_done = len([t for t in study.trials if t.state.is_finished()])
    if o["n_trials"] - n_done > 0:
        study.optimize(objective, n_trials=o["n_trials"] - n_done, show_progress_bar=False)
    best = study.best_params

    import time as _t

    per_seed, grad_var_last, fit_times, latencies = [], [], [], []
    for s in seeds:
        set_all_seeds(s)
        m = VQClassifier(
            n_qubits,
            depth=best["depth"],
            n_classes=n_classes,
            lr=best["lr"],
            batch_size=best["batch_size"],
            max_epochs=cfg_q["vqc"]["max_epochs"],
            patience=cfg_q["vqc"]["patience"],
            seed=s,
        )
        m.fit(A_trc, y_trc, A_vas, y_vas)
        t0 = _t.perf_counter()
        proba = m.predict_proba(A_te)  # reported inference: CPU float64 path
        latencies.append((_t.perf_counter() - t0) / len(A_te))
        fit_times.append(m.fit_seconds)
        per_seed.append(compute_all(y_te, proba))
        grad_var_last.append(m.history["grad_norm_var"][-1] if m.history["grad_norm_var"] else 0.0)
    agg = aggregate_seeds(per_seed)
    agg["train_time_s"] = {
        "mean": float(np.mean(fit_times)),
        "std": float(np.std(fit_times, ddof=1)),
    }
    agg["infer_latency_s"] = {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies, ddof=1)),
    }
    result = {
        "qubits": n_qubits,
        "best_params": best,
        "aggregate": agg,
        # persisted so headline comparisons are paired-testable from artifacts
        # (CLAUDE.md section 5: paired Wilcoxon across seeds); aggregate alone
        # only supports a margin, not a significance claim
        "per_seed": per_seed,
        "grad_norm_var_final_mean": float(np.mean(grad_var_last)),
        "subset_size": int(len(sub)),
        "engine": {
            "device": m.device_name,
            "diff_method": m.diff_method,
            "grad_method": m.grad_method,
            "chunk": m.chunk,
        },
    }
    run.write_json(f"vqc_q{n_qubits}.json", result)
    log.info(
        "VQC q%d | depth=%d | test macro-F1 %.4f | grad_var_final=%.2e",
        n_qubits,
        best["depth"],
        agg["macro_f1"]["mean"],
        np.mean(grad_var_last),
    )
    mlflow_log(
        f"vqc_{tag}_q{n_qubits}",
        {"qubits": n_qubits, **best},
        {f"test_{k}": v["mean"] for k, v in agg.items()},
        tags={"phase": "quantum", "family": "vqc"},
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--model", choices=["qkernel", "vqc"], default="qkernel")
    ap.add_argument("--encodings", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    cfg_c = load_yaml(REPO_ROOT / "configs" / "classical.yaml")
    seeds = args.seeds or cfg_c["evaluation"]["seeds"]
    storage_dir = REPO_ROOT / cfg_c["tuning"]["optuna_storage_dir"]
    encodings = args.encodings or cfg_q["qkernel"]["encodings"]

    reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
    tag = f"{args.variant}_q{args.qubits}"

    def storage_for(study: str) -> str:
        return f"sqlite:///{(storage_dir / f'optuna_{study}.db').as_posix()}"

    with RunDir(f"quantum_{args.model}_{tag}", config=vars(args), seeds=seeds) as run:
        out = {}
        if args.model == "qkernel":
            for enc in encodings:
                out[enc] = run_qkernel(
                    reg, enc, args.qubits, seeds, cfg_q, storage_for(f"qk_{enc}_{tag}"), tag, run
                )
        else:
            out["vqc"] = run_vqc(
                reg, args.qubits, seeds, cfg_q, storage_for(f"vqc_{tag}"), tag, run
            )
        run.write_json("results.json", out)
        run.metrics = {k: v["aggregate"]["macro_f1"]["mean"] for k, v in out.items()}
        print(json.dumps(run.metrics, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
