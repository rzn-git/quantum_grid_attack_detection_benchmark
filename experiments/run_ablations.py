"""Phase 3 ablations + data-efficiency curves (RQ3).

  - shot-noise study: analytic vs 1024 vs 4096 shots on the best quantum kernel
  - feature-map / qubit sweep (macro-F1 grid)
  - VQC depth sweep with gradient-variance (barren-plateau signature)
  - data-efficiency curves: best classical + both quantum on {250,500,1000,2000}
  - optional single depolarizing-noise point (default.mixed)
  - kernel geometry: concentration test + near-identity degeneracy of cached Grams
  - relabel control: Huang et al. quantum-easy labels — pipeline positive control
  - bandwidth curve: dequantization <-> degeneracy axis, in and beyond the tuning box
  - bandwidth fragility: flip rate vs bandwidth — does accuracy-tuning cause fragility?
  - dequantization check: prediction agreement, tuned quantum SVM vs classical RBF SVM

Run:  python -m experiments.run_ablations --variant binary
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.svm import SVC

from qgridbench.eval.metrics import compute_all
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.models.quantum.qkernel import (
    compute_density_matrices,
    compute_states,
    cosine_fidelity_kernel,
    fidelity_kernel,
    find_cached_kernel,
    geometric_difference,
    kernel_target_alignment,
    noisy_fidelity_kernel,
    quantum_easy_labels,
    shot_noise_kernel,
)
from qgridbench.models.quantum.vqc import VQClassifier
from qgridbench.protocol import kernel_subset, prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)


def _qk_f1(K_tr, K_te, ytr, yte, C=1.0):
    clf = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K_tr, ytr)
    d = clf.decision_function(K_te)
    p1 = 1.0 / (1.0 + np.exp(-d))
    return compute_all(yte, np.column_stack([1 - p1, p1]))["macro_f1"]


def shot_study(reg, encoding, n_qubits, cap, shots_list, seeds):
    sub = kernel_subset(reg, cap, seed=0)
    Atr, ytr = reg.angles["train"][sub], reg.y["train"][sub]
    Ate, yte = reg.angles["test"], reg.y["test"]
    K_tr = fidelity_kernel(compute_states(Atr, encoding), compute_states(Atr, encoding))
    states_te = compute_states(Ate, encoding)
    K_te = fidelity_kernel(states_te, compute_states(Atr, encoding))
    out = {}
    for shots in shots_list:
        if shots == "analytic":
            out["analytic"] = {"mean": _qk_f1(K_tr, K_te, ytr, yte), "std": 0.0}
        else:
            vals = []
            for s in seeds:
                rng = spawn_rng(s, f"shots_{shots}")
                Ktr_n = shot_noise_kernel(K_tr, shots, rng)
                Kte_n = shot_noise_kernel(K_te, shots, rng, symmetric=False)
                vals.append(_qk_f1(Ktr_n, Kte_n, ytr, yte))
            out[str(shots)] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
    return out


def noise_model_study(reg, encoding, n_qubits, cap, p_levels, test_cap=2000):
    """Depolarizing-noise point: the simulator -> hardware bridge (study protocol §6.5).

    Shot noise (above) models finite sampling of an ideal device; this models the
    device being *wrong*, which is the qualitatively different failure. Runs on a
    reduced subset because a density matrix is 4**d complex entries per sample
    against 2**d for a statevector -- 256x the memory at 8 qubits. The p=0.0 row is
    computed through the SAME mixed-state path, so the comparison isolates noise
    rather than confounding it with a change of method.
    """
    sub = kernel_subset(reg, cap, seed=0)
    Atr, ytr = reg.angles["train"][sub], reg.y["train"][sub]
    te = stratified_cap(reg.X["test"], reg.y["test"], test_cap, seed=0)
    Ate, yte = reg.angles["test"][te], reg.y["test"][te]

    out = {}
    for p in p_levels:
        rho_tr = compute_density_matrices(Atr, encoding, p)
        rho_te = compute_density_matrices(Ate, encoding, p)
        K_tr = noisy_fidelity_kernel(rho_tr, rho_tr)
        K_te = noisy_fidelity_kernel(rho_te, rho_tr)
        out[str(p)] = {
            "macro_f1": _qk_f1(K_tr, K_te, ytr, yte),
            "kta": kernel_target_alignment(K_tr, ytr),
            "mean_purity": float(np.diag(K_tr).mean()),
            "n_train": int(len(Atr)),
            "n_test": int(len(Ate)),
        }
        log.info(
            "noise p=%.3f | macro-F1 %.4f | KTA %.4f | purity %.4f",
            p,
            out[str(p)]["macro_f1"],
            out[str(p)]["kta"],
            out[str(p)]["mean_purity"],
        )
    return out


def feature_map_qubit_sweep(variant, encodings, qubit_list, cap, seed=0):
    grid = {}
    for q in qubit_list:
        reg = prepare_regime(variant, "pca", n_components=q, seed=0)
        sub = kernel_subset(reg, cap, seed=0)
        Atr, ytr = reg.angles["train"][sub], reg.y["train"][sub]
        Ate, yte = reg.angles["test"], reg.y["test"]
        for enc in encodings:
            K_tr = fidelity_kernel(compute_states(Atr, enc), compute_states(Atr, enc))
            K_te = fidelity_kernel(compute_states(Ate, enc), compute_states(Atr, enc))
            grid[f"{enc}_q{q}"] = float(_qk_f1(K_tr, K_te, ytr, yte))
            log.info("sweep %s q%d -> %.4f", enc, q, grid[f"{enc}_q{q}"])
    return grid


def vqc_depth_sweep(reg, n_qubits, depths, cap, seeds):
    sub = kernel_subset(reg, cap, seed=0)
    Atr, ytr = reg.angles["train"][sub], reg.y["train"][sub]
    Ava, yva = reg.angles["val"][:500], reg.y["val"][:500]
    Ate, yte = reg.angles["test"], reg.y["test"]
    out = {}
    for depth in depths:
        f1s, gvars = [], []
        for s in seeds:
            set_all_seeds(s)
            m = VQClassifier(
                n_qubits,
                depth=depth,
                n_classes=len(np.unique(ytr)),
                lr=0.05,
                batch_size=64,
                max_epochs=40,
                patience=8,
                seed=s,
            )
            m.fit(Atr, ytr, Ava, yva)
            f1s.append(compute_all(yte, m.predict_proba(Ate))["macro_f1"])
            gvars.append(np.mean(m.history["grad_norm_var"]))
        out[f"depth_{depth}"] = {
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s, ddof=1)),
            "grad_norm_var_mean": float(np.mean(gvars)),
        }
        log.info("VQC depth %d -> F1 %.4f | grad_var %.2e", depth, np.mean(f1s), np.mean(gvars))
    return out


def barren_plateau_scan(variant, qubit_list, depth, n_inits=40, n_samples=64, cap=2000):
    """Textbook barren-plateau measurement: Var[dL/dtheta] at RANDOM initialisation.

    The section-6.4 depth sweep found gradient variance FLAT across depth 2/4/6 at 8
    qubits, and the tuned q8-vs-q12 finals then showed a 2.27x drop — but those two
    runs used different tuned learning rates and are measured after training, so the
    comparison is confounded. This is the controlled version, and the one the barren-
    plateau literature actually specifies (McClean et al. 2018): sample many random
    parameter sets, take the gradient of the loss w.r.t. a single fixed parameter,
    and report its variance across the draws. No training, so nothing about the
    optimiser or learning rate can leak in, and it is cheap.

    Reports variance for one fixed coordinate (the standard estimator) and the mean
    squared gradient norm over all coordinates (a scale-free companion).
    """
    import torch

    out = {
        "depth": int(depth),
        "n_inits": int(n_inits),
        "n_samples": int(n_samples),
        "by_qubits": {},
    }
    for q in qubit_list:
        reg = prepare_regime(variant, "pca", n_components=q, seed=0)
        sub = kernel_subset(reg, cap, seed=0)
        A = reg.angles["train"][sub][:n_samples]
        y = reg.y["train"][sub][:n_samples]
        n_classes = len(np.unique(reg.y["train"]))

        model = VQClassifier(q, depth=depth, n_classes=n_classes, seed=0)
        Xt = torch.as_tensor(A, dtype=torch.float64)
        yt = torch.as_tensor(y, dtype=torch.long)
        shape = model.weights.shape

        g0, norms = [], []
        for i in range(n_inits):
            gen = torch.Generator().manual_seed(1000 + i)
            # uniform over the full rotation range — the standard BP initialisation,
            # NOT the small-sigma init used for training
            w = 2 * np.pi * torch.rand(*shape, generator=gen, dtype=torch.float64) - np.pi
            w.requires_grad_(True)
            loss = model._ce_loss(model._logits(Xt, weights=w), yt)
            (grad,) = torch.autograd.grad(loss, w)
            g0.append(float(grad.reshape(-1)[0]))  # one fixed coordinate
            norms.append(float((grad**2).mean()))  # mean squared over all
        out["by_qubits"][str(q)] = {
            "var_single_param": float(np.var(g0, ddof=1)),
            "mean_sq_grad": float(np.mean(norms)),
            "n_params": int(np.prod(shape)),
        }
        log.info(
            "barren-plateau q%d depth%d: Var[g_0]=%.3e  mean|g|^2=%.3e  (%d params)",
            q,
            depth,
            out["by_qubits"][str(q)]["var_single_param"],
            out["by_qubits"][str(q)]["mean_sq_grad"],
            out["by_qubits"][str(q)]["n_params"],
        )
    return out


def kernel_cap_cost(variant, dims, cap, seeds, models=("xgb", "lgbm")):
    """What the O(N^2) kernel cap costs the comparison — the dominant handicap.

    The 2,000-sample cap exists because a fidelity kernel needs O(N^2) circuit
    evaluations; classical models are capped identically so the matched-dimensionality
    comparison is fair. It IS fair — but it is also expensive, and reporting ~0.60
    accuracies without saying why understates what the feature space supports by
    roughly 0.19 macro-F1. Measured here so the paper can state it with a number
    instead of a hedge: at the cap, classical models lose ~5x more accuracy than the
    entire quantum-classical gap they are being compared across.

    Also emits a train-size curve, which shows whether the cap sits on a plateau
    (harmless) or on the steep part of the curve (costly). It is the latter.
    """
    out = {"cap": int(cap), "per_regime": {}, "train_size_curve": {}}
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    for d in dims:
        reg = prepare_regime(variant, "pca", n_components=d, seed=0)
        sub = kernel_subset(reg, cap, seed=0)
        Xte, yte = reg.X["test"], reg.y["test"]
        for model in models:
            params = best.get(model, {})
            capped = _fit_score(model, reg.X["train"][sub], reg.y["train"][sub], Xte, yte, params)
            full = _fit_score(model, reg.X["train"], reg.y["train"], Xte, yte, params)
            out["per_regime"][f"pca{d}_{model}"] = {
                "capped": capped,
                "full_train": full,
                "cap_cost": full - capped,
                "n_capped": int(len(sub)),
                "n_full": int(len(reg.y["train"])),
            }
            log.info(
                "cap cost pca%d %s: capped %.4f -> full %.4f (%+.4f)",
                d,
                model,
                capped,
                full,
                full - capped,
            )
    # where does the cap sit on the learning curve?
    reg = prepare_regime(variant, "pca", n_components=dims[0], seed=0)
    for n in (500, 1000, 2000, 5000, 10000, 20000, len(reg.y["train"])):
        idx = stratified_cap(reg.X["train"], reg.y["train"], n, seed=0)
        out["train_size_curve"][str(len(idx))] = _fit_score(
            "xgb",
            reg.X["train"][idx],
            reg.y["train"][idx],
            reg.X["test"],
            reg.y["test"],
            best.get("xgb", {}),
        )
    return out


def _fit_score(model, Xtr, ytr, Xte, yte, params, seed=0):
    set_all_seeds(seed)
    est = build(model, params, seed)
    fit_model(model, est, Xtr, ytr)
    return float(compute_all(yte, est.predict_proba(Xte))["macro_f1"])


def _offdiag_stats(K, shots_list):
    """Mean/std/percentile stats of the off-diagonal Gram entries + the ratio of
    shot-noise std to the informative spread (>=1 would mean sampling swamps signal)."""
    v = K[np.triu_indices_from(K, k=1)]
    mean, std = float(v.mean()), float(v.std())
    row = {
        "mean": mean,
        "std": std,
        "var": float(v.var()),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "frac_below_0.01": float(np.mean(v < 0.01)),
        "frac_below_0.001": float(np.mean(v < 0.001)),
    }
    for s in shots_list:
        shot_std = float(np.sqrt(mean * (1 - mean) / s))
        row[f"shot_std_ratio_{s}"] = shot_std / std if std > 0 else float("inf")
    return row


def kernel_geometry(variant, cfg_geo):
    """Is exponential concentration (Thanasilp et al.) behind the low KTA and the
    fragmented QSVM surface? Two measurements, one verdict.

    (1) Every cached tuned-bandwidth Gram — the matrices every reported number
    actually used — gets off-diagonal stats. This is where the near-identity
    degeneracy shows: cells whose median off-diagonal fidelity is ~1e-4 are
    Grams that are far from ANY classical kernel (large g) because almost all
    state pairs are orthogonal, not because the geometry is usefully exotic.
    (2) A controlled fixed-bandwidth scan over qubit counts on this variant's
    regimes — tuned bandwidths differ per cell, so only a fixed-bandwidth scan
    can measure the concentration rate. 2^-n concentration = log2-var slope -1.0.
    """
    shots_list = cfg_geo["shots"]
    cached = []
    for meta_path, meta in sorted(
        _iter_cached_grams(), key=lambda x: (x[1]["tag"], x[1]["encoding"])
    ):
        K = np.load(meta_path.with_suffix(".npz"))["K"]
        cached.append(
            {
                "cell": f"{meta['encoding']}_q{meta['qubits']}_{meta['tag']}",
                "bandwidth": meta["bandwidth"],
                **_offdiag_stats(K, shots_list),
            }
        )
        log.info(
            "cached %s: median %.2e  frac<0.01 %.3f",
            cached[-1]["cell"],
            cached[-1]["median"],
            cached[-1]["frac_below_0.01"],
        )

    scan, bw = [], cfg_geo["scan_bandwidth"]
    for q in cfg_geo["scan_qubits"]:
        reg = prepare_regime(variant, "pca", n_components=q, seed=0)
        sub = kernel_subset(reg, 2000, seed=0)
        A, y = reg.angles["train"][sub], reg.y["train"][sub]
        small = stratified_cap(A, y, cfg_geo["scan_subset"], seed=0)
        for enc in ("angle_ry", "zz"):
            states = compute_states(A[small], enc, bandwidth=bw)
            K = fidelity_kernel(states, states)
            scan.append({"encoding": enc, "qubits": q, **_offdiag_stats(K, shots_list)})
            log.info("scan %s q%d: var %.3e", enc, q, scan[-1]["var"])

    slopes = {}
    for enc in ("angle_ry", "zz"):
        pts = [(r["qubits"], r["var"]) for r in scan if r["encoding"] == enc and r["var"] > 0]
        n, lv = np.array([p[0] for p in pts]), np.log2([p[1] for p in pts])
        slopes[enc] = float(np.polyfit(n, lv, 1)[0])
    log.info("log2-variance slope per qubit: %s (2^-n concentration = -1.0)", slopes)
    return {
        "cached_tuned_grams": cached,
        "fixed_bandwidth_scan": {"bandwidth": bw, "rows": scan},
        "log2_var_slope_per_qubit": slopes,
    }


def _iter_cached_grams():
    from qgridbench.models.quantum.qkernel import KERNEL_CACHE

    for meta_path in sorted(KERNEL_CACHE.glob("*.json")):
        yield meta_path, json.loads(meta_path.read_text())


def _latest_qkernel_results(tag):
    """rbf_gamma + diagnostics for a tag, from the newest qkernel run dir (fail loud)."""
    dirs = sorted((REPO_ROOT / "results" / "runs").glob(f"*_quantum_qkernel_{tag}"))
    if not dirs:
        raise FileNotFoundError(f"no qkernel run dir for tag {tag}")
    return json.loads((dirs[-1] / "results.json").read_text())


def relabel_control(variant, qubit_list, cfg_rel):
    """Huang et al. positive control: can this pipeline detect a quantum win at all?

    For each cached Gram, build labels from the g eigenvector (quantum_easy_labels)
    and pit the quantum kernel against classical models RE-TUNED on the relabeled
    task, on the same features, subset, and cap. A quantum win here proves the
    Phase-3 null is a property of the grid labels, not a pipeline artifact — and
    the size of the win should track the measured g.
    """
    from sklearn.metrics import f1_score
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.model_selection import StratifiedKFold, train_test_split

    C_grid = cfg_rel["svm_C_grid"]
    gcfg = cfg_rel["rbf_gamma_grid"]
    gamma_grid = np.logspace(gcfg["low"], gcfg["high"], gcfg["num"])
    xgb_grid = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        for n in cfg_rel["xgb_grid"]["n_estimators"]
        for d in cfg_rel["xgb_grid"]["max_depth"]
        for lr in cfg_rel["xgb_grid"]["learning_rate"]
    ]

    def cv_folds(y, tr):
        return list(StratifiedKFold(3, shuffle=True, random_state=0).split(tr, y[tr]))

    def qsvm_f1(K, y, tr, te):
        def cv(C):
            scores = []
            for t, v in cv_folds(y, tr):
                clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
                clf.fit(K[np.ix_(tr[t], tr[t])], y[tr[t]])
                pred = clf.predict(K[np.ix_(tr[v], tr[t])])
                scores.append(f1_score(y[tr[v]], pred, average="macro"))
            return np.mean(scores)

        best_C = max(C_grid, key=cv)
        clf = SVC(kernel="precomputed", C=best_C, class_weight="balanced")
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        return float(f1_score(y[te], clf.predict(K[np.ix_(te, tr)]), average="macro"))

    def rbf_f1(X, y, tr, te):
        def cv(p):
            g, C = p
            scores = []
            for t, v in cv_folds(y, tr):
                clf = SVC(gamma=g, C=C, class_weight="balanced").fit(X[tr[t]], y[tr[t]])
                scores.append(f1_score(y[tr[v]], clf.predict(X[tr[v]]), average="macro"))
            return np.mean(scores)

        best_g, best_C = max(((g, C) for g in gamma_grid for C in C_grid), key=cv)
        clf = SVC(gamma=best_g, C=best_C, class_weight="balanced").fit(X[tr], y[tr])
        return float(f1_score(y[te], clf.predict(X[te]), average="macro"))

    def xgb_f1(X, y, tr, te):
        def cv(params):
            scores = []
            for t, v in cv_folds(y, tr):
                est = build("xgb", params, 0)
                fit_model("xgb", est, X[tr[t]], y[tr[t]])
                scores.append(f1_score(y[tr[v]], est.predict(X[tr[v]]), average="macro"))
            return np.mean(scores)

        best = max(xgb_grid, key=cv)
        est = build("xgb", best, 0)
        fit_model("xgb", est, X[tr], y[tr])
        return float(f1_score(y[te], est.predict(X[te]), average="macro"))

    rows = []
    for q in qubit_list:
        tag = f"{variant}_q{q}"
        reg = prepare_regime(variant, "pca", n_components=q, seed=0)
        sub = kernel_subset(reg, 2000, seed=0)
        X_pca, A = reg.X["train"][sub], reg.angles["train"][sub]
        for enc in ("angle_ry", "zz"):
            try:
                K_q, meta = find_cached_kernel(encoding=enc, qubits=q, tag=tag)
            except FileNotFoundError:
                log.info("relabel %s q%d %s: no cached Gram, cell skipped", variant, q, enc)
                continue
            if K_q.shape != (len(sub), len(sub)):
                raise ValueError(f"{tag} {enc}: Gram shape {K_q.shape} != subset {len(sub)}")
            # integrity: a recomputed corner at the cached bandwidth must match
            corner_states = compute_states(A[:5], enc, bandwidth=meta["bandwidth"])
            if not np.allclose(
                fidelity_kernel(corner_states, corner_states), K_q[:5, :5], atol=1e-4
            ):
                raise ValueError(f"{tag} {enc}: cached Gram does not match recomputed corner")

            res = _latest_qkernel_results(tag)[enc]
            K_c = rbf_kernel(X_pca, gamma=res["rbf_gamma"])
            y_new, g = quantum_easy_labels(K_c, K_q)
            g_reported = res["diagnostics"]["geometric_difference"]
            if abs(g - g_reported) > 0.05 * g_reported:
                raise ValueError(f"{tag} {enc}: recomputed g {g:.2f} != reported {g_reported:.2f}")

            # win measured over several train/test resamples -> mean +/- std, so the
            # paper never cites a single-split number (single-split was the scout form)
            idx = np.arange(len(y_new))
            per_seed = []
            for split_seed in cfg_rel["split_seeds"]:
                tr, te = train_test_split(
                    idx, test_size=cfg_rel["test_size"], stratify=y_new, random_state=split_seed
                )
                f1s = {
                    "f1_qsvm": qsvm_f1(K_q, y_new, tr, te),
                    "f1_rbf_retuned": rbf_f1(X_pca, y_new, tr, te),
                    "f1_xgb_retuned": xgb_f1(X_pca, y_new, tr, te),
                }
                f1s["quantum_win"] = f1s["f1_qsvm"] - max(
                    f1s["f1_rbf_retuned"], f1s["f1_xgb_retuned"]
                )
                per_seed.append({"split_seed": split_seed, **f1s})
            row = {
                "cell": f"{enc}_q{q}_{variant}",
                "g": g,
                "kta_q_grid_labels": res["diagnostics"]["kta_quantum"],
                "kta_q_easy_labels": kernel_target_alignment(K_q, y_new),
                "kta_rbf_easy_labels": kernel_target_alignment(K_c, y_new),
                "per_seed": per_seed,
            }
            for k in ("f1_qsvm", "f1_rbf_retuned", "f1_xgb_retuned", "quantum_win"):
                vals = np.array([s[k] for s in per_seed])
                row[f"{k}_mean"] = float(vals.mean())
                row[f"{k}_std"] = float(vals.std(ddof=1))
            rows.append(row)
            log.info(
                "relabel %s: g %.1f  qsvm %.3f  win %+.4f +/- %.4f over %d splits",
                row["cell"],
                g,
                row["f1_qsvm_mean"],
                row["quantum_win_mean"],
                row["quantum_win_std"],
                len(per_seed),
            )
    if not rows:
        raise FileNotFoundError(f"relabel_control: no cached Grams for variant {variant}")
    return rows


def bandwidth_curve(variant, cfg_bw):
    """The bandwidth axis: dequantization <-> degeneracy, and where tuning landed.

    Small bandwidth restricts the encoding to a near-classical kernel (Canatar et
    al. 2023; arXiv:2503.05602); large bandwidth spreads states toward mutual
    orthogonality and a near-identity Gram (Shaydulin & Wild, PRA 106.042407) —
    the degeneracy this project measured in the tuned zz q12/16 cells, whose
    Optuna optima (1.94/1.92) sit essentially AT the 2.0 search-box edge. This
    sweep traces geometry (median fidelity, Gram variance, top-eigenvalue share),
    diagnostics (KTA, g vs the SAME tuned classical RBF the reported diagnostics
    used), and VAL macro-F1 (C re-tuned per point) across bandwidths in and
    beyond the box. The test split is never touched.

    Question it answers: does ANY bandwidth give large g without fidelity
    collapse — i.e., is the map's exotic geometry ever populated on this data?
    """
    from sklearn.metrics import f1_score
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.model_selection import StratifiedKFold

    C_grid = cfg_bw["svm_C_grid"]

    def val_f1(K_tr, y_tr, K_va, y_va):
        def cv(C):
            scores = []
            folds = StratifiedKFold(3, shuffle=True, random_state=0).split(K_tr, y_tr)
            for t, v in folds:
                clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
                clf.fit(K_tr[np.ix_(t, t)], y_tr[t])
                pred = clf.predict(K_tr[np.ix_(v, t)])
                scores.append(f1_score(y_tr[v], pred, average="macro"))
            return np.mean(scores)

        best_C = max(C_grid, key=cv)
        clf = SVC(kernel="precomputed", C=best_C, class_weight="balanced").fit(K_tr, y_tr)
        return float(f1_score(y_va, clf.predict(K_va), average="macro")), float(best_C)

    rows = []
    for enc, q in cfg_bw["cells"]:
        tag = f"{variant}_q{q}"
        try:
            res = _latest_qkernel_results(tag)[enc]
        except (FileNotFoundError, KeyError):
            log.info("bandwidth %s: no tuned run dir for %s q%d, cell skipped", variant, enc, q)
            continue
        reg = prepare_regime(variant, "pca", n_components=q, seed=0)
        sub = kernel_subset(reg, 2000, seed=0)
        A_tr, y_tr = reg.angles["train"][sub], reg.y["train"][sub]
        X_pca = reg.X["train"][sub]
        va = stratified_cap(reg.angles["val"], reg.y["val"], cfg_bw["val_subset"], seed=0)
        A_va, y_va = reg.angles["val"][va], reg.y["val"][va]

        K_c = rbf_kernel(X_pca, gamma=res["rbf_gamma"])
        tuned_bw = res["best_params"]["bandwidth"]

        for bw in cfg_bw["bandwidths"]:
            s_tr = compute_states(A_tr, enc, bandwidth=bw)
            K_tr = fidelity_kernel(s_tr, s_tr)
            K_va = fidelity_kernel(compute_states(A_va, enc, bandwidth=bw), s_tr)
            offd = K_tr[np.triu_indices_from(K_tr, k=1)]
            evals = np.linalg.eigvalsh(K_tr)
            f1, best_C = val_f1(K_tr, y_tr, K_va, y_va)
            rows.append(
                {
                    "cell": f"{enc}_q{q}_{variant}",
                    "bandwidth": float(bw),
                    "in_tuning_box": bool(bw <= cfg_bw["tuning_box_high"]),
                    "tuned_bandwidth": float(tuned_bw),
                    "median_offdiag": float(np.median(offd)),
                    "frac_below_0.01": float(np.mean(offd < 0.01)),
                    "var_offdiag": float(offd.var()),
                    "top_eig_share": float(evals[-1] / evals.sum()),
                    "kta_grid_labels": kernel_target_alignment(K_tr, y_tr),
                    "g_vs_tuned_rbf": geometric_difference(K_c, K_tr),
                    "val_macro_f1": f1,
                    "best_C": best_C,
                }
            )
            r = rows[-1]
            log.info(
                "bw %s bw=%.2f: median %.2e  g %.1f  KTA %.4f  valF1 %.4f",
                r["cell"],
                bw,
                r["median_offdiag"],
                r["g_vs_tuned_rbf"],
                r["kta_grid_labels"],
                r["val_macro_f1"],
            )
    if not rows:
        raise FileNotFoundError(f"bandwidth_curve: no tuned cells for variant {variant}")
    return rows


def bandwidth_fragility(variant, cfg_frag):
    """Does accuracy-tuning CAUSE the fragility? Flip rate vs bandwidth, one QSVM per point.

    Phase 4 measured the deployed QSVM flipping 45% of predictions under 0.5-sigma
    random L-inf noise; the kernel-geometry stage tied that to a spiky near-orthogonal
    Gram. This closes the causal loop: rebuild the same QSVM at bandwidths spanning
    near-constant -> healthy -> degenerate, and measure the flip rate at each. The
    perturbation set is generated ONCE and reused across bandwidths, so the
    comparison is paired. Flip rate is measured against each model's OWN clean
    predictions (the Phase-4 control convention); clean macro-F1 and the predicted
    positive rate are reported beside it, because a near-constant kernel is expected
    to be "stable" only by collapsing to the prior.
    """
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    enc, q = cfg_frag["cell"]
    reg = prepare_regime(variant, "pca", n_components=q, seed=0)
    sub = kernel_subset(reg, 2000, seed=0)
    A_tr, y_tr = reg.angles["train"][sub], reg.y["train"][sub]
    scale = reg.feature_scale
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_frag["eval_subset"], seed=0)
    X_ev, y_ev = reg.X["test"][ev], reg.y["test"][ev]
    C_grid = cfg_frag["svm_C_grid"]

    # one shared perturbation set: {radius: [delta_draw1, delta_draw2, ...]}
    rng = np.random.default_rng(0)
    deltas = {
        r: [rng.uniform(-1.0, 1.0, X_ev.shape) * r * scale for _ in range(cfg_frag["n_draws"])]
        for r in cfg_frag["radii_sigma"]
    }

    def cv_c(K_tr, C):
        scores = []
        for t, v in StratifiedKFold(3, shuffle=True, random_state=0).split(K_tr, y_tr):
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_tr[np.ix_(t, t)], y_tr[t])
            pred = clf.predict(K_tr[np.ix_(v, t)])
            scores.append(f1_score(y_tr[v], pred, average="macro"))
        return np.mean(scores)

    rows = []
    for bw in cfg_frag["bandwidths"]:
        s_tr = compute_states(A_tr, enc, bandwidth=bw)
        K_tr = fidelity_kernel(s_tr, s_tr)
        best_C = max(C_grid, key=lambda C: cv_c(K_tr, C))
        clf = SVC(kernel="precomputed", C=best_C, class_weight="balanced").fit(K_tr, y_tr)

        def predict(X, _s_tr=s_tr, _clf=clf, _bw=bw):
            A = reg.angle_scaler.transform(X)
            return _clf.predict(fidelity_kernel(compute_states(A, enc, bandwidth=_bw), _s_tr))

        base = predict(X_ev)
        flips = {}
        for r, dlist in deltas.items():
            rates = [float(np.mean(predict(X_ev + d) != base)) for d in dlist]
            flips[str(r)] = {"mean": float(np.mean(rates)), "draws": rates}
        rows.append(
            {
                "cell": f"{enc}_q{q}_{variant}",
                "bandwidth": float(bw),
                "best_C": float(best_C),
                "clean_macro_f1": float(f1_score(y_ev, base, average="macro")),
                "pred_positive_rate": float(np.mean(base == 1)),
                "true_positive_rate": float(np.mean(y_ev == 1)),
                "flip_rate": flips,
            }
        )
        log.info(
            "fragility bw=%.2f: clean F1 %.3f  pred-pos %.2f  flip@0.5 %.2f",
            bw,
            rows[-1]["clean_macro_f1"],
            rows[-1]["pred_positive_rate"],
            flips[str(cfg_frag["radii_sigma"][-1])]["mean"],
        )
    return rows


def dequantization_check(variant, qubit_list, cfg_deq):
    """Prediction agreement between each tuned quantum-kernel SVM and a best-effort
    classical RBF SVM on the same subset — the dequantization claim, measured.

    Reports raw agreement AND Cohen's kappa (raw agreement is inflated by the 71/29
    prior), plus a REFERENCE pair — the agreement between two best-effort classical
    families (RBF vs XGBoost) on the same split — because near-chance models disagree
    on noisy boundaries for free, and "quantum vs classical agree X%" is meaningless
    until compared with how much two classical families agree with each other.
    Scored on the val subsample; the test split stays frozen. Binary only —
    fails loud on a multiclass variant rather than silently averaging OvR heads.
    """
    from sklearn.metrics import cohen_kappa_score, f1_score
    from sklearn.model_selection import StratifiedKFold

    C_grid = cfg_deq["svm_C_grid"]
    gcfg = cfg_deq["rbf_gamma_grid"]
    gamma_grid = np.logspace(gcfg["low"], gcfg["high"], gcfg["num"])
    xgb_grid = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        for n in (200, 400)
        for d in (3, 6)
        for lr in (0.1, 0.3)
    ]

    rows = []
    for q in qubit_list:
        tag = f"{variant}_q{q}"
        reg = prepare_regime(variant, "pca", n_components=q, seed=0)
        if len(np.unique(reg.y["train"])) != 2:
            raise ValueError("dequantization_check supports binary variants only")
        sub = kernel_subset(reg, 2000, seed=0)
        X_tr, A_tr, y_tr = reg.X["train"][sub], reg.angles["train"][sub], reg.y["train"][sub]
        va = stratified_cap(reg.X["val"], reg.y["val"], cfg_deq["val_subset"], seed=0)
        X_va, A_va, y_va = reg.X["val"][va], reg.angles["val"][va], reg.y["val"][va]

        def cv_rbf(gamma, C):
            scores = []
            for t, v in StratifiedKFold(3, shuffle=True, random_state=0).split(X_tr, y_tr):
                clf = SVC(gamma=gamma, C=C, class_weight="balanced").fit(X_tr[t], y_tr[t])
                scores.append(f1_score(y_tr[v], clf.predict(X_tr[v]), average="macro"))
            return np.mean(scores)

        def cv_xgb(params):
            scores = []
            for t, v in StratifiedKFold(3, shuffle=True, random_state=0).split(X_tr, y_tr):
                est = build("xgb", params, 0)
                fit_model("xgb", est, X_tr[t], y_tr[t])
                scores.append(f1_score(y_tr[v], est.predict(X_tr[v]), average="macro"))
            return np.mean(scores)

        # reference pair: two best-effort CLASSICAL families on the identical split
        best_xgb = max(xgb_grid, key=cv_xgb)
        est_x = build("xgb", best_xgb, 0)
        fit_model("xgb", est_x, X_tr, y_tr)
        pred_x = est_x.predict(X_va)

        for enc in ("angle_ry", "zz"):
            try:
                K_q, meta = find_cached_kernel(encoding=enc, qubits=q, tag=tag)
            except FileNotFoundError:
                log.info("dequant %s q%d %s: no cached Gram, cell skipped", variant, q, enc)
                continue
            res = _latest_qkernel_results(tag)[enc]
            bw = meta["bandwidth"]
            s_tr = compute_states(A_tr, enc, bandwidth=bw)
            if not np.allclose(fidelity_kernel(s_tr[:5], s_tr[:5]), K_q[:5, :5], atol=1e-4):
                raise ValueError(f"{tag} {enc}: cached Gram does not match recomputed corner")
            K_va = fidelity_kernel(compute_states(A_va, enc, bandwidth=bw), s_tr)
            clf_q = SVC(kernel="precomputed", C=res["best_params"]["C"], class_weight="balanced")
            clf_q.fit(K_q, y_tr)
            pred_q = clf_q.predict(K_va)

            best_g, best_C = max(
                ((g, C) for g in gamma_grid for C in C_grid), key=lambda p: cv_rbf(*p)
            )
            clf_c = SVC(gamma=best_g, C=best_C, class_weight="balanced").fit(X_tr, y_tr)
            pred_c = clf_c.predict(X_va)

            rows.append(
                {
                    "cell": f"{enc}_q{q}_{variant}",
                    "bandwidth": float(bw),
                    "agreement": float(np.mean(pred_q == pred_c)),
                    "cohen_kappa": float(cohen_kappa_score(pred_q, pred_c)),
                    "ref_agreement_rbf_xgb": float(np.mean(pred_c == pred_x)),
                    "ref_kappa_rbf_xgb": float(cohen_kappa_score(pred_c, pred_x)),
                    "f1_qsvm_val": float(f1_score(y_va, pred_q, average="macro")),
                    "f1_rbf_val": float(f1_score(y_va, pred_c, average="macro")),
                    "f1_xgb_val": float(f1_score(y_va, pred_x, average="macro")),
                    "rbf_gamma": float(best_g),
                    "rbf_C": float(best_C),
                }
            )
            r = rows[-1]
            log.info(
                "dequant %s: agree %.3f (kappa %.3f) | ref rbf-xgb %.3f (kappa %.3f) | "
                "F1 q %.3f / rbf %.3f / xgb %.3f",
                r["cell"],
                r["agreement"],
                r["cohen_kappa"],
                r["ref_agreement_rbf_xgb"],
                r["ref_kappa_rbf_xgb"],
                r["f1_qsvm_val"],
                r["f1_rbf_val"],
                r["f1_xgb_val"],
            )
    if not rows:
        raise FileNotFoundError(f"dequantization_check: no cached Grams for variant {variant}")
    return rows


def data_efficiency(variant, n_qubits, sizes, seeds, cap):
    """best classical (xgb, full) + qkernel(angle_ry) + vqc on subset sizes."""
    reg_full = prepare_regime(variant, "full", seed=0)
    reg_pca = prepare_regime(variant, "pca", n_components=n_qubits, seed=0)
    yte_full = reg_full.y["test"]
    out = {"xgb_full": {}, "qkernel_angle_ry": {}, "vqc": {}}
    for size in sizes:
        idx = stratified_cap(reg_full.X["train"], reg_full.y["train"], size, seed=0)
        # classical xgb (full features)
        f_xgb = []
        for s in seeds:
            set_all_seeds(s)
            est = build(
                "xgb",
                {
                    "n_estimators": 200,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "min_child_weight": 1,
                    "reg_lambda": 1.0,
                },
                s,
            )
            fit_model("xgb", est, reg_full.X["train"][idx], reg_full.y["train"][idx])
            f_xgb.append(compute_all(yte_full, est.predict_proba(reg_full.X["test"]))["macro_f1"])
        out["xgb_full"][str(size)] = {
            "mean": float(np.mean(f_xgb)),
            "std": float(np.std(f_xgb, ddof=1)),
        }
        # quantum kernel (pca/angle)
        idxp = stratified_cap(reg_pca.X["train"], reg_pca.y["train"], size, seed=0)
        Atr, ytr = reg_pca.angles["train"][idxp], reg_pca.y["train"][idxp]
        Ate, yte = reg_pca.angles["test"], reg_pca.y["test"]
        K_tr = fidelity_kernel(compute_states(Atr, "angle_ry"), compute_states(Atr, "angle_ry"))
        K_te = fidelity_kernel(compute_states(Ate, "angle_ry"), compute_states(Atr, "angle_ry"))
        out["qkernel_angle_ry"][str(size)] = {
            "mean": float(_qk_f1(K_tr, K_te, ytr, yte)),
            "std": 0.0,
        }
        # vqc
        f_vqc = []
        for s in seeds:
            set_all_seeds(s)
            m = VQClassifier(
                n_qubits,
                depth=4,
                n_classes=len(np.unique(ytr)),
                lr=0.05,
                batch_size=64,
                max_epochs=40,
                patience=8,
                seed=s,
            )
            m.fit(Atr, ytr, reg_pca.angles["val"][:500], reg_pca.y["val"][:500])
            f_vqc.append(compute_all(yte, m.predict_proba(Ate))["macro_f1"])
        out["vqc"][str(size)] = {"mean": float(np.mean(f_vqc)), "std": float(np.std(f_vqc, ddof=1))}
        log.info("data-eff size %d done", size)
    return out


def confusion_three_class(variant, regime, n_components, tuned_run, seeds, cap):
    """Per-class metrics + confusion matrices for the three-class task (study protocol §8.8).

    The operationally load-bearing cell is natural-vs-attack: a grid operator's first question
    of any detector is how often it calls a legitimate disturbance an attack, and macro-F1
    cannot answer it. Aggregate metrics were the only thing the Phase-2 runs persisted, so this
    re-scores the ALREADY-TUNED configs over the same 10 seeds. No search is re-run and no
    hyperparameter is re-selected, so the test-split-touched-once protocol is unaffected.

    Counts are summed over seeds (a per-seed mean of an integer matrix is not a confusion
    matrix); rates are normalised per true class so the 5.6/23.4/71.0 imbalance cannot make a
    rare-class error look small.
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    tuned = json.loads((REPO_ROOT / tuned_run / "results.json").read_text())
    reg = prepare_regime(variant, regime, n_components=n_components, seed=0)
    Xtr, ytr, Xte, yte = reg.X["train"], reg.y["train"], reg.X["test"], reg.y["test"]
    cap_idx = stratified_cap(Xtr, ytr, cap, seed=0) if cap else np.arange(len(ytr))
    labels = sorted(np.unique(yte).tolist())
    out = {
        "regime": regime,
        "n_components": n_components,
        "tuned_run": str(tuned_run),
        "seeds": list(seeds),
        "labels": labels,
        "class_support": {str(c): int((yte == c).sum()) for c in labels},
        "per_model": {},
    }
    for name, blk in tuned.items():
        if "best_params" not in blk:
            continue
        cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
        prf = []
        for sd in seeds:
            set_all_seeds(sd)
            est = build(name, blk["best_params"], sd)
            fit_model(name, est, Xtr[cap_idx], ytr[cap_idx])
            pred = est.predict(Xte)
            cm += confusion_matrix(yte, pred, labels=labels)
            p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=labels, zero_division=0)
            prf.append(np.stack([p, r, f]))
        prf = np.stack(prf)  # (seeds, 3, n_classes)
        row = cm.sum(axis=1, keepdims=True)
        out["per_model"][name] = {
            "confusion_counts": cm.tolist(),
            "confusion_rates": (cm / np.maximum(row, 1)).tolist(),
            "per_class": {
                str(c): {
                    "precision": {
                        "mean": float(prf[:, 0, i].mean()),
                        "std": float(prf[:, 0, i].std(ddof=1)),
                    },
                    "recall": {
                        "mean": float(prf[:, 1, i].mean()),
                        "std": float(prf[:, 1, i].std(ddof=1)),
                    },
                    "f1": {
                        "mean": float(prf[:, 2, i].mean()),
                        "std": float(prf[:, 2, i].std(ddof=1)),
                    },
                }
                for i, c in enumerate(labels)
            },
        }
        # labels are 0=no-event, 1=natural, 2=attack: the two cells an operator acts on
        nat, atk = labels.index(1), labels.index(2)
        fa = float(cm[nat, atk] / max(cm[nat].sum(), 1))
        miss = float(cm[atk, nat] / max(cm[atk].sum(), 1))
        out["per_model"][name]["natural_called_attack"] = fa
        out["per_model"][name]["attack_called_natural"] = miss
        log.info(
            "confusion %s (%s): natural->attack %.3f, attack->natural %.3f, per-class F1 %s",
            name,
            regime,
            fa,
            miss,
            {
                str(c): round(out["per_model"][name]["per_class"][str(c)]["f1"]["mean"], 3)
                for c in labels
            },
        )
    return out


def bandwidth_fragility_classical(variant, cfg_frag, gamma_mults=(100.0, 10.0, 1.0, 0.1, 0.01)):
    """The MATCHED CLASSICAL CONTROL for the quantum fragility co-peak (P16a / N5).

    `bandwidth_fragility` measured, on the quantum side only, that random-flip rate co-peaks
    with clean accuracy across encoding bandwidth. Read alone that is a property of a sweep,
    not of a family: kernel machines might all behave this way. Fawzi et al. report the
    opposite direction classically -- a NARROWER RBF is more adversarially robust -- so the
    question is whether the co-peak inverts a classical relation or merely restates one.

    This runs the identical protocol on the classical RBF with only the kernel swapped: the
    same shared perturbation set (generated once, reused across widths, so the comparison is
    paired), the same C selected per width by 3-fold CV from the same grid, the same 100
    evaluation points, the same draws. Sweeping gamma about the tuned value plays the role
    bandwidth plays quantumly.

    KILL: if the classical curve ALSO co-peaks, there is no inversion -- the co-peak is a
    kernel-machine property and the quantum framing is dropped.
    """
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC as _SVC

    _, q = cfg_frag["cell"]
    reg = prepare_regime(variant, "pca", n_components=q, seed=0)
    sub = kernel_subset(reg, 2000, seed=0)
    Xtr, ytr = reg.X["train"][sub], reg.y["train"][sub]
    scale = reg.feature_scale
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_frag["eval_subset"], seed=0)
    X_ev, y_ev = reg.X["test"][ev], reg.y["test"][ev]
    best = json.loads((REPO_ROOT / "results/adv_best_params_binary_q8.json").read_text())
    g_tuned = float(best["rbf_svm"]["gamma"])

    rng = np.random.default_rng(0)  # SAME seed as the quantum sweep -> same perturbations
    deltas = {
        r: [rng.uniform(-1.0, 1.0, X_ev.shape) * r * scale for _ in range(cfg_frag["n_draws"])]
        for r in cfg_frag["radii_sigma"]
    }
    rows = []
    for m in gamma_mults:
        g = g_tuned * m

        def cv(C, _g=g):
            sc = []
            for t, v in StratifiedKFold(3, shuffle=True, random_state=0).split(Xtr, ytr):
                clf = _SVC(C=C, gamma=_g, class_weight="balanced").fit(Xtr[t], ytr[t])
                sc.append(f1_score(ytr[v], clf.predict(Xtr[v]), average="macro"))
            return float(np.mean(sc))

        best_C = max(cfg_frag["svm_C_grid"], key=cv)
        clf = _SVC(C=best_C, gamma=g, class_weight="balanced").fit(Xtr, ytr)
        base = clf.predict(X_ev)
        flips = {}
        for r, dlist in deltas.items():
            rates = [float(np.mean(clf.predict(X_ev + d) != base)) for d in dlist]
            flips[str(r)] = {"mean": float(np.mean(rates)), "draws": rates}
        rows.append(
            {
                "gamma": float(g),
                "gamma_mult": float(m),
                "best_C": float(best_C),
                "clean_macro_f1": float(f1_score(y_ev, base, average="macro")),
                "pred_positive_rate": float(np.mean(base == 1)),
                "flip_rate": flips,
            }
        )
        log.info(
            "classical fragility gamma=%.4g (x%.3g): clean F1 %.3f  pred-pos %.2f  flip@0.5 %.2f",
            g,
            m,
            rows[-1]["clean_macro_f1"],
            rows[-1]["pred_positive_rate"],
            flips[str(cfg_frag["radii_sigma"][-1])]["mean"],
        )
    top = cfg_frag["radii_sigma"][-1]
    acc_arg = max(rows, key=lambda r: r["clean_macro_f1"])
    flip_arg = max(rows, key=lambda r: r["flip_rate"][str(top)]["mean"])
    verdict = {
        "accuracy_optimal_gamma": acc_arg["gamma"],
        "fragility_optimal_gamma": flip_arg["gamma"],
        "co_peaks": bool(acc_arg["gamma"] == flip_arg["gamma"]),
        "flip_at_accuracy_optimum": acc_arg["flip_rate"][str(top)]["mean"],
        "max_flip": flip_arg["flip_rate"][str(top)]["mean"],
    }
    log.info(
        "classical fragility verdict: accuracy-optimal gamma %.4g, fragility-optimal %.4g, "
        "co-peaks=%s (flip at accuracy optimum %.2f vs max %.2f)",
        verdict["accuracy_optimal_gamma"],
        verdict["fragility_optimal_gamma"],
        verdict["co_peaks"],
        verdict["flip_at_accuracy_optimum"],
        verdict["max_flip"],
    )
    return {"rows": rows, "verdict": verdict}


def entanglement_isolation(variant, cfg_frag, strengths=(0.0, 0.25, 0.5, 0.75, 1.0), gate_n=48):
    """Does ENTANGLEMENT carry the accuracy/fragility inversion? (P17b)

    P17a killed the boundedness explanation but compared two maps differing in four ways at
    once (entanglement, rotation structure, repetitions, quadratic pair scaling), so it named
    entanglement as the candidate without isolating it. The ZZ ring block is
    `CNOT; RZ(2 s x_i x_j); CNOT`: at s=0 the rotation is the identity and the CNOTs cancel
    exactly, leaving the SAME H+RZ product map with the SAME repetitions and scaling. So
    sweeping s varies entanglement and nothing else, and s=1 is the deployed circuit.

    Verdict per strength is the pre-registered ratio: flip rate at that strength's
    accuracy-optimal bandwidth over its own maximum flip rate (ZZ scores 1.00, the P17a
    product arm 0.51). KILL: ratio flat across s (max - min <= 0.15) => entanglement is not
    the axis.
    """
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    enc, q = cfg_frag["cell"]
    if enc != "zz":
        raise ValueError(f"entanglement isolation needs the zz map, got '{enc}'")
    reg = prepare_regime(variant, "pca", n_components=q, seed=0)
    sub = kernel_subset(reg, 2000, seed=0)
    A_tr, y_tr = reg.angles["train"][sub], reg.y["train"][sub]
    scale = reg.feature_scale
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_frag["eval_subset"], seed=0)
    X_ev, y_ev = reg.X["test"][ev], reg.y["test"][ev]

    rng = np.random.default_rng(0)  # SAME seed as every other arm -> same perturbations
    deltas = {
        r: [rng.uniform(-1.0, 1.0, X_ev.shape) * r * scale for _ in range(cfg_frag["n_draws"])]
        for r in cfg_frag["radii_sigma"]
    }

    # GATE 1 (anchor): s=1.0 must BE the shipped circuit, else the sweep measures a different
    # model than the paper's and nothing from it is usable.
    g = A_tr[:gate_n].astype(np.float64)
    anchor = float(
        np.abs(
            compute_states(g, "zz", dtype=np.complex128, entangle_strength=1.0)
            - compute_states(g, "zz", dtype=np.complex128)
        ).max()
    )
    if not anchor == 0.0:
        raise RuntimeError(
            f"s=1.0 does not reproduce the shipped zz circuit: max|dev| {anchor:.3e}"
        )
    # GATE 2 (product limit): s=0.0 must be genuinely unentangled -- Schmidt rank 1 across a
    # 1-vs-rest cut, else "the ring vanished" is an assumption rather than a fact.
    s0 = compute_states(g, "zz", dtype=np.complex128, entangle_strength=0.0)
    second = float(max(np.linalg.svd(v.reshape(2, -1), compute_uv=False)[1] for v in s0))
    if not second < 1e-10:
        raise RuntimeError(f"s=0.0 is still entangled: second Schmidt value {second:.3e}")
    log.info("gates passed: s=1 anchor dev %.1e, s=0 second Schmidt value %.1e", anchor, second)

    def cv_c(K_tr, C):
        scores = []
        for t, v in StratifiedKFold(3, shuffle=True, random_state=0).split(K_tr, y_tr):
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_tr[np.ix_(t, t)], y_tr[t])
            scores.append(f1_score(y_tr[v], clf.predict(K_tr[np.ix_(v, t)]), average="macro"))
        return float(np.mean(scores))

    top, rows = str(cfg_frag["radii_sigma"][-1]), []
    for s in strengths:
        for bw in cfg_frag["bandwidths"]:
            s_tr = compute_states(A_tr, enc, bandwidth=bw, entangle_strength=s)
            K_tr = fidelity_kernel(s_tr, s_tr)
            best_C = max(cfg_frag["svm_C_grid"], key=lambda C: cv_c(K_tr, C))
            clf = SVC(kernel="precomputed", C=best_C, class_weight="balanced").fit(K_tr, y_tr)

            def predict(X, _s_tr=s_tr, _clf=clf, _bw=bw, _s=s):
                A = reg.angle_scaler.transform(X)
                st = compute_states(A, enc, bandwidth=_bw, entangle_strength=_s)
                return _clf.predict(fidelity_kernel(st, _s_tr))

            base = predict(X_ev)
            flips = {}
            for r, dlist in deltas.items():
                rates = [float(np.mean(predict(X_ev + d) != base)) for d in dlist]
                flips[str(r)] = {"mean": float(np.mean(rates)), "draws": rates}
            rows.append(
                {
                    "entangle_strength": float(s),
                    "bandwidth": float(bw),
                    "best_C": float(best_C),
                    "clean_macro_f1": float(f1_score(y_ev, base, average="macro")),
                    "pred_positive_rate": float(np.mean(base == 1)),
                    "flip_rate": flips,
                }
            )
            log.info(
                "entangle s=%.2f bw=%.2f: clean F1 %.3f  pred-pos %.2f  flip@0.5 %.2f",
                s,
                bw,
                rows[-1]["clean_macro_f1"],
                rows[-1]["pred_positive_rate"],
                flips[top]["mean"],
            )

    per_strength = {}
    for s in strengths:
        rs = [r for r in rows if r["entangle_strength"] == float(s)]
        acc = max(rs, key=lambda r: r["clean_macro_f1"])
        mx = max(r["flip_rate"][top]["mean"] for r in rs)
        per_strength[str(s)] = {
            "accuracy_optimal_bandwidth": acc["bandwidth"],
            "flip_at_accuracy_optimum": acc["flip_rate"][top]["mean"],
            "max_flip": mx,
            "ratio": float(acc["flip_rate"][top]["mean"] / mx) if mx > 0 else 0.0,
            "clean_macro_f1_at_optimum": acc["clean_macro_f1"],
            "co_peaks": bool(acc["flip_rate"][top]["mean"] == mx),
        }
    ratios = [per_strength[str(s)]["ratio"] for s in strengths]
    spread = float(max(ratios) - min(ratios))
    verdict = {
        "per_strength": per_strength,
        "ratio_by_strength": ratios,
        "ratio_spread": spread,
        "monotone_increasing": bool(all(b >= a for a, b in zip(ratios, ratios[1:]))),
        # the pre-registered call, computed rather than read by eye
        "outcome": "killed_flat" if spread <= 0.15 else "entanglement_axis_supported",
    }
    log.info(
        "entanglement verdict: ratios %s  spread %.3f  monotone=%s -> %s",
        [round(r, 3) for r in ratios],
        spread,
        verdict["monotone_increasing"],
        verdict["outcome"],
    )
    return {"rows": rows, "verdict": verdict}


def margin_gradient_mechanism(variant, cfg_frag, h=0.01):
    """Is the co-peak a FIRST-ORDER effect -- fragility = margin / gradient? (P17d)

    P17c showed the inversion tracks stationarity but gave no explanation. Under a first-order
    expansion a perturbation delta shifts the decision value by grad.delta; the sweeps' deltas
    are U(-eps*sigma, eps*sigma) per coordinate, so grad.delta has mean 0 and standard
    deviation s = eps*||g||_2/sqrt(3) with g the gradient in sigma-units. A point flips when
    the decision value crosses zero, so by the CLT over the coordinates

        predicted_flip(x) = 2 * Phi(-|f(x)| / s(x)).

    That predicts the RANDOM flip rate the sweeps measured, not a worst-case L-inf bound.

    KILL: Pearson r between predicted and measured flip rate across cells < 0.70 means
    fragility here is not first-order and no mechanism is claimed. This is a real risk -- the
    fidelity kernel's decision regions interleave at fine scale, which is where a first-order
    expansion breaks.
    """
    from scipy.stats import norm, pearsonr
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    enc, q = cfg_frag["cell"]
    reg = prepare_regime(variant, "pca", n_components=q, seed=0)
    sub = kernel_subset(reg, 2000, seed=0)
    A_tr, y_tr = reg.angles["train"][sub], reg.y["train"][sub]
    scale = reg.feature_scale
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_frag["eval_subset"], seed=0)
    X_ev, y_ev = reg.X["test"][ev], reg.y["test"][ev]
    eps = float(cfg_frag["radii_sigma"][-1])

    rng = np.random.default_rng(0)  # SAME seed as every other arm
    deltas = {
        r: [rng.uniform(-1.0, 1.0, X_ev.shape) * r * scale for _ in range(cfg_frag["n_draws"])]
        for r in cfg_frag["radii_sigma"]
    }

    # (label, encoding, entangle_strength, n_reps, stationary)
    arms = [
        ("ry_x1", "angle_ry", None, None, True),
        ("hrz_x1", "zz", 0.0, 1, True),
        ("hrz_x2", "zz", 0.0, 2, False),
        ("zz_deployed", "zz", 1.0, 2, False),
    ]

    def states(A, bw, e, s, nr):
        if e == "angle_ry":
            return compute_states(A, e, bandwidth=bw)
        return compute_states(A, e, bandwidth=bw, entangle_strength=s, n_reps=nr)

    def cv_c(K_tr, C):
        sc = []
        for t, v in StratifiedKFold(3, shuffle=True, random_state=0).split(K_tr, y_tr):
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_tr[np.ix_(t, t)], y_tr[t])
            sc.append(f1_score(y_tr[v], clf.predict(K_tr[np.ix_(v, t)]), average="macro"))
        return float(np.mean(sc))

    rows = []
    for label, e, s, nr, stationary in arms:
        for bw in cfg_frag["bandwidths"]:
            s_tr = states(A_tr, bw, e, s, nr)
            K_tr = fidelity_kernel(s_tr, s_tr)
            best_C = max(cfg_frag["svm_C_grid"], key=lambda C: cv_c(K_tr, C))
            clf = SVC(kernel="precomputed", C=best_C, class_weight="balanced").fit(K_tr, y_tr)

            def dec(X, _s_tr=s_tr, _clf=clf, _bw=bw, _e=e, _s=s, _nr=nr):
                A = reg.angle_scaler.transform(X)
                return _clf.decision_function(fidelity_kernel(states(A, _bw, _e, _s, _nr), _s_tr))

            f0 = dec(X_ev)
            # central differences in sigma-units, uniform across arms
            g = np.empty_like(X_ev)
            for j in range(X_ev.shape[1]):
                step = np.zeros_like(X_ev)
                step[:, j] = h * scale[j]
                g[:, j] = (dec(X_ev + step) - dec(X_ev - step)) / (2.0 * h)
            gnorm = np.linalg.norm(g, axis=1)
            sd = eps * gnorm / np.sqrt(3.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                pred_pt = np.where(
                    sd > 0, 2.0 * norm.cdf(-np.abs(f0) / np.maximum(sd, 1e-300)), 0.0
                )

            # binary SVC predicts sign(decision_function), so a sign change IS a prediction
            # flip -- reuse f0 rather than recomputing the states a second time
            base = clf.classes_[(f0 > 0).astype(int)]
            measured = float(np.mean([np.mean(dec(X_ev + d) * f0 < 0) for d in deltas[eps]]))
            rows.append(
                {
                    "arm": label,
                    "stationary": bool(stationary),
                    "bandwidth": float(bw),
                    "best_C": float(best_C),
                    "clean_macro_f1": float(f1_score(y_ev, base, average="macro")),
                    "predicted_flip": float(np.mean(pred_pt)),
                    "measured_flip": measured,
                    "median_abs_margin": float(np.median(np.abs(f0))),
                    "median_grad_norm": float(np.median(gnorm)),
                    "median_ratio": float(np.median(np.abs(f0) / np.maximum(gnorm, 1e-300))),
                    "frac_pred_underflow": float(np.mean(pred_pt < 1e-6)),
                    "degenerate": bool(f1_score(y_ev, base, average="macro") < 0.51),
                }
            )
            log.info(
                "mechanism %s bw=%.2f: predicted %.3f  measured %.3f  |f| %.3g  ||g|| %.3g",
                label,
                bw,
                rows[-1]["predicted_flip"],
                measured,
                rows[-1]["median_abs_margin"],
                rows[-1]["median_grad_norm"],
            )

    P = np.array([r["predicted_flip"] for r in rows])
    M = np.array([r["measured_flip"] for r in rows])
    r_all, p_all = pearsonr(P, M)
    keep = np.array([not r["degenerate"] for r in rows])
    r_nd, p_nd = pearsonr(P[keep], M[keep]) if keep.sum() > 2 else (float("nan"), float("nan"))

    decomposition = {}
    for label, _e, _s, _nr, stationary in arms:
        rs = [r for r in rows if r["arm"] == label]
        acc = max(rs, key=lambda r: r["clean_macro_f1"])
        med_m = float(np.median([r["median_abs_margin"] for r in rs]))
        med_g = float(np.median([r["median_grad_norm"] for r in rs]))
        decomposition[label] = {
            "stationary": bool(stationary),
            "accuracy_optimal_bandwidth": acc["bandwidth"],
            "margin_at_opt_over_sweep_median": float(acc["median_abs_margin"] / max(med_m, 1e-300)),
            "grad_at_opt_over_sweep_median": float(acc["median_grad_norm"] / max(med_g, 1e-300)),
            "ratio_at_opt_over_sweep_median": float(
                acc["median_ratio"] / max(float(np.median([r["median_ratio"] for r in rs])), 1e-300)
            ),
        }
    verdict = {
        "pearson_r_all_cells": float(r_all),
        "p_value_all_cells": float(p_all),
        "pearson_r_excluding_degenerate": float(r_nd),
        "n_cells": len(rows),
        "n_degenerate": int((~keep).sum()),
        "outcome": "first_order_supported" if r_all >= 0.70 else "killed_not_first_order",
        "decomposition": decomposition,
        "fd_step_sigma": h,
    }
    log.info(
        "mechanism verdict: r(predicted, measured) = %.3f (p=%.2g, n=%d) -> %s",
        r_all,
        p_all,
        len(rows),
        verdict["outcome"],
    )
    return {"rows": rows, "verdict": verdict}


def rotation_structure_isolation(variant, cfg_frag, reps=(1, 3), gate_n=48, gate_tol=1e-12):
    """Is STATIONARITY the axis behind the accuracy/fragility inversion? (P17c)

    P17b left two candidates -- rotation structure (H+RZ vs RY) and repetition count -- but
    the algebra collapses them into one. At ONE repetition the H+RZ overlap is
    prod_i cos^2(c(x_i - y_i)): a function of x - y alone, the same functional family as the
    RY map at twice the bandwidth. At TWO the per-wire state is
    e^{-it}cos t|0> - i e^{it}sin t|1>, whose overlap depends on t_x and t_y separately. And
    RY cannot break this, since RY(t)RY(t) = RY(2t). So the contrast is stationary vs
    non-stationary, and the existing arms already straddle it (RY 0.51, H+RZ x2 0.959,
    classical RBF 0.43).

    Decisive cell: H+RZ at ONE repetition uses the same gates as the 0.959 arm but is
    stationary. If the gate identity mattered it scores HIGH; if stationarity is the axis it
    scores LOW. Opposite predictions, so the cell cannot come out uninformative.

    KILL: ratio >= 0.90 => stationarity is not the axis. [0.75, 0.90) => INDETERMINATE,
    reported as such. Only < 0.75 supports the hypothesis.
    """
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    enc, q = cfg_frag["cell"]
    reg = prepare_regime(variant, "pca", n_components=q, seed=0)
    sub = kernel_subset(reg, 2000, seed=0)
    A_tr, y_tr = reg.angles["train"][sub], reg.y["train"][sub]
    scale = reg.feature_scale
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_frag["eval_subset"], seed=0)
    X_ev, y_ev = reg.X["test"][ev], reg.y["test"][ev]

    rng = np.random.default_rng(0)  # SAME seed as every other arm -> same perturbations
    deltas = {
        r: [rng.uniform(-1.0, 1.0, X_ev.shape) * r * scale for _ in range(cfg_frag["n_draws"])]
        for r in cfg_frag["radii_sigma"]
    }

    def cv_c(K_tr, C):
        scores = []
        for t, v in StratifiedKFold(3, shuffle=True, random_state=0).split(K_tr, y_tr):
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_tr[np.ix_(t, t)], y_tr[t])
            scores.append(f1_score(y_tr[v], clf.predict(K_tr[np.ix_(v, t)]), average="macro"))
        return float(np.mean(scores))

    top, rows, gate = str(cfg_frag["radii_sigma"][-1]), [], []
    for nr in reps:
        for bw in cfg_frag["bandwidths"]:
            # GATE: at one repetition the arm must BE the stationary closed form (at twice the
            # bandwidth), else it is not the kernel the hypothesis is about. Only n_reps=1 has
            # a closed form to check against; higher reps are non-stationary by construction.
            if nr == 1:
                g = A_tr[:gate_n].astype(np.float64)
                st = compute_states(
                    g, enc, bandwidth=bw, dtype=np.complex128, entangle_strength=0.0, n_reps=1
                )
                dev = float(
                    np.abs(
                        fidelity_kernel(st, st, exact=True) - cosine_fidelity_kernel(g, g, 2.0 * bw)
                    ).max()
                )
                if not dev < gate_tol:
                    raise RuntimeError(
                        f"H+RZ x1 at bandwidth {bw} is not the stationary closed form: "
                        f"max|dev| {dev:.3e} >= {gate_tol:.0e}"
                    )
                gate.append({"n_reps": 1, "bandwidth": float(bw), "max_abs_dev": dev})

            s_tr = compute_states(A_tr, enc, bandwidth=bw, entangle_strength=0.0, n_reps=nr)
            K_tr = fidelity_kernel(s_tr, s_tr)
            best_C = max(cfg_frag["svm_C_grid"], key=lambda C: cv_c(K_tr, C))
            clf = SVC(kernel="precomputed", C=best_C, class_weight="balanced").fit(K_tr, y_tr)

            def predict(X, _s_tr=s_tr, _clf=clf, _bw=bw, _nr=nr):
                A = reg.angle_scaler.transform(X)
                st = compute_states(A, enc, bandwidth=_bw, entangle_strength=0.0, n_reps=_nr)
                return _clf.predict(fidelity_kernel(st, _s_tr))

            base = predict(X_ev)
            flips = {}
            for r, dlist in deltas.items():
                rates = [float(np.mean(predict(X_ev + d) != base)) for d in dlist]
                flips[str(r)] = {"mean": float(np.mean(rates)), "draws": rates}
            rows.append(
                {
                    "n_reps": int(nr),
                    "stationary": bool(nr == 1),
                    "bandwidth": float(bw),
                    "best_C": float(best_C),
                    "clean_macro_f1": float(f1_score(y_ev, base, average="macro")),
                    "pred_positive_rate": float(np.mean(base == 1)),
                    "flip_rate": flips,
                }
            )
            log.info(
                "rotation nreps=%d bw=%.2f: clean F1 %.3f  pred-pos %.2f  flip@0.5 %.2f",
                nr,
                bw,
                rows[-1]["clean_macro_f1"],
                rows[-1]["pred_positive_rate"],
                flips[top]["mean"],
            )

    per_reps = {}
    for nr in reps:
        rs = [r for r in rows if r["n_reps"] == nr]
        acc = max(rs, key=lambda r: r["clean_macro_f1"])
        mx = max(r["flip_rate"][top]["mean"] for r in rs)
        ratio = float(acc["flip_rate"][top]["mean"] / mx) if mx > 0 else 0.0
        per_reps[str(nr)] = {
            "stationary": bool(nr == 1),
            "accuracy_optimal_bandwidth": acc["bandwidth"],
            "clean_macro_f1_at_optimum": acc["clean_macro_f1"],
            "flip_at_accuracy_optimum": acc["flip_rate"][top]["mean"],
            "max_flip": mx,
            "ratio": ratio,
            "flip_profile": [r["flip_rate"][top]["mean"] for r in rs],
        }
    decisive = per_reps["1"]["ratio"]
    verdict = {
        "per_reps": per_reps,
        "decisive_ratio_nreps1": decisive,
        # the pre-registered three-way call, computed rather than read by eye
        "outcome": (
            "stationarity_supported"
            if decisive < 0.75
            else ("killed" if decisive >= 0.90 else "indeterminate")
        ),
        "reference_ratios": {"ry_x1_P17a": 0.51, "hrz_x2_P17b": 0.959, "classical_rbf_P16a": 0.43},
        "numerical_gate": gate,
    }
    log.info(
        "rotation verdict: nreps=1 (stationary) ratio %.3f vs nreps=2 0.959 -> %s",
        decisive,
        verdict["outcome"],
    )
    return {"rows": rows, "verdict": verdict}


def bandwidth_fragility_bounded(variant, cfg_frag, gate_slice=64, gate_tol=1e-10):
    """Is the accuracy/fragility INVERSION quantum, or just bounded-and-periodic? (P17a / N6)

    P16a contrasted the fidelity kernel against the classical RBF and found the inversion:
    the RBF's accuracy-optimal width is not its fragility maximum, the fidelity kernel's is.
    But those two kernels differ on TWO axes at once -- quantum vs classical, and decaying
    vs bounded periodic -- so that contrast cannot say which axis carries the effect. This
    is the same confound P8a found in the end-state claim, where the family framing turned
    out to be a length-scale property.

    The disambiguating arm is the product RY fidelity kernel in closed form: bounded and
    periodic like the quantum one, no entanglement, no simulator. Same bandwidth grid, same
    rng seed (so the perturbation set is byte-identical to both existing arms), same C grid
    and CV, same eval points and draws -- only the kernel changes.

    KILL: if this arm behaves like the RBF (optima separated AND flip at the accuracy
    optimum below 0.75x the sweep max), boundedness is not sufficient, and the paper's
    feature-map-scoped wording stands unchanged.
    """
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    _, q = cfg_frag["cell"]
    reg = prepare_regime(variant, "pca", n_components=q, seed=0)
    sub = kernel_subset(reg, 2000, seed=0)
    A_tr, y_tr = reg.angles["train"][sub], reg.y["train"][sub]
    scale = reg.feature_scale
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_frag["eval_subset"], seed=0)
    X_ev, y_ev = reg.X["test"][ev], reg.y["test"][ev]

    rng = np.random.default_rng(0)  # SAME seed as both existing arms -> same perturbations
    deltas = {
        r: [rng.uniform(-1.0, 1.0, X_ev.shape) * r * scale for _ in range(cfg_frag["n_draws"])]
        for r in cfg_frag["radii_sigma"]
    }

    def cv_c(K_tr, C):
        scores = []
        for t, v in StratifiedKFold(3, shuffle=True, random_state=0).split(K_tr, y_tr):
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_tr[np.ix_(t, t)], y_tr[t])
            scores.append(f1_score(y_tr[v], clf.predict(K_tr[np.ix_(v, t)]), average="macro"))
        return float(np.mean(scores))

    rows, gate = [], []
    for bw in cfg_frag["bandwidths"]:
        # NUMERICAL GATE (study protocol §2): the closed form must reproduce the shipped
        # simulator, else this arm is a lookalike rather than the unentangled kernel.
        # Run at float64/complex128 on BOTH sides so the gate measures the identity and
        # not the pipeline's storage precision: angles are stored float32, which alone
        # separates the paths by 8.7e-9 (bw 0.1) to 2.9e-7 (bw 3.0), while at float64
        # they agree to ~3e-15. The SWEEP below deliberately keeps the stored float32
        # angles, because every other arm of this comparison uses them.
        g = A_tr[:gate_slice].astype(np.float64)
        dev = float(
            np.abs(
                cosine_fidelity_kernel(g, g, bw)
                - fidelity_kernel(
                    compute_states(g, "angle_ry", bandwidth=bw, dtype=np.complex128),
                    compute_states(g, "angle_ry", bandwidth=bw, dtype=np.complex128),
                    exact=True,
                )
            ).max()
        )
        if not dev < gate_tol:
            raise RuntimeError(
                f"closed-form kernel disagrees with the simulator at bandwidth {bw}: "
                f"max|dev| {dev:.3e} >= {gate_tol:.0e}"
            )
        gate.append({"bandwidth": float(bw), "max_abs_dev": dev})

        K_tr = cosine_fidelity_kernel(A_tr, A_tr, bw)
        best_C = max(cfg_frag["svm_C_grid"], key=lambda C: cv_c(K_tr, C))
        clf = SVC(kernel="precomputed", C=best_C, class_weight="balanced").fit(K_tr, y_tr)

        def predict(X, _clf=clf, _bw=bw):
            A = reg.angle_scaler.transform(X)
            return _clf.predict(cosine_fidelity_kernel(A, A_tr, _bw))

        base = predict(X_ev)
        flips = {}
        for r, dlist in deltas.items():
            rates = [float(np.mean(predict(X_ev + d) != base)) for d in dlist]
            flips[str(r)] = {"mean": float(np.mean(rates)), "draws": rates}
        rows.append(
            {
                "cell": f"cosine_q{q}_{variant}",
                "bandwidth": float(bw),
                "best_C": float(best_C),
                "clean_macro_f1": float(f1_score(y_ev, base, average="macro")),
                "pred_positive_rate": float(np.mean(base == 1)),
                "true_positive_rate": float(np.mean(y_ev == 1)),
                "flip_rate": flips,
            }
        )
        log.info(
            "bounded fragility bw=%.2f: clean F1 %.3f  pred-pos %.2f  flip@0.5 %.2f  (gate %.1e)",
            bw,
            rows[-1]["clean_macro_f1"],
            rows[-1]["pred_positive_rate"],
            flips[str(cfg_frag["radii_sigma"][-1])]["mean"],
            dev,
        )

    top = str(cfg_frag["radii_sigma"][-1])
    acc_arg = max(rows, key=lambda r: r["clean_macro_f1"])
    flip_arg = max(rows, key=lambda r: r["flip_rate"][top]["mean"])
    flip_at_acc = acc_arg["flip_rate"][top]["mean"]
    max_flip = flip_arg["flip_rate"][top]["mean"]
    co_peaks = bool(acc_arg["bandwidth"] == flip_arg["bandwidth"])
    verdict = {
        "accuracy_optimal_bandwidth": acc_arg["bandwidth"],
        "fragility_optimal_bandwidth": flip_arg["bandwidth"],
        "co_peaks": co_peaks,
        "flip_at_accuracy_optimum": flip_at_acc,
        "max_flip": max_flip,
        "flip_ratio_at_optimum": float(flip_at_acc / max_flip) if max_flip > 0 else 0.0,
        # the pre-registered three-way call, computed here so no reading happens by eye
        "outcome": (
            "co_peak"
            if co_peaks
            else ("killed" if flip_at_acc < 0.75 * max_flip else "indeterminate")
        ),
        "numerical_gate": gate,
    }
    log.info(
        "bounded fragility verdict: accuracy-optimal bw %.2f, fragility-optimal %.2f, "
        "co-peaks=%s, flip at optimum %.2f vs max %.2f -> %s",
        verdict["accuracy_optimal_bandwidth"],
        verdict["fragility_optimal_bandwidth"],
        co_peaks,
        flip_at_acc,
        max_flip,
        verdict["outcome"],
    )
    return {"rows": rows, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument(
        "--tuned-full",
        default="results/runs/merged_accuracy_triple_full",
        help="run dir supplying tuned three-class full-feature params for --only confusion",
    )
    ap.add_argument(
        "--tuned-pca",
        default="results/runs/merged_accuracy_triple_pca8",
        help="run dir supplying tuned three-class PCA params for --only confusion",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=(
            "subset of {shots,sweep,depth,data_eff,noise_model,cap_cost,barren,entangle,"
            "geometry,relabel,bandwidth,fragility,fragility_classical,rotation,mechanism,"
            "fragility_bounded,confusion,dequant}"
        ),
    )
    args = ap.parse_args()

    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    cfg_c = load_yaml(REPO_ROOT / "configs" / "classical.yaml")
    seeds = args.seeds or cfg_c["evaluation"]["seeds"]
    cap = cfg_q["qkernel"]["subset_cap"]
    do = set(args.only) if args.only else {"shots", "sweep", "depth", "data_eff"}
    # noise_model is opt-in: density matrices are 4**d per sample

    reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
    with RunDir(f"ablations_{args.variant}_q{args.qubits}", config=vars(args), seeds=seeds) as run:
        if "shots" in do:
            run.write_json(
                "shot_study.json",
                shot_study(reg, "angle_ry", args.qubits, cap, cfg_q["ablations"]["shots"], seeds),
            )
        if "sweep" in do:
            run.write_json(
                "feature_map_qubit_sweep.json",
                feature_map_qubit_sweep(
                    args.variant, cfg_q["qkernel"]["encodings"], cfg_q["qkernel"]["qubits"], cap
                ),
            )
        if "depth" in do:
            run.write_json(
                "vqc_depth_sweep.json",
                vqc_depth_sweep(reg, args.qubits, cfg_q["vqc"]["depths"], cap, seeds),
            )
        if "noise_model" in do:
            nm = cfg_q["ablations"]["noise_model"]
            run.write_json(
                "noise_model.json",
                noise_model_study(
                    reg,
                    nm["encoding"],
                    args.qubits,
                    nm["subset_cap"],
                    nm["depolarizing_p"],
                    test_cap=nm["test_cap"],
                ),
            )
        if "barren" in do:
            run.write_json(
                "barren_plateau_scan.json",
                barren_plateau_scan(args.variant, cfg_q["qkernel"]["qubits"], depth=6),
            )
        if "cap_cost" in do:
            run.write_json(
                "kernel_cap_cost.json",
                kernel_cap_cost(args.variant, cfg_q["qkernel"]["qubits"], cap, seeds),
            )
        if "geometry" in do:
            run.write_json(
                "kernel_geometry.json",
                kernel_geometry(args.variant, cfg_q["ablations"]["kernel_geometry"]),
            )
        if "relabel" in do:
            run.write_json(
                "relabel_control.json",
                relabel_control(
                    args.variant, cfg_q["qkernel"]["qubits"], cfg_q["ablations"]["relabel_control"]
                ),
            )
        if "bandwidth" in do:
            run.write_json(
                "bandwidth_curve.json",
                bandwidth_curve(args.variant, cfg_q["ablations"]["bandwidth_curve"]),
            )
        if "fragility" in do:
            run.write_json(
                "bandwidth_fragility.json",
                bandwidth_fragility(args.variant, cfg_q["ablations"]["bandwidth_fragility"]),
            )
        if "dequant" in do:
            run.write_json(
                "dequantization_check.json",
                dequantization_check(
                    args.variant, cfg_q["qkernel"]["qubits"], cfg_q["ablations"]["dequantization"]
                ),
            )
        if "fragility_classical" in do:
            run.write_json(
                "bandwidth_fragility_classical.json",
                bandwidth_fragility_classical(
                    args.variant, cfg_q["ablations"]["bandwidth_fragility"]
                ),
            )
        if "mechanism" in do:
            run.write_json(
                "margin_gradient_mechanism.json",
                margin_gradient_mechanism(args.variant, cfg_q["ablations"]["bandwidth_fragility"]),
            )
        if "rotation" in do:
            run.write_json(
                "rotation_structure_isolation.json",
                rotation_structure_isolation(
                    args.variant, cfg_q["ablations"]["bandwidth_fragility"]
                ),
            )
        if "entangle" in do:
            run.write_json(
                "entanglement_isolation.json",
                entanglement_isolation(args.variant, cfg_q["ablations"]["bandwidth_fragility"]),
            )
        if "fragility_bounded" in do:
            run.write_json(
                "bandwidth_fragility_bounded.json",
                bandwidth_fragility_bounded(
                    args.variant, cfg_q["ablations"]["bandwidth_fragility"]
                ),
            )
        if "confusion" in do:
            run.write_json(
                "confusion_triple_full.json",
                confusion_three_class("triple", "full", None, args.tuned_full, seeds, None),
            )
            run.write_json(
                "confusion_triple_pca8.json",
                confusion_three_class("triple", "pca", args.qubits, args.tuned_pca, seeds, cap),
            )
        if "data_eff" in do:
            run.write_json(
                "data_efficiency.json",
                data_efficiency(
                    args.variant, args.qubits, cfg_c["data_efficiency"]["sizes"], seeds, cap
                ),
            )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
