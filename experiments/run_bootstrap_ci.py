"""P26: paired bootstrap over test points for the matched-dimensionality margins.

THE GAP THIS CLOSES. The quantum-kernel SVM is deterministic given its fixed training
subset, so the paper's matched-dimensionality comparisons rest on margins against the
classical seed spread (methodology, external review item 9). This runner adds the standard
fix: a percentile bootstrap over test points of the PAIRED difference in macro-F1 between
the tuned XGBoost and the tuned best quantum kernel, at every dimensionality and every
XGBoost seed.

Exact resampling, not simulation of it: binary macro-F1 depends only on the contingency
counts, so an i.i.d. resample of test points is a multinomial resample of the 8-cell joint
distribution (label x classical prediction x quantum prediction). B draws are vectorized.

Positive control gating interpretation (pre-registration 2026-08-20T17:45Z): the refit
models must reproduce the archived RQ1 artifacts before any interval is read — the
deterministic QSVM to 1e-9, the XGBoost 10-seed mean within 0.003. Reproduction failure
means the run is discarded, not reinterpreted.

Run:  python -m experiments.run_bootstrap_ci
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from qgridbench.models.classical.zoo import build, fit_model
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
from qgridbench.protocol import kernel_subset, prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
XGB_SEEDS = list(range(10))
B = 10_000
# Best quantum kernel per dimensionality, from the RQ1 artifacts.
BEST_ENCODING = {8: "zz", 12: "zz", 16: "angle_ry"}
# Archived references are READ FROM THE ARTIFACTS at runtime, never transcribed: the
# first run of this script hard-coded them and the positive control killed the run on a
# wrong constant while the refit was bit-exact (decision log 2026-08-20). The artifact
# is the single source of truth; a copied literal is a second one that can drift.


def qkernel_artifact(dim: int) -> dict:
    """Best params + archived macro-F1 for the best encoding at this dimensionality."""
    enc = BEST_ENCODING[dim]
    run = sorted(glob.glob(f"results/runs/*quantum_qkernel_binary_q{dim}"))[-1]
    return json.loads((REPO_ROOT / run / f"qkernel_{enc}_q{dim}.json").read_text())


def archived_references(dim: int) -> tuple[float, float]:
    """Archived QSVM macro-F1 (exact) and XGBoost 10-seed mean, read from the run dirs."""
    qsvm = qkernel_artifact(dim)["aggregate"]["macro_f1"]["mean"]
    run = sorted(glob.glob(f"results/runs/*merged_rq1_binary_pca{dim}"))[-1]
    xgb = json.loads((REPO_ROOT / run / "results.json").read_text())["xgb"]
    return float(qsvm), float(xgb["aggregate"]["macro_f1"]["mean"])


def xgb_params(dim: int) -> dict:
    db = (REPO_ROOT / "results" / f"optuna_cls_xgb_binary_pca{dim}.db").as_posix()
    study = optuna.load_study(study_name=f"cls_xgb_binary_pca{dim}", storage=f"sqlite:///{db}")
    return study.best_params


def macro_f1_from_counts(tp0, fp0, fn0, tp1, fp1, fn1):
    """Vectorized binary macro-F1 from per-class TP/FP/FN counts (arrays over draws)."""
    f1_0 = 2 * tp0 / np.maximum(2 * tp0 + fp0 + fn0, 1)
    f1_1 = 2 * tp1 / np.maximum(2 * tp1 + fp1 + fn1, 1)
    return (f1_0 + f1_1) / 2


def paired_bootstrap_ci(y, pred_c, pred_q, rng):
    """Percentile 95% CI of macro-F1(classical) - macro-F1(quantum), paired over test points."""
    cell = (y * 4 + pred_c * 2 + pred_q).astype(np.int64)  # 8-cell joint
    counts = np.bincount(cell, minlength=8)
    draws = rng.multinomial(len(y), counts / counts.sum(), size=B).astype(np.float64)

    def f1_of(pred_bit):
        # pred = the chosen model's prediction bit inside the cell code
        pred = np.array([(c >> pred_bit) & 1 for c in range(8)])
        lab = np.array([(c >> 2) & 1 for c in range(8)])
        tp1 = draws[:, (lab == 1) & (pred == 1)].sum(axis=1)
        fp1 = draws[:, (lab == 0) & (pred == 1)].sum(axis=1)
        fn1 = draws[:, (lab == 1) & (pred == 0)].sum(axis=1)
        tp0 = draws[:, (lab == 0) & (pred == 0)].sum(axis=1)
        fp0 = fn1
        fn0 = fp1
        return macro_f1_from_counts(tp0, fp0, fn0, tp1, fp1, fn1)

    delta = f1_of(1) - f1_of(0)  # classical bit is 2^1, quantum bit is 2^0
    lo, hi = np.percentile(delta, [2.5, 97.5])
    return float(lo), float(hi), float(delta.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dims", type=int, nargs="*", default=[8, 12, 16])
    args = ap.parse_args()

    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    cap = cfg_q["qkernel"]["subset_cap"]

    cfg = {
        "variant": VARIANT,
        "dims": args.dims,
        "xgb_seeds": XGB_SEEDS,
        "B": B,
        "best_encoding": {str(k): v for k, v in BEST_ENCODING.items()},
        "preregistration": "decision log 2026-08-20T17:45Z (P26)",
    }
    with RunDir(f"bootstrap_ci_{VARIANT}", config=cfg, seeds=XGB_SEEDS) as run:
        out = {}
        all_exclude_zero = True
        for dim in args.dims:
            archived_qsvm, archived_xgb = archived_references(dim)
            art = qkernel_artifact(dim)
            enc, bw, C = (
                BEST_ENCODING[dim],
                art["best_params"]["bandwidth"],
                art["best_params"]["C"],
            )
            reg = prepare_regime(VARIANT, "pca", n_components=dim, seed=0)
            sub = kernel_subset(reg, cap, seed=0)
            yte = reg.y["test"]

            # --- deterministic quantum arm, exactly the run_quantum path -----------
            A_tr = reg.angles["train"][sub]
            s_tr = compute_states(A_tr, enc, bandwidth=bw)
            K_tr = fidelity_kernel(s_tr, s_tr)
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_tr, reg.y["train"][sub])
            A_te = reg.angles["test"]
            K_te = np.empty((len(A_te), len(A_tr)))
            chunk = 2000
            for s in range(0, len(A_te), chunk):
                se = compute_states(A_te[s : s + chunk], enc, bandwidth=bw)
                K_te[s : s + chunk] = fidelity_kernel(se, s_tr)
            pred_q = clf.predict(K_te).astype(np.int64)
            f1_q = float(f1_score(yte, pred_q, average="macro"))
            dq = f1_q - archived_qsvm
            log.info("pca%d %s QSVM refit %.9f (archive delta %+.1e)", dim, enc, f1_q, dq)
            if abs(dq) > 1e-9:
                raise SystemExit(
                    f"POSITIVE CONTROL FAILED: pca{dim} QSVM {f1_q!r} != archived "
                    f"{archived_qsvm!r} — run discarded per pre-registration"
                )

            # --- classical arm, exactly the run_classical eval path ----------------
            xp = xgb_params(dim)
            rng = np.random.default_rng(0)
            cells = []
            xgb_f1 = []
            for s in XGB_SEEDS:
                set_all_seeds(s)
                est = build("xgb", xp, s)
                fit_model("xgb", est, reg.X["train"][sub], reg.y["train"][sub])
                pred_c = est.predict(reg.X["test"]).astype(np.int64)
                f1_c = float(f1_score(yte, pred_c, average="macro"))
                xgb_f1.append(f1_c)
                lo, hi, mean = paired_bootstrap_ci(yte, pred_c, pred_q, rng)
                cells.append(
                    {
                        "seed": s,
                        "xgb_f1": f1_c,
                        "delta_point": f1_c - f1_q,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "delta_boot_mean": mean,
                        "excludes_zero": bool(lo > 0),
                    }
                )
                if lo <= 0:
                    all_exclude_zero = False
                log.info(
                    "pca%d seed %d | xgb %.4f qsvm %.4f | delta %.4f CI [%.4f, %.4f] %s",
                    dim,
                    s,
                    f1_c,
                    f1_q,
                    f1_c - f1_q,
                    lo,
                    hi,
                    "excl-0" if lo > 0 else "CONTAINS 0",
                )

            xgb_mean = float(np.mean(xgb_f1))
            dx = xgb_mean - archived_xgb
            log.info("pca%d xgb 10-seed mean %.4f (archive delta %+.4f)", dim, xgb_mean, dx)
            if abs(dx) > 0.003:
                raise SystemExit(
                    f"POSITIVE CONTROL FAILED: pca{dim} xgb mean {xgb_mean:.4f} vs archived "
                    f"{archived_xgb:.4f} — run discarded per pre-registration"
                )
            out[f"pca{dim}"] = {
                "encoding": enc,
                "qsvm_f1": f1_q,
                "xgb_mean": xgb_mean,
                "cells": cells,
                "ci_lo_min": float(min(c["ci_lo"] for c in cells)),
                "ci_hi_max": float(max(c["ci_hi"] for c in cells)),
                "control_qsvm_delta": dq,
                "control_xgb_delta": dx,
            }
            run.write_json("results_partial.json", out)

        verdict = (
            "H1 CONFIRMED — every one of the 30 paired 95% CIs excludes zero"
            if all_exclude_zero
            else "KILL — at least one CI contains zero; weaken the margin sentence cell by cell"
        )
        run.metrics = {
            "all_cis_exclude_zero": all_exclude_zero,
            **{f"pca{d}_ci_lo_min": out[f"pca{d}"]["ci_lo_min"] for d in args.dims},
        }
        run.write_json("results.json", {"per_dim": out, "verdict": verdict, "B": B})
        log.info("VERDICT: %s", verdict)
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
