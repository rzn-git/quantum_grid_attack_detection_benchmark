"""P27c: re-tune BOTH matched arms (PCA-8 XGBoost, ZZ fidelity kernel q8) under whole-file splits.

THE GAP THIS CLOSES (follow-up brief Task 1, optional extension). The archived matched-regime
group-split arms (control run 20260816T023620Z_group_split_control_binary) reuse
hyperparameters tuned under the row-level
protocol — the same confound P27 measured for LightGBM, applied to the matched-dimensionality
contrast (+0.0154 ± 0.0149, sign-unstable). This runner re-tunes each arm per whole-file
partition at the paper's equal budget and asks whether the contrast moves.

Pre-registered 2026-09-01T02:05Z (P27c): per-arm recovery read on P27's bars (< 0.02 not
tuning-limited; >= 0.05 scoped); contrast shift vs the archived +0.0154 read on the same bars;
asymmetric recovery between arms is its own finding. Positive controls: the row-tuned configs
must reproduce the archived per-partition matched scores within 0.002.

Run:  python -m experiments.run_whole_file_matched_retune [--partitions 5] [--trials 50]
"""

from __future__ import annotations

import argparse

import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from experiments.run_group_split_control import group_splits, reconstruct_groups, tuned_params
from experiments.run_whole_file_retune import EXPECTED_TEST_GROUPS, assert_file_disjoint
from qgridbench.features.reduce import AngleScaler, build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap, suggest_params
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

VARIANT = "binary"
ENCODING = "zz"
CAP = 2000  # the matched contract's shared training cap
TUNING_SEEDS = [0, 1, 2]  # XGBoost search seeds (classical protocol)
# Archived matched-regime results on the identical partitions
# (run 20260816T023620Z_group_split_control_binary).
ARCHIVED_MATCHED = {
    0: {"cls": 0.5347356767693852, "q": 0.5070432072675171},
    1: {"cls": 0.5134124992678517, "q": 0.5252651897704205},
    2: {"cls": 0.5393199606824701, "q": 0.5092700931536661},
    3: {"cls": 0.5028339873218358, "q": 0.4872450613402072},
    4: {"cls": 0.5258846303907314, "q": 0.5104880117084735},
}
ARCHIVED_CONTRAST_MEAN = 0.0154


def fit_xgb_and_score(params, Z, y, splits, sub, score_split, seed):
    set_all_seeds(seed)
    est = build("xgb", params, seed)
    fit_model("xgb", est, Z["train"][sub], y[splits["train"]][sub])
    return float(f1_score(y[splits[score_split]], est.predict(Z[score_split]), average="macro"))


def qk_score(bw, C, A_tr_sub, y_tr_sub, A_eval, y_eval):
    s_tr = compute_states(A_tr_sub, ENCODING, bandwidth=bw)
    clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
    clf.fit(fidelity_kernel(s_tr, s_tr), y_tr_sub)
    K_eval = fidelity_kernel(compute_states(A_eval, ENCODING, bandwidth=bw), s_tr)
    return float(f1_score(y_eval, clf.predict(K_eval), average="macro"))


def make_study(name):
    db = (REPO_ROOT / "results" / f"{name}.db").as_posix()
    return optuna.create_study(
        direction="maximize",
        study_name=name,
        storage=f"sqlite:///{db}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
    )


def top_up(study, objective, n_trials):
    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--partitions", type=int, default=5)
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

    _, best = tuned_params()  # row-tuned matched params: best["xgb"], best["qk_bw"], best["qk_C"]
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    bw_cfg, c_cfg = cfg_q["qkernel"]["bandwidth"], cfg_q["qkernel"]["svm_C"]
    va_cap = cfg_q["qkernel"]["tuning_val_subset"]
    X, y, gid = reconstruct_groups()

    cfg = {
        "variant": VARIANT,
        "encoding": ENCODING,
        "cap": CAP,
        "n_partitions": args.partitions,
        "n_trials": args.trials,
        "tuning_seeds_xgb": TUNING_SEEDS,
        "row_tuned_source": "results/adv_best_params_binary_q8.json",
        "preregistration": "decision log 2026-09-01T02:05Z (P27c)",
    }
    with RunDir(f"whole_file_matched_retune_{VARIANT}_q8", config=cfg) as run:
        rng = np.random.default_rng(0)  # same stream as the archived control -> same partitions
        rows = []
        for p_idx in range(args.partitions):
            splits, held = group_splits(gid, y, rng)
            if held[2] != EXPECTED_TEST_GROUPS[p_idx]:
                raise SystemExit(
                    f"PARTITION STREAM DIVERGED: p{p_idx} test files {held[2]} vs "
                    f"expected {EXPECTED_TEST_GROUPS[p_idx]}"
                )
            assert_file_disjoint(splits, gid, p_idx)
            pipe = build_preprocessor("pca", n_components=8, seed=0)
            Z = fit_transform_splits(pipe, X, splits)
            ytr, yva, yte = y[splits["train"]], y[splits["val"]], y[splits["test"]]
            sub = stratified_cap(Z["train"], ytr, CAP, seed=0)

            sc = AngleScaler().fit(Z["train"])  # train-fitted, leakage rule (as score_matched)
            A_tr_sub = sc.transform(Z["train"][sub])
            A_va = sc.transform(Z["val"])
            va_sub = stratified_cap(A_va, yva, va_cap, seed=0)
            A_va_s, y_va_s = A_va[va_sub], yva[va_sub]
            A_te = sc.transform(Z["test"])

            # positive controls: row-tuned configs must reproduce the archived matched scores
            ctrl_cls = fit_xgb_and_score(best["xgb"], Z, y, splits, sub, "test", seed=0)
            ctrl_q = qk_score(best["qk_bw"], best["qk_C"], A_tr_sub, ytr[sub], A_te, yte)
            d_cls = ctrl_cls - ARCHIVED_MATCHED[p_idx]["cls"]
            d_q = ctrl_q - ARCHIVED_MATCHED[p_idx]["q"]
            log.info(
                "p%d controls | xgb %.4f (delta %+.4f) | qk %.4f (delta %+.4f)",
                p_idx,
                ctrl_cls,
                d_cls,
                ctrl_q,
                d_q,
            )
            if abs(d_cls) > 0.002 or abs(d_q) > 0.002:
                raise SystemExit(
                    f"POSITIVE CONTROL FAILED: p{p_idx} xgb delta {d_cls:+.4f}, "
                    f"qk delta {d_q:+.4f} — run discarded"
                )

            def xgb_objective(trial):
                params = suggest_params("xgb", trial)
                return float(
                    np.mean(
                        [
                            fit_xgb_and_score(params, Z, y, splits, sub, "val", s)
                            for s in TUNING_SEEDS
                        ]
                    )
                )

            def qk_objective(trial):
                bw = trial.suggest_float(
                    "bandwidth", bw_cfg["low"], bw_cfg["high"], log=bw_cfg["log"]
                )
                C = trial.suggest_float("C", c_cfg["low"], c_cfg["high"], log=c_cfg["log"])
                return qk_score(bw, C, A_tr_sub, ytr[sub], A_va_s, y_va_s)

            xgb_study = make_study(f"group_matched_retune_{VARIANT}_p{p_idx}_xgb")
            top_up(xgb_study, xgb_objective, args.trials)
            qk_study = make_study(f"group_matched_retune_{VARIANT}_p{p_idx}_qk")
            top_up(qk_study, qk_objective, args.trials)

            retuned_cls = fit_xgb_and_score(xgb_study.best_params, Z, y, splits, sub, "test", 0)
            retuned_q = qk_score(
                qk_study.best_params["bandwidth"],
                qk_study.best_params["C"],
                A_tr_sub,
                ytr[sub],
                A_te,
                yte,
            )
            rec = {
                "partition": p_idx,
                "test_groups": held[2],
                "row_tuned_cls": ctrl_cls,
                "row_tuned_q": ctrl_q,
                "row_tuned_contrast": ctrl_cls - ctrl_q,
                "retuned_cls": retuned_cls,
                "retuned_q": retuned_q,
                "retuned_contrast": retuned_cls - retuned_q,
                "recovery_cls": retuned_cls - ctrl_cls,
                "recovery_q": retuned_q - ctrl_q,
                "xgb_best_params": xgb_study.best_params,
                "xgb_best_val": xgb_study.best_value,
                "qk_best_params": qk_study.best_params,
                "qk_best_val": qk_study.best_value,
            }
            rows.append(rec)
            run.write_json("results_partial.json", {"rows": rows})
            log.info(
                "p%d | cls %.4f->%.4f (%+.4f) | qk %.4f->%.4f (%+.4f) | contrast %+.4f->%+.4f",
                p_idx,
                ctrl_cls,
                retuned_cls,
                rec["recovery_cls"],
                ctrl_q,
                retuned_q,
                rec["recovery_q"],
                rec["row_tuned_contrast"],
                rec["retuned_contrast"],
            )

        def agg(key):
            v = np.array([r[key] for r in rows])
            return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

        rc_m, rc_s = agg("recovery_cls")
        rq_m, rq_s = agg("recovery_q")
        con_m, con_s = agg("retuned_contrast")
        summary = {
            "recovery_cls_mean": rc_m,
            "recovery_cls_std": rc_s,
            "recovery_q_mean": rq_m,
            "recovery_q_std": rq_s,
            "retuned_contrast_mean": con_m,
            "retuned_contrast_std": con_s,
            "retuned_contrast_per_partition": [r["retuned_contrast"] for r in rows],
            "archived_contrast_mean": ARCHIVED_CONTRAST_MEAN,
            "contrast_shift": con_m - ARCHIVED_CONTRAST_MEAN,
            "sign_flips": int(sum(1 for r in rows if r["retuned_contrast"] < 0)),
        }
        shift = abs(summary["contrast_shift"])
        summary["verdict"] = (
            "H1 — |contrast shift| < 0.02: the matched contrast is not tuning-limited"
            if shift < 0.02
            else "KILL — |contrast shift| >= 0.05: the matched-contrast sentence is scoped"
            if shift >= 0.05
            else "INTERMEDIATE — report as measured, scope proportionally"
        )
        run.metrics = summary
        run.write_json("results.json", {"rows": rows, "summary": summary})
        log.info(
            "VERDICT: %s (cls recovery %+.4f±%.4f, qk recovery %+.4f±%.4f, "
            "contrast %+.4f±%.4f vs archived %+.4f)",
            summary["verdict"],
            rc_m,
            rc_s,
            rq_m,
            rq_s,
            con_m,
            con_s,
            ARCHIVED_CONTRAST_MEAN,
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
