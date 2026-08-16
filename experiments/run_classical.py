"""Phase 2: classical baselines — Optuna tuning (seeds {0,1,2}) then 10-seed eval.

Protocol (CLAUDE.md section 5):
  - equal budget: 50 Optuna trials per model, objective = mean val macro-F1 over
    tuning seeds
  - winning config retrained + evaluated over all 10 seeds
  - test split touched once per final config
  - metrics + wall time + inference latency recorded per seed
Run:  python -m experiments.run_classical --variant binary --regime full
      python -m experiments.run_classical --variant binary --regime pca --pca 8
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
from sklearn.metrics import f1_score

from qgridbench.eval.metrics import compute_all
from qgridbench.models.classical.zoo import (
    MODEL_NAMES,
    build,
    fit_model,
    predict_proba_timed,
    stratified_cap,
    suggest_params,
)
from qgridbench.protocol import aggregate_seeds, prepare_regime
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


def tune_model(name, reg, tuning_seeds, n_trials, subset_cap, storage, study_name):
    """Return best params by mean val macro-F1 over tuning seeds."""
    Xtr, ytr, Xva, yva = reg.X["train"], reg.y["train"], reg.X["val"], reg.y["val"]
    cap_idx = stratified_cap(Xtr, ytr, subset_cap, seed=0) if subset_cap else np.arange(len(ytr))
    Xtr_c, ytr_c = Xtr[cap_idx], ytr[cap_idx]

    def objective(trial):
        params = suggest_params(name, trial)
        scores = []
        for s in tuning_seeds:
            set_all_seeds(s)
            est = build(name, params, s)
            fit_model(name, est, Xtr_c, ytr_c)
            scores.append(f1_score(yva, est.predict(Xva), average="macro"))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    # top up to the equal budget, never past it (resume-safe)
    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    return study.best_params, study.best_value


def eval_model(name, best_params, reg, seeds, subset_cap):
    Xtr, ytr = reg.X["train"], reg.y["train"]
    Xte, yte = reg.X["test"], reg.y["test"]
    cap_idx = stratified_cap(Xtr, ytr, subset_cap, seed=0) if subset_cap else np.arange(len(ytr))
    per_seed, train_times, latencies = [], [], []
    for s in seeds:
        set_all_seeds(s)
        est = build(name, best_params, s)
        train_times.append(fit_model(name, est, Xtr[cap_idx], ytr[cap_idx]))
        proba, latency = predict_proba_timed(est, Xte)
        latencies.append(latency)
        per_seed.append(compute_all(yte, proba))
    agg = aggregate_seeds(per_seed)
    agg["train_time_s"] = {
        "mean": float(np.mean(train_times)),
        "std": float(np.std(train_times, ddof=1)),
    }
    agg["infer_latency_s"] = {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies, ddof=1)),
    }
    return {"best_params": best_params, "per_seed": per_seed, "aggregate": agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--regime", choices=["full", "pca"], default="full")
    ap.add_argument("--pca", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=MODEL_NAMES)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument(
        "--subset-cap",
        type=int,
        default=None,
        help="cap train subset (PCA-regime fairness with quantum)",
    )
    args = ap.parse_args()

    cfg = load_yaml(REPO_ROOT / "configs" / "classical.yaml")
    n_trials = args.n_trials or cfg["tuning"]["n_trials"]
    tuning_seeds = cfg["tuning"]["tuning_seeds"]
    seeds = args.seeds or cfg["evaluation"]["seeds"]
    storage_dir = REPO_ROOT / cfg["tuning"]["optuna_storage_dir"]
    svm_cap = cfg["svm_train_cap"]

    reg = prepare_regime(args.variant, args.regime, n_components=args.pca, seed=0)
    tag = f"{args.variant}_{args.regime}{args.pca or ''}"

    with RunDir(
        f"classical_{tag}", config={**vars(args), "n_trials": n_trials}, seeds=seeds
    ) as run:
        results = {}
        for name in args.models:
            study_name = f"cls_{name}_{tag}"
            # one sqlite file per study: parallel model processes never contend
            storage = f"sqlite:///{(storage_dir / f'optuna_{study_name}.db').as_posix()}"
            # kernel-SVM cap applies wherever no tighter cap is set (decision log)
            cap = args.subset_cap
            if name == "rbf_svm":
                cap = min(cap, svm_cap) if cap else svm_cap
            best_params, best_val = tune_model(
                name, reg, tuning_seeds, n_trials, cap, storage, study_name
            )
            res = eval_model(name, best_params, reg, seeds, cap)
            res["best_val_macro_f1"] = best_val
            results[name] = res
            f1 = res["aggregate"]["macro_f1"]
            log.info(
                "%s | val=%.4f | test macro-F1 %.4f +/- %.4f", name, best_val, f1["mean"], f1["std"]
            )
            mlflow_log(
                f"classical_{name}_{tag}",
                {
                    "model": name,
                    "variant": args.variant,
                    "regime": args.regime,
                    "pca": args.pca,
                    **best_params,
                },
                {f"test_{k}": v["mean"] for k, v in res["aggregate"].items()},
                tags={"phase": "classical"},
            )
        run.write_json("results.json", results)
        run.metrics = {n: r["aggregate"]["macro_f1"]["mean"] for n, r in results.items()}
        print(json.dumps(run.metrics, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
