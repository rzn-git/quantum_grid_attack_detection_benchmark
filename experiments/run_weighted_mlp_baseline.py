"""E-M4b: the class-weighted MLP as the MLP baseline (supervisor review M4b).

Re-tunes the MLP with balanced sample weights under the equal 50-trial budget at PCA-8 (the
Table II protocol), evaluates it over the ten test seeds, and measures it under the Table III
protocol at 0.5 sigma: the published MLP-surrogate PGD transferred to it (the shared protocol,
replayed seed by seed from the same attack RNG stream) and its own white-box PGD. It then
recomputes the 12-pair quantum-versus-classical Holm family with the weighted MLP in place of
the unweighted one. Pre-registration: decision log 2026-11-16T05:15Z, section E-M4b.

Run: python -m experiments.run_weighted_mlp_baseline [--threads N] [--smoke]

Arms (Table II protocol, 2000-row cap, ten seeds):
  U   published hyperparameters, unweighted fit   (anchor: must reproduce Table II's MLP row)
  W0  published hyperparameters, balanced fit     (the weighting effect alone)
  W1  re-tuned under the balanced fit             (the candidate baseline row)
"""

from __future__ import annotations

import argparse
import os
import sys


def _threads_from_argv(default: int = 4) -> int:
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return default


_N = str(_threads_from_argv())
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = _N

import json  # noqa: E402

import numpy as np  # noqa: E402
import optuna  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from qgridbench.attacks.evasion import pgd  # noqa: E402
from qgridbench.attacks.gradients import GradProvider  # noqa: E402
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.eval.metrics import compute_all  # noqa: E402
from qgridbench.eval.stats import holm_bonferroni, paired_wilcoxon  # noqa: E402
from qgridbench.models.classical.zoo import (  # noqa: E402
    build,
    fit_model,
    stratified_cap,
    suggest_params,
)
from qgridbench.protocol import aggregate_seeds, kernel_subset, prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import set_all_seeds, spawn_rng  # noqa: E402

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

TABLE2_RUN = "20260813T064540Z_classical_binary_pca8"  # Table II source (make_report rq1)
TABLE3_RUN = "20260813T183448Z_adversarial_binary_q8"  # Table III source (robustness_stats)
EPS = 0.5
ANCHOR_TOL_MEAN = 0.005
QUANTUM = ("qsvm", "vqc")


def _f1(est, X, y):
    return float(f1_score(y, est.predict_proba(X).argmax(1), average="macro"))


def tune_balanced(reg, tuning_seeds, n_trials, cap, storage, study_name):
    """Equal-budget Optuna search for the MLP under the balanced fit (val macro-F1)."""
    Xtr, ytr, Xva, yva = reg.X["train"], reg.y["train"], reg.X["val"], reg.y["val"]
    idx = stratified_cap(Xtr, ytr, cap, seed=0)

    def objective(trial):
        params = suggest_params("mlp", trial)
        scores = []
        for s in tuning_seeds:
            set_all_seeds(s)
            est = build("mlp", params, s)
            fit_model("mlp", est, Xtr[idx], ytr[idx], balance_via_sample_weight=True)
            scores.append(f1_score(yva, est.predict(Xva), average="macro"))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    return study.best_params, float(study.best_value)


def eval_arm(reg, params, seeds, cap, balanced):
    """Table II evaluation: fit on the capped train subset, score the full test split."""
    Xtr, ytr, Xte, yte = reg.X["train"], reg.y["train"], reg.X["test"], reg.y["test"]
    idx = stratified_cap(Xtr, ytr, cap, seed=0)
    per_seed, pos_rate = [], []
    for s in seeds:
        set_all_seeds(s)
        est = build("mlp", params, s)
        fit_model("mlp", est, Xtr[idx], ytr[idx], balance_via_sample_weight=balanced)
        proba = est.predict_proba(Xte)
        per_seed.append(compute_all(yte, proba))
        pos_rate.append(float((proba.argmax(1) == 1).mean()))
    return {
        "params": params,
        "balanced": balanced,
        "per_seed": per_seed,
        "aggregate": aggregate_seeds(per_seed),
        "positive_rate_mean": float(np.mean(pos_rate)),
    }


def stratified_floor(n_eval: int) -> float:
    """The Table III stratified-random floor on the same evaluation subset (2,000 draws)."""
    _X, _y, sp = load_processed("binary")
    y_tr, y_te = _y[sp["train"]], _y[sp["test"]]
    y_ev = y_te[stratified_cap(y_te, y_te, n_eval, seed=0)]
    cls, cnt = np.unique(y_tr, return_counts=True)
    rng = np.random.default_rng(0)
    return float(
        np.mean(
            [
                f1_score(y_ev, rng.choice(cls, size=len(y_ev), p=cnt / cnt.sum()), average="macro")
                for _ in range(2000)
            ]
        )
    )


def robustness(reg, published, params_u, params_w, seeds, cfg_adv, sub_cap):
    """Table III protocol, replayed: per seed, the same attack RNG stream as run_adversarial.

    In run_adversarial each PGD call draws one uniform random start of shape X.shape from the
    per-seed stream `spawn_rng(seed, "adv")`, in this order: for every epsilon, the MLP's and
    then the VQC's white-box PGD; then the MLP surrogate's PGD for every epsilon. Replaying the
    MLP calls and drawing the VQC's start without its model reproduces the published MLP
    white-box set and the shared transfer set exactly, so the weighted MLP sees the very
    perturbations every other family in Table III saw.
    """
    eps_grid = cfg_adv["evasion"]["epsilons"]
    pgd_cfg = cfg_adv["evasion"]["pgd"]
    ev_idx = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    Xte, yte = reg.X["test"][ev_idx], reg.y["test"][ev_idx]
    scale = reg.feature_scale
    sub = kernel_subset(reg, sub_cap, seed=0)
    Xtr, ytr = reg.X["train"][sub], reg.y["train"][sub]

    def run_pgd(gp, rng):
        return pgd(
            gp,
            Xte,
            yte,
            EPS,
            scale,
            n_steps=pgd_cfg["n_steps"],
            step_frac=pgd_cfg["step_frac"],
            rng=rng,
        )

    rows = []
    for s in seeds:
        set_all_seeds(s)
        mlp_u = build("mlp", params_u, s)
        fit_model("mlp", mlp_u, Xtr, ytr)
        gp_u = GradProvider(mlp_u, "mlp")
        rng = spawn_rng(s, "adv")
        wb_u_set, shared_set = None, None
        for e in eps_grid:  # loop 1: sources mlp, vqc (white-box, each attacks itself)
            if e == EPS:
                wb_u_set = run_pgd(gp_u, rng)
            else:
                pgd(
                    gp_u,
                    Xte,
                    yte,
                    e,
                    scale,
                    n_steps=pgd_cfg["n_steps"],
                    step_frac=pgd_cfg["step_frac"],
                    rng=rng,
                )
            rng.uniform(-1.0, 1.0, size=Xte.shape)  # the VQC's random start
        for e in eps_grid:  # loop 2: the MLP surrogate for the non-differentiable families
            if e == EPS:
                shared_set = run_pgd(gp_u, rng)
            else:
                rng.uniform(-1.0, 1.0, size=Xte.shape)

        pub = published[s]["evasion"]
        rec_u = {
            "clean": _f1(mlp_u, Xte, yte),
            "whitebox": _f1(mlp_u, wb_u_set, yte),
            "published_clean": pub["clean"]["mlp"],
            "published_whitebox": pub["curves"]["mlp"]["pgd"][str(EPS)],
        }

        set_all_seeds(s)
        mlp_w = build("mlp", params_w, s)
        fit_model("mlp", mlp_w, Xtr, ytr, balance_via_sample_weight=True)
        wb_w_set = run_pgd(GradProvider(mlp_w, "mlp"), spawn_rng(s, "m4b"))
        rec_w = {
            "clean": _f1(mlp_w, Xte, yte),
            "shared": _f1(mlp_w, shared_set, yte),
            "whitebox": _f1(mlp_w, wb_w_set, yte),
            "positive_rate": float((mlp_w.predict_proba(Xte).argmax(1) == 1).mean()),
        }
        rows.append({"seed": s, "unweighted": rec_u, "weighted": rec_w})
        log.info(
            "seed %d | U clean %.4f (pub %.4f) wb %.4f (pub %.4f) | W clean %.4f shared %.4f "
            "wb %.4f",
            s,
            rec_u["clean"],
            rec_u["published_clean"],
            rec_u["whitebox"],
            rec_u["published_whitebox"],
            rec_w["clean"],
            rec_w["shared"],
            rec_w["whitebox"],
        )
    return rows


def holm_family(published, weighted_shared):
    """The robustness_stats family (quantum x classical at 0.5 sigma), MLP replaced."""
    fams = list(published[0]["evasion"]["curves"])
    series = {
        f: np.array([p["evasion"]["curves"][f]["pgd"][str(EPS)] for p in published]) for f in fams
    }
    series["mlp_weighted"] = np.asarray(weighted_shared)
    classical = [f for f in fams if f not in QUANTUM and f != "mlp"] + ["mlp_weighted"]
    raw, detail = {}, {}
    for q in QUANTUM:
        for c in classical:
            st = paired_wilcoxon(series[q], series[c])
            raw[f"{q}_vs_{c}"], detail[f"{q}_vs_{c}"] = st["p_value"], st
    corrected = holm_bonferroni(raw)
    for k, st in detail.items():
        st["p_holm"] = corrected[k]
        st["significant_holm_0.05"] = bool(corrected[k] < 0.05)
    return {
        "pairs": detail,
        "n_comparisons": len(raw),
        "n_significant_holm": sum(v["significant_holm_0.05"] for v in detail.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="2 seeds, 3 trials, separate study")
    args = ap.parse_args()

    cfg_c = load_yaml(REPO_ROOT / "configs" / "classical.yaml")
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    seeds = [0, 1] if args.smoke else cfg_c["evaluation"]["seeds"]
    n_trials = 3 if args.smoke else cfg_c["tuning"]["n_trials"]
    tuning_seeds = cfg_c["tuning"]["tuning_seeds"]
    cap = 2000  # Table II's classical PCA-regime cap (run config of TABLE2_RUN)
    study_name = "cls_mlp_balanced_binary_pca8" + ("_smoke" if args.smoke else "")
    storage_dir = REPO_ROOT / cfg_c["tuning"]["optuna_storage_dir"]
    storage = f"sqlite:///{(storage_dir / f'optuna_{study_name}.db').as_posix()}"

    t2 = json.loads((REPO_ROOT / "results" / "runs" / TABLE2_RUN / "results.json").read_text())
    params_u = t2["mlp"]["best_params"]
    pub_t2 = t2["mlp"]["aggregate"]["macro_f1"]["mean"]
    t3 = json.loads((REPO_ROOT / "results" / "runs" / TABLE3_RUN / "results.json").read_text())
    published = t3["per_seed"]
    params_adv = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    if params_adv["mlp"] != params_u:
        raise ValueError("Table II and Table III MLP hyperparameters differ; the anchor is void")

    reg = prepare_regime("binary", "pca", n_components=8, seed=0)
    config = {**vars(args), "seeds": seeds, "n_trials": n_trials, "cap": cap, "eps": EPS}
    with RunDir("weighted_mlp_baseline_binary_pca8", config=config, seeds=seeds) as run:
        out = {"table2_run": TABLE2_RUN, "table3_run": TABLE3_RUN}

        out["U"] = eval_arm(reg, params_u, seeds, cap, balanced=False)
        u_mean = out["U"]["aggregate"]["macro_f1"]["mean"]
        log.info("U (anchor) macro-F1 %.4f against published %.4f", u_mean, pub_t2)
        if not args.smoke and abs(u_mean - pub_t2) > ANCHOR_TOL_MEAN:
            raise AssertionError(f"Table II anchor failed: {u_mean:.4f} vs {pub_t2:.4f}")
        run.write_json("results_partial.json", out)

        out["W0"] = eval_arm(reg, params_u, seeds, cap, balanced=True)
        best_w, best_val = tune_balanced(reg, tuning_seeds, n_trials, cap, storage, study_name)
        out["W1"] = eval_arm(reg, best_w, seeds, cap, balanced=True)
        out["W1"]["best_val_macro_f1"] = best_val
        run.write_json("results_partial.json", out)

        prior = float(reg.y["test"].mean())
        pos_u, pos_w = out["U"]["positive_rate_mean"], out["W1"]["positive_rate_mean"]
        out["control_positive_rate"] = {
            "test_prior": prior,
            "U": pos_u,
            "W1": pos_w,
            "pass": abs(pos_w - prior) < abs(pos_u - prior),
        }
        if not out["control_positive_rate"]["pass"]:
            raise AssertionError(f"positive control failed: {out['control_positive_rate']}")

        rows = robustness(
            reg, published, params_u, best_w, seeds, cfg_adv, cfg_q["qkernel"]["subset_cap"]
        )
        out["robustness_rows"] = rows
        d_clean = np.mean(
            [r["unweighted"]["clean"] - r["unweighted"]["published_clean"] for r in rows]
        )
        d_wb = np.mean(
            [r["unweighted"]["whitebox"] - r["unweighted"]["published_whitebox"] for r in rows]
        )
        out["control_table3_anchor"] = {
            "mean_delta_clean": float(d_clean),
            "mean_delta_whitebox": float(d_wb),
            "pass": bool(abs(d_clean) <= ANCHOR_TOL_MEAN and abs(d_wb) <= ANCHOR_TOL_MEAN),
        }
        if not out["control_table3_anchor"]["pass"]:
            raise AssertionError(f"Table III anchor failed: {out['control_table3_anchor']}")

        floor = stratified_floor(cfg_adv["eval_subset_adv"])
        w = {k: np.array([r["weighted"][k] for r in rows]) for k in ("clean", "shared", "whitebox")}
        out["weighted_table3_row"] = {
            "floor": floor,
            "clean_mean": float(w["clean"].mean()),
            "clean_std": float(w["clean"].std(ddof=1)) if len(rows) > 1 else 0.0,
            "shared_mean": float(w["shared"].mean()),
            "shared_std": float(w["shared"].std(ddof=1)) if len(rows) > 1 else 0.0,
            "f1_minus_floor_shared": float(w["shared"].mean() - floor),
            "retention_shared": float(w["shared"].mean() / w["clean"].mean()),
            "whitebox_mean": float(w["whitebox"].mean()),
            "retention_whitebox": float(w["whitebox"].mean() / w["clean"].mean()),
        }
        if not args.smoke:
            out["holm_family_weighted"] = holm_family(published, w["shared"])

        w1 = out["W1"]["aggregate"]["macro_f1"]["mean"]
        t2_floor = 0.501  # Table II stratified-random row
        out["branch"] = (
            "weighted_becomes_baseline" if w1 - t2_floor >= 0.02 else "keep_unweighted_with_note"
        )
        run.write_json("results.json", out)
        run.metrics = {
            "U": u_mean,
            "W0": out["W0"]["aggregate"]["macro_f1"]["mean"],
            "W1": w1,
            "branch": out["branch"],
        }
        print(json.dumps(run.metrics, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
