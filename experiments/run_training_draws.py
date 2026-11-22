"""Does the classical lead of the matched contest hold when the 2000 training rows are redrawn?

Registered 2026-11-22 (experiment `training_draws`, decision log). The quantum kernel SVM is
deterministic given its training subset, and every published matched-contest score uses one
stratified draw of that subset (seed 0). Neither the seed spread of the classical models nor a
bootstrap over test points measures the draw. This runner redraws the subset (draw 0 is the
published one), refits the six classical models and both QSVMs on each draw with the tuned
parameters of the published run, and scores every fit on the full test split. Nothing is
retuned per draw.

Registered statistic, per dimensionality: d_k = XGBoost's macro-F1 (mean over the model seeds)
minus the published best quantum model's, on draw k.
  REAL             d_k > 0 on at least `min_same_sign` draws, the exact two-sided Wilcoxon
                   signed-rank p is under 0.05, and the mean is at least `margin`
  REVERSED         the mirror image
  NOT_ESTABLISHED  every other outcome
Anchor read first: on draw 0 each QSVM reproduces its published macro-F1.

Run: python -m experiments.run_training_draws --pca 8 [--draws 10] [--model-seeds 0 1 2]
     [--workers 4] [--threads 2] [--draw-ids 0 1 ...]

--draw-ids runs a subset of the draws (the statistic is computed only when every draw of
--draws is present); --workers runs draws side by side, each with --threads BLAS threads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml

log = get_logger(__name__)

VARIANT = "binary"
CLASSICAL = ("logreg", "rbf_svm", "rf", "xgb", "lgbm", "mlp")
ENCODINGS = ("angle_ry", "zz")
# the published best model of each group at each dimensionality (Table "matched contest")
PAIR = {8: ("xgb", "qsvm_zz"), 12: ("xgb", "qsvm_zz"), 16: ("xgb", "qsvm_angle_ry")}
MARGIN = 0.02  # the margin of the gas accuracy rule: a lead under it reads as parity
MIN_SAME_SIGN_SHARE = 0.9
ANCHOR_TOL_QSVM = 1e-6
ANCHOR_TOL_XGB = 0.005  # XGBoost's fit moves with the thread count
TEST_CHUNK = 2000
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def published_path(pca: int):
    return REPO_ROOT / "results" / "runs" / f"merged_rq1_{VARIANT}_pca{pca}" / "results.json"


def lead_statistic(d: np.ndarray, margin: float = MARGIN) -> dict:
    """The registered reading of the per-draw differences (classical minus quantum)."""
    from scipy.stats import wilcoxon

    d = np.asarray(d, dtype=float)
    n = len(d)
    n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
    need = int(np.ceil(MIN_SAME_SIGN_SHARE * n))
    p = float(wilcoxon(d, alternative="two-sided", method="exact").pvalue) if d.any() else 1.0
    mean = float(d.mean())
    if n_pos >= need and p < 0.05 and mean >= margin:
        label = "REAL"
    elif n_neg >= need and p < 0.05 and mean <= -margin:
        label = "REVERSED"
    else:
        label = "NOT_ESTABLISHED"
    return {
        "branch": label,
        "n_draws": n,
        "mean": mean,
        "std": float(d.std(ddof=1)) if n > 1 else 0.0,
        "min": float(d.min()),
        "max": float(d.max()),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "wilcoxon_p_two_sided": p,
        "margin": margin,
        "min_same_sign": need,
    }


_STATE: dict = {}


def _init_worker(pca: int, threads: int) -> None:
    for var in THREAD_VARS:
        os.environ[var] = str(threads)
    from qgridbench.protocol import prepare_regime

    _STATE["reg"] = prepare_regime(VARIANT, "pca", n_components=pca, seed=0)
    _STATE["tuned"] = json.loads(published_path(pca).read_text())
    _STATE["threads"] = threads


def score_draw(draw: int, cap: int, model_seeds: list[int]) -> dict:
    """Every model's test macro-F1 (and class-1 probabilities) on one draw of the subset."""
    from experiments.run_adversarial import _svc_on_kernel, _svc_proba
    from qgridbench.eval.metrics import compute_all
    from qgridbench.models.classical.zoo import build, fit_model, predict_proba_timed
    from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
    from qgridbench.protocol import kernel_subset
    from qgridbench.utils.seeding import set_all_seeds

    reg, tuned, threads = _STATE["reg"], _STATE["tuned"], _STATE["threads"]
    t0 = time.perf_counter()
    sub = kernel_subset(reg, cap, seed=draw)
    Xtr, ytr = reg.X["train"][sub], reg.y["train"][sub]
    Xte, yte = reg.X["test"], reg.y["test"]
    n_classes = len(np.unique(reg.y["train"]))
    scores: dict = {}
    proba_1: dict = {}
    for name in CLASSICAL:
        scores[name] = {}
        for s in model_seeds:
            set_all_seeds(s)
            est = build(name, tuned[name]["best_params"], s)
            if name in ("rf", "xgb", "lgbm"):
                est.set_params(n_jobs=threads)
            fit_model(name, est, Xtr, ytr)
            proba, _ = predict_proba_timed(est, Xte)
            scores[name][str(s)] = compute_all(yte, proba)["macro_f1"]
            proba_1[f"{name}|{s}"] = proba[:, 1].astype(np.float32)
    A_tr, A_te = reg.angles["train"][sub], reg.angles["test"]
    for enc in ENCODINGS:
        best = tuned[f"qsvm_{enc}"]["best_params"]
        states_tr = compute_states(A_tr, enc, bandwidth=best["bandwidth"])
        clf = _svc_on_kernel(n_classes, best["C"]).fit(fidelity_kernel(states_tr, states_tr), ytr)
        K_te = np.empty((len(A_te), len(A_tr)))
        for a in range(0, len(A_te), TEST_CHUNK):
            se = compute_states(A_te[a : a + TEST_CHUNK], enc, bandwidth=best["bandwidth"])
            K_te[a : a + TEST_CHUNK] = fidelity_kernel(se, states_tr)
        proba = _svc_proba(clf, K_te, n_classes)
        scores[f"qsvm_{enc}"] = compute_all(yte, proba)["macro_f1"]
        proba_1[f"qsvm_{enc}|0"] = proba[:, 1].astype(np.float32)
    return {
        "draw": draw,
        "n_rows": int(len(sub)),
        "positive_share": float(ytr.mean()),
        "subset_sha256": hashlib.sha256(np.ascontiguousarray(sub).tobytes()).hexdigest(),
        "scores": scores,
        "proba_1": proba_1,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def anchors(draw0: dict, published: dict, model_seeds: list[int]) -> dict:
    """Draw 0 against the published run: each QSVM exactly, XGBoost within its thread tolerance."""
    out = {}
    for enc in ENCODINGS:
        name = f"qsvm_{enc}"
        pub = published[name]["aggregate"]["macro_f1"]["mean"]
        got = draw0["scores"][name]
        out[name] = {
            "published": pub,
            "draw0": got,
            "abs_diff": abs(got - pub),
            "pass": bool(abs(got - pub) <= ANCHOR_TOL_QSVM),
        }
    pub = float(np.mean([published["xgb"]["per_seed"][s]["macro_f1"] for s in model_seeds]))
    got = float(np.mean([draw0["scores"]["xgb"][str(s)] for s in model_seeds]))
    out["xgb"] = {
        "published": pub,
        "draw0": got,
        "abs_diff": abs(got - pub),
        "pass": bool(abs(got - pub) <= ANCHOR_TOL_XGB),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pca", type=int, required=True, choices=sorted(PAIR))
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--draw-ids", type=int, nargs="*", default=None)
    args = ap.parse_args()
    draw_ids = args.draw_ids if args.draw_ids else list(range(args.draws))
    cap = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")["qkernel"]["subset_cap"]
    pub_path = published_path(args.pca)
    published = json.loads(pub_path.read_text())
    classical, quantum = PAIR[args.pca]

    with RunDir(
        f"training_draws_{VARIANT}_pca{args.pca}",
        config={
            "variant": VARIANT,
            "pca": args.pca,
            "subset_cap": cap,
            "draws": args.draws,
            "draw_ids": draw_ids,
            "model_seeds": args.model_seeds,
            "pair": [classical, quantum],
            "workers": args.workers,
            "threads_per_worker": args.threads,
            "tuned_from": pub_path.relative_to(REPO_ROOT).as_posix(),
            "tuned_from_sha256": hashlib.sha256(pub_path.read_bytes()).hexdigest(),
            "subset_stream": "kernel_subset(reg, cap, seed=draw): sklearn train_test_split",
        },
        seeds=args.model_seeds,
    ) as run:
        from experiments.exports import Exports

        ex = Exports(VARIANT, None)
        per_draw: dict[int, dict] = {}
        probas_by_draw: dict[int, dict] = {}
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(args.pca, args.threads)
        ) as pool:
            futures = {pool.submit(score_draw, k, cap, args.model_seeds): k for k in draw_ids}
            for fut in as_completed(futures):
                rec = fut.result()
                probas = rec.pop("proba_1")
                per_draw[rec["draw"]] = rec
                run.write_json(f"draw_{rec['draw']:02d}.json", rec)
                log.info(
                    "draw %d: xgb %.4f, %s %.4f (%.0f s)",
                    rec["draw"],
                    float(np.mean(list(rec["scores"]["xgb"].values()))),
                    quantum,
                    rec["scores"][quantum],
                    rec["seconds"],
                )
                probas_by_draw[rec["draw"]] = probas

        from qgridbench.protocol import prepare_regime

        yte = prepare_regime(VARIANT, "pca", n_components=args.pca, seed=0).y["test"]
        pos = np.arange(len(yte))
        for k in sorted(per_draw):
            for key, p1 in probas_by_draw[k].items():
                model, seed = key.split("|")
                proba = np.column_stack([1 - p1, p1])
                ex.predictions(int(seed), f"draw{k}", model, proba, yte, pos)
            for name in CLASSICAL:
                for s, v in per_draw[k]["scores"][name].items():
                    ex.metric(int(s), name, f"draw{k}", "macro_f1", v)
            for enc in ENCODINGS:
                ex.metric(
                    0, f"qsvm_{enc}", f"draw{k}", "macro_f1", per_draw[k]["scores"][f"qsvm_{enc}"]
                )
        ex.extra["threads_per_worker"] = args.threads
        ex.extra["tuned_from"] = pub_path.relative_to(REPO_ROOT).as_posix()
        ex.write(run.path)

        out: dict = {"per_draw": [per_draw[k] for k in sorted(per_draw)]}
        if 0 in per_draw:
            out["anchors"] = anchors(per_draw[0], published, args.model_seeds)
        shas = [per_draw[k]["subset_sha256"] for k in sorted(per_draw)]
        out["draws_distinct"] = len(set(shas)) == len(shas)
        complete = sorted(per_draw) == list(range(args.draws))
        if complete:
            cls = np.array(
                [
                    np.mean(list(per_draw[k]["scores"][classical].values()))
                    for k in range(args.draws)
                ]
            )
            qnt = np.array([per_draw[k]["scores"][quantum] for k in range(args.draws)])
            out["registered"] = {
                "pair": [classical, quantum],
                "classical_per_draw": cls.tolist(),
                "quantum_per_draw": qnt.tolist(),
                "difference_per_draw": (cls - qnt).tolist(),
                **lead_statistic(cls - qnt),
            }
            # secondary, selected on the test split: best classical minus best quantum per draw
            best_c = np.array(
                [
                    max(np.mean(list(per_draw[k]["scores"][m].values())) for m in CLASSICAL)
                    for k in range(args.draws)
                ]
            )
            best_q = np.array(
                [
                    max(per_draw[k]["scores"][f"qsvm_{e}"] for e in ENCODINGS)
                    for k in range(args.draws)
                ]
            )
            out["secondary_best_of_group"] = {
                "difference_per_draw": (best_c - best_q).tolist(),
                **lead_statistic(best_c - best_q),
            }
            run.metrics = {
                "lead_mean": out["registered"]["mean"],
                "lead_std": out["registered"]["std"],
                "n_positive": out["registered"]["n_positive"],
            }
            log.info(
                "registered: %s",
                json.dumps({k: v for k, v in out["registered"].items() if "per_draw" not in k}),
            )
        else:
            log.warning(
                "draws %s of %d present: no statistic computed", sorted(per_draw), args.draws
            )
        run.write_json("results.json", out)
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
