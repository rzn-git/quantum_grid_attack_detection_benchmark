"""D1 — does file DIVERSITY close the cross-file gap, or is it the seventh lever that does not?

PRE-REGISTRATION (BACKLOG `Validate` row D1, winds review 2026-08-22T21:11Z; frozen before the
run). Whole-file macro-F1 tops out near 0.595 against 0.905 within-file — a gap of roughly 0.31 —
and every lever measured so far closes only about 0.10 of it: dimensionality +0.013, raw features
+0.028, training volume +0.028 (all W1), choice of 8-dimensional projection +0.000 (N2), being in
the target's own domain -0.021 (W3). Every one of those arms trains on ALL NINE training files, and
in W1's full-volume level arm diversity is CONFOUNDED with volume because more rows and more files
arrived together. So the question a practitioner actually asks — "will collecting more source files
fix this?" — is the one lever never tested.

METHOD. Fixed held-out test files per partition. Vary k, the number of distinct TRAINING files, over
{1, 2, 3, 5, 7, 9}. Draw R random file-subsets at each k so one unlucky file cannot drive the curve
(at k = 9 there is only one subset, so R collapses to 1 there by construction).

  ARM A  diversity ISOLATED: total training rows held FIXED at 2,000 stratified, drawn from the
         pooled rows of the k files. This is the arm that answers the question.
  ARM B  diversity + volume: rows-per-file held fixed at 2000/9, so total rows scale with k.
         B - A at each k is the volume contribution, which decomposes W1's confound.

DECISION RULE, fixed now, on ARM A, XGBoost, full features:
  k=9 beats k=1 by > 0.05 excess-over-floor  -> file diversity IS a lever; collecting more source
                                                files is a real mitigation and the paper says so
  k=9 beats k=1 by <= 0.05                   -> file diversity is NOT a lever; the
                                                engineering-boundary map closes with all seven
                                                entries measured and the paper states plainly that
                                                collecting more source files does not close the gap
A null is the more useful outcome here and is worth as much as a positive.

CONTROLS.
  positive (blocking)  ARM A at k = 9, full features, XGBoost is EXACTLY W1's full-regime design
                       point — all nine training files, `stratified_cap(..., 2000, seed=0)`, same
                       preprocessor — and must reproduce +0.0693 within 0.005
  negative (blocking)  the SAME curve scored on held-out ROWS FROM THE TRAINING FILES instead of
                       held-out files. There k should buy little and the level should sit near
                       0.90; if that curve rises as steeply as the cross-file one, k is proxying
                       for generic sample diversity and the cross-file reading is void
  reported             achieved class balance per arm — one file may not supply a balanced
                       stratified draw at k = 1, and that must be visible rather than silently
                       dropped

The quantum arm runs at the ENDPOINTS ONLY (k = 1 and k = 9, PCA-8, Arm A): the kernel is O(N^2)
circuit evaluations and the endpoints carry the decision, so the interior k values would cost an
hour to confirm a curve the classical arms already trace.

Run:  python -m experiments.run_file_diversity_curve
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from experiments.run_group_split_control import group_splits, reconstruct_groups
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import AngleScaler, build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
CAP = 2000
K_VALUES = [1, 2, 3, 5, 7, 9]
SUBSET_DRAWS = 3
DIM = 8
W1_FULL_XGB_EXCESS = 0.0693  # decision log 2026-08-22T15:14Z, h1.best_excess_per_regime.full
W1_TOL = 0.005
LEVER_BAR = 0.05
QUANTUM_K = [1, 9]


def full_regime_params(model: str) -> dict:
    """Hyperparameters tuned on the FULL feature set — the regime the classical arms run in."""
    name = f"cls_{model}_{VARIANT}_full"
    db = (REPO_ROOT / "results" / f"optuna_{name}.db").as_posix()
    return optuna.load_study(study_name=name, storage=f"sqlite:///{db}").best_params


def pca8_params() -> dict:
    """PCA-8 tuned XGBoost, for the endpoint arm that carries the quantum comparison."""
    db = (REPO_ROOT / "results" / "optuna_cls_xgb_binary_pca8.db").as_posix()
    return optuna.load_study(
        study_name="cls_xgb_binary_pca8", storage=f"sqlite:///{db}"
    ).best_params


def file_subsets(files: list[int], k: int, draws: int, rng: np.random.Generator) -> list[list[int]]:
    """`draws` distinct random k-subsets; at k = len(files) there is exactly one."""
    if k >= len(files):
        return [sorted(files)]
    seen, out = set(), []
    for _ in range(draws * 20):
        s = tuple(sorted(rng.choice(files, size=k, replace=False).tolist()))
        if s not in seen:
            seen.add(s)
            out.append(list(s))
        if len(out) == draws:
            break
    return out


def class_balance(y: np.ndarray) -> dict:
    vals, counts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts, strict=True)}


def fit_score(model: str, params: dict, Ztr, ytr, targets: dict, seed: int) -> dict:
    """One fit, scored on every named target split (cross-file test and within-file rows)."""
    set_all_seeds(seed)
    m = build(model, params, seed)
    fit_model(model, m, Ztr, ytr)
    return {
        name: float(f1_score(yt, m.predict(Zt), average="macro"))
        for name, (Zt, yt) in targets.items()
    }


def qsvm_score(Ztr, ytr, Zte, yte, best, seed: int) -> float:
    """The tuned ZZ fidelity kernel on an 8-dimensional view — endpoints only."""
    sc = AngleScaler().fit(Ztr)
    s_tr = compute_states(sc.transform(Ztr), "zz", bandwidth=best["qk_bw"])
    set_all_seeds(seed)
    qk = SVC(kernel="precomputed", C=best["qk_C"], class_weight="balanced")
    qk.fit(fidelity_kernel(s_tr, s_tr), ytr)
    s_te = compute_states(sc.transform(Zte), "zz", bandwidth=best["qk_bw"])
    return float(f1_score(yte, qk.predict(fidelity_kernel(s_te, s_tr)), average="macro"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--no-quantum", action="store_true")
    ap.add_argument(
        "--models",
        nargs="*",
        default=["xgb", "lgbm"],
        help="classical arms. Added 2026-08-23 so the MLP can be swept on the SAME curve — "
        "D2c's validity arm suggested the diversity lever may run OPPOSITE for a neural net. "
        "The default reproduces the original run exactly.",
    )
    args = ap.parse_args()

    params = {m: full_regime_params(m) for m in args.models}
    qbest = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    xgb_pca8 = pca8_params()

    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # shipped stream — partitions must match every prior run
    parts = []
    for p in range(args.partitions):
        sp, held = group_splits(gid, y, rng_part)
        parts.append((p, sp, held[2]))

    cfg = {
        "variant": VARIANT,
        "k_values": K_VALUES,
        "subset_draws": SUBSET_DRAWS,
        "cap": CAP,
        "arms": {
            "A": "total rows FIXED at 2000 — diversity isolated",
            "B": "rows-per-file fixed at 2000/9 — volume scales with k",
        },
        "partitions": args.partitions,
        "seeds": args.seeds,
        "lever_bar": LEVER_BAR,
        "w1_reference_full_xgb": W1_FULL_XGB_EXCESS,
        "quantum_k": [] if args.no_quantum else QUANTUM_K,
        "note": "held-out FILES are fixed per partition; only the training files vary",
    }

    with RunDir(f"file_diversity_curve_{VARIANT}", config=cfg, seeds=args.seeds) as run:
        rows: list[dict] = []
        for p, sp, held in parts:
            pipe = build_preprocessor("full", n_components=None, seed=0)
            Z = fit_transform_splits(pipe, X, sp)
            ytr_all, yte = y[sp["train"]], y[sp["test"]]
            gtr = gid[sp["train"]]
            files = sorted(np.unique(gtr).tolist())
            floor = trivial_baselines(ytr_all, yte)["baseline_stratified_random"]["aggregate"][
                "macro_f1"
            ]["mean"]

            # PCA-8 view, fit on the FULL training split once — the endpoints' quantum regime.
            # Fitting it per k would confound the projection with the diversity being measured.
            pca = PCA(n_components=DIM, random_state=0).fit(Z["train"])

            rng_sub = np.random.default_rng(9000 + p)
            for k in K_VALUES:
                for si, subset in enumerate(file_subsets(files, k, SUBSET_DRAWS, rng_sub)):
                    m_sub = np.isin(gtr, subset)
                    Zsub, ysub = Z["train"][m_sub], ytr_all[m_sub]

                    # NEGATIVE CONTROL target: held-out ROWS from the TRAINING files. Carved from
                    # the same subset so it is genuinely within-file, and disjoint from the rows
                    # the model trains on.
                    holdout = np.random.default_rng(100 + si).permutation(len(ysub))
                    n_hold = max(200, int(0.2 * len(ysub)))
                    hold_idx, pool_idx = holdout[:n_hold], holdout[n_hold:]
                    Zpool, ypool = Zsub[pool_idx], ysub[pool_idx]
                    Zhold, yhold = Zsub[hold_idx], ysub[hold_idx]

                    for arm, n_rows in (("A", CAP), ("B", int(round(CAP / len(files))) * k)):
                        n = min(n_rows, len(ypool))
                        idx = stratified_cap(Zpool, ypool, n, seed=0)
                        Ztr, ytr = Zpool[idx], ypool[idx]
                        targets = {"crossfile": (Z["test"], yte), "withinfile": (Zhold, yhold)}
                        for model in args.models:
                            for s in args.seeds:
                                sc = fit_score(model, params[model], Ztr, ytr, targets, s)
                                rows.append(
                                    {
                                        "partition": p,
                                        "k": k,
                                        "subset": si,
                                        "files": subset,
                                        "arm": arm,
                                        "model": model,
                                        "seed": s,
                                        "n_train": int(len(ytr)),
                                        "floor": floor,
                                        "class_balance": class_balance(ytr),
                                        "crossfile_f1": sc["crossfile"],
                                        "withinfile_f1": sc["withinfile"],
                                        "crossfile_excess": sc["crossfile"] - floor,
                                    }
                                )
                        # quantum endpoints: Arm A only, PCA-8, k in {1, 9}
                        if arm == "A" and k in QUANTUM_K and not args.no_quantum:
                            Ptr, Pte = pca.transform(Ztr), pca.transform(Z["test"])
                            for s in args.seeds:
                                q = qsvm_score(Ptr, ytr, Pte, yte, qbest, s)
                                set_all_seeds(s)
                                xm = build("xgb", xgb_pca8, s)
                                fit_model("xgb", xm, Ptr, ytr)
                                rows.append(
                                    {
                                        "partition": p,
                                        "k": k,
                                        "subset": si,
                                        "files": subset,
                                        "arm": "A_pca8",
                                        "model": "qsvm",
                                        "seed": s,
                                        "n_train": int(len(ytr)),
                                        "floor": floor,
                                        "crossfile_f1": q,
                                        "crossfile_excess": q - floor,
                                        "xgb_pca8_f1": float(
                                            f1_score(yte, xm.predict(Pte), average="macro")
                                        ),
                                    }
                                )
                    log.info(
                        "p%d k=%d subset%d %s  A/xgb %+0.4f | floor %.4f",
                        p,
                        k,
                        si,
                        subset,
                        float(
                            np.mean(
                                [
                                    r["crossfile_excess"]
                                    for r in rows
                                    if r["partition"] == p
                                    and r["k"] == k
                                    and r["subset"] == si
                                    and r["arm"] == "A"
                                    and r["model"] == "xgb"
                                ]
                            )
                        ),
                        floor,
                    )
                    run.write_json("results_partial.json", {"rows": rows})

        # ---- aggregate ------------------------------------------------------------------
        def mean_over(pred, key="crossfile_excess"):
            v = [r[key] for r in rows if pred(r)]
            return float(np.mean(v)) if v else float("nan")

        curve = {}
        for arm in ("A", "B"):
            curve[arm] = {}
            for model in args.models:
                curve[arm][model] = {
                    str(k): {
                        "crossfile_excess": mean_over(
                            lambda r, k=k, a=arm, m=model: (
                                r["k"] == k and r["arm"] == a and r["model"] == m
                            )
                        ),
                        "withinfile_f1": mean_over(
                            lambda r, k=k, a=arm, m=model: (
                                r["k"] == k and r["arm"] == a and r["model"] == m
                            ),
                            "withinfile_f1",
                        ),
                        "n_train": mean_over(
                            lambda r, k=k, a=arm, m=model: (
                                r["k"] == k and r["arm"] == a and r["model"] == m
                            ),
                            "n_train",
                        ),
                    }
                    for k in K_VALUES
                }
        quantum = {
            str(k): {
                "qsvm_excess": mean_over(lambda r, k=k: r["k"] == k and r["model"] == "qsvm"),
                "xgb_pca8_f1": mean_over(
                    lambda r, k=k: r["k"] == k and r["model"] == "qsvm", "xgb_pca8_f1"
                ),
            }
            for k in QUANTUM_K
        }

        # ---- controls ---------------------------------------------------------------------
        pos = curve["A"]["xgb"]["9"]["crossfile_excess"]
        neg_1 = curve["A"]["xgb"]["1"]["withinfile_f1"]
        neg_9 = curve["A"]["xgb"]["9"]["withinfile_f1"]
        cross_gain = pos - curve["A"]["xgb"]["1"]["crossfile_excess"]
        controls = {
            "positive_k9_full_xgb_excess": pos,
            "positive_delta_vs_w1": float(pos - W1_FULL_XGB_EXCESS),
            "positive_pass": bool(abs(pos - W1_FULL_XGB_EXCESS) <= W1_TOL),
            "negative_withinfile_k1": neg_1,
            "negative_withinfile_k9": neg_9,
            "negative_withinfile_gain": float(neg_9 - neg_1),
            "negative_crossfile_gain": float(cross_gain),
            "negative_pass": bool((neg_9 - neg_1) < cross_gain or cross_gain <= 0),
            "negative_note": (
                "if the within-file curve rises as steeply as the cross-file one, k is proxying "
                "for generic sample diversity and the cross-file reading is void"
            ),
        }
        controls["pass"] = bool(controls["positive_pass"] and controls["negative_pass"])
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        # ---- decision rule ------------------------------------------------------------------
        verdict = {
            "controls": controls,
            "curve": curve,
            "quantum_endpoints": quantum,
            "arm_a_xgb_k1": curve["A"]["xgb"]["1"]["crossfile_excess"],
            "arm_a_xgb_k9": curve["A"]["xgb"]["9"]["crossfile_excess"],
            "diversity_gain": float(cross_gain),
            # per-model k=1 -> k=9 gain on Arm A (fixed row budget), so a lever that runs
            # in OPPOSITE directions for different learners is visible rather than hidden
            # behind the XGBoost figure the decision rule reads.
            "diversity_gain_by_model": {
                m: float(
                    curve["A"][m]["9"]["crossfile_excess"] - curve["A"][m]["1"]["crossfile_excess"]
                )
                for m in args.models
            },
            "volume_contribution_at_k9": float(
                curve["B"]["xgb"]["9"]["crossfile_excess"]
                - curve["A"]["xgb"]["9"]["crossfile_excess"]
            ),
            "lever_bar": LEVER_BAR,
            "branch": (
                "diversity_IS_a_lever" if cross_gain > LEVER_BAR else "diversity_is_NOT_a_lever"
            ),
            "rule": (
                f"Arm A, XGBoost, full features: k=9 minus k=1 > {LEVER_BAR} -> diversity is a "
                "lever; else the engineering-boundary map closes"
            ),
            "rows": rows,
        }
        run.write_json("results.json", verdict)
        run.metrics = {"diversity_gain": cross_gain, "k9_excess": pos}
        log.info(
            "VERDICT %s | k=1 %+0.4f -> k=9 %+0.4f (gain %+0.4f, bar %0.2f)",
            verdict["branch"],
            verdict["arm_a_xgb_k1"],
            verdict["arm_a_xgb_k9"],
            cross_gain,
            LEVER_BAR,
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
