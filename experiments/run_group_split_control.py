"""Group-split control: the paired-protocol exhibit behind the split-protocol instance.

THE INSTANCE. Every published split in this study is stratified over POOLED rows. The
MSU/ORNL data ships as 15 source files whose rows are near-duplicates within a file, so a
row-level split puts near-copies on both sides of the train/test boundary. P12a (decision
log 2026-08-15T22:06Z) measured the cost: the full-feature ceiling falls ~0.31 macro-F1
when whole source files are held out. This run is the shipped, artifact-backed form of that
probe, with the two defects of the probe corrected (decision log 2026-08-16T02:35Z,
condition 2):

  * the quantum arm runs at the TUNED hyperparameters the paper reports
    (`results/adv_best_params_binary_q8.json`), not at bandwidth 1.0 / C 1.0;
  * every number lands in ONE run artifact — the probe's tuned-level and matched-cap
    figures were quoted in the log but never archived.

Three arms per split, identical transforms (only the split indices change):
  LEVEL    — full features, tuned LightGBM (Optuna study `cls_lgbm_binary_full`).
  MATCHED  — PCA-8, tuned XGBoost vs tuned ZZ fidelity-kernel SVM, BOTH trained on the
             identical 2,000-row stratified subset (the paper's matched contract).
  AUDIT    — stratified-random floor for the split, and the nearest-neighbour leakage
             audit in standardized full-feature space (the mechanism number).

Splits: the shipped row-level baseline, then N whole-file 60/20/20 partitions drawn from
`np.random.default_rng(0)` — the same stream as the probe, so partition sets match it.
Groups are "whole source files held out": whether files are physical events is NOT
established (P12a non-establishment (i)) and nothing here claims it.

Run:  python -m experiments.run_group_split_control
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

from qgridbench.data.preprocess import load_processed
from qgridbench.data.validate import load_variant_frames
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import AngleScaler, build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)
VARIANT = "binary"

# Row-level references this run must land near before its numbers are quotable
# (logged as deltas, not gates): MODEL_CARD headline means over 10 seeds.
PUBLISHED_FULL_LGBM = 0.9056
PUBLISHED_QSVM_ZZ_PCA8 = 0.5644


def reconstruct_groups():
    """Group id per PROCESSED row, following preprocess_variant's steps exactly."""
    cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    label_col = cfg["schema"]["label_column"]
    frames = load_variant_frames(VARIANT)
    gid_raw = np.concatenate([np.full(len(f), i, dtype=np.int64) for i, f in enumerate(frames)])
    pooled = pd.concat(frames, ignore_index=True)

    label_map = cfg["labels"].get(VARIANT)
    y = pooled[label_col].map(label_map).to_numpy(dtype=np.int64)
    X = pooled.drop(columns=[label_col]).to_numpy(dtype=np.float64)
    X[np.isinf(X)] = np.nan

    df = pd.DataFrame(X)
    df["__y"] = y
    keep = (~df.duplicated()).to_numpy()
    return X[keep], y[keep], gid_raw[keep]


def group_splits(gid, y, rng):
    """60/20/20 by GROUP, not by row: whole source files are held out."""
    groups = np.unique(gid)
    rng.shuffle(groups)
    n = len(groups)
    tr_g, va_g = groups[: int(0.6 * n)], groups[int(0.6 * n) : int(0.8 * n)]
    te_g = groups[int(0.8 * n) :]
    idx = np.arange(len(y))
    return {
        "train": idx[np.isin(gid, tr_g)],
        "val": idx[np.isin(gid, va_g)],
        "test": idx[np.isin(gid, te_g)],
    }, (sorted(tr_g.tolist()), sorted(va_g.tolist()), sorted(te_g.tolist()))


def tuned_params():
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    db = (REPO_ROOT / "results" / "optuna_cls_lgbm_binary_full.db").as_posix()
    study = optuna.load_study(study_name="cls_lgbm_binary_full", storage=f"sqlite:///{db}")
    return study.best_params, best


def score_level(lgbm_params, splits, X, y, seed=0):
    pipe = build_preprocessor("full", n_components=None, seed=seed)
    Z = fit_transform_splits(pipe, X, splits)
    set_all_seeds(seed)
    est = build("lgbm", lgbm_params, seed)
    fit_model("lgbm", est, Z["train"], y[splits["train"]])
    return float(f1_score(y[splits["test"]], est.predict(Z["test"]), average="macro"))


def score_matched(best, splits, X, y, seed=0):
    """PCA-8; both arms train on the identical 2,000-row stratified subset."""
    pipe = build_preprocessor("pca", n_components=8, seed=seed)
    Z = fit_transform_splits(pipe, X, splits)
    ytr, yte = y[splits["train"]], y[splits["test"]]
    sub = stratified_cap(Z["train"], ytr, 2000, seed=0)

    set_all_seeds(seed)
    xgb = build("xgb", best["xgb"], seed)
    fit_model("xgb", xgb, Z["train"][sub], ytr[sub])
    cls = float(f1_score(yte, xgb.predict(Z["test"]), average="macro"))

    sc = AngleScaler().fit(Z["train"])  # train-fitted, leakage rule (same as prepare_regime)
    s_tr = compute_states(sc.transform(Z["train"][sub]), "zz", bandwidth=best["qk_bw"])
    qk = SVC(kernel="precomputed", C=best["qk_C"], class_weight="balanced")
    qk.fit(fidelity_kernel(s_tr, s_tr), ytr[sub])
    s_te = compute_states(sc.transform(Z["test"]), "zz", bandwidth=best["qk_bw"])
    q = float(f1_score(yte, qk.predict(fidelity_kernel(s_te, s_tr)), average="macro"))
    return {"classical": cls, "quantum": q, "contrast": cls - q}


def audit(splits, X, y, seed=0):
    """Stratified-random floor + nearest-neighbour leakage in standardized full-feature space."""
    floor = trivial_baselines(y[splits["train"]], y[splits["test"]])
    pipe = build_preprocessor("full", n_components=None, seed=seed)
    Z = fit_transform_splits(pipe, X, splits)
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Z["train"])
    d = nn.kneighbors(Z["test"])[0].ravel()
    return {
        "floor_macro_f1": floor["baseline_stratified_random"]["aggregate"]["macro_f1"]["mean"],
        "nn_frac_within_0.01": float((d < 0.01).mean()),
        "nn_frac_within_0.1": float((d < 0.1).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=5)
    args = ap.parse_args()

    lgbm_params, best = tuned_params()
    X, y, gid = reconstruct_groups()
    Xp, yp, splits_row = load_processed(VARIANT)
    assert X.shape == Xp.shape, f"group reconstruction misaligned: {X.shape} vs {Xp.shape}"
    assert np.array_equal(y, yp), "label vector diverged from the shipped artifact"
    log.info("groups reconstructed and ALIGNED: %d rows, %d groups", len(gid), len(np.unique(gid)))

    cfg = {
        "lgbm_source": "optuna study cls_lgbm_binary_full (best_params)",
        "matched_source": "results/adv_best_params_binary_q8.json",
        "qk_bw": best["qk_bw"],
        "qk_C": best["qk_C"],
        "n_partitions": args.partitions,
    }
    with RunDir(f"group_split_control_{VARIANT}", config=cfg) as run:
        rows = []

        def evaluate(split_name, partition, splits, held=None):
            r = {
                "split": split_name,
                "partition": partition,
                "test_groups": held,
                "level_full_lgbm": score_level(lgbm_params, splits, X, y),
                **{f"matched_{k}": v for k, v in score_matched(best, splits, X, y).items()},
                **audit(splits, X, y),
            }
            rows.append(r)
            run.write_json("results_partial.json", {"rows": rows})
            log.info(
                "%s p%s | level %.4f | matched cls %.4f q %.4f contrast %+.4f | floor %.4f "
                "| nn<0.1 %.1f%%",
                split_name,
                partition,
                r["level_full_lgbm"],
                r["matched_classical"],
                r["matched_quantum"],
                r["matched_contrast"],
                r["floor_macro_f1"],
                100 * r["nn_frac_within_0.1"],
            )
            return r

        row_ref = evaluate("row (shipped)", None, {k: np.asarray(v) for k, v in splits_row.items()})
        log.info(
            "row-level reproduction deltas vs published: level %+0.4f (vs %.4f) | "
            "qsvm %+0.4f (vs %.4f)",
            row_ref["level_full_lgbm"] - PUBLISHED_FULL_LGBM,
            PUBLISHED_FULL_LGBM,
            row_ref["matched_quantum"] - PUBLISHED_QSVM_ZZ_PCA8,
            PUBLISHED_QSVM_ZZ_PCA8,
        )

        rng = np.random.default_rng(0)  # same stream as the probe: partition sets match it
        for p in range(args.partitions):
            sp, held = group_splits(gid, y, rng)
            evaluate("group", p, sp, held[2])

        grp = [r for r in rows if r["split"] == "group"]

        def agg(key):
            v = [r[key] for r in grp]
            return float(np.mean(v)), float(np.std(v))

        level_m, level_s = agg("level_full_lgbm")
        cls_m, cls_s = agg("matched_classical")
        q_m, q_s = agg("matched_quantum")
        con_m, con_s = agg("matched_contrast")
        floor_m, _ = agg("floor_macro_f1")
        verdict = {
            "level_row": row_ref["level_full_lgbm"],
            "level_group_mean": level_m,
            "level_group_std": level_s,
            "level_drop": row_ref["level_full_lgbm"] - level_m,
            "matched_classical_row": row_ref["matched_classical"],
            "matched_quantum_row": row_ref["matched_quantum"],
            "matched_contrast_row": row_ref["matched_contrast"],
            "matched_classical_group_mean": cls_m,
            "matched_classical_group_std": cls_s,
            "matched_quantum_group_mean": q_m,
            "matched_quantum_group_std": q_s,
            "matched_contrast_group_mean": con_m,
            "matched_contrast_group_std": con_s,
            "contrast_shift": row_ref["matched_contrast"] - con_m,
            "floor_group_mean": floor_m,
            "row_repro_delta_level": row_ref["level_full_lgbm"] - PUBLISHED_FULL_LGBM,
            "row_repro_delta_qsvm": row_ref["matched_quantum"] - PUBLISHED_QSVM_ZZ_PCA8,
        }
        run.write_json("results.json", {"rows": rows, "verdict": verdict})
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
