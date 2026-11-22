"""R4-22 H4: the split gap on the synthetic FDIA variants (row split against day-grouped folds).

Usage: uv run python -m experiments.run_fdia_split_gap --variants fdia14_binary fdia118_binary \
           --classical-runs results/runs/<stamp>_classical_fdia14_binary_full ... --threads 4

For each variant, Table I's three arms are scored with `run_group_split_control`'s own functions
(imported, not copied): LEVEL, tuned full-feature LightGBM; MATCHED, PCA-8 XGBoost against the
ZZ-QSVM on the identical 2,000-row subset at the variant's tuned parameters; AUDIT, the
stratified-random floor and the nearest-neighbour distance audit. Splits:
  row      the shipped stratified 60/20/20 row split (splits_<variant>.npz)
  day      5 disjoint day folds (E-DF's rule, qgridbench.data.fdia_synth.day_fold_splits):
           every day is a test day exactly once
  episode  5 disjoint episode folds, the same rule on episode ids (days shared, episodes
           held out); the optional third rung, reported with no branch
Mechanism number: the share of test rows whose nearest training row (standardised full
features) comes from the same episode.

Pre-registration: decision log 2026-11-16T12:17Z, "PRE-RUN AMENDMENT (R4-22", H4.
  G = row-split LightGBM macro-F1 - mean over the 5 day folds.
  HYPOTHESIS  G >= 0.05 on every variant;  KILL  G < 0.02 on every variant;  else INTERMEDIATE.
Anchor (blocking): the row-split LEVEL reproduces the variant's run_classical seed-0 LightGBM
test macro-F1 within 1e-6 (same thread count).
"""

import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, required=True)
THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_var] = str(THREADS)

import json  # noqa: E402

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from experiments.exports import Exports  # noqa: E402
from experiments.run_group_split_control import audit, score_level, score_matched  # noqa: E402
from qgridbench.data.fdia_synth import day_fold_splits, fdia_config, load_aux  # noqa: E402
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.features.reduce import (  # noqa: E402
    AngleScaler,  # noqa: E402
    build_preprocessor,
    fit_transform_splits,
)
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap  # noqa: E402
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel  # noqa: E402
from qgridbench.protocol import best_params_path  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger  # noqa: E402
from qgridbench.utils.seeding import set_all_seeds  # noqa: E402

log = get_logger(__name__)

G_HYP, G_KILL = 0.05, 0.02
ANCHOR_TOL = 1e-6
SIDES = ("train", "val", "test")


def branch(gaps: list[float]) -> str:
    if all(g >= G_HYP for g in gaps):
        return "HYPOTHESIS"
    if all(g < G_KILL for g in gaps):
        return "KILL"
    return "INTERMEDIATE"


def episode_ids(aux: dict, per_day: int) -> np.ndarray:
    return aux["day"].astype(np.int64) * per_day + aux["episode"].astype(np.int64)


def same_episode_share(splits: dict, X: np.ndarray, ep: np.ndarray) -> float:
    pipe = build_preprocessor("full", n_components=None, seed=0)
    Z = fit_transform_splits(pipe, X, splits)
    nn = NearestNeighbors(n_neighbors=1).fit(Z["train"])
    j = nn.kneighbors(Z["test"], return_distance=False).ravel()
    return float(np.mean(ep[splits["train"]][j] == ep[splits["test"]]))


def tuned(variant: str) -> tuple[dict, dict]:
    best = json.loads(best_params_path(variant, 8).read_text())
    name = f"cls_lgbm_{variant}_full"
    db = (REPO_ROOT / "results" / f"optuna_{name}.db").as_posix()
    study = optuna.load_study(study_name=name, storage=f"sqlite:///{db}")
    return study.best_params, best


def anchor_value(classical_run: str) -> float:
    path = REPO_ROOT / classical_run
    if json.loads((path / "metrics.json").read_text())["status"] != "ok":
        raise RuntimeError(f"{classical_run} did not finish ok")
    res = json.loads((path / "results.json").read_text())
    return float(res["lgbm"]["per_seed"][0]["macro_f1"])


def arm_probabilities(lgbm_params, best, sp, X, y, seed=0) -> dict[str, np.ndarray]:
    """The three models of score_level / score_matched refitted the same way, returning their
    test probabilities for the exports; check_against_table_i ties them to the scored F1."""
    out = {}
    pipe = build_preprocessor("full", n_components=None, seed=seed)
    Z = fit_transform_splits(pipe, X, sp)
    set_all_seeds(seed)
    est = build("lgbm", lgbm_params, seed)
    fit_model("lgbm", est, Z["train"], y[sp["train"]])
    out["level_full_lgbm"] = est.predict_proba(Z["test"])
    pipe = build_preprocessor("pca", n_components=8, seed=seed)
    Z = fit_transform_splits(pipe, X, sp)
    ytr = y[sp["train"]]
    sub = stratified_cap(Z["train"], ytr, 2000, seed=0)
    set_all_seeds(seed)
    xgb = build("xgb", best["xgb"], seed)
    fit_model("xgb", xgb, Z["train"][sub], ytr[sub])
    out["matched_classical"] = xgb.predict_proba(Z["test"])
    sc = AngleScaler().fit(Z["train"])
    s_tr = compute_states(sc.transform(Z["train"][sub]), "zz", bandwidth=best["qk_bw"])
    qk = SVC(kernel="precomputed", C=best["qk_C"], class_weight="balanced")
    qk.fit(fidelity_kernel(s_tr, s_tr), ytr[sub])
    s_te = compute_states(sc.transform(Z["test"]), "zz", bandwidth=best["qk_bw"])
    p1 = 1.0 / (1.0 + np.exp(-qk.decision_function(fidelity_kernel(s_te, s_tr))))
    out["matched_quantum"] = np.column_stack([1 - p1, p1])  # label = decision > 0
    return out


def check_against_table_i(row: dict, probs: dict, y_test: np.ndarray) -> dict:
    """Blocking: each refitted model's macro-F1 equals the Table I function's exactly."""
    diffs = {}
    for k, p in probs.items():
        f1 = float(f1_score(y_test, p.argmax(1), average="macro"))
        diffs[k] = f1 - row[k]
        if f1 != row[k]:
            raise AssertionError(f"exports refit of {k} gives {f1}, Table I's function {row[k]}")
    return diffs


def score_split(name, fold, sp, X, y, ep, lgbm_params, best) -> tuple[dict, dict]:
    row = {
        "split": name,
        "fold": fold,
        "n_test": int(len(sp["test"])),
        "level_full_lgbm": score_level(lgbm_params, sp, X, y),
        **{f"matched_{k}": v for k, v in score_matched(best, sp, X, y).items()},
        **audit(sp, X, y),
        "nn_same_episode_share": same_episode_share(sp, X, ep),
    }
    probs = arm_probabilities(lgbm_params, best, sp, X, y)
    check_against_table_i(row, probs, y[sp["test"]])
    return row, probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--classical-runs", nargs="+", required=True, help="one per variant, in order")
    ap.add_argument("--threads", type=int, default=THREADS)
    args = ap.parse_args()
    if len(args.variants) != len(args.classical_runs):
        raise ValueError("one --classical-runs entry per variant")
    cfg = fdia_config()
    sp_cfg, per_day = cfg["split"], cfg["episodes"]["per_day"]
    config = {**vars(args), "n_folds": sp_cfg["n_folds"], "fold_seed": sp_cfg["fold_seed"]}
    with RunDir("fdia_split_gap", config=config) as run:
        out = {}
        for variant, crun in zip(args.variants, args.classical_runs, strict=True):
            lgbm_params, best = tuned(variant)
            X, y, splits_row = load_processed(variant)
            X = X.astype(np.float64)
            aux = load_aux(variant)
            ep = episode_ids(aux, per_day)
            exp = Exports(variant, best_params_path(variant, 8).relative_to(REPO_ROOT).as_posix())
            exp.extra = {
                "threads": THREADS,
                "lgbm_study": f"cls_lgbm_{variant}_full",
                "fold_predictions": "predictions_folds.parquet (row_id = row of the "
                "variant's data file; fold splits are not the shipped test split)",
            }
            fold_preds = []

            def record(row, probs, sp):
                cond = row["split"] if row["fold"] is None else f"{row['split']}_fold{row['fold']}"
                for k in ("level_full_lgbm", "matched_classical", "matched_quantum"):
                    exp.metric(0, k, cond, "macro_f1", row[k])
                for k in (
                    "floor_macro_f1",
                    "nn_frac_within_0.01",
                    "nn_frac_within_0.1",
                    "nn_same_episode_share",
                ):
                    exp.metric(0, "audit", cond, k, row[k])
                for k, pr in probs.items():
                    if row["fold"] is None:
                        exp.predictions(0, cond, k, pr, y[sp["test"]], np.arange(len(sp["test"])))
                    else:
                        fold_preds.append(
                            pd.DataFrame(
                                {
                                    "seed": 0,
                                    "condition": cond,
                                    "model": k,
                                    "row_id": sp["test"],
                                    "y_true": y[sp["test"]],
                                    "proba_1": pr[:, 1],
                                    "label": pr.argmax(1),
                                }
                            )
                        )

            row0, probs0 = score_split("row", None, splits_row, X, y, ep, lgbm_params, best)
            record(row0, probs0, splits_row)
            rows = [row0]
            ref = anchor_value(crun)
            anchor = {"ours": rows[0]["level_full_lgbm"], "classical_seed0": ref}
            anchor["passed"] = abs(anchor["ours"] - ref) <= ANCHOR_TOL
            log.info("%s anchor %.10f vs %.10f", variant, anchor["ours"], ref)
            if not anchor["passed"]:
                run.write_json("anchor_failed.json", {variant: anchor})
                raise AssertionError(f"{variant} anchor failed: {anchor}")
            for name, gid in (("day", aux["day"]), ("episode", ep)):
                for k in range(sp_cfg["n_folds"]):
                    sp = day_fold_splits(gid, k, sp_cfg["n_folds"], sp_cfg["fold_seed"])
                    rk, pk = score_split(name, k, sp, X, y, ep, lgbm_params, best)
                    record(rk, pk, sp)
                    rows.append(rk)
                    run.write_json("results_partial.json", {**out, variant: rows})
                    log.info(
                        "%s %s fold %d level %.4f", variant, name, k, rows[-1]["level_full_lgbm"]
                    )
            summ = {}
            for name in ("day", "episode"):
                rr = [r for r in rows if r["split"] == name]
                for key in ("level_full_lgbm", "matched_classical", "matched_quantum"):
                    v = np.array([r[key] for r in rr])
                    summ[f"{name}_{key}"] = {"mean": float(v.mean()), "std": float(v.std(ddof=1))}
                    summ[f"gap_{name}_{key}"] = rows[0][key] - float(v.mean())
            out[variant] = {"anchor": anchor, "rows": rows, "summary": summ}
            exp_dir = run.path / f"exports_{variant}"  # one Exports per variant (its own splits)
            exp_dir.mkdir()
            exp.write(exp_dir)
            pd.concat(fold_preds, ignore_index=True).to_parquet(
                exp_dir / "predictions_folds.parquet"
            )
        gaps = [out[v]["summary"]["gap_day_level_full_lgbm"] for v in args.variants]
        verdict = {"G": dict(zip(args.variants, gaps, strict=True)), "branch": branch(gaps)}
        run.write_json("results.json", {"variants": out, "verdict": verdict})
        run.metrics.update(verdict)
        log.info("H4 %s", json.dumps(verdict))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
