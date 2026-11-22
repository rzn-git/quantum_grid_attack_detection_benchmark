"""NS15: are the gas tree ensembles fragile only on fill-duplicated eval rows? An archive-only
re-score of the published R4-12 gas exports, split by duplicate status.

Pre-registration: decision log 2026-11-19T08:25Z, "PRE-REGISTRATION (round 5", NS15.

A test row is a duplicate if its PCA-8 model input (`prepare_regime("gas_binary", "pca", 8)`)
equals a row of the zoo's 2,000-row training subset (`kernel_subset(seed=0)`, the subset every
FittedZoo trains on) exactly: the definition of the 13:37Z probe. The reconstruction is checked
against the exports: every random-control and Square crafted row lies inside its eps box around
the reconstructed input (|x_adv - x| / scale <= eps).

Exports re-scored (every family; the trees carry the statistic):
  random   RANDOM_RUN: instability = share of rows whose three random draws do not all give
           the same label, and the error rate, per eps.
  hsj      HSJ_RUNS: per-row L-inf distortion in standardized units (run_adversarial's
           measure), per restart; success = adversarial label differs from the true label.
  square   SQUARE_RUN: success = share of clean-correct rows misclassified, per eps.
  perclass PERCLASS_RUN: macro-F1 under the shared and white-box PGD sets at 0.5 sigma.
Statistics per tree: R_h = median non-duplicate distortion / median duplicate distortion
(successful attacks, pooled over restarts and seeds); dS = Square success on duplicate rows
minus on non-duplicate rows at eps 0.1 (pooled over seeds).
Branch: KILL if every tree has R_h <= 2 and |dS| <= 0.05; CONFIRMED if every tree has R_h >= 10
or dS >= 0.10; PARTIAL otherwise. A non-duplicate set under 20 distinct rows is not read.
Fix check (no branch): distinct training rows and label-conflicting duplicate groups under the
median fill and under median fill plus missing indicators, full features and PCA-8.

Run:
    CUDA_VISIBLE_DEVICES=-1 uv run python -m experiments.run_gas_duplicate_rescore
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from qgridbench.data.preprocess import load_processed
from qgridbench.features.reduce import build_preprocessor
from qgridbench.protocol import kernel_subset, prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml

log = get_logger(__name__)

VARIANT = "gas_binary"
RUNS = REPO_ROOT / "results" / "runs"
RANDOM_RUN = RUNS / "20261118T121646Z_adversarial_gas_binary_q8_random"
HSJ_RUNS = [
    RUNS / "20261118T122227Z_probe_hsj_restarts_gas_binary_q8",  # xgb lgbm rbf_svm logreg mlp
    RUNS / "20261118T122357Z_probe_hsj_restarts_gas_binary_q8",  # rf
    RUNS / "20261118T121915Z_probe_hsj_restarts_gas_binary_q8",  # qsvm vqc (reported only)
]
SQUARE_RUN = RUNS / "20261118T122649Z_square_blackbox_gas_binary_q8"
PERCLASS_RUN = RUNS / "20261118T125448Z_evasion_per_class_gas_binary_q8"
TREES = ("rf", "xgb", "lgbm")
STAT_EPS = "0.1"
MIN_ROWS = 20
BOX_TOL = 1e-6
R_KILL, R_CONF = 2.0, 10.0
S_KILL, S_CONF = 0.05, 0.10


def _row_set(X: np.ndarray) -> set:
    return {r.tobytes() for r in np.ascontiguousarray(X)}


def _split_key(key: str) -> tuple[str, str, str]:
    seed, rest = key.split("|", 1)
    cond, source = rest.rsplit("|", 1)
    return seed, cond, source


def dup_flags(reg, sub) -> np.ndarray:
    train = _row_set(reg.X["train"][sub])
    return np.array([r.tobytes() in train for r in np.ascontiguousarray(reg.X["test"])])


def box_check(run, preds: pd.DataFrame, X: np.ndarray, scale: np.ndarray) -> dict:
    """Every crafted row of a bounded attack lies in its eps box around the reconstructed X."""
    z = np.load(run / "crafted_inputs.npz")
    worst, n = 0.0, 0
    for key in z.files:
        seed, cond, source = _split_key(key)
        eps = float(cond.split("@")[1].split("|")[0])
        model = source if source in set(preds["model"]) else None
        sel = preds[(preds["seed"] == int(seed)) & (preds["condition"] == cond.split("|for_")[0])]
        if model is not None:
            sel = sel[sel["model"] == model]
        else:
            sel = sel[sel["model"] == sel["model"].iloc[0]]
        pos = sel["test_pos"].to_numpy()
        Xa = z[key]
        if len(pos) != len(Xa):
            raise AssertionError(f"{run.name} {key}: {len(Xa)} crafted rows, {len(pos)} positions")
        r = float((np.abs(Xa - X[pos]) / scale).max() / eps)
        worst, n = max(worst, r), n + 1
    if worst > 1 + BOX_TOL:
        raise AssertionError(f"{run.name}: crafted rows leave the eps box (max ratio {worst})")
    return {"run": run.name, "n_blocks": n, "max_linf_over_eps": worst}


def random_by_dup(dup: np.ndarray) -> list[dict]:
    p = pd.read_parquet(RANDOM_RUN / "predictions.parquet")
    p["eps"] = p["condition"].str.extract(r"random@([0-9.]+)")[0]
    p["dup"] = dup[p["test_pos"].to_numpy()]
    out = []
    for (model, eps, seed), g in p.groupby(["model", "eps", "seed"]):
        w = g.pivot_table(index="test_pos", columns="condition", values="label")
        unstable = (w.nunique(axis=1) > 1).to_numpy()
        d = dup[w.index.to_numpy()]
        err = g.assign(e=g["label"] != g["y_true"]).groupby("dup")["e"].mean()
        for flag in (True, False):
            out.append(
                {
                    "model": model,
                    "eps": eps,
                    "seed": seed,
                    "dup": flag,
                    "n_rows": int((d == flag).sum()),
                    "instability": float(unstable[d == flag].mean()) if (d == flag).any() else None,
                    "error_rate": float(err.get(flag, np.nan)),
                }
            )
    return out


def hsj_by_dup(dup: np.ndarray, X: np.ndarray, y_test: np.ndarray, scale: np.ndarray) -> dict:
    rows = []
    for run in HSJ_RUNS:
        p = pd.read_parquet(run / "predictions.parquet")
        z = np.load(run / "crafted_inputs.npz")
        clean = p[p["condition"] == "clean"].set_index(["seed", "model", "test_pos"])["label"]
        for key in z.files:
            seed, cond, model = _split_key(key)
            sel = p[(p["seed"] == int(seed)) & (p["condition"] == cond) & (p["model"] == model)]
            pos = sel["test_pos"].to_numpy()
            Xa = z[key]
            clean_label = clean.loc[[(int(seed), model, int(tp)) for tp in pos]].to_numpy()
            if len(pos) != len(Xa):
                raise AssertionError(f"{run.name} {key}: row count differs")
            if not np.array_equal(sel["y_true"].to_numpy(), y_test[pos]):
                raise AssertionError(f"{run.name} {key}: labels differ from the test split")
            dist = (np.abs(Xa - X[pos]) / scale).max(axis=1)
            for i, tp in enumerate(pos):
                rows.append(
                    {
                        "run": run.name,
                        "model": model,
                        "seed": int(seed),
                        "restart": cond,
                        "test_pos": int(tp),
                        "dup": bool(dup[tp]),
                        "linf_sigma": float(dist[i]),
                        "success": bool(sel["label"].iloc[i] != sel["y_true"].iloc[i]),
                        # the attack found an adversarial point: the row moved and its label
                        # left the clean prediction. A row HopSkipJump could not start on comes
                        # back unmoved, and when it was already misclassified the registered
                        # `success` counts it at distortion zero (decision log 2026-11-20T02:03Z)
                        "found": bool(dist[i] > 0 and sel["label"].iloc[i] != clean_label[i]),
                    }
                )
    df = pd.DataFrame(rows)
    summary = {}
    for model, g in df.groupby("model"):
        s = {}
        for flag, name in ((True, "dup"), (False, "nondup")):
            h = g[g["dup"] == flag]
            ok = h[h["success"]]
            s[name] = {
                "distinct_rows": int(h["test_pos"].nunique()),
                "median_linf_all": float(h["linf_sigma"].median()) if len(h) else None,
                "median_linf_success": float(ok["linf_sigma"].median()) if len(ok) else None,
                "success_rate": float(h["success"].mean()) if len(h) else None,
                "n_success": int(len(ok)),
                "median_linf_found": (
                    float(h[h["found"]]["linf_sigma"].median()) if h["found"].any() else None
                ),
                "n_found": int(h["found"].sum()),
                "n_success_unmoved": int((ok["linf_sigma"] == 0).sum()),
            }
        a, b = s["nondup"]["median_linf_success"], s["dup"]["median_linf_success"]
        s["R_h"] = (float("inf") if b == 0 else a / b) if a is not None and b is not None else None
        # the same ratio over the rows the attack found (added 2026-11-20; the branch reads R_h)
        fa, fb = s["nondup"]["median_linf_found"], s["dup"]["median_linf_found"]
        s["R_h_found"] = fa / fb if fa is not None and fb else None
        s["readable"] = s["nondup"]["distinct_rows"] >= MIN_ROWS
        summary[model] = s
    return {"summary": summary, "rows": df}


def square_by_dup(dup: np.ndarray) -> dict:
    p = pd.read_parquet(SQUARE_RUN / "predictions.parquet")
    p["dup"] = dup[p["test_pos"].to_numpy()]
    clean = p[p["condition"] == "clean"].set_index(["seed", "model", "test_pos"])["label"]
    out, per = {}, []
    for cond in sorted(c for c in p["condition"].unique() if c.startswith("square@")):
        eps = cond.split("@")[1]
        a = p[p["condition"] == cond].copy()
        a["clean_label"] = clean.loc[list(zip(a["seed"], a["model"], a["test_pos"]))].to_numpy()
        a = a[a["clean_label"] == a["y_true"]]
        a["success"] = a["label"] != a["y_true"]
        for model, g in a.groupby("model"):
            cell = {}
            for flag, name in ((True, "dup"), (False, "nondup")):
                h = g[g["dup"] == flag]
                cell[name] = {
                    "clean_correct_rows": int(len(h)),
                    "distinct_rows": int(h["test_pos"].nunique()),
                    "success": float(h["success"].mean()) if len(h) else None,
                }
                per.append({"model": model, "eps": eps, "dup": flag, **cell[name]})
            d, n = cell["dup"]["success"], cell["nondup"]["success"]
            cell["dS"] = d - n if d is not None and n is not None else None
            cell["readable"] = cell["nondup"]["distinct_rows"] >= MIN_ROWS
            out.setdefault(model, {})[eps] = cell
    return {"summary": out, "rows": pd.DataFrame(per)}


def perclass_by_dup(dup: np.ndarray) -> list[dict]:
    from sklearn.metrics import f1_score

    p = pd.read_parquet(PERCLASS_RUN / "predictions.parquet")
    p["dup"] = dup[p["test_pos"].to_numpy()]
    out = []
    for (model, cond, seed, flag), g in p.groupby(["model", "condition", "seed", "dup"]):
        out.append(
            {
                "model": model,
                "condition": cond,
                "seed": seed,
                "dup": flag,
                "n_rows": int(len(g)),
                "macro_f1": float(f1_score(g["y_true"], g["label"], average="macro")),
                "error_rate": float((g["label"] != g["y_true"]).mean()),
            }
        )
    return out


def fix_check(reg_full_X, y_tr) -> dict:
    out = {}
    for regime in ("full", "pca"):
        for ind in (False, True):
            pipe = build_preprocessor(
                regime, n_components=8 if regime == "pca" else None, seed=0, add_indicator=ind
            )
            Z = pipe.fit_transform(reg_full_X)
            df = pd.DataFrame(Z)
            key = pd.util.hash_pandas_object(df, index=False).to_numpy()
            g = pd.DataFrame({"k": key, "y": y_tr})
            sizes = g.groupby("k")["y"].agg(["size", "nunique"])
            multi = sizes[sizes["size"] > 1]
            out[f"{regime}|indicators={ind}"] = {
                "rows": int(len(Z)),
                "distinct_rows": int(len(sizes)),
                "duplicate_groups": int(len(multi)),
                "label_conflicting_groups": int((multi["nunique"] > 1).sum()),
                "rows_in_conflicting_groups": int(multi.loc[multi["nunique"] > 1, "size"].sum()),
                "n_features": int(Z.shape[1]),
            }
    return out


def branch(hsj: dict, sq: dict) -> dict:
    per = {}
    for t in TREES:
        rh = hsj[t]["R_h"]
        cell = sq[t][STAT_EPS]
        ds = cell["dS"]
        readable = hsj[t]["readable"] and cell["readable"]
        per[t] = {
            "R_h": rh,
            "dS_eps_0.1": ds,
            "readable": readable,
            "artifact": bool(readable and ((rh is not None and rh >= R_CONF) or ds >= S_CONF)),
            "none": bool(readable and rh is not None and rh <= R_KILL and abs(ds) <= S_KILL),
        }
    if all(v["none"] for v in per.values()):
        verdict = "KILL"
    elif all(v["artifact"] for v in per.values()):
        verdict = "CONFIRMED"
    else:
        verdict = "PARTIAL"
    return {"per_tree": per, "verdict": verdict}


def main() -> None:
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    reg = prepare_regime(VARIANT, "pca", n_components=8, seed=0)
    sub = kernel_subset(reg, cfg_q["qkernel"]["subset_cap"], seed=0)
    X, scale = reg.X["test"], reg.feature_scale
    dup = dup_flags(reg, sub)
    cfg = {
        "experiment": "NS15",
        "preregistration": "decision log 2026-11-19T08:25Z, PRE-REGISTRATION (round 5)",
        "duplicate_definition": "PCA-8 test input equals a row of kernel_subset(seed=0) exactly",
        "exports": {
            "random": RANDOM_RUN.name,
            "hsj": [r.name for r in HSJ_RUNS],
            "square": SQUARE_RUN.name,
            "perclass": PERCLASS_RUN.name,
        },
        "statistic_eps": STAT_EPS,
        "min_rows": MIN_ROWS,
    }
    with RunDir(f"gas_duplicate_rescore_{VARIANT}", config=cfg, seeds=[0]) as run:
        sq_preds = pd.read_parquet(SQUARE_RUN / "predictions.parquet")
        rnd_preds = pd.read_parquet(RANDOM_RUN / "predictions.parquet")
        checks = [
            box_check(SQUARE_RUN, sq_preds, X, scale),
            box_check(RANDOM_RUN, rnd_preds, X, scale),
        ]
        log.info("reconstruction checks %s", json.dumps(checks))
        hsj = hsj_by_dup(dup, X, reg.y["test"], scale)
        sq = square_by_dup(dup)
        rnd = random_by_dup(dup)
        pc = perclass_by_dup(dup)
        Xraw, yraw, splits = load_processed(VARIANT)
        tr = np.asarray(splits["train"])
        fx = fix_check(np.asarray(Xraw, dtype=np.float64)[tr], np.asarray(yraw)[tr])
        br = branch(hsj["summary"], sq["summary"])
        eval_sets = {
            "hsj_eval_rows": sorted(set(hsj["rows"]["test_pos"])),
            "square_eval_rows": sorted(set(sq_preds["test_pos"])),
        }
        dup_counts = {k: {"rows": len(v), "dup": int(dup[v].sum())} for k, v in eval_sets.items()}
        res = {
            "config": cfg,
            "reconstruction_checks": checks,
            "test_rows": int(len(dup)),
            "test_rows_dup": int(dup.sum()),
            "eval_dup_counts": dup_counts,
            "hsj": hsj["summary"],
            "square": sq["summary"],
            "fix_check": fx,
            "branch": br,
        }
        run.write_json("results.json", res)
        hsj["rows"].to_csv(run.path / "hsj_rows_by_dup.csv", index=False)
        sq["rows"].to_csv(run.path / "square_by_dup.csv", index=False)
        pd.DataFrame(rnd).to_csv(run.path / "random_by_dup.csv", index=False)
        pd.DataFrame(pc).to_csv(run.path / "perclass_by_dup.csv", index=False)
        pd.DataFrame({"test_pos": np.arange(len(dup)), "dup": dup}).to_csv(
            run.path / "test_rows_dup.csv", index=False
        )
        (run.path / "exports_provenance.json").write_text(
            json.dumps(
                {
                    "env": {
                        k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
                    },
                    "inputs": cfg["exports"],
                    "random_streams": [],
                    "notes": "archive-only re-score; no model is fitted",
                },
                indent=1,
            )
        )
        run.metrics.update({"branch": br["verdict"]})
        log.info("DUP %s", json.dumps(dup_counts))
        log.info("BRANCH %s", json.dumps(br))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
