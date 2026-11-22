"""Round 5: the split exhibit's level arm (full features, tuned LightGBM) read per class, on two
split designs. Pre-registration: decision log 2026-11-19T08:25Z, "PRE-REGISTRATION (round 5".

--design file (NS13: is the row-split optimism concentrated in the natural class?). The row
split and the five whole-file partitions of `run_group_split_control` (same groups, same
`default_rng(0)` stream). Per split: macro-F1, per-class recall, precision and F1, confusion
matrix. Anchor: macro-F1 equals the archived level values (`ARCHIVED_FILE_RUN`) within 1e-6.
Statistic: dR_c = recall_c(row) - mean over partitions of recall_c(group), c in {natural,
attack}; dF1_c likewise; natural share of the macro-F1 drop = dF1_nat / (dF1_nat + dF1_att).
Branch: KILL if dR_nat <= dR_att; HOLDS if dR_nat > dR_att and the natural share > 0.5;
PARTIAL otherwise.

--design family (NS14: do unseen attack families fail silent on the grid?). Scenario markers
come from the multiclass files, which align with the binary files row for row (dataset README
Tables I-V; checked in the run). Natural rows keep the shipped row split. Arms:
  R     the shipped row split;
  F<k>  attack family k held out: train = shipped-train natural rows + shipped-train rows of the
        other two families; test = shipped-test natural rows + every row of family k;
  S<k>  negative control: as many attack rows as family k, drawn at random from all attack rows
        (default_rng(k)), held out in the same way.
Per arm: macro-F1 against the arm's stratified-random floor, and the recall of the held-out
attack rows. Branch: KILL if every F<k> scores above its floor; REPLICATES if at least two of
three fall below; PARTIAL if one does; VOID if any S<k> differs from R by more than 0.05.

Every fit is seed 0 with `cls_lgbm_binary_full`'s best parameters; every transform is fit on the
arm's train rows. On full features LightGBM's fit depends on the thread count at about 1e-4
macro-F1 (3 threads missed the archived row level by 8.6e-5; 24 threads, the archived run's
OpenMP default, reproduces all six levels exactly: decision log, NS13 finding), so the exact
anchor needs one worker at 24 threads; a unit takes about 20 s. Units (one fit each) run in
`--workers` spawned processes of `--threads` threads; each unit writes `units/<arm>.npz` as it
lands, and `--resume <run dir>` reuses them.

Run:
    CUDA_VISIBLE_DEVICES=-1 uv run python -m experiments.run_level_per_class --design file \
        --workers 1 --threads 24
    CUDA_VISIBLE_DEVICES=-1 uv run python -m experiments.run_level_per_class --design family \
        --workers 1 --threads 24
"""

from __future__ import annotations

import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=3)
THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
for _var in THREAD_VARS:
    os.environ[_var] = str(THREADS)

import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from experiments.run_group_split_control import (  # noqa: E402
    group_splits,
    reconstruct_groups,
    tuned_params,
)
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.data.validate import load_variant_frames  # noqa: E402
from qgridbench.eval.baselines import trivial_baselines  # noqa: E402
from qgridbench.features.reduce import build_preprocessor  # noqa: E402
from qgridbench.models.classical.zoo import build, fit_model  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import set_all_seeds  # noqa: E402

log = get_logger(__name__)

VARIANT = "binary"
SEED = 0
N_PARTITIONS = 5
ANCHOR_TOL = 1e-6
VOID_TOL = 0.05
ARCHIVED_FILE_RUN = REPO_ROOT / "results/runs/20260816T023620Z_group_split_control_binary"
OPTUNA_DB = REPO_ROOT / "results" / "optuna_cls_lgbm_binary_full.db"
README = "data/raw/PowerSystem_Dataset_README.pdf, Tables I-V"
NATURAL = [1, 2, 3, 4, 5, 6, 13, 14, 41]
ATTACK_FAMILIES = {
    "data_injection": list(range(7, 13)),
    "remote_tripping": list(range(15, 21)),
    "relay_setting_change": list(range(21, 31)) + list(range(35, 41)),
}
CLASSES = {0: "natural", 1: "attack"}

_DATA: dict = {}


def _init_worker(X: np.ndarray, y: np.ndarray, threads: int) -> None:
    if os.environ.get("OMP_NUM_THREADS") != str(threads):
        raise AssertionError("worker thread variables differ from --threads")
    _DATA["X"], _DATA["y"] = X, y


def fit_score(spec: dict) -> dict:
    """One fit: the shipped transform and the tuned LightGBM on spec's train rows, scored on its
    test rows. Returns the metrics and the per-row arrays for the exports."""
    X, y = _DATA["X"], _DATA["y"]
    tr, te = spec["train"], spec["test"]
    t0 = time.perf_counter()
    pipe = build_preprocessor("full", n_components=None, seed=SEED)
    Ztr = pipe.fit_transform(X[tr])
    Zte = pipe.transform(X[te])
    set_all_seeds(SEED)
    est = build("lgbm", spec["params"], SEED)
    fit_model("lgbm", est, Ztr, y[tr])
    proba = est.predict_proba(Zte)[:, 1]
    pred = est.predict(Zte)
    yte = y[te]
    prec, rec, f1c, sup = precision_recall_fscore_support(yte, pred, labels=[0, 1], zero_division=0)
    floor = trivial_baselines(y[tr], yte)["baseline_stratified_random"]["aggregate"]["macro_f1"]
    out = {
        "arm": spec["arm"],
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "train_attack_share": float(y[tr].mean()),
        "test_attack_share": float(yte.mean()),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "floor_macro_f1": float(floor["mean"]),
        "per_class": {
            CLASSES[c]: {
                "recall": float(rec[c]),
                "precision": float(prec[c]),
                "f1": float(f1c[c]),
                "support": int(sup[c]),
            }
            for c in (0, 1)
        },
        "confusion_true_by_pred": confusion_matrix(yte, pred, labels=[0, 1]).tolist(),
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_rss_gb": round(psutil.Process().memory_info().peak_wset / 2**30, 3)
        if hasattr(psutil.Process().memory_info(), "peak_wset")
        else None,
    }
    if spec.get("held") is not None:
        held = np.isin(te, spec["held"])
        out["held_attack_rows"] = int(held.sum())
        out["held_attack_recall"] = float((pred[held] == 1).mean())
        out["seen_attack_recall"] = (
            float((pred[(yte == 1) & ~held] == 1).mean()) if ((yte == 1) & ~held).any() else None
        )
    arrays = {"test": te, "proba_1": proba, "label": pred.astype(np.int64)}
    return {"metrics": out, "arrays": arrays}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def binary_markers(Xp: np.ndarray, yp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scenario marker per processed binary row, read by position from the multiclass files,
    and the float64 feature rows (the split exhibit's input, `reconstruct_groups`).

    Repeats preprocess_variant's pooling, inf policy and (features, label) dedupe on the binary
    files, and checks the result equals the processed artifact (stored as float32), the per-file
    row counts match, and the natural markers are exactly the label-0 rows. The fits use the
    float64 rows: on the float32 artifact the row-split level is 0.9058, not the exhibit's
    0.9053 (first NS14 run, superseded)."""
    cfg = load_yaml(REPO_ROOT / "configs" / "data.yaml")
    lc = cfg["schema"]["label_column"]
    fb, fm = load_variant_frames(VARIANT), load_variant_frames("multiclass")
    if [len(f) for f in fb] != [len(f) for f in fm]:
        raise AssertionError("binary and multiclass files differ in row counts")
    pb = pd.concat(fb, ignore_index=True)
    pm = pd.concat(fm, ignore_index=True)
    y = pb[lc].map(cfg["labels"][VARIANT]).to_numpy(dtype=np.int64)
    X = pb.drop(columns=[lc]).to_numpy(dtype=np.float64)
    X[np.isinf(X)] = np.nan
    df = pd.DataFrame(X)
    df["__y"] = y
    keep = (~df.duplicated()).to_numpy()
    X, y = X[keep], y[keep]
    marker = pm[lc].to_numpy(dtype=np.int64)[keep]
    if X.shape != Xp.shape or not np.array_equal(y, yp):
        raise AssertionError("binary rows do not reproduce the processed artifact")
    X32 = X.astype(Xp.dtype)  # the processed artifact stores float32
    if not np.array_equal(np.isnan(X32), np.isnan(Xp)) or not np.array_equal(
        np.nan_to_num(X32), np.nan_to_num(Xp)
    ):
        raise AssertionError("binary feature values differ from the processed artifact")
    if not np.array_equal(np.isin(marker, NATURAL), yp == 0):
        raise AssertionError("natural markers are not exactly the label-0 rows")
    fam_of = np.full(len(marker), "", dtype=object)
    for k, ms in ATTACK_FAMILIES.items():
        fam_of[np.isin(marker, ms)] = k
    if np.any((yp == 1) & (fam_of == "")):
        raise AssertionError(f"attack markers without a family: {np.unique(marker[fam_of == ''])}")
    return marker, X


def file_specs(params: dict) -> tuple[list[dict], dict]:
    X, y, gid = reconstruct_groups()
    Xp, yp, sp = load_processed(VARIANT)
    if X.shape != Xp.shape or not np.array_equal(y, yp):
        raise AssertionError("group reconstruction misaligned with the processed artifact")
    specs = [{"arm": "row", "train": np.asarray(sp["train"]), "test": np.asarray(sp["test"])}]
    rng = np.random.default_rng(0)  # run_group_split_control's stream: the same partitions
    held = {}
    for p in range(N_PARTITIONS):
        s, groups = group_splits(gid, y, rng)
        specs.append({"arm": f"group{p}", "train": s["train"], "test": s["test"]})
        held[f"group{p}"] = groups[2]
    for s in specs:
        s["params"] = params
    return specs, {"X": X, "y": y, "test_groups": held, "gid": gid}


def family_specs(params: dict) -> tuple[list[dict], dict]:
    Xp, yp, sp = load_processed(VARIANT)
    marker, X64 = binary_markers(Xp, yp)
    tr0, te0 = np.asarray(sp["train"]), np.asarray(sp["test"])
    natural = yp == 0
    idx = np.arange(len(yp))
    attack_idx = idx[~natural]
    specs = [{"arm": "R", "train": tr0, "test": te0, "held": None}]
    for i, (k, ms) in enumerate(ATTACK_FAMILIES.items()):
        fam = np.isin(marker, ms)
        held = idx[fam]
        specs.append(
            {
                "arm": f"F_{k}",
                "train": np.sort(tr0[~fam[tr0]]),
                "test": np.sort(np.concatenate([te0[natural[te0]], held])),
                "held": held,
            }
        )
        rnd = np.zeros(len(yp), dtype=bool)
        rnd[np.random.default_rng(i).choice(attack_idx, size=len(held), replace=False)] = True
        specs.append(
            {
                "arm": f"S_{k}",
                "train": np.sort(tr0[~rnd[tr0]]),
                "test": np.sort(np.concatenate([te0[natural[te0]], idx[rnd]])),
                "held": idx[rnd],
            }
        )
    for s in specs:
        s["params"] = params
        if np.intersect1d(s["train"], s["test"]).size:
            raise AssertionError(f"{s['arm']}: train and test overlap")
    counts = {
        k: {"markers": ms, "rows": int(np.isin(marker, ms).sum())}
        for k, ms in ATTACK_FAMILIES.items()
    }
    return specs, {"X": X64, "y": yp, "marker": marker, "family_rows": counts}


def verdict_file(m: dict) -> dict:
    row = m["row"]
    grp = [m[f"group{p}"] for p in range(N_PARTITIONS)]

    def d(c, key):
        return row["per_class"][c][key] - float(np.mean([g["per_class"][c][key] for g in grp]))

    dR = {c: d(c, "recall") for c in ("natural", "attack")}
    dF = {c: d(c, "f1") for c in ("natural", "attack")}
    share = dF["natural"] / (dF["natural"] + dF["attack"])
    if dR["natural"] <= dR["attack"]:
        branch = "KILL"
    elif share > 0.5:
        branch = "HOLDS"
    else:
        branch = "PARTIAL"
    return {
        "delta_recall": dR,
        "delta_f1": dF,
        "natural_share_of_macro_f1_drop": share,
        "macro_f1_row": row["macro_f1"],
        "macro_f1_group_mean": float(np.mean([g["macro_f1"] for g in grp])),
        "macro_f1_group_std": float(np.std([g["macro_f1"] for g in grp], ddof=1)),
        "group_recall": {
            c: {
                "mean": float(np.mean([g["per_class"][c]["recall"] for g in grp])),
                "min": float(np.min([g["per_class"][c]["recall"] for g in grp])),
                "max": float(np.max([g["per_class"][c]["recall"] for g in grp])),
            }
            for c in ("natural", "attack")
        },
        "branch": branch,
    }


def anchor_file(m: dict) -> dict:
    arch = json.loads((ARCHIVED_FILE_RUN / "results.json").read_text())["rows"]
    ref = {"row": arch[0]["level_full_lgbm"]}
    for r in arch[1:]:
        ref[f"group{r['partition']}"] = r["level_full_lgbm"]
    diff = {k: m[k]["macro_f1"] - v for k, v in ref.items()}
    return {
        "archived_run": ARCHIVED_FILE_RUN.name,
        "diff": diff,
        "max_abs_diff": max(abs(v) for v in diff.values()),
        "pass": all(abs(v) <= ANCHOR_TOL for v in diff.values()),
    }


def verdict_family(m: dict) -> dict:
    R = m["R"]
    fam = {k: m[f"F_{k}"] for k in ATTACK_FAMILIES}
    ctl = {k: m[f"S_{k}"] for k in ATTACK_FAMILIES}
    below = {k: v["macro_f1"] < v["floor_macro_f1"] for k, v in fam.items()}
    ctl_gap = {k: v["macro_f1"] - R["macro_f1"] for k, v in ctl.items()}
    n_below = sum(below.values())
    if any(abs(g) > VOID_TOL for g in ctl_gap.values()):
        branch = "VOID"
    elif n_below == 0:
        branch = "KILL"
    elif n_below >= 2:
        branch = "REPLICATES"
    else:
        branch = "PARTIAL"
    return {
        "per_family": {
            k: {
                "macro_f1": v["macro_f1"],
                "floor": v["floor_macro_f1"],
                "below_floor": bool(below[k]),
                "held_attack_recall": v["held_attack_recall"],
                "natural_recall": v["per_class"]["natural"]["recall"],
                "control_macro_f1": ctl[k]["macro_f1"],
                "control_held_recall": ctl[k]["held_attack_recall"],
                "control_minus_R": ctl_gap[k],
            }
            for k, v in fam.items()
        },
        "R_macro_f1": R["macro_f1"],
        "n_below_floor": n_below,
        "branch": branch,
    }


def run_units(specs, data, workers, threads, run_path: Path, resume: Path | None) -> dict:
    units_dir = run_path / "units"
    units_dir.mkdir(exist_ok=True)
    done: dict = {}
    if resume:
        part = json.loads((resume / "results_partial.json").read_text())["units"]
        for arm, met in part.items():
            z = np.load(resume / "units" / f"{arm}.npz")
            done[arm] = {"metrics": met, "arrays": {k: z[k] for k in z.files}}
            np.savez(units_dir / f"{arm}.npz", **done[arm]["arrays"])
    todo = [s for s in specs if s["arm"] not in done]
    log.info("units: %d to run, %d resumed", len(todo), len(done))

    def save(res):
        arm = res["metrics"]["arm"]
        done[arm] = res
        np.savez(units_dir / f"{arm}.npz", **res["arrays"])
        partial = {"units": {a: r["metrics"] for a, r in done.items()}}
        (run_path / "results_partial.json").write_text(json.dumps(partial, indent=1))
        mt = res["metrics"]
        log.info(
            "%s: macro-F1 %.4f (floor %.4f) | recall natural %.3f attack %.3f%s (%.0f s)",
            arm,
            mt["macro_f1"],
            mt["floor_macro_f1"],
            mt["per_class"]["natural"]["recall"],
            mt["per_class"]["attack"]["recall"],
            f" | held {mt['held_attack_recall']:.3f}" if "held_attack_recall" in mt else "",
            mt["wall_s"],
        )

    if todo:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(todo)),
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(data["X"], data["y"], threads),
        ) as ex:
            futs = [ex.submit(fit_score, s) for s in todo]
            for f in as_completed(futs):
                save(f.result())
    if set(done) != {s["arm"] for s in specs}:
        raise AssertionError("finished units do not match the planned units")
    return done


def write_exports(run_path: Path, done: dict, data: dict, extra: dict) -> None:
    rows = []
    for arm, res in done.items():
        a = res["arrays"]
        te = a["test"]
        frame = {
            "arm": arm,
            "row_index": te,
            "y_true": data["y"][te],
            "proba_1": a["proba_1"],
            "label": a["label"],
        }
        if "marker" in data:
            frame["marker"] = data["marker"][te]
        if "gid" in data:
            frame["file_group"] = data["gid"][te]
        rows.append(pd.DataFrame(frame))
    pd.concat(rows, ignore_index=True).to_parquet(run_path / "predictions.parquet", index=False)
    flat = []
    for arm, res in done.items():
        mt = res["metrics"]
        r = {k: v for k, v in mt.items() if not isinstance(v, (dict, list))}
        for c, d in mt["per_class"].items():
            for k, v in d.items():
                r[f"{c}_{k}"] = v
        flat.append(r)
    pd.DataFrame(flat).sort_values("arm").to_csv(run_path / "per_arm.csv", index=False)
    prov = {
        "env": {k: os.environ.get(k) for k in (*THREAD_VARS, "CUDA_VISIBLE_DEVICES")},
        "random_streams": extra["streams"],
        "tuned_source": {
            "study": "cls_lgbm_binary_full",
            "path": str(OPTUNA_DB.relative_to(REPO_ROOT)),
            "sha256": _sha256(OPTUNA_DB),
        },
        "splits_file": "data/processed splits for 'binary' (load_processed)",
        "seed": SEED,
        "notes": extra.get("notes", ""),
    }
    (run_path / "exports_provenance.json").write_text(json.dumps(prov, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", choices=["file", "family"], required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--resume", type=Path, help="run directory of an interrupted run")
    args = ap.parse_args()
    if args.threads != THREADS:
        raise AssertionError("early thread parse differs from argparse")
    if args.workers * args.threads > psutil.cpu_count(logical=True):
        raise ValueError("workers x threads exceeds the logical CPU count")
    params, _ = tuned_params()
    if args.design == "file":
        specs, data = file_specs(params)
        name = f"level_per_class_file_{VARIANT}"
        streams = ["default_rng(0) (run_group_split_control's partition stream)"]
        experiment = "NS13"
    else:
        specs, data = family_specs(params)
        name = f"level_per_class_family_{VARIANT}"
        streams = [f"default_rng({i}) (S_{k})" for i, k in enumerate(ATTACK_FAMILIES)]
        experiment = "NS14"
    cfg = {
        "experiment": experiment,
        "design": args.design,
        "preregistration": "decision log 2026-11-19T08:25Z, PRE-REGISTRATION (round 5)",
        "model": "lgbm, full features, cls_lgbm_binary_full best_params, seed 0",
        "arms": [s["arm"] for s in specs],
        "workers": args.workers,
        "threads": args.threads,
        "resume": str(args.resume) if args.resume else None,
    }
    if args.design == "family":
        cfg["scenario_map"] = {"natural": NATURAL, **ATTACK_FAMILIES, "source": README}
    with RunDir(name, config=cfg, seeds=[SEED]) as run:
        done = run_units(specs, data, args.workers, args.threads, run.path, args.resume)
        m = {a: r["metrics"] for a, r in done.items()}
        if args.design == "file":
            anchor = anchor_file(m)
            verdict = verdict_file(m)
            extra = {"test_groups": data["test_groups"]}
        else:
            anchor = {
                "R_vs_archived_row_level": m["R"]["macro_f1"]
                - json.loads((ARCHIVED_FILE_RUN / "results.json").read_text())["rows"][0][
                    "level_full_lgbm"
                ]
            }
            anchor["pass"] = abs(anchor["R_vs_archived_row_level"]) <= ANCHOR_TOL
            verdict = verdict_family(m)
            extra = {"family_rows": data["family_rows"]}
        res = {"config": cfg, "anchor": anchor, "verdict": verdict, "units": m, **extra}
        run.write_json("results.json", res)
        write_exports(run.path, done, data, {"streams": streams})
        run.metrics.update({"branch": verdict["branch"], "anchor_pass": anchor["pass"]})
        log.info("ANCHOR %s", json.dumps(anchor))
        log.info("VERDICT %s", json.dumps(verdict))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
