"""R4-12 gas pipeline: missingness control, split ladder, X3 framing control, grouped contests.

Pre-registration: decision log 2026-11-16T11:43Z, "PRE-RUN AMENDMENT (R4-12". Variant
`gas_binary` (qgridbench.data.gas). Grouped arms reuse the R-arm tuned hyperparameters
(ruling 3): LightGBM full-feature parameters from its Optuna study, PCA-8 parameters from
`protocol.best_params_path("gas_binary", 8)`. Every fit is seed 0; every transform is fit on
the arm's train split.

Arms: R (row-level, the shipped split), Rraw (R's rule on the undeduplicated capture, context
only), T<p> (G-time, 15 blocks, partition p), Y<p> (G-type, attack-name holdout), S<p>
(shuffled control: T<p>'s side sizes, rows assigned at random).

STAGES (each a set of independent units, run in a process pool)
  missingness  control M (ruling 2): LightGBM (full) and QSVM-ZZ (PCA-8) under R, T0, T1, T2,
               median imputation against median + missing indicators.
               all |delta| <= 0.01 -> not load-bearing; any > 0.02 -> STOP (exit 2) before any
               gas number is quoted; otherwise report delta beside each number.
  ladder       H-L1: LightGBM (full) R - mean(T0..T4); >= 0.05 PORTS, < 0.02 DOES NOT PORT,
               else PARTIAL. Y and Rraw reported, no branch. Neighbour audit (share of test
               rows with a train row closer than 0.1, standardized full features) per arm.
               Controls: R reproduces the Phase-1 LightGBM seed-0 test macro-F1 within 1e-6
               (--phase1-lgbm-run); every S<p> within 0.02 of R, else the ladder is VOID.
  framing      H-L2 (X3, registered verbatim): per G-time partition and held test block b,
               LightGBM trained on the other test blocks (target domain) against the same
               model on a stratified source subset of equal size, both scored on b.
               mean(target - size-matched) > 0.05 KILLS the X3 framing; W3's two-sided
               |delta| <= 0.05 is reported too. G-type secondary: eval set = attack-name group
               g plus a fixed half of the test-domain normal rows (default_rng(p)); target
               training set = the other test groups plus the other normal half.
               Negative control: a domain classifier on two random halves of the train
               domain reads AUC <= 0.6 (W3).
  contest      H-C2 under grouped arms (reported, no branch): all eight families at PCA-8 on
               the 2,000-row cap, best classical minus best quantum per partition.

Run:  uv run python -m experiments.run_gas_split_ladder --stages missingness --workers 2 --threads 4
      uv run python -m experiments.run_gas_split_ladder --stages ladder framing contest \
          --phase1-lgbm-run results/runs/<stamp>_classical_gas_binary_full --workers 4 --threads 4
      (resume: --resume results/runs/<stamp>_gas_split_ladder_gas_binary)
      --variant gas_binary_slog runs every stage on that derived variant (R5-G1 option B).
"""

from __future__ import annotations

import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=4)
_pre.add_argument("--variant", default=None)
_pre_args = _pre.parse_known_args(sys.argv[1:])[0]
THREADS = _pre_args.threads
# --variant reaches the spawned workers through the environment (their argv is not ours)
if _pre_args.variant is not None:
    os.environ["QGB_GAS_VARIANT"] = _pre_args.variant
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
for _var in THREAD_VARS:
    os.environ[_var] = str(THREADS)

import json  # noqa: E402
import multiprocessing  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import psutil  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from experiments.run_shift_decomposition import domain_auc, floor_of  # noqa: E402
from qgridbench.data.gas import (  # noqa: E402
    apply_variant_transform,
    attack_name_splits,
    base_variant,
    gas_config,
    gas_features,
    label_vector,
    load_aux,
    load_gas_frame,
    row_split,
    time_block_splits,
)
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.features.reduce import (  # noqa: E402
    AngleScaler,
    build_preprocessor,
    fit_transform_splits,
)
from qgridbench.models.classical.zoo import (  # noqa: E402
    MODEL_NAMES,
    build,
    fit_model,
    stratified_cap,
)
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel  # noqa: E402
from qgridbench.models.quantum.vqc import VQClassifier  # noqa: E402
from qgridbench.protocol import best_params_path  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import set_all_seeds  # noqa: E402

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

VARIANT = os.environ.get("QGB_GAS_VARIANT", "gas_binary")
base_variant(VARIANT)  # fails loud on an unknown variant
QUBITS = 8
CAP = 2000
SEED = 0
RADIUS = 0.1
MISS_OK, MISS_STOP = 0.01, 0.02
LADDER_PORT, LADDER_NOT = 0.05, 0.02
X3_KILL = 0.05
SHUFFLE_TOL = 0.02
DOMAIN_AUC_MAX = 0.6
ANCHOR_TOL = 1e-6
MEM_FRACTION = 0.8
STAGES = ("missingness", "ladder", "framing", "contest")


# ---------------------------------------------------------------- data and arms -------------


def _data():
    X, y, sp_r = load_processed(VARIANT)
    return X, y, sp_r, load_aux(VARIANT), gas_config()


def arm_splits(arm: str, X, y, sp_r, aux, cfg) -> dict[str, np.ndarray]:
    g = cfg["groups"]
    kind, p = arm[0], (int(arm[1:]) if len(arm) > 1 and arm[1:].isdigit() else None)
    if arm == "R":
        return sp_r
    if kind == "T":
        return time_block_splits(aux["block"], p, g["time_blocks"])
    if kind == "Y":
        return attack_name_splits(
            sp_r, y == 0, aux["specific"], aux["categorized"], p, g["attack_names"], g["name_cycle"]
        )
    if kind == "S":
        t = time_block_splits(aux["block"], p, g["time_blocks"])
        perm = np.random.default_rng(1000 + p).permutation(len(y))
        n_tr, n_va = len(t["train"]), len(t["val"])
        return {
            "train": np.sort(perm[:n_tr]),
            "val": np.sort(perm[n_tr : n_tr + n_va]),
            "test": np.sort(perm[n_tr + n_va :]),
        }
    raise ValueError(f"unknown arm {arm}")


def lgbm_full_params() -> dict:
    name = f"cls_lgbm_{VARIANT}_full"
    db = (REPO_ROOT / "results" / f"optuna_{name}.db").as_posix()
    st = optuna.load_study(study_name=name, storage=f"sqlite:///{db}")
    if len([t for t in st.trials if t.state.is_finished()]) < 50:
        raise RuntimeError(f"{name} holds fewer than the 50-trial budget")
    return st.best_params


def pca_params() -> dict:
    return json.loads(best_params_path(VARIANT, QUBITS).read_text())


def _f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro"))


def lgbm_f1(params, X, y, sp, add_indicator=False) -> float:
    Z = fit_transform_splits(build_preprocessor("full", add_indicator=add_indicator), X, sp)
    set_all_seeds(SEED)
    est = build("lgbm", params, SEED)
    fit_model("lgbm", est, Z["train"], y[sp["train"]])
    return _f1(y[sp["test"]], est.predict_proba(Z["test"]).argmax(1))


def qsvm_f1(bw, C, X, y, sp, add_indicator=False, encoding="zz") -> float:
    pipe = build_preprocessor("pca", n_components=QUBITS, seed=SEED, add_indicator=add_indicator)
    Z = fit_transform_splits(pipe, X, sp)
    ytr = y[sp["train"]]
    sub = stratified_cap(Z["train"], ytr, CAP, seed=SEED)
    sc = AngleScaler().fit(Z["train"])
    s_tr = compute_states(sc.transform(Z["train"][sub]), encoding, bandwidth=bw)
    clf = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(
        fidelity_kernel(s_tr, s_tr), ytr[sub]
    )
    A_te = sc.transform(Z["test"])
    pred = np.concatenate(
        [
            clf.predict(fidelity_kernel(compute_states(A_te[i : i + 2000], encoding, bw), s_tr))
            for i in range(0, len(A_te), 2000)
        ]
    )
    return _f1(y[sp["test"]], pred)


def neighbour_share(X, sp) -> float:
    Z = fit_transform_splits(build_preprocessor("full"), X, sp)
    dist = NearestNeighbors(n_neighbors=1).fit(Z["train"]).kneighbors(Z["test"])[0][:, 0]
    return float((dist < RADIUS).mean())


def realized(y, sp) -> dict:
    return {
        "sizes": {s: int(len(v)) for s, v in sp.items()},
        "attack_rate": {s: float(y[v].mean()) for s, v in sp.items()},
        "floor": floor_of(y[sp["train"]], y[sp["test"]]),
    }


# ---------------------------------------------------------------- units ---------------------


def unit_missingness(arm: str, model: str) -> dict:
    X, y, sp_r, aux, cfg = _data()
    sp = arm_splits(arm, X, y, sp_r, aux, cfg)
    if model == "lgbm":
        params = lgbm_full_params()
        a, b = (lgbm_f1(params, X, y, sp, ind) for ind in (False, True))
    else:
        bp = pca_params()
        a, b = (qsvm_f1(bp["qk_bw"], bp["qk_C"], X, y, sp, ind) for ind in (False, True))
    return {"f1_median": a, "f1_indicator": b, "delta": b - a, **realized(y, sp)}


def _raw_r():
    cfg = gas_config()
    df = load_gas_frame(cfg)
    X = df[gas_features(cfg)].to_numpy(dtype=np.float64).astype(np.float32)
    X = apply_variant_transform(X, VARIANT, cfg)
    y = label_vector(df, cfg, base_variant(VARIANT, cfg))
    sp_cfg = cfg["preprocess"]["split"]
    return X, y, row_split(y, sp_cfg["ratios"], sp_cfg["seed"])


def unit_ladder(arm: str) -> dict:
    params = lgbm_full_params()
    if arm == "Rraw":
        X, y, sp = _raw_r()
        return {"f1": lgbm_f1(params, X, y, sp), **realized(y, sp)}
    X, y, sp_r, aux, cfg = _data()
    sp = arm_splits(arm, X, y, sp_r, aux, cfg)
    out = {"f1": lgbm_f1(params, X, y, sp), "neighbour_share": neighbour_share(X, sp)}
    if arm[0] == "T":
        out["test_blocks"] = sorted(np.unique(aux["block"][sp["test"]]).tolist())
    if arm[0] == "Y":
        att = sp["test"][y[sp["test"]] == 1]
        out["test_types"] = sorted(np.unique(aux["specific"][att]).tolist())
    return {**out, **realized(y, sp)}


def unit_framing(arm: str) -> dict:
    X, y, sp_r, aux, cfg = _data()
    params = lgbm_full_params()
    p = int(arm[1:])
    sp = arm_splits(arm, X, y, sp_r, aux, cfg)
    Z = fit_transform_splits(build_preprocessor("full"), X, sp)
    Ztr, ytr, Zte, yte = Z["train"], y[sp["train"]], Z["test"], y[sp["test"]]

    def score(Za, ya, Zb, yb) -> float:
        set_all_seeds(SEED)
        est = build("lgbm", params, SEED)
        fit_model("lgbm", est, Za, ya)
        return _f1(yb, est.predict_proba(Zb).argmax(1))

    folds = []
    if arm[0] == "T":
        held = aux["block"][sp["test"]]
        units = [(int(b), held == b, held != b) for b in np.unique(held)]
    else:
        spec = aux["specific"][sp["test"]]
        normal = np.flatnonzero(yte == 0)
        perm = np.random.default_rng(p).permutation(normal)
        ne = np.zeros(len(yte), bool)
        ne[perm[: len(perm) // 2]] = True
        nt = (yte == 0) & ~ne
        units = []
        for grp in cfg["groups"]["attack_names"]:
            g = np.isin(spec, grp) & (yte == 1)
            if g.any():
                units.append((grp, g | ne, ((yte == 1) & ~g) | nt))
    for key, ev, tr in units:
        if len(np.unique(yte[ev])) < 2 or len(np.unique(yte[tr])) < 2:
            folds.append({"held": key, "skipped": "single-class fold"})
            continue
        n_tgt = int(tr.sum())
        sub = stratified_cap(Ztr, ytr, n_tgt, seed=SEED)
        folds.append(
            {
                "held": key,
                "n_target_train": n_tgt,
                "n_eval": int(ev.sum()),
                "target_f1": score(Zte[tr], yte[tr], Zte[ev], yte[ev]),
                "source_matched_f1": score(Ztr[sub], ytr[sub], Zte[ev], yte[ev]),
                "source_full_f1": score(Ztr, ytr, Zte[ev], yte[ev]),
                "floor": floor_of(yte[tr], yte[ev]),
            }
        )
    perm = np.random.default_rng(p).permutation(len(Ztr))
    half = len(perm) // 2
    neg_auc = domain_auc(Ztr[perm[:half]], Ztr[perm[half:]], params, seed=SEED)
    return {"folds": folds, "negative_domain_auc": neg_auc, **realized(y, sp)}


def unit_contest(arm: str) -> dict:
    X, y, sp_r, aux, cfg = _data()
    sp = arm_splits(arm, X, y, sp_r, aux, cfg)
    bp = pca_params()
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    Z = fit_transform_splits(build_preprocessor("pca", n_components=QUBITS, seed=SEED), X, sp)
    ytr, yva, yte = y[sp["train"]], y[sp["val"]], y[sp["test"]]
    sub = stratified_cap(Z["train"], ytr, CAP, seed=SEED)
    f1 = {}
    for m in MODEL_NAMES:
        set_all_seeds(SEED)
        est = build(m, bp[m], SEED)
        fit_model(m, est, Z["train"][sub], ytr[sub])
        f1[m] = _f1(yte, est.predict(Z["test"]))
    f1["qsvm_zz"] = qsvm_f1(bp["qk_bw"], bp["qk_C"], X, y, sp, encoding="zz")
    ry_name = f"qk_angle_ry_{VARIANT}_q{QUBITS}"
    ry_db = (REPO_ROOT / "results" / f"optuna_{ry_name}.db").as_posix()
    qk_ry = optuna.load_study(study_name=ry_name, storage=f"sqlite:///{ry_db}").best_params
    f1["qsvm_angle_ry"] = qsvm_f1(qk_ry["bandwidth"], qk_ry["C"], X, y, sp, encoding="angle_ry")
    sc = AngleScaler().fit(Z["train"])
    va = stratified_cap(Z["val"], yva, cfg_q["qkernel"]["tuning_val_subset"], seed=SEED)
    set_all_seeds(SEED)
    vqc = VQClassifier(
        QUBITS,
        depth=bp["vqc_depth"],
        n_classes=2,
        lr=bp["vqc_lr"],
        batch_size=bp["vqc_batch"],
        max_epochs=cfg_q["vqc"]["max_epochs"],
        patience=cfg_q["vqc"]["patience"],
        seed=SEED,
    )
    vqc.fit(sc.transform(Z["train"][sub]), ytr[sub], sc.transform(Z["val"][va]), yva[va])
    f1["vqc"] = _f1(yte, vqc.predict(sc.transform(Z["test"])))
    best_c = max(MODEL_NAMES, key=lambda m: f1[m])
    best_q = max(("qsvm_zz", "qsvm_angle_ry", "vqc"), key=lambda m: f1[m])
    return {
        "f1": f1,
        "best_classical": best_c,
        "best_quantum": best_q,
        "delta": f1[best_c] - f1[best_q],
        **realized(y, sp),
    }


UNIT_FN = {
    "missingness": unit_missingness,
    "ladder": unit_ladder,
    "framing": unit_framing,
    "contest": unit_contest,
}


def run_unit(spec: dict) -> dict:
    for v in THREAD_VARS:
        if os.environ.get(v) != str(THREADS):
            raise AssertionError(f"worker {os.getpid()}: {v}={os.environ.get(v)} != {THREADS}")
    args = spec["args"]
    out = UNIT_FN[spec["stage"]](*args)
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", mi.rss) / 2**30
    return {**spec, "result": out, "peak_rss_gb": peak}


def specs_for(stage: str, partitions: int) -> list[dict]:
    if stage == "missingness":
        return [
            {"stage": stage, "key": f"M_{a}_{m}", "args": [a, m]}
            for a in ("R", "T0", "T1", "T2")
            for m in ("lgbm", "qsvm")
        ]
    if stage == "ladder":
        arms = ["R", "Rraw"] + [f"{k}{p}" for k in "TYS" for p in range(partitions)]
        return [{"stage": stage, "key": f"L_{a}", "args": [a]} for a in arms]
    arms = [f"{k}{p}" for k in "TY" for p in range(partitions)]
    tag = "F" if stage == "framing" else "C"
    return [{"stage": stage, "key": f"{tag}_{a}", "args": [a]} for a in arms]


# ---------------------------------------------------------------- verdicts ------------------


def verdict_missingness(units: dict) -> dict:
    cells = {k: u["result"]["delta"] for k, u in units.items() if k.startswith("M_")}
    worst = max(abs(v) for v in cells.values())
    branch = (
        "not_load_bearing"
        if worst <= MISS_OK
        else "STOP"
        if worst > MISS_STOP
        else "report_beside_numbers"
    )
    return {"delta_by_cell": cells, "max_abs_delta": worst, "branch": branch}


def verdict_ladder(units: dict, partitions: int, phase1_seed0: float | None) -> dict:
    f = {k[2:]: u["result"]["f1"] for k, u in units.items() if k.startswith("L_")}
    t = [f[f"T{p}"] for p in range(partitions)]
    yy = [f[f"Y{p}"] for p in range(partitions)]
    drop = f["R"] - float(np.mean(t))
    shuffled = {f"S{p}": f[f"S{p}"] - f["R"] for p in range(partitions)}
    shuffle_ok = all(abs(v) <= SHUFFLE_TOL for v in shuffled.values())
    anchor = None
    if phase1_seed0 is not None:
        anchor = {
            "phase1_seed0": phase1_seed0,
            "ladder_R": f["R"],
            "abs_diff": abs(f["R"] - phase1_seed0),
            "pass": abs(f["R"] - phase1_seed0) <= ANCHOR_TOL,
        }
    branch = "PORTS" if drop >= LADDER_PORT else "DOES_NOT_PORT" if drop < LADDER_NOT else "PARTIAL"
    return {
        "f1": f,
        "g_time_mean": float(np.mean(t)),
        "g_time_std": float(np.std(t, ddof=1)),
        "g_type_mean": float(np.mean(yy)),
        "g_type_std": float(np.std(yy, ddof=1)),
        "drop_R_minus_g_time": drop,
        "drop_R_minus_g_type": f["R"] - float(np.mean(yy)),
        "drop_Rraw_minus_R": f["Rraw"] - f["R"],
        "neighbour_share": {
            k[2:]: u["result"]["neighbour_share"]
            for k, u in units.items()
            if k.startswith("L_") and "neighbour_share" in u["result"]
        },
        "shuffled_minus_R": shuffled,
        "shuffled_control_pass": shuffle_ok,
        "anchor": anchor,
        "branch_H_L1": branch if shuffle_ok else "VOID (shuffled control failed)",
    }


def verdict_framing(units: dict, axis: str) -> dict:
    rows = [
        {"partition": k, **fo}
        for k, u in units.items()
        if k.startswith(f"F_{axis}")
        for fo in u["result"]["folds"]
        if "skipped" not in fo
    ]
    if not rows:
        return {"n_folds": 0}
    d = [r["target_f1"] - r["source_matched_f1"] for r in rows]
    neg = {
        k: u["result"]["negative_domain_auc"] for k, u in units.items() if k.startswith(f"F_{axis}")
    }
    return {
        "n_folds": len(rows),
        "n_skipped": sum(
            1
            for k, u in units.items()
            if k.startswith(f"F_{axis}")
            for fo in u["result"]["folds"]
            if "skipped" in fo
        ),
        "target_mean": float(np.mean([r["target_f1"] for r in rows])),
        "source_matched_mean": float(np.mean([r["source_matched_f1"] for r in rows])),
        "source_full_mean": float(np.mean([r["source_full_f1"] for r in rows])),
        "delta_target_minus_matched": float(np.mean(d)),
        "delta_by_fold": d,
        "x3_kill": bool(np.mean(d) > X3_KILL),
        "w3_two_sided_within_0.05": bool(abs(np.mean(d)) <= X3_KILL),
        "negative_domain_auc": neg,
        "negative_control_pass": all(v <= DOMAIN_AUC_MAX for v in neg.values()),
    }


def verdict_contest(units: dict, axis: str) -> dict:
    cells = {k: u["result"] for k, u in units.items() if k.startswith(f"C_{axis}")}
    if not cells:
        return {}
    return {
        "delta_by_partition": {k: c["delta"] for k, c in cells.items()},
        "delta_mean": float(np.mean([c["delta"] for c in cells.values()])),
        "best_by_partition": {
            k: (c["best_classical"], c["best_quantum"]) for k, c in cells.items()
        },
        "family_mean_f1": {
            m: float(np.mean([c["f1"][m] for c in cells.values()]))
            for m in next(iter(cells.values()))["f1"]
        },
    }


# ---------------------------------------------------------------- main ----------------------


def memory_guard(workers: int, rss_gb: float) -> dict:
    avail = psutil.virtual_memory().available / 2**30
    out = {"workers": workers, "worker_rss_gb": rss_gb, "available_gb": avail}
    if workers * rss_gb > MEM_FRACTION * avail:
        raise MemoryError(f"{workers} x {rss_gb:.2f} GB > {MEM_FRACTION:.0%} of {avail:.1f} GB")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stages", nargs="+", choices=STAGES, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--variant", default=VARIANT, help="gas variant, base or derived")
    ap.add_argument("--partitions", type=int, default=5)
    ap.add_argument("--worker-rss-gb", type=float, default=1.25, help="measured probe peak")
    ap.add_argument("--phase1-lgbm-run", default=None, help="run_classical lgbm full run dir")
    ap.add_argument("--resume", default=None, help="earlier run dir; its finished units are kept")
    args = ap.parse_args()
    if args.variant != VARIANT:
        raise SystemExit(f"--variant {args.variant} != module variant {VARIANT}")

    mem = memory_guard(args.workers, args.worker_rss_gb)
    phase1_seed0 = None
    if "ladder" in args.stages:
        if not args.phase1_lgbm_run:
            raise SystemExit("--phase1-lgbm-run is required for the ladder anchor")
        res = json.loads((REPO_ROOT / args.phase1_lgbm_run / "results.json").read_text())
        phase1_seed0 = res["lgbm"]["per_seed"][0]["macro_f1"]

    done: dict[str, dict] = {}
    if args.resume:
        for f in sorted((REPO_ROOT / args.resume / "units").glob("*.json")):
            done[f.stem] = json.loads(f.read_text())
    specs = [s for st in args.stages for s in specs_for(st, args.partitions)]
    todo = [s for s in specs if s["key"] not in done]
    cfg = {**vars(args), "memory": mem, "n_units": len(specs), "resumed": len(specs) - len(todo)}
    with RunDir(f"gas_split_ladder_{VARIANT}", config=cfg, seeds=[SEED]) as run:
        udir = run.path / "units"
        udir.mkdir()
        for k, u in done.items():
            (udir / f"{k}.json").write_text(json.dumps(u, indent=1, default=str))
        failed = []
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as ex:
            futs = {ex.submit(run_unit, s): s for s in todo}
            for fut in as_completed(futs):
                exc = fut.exception()
                if exc is not None:
                    log.error("unit %s failed", futs[fut]["key"], exc_info=exc)
                    failed.append(futs[fut]["key"])
                    continue
                u = fut.result()
                done[u["key"]] = u
                (udir / f"{u['key']}.json").write_text(json.dumps(u, indent=1, default=str))
                log.info("unit %s done (peak %.2f GB)", u["key"], u["peak_rss_gb"])
        if failed:
            raise RuntimeError(f"units failed: {failed} (finished units are in {udir})")

        verdict: dict = {}
        if "missingness" in args.stages:
            verdict["missingness"] = verdict_missingness(done)
        if "ladder" in args.stages:
            verdict["ladder"] = verdict_ladder(done, args.partitions, phase1_seed0)
        if "framing" in args.stages:
            verdict["framing_g_time"] = verdict_framing(done, "T")
            verdict["framing_g_type"] = verdict_framing(done, "Y")
        if "contest" in args.stages:
            verdict["contest_g_time"] = verdict_contest(done, "T")
            verdict["contest_g_type"] = verdict_contest(done, "Y")
        verdict["peak_rss_gb_max"] = max(u["peak_rss_gb"] for u in done.values())
        run.write_json("results.json", {"verdict": verdict, "units": done})
        run.metrics.update({"stages": args.stages})
        log.info("VERDICT %s", json.dumps(verdict, indent=1, default=str))
        print(f"-> {run.path}")
        anchor = verdict.get("ladder", {}).get("anchor")
        if anchor is not None and not anchor["pass"]:
            raise AssertionError(f"ladder R anchor failed: {anchor}")
        if verdict.get("missingness", {}).get("branch") == "STOP":
            sys.exit(2)


if __name__ == "__main__":
    main()
