"""R4-07: an XGBoost surrogate attacked with the Square schedule; its crafted set transfers to
every family, in the physically consistent relay space (part b) and in the PCA space (part a).

Pre-registration: decision log "2026-11-16T11:07Z — PRE-REGISTRATION (round 4", R4-07 (part a),
and "Amendment to R4-07" in the 11:25Z entry (part b). The attack is V2-E2's Square-schedule
search (`run_square_blackbox.square_attack`, margin loss, V2-E2's max_iter 600, 602 queries per
sample), run against the XGBoost surrogate's scores.

--arm consistent (part b; NS6 for a tree surrogate). E-M1c's joint attack space
(`run_consistent_relay_attack.JointSpace`, mode consistent, all four relays, 56 controlled
channels, frame angle unit, magnitude floor, deadband, E-M1c's 500 eval rows). Square proposes
moves v on the 56 controlled channels inside the box; each proposal is projected onto the box
(floor, unattackable rows fixed), mapped through the relay maps (`JointSpace.rows`) to a full
raw row, through the frozen pipeline (`JointSpace.Z`) and scored by the zoo's XGBoost (T = 4,
E-M1c's fit). Square on the symmetric eps box with the projection inside the oracle evaluates
exactly the points of Square run on the projected box (`square_in_box`; the test proves it).
The random null is E-M1c's random-sign control replayed from its own stream
(`spawn_rng(seed, "m1a2_consistent_random_R1+R2+R3+R4")`, one draw per eps in E-M1c's order).
Statistic: E(XGB -> t, eps) = seed mean of (random F1 - transfer F1); branch at eps 0.5:
TREE_TRANSFER_SURVIVES if E(XGB -> RF) or E(XGB -> LightGBM) >= 0.05, else
PHYSICS_REMOVES_IT_FOR_TREES_TOO. Every family and eps is reported.
Controls (raised after results.json is written): positive, XGBoost's own Square F1 is below
its random F1 at every eps (seed means); clean F1 per (seed, family) equals E-M1c's exactly;
the replayed random F1 per (seed, eps, family) equals E-M1c's exactly. Control 4 (every
crafted row satisfies the derived map) and control 3 run in the unit through
`JointSpace.check`, which raises.

--arm pca (part a). Table III's eval subset (500 rows) and seeds 0-9 in the PCA-8 space.
The surrogate is XGBoost fitted at V2-E2's OpenMP thread count (10, read from its
compute_env.json, raised for the fit only with threadpoolctl), attacked by Square at eps 0.5
on V2-E2's stream `spawn_rng(seed, "square_xgb_0.5")`; the null is V2-E2's random draw
(`default_rng(500 + seed)`, uniform in the box). The targets are Table III's models: the zoo at
T, with XGBoost refitted at Table III's thread count (3, the count E-NS3a's exact replay used;
every other family is thread-invariant). The VQC target is Table I's configuration (depth 6,
learning rate 0.0175; ruling 2026-11-16T13:33Z), read from
results/adv_best_params_binary_q8_table1.json, which must equal the grid file plus R4-23's
depth-6 overrides. Statistic: per target, the transfer retention (seed mean attacked F1 / seed
mean clean F1, Table III's ratio of means) minus its MLP-surrogate transfer retention: E-NS3a's
exact replay of Table III's shared MLP-crafted set (equal to Table III's retention column for
the six non-differentiable families), and for the VQC R4-23's depth-6 MLP-surrogate retention,
with Table III's depth-4 figures reported beside it. SURROGATE_DEPENDENT if any |change| >= 0.10
or the QSVM loses its retention lead over the classical families; SURROGATE_ROBUST if every
|change| <= 0.05 (and the lead holds); INTERMEDIATE otherwise.
Anchors (raised after results.json is written): the surrogate's clean, Square and random F1
reproduce V2-E2's XGBoost cells within 1e-6 at V2-E2's seeds; every target's clean F1
reproduces Table III's within 1e-6 at every seed, the VQC's R4-23's depth-6 arm's.

Part (b)'s VQC stays the grid file's depth-4 model (E-M1c's): it is a reported target only, and
its clean F1 must equal E-M1c's.

Units are seeds (each fits its own zoo; every stream depends on the seed and eps only), run in
`--workers` processes of `--threads` threads (one worker: the units run in the parent
process); a memory guard (W x measured worker peak within 80% of free RAM, and the total within
the R4-07 budget of 1.3 GB) and --resume.
--smoke runs seed 0 at full size, which is the seed-0 anchor.

Run (thread budget W x T; --threads is read before numpy loads, in every worker):
    uv run python -m experiments.run_tree_surrogate_transfer --arm consistent --smoke \
        --workers 1 --threads 4 --worker-rss-gb 1.25
    uv run python -m experiments.run_tree_surrogate_transfer --arm consistent \
        --workers 1 --threads 4 --worker-rss-gb 1.25
    uv run python -m experiments.run_tree_surrogate_transfer --arm pca --smoke \
        --workers 1 --threads 4 --worker-rss-gb 1.0
    uv run python -m experiments.run_tree_surrogate_transfer --arm pca \
        --workers 1 --threads 4 --worker-rss-gb <smoke peak>
"""

from __future__ import annotations

import os
import sys


def _early_threads(argv: list[str]) -> str | None:
    """--threads, read BEFORE numpy/torch load so the BLAS/OpenMP pools obey it."""
    for i, a in enumerate(argv):
        if a == "--threads":
            return argv[i + 1]
        if a.startswith("--threads="):
            return a.split("=", 1)[1]
    return None


_THREADS = _early_threads(sys.argv)
if _THREADS is None and not {"-h", "--help"} & set(sys.argv):
    raise SystemExit("--threads is required (it is applied before numpy is imported)")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
for _var in THREAD_VARS:
    os.environ[_var] = _THREADS or "1"

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import yaml  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import experiments.run_consistent_relay_attack as crr  # noqa: E402
from experiments.run_adversarial import FittedZoo, _f1  # noqa: E402
from experiments.run_channel_restricted_attack import (  # noqa: E402
    ENCODING,
    EPS_GRID,
    FAMILIES,
    QUBITS,
    RELAYS,
    VARIANT,
    _ms,
)
from experiments.run_square_blackbox import square_attack  # noqa: E402
from experiments.runner_guard import assert_clean_predictor  # noqa: E402
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap  # noqa: E402
from qgridbench.protocol import prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import spawn_rng  # noqa: E402

log = get_logger(__name__)

SURROGATE = "xgb"
RUNS = REPO_ROOT / "results" / "runs"
V2E2_RUN = RUNS / "20260901T105256Z_square_blackbox_binary_q8"
TABLE_III_RUN = RUNS / "20260813T183448Z_adversarial_binary_q8"
NS3A_RUN = RUNS / "20261116T072807Z_evasion_per_class_binary_q8"  # exact replay of Table III
R4_23_RUN = RUNS / "20261116T122057Z_vqc_config_check_binary_q8"  # depth-6 (Table I) VQC arm
GRID_PARAMS = REPO_ROOT / "results" / "adv_best_params_binary_q8.json"
TABLE1_PARAMS = REPO_ROOT / "results" / "adv_best_params_binary_q8_table1.json"
EM1C_RUN = crr.FRAME_JOINT_RESULTS  # E-M1c, T = 4
TABLE_III_XGB_THREADS = 3  # E-NS3a's replay (T = 3) reproduces Table III's XGBoost exactly
RANDOM_CONTROL_SEED_OFFSET = 500  # V2-E2's random null: default_rng(500 + seed)
PCA_EPS = 0.5
PCA_SEEDS = list(range(10))  # Table III's seeds
CONSISTENT_SEEDS = [0, 1, 2]  # E-M1c's seeds
B_TREES = ("rf", "lgbm")  # the amendment's two targets
B_MARGIN = 0.05
A_MOVE, A_ROBUST = 0.10, 0.05
ANCHOR_TOL = 1e-6
MAX_TOTAL_GB = 1.3  # the R4-07 budget (resume brief, 2026-11-16): all its processes together


def square_in_box(proba_of_v, lo, hi, eps, y, rng, max_iter):
    """V2-E2's Square schedule on v in the box [lo, hi], a subset of [-eps, eps], from v = 0.

    square_attack runs on the symmetric box [-eps, eps] (scale 1); the oracle scores the
    projection clip(V, lo, hi). Clipping is elementwise and the box sits inside the symmetric
    one, so every scored point, every accept decision and every RNG draw equal those of Square
    run on [lo, hi] itself; the returned v is the projection of the stored point."""
    if np.any(lo < -eps) or np.any(hi > eps) or np.any(lo > 0) or np.any(hi < 0):
        raise ValueError("the box must hold v = 0 and lie inside [-eps, eps]")
    n, d = lo.shape
    Va, qps = square_attack(
        lambda V: proba_of_v(np.clip(V, lo, hi)),
        np.zeros((n, d)),
        y,
        eps,
        np.ones(d),
        rng,
        max_iter,
    )
    return np.clip(Va, lo, hi), qps


def _seed_mean(units, fn) -> float:
    return float(np.mean([fn(u) for u in units]))


def _peak_rss() -> float:
    return round(crr._peak_rss_gb(), 3)


def memory_guard(workers: int, worker_rss_gb: float, what: str) -> dict:
    """run_consistent_relay_attack's 80% guard, plus the budget's total cap. With one worker the
    units run in this process (see _run), so the total is that process's measured peak; with
    more, W worker peaks plus the parent."""
    out = crr.memory_guard(workers, worker_rss_gb, what)
    parent = psutil.Process().memory_info().rss / 2**30
    total = worker_rss_gb if workers == 1 else workers * worker_rss_gb + parent
    out.update(parent_rss_gb=parent, total_gb=total, max_total_gb=MAX_TOTAL_GB)
    if total > MAX_TOTAL_GB:
        raise MemoryError(f"{what}: {total:.2f} GB > {MAX_TOTAL_GB} GB (R4-07 budget)")
    return out


# --------------------------------------------------------------------------- part (b)


def unit_consistent(spec: dict) -> dict:
    """One seed: E-M1c's zoo at T, Square on its XGBoost in the joint consistent space at each
    eps, the crafted set and E-M1c's replayed random set scored on every family."""
    sh = crr._SHARED
    ctx, seed = sh.ctx, spec["seed"]
    t0 = time.perf_counter()
    zoo = FittedZoo(ctx.reg, ENCODING, QUBITS, seed, ctx.subset_cap, ctx.best_params)
    t_fit = time.perf_counter() - t0
    y = ctx.y
    clean = {f: _f1(y, zoo.clean_proba(f, ctx.Z0, y)) for f in FAMILIES}
    space = crr.JointSpace("consistent", RELAYS, sh, "cpu")
    tag = crr.joint_tag(RELAYS)
    rng_rand = spawn_rng(seed, f"m1a2_consistent_random_{tag}")  # E-M1c's random stream
    records, checks, timing = [], [], {}

    def record(eps, fam, kind, Z):
        records.append(
            {
                "seed": seed,
                "eps": eps,
                "family": fam,
                "kind": kind,
                "clean_f1": clean[fam],
                "attacked_f1": _f1(y, zoo.predict_proba(fam, Z)),
            }
        )

    v0 = np.zeros_like(space.x0c)
    X = space.rows(v0)
    checks.append({"eps": 0.0, "kind": "projection_only"} | space.check(v0, X, 1.0, "proj"))
    for fam in FAMILIES:
        record(0.0, fam, "projection_only", space.Z(X))
    for eps in spec["eps_grid"]:
        lo, hi = space.box(eps)
        t1 = time.perf_counter()
        v, qps = square_in_box(
            lambda V: zoo.predict_proba(SURROGATE, space.Z(space.rows(V))),
            lo,
            hi,
            eps,
            y,
            spawn_rng(seed, f"r407b_square_{SURROGATE}_{eps}"),
            spec["max_iter"],
        )
        timing[f"square_{eps}"] = round(time.perf_counter() - t1, 1)
        X = space.rows(v)
        c = space.check(v, X, eps, f"{tag} s{seed} square_xgb eps {eps}")  # controls 3, 4
        checks.append({"eps": eps, "kind": f"transfer_{SURROGATE}", "queries": qps} | c)
        Z = space.Z(X)
        for fam in FAMILIES:  # the xgb cell is the surrogate's own Square F1
            record(eps, fam, f"transfer_{SURROGATE}", Z)
        v = space.random_sign(eps, rng_rand)
        X = space.rows(v)
        c = space.check(v, X, eps, f"{tag} s{seed} random eps {eps}")
        checks.append({"eps": eps, "kind": "random"} | c)
        for fam in FAMILIES:
            record(eps, fam, "random", space.Z(X))
        log.info(
            "seed %d eps %.2f: xgb square %.4f random %.4f | rf %.4f/%.4f lgbm %.4f/%.4f (%.0f s)",
            seed,
            eps,
            *(
                next(
                    r["attacked_f1"]
                    for r in records
                    if r["eps"] == eps and r["family"] == f and r["kind"] == k
                )
                for f in ("xgb", "rf", "lgbm")
                for k in (f"transfer_{SURROGATE}", "random")
            ),
            time.perf_counter() - t0,
        )
    return {
        "unit": f"s{seed}",
        "seed": seed,
        "records": records,
        "checks": checks,
        "timing_s": {"zoo_fit": round(t_fit, 1), **timing},
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_rss_gb": _peak_rss(),
        "pid": os.getpid(),
    }


def read_consistent(units: list[dict], eps_grid: list[float], em1c: dict) -> dict:
    """The statistic, the branch, the positive control and the two E-M1c identities."""
    units = sorted(units, key=lambda u: u["seed"])
    seeds = [u["seed"] for u in units]
    f1 = {
        (r["seed"], r["eps"], r["family"], r["kind"]): r["attacked_f1"]
        for u in units
        for r in u["records"]
    }
    clean = {(r["seed"], r["family"]): r["clean_f1"] for u in units for r in u["records"]}
    tk = f"transfer_{SURROGATE}"

    def ms(fn):
        return _ms([fn(s) for s in seeds])

    matrix = {}
    for fam in FAMILIES:
        for eps in eps_grid:
            matrix[f"{fam}|{eps}"] = {
                "clean_f1": ms(lambda s, f=fam: clean[(s, f)]),
                "transfer_f1": ms(lambda s, f=fam, e=eps: f1[(s, e, f, tk)]),
                "random_f1": ms(lambda s, f=fam, e=eps: f1[(s, e, f, "random")]),
                "E_transfer_minus_random_damage": ms(
                    lambda s, f=fam, e=eps: f1[(s, e, f, "random")] - f1[(s, e, f, tk)]
                ),
            }
    E = {k: v["E_transfer_minus_random_damage"]["mean"] for k, v in matrix.items()}
    top = max(eps_grid)
    stat = {t: E[f"{t}|{top}"] for t in B_TREES}
    branch = (
        "TREE_TRANSFER_SURVIVES"
        if any(v >= B_MARGIN for v in stat.values())
        else "PHYSICS_REMOVES_IT_FOR_TREES_TOO"
    )
    positive = {
        str(eps): {
            "xgb_square_f1": matrix[f"xgb|{eps}"]["transfer_f1"]["mean"],
            "xgb_random_f1": matrix[f"xgb|{eps}"]["random_f1"]["mean"],
            "pass": bool(
                matrix[f"xgb|{eps}"]["transfer_f1"]["mean"]
                < matrix[f"xgb|{eps}"]["random_f1"]["mean"]
            ),
        }
        for eps in eps_grid
    }
    ref = {
        (r["seed"], r["eps"], r["family"], r["kind"]): r
        for r in em1c["records"]
        if r["mode"] == "joint"
    }
    clean_diff = {
        f"{s}|{f}": clean[(s, f)] - ref[(s, 0.0, f, "projection_only")]["clean_f1"]
        for s in seeds
        for f in FAMILIES
    }
    random_diff = {
        f"{s}|{e}|{f}": f1[(s, e, f, "random")] - ref[(s, e, f, "random")]["attacked_f1"]
        for s in seeds
        for e in eps_grid
        for f in FAMILIES
    }
    proj_diff = {
        f"{s}|{f}": f1[(s, 0.0, f, "projection_only")]
        - ref[(s, 0.0, f, "projection_only")]["attacked_f1"]
        for s in seeds
        for f in FAMILIES
    }
    mlp_e = {
        f"{row['target']}|{row['eps']}": row["transfer_minus_random_damage"]["mean"]
        for row in em1c["transfer_matrix"]
        if row["source"] == "mlp"
    }
    return {
        "seeds": seeds,
        "eps_grid": eps_grid,
        "matrix": matrix,
        "statistic": {
            "eps": top,
            "E_xgb_to": stat,
            "margin": B_MARGIN,
            "E_all_targets_at_top_eps": {f: E[f"{f}|{top}"] for f in FAMILIES},
            "E_all": E,
            "em1c_mlp_source_E_same_cells": {k: mlp_e.get(k) for k in E},
        },
        "branch": branch,
        "complete": seeds == CONSISTENT_SEEDS and list(eps_grid) == EPS_GRID,
        "controls": {
            "positive_xgb_square_beats_random": positive,
            "positive_pass": all(p["pass"] for p in positive.values()),
            "clean_equals_em1c": {
                "max_abs_diff": max(abs(v) for v in clean_diff.values()),
                "n_cells": len(clean_diff),
                "pass": all(v == 0.0 for v in clean_diff.values()),
            },
            "random_equals_em1c": {
                "max_abs_diff": max(abs(v) for v in random_diff.values()),
                "n_cells": len(random_diff),
                "pass": all(v == 0.0 for v in random_diff.values()),
            },
            "projection_only_equals_em1c_recorded": {
                "max_abs_diff": max(abs(v) for v in proj_diff.values()),
                "n_cells": len(proj_diff),
            },
            "control4_map_every_crafted_row": "JointSpace.check raised on any violation",
        },
    }


# --------------------------------------------------------------------------- part (a)

_PCA: SimpleNamespace | None = None


def _pca_context() -> SimpleNamespace:
    global _PCA
    if _PCA is None:
        cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
        cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
        best = json.loads(GRID_PARAMS.read_text())
        best_t1 = json.loads(TABLE1_PARAMS.read_text())
        overrides = yaml.safe_load((R4_23_RUN / "params.yaml").read_text())["config"][
            "depth6_overrides"
        ]["params"]
        check_table1_params(best, best_t1, overrides)
        reg = prepare_regime(VARIANT, "pca", n_components=QUBITS, seed=0)
        ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
        _PCA = SimpleNamespace(
            best=best,
            best_t1=best_t1,
            reg=reg,
            subset_cap=cfg_q["qkernel"]["subset_cap"],
            X=reg.X["test"][ev],
            y=reg.y["test"][ev],
            scale=reg.feature_scale,
        )
    return _PCA


def check_table1_params(grid: dict, table1: dict, overrides: dict) -> None:
    """The Table I file (ruling 2026-11-16T13:33Z) is the grid file with R4-23's depth-6 VQC
    settings and nothing else changed; any other difference raises."""
    if set(grid) != set(table1):
        raise AssertionError(f"Table I params keys differ: {sorted(set(grid) ^ set(table1))}")
    bad = {k: (grid[k], table1[k]) for k in grid if k not in overrides and grid[k] != table1[k]}
    bad |= {k: (v, table1.get(k)) for k, v in overrides.items() if table1.get(k) != v}
    if bad:
        raise AssertionError(f"Table I params differ from grid + R4-23 depth-6 overrides: {bad}")


def _fit_xgb(c: SimpleNamespace, zoo: FittedZoo, seed: int, omp_threads: int):
    """XGBoost on the zoo's subset with the zoo's recipe, its OpenMP pool set to omp_threads
    for the fit (the fitted trees depend on it; n_jobs alone is capped by the pool)."""
    est = build("xgb", c.best.get("xgb", {}), seed)
    with threadpool_limits(limits=omp_threads, user_api="openmp"):
        fit_model("xgb", est, zoo.Xtr, zoo.ytr)
    return est


def unit_pca(spec: dict) -> dict:
    """One seed: Table III's zoo with Table I's VQC (depth 6, ruling 13:33Z; XGBoost at Table
    III's thread count), the 10-thread XGBoost surrogate, Square on it at eps 0.5, its set and
    V2-E2's random set on every family."""
    c, seed = _pca_context(), spec["seed"]
    X, y, scale, eps = c.X, c.y, c.scale, spec["eps"]
    t0 = time.perf_counter()
    zoo = FittedZoo(c.reg, ENCODING, QUBITS, seed, c.subset_cap, c.best_t1)
    zoo.models["xgb"] = _fit_xgb(c, zoo, seed, spec["target_xgb_threads"])
    sur = _fit_xgb(c, zoo, seed, spec["surrogate_threads"])
    t_fit = time.perf_counter() - t0
    p0 = sur.predict_proba(X)
    assert_clean_predictor("xgb_surrogate", y, p0, p0.argmax(1))
    t1 = time.perf_counter()
    rng = spawn_rng(seed, f"square_{SURROGATE}_{eps}")  # V2-E2's stream
    Xa, qps = square_attack(sur.predict_proba, X, y, eps, scale, rng, spec["max_iter"])
    t_sq = time.perf_counter() - t1
    linf = float((np.abs(Xa - X) / scale).max() / eps)
    if linf > 1.0 + 1e-9:
        raise AssertionError(f"Square left the eps ball: linf/eps = {linf}")
    u = np.random.default_rng(RANDOM_CONTROL_SEED_OFFSET + seed).uniform(-1, 1, X.shape)
    Xr = X + u * eps * scale
    out = {
        "unit": f"s{seed}",
        "seed": seed,
        "surrogate": {
            "clean_f1": _f1(y, p0),
            "square_f1": _f1(y, sur.predict_proba(Xa)),
            "random_f1": _f1(y, sur.predict_proba(Xr)),
            "queries_per_sample": qps,
            "linf_over_eps": linf,
        },
        "targets": {},
    }
    for f in FAMILIES:
        out["targets"][f] = {
            "clean_f1": _f1(y, zoo.clean_proba(f, X, y)),
            "transfer_f1": _f1(y, zoo.predict_proba(f, Xa)),
            "random_f1": _f1(y, zoo.predict_proba(f, Xr)),
        }
    s_ = out["surrogate"]
    log.info(
        "seed %d: surrogate clean %.4f square %.4f random %.4f | %s",
        seed,
        s_["clean_f1"],
        s_["square_f1"],
        s_["random_f1"],
        " ".join(
            f"{f}:{v['clean_f1']:.3f}->{v['transfer_f1']:.3f}" for f, v in out["targets"].items()
        ),
    )
    out.update(
        timing_s={"fit": round(t_fit, 1), "square": round(t_sq, 1)},
        wall_s=round(time.perf_counter() - t0, 1),
        peak_rss_gb=_peak_rss(),
        pid=os.getpid(),
    )
    return out


def _table_iii_reference() -> dict:
    """Per family: Table III's per-seed clean F1, its printed retention, and the MLP-surrogate
    transfer retention of E-NS3a's exact replay (the comparator). The VQC is Table I's (ruling
    2026-11-16T13:33Z): its clean F1 per seed and its comparator come from R4-23's depth-6 arm
    (its MLP-surrogate retention), with Table III's depth-4 figures reported beside them."""
    pub = json.loads((TABLE_III_RUN / "results.json").read_text())["per_seed"]
    ns3a = json.loads((NS3A_RUN / "results.json").read_text())
    if not ns3a["anchor"]["passed"]:
        raise AssertionError("E-NS3a's replay of Table III did not pass its anchor")
    fams = ns3a["summary"]["families"]
    out = {}
    for f in FAMILIES:
        c = [p["evasion"]["clean"][f] for p in pub]
        a = [p["evasion"]["curves"][f]["pgd"]["0.5"] for p in pub]
        out[f] = {
            "clean_f1_per_seed": c,
            "table_iii_retention_column": float(np.mean(a) / np.mean(c)),
            "table_iii_column_attack": fams[f]["attack_kind_in_table_iii"],
            "mlp_surrogate_transfer_retention": fams[f]["shared_retention"],
        }
        if fams[f]["attack_kind_in_table_iii"] == "shared":
            gap = abs(out[f]["table_iii_retention_column"] - fams[f]["shared_retention"])
            if gap > 1e-12:
                raise AssertionError(f"{f}: E-NS3a shared retention differs from Table III")
    r423 = json.loads((R4_23_RUN / "results.json").read_text())
    if not all(a["passed"] for a in r423["anchors"].values()):
        raise AssertionError("R4-23 did not pass its anchors")
    d6 = json.loads((R4_23_RUN / "table_iii_depth6" / "results.json").read_text())["per_seed"]
    v4 = out["vqc"]
    out["vqc"] = {
        "clean_f1_per_seed": [p["evasion"]["clean"]["vqc"] for p in d6],
        "mlp_surrogate_transfer_retention": r423["arms"]["depth6"]["figures"][
            "retention_mlp_surrogate->vqc"
        ],
        "comparator_source": f"{R4_23_RUN.name} depth6 retention_mlp_surrogate->vqc",
        "depth4_table_iii_retention_column": v4["table_iii_retention_column"],
        "depth4_table_iii_column_attack": v4["table_iii_column_attack"],
        "depth4_ns3a_shared_retention": v4["mlp_surrogate_transfer_retention"],
        "depth4_r4_23_mlp_surrogate_retention": r423["arms"]["depth4"]["figures"][
            "retention_mlp_surrogate->vqc"
        ],
    }
    return out


def branch_pca(ret: dict, ref: dict) -> dict:
    """Part (a)'s registered branch from per-family retentions under the two surrogates."""
    change = {f: ret[f] - ref[f] for f in ret}
    classical = [f for f in ret if f not in ("qsvm", "vqc")]
    lead_ref = ref["qsvm"] - max(ref[f] for f in classical)
    lead = ret["qsvm"] - max(ret[f] for f in classical)
    lost = lead_ref > 0 and lead <= 0
    if any(abs(v) >= A_MOVE for v in change.values()) or lost:
        verdict = "SURROGATE_DEPENDENT"
    elif all(abs(v) <= A_ROBUST for v in change.values()):
        verdict = "SURROGATE_ROBUST"
    else:
        verdict = "INTERMEDIATE"
    return {
        "change_xgb_minus_mlp_surrogate": change,
        "families_moved_at_least_0.10": sorted(f for f, v in change.items() if abs(v) >= A_MOVE),
        "qsvm_lead_over_classical_mlp_surrogate": lead_ref,
        "qsvm_lead_over_classical_xgb_surrogate": lead,
        "qsvm_lost_lead": bool(lost),
        "verdict": verdict,
    }


def read_pca(units: list[dict], v2e2: dict) -> dict:
    units = sorted(units, key=lambda u: u["seed"])
    seeds = [u["seed"] for u in units]
    ref = _table_iii_reference()
    per_family = {}
    for f in FAMILIES:
        cl = _seed_mean(units, lambda u, f=f: u["targets"][f]["clean_f1"])
        tr = _seed_mean(units, lambda u, f=f: u["targets"][f]["transfer_f1"])
        rd = _seed_mean(units, lambda u, f=f: u["targets"][f]["random_f1"])
        per_family[f] = {
            "clean_f1": cl,
            "transfer_f1": tr,
            "random_f1": rd,
            "transfer_retention": tr / cl,
            "random_retention": rd / cl,
            "E_transfer_minus_random_damage": rd - tr,
            **{k: v for k, v in ref[f].items() if k != "clean_f1_per_seed"},
        }
    ret = {f: per_family[f]["transfer_retention"] for f in FAMILIES}
    ret_ref = {f: per_family[f]["mlp_surrogate_transfer_retention"] for f in FAMILIES}
    sur = {
        k: _seed_mean(units, lambda u, k=k: u["surrogate"][k])
        for k in ("clean_f1", "square_f1", "random_f1")
    }
    ek = str(PCA_EPS)
    v2 = {p["seed"]: p for p in v2e2["per_seed"]}
    anchor_sur = {}
    for u in units:
        s = u["seed"]
        if s in v2:
            p = v2[s]
            anchor_sur[f"{s}|clean"] = u["surrogate"]["clean_f1"] - p["clean"][SURROGATE]
            anchor_sur[f"{s}|square"] = u["surrogate"]["square_f1"] - p["square"][ek][SURROGATE]
            anchor_sur[f"{s}|random"] = u["surrogate"]["random_f1"] - p["random"][ek][SURROGATE]
    anchor_tgt = {
        f"{u['seed']}|{f}": u["targets"][f]["clean_f1"] - ref[f]["clean_f1_per_seed"][u["seed"]]
        for u in units
        for f in FAMILIES
    }
    return {
        "seeds": seeds,
        "eps": PCA_EPS,
        "per_family": per_family,
        "surrogate_seed_means": sur,
        "branch": branch_pca(ret, ret_ref),
        "complete": seeds == PCA_SEEDS,
        "anchors": {
            "surrogate_vs_v2e2": {
                "cells": anchor_sur,
                "max_abs_diff": max((abs(v) for v in anchor_sur.values()), default=None),
                "pass": bool(anchor_sur) and all(abs(v) <= ANCHOR_TOL for v in anchor_sur.values()),
            },
            "targets_clean_vs_table_iii_vqc_vs_r4_23_depth6": {
                "max_abs_diff": max(abs(v) for v in anchor_tgt.values()),
                "failed": {k: v for k, v in anchor_tgt.items() if abs(v) > ANCHOR_TOL},
                "pass": all(abs(v) <= ANCHOR_TOL for v in anchor_tgt.values()),
            },
        },
        "controls": {
            "surrogate_square_below_random": bool(sur["square_f1"] < sur["random_f1"]),
        },
    }


# --------------------------------------------------------------------------- main


def _run(fn, specs, sh, workers, run, resume) -> list[dict]:
    """The seed units not in --resume. One worker runs them in this process, after the pool
    initializer's own thread check (a spawned worker would hold a second copy of the context);
    more run under run_pool_all. sh carries the thread count (part b: the Shared context)."""
    done = json.loads(resume.read_text())["units"] if resume else []
    have = {u["unit"] for u in done}
    todo = [sp for sp in specs if sp["unit"] not in have]
    log.info("units: %d to run, %d resumed", len(todo), len(done))

    def save(res):
        done.append(res)
        run.write_json("results_partial.json", {"units": done})

    if workers == 1:
        crr._init_worker(sh)
        for sp in todo:
            save(fn(sp))
    else:
        crr.run_pool_all(fn, todo, sh, workers, save)
    if {u["unit"] for u in done} != {sp["unit"] for sp in specs}:
        raise AssertionError("finished units do not match the planned units")
    return done


def main_consistent(args, T: int) -> None:
    em1c = json.loads((EM1C_RUN / "results.json").read_text())
    if T != em1c["config"]["threads"]:
        raise SystemExit(f"part (b) needs E-M1c's T={em1c['config']['threads']} (XGBoost)")
    v2e2 = yaml.safe_load((V2E2_RUN / "params.yaml").read_text())["config"]
    seeds = [0] if args.smoke else CONSISTENT_SEEDS
    eps_grid = EPS_GRID
    sh, diag = crr.build_shared(
        SimpleNamespace(smoke=False, angle_unit="frame", magnitude_unit="scale"), T, RELAYS
    )
    if len(sh.ctx.y) != em1c["config"]["n_eval"]:
        raise AssertionError("eval rows differ from E-M1c's")
    cfg = {
        "experiment": "R4-07(b)",
        "arm": "consistent",
        "surrogate": SURROGATE,
        "attack": "Square schedule (run_square_blackbox.square_attack), margin loss",
        "max_iter": v2e2["max_iter"],
        "relays": RELAYS,
        "angle_unit": "frame",
        "seeds": seeds,
        "eps_grid": eps_grid,
        "n_eval": int(len(sh.ctx.y)),
        "threads": T,
        "workers": args.workers,
        "smoke": args.smoke,
        "em1c_run": EM1C_RUN.name,
        "v2e2_run": V2E2_RUN.name,
        "square_stream": "spawn_rng(seed, 'r407b_square_xgb_<eps>')",
        "random_stream": "spawn_rng(seed, 'm1a2_consistent_random_R1+R2+R3+R4') (E-M1c's)",
        "resume": str(args.resume) if args.resume else None,
        "preregistration": "decision log 2026-11-16T11:25Z, Amendment to R4-07 (part b)",
    }
    specs = [
        {"unit": f"s{s}", "seed": s, "eps_grid": eps_grid, "max_iter": v2e2["max_iter"]}
        for s in seeds
    ]
    name = f"tree_surrogate_transfer_{VARIANT}_q{QUBITS}_consistent" + (
        "_smoke" if args.smoke else ""
    )
    with RunDir(name, config=cfg, seeds=seeds) as run:
        run.write_json("controls_map.json", diag)
        guard = memory_guard(args.workers, args.worker_rss_gb, name)
        units = _run(unit_consistent, specs, sh, args.workers, run, args.resume)
        out = read_consistent(units, eps_grid, em1c)
        res = {
            "config": cfg,
            "memory_guard": guard,
            "branch": out["branch"],
            "statistic": out["statistic"],
            "controls": out["controls"],
            "complete": out["complete"],
            "matrix": out["matrix"],
            "units_meta": [
                {k: u[k] for k in ("unit", "timing_s", "wall_s", "peak_rss_gb", "pid")}
                for u in sorted(units, key=lambda u: u["seed"])
            ],
            "checks": {u["unit"]: u["checks"] for u in units},
            "records": [r for u in sorted(units, key=lambda u: u["seed"]) for r in u["records"]],
            "parent_peak_rss_gb": _peak_rss(),
        }
        run.write_json("results.json", res)
        ctl = out["controls"]
        run.metrics.update(
            {
                "branch": out["branch"],
                "E_xgb_to_rf": out["statistic"]["E_xgb_to"]["rf"],
                "E_xgb_to_lgbm": out["statistic"]["E_xgb_to"]["lgbm"],
                "positive_pass": ctl["positive_pass"],
                "clean_equals_em1c": ctl["clean_equals_em1c"]["pass"],
                "random_equals_em1c": ctl["random_equals_em1c"]["pass"],
                "max_worker_peak_rss_gb": max(u["peak_rss_gb"] for u in units),
            }
        )
        log.info("STATISTIC %s", json.dumps(out["statistic"]["E_xgb_to"]))
        log.info("BRANCH %s (complete %s)", out["branch"], out["complete"])
        log.info("CONTROLS %s", json.dumps(ctl))
        print(f"-> {run.path}")
        failed = [
            k
            for k, ok in (
                ("positive", ctl["positive_pass"]),
                ("clean_equals_em1c", ctl["clean_equals_em1c"]["pass"]),
                ("random_equals_em1c", ctl["random_equals_em1c"]["pass"]),
            )
            if not ok
        ]
        if failed:
            raise AssertionError(f"R4-07(b) controls failed: {failed} (results.json)")


def main_pca(args, T: int) -> None:
    v2e2_cfg = yaml.safe_load((V2E2_RUN / "params.yaml").read_text())["config"]
    v2e2_threads = int(json.loads((V2E2_RUN / "compute_env.json").read_text())["omp_num_threads"])
    if PCA_EPS not in v2e2_cfg["epsilons"]:
        raise AssertionError(f"eps {PCA_EPS} not in V2-E2's grid {v2e2_cfg['epsilons']}")
    v2e2 = json.loads((V2E2_RUN / "results.json").read_text())
    seeds = [0] if args.smoke else PCA_SEEDS
    cfg = {
        "experiment": "R4-07(a)",
        "arm": "pca",
        "surrogate": SURROGATE,
        "surrogate_fit_omp_threads": v2e2_threads,
        "target_xgb_fit_omp_threads": TABLE_III_XGB_THREADS,
        "attack": "Square schedule (run_square_blackbox.square_attack), margin loss",
        "max_iter": v2e2_cfg["max_iter"],
        "eps": PCA_EPS,
        "seeds": seeds,
        "threads": T,
        "workers": args.workers,
        "smoke": args.smoke,
        "v2e2_run": V2E2_RUN.name,
        "table_iii_run": TABLE_III_RUN.name,
        "ns3a_run": NS3A_RUN.name,
        "target_params": str(TABLE1_PARAMS.relative_to(REPO_ROOT)),
        "surrogate_params": str(GRID_PARAMS.relative_to(REPO_ROOT)),
        "vqc_reference_run": R4_23_RUN.name,
        "vqc_ruling": "decision log 2026-11-16T13:33Z (R4-23 MATERIAL): Table I VQC, depth 6",
        "square_stream": "spawn_rng(seed, 'square_xgb_0.5') (V2-E2's)",
        "random_stream": "default_rng(500 + seed) uniform (V2-E2's)",
        "resume": str(args.resume) if args.resume else None,
        "preregistration": "decision log 2026-11-16T11:07Z, R4-07 (part a)",
    }
    specs = [
        {
            "unit": f"s{s}",
            "seed": s,
            "eps": PCA_EPS,
            "max_iter": v2e2_cfg["max_iter"],
            "surrogate_threads": v2e2_threads,
            "target_xgb_threads": TABLE_III_XGB_THREADS,
        }
        for s in seeds
    ]
    name = f"tree_surrogate_transfer_{VARIANT}_q{QUBITS}_pca" + ("_smoke" if args.smoke else "")
    with RunDir(name, config=cfg, seeds=seeds) as run:
        guard = memory_guard(args.workers, args.worker_rss_gb, name)
        units = _run(unit_pca, specs, SimpleNamespace(threads=T), args.workers, run, args.resume)
        out = read_pca(units, v2e2)
        res = {
            "config": cfg,
            "memory_guard": guard,
            **out,
            "units": sorted(units, key=lambda u: u["seed"]),
            "parent_peak_rss_gb": _peak_rss(),
        }
        run.write_json("results.json", res)
        anc = out["anchors"]
        run.metrics.update(
            {
                "branch": out["branch"]["verdict"],
                "anchor_surrogate_pass": anc["surrogate_vs_v2e2"]["pass"],
                "anchor_targets_pass": anc["targets_clean_vs_table_iii_vqc_vs_r4_23_depth6"][
                    "pass"
                ],
                "max_worker_peak_rss_gb": max(u["peak_rss_gb"] for u in units),
            }
        )
        log.info("BRANCH %s", json.dumps(out["branch"]))
        log.info("ANCHORS %s", json.dumps(anc))
        print(f"-> {run.path}")
        failed = [k for k, a in anc.items() if not a["pass"]]
        if failed:
            raise AssertionError(f"R4-07(a) anchors failed: {failed} (results.json)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["consistent", "pca"], required=True)
    ap.add_argument("--threads", type=int, required=True, help="BLAS/OpenMP/torch per worker")
    ap.add_argument("--workers", type=int, required=True, help="parallel seed units")
    ap.add_argument(
        "--worker-rss-gb", type=float, required=True, help="per-worker peak RSS (smoke measures)"
    )
    ap.add_argument("--smoke", action="store_true", help="seed 0 at full size (seed-0 anchor)")
    ap.add_argument("--resume", type=Path, help="results_partial.json of an interrupted run")
    args = ap.parse_args()
    if str(args.threads) != _THREADS:
        raise AssertionError(f"early thread parse {_THREADS} != argparse {args.threads}")
    n_cpu = psutil.cpu_count(logical=True)
    if args.workers * args.threads > n_cpu:
        raise ValueError(f"workers x threads = {args.workers * args.threads} > {n_cpu} CPUs")
    if args.arm == "consistent":
        main_consistent(args, args.threads)
    else:
        main_pca(args, args.threads)


if __name__ == "__main__":
    main()
