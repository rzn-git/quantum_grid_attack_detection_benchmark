"""NS12: in the physically consistent relay space, does a surrogate's transferred attack land
mostly on its own family? R4-07(b) with random-forest, LightGBM and QSVM surrogates.

Pre-registration: decision log 2026-11-19T08:25Z, "PRE-REGISTRATION (round 5", NS12.

Everything but the surrogate is R4-07(b)'s (`run_tree_surrogate_transfer --arm consistent`):
E-M1c's joint consistent space (four relays, derived map enforced, frame angle unit), its 500
eval rows, its zoo at T = 4, seeds 0-2, eps 0.1 / 0.25 / 0.5, the Square schedule (600
iterations, margin loss) on the surrogate's scores inside the projected box (`square_in_box`),
and E-M1c's random-sign null replayed from its own stream, one draw per eps in E-M1c's order.
Square streams: `spawn_rng(seed, "ns12_square_<surrogate>_<eps>")`; the XGBoost anchor unit
(seed 0) uses R4-07(b)'s `r407b_square_xgb_<eps>` and must reproduce its transfer F1 in every
(eps, family) cell. The XGBoost surrogate's statistic is read from R4-07(b) (3 seeds).

Statistic: E(s -> t, eps) = seed mean of (random F1 - transfer F1). Non-tree targets: QSVM,
RBF-SVM, logistic regression, MLP (E-M1c's depth-4 VQC is reported only). Branch at eps 0.5:
  KILL          the random-forest surrogate moves neither LightGBM nor XGBoost by 0.05, or the
                QSVM surrogate's largest non-self E lands on a tree;
  FAMILY_LOCAL  for each tree surrogate the mean E over the other two trees exceeds the mean E
                over the four non-tree targets and some other tree has E >= 0.05, and the QSVM
                surrogate's largest non-self E lands on a non-tree target;
  PARTIAL       otherwise.
Controls, raised after results.json is written: each surrogate's own Square F1 is below its
random F1 at every eps (seed means); clean and replayed random F1 equal E-M1c's exactly; the
anchor unit equals R4-07(b) within 1e-12; every crafted row satisfies the derived map
(`JointSpace.check` raises inside the unit).

Units: (surrogate, seed) with all three eps, except the QSVM surrogate, split per eps (its
Square makes 602 kernel calls on 500 rows per eps). They run in `--workers` spawned processes of
`--threads` threads (T = 4, E-M1c's, which XGBoost's fit needs), longest first; each finished
unit is written to results_partial.json and `--resume` reuses it.

Run:
    CUDA_VISIBLE_DEVICES=-1 uv run python -m experiments.run_surrogate_family_transfer \
        --workers 6 --threads 4 --worker-rss-gb 1.3
"""

from __future__ import annotations

import os
import sys


def _early_threads(argv: list[str]) -> str | None:
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
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

import experiments.run_consistent_relay_attack as crr  # noqa: E402
from experiments.run_adversarial import FittedZoo, _f1  # noqa: E402
from experiments.run_channel_restricted_attack import (  # noqa: E402
    ENCODING,
    EPS_GRID,
    FAMILIES,
    QUBITS,
    RELAYS,
    _ms,
)
from experiments.run_tree_surrogate_transfer import square_in_box  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger  # noqa: E402
from qgridbench.utils.seeding import spawn_rng  # noqa: E402

log = get_logger(__name__)

RUNS = REPO_ROOT / "results" / "runs"
R407B_RUN = RUNS / "20261116T165641Z_tree_surrogate_transfer_binary_q8_consistent"
EM1C_RUN = crr.FRAME_JOINT_RESULTS
NEW_SURROGATES = ("rf", "lgbm", "qsvm")
TREES = ("rf", "xgb", "lgbm")
NONTREE = ("qsvm", "rbf_svm", "logreg", "mlp")
SEEDS = [0, 1, 2]
MAX_ITER = 600
MARGIN = 0.05
ANCHOR_TOL = 1e-12


def _stream(sur: str, eps: float) -> str:
    return f"r407b_square_xgb_{eps}" if sur == "xgb" else f"ns12_square_{sur}_{eps}"


def unit(spec: dict) -> dict:
    """One (surrogate, seed[, eps]): E-M1c's zoo, Square on the surrogate in the joint
    consistent space at each requested eps, its set and E-M1c's replayed random set scored on
    every family."""
    sh = crr._SHARED
    ctx, seed, sur = sh.ctx, spec["seed"], spec["surrogate"]
    t0 = time.perf_counter()
    zoo = FittedZoo(ctx.reg, ENCODING, QUBITS, seed, ctx.subset_cap, ctx.best_params)
    t_fit = time.perf_counter() - t0
    y = ctx.y
    clean = {f: _f1(y, zoo.clean_proba(f, ctx.Z0, y)) for f in FAMILIES}
    space = crr.JointSpace("consistent", RELAYS, sh, "cpu")
    rng_rand = spawn_rng(seed, f"m1a2_consistent_random_{crr.joint_tag(RELAYS)}")
    records, preds, crafted, checks, timing = [], [], {}, [], {}

    def score(eps, kind, Z):
        for fam in FAMILIES:
            p = zoo.predict_proba(fam, Z)
            records.append(
                {
                    "surrogate": sur,
                    "seed": seed,
                    "eps": eps,
                    "family": fam,
                    "kind": kind,
                    "clean_f1": clean[fam],
                    "attacked_f1": _f1(y, p),
                }
            )
            preds.append(
                {
                    "surrogate": sur,
                    "seed": seed,
                    "eps": eps,
                    "kind": kind,
                    "model": fam,
                    "proba_1": p[:, 1],
                    "label": p.argmax(1),
                }
            )

    for eps in EPS_GRID:  # E-M1c's order: the random stream advances once per eps
        v_rand = space.random_sign(eps, rng_rand)
        if eps not in spec["eps_grid"]:
            continue
        lo, hi = space.box(eps)
        t1 = time.perf_counter()
        v, qps = square_in_box(
            lambda V: zoo.predict_proba(sur, space.Z(space.rows(V))),
            lo,
            hi,
            eps,
            y,
            spawn_rng(seed, _stream(sur, eps)),
            MAX_ITER,
        )
        timing[f"square_{eps}"] = round(time.perf_counter() - t1, 1)
        X = space.rows(v)
        c = space.check(v, X, eps, f"s{seed} square_{sur} eps {eps}")
        checks.append({"eps": eps, "kind": "transfer", "queries": qps} | c)
        Z = space.Z(X)
        crafted[f"{seed}|transfer@{eps}|{sur}"] = Z
        score(eps, "transfer", Z)
        X = space.rows(v_rand)
        c = space.check(v_rand, X, eps, f"s{seed} random eps {eps}")
        checks.append({"eps": eps, "kind": "random"} | c)
        Z = space.Z(X)
        crafted[f"{seed}|random@{eps}|{sur}"] = Z
        score(eps, "random", Z)
        own = {
            k: next(
                r["attacked_f1"]
                for r in records
                if r["eps"] == eps and r["family"] == sur and r["kind"] == k
            )
            for k in ("transfer", "random")
        }
        log.info(
            "%s s%d eps %.2f: own square %.4f random %.4f (%.0f s)",
            sur,
            seed,
            eps,
            own["transfer"],
            own["random"],
            time.perf_counter() - t0,
        )
    return {
        "unit": spec["unit"],
        "surrogate": sur,
        "seed": seed,
        "records": records,
        "preds": preds,
        "crafted": crafted,
        "checks": checks,
        "timing_s": {"zoo_fit": round(t_fit, 1), **timing},
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_rss_gb": round(crr._peak_rss_gb(), 3),
        "pid": os.getpid(),
    }


def build_specs() -> list[dict]:
    specs = []
    for s in SEEDS:  # the QSVM units first: they are the long ones
        for e in EPS_GRID:
            specs.append(
                {"unit": f"qsvm_s{s}_e{e}", "surrogate": "qsvm", "seed": s, "eps_grid": [e]}
            )
    for sur in ("rf", "lgbm"):
        for s in SEEDS:
            specs.append({"unit": f"{sur}_s{s}", "surrogate": sur, "seed": s, "eps_grid": EPS_GRID})
    specs.append({"unit": "xgb_s0_anchor", "surrogate": "xgb", "seed": 0, "eps_grid": EPS_GRID})
    return specs


def read(units: list[dict], em1c: dict, r407b: dict) -> dict:
    recs = [r for u in units for r in u["records"]]
    f1 = {
        (r["surrogate"], r["seed"], r["eps"], r["family"], r["kind"]): r["attacked_f1"]
        for r in recs
    }
    clean = {(r["seed"], r["family"]): r["clean_f1"] for r in recs}
    top = max(EPS_GRID)

    E, matrix = {}, {}
    for sur in NEW_SURROGATES:
        for fam in FAMILIES:
            for eps in EPS_GRID:
                vals = [
                    f1[(sur, s, eps, fam, "random")] - f1[(sur, s, eps, fam, "transfer")]
                    for s in SEEDS
                ]
                matrix[f"{sur}|{fam}|{eps}"] = {
                    "transfer_f1": _ms([f1[(sur, s, eps, fam, "transfer")] for s in SEEDS]),
                    "random_f1": _ms([f1[(sur, s, eps, fam, "random")] for s in SEEDS]),
                    "E": _ms(vals),
                }
                E[(sur, fam, eps)] = float(np.mean(vals))
    for fam in FAMILIES:  # the XGBoost surrogate: R4-07(b), 3 seeds
        for eps in EPS_GRID:
            E[("xgb", fam, eps)] = r407b["statistic"]["E_all"][f"{fam}|{eps}"]
            matrix[f"xgb|{fam}|{eps}"] = {"E_from_r407b": E[("xgb", fam, eps)]}

    per_tree = {}
    for s in TREES:
        others = [t for t in TREES if t != s]
        m_tree = float(np.mean([E[(s, t, top)] for t in others]))
        m_non = float(np.mean([E[(s, t, top)] for t in NONTREE]))
        per_tree[s] = {
            "E_other_trees": {t: E[(s, t, top)] for t in others},
            "E_nontree": {t: E[(s, t, top)] for t in NONTREE},
            "mean_other_trees": m_tree,
            "mean_nontree": m_non,
            "local": bool(m_tree > m_non and max(E[(s, t, top)] for t in others) >= MARGIN),
        }
    q_targets = [t for t in FAMILIES if t not in ("qsvm", "vqc")]
    q_E = {t: E[("qsvm", t, top)] for t in q_targets}
    q_arg = max(q_E, key=q_E.get)
    rf_moves = max(E[("rf", "lgbm", top)], E[("rf", "xgb", top)]) >= MARGIN
    if not rf_moves or q_arg in TREES:
        branch = "KILL"
    elif all(v["local"] for v in per_tree.values()):
        branch = "FAMILY_LOCAL"
    else:
        branch = "PARTIAL"

    positive = {
        sur: {
            str(eps): {
                "own_square_f1": float(
                    np.mean([f1[(sur, s, eps, sur, "transfer")] for s in SEEDS])
                ),
                "own_random_f1": float(np.mean([f1[(sur, s, eps, sur, "random")] for s in SEEDS])),
            }
            for eps in EPS_GRID
        }
        for sur in NEW_SURROGATES
    }
    for sur in positive:
        for cell in positive[sur].values():
            cell["pass"] = bool(cell["own_square_f1"] < cell["own_random_f1"])

    ref = {
        (r["seed"], r["eps"], r["family"], r["kind"]): r
        for r in em1c["records"]
        if r["mode"] == "joint"
    }
    clean_diff = [
        clean[(s, f)] - ref[(s, 0.0, f, "projection_only")]["clean_f1"]
        for s in SEEDS
        for f in FAMILIES
    ]
    random_diff = [
        f1[(sur, s, e, f, "random")] - ref[(s, e, f, "random")]["attacked_f1"]
        for (sur, s, e, f, k) in f1
        if k == "random"
    ]
    r_tr = {
        (r["seed"], r["eps"], r["family"]): r["attacked_f1"]
        for r in r407b["records"]
        if r["kind"] == "transfer_xgb"
    }
    anchor_diff = {
        f"{e}|{f}": f1[("xgb", 0, e, f, "transfer")] - r_tr[(0, e, f)]
        for e in EPS_GRID
        for f in FAMILIES
    }
    controls = {
        "positive": positive,
        "positive_pass": all(c["pass"] for p in positive.values() for c in p.values()),
        "clean_equals_em1c": {
            "max_abs_diff": max(abs(x) for x in clean_diff),
            "pass": all(x == 0.0 for x in clean_diff),
        },
        "random_equals_em1c": {
            "max_abs_diff": max(abs(x) for x in random_diff),
            "n_cells": len(random_diff),
            "pass": all(x == 0.0 for x in random_diff),
        },
        "anchor_xgb_seed0_equals_r407b": {
            "max_abs_diff": max(abs(x) for x in anchor_diff.values()),
            "pass": all(abs(x) <= ANCHOR_TOL for x in anchor_diff.values()),
        },
        "control4_map_every_crafted_row": "JointSpace.check raised on any violation",
    }
    return {
        "eps": top,
        "branch": branch,
        "per_tree_surrogate": per_tree,
        "qsvm_surrogate": {"E_non_self": q_E, "argmax": q_arg},
        "rf_moves_a_tree": bool(rf_moves),
        "E_all": {f"{s}|{t}|{e}": v for (s, t, e), v in E.items()},
        "matrix": matrix,
        "controls": controls,
    }


def save_unit_arrays(run_path: Path, u: dict) -> None:
    """A finished unit's crafted inputs and per-row outputs, so --resume loses nothing."""
    arrays = dict(u["crafted"])
    for p in u["preds"]:
        key = f"{p['seed']}|{p['kind']}@{p['eps']}|{p['model']}"
        arrays[f"{key}|proba_1"] = p["proba_1"]
        arrays[f"{key}|label"] = p["label"]
    np.savez_compressed(run_path / f"unit_{u['unit']}.npz", **arrays)


def _resumed(partial: Path | None) -> list[dict]:
    return json.loads(partial.read_text())["units"] if partial else []


def restore_unit(run_dir: Path, slim: dict) -> dict:
    """A resumed unit with its crafted inputs and per-row outputs read back from its npz."""
    z = np.load(run_dir / f"unit_{slim['unit']}.npz")
    crafted, preds = {}, []
    for k in z.files:
        parts = k.split("|")
        if len(parts) == 3:
            crafted[k] = z[k]
        elif parts[-1] == "proba_1":
            seed, kind_eps, model = parts[0], parts[1], parts[2]
            kind, eps = kind_eps.split("@")
            preds.append(
                {
                    "surrogate": slim["surrogate"],
                    "seed": int(seed),
                    "eps": float(eps),
                    "kind": kind,
                    "model": model,
                    "proba_1": z[k],
                    "label": z["|".join(parts[:-1] + ["label"])],
                }
            )
    return {**slim, "crafted": crafted, "preds": preds}


def write_exports(run_path: Path, units: list[dict], ctx) -> None:
    rows = []
    for u in units:
        for p in u["preds"]:
            rows.append(
                pd.DataFrame(
                    {
                        "surrogate": p["surrogate"],
                        "seed": p["seed"],
                        "eps": p["eps"],
                        "kind": p["kind"],
                        "model": p["model"],
                        "eval_pos": np.arange(len(ctx.y)),
                        "y_true": ctx.y,
                        "proba_1": p["proba_1"],
                        "label": p["label"],
                    }
                )
            )
    pd.concat(rows, ignore_index=True).to_parquet(run_path / "predictions.parquet", index=False)
    crafted = {k: v for u in units for k, v in u["crafted"].items()}
    np.savez_compressed(run_path / "crafted_inputs.npz", **crafted)
    recs = [r for u in units for r in u["records"]]
    pd.DataFrame(recs).to_csv(run_path / "per_seed.csv", index=False)
    streams = sorted({_stream(u["surrogate"], e) for u in units for e in EPS_GRID})
    (run_path / "exports_provenance.json").write_text(
        json.dumps(
            {
                "env": {
                    k: os.environ.get(k)
                    for k in (*THREAD_VARS, "CUDA_VISIBLE_DEVICES", "QGB_VQC_CHUNK")
                },
                "random_streams": streams + ["m1a2_consistent_random_R1+R2+R3+R4 (E-M1c's)"],
                "eval_rows": "E-M1c's 500 rows (eval_pos indexes them in its order)",
                "crafted_inputs": "PCA-8 model inputs Z, keyed seed|kind@eps|surrogate",
                "reference_runs": [EM1C_RUN.name, R407B_RUN.name],
            },
            indent=1,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--worker-rss-gb", type=float, required=True)
    ap.add_argument("--resume", type=Path, help="results_partial.json of an interrupted run")
    ap.add_argument("--units", nargs="*", help="smoke run: only these unit names")
    args = ap.parse_args()
    if str(args.threads) != _THREADS:
        raise AssertionError("early thread parse differs from argparse")
    if args.workers * args.threads > psutil.cpu_count(logical=True):
        raise ValueError("workers x threads exceeds the logical CPU count")
    em1c = json.loads((EM1C_RUN / "results.json").read_text())
    r407b = json.loads((R407B_RUN / "results.json").read_text())
    if args.threads != em1c["config"]["threads"]:
        raise SystemExit(f"NS12 needs E-M1c's T={em1c['config']['threads']} (XGBoost)")
    sh, diag = crr.build_shared(
        SimpleNamespace(smoke=False, angle_unit="frame", magnitude_unit="scale"),
        args.threads,
        RELAYS,
    )
    if len(sh.ctx.y) != em1c["config"]["n_eval"]:
        raise AssertionError("eval rows differ from E-M1c's")
    specs = build_specs()
    if args.units:
        unknown = set(args.units) - {s["unit"] for s in specs}
        if unknown:
            raise SystemExit(f"unknown units: {sorted(unknown)}")
        specs = [s for s in specs if s["unit"] in args.units]
    cfg = {
        "experiment": "NS12",
        "preregistration": "decision log 2026-11-19T08:25Z, PRE-REGISTRATION (round 5)",
        "surrogates": list(NEW_SURROGATES) + ["xgb (seed-0 anchor; statistic from R4-07(b))"],
        "attack": "Square schedule (run_square_blackbox.square_attack), margin loss",
        "max_iter": MAX_ITER,
        "relays": RELAYS,
        "angle_unit": "frame",
        "seeds": SEEDS,
        "eps_grid": EPS_GRID,
        "n_eval": int(len(sh.ctx.y)),
        "threads": args.threads,
        "workers": args.workers,
        "units": [s["unit"] for s in specs],
        "em1c_run": EM1C_RUN.name,
        "r407b_run": R407B_RUN.name,
        "resume": str(args.resume) if args.resume else None,
    }
    name = "surrogate_family_transfer_binary_q8_consistent" + ("_smoke" if args.units else "")
    with RunDir(name, config=cfg, seeds=SEEDS) as run:
        run.write_json("controls_map.json", diag)
        guard = crr.memory_guard(args.workers, args.worker_rss_gb, "NS12")
        done = [restore_unit(args.resume.parent, u) for u in _resumed(args.resume)]
        for u in done:
            save_unit_arrays(run.path, u)
        have = {u["unit"] for u in done}
        todo = [s for s in specs if s["unit"] not in have]
        log.info("units: %d to run, %d resumed", len(todo), len(done))

        def save(res):
            done.append(res)
            slim = [{k: v for k, v in u.items() if k not in ("preds", "crafted")} for u in done]
            run.write_json("results_partial.json", {"units": slim})
            save_unit_arrays(run.path, res)

        if args.workers == 1:
            crr._init_worker(sh)
            for sp in todo:
                save(unit(sp))
        else:
            crr.run_pool_all(unit, todo, sh, args.workers, save)
        if {u["unit"] for u in done} != {s["unit"] for s in specs}:
            raise AssertionError("finished units do not match the planned units")
        if args.units:
            log.info("smoke: %s", json.dumps([u["timing_s"] for u in done]))
            if "xgb_s0_anchor" in args.units:
                r_tr = {
                    (r["eps"], r["family"]): r["attacked_f1"]
                    for r in r407b["records"]
                    if r["kind"] == "transfer_xgb" and r["seed"] == 0
                }
                a = next(u for u in done if u["unit"] == "xgb_s0_anchor")
                diff = max(
                    abs(r["attacked_f1"] - r_tr[(r["eps"], r["family"])])
                    for r in a["records"]
                    if r["kind"] == "transfer"
                )
                log.info("smoke anchor max |diff| vs R4-07(b): %g", diff)
                run.metrics.update({"anchor_max_abs_diff": diff})
            print(f"-> {run.path}")
            return
        out = read(done, em1c, r407b)
        res = {
            "config": cfg,
            "memory_guard": guard,
            **out,
            "units_meta": [
                {k: u[k] for k in ("unit", "timing_s", "wall_s", "peak_rss_gb", "pid")}
                for u in done
            ],
            "checks": {u["unit"]: u["checks"] for u in done},
            "records": [r for u in done for r in u["records"]],
        }
        run.write_json("results.json", res)
        write_exports(run.path, done, sh.ctx)
        ctl = out["controls"]
        run.metrics.update(
            {
                "branch": out["branch"],
                "positive_pass": ctl["positive_pass"],
                "clean_equals_em1c": ctl["clean_equals_em1c"]["pass"],
                "random_equals_em1c": ctl["random_equals_em1c"]["pass"],
                "anchor_pass": ctl["anchor_xgb_seed0_equals_r407b"]["pass"],
                "max_worker_peak_rss_gb": max(u["peak_rss_gb"] for u in done),
            }
        )
        log.info("BRANCH %s", out["branch"])
        log.info("PER_TREE %s", json.dumps(out["per_tree_surrogate"]))
        log.info("QSVM %s", json.dumps(out["qsvm_surrogate"]))
        log.info("CONTROLS %s", json.dumps(ctl))
        print(f"-> {run.path}")
        failed = [
            k
            for k, ok in (
                ("positive", ctl["positive_pass"]),
                ("clean_equals_em1c", ctl["clean_equals_em1c"]["pass"]),
                ("random_equals_em1c", ctl["random_equals_em1c"]["pass"]),
                ("anchor", ctl["anchor_xgb_seed0_equals_r407b"]["pass"]),
            )
            if not ok
        ]
        if failed:
            raise AssertionError(f"NS12 controls failed: {failed} (results.json)")


if __name__ == "__main__":
    main()
