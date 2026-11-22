"""R4-23: does any Table III VQC figure, or the abstract's surrogate asymmetry, move when the
robustness suite attacks Table I's VQC instead of the revoked depth-4 configuration?

Table III's run (`20260813T183448Z_adversarial_binary_q8`) read `results/adv_best_params_binary_q8
.json`, whose VQC is depth 4, learning rate 0.0397 (the deleted custom engine's study); Table I
reports depth 6, learning rate 0.0175 (the PennyLane study `vqc_binary_q8_q8`). Two arms, one unit
per (arm, seed), each in its own process:
  depth4  the grid file as is (the anchor arm)
  depth6  a copy of it with vqc_depth, vqc_lr and vqc_batch from the PennyLane study's best trial
Each unit fits `FittedZoo` (Table III's encoding, qubits and tuned file, read from its params.yaml),
replays `run_adversarial.evasion_and_transfer` on a fresh `spawn_rng(seed, "adv")` stream (Table
III's evasion stage), then `run_surrogate_control.seed_record` on its own fresh stream (the
published surrogate control, `20260815T195529Z_surrogate_control_binary_q8`), and reads every
family's clean predictions per class.

Anchors (blocking, depth-4 arm, every seed run): every leaf of the evasion record (clean, FGSM and
PGD curves at every epsilon, transfer matrix) equals Table III's within 1e-6 for every family; every
surrogate-control F1 and positive rate equals the archived run's within 1e-6, XGBoost's excepted
(reported, see SURROGATE_THREAD_DEPENDENT); with all ten seeds, the depth-4 F, R and F / R equal the
archived summary's (10.16) within 1e-6.

Statistic (decision log 2026-11-16T11:52Z, R4-23): per VQC-involving figure, |depth6 - depth4| of
the seed means: VQC clean F1; VQC white-box FGSM and PGD F1 at the top epsilon; transfer retention
MLP -> VQC (the MLP's white-box set and the MLP-surrogate set) and VQC -> every family (adversarial
F1 mean over the target's clean F1 mean, Table III's retention); the asymmetry difference F - R and
ratio F / R on `run_surrogate_control`'s definition (F the VQC's drop under MLP-crafted examples,
R the MLP's drop under VQC-crafted examples). CONSISTENT when every figure moves by less than 0.05
and the depth-6 ratio lies in [0.5, 2] x the published ratio; MATERIAL otherwise.
Controls: the runner guard on every clean prediction (`FittedZoo.clean_proba`); per-class counts of
every family's clean predictions and of the eval rows.

Run (W x T threads; --threads is applied before numpy loads, in every worker; Table III's anchor
holds at 3 threads, E-NS3a; W x worker peak + the parent's resident set must fit --mem-budget-gb and
80% of free RAM):
  uv run python -m experiments.run_vqc_config_check --smoke --workers 2 --threads 3 \
      --worker-rss-gb 1.2 --mem-budget-gb 2.5
  uv run python -m experiments.run_vqc_config_check --workers 2 --threads 3 --worker-rss-gb 1.2 \
      --mem-budget-gb 2.5
Resume: add --resume results/runs/<stamp>_vqc_config_check_binary_q8 (finished units are skipped).
"""

import argparse
import os
import sys

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int)
_THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
if _THREADS is not None:
    for _var in THREAD_VARS:
        os.environ[_var] = str(_THREADS)
elif __name__ == "__main__" and not {"-h", "--help"} & set(sys.argv):
    raise SystemExit("--threads is required (it is applied before numpy is imported)")

import json  # noqa: E402
import multiprocessing  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402

log = get_logger(__name__)

TABLE_III_RUN = REPO_ROOT / "results" / "runs" / "20260813T183448Z_adversarial_binary_q8"
SURROGATE_RUN = REPO_ROOT / "results" / "runs" / "20260815T195529Z_surrogate_control_binary_q8"
TABLE_I_STUDY = ("vqc_binary_q8_q8", "vqc_binary_q8")  # (study name, Optuna storage stem)
STUDY_TRIALS = 50
ANCHOR_TOL = 1e-6
MOVE_TOL = 0.05
RATIO_BAND = (0.5, 2.0)
ARMS = ("depth4", "depth6")
# The archived surrogate control ran at an unrecorded thread count other than Table III's, and
# XGBoost's hist fit depends on the thread count (decision log 2026-08-15T20:09Z): at Table III's
# 3 threads its XGBoost leaves cannot reproduce, so they are reported beside the anchor, not gated.
# The asymmetry reads only the MLP and the VQC.
SURROGATE_THREAD_DEPENDENT = ("xgb",)
MEM_FRACTION = 0.8


def table_iii_config() -> dict:
    """Table III's encoding, variant, qubits, seeds and tuned file, from its own params.yaml."""
    p = yaml.safe_load((TABLE_III_RUN / "params.yaml").read_text())
    return {**p["config"], "seeds": p["seeds"]}


def table_i_vqc() -> dict:
    """The PennyLane study's best VQC trial, as tuned-file keys (Table I's VQC row)."""
    from experiments.assemble_best_params import _best

    best, prov = _best(*TABLE_I_STUDY, STUDY_TRIALS)
    return {
        "params": {
            "vqc_depth": best["depth"],
            "vqc_lr": best["lr"],
            "vqc_batch": best["batch_size"],
        },
        "provenance": prov,
    }


def arm_params(base: dict, overrides: dict) -> dict:
    """A copy of the tuned file with the arm's VQC keys replaced; every key must already exist."""
    missing = set(overrides) - set(base)
    if missing:
        raise KeyError(f"override keys absent from the tuned file: {sorted(missing)}")
    return {**base, **overrides}


def _json(obj):
    """JSON round trip: float epsilon keys become the strings the archived runs carry."""
    return json.loads(json.dumps(obj))


def _peak_rss_gb() -> float:
    """Peak resident set of this process (Windows peak working set; ru_maxrss elsewhere).

    `run_consistent_relay_attack` carries the same reading, but importing it loads torch and its
    whole chain: 0.5 GB measured in the parent, a fifth of this run's 2.5 GB budget."""
    import psutil

    if sys.platform == "win32":
        return psutil.Process().memory_info().peak_wset / 2**30
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def memory_guard(workers: int, worker_rss_gb: float, budget_gb: float) -> dict:
    """W x worker peak + this process's resident set must fit the budget and 80% of free RAM."""
    import psutil

    parent = psutil.Process().memory_info().rss / 2**30
    avail = psutil.virtual_memory().available / 2**30
    need = workers * worker_rss_gb + parent
    out = {
        "workers": workers,
        "worker_rss_gb": worker_rss_gb,
        "parent_rss_gb": round(parent, 3),
        "need_gb": round(need, 3),
        "budget_gb": budget_gb,
        "available_gb": round(avail, 3),
    }
    log.info("memory %s", out)
    if need > min(budget_gb, MEM_FRACTION * avail):
        raise MemoryError(f"need {need:.2f} GB > budget {budget_gb} or 80% of {avail:.1f} GB free")
    return out


def _init_worker(threads: int):
    import torch

    got = {v: os.environ.get(v) for v in THREAD_VARS}
    if any(x != str(threads) for x in got.values()) or torch.get_num_threads() != threads:
        raise AssertionError(
            f"worker {os.getpid()}: thread caps {got}, torch {torch.get_num_threads()}, "
            f"expected {threads}"
        )


def run_unit(spec: dict) -> dict:
    """One (arm, seed): fit the zoo, Table III's evasion stage, the surrogate control, per class."""
    from experiments.run_adversarial import CLASSICAL, FittedZoo, evasion_and_transfer
    from experiments.run_evasion_per_class import per_class
    from experiments.run_surrogate_control import seed_record
    from qgridbench.models.classical.zoo import stratified_cap
    from qgridbench.protocol import prepare_regime
    from qgridbench.utils.seeding import spawn_rng

    t0 = time.perf_counter()
    s, cfg = spec["seed"], spec["table_iii"]
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    reg = prepare_regime(cfg["variant"], "pca", n_components=cfg["qubits"], seed=0)
    scale = reg.feature_scale
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    X, y = reg.X["test"][ev], reg.y["test"][ev]
    bp = spec["params"]
    zoo = FittedZoo(reg, cfg["encoding"], cfg["qubits"], s, cfg_q["qkernel"]["subset_cap"], bp)
    t_fit = time.perf_counter() - t0

    evasion = evasion_and_transfer(zoo, X, y, cfg_adv, spawn_rng(s, "adv"), scale)
    eps = max(cfg_adv["evasion"]["epsilons"])
    surrogate = seed_record(zoo, bp, X, y, eps, scale, cfg_adv["evasion"]["pgd"], s)
    clean_per_class = {
        f: per_class(y, zoo.clean_proba(f, X, y).argmax(1)) for f in CLASSICAL + ["qsvm", "vqc"]
    }
    rec = {
        "arm": spec["arm"],
        "seed": s,
        "evasion": _json(evasion),
        "surrogate": _json(surrogate),
        "clean_per_class": clean_per_class,
        "vqc_config": {k: bp[k] for k in ("vqc_depth", "vqc_lr", "vqc_batch", "vqc_epochs")},
        "fit_s": round(t_fit, 1),
        "wall_s": round(time.perf_counter() - t0, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
    }
    log.info(
        "%s seed %d | vqc clean %.4f pgd@%s %.4f | %.0fs, %.2f GB",
        spec["arm"],
        s,
        evasion["clean"]["vqc"],
        eps,
        evasion["curves"]["vqc"]["pgd"][eps],
        rec["wall_s"],
        rec["peak_rss_gb"],
    )
    return rec


def leaf_diffs(ours, ref, path: str = "") -> dict:
    """|ours - ref| at every numeric leaf; a missing or extra key fails loud."""
    if isinstance(ref, dict):
        if not isinstance(ours, dict) or set(ours) != set(ref):
            raise KeyError(f"key mismatch at '{path}': {sorted(ours)} vs {sorted(ref)}")
        out = {}
        for k in ref:
            out.update(leaf_diffs(ours[k], ref[k], f"{path}/{k}"))
        return out
    return {path: abs(float(ours) - float(ref))}


def anchor(units: list[dict], reference: dict[int, dict], key: str, exclude=()) -> dict:
    """Every depth-4 unit's `key` record against `reference[seed]`, within ANCHOR_TOL.

    Leaves of a model in `exclude` (the last path component) are reported, not gated."""
    diffs, excluded = {}, {}
    for u in units:
        if u["arm"] != "depth4":
            continue
        for p, d in leaf_diffs(u[key], reference[u["seed"]]).items():
            (excluded if p.rsplit("/", 1)[-1] in exclude else diffs)[f"seed{u['seed']}{p}"] = d
    if not diffs:
        raise ValueError(f"no depth-4 units to anchor '{key}'")
    worst = max(diffs, key=diffs.get)
    out = {
        "n_leaves": len(diffs),
        "max_abs_diff": diffs[worst],
        "worst": worst,
        "tol": ANCHOR_TOL,
        "passed": diffs[worst] <= ANCHOR_TOL,
    }
    if exclude:
        out["excluded_models"] = list(exclude)
        out["excluded_n_leaves"] = len(excluded)
        out["excluded_max_abs_diff"] = max(excluded.values()) if excluded else None
    return out


def asymmetry_anchor(summary: dict, reference: dict) -> dict:
    """The depth-4 arm's F, R and F / R against the archived surrogate control's summary."""
    keys = ("vqc_damage", "reverse_damage", "forward_over_reverse")
    diffs = {k: abs(summary["mlp"][k] - reference["mlp"][k]) for k in keys}
    return {
        "abs_diff": diffs,
        "tol": ANCHOR_TOL,
        "passed": max(diffs.values()) <= ANCHOR_TOL,
    }


def figures(units: list[dict], surrogate_summary: dict) -> dict:
    """The registered VQC-involving figures of one arm, as seed means."""
    ev = [u["evasion"] for u in units]
    top = max(ev[0]["curves"]["vqc"]["pgd"], key=float)

    def mean(fn):
        return float(np.mean([fn(e) for e in ev]))

    clean = {t: mean(lambda e, t=t: e["clean"][t]) for t in ev[0]["clean"]}
    out = {
        "vqc_clean_f1": clean["vqc"],
        f"vqc_whitebox_fgsm_f1@{top}": mean(lambda e: e["curves"]["vqc"]["fgsm"][top]),
        f"vqc_whitebox_pgd_f1@{top}": mean(lambda e: e["curves"]["vqc"]["pgd"][top]),
    }
    for src in ("mlp", "mlp_surrogate"):
        out[f"retention_{src}->vqc"] = (
            mean(lambda e, s=src: e["transfer_matrix"][s]["vqc"]) / clean["vqc"]
        )
    for t in ev[0]["transfer_matrix"]["vqc"]:
        out[f"retention_vqc->{t}"] = mean(lambda e, t=t: e["transfer_matrix"]["vqc"][t]) / clean[t]
    m = surrogate_summary["mlp"]
    out["asymmetry_difference"] = m["vqc_damage"] - m["reverse_damage"]
    out["asymmetry_ratio"] = m["forward_over_reverse"]
    return out


def compare(fig4: dict, fig6: dict, published_ratio: float) -> dict:
    """The registered branch: every figure but the ratio moves < MOVE_TOL; the ratio in band."""
    if set(fig4) != set(fig6):
        raise KeyError("the two arms carry different figures")
    moves = {k: abs(fig6[k] - fig4[k]) for k in fig4 if k != "asymmetry_ratio"}
    lo, hi = RATIO_BAND[0] * published_ratio, RATIO_BAND[1] * published_ratio
    ratio_in_band = lo <= fig6["asymmetry_ratio"] <= hi
    over = sorted(k for k, v in moves.items() if v >= MOVE_TOL)
    return {
        "abs_change": moves,
        "ratio_change": fig6["asymmetry_ratio"] - fig4["asymmetry_ratio"],
        "largest_move": max(moves, key=moves.get),
        "figures_at_or_over_tol": over,
        "ratio_band": [lo, hi],
        "depth6_ratio_in_band": ratio_in_band,
        "branch": "CONSISTENT" if not over and ratio_in_band else "MATERIAL",
    }


def per_class_counts(units: list[dict]) -> dict:
    """Eval-row class counts (one pair across all units) and seed-mean clean per-class rates."""
    pairs = {(r["n_attack"], r["n_natural"]) for u in units for r in u["clean_per_class"].values()}
    if len(pairs) != 1:
        raise AssertionError(f"eval rows differ across units: {sorted(pairs)}")
    fams = units[0]["clean_per_class"]
    return {
        "n_attack_n_natural": list(pairs.pop()),
        "per_family": {
            f: {
                k: float(np.mean([u["clean_per_class"][f][k] for u in units]))
                for k in ("attack_recall", "false_alarm")
            }
            for f in fams
        },
    }


def main() -> None:
    from qgridbench.protocol import best_params_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--worker-rss-gb", type=float, required=True)
    ap.add_argument("--mem-budget-gb", type=float, required=True)
    ap.add_argument("--smoke", action="store_true", help="seed 0 of both arms; not results")
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()

    t3 = table_iii_config()
    bp_path = best_params_path(t3["variant"], t3["qubits"])
    if bp_path.relative_to(REPO_ROOT).as_posix() != t3["best_params"]:
        raise ValueError(f"tuned file {bp_path} is not Table III's {t3['best_params']}")
    base = json.loads(bp_path.read_text())
    # read in a throwaway process: Optuna would stay resident in the parent through the pool
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as ex:
        t1 = ex.submit(table_i_vqc).result()
    params = {"depth4": base, "depth6": arm_params(base, t1["params"])}
    seeds = [0] if args.smoke else t3["seeds"]
    published = json.loads((TABLE_III_RUN / "results.json").read_text())["per_seed"]
    published = {s: r["evasion"] for s, r in zip(t3["seeds"], published, strict=True)}
    sur_pub = json.loads((SURROGATE_RUN / "results.json").read_text())
    sur_seeds = yaml.safe_load((SURROGATE_RUN / "params.yaml").read_text())["seeds"]
    sur_ref = dict(zip(sur_seeds, sur_pub["per_seed"], strict=True))
    published_ratio = sur_pub["summary"]["mlp"]["forward_over_reverse"]

    done = []
    if args.resume:
        done = json.loads((args.resume / "results_partial.json").read_text())["units"]
        log.info("resume from %s: %d finished units", args.resume, len(done))
    have = {(u["arm"], u["seed"]) for u in done}
    specs = [
        {"arm": a, "seed": s, "params": params[a], "table_iii": t3}
        for a in ARMS
        for s in seeds
        if (a, s) not in have
    ]

    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    config = {
        **{k: v for k, v in vars(args).items() if k != "resume"},
        "resume": str(args.resume) if args.resume else None,
        "table_iii_run": TABLE_III_RUN.name,
        "surrogate_run": SURROGATE_RUN.name,
        "table_iii_config": t3,
        "depth6_overrides": t1,
        "branch": branch,
    }
    name = "vqc_config_check_binary_q8" + ("_smoke" if args.smoke else "")
    with RunDir(name, config=config, seeds=seeds) as run:
        (run.path / "git_status_at_launch.txt").write_text(git_status)
        guard = memory_guard(args.workers, args.worker_rss_gb, args.mem_budget_gb)
        units, failed = list(done), []
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(args.threads,),
        ) as ex:
            futs = {ex.submit(run_unit, sp): (sp["arm"], sp["seed"]) for sp in specs}
            for f in as_completed(futs):
                exc = f.exception()
                if exc is not None:
                    log.error("unit %s failed", futs[f], exc_info=exc)
                    failed.append(futs[f])
                    continue
                units.append(f.result())
                run.write_json("results_partial.json", {"units": units})
        if failed:
            raise RuntimeError(f"{len(failed)} units failed (others in results_partial.json)")
        units.sort(key=lambda u: (u["arm"], u["seed"]))
        # loaded only now: the analysis imports would otherwise sit in the parent during the pool
        from experiments.run_surrogate_control import summarize
        from qgridbench.eval.baselines import trivial_baselines
        from qgridbench.eval.stats import robustness_stats
        from qgridbench.models.classical.zoo import stratified_cap
        from qgridbench.protocol import prepare_regime

        anchors = {
            "table_iii": anchor(units, published, "evasion"),
            "surrogate_control": anchor(units, sur_ref, "surrogate", SURROGATE_THREAD_DEPENDENT),
        }
        reg = prepare_regime(t3["variant"], "pca", n_components=t3["qubits"], seed=0)
        cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
        ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
        yte = reg.y["test"][ev]
        floor = trivial_baselines(reg.y["train"], yte)["baseline_stratified_random"]["aggregate"][
            "macro_f1"
        ]["mean"]
        eps = max(cfg_adv["evasion"]["epsilons"])

        arms = {}
        for a in ARMS:
            au = [u for u in units if u["arm"] == a]
            sur = summarize([u["surrogate"] for u in au], floor, eps, float(yte.mean()))
            tdir = run.path / f"table_iii_{a}"
            tdir.mkdir()
            (tdir / "results.json").write_text(
                json.dumps({"per_seed": [{"evasion": u["evasion"]} for u in au]}, indent=2)
            )
            arms[a] = {
                "figures": figures(au, sur),
                "table_iii_stats": robustness_stats(tdir),
                "surrogate_summary": sur,
                "per_class": per_class_counts(au),
                "unit_wall_s": [u["wall_s"] for u in au],
                "unit_peak_rss_gb": [u["peak_rss_gb"] for u in au],
            }
        if [u["seed"] for u in units if u["arm"] == "depth4"] == sur_seeds:
            anchors["asymmetry"] = asymmetry_anchor(
                arms["depth4"]["surrogate_summary"], sur_pub["summary"]
            )
        verdict = compare(arms["depth4"]["figures"], arms["depth6"]["figures"], published_ratio)
        verdict["published_ratio"] = published_ratio
        out = {
            "smoke": args.smoke,
            "anchors": anchors,
            "verdict": verdict,
            "arms": arms,
            "memory_guard": guard,
        }
        run.write_json("results.json", {**out, "units": units})
        run.metrics.update(
            {
                "anchor_table_iii_passed": anchors["table_iii"]["passed"],
                "anchor_surrogate_passed": anchors["surrogate_control"]["passed"],
                "anchor_asymmetry_passed": anchors["asymmetry"]["passed"]
                if "asymmetry" in anchors
                else None,
                "branch": verdict["branch"],
                "max_unit_peak_rss_gb": max(u["peak_rss_gb"] for u in units),
                "max_unit_wall_s": max(u["wall_s"] for u in units),
                "parent_peak_rss_gb": round(_peak_rss_gb(), 3),
            }
        )
        log.info("ANCHORS %s", json.dumps(anchors))
        log.info("VERDICT %s", json.dumps(verdict, indent=1))
        print(f"-> {run.path}")
        bad = [k for k, v in anchors.items() if not v["passed"]]
        if bad:
            raise AssertionError(f"anchor failed: {[(k, anchors[k]['worst']) for k in bad]}")


if __name__ == "__main__":
    main()
