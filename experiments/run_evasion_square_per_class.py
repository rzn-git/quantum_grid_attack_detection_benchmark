"""E-NS3b: per-class evasion under the strongest common attacker (Square schedule, 0.5 sigma).

Replays V2-E2's Square-schedule score-based search (`run_square_blackbox.square_attack`,
artifact `20260901T105256Z_square_blackbox_binary_q8`) at eps = 0.5 sigma for all eight
families and reads it per class: attack-class recall and natural-class false alarm, clean,
under Square and under V2-E2's random control. Evasion is clean minus attacked attack recall.
A secondary arm (new, no anchor) runs Table III's white-box PGD (BCE) at 0.5 sigma against the
RBF-SVM (analytic gradient) and the QSVM (`QSVMWhiteBox`, numerically gated) on the same rows.

Pre-registration: decision log "2026-11-16T09:19Z — PRE-REGISTRATION", section E-NS3b.
Anchor (blocking): each (seed, family) clean, Square and random macro-F1 reproduces V2-E2
within 1e-6. Statistic: each tree's Square evasion against its E-NS3a transfer evasion
(read from `20261116T072807Z_evasion_per_class_binary_q8`); HYPOTHESIS if any tree gains
>= 0.10, KILL if all three are within 0.05, INTERMEDIATE otherwise.

Units are (kind, seed, family): kind "square" for the eight families, "whitebox" for the
RBF-SVM and QSVM. Each unit draws its own stream (`spawn_rng(seed, "square_<m>_0.5")`,
`spawn_rng(seed, "ns3b_wb_<m>")`), so units are independent; a worker keeps one fitted zoo and
refits only when the seed changes (a fit costs ~33 s, most units less).

--smoke runs seed 0 only: Square on logreg and mlp (anchored like the full run) and the whole
white-box arm (RBF-SVM and QSVM), which measures the QSVM white-box cost.

Run (smoke, then full; --worker-rss-gb is the smoke's measured per-worker peak):
  uv run python -m experiments.run_evasion_square_per_class --smoke --workers 2 --threads 5 \
      --worker-rss-gb 1.1
  uv run python -m experiments.run_evasion_square_per_class --workers 2 --threads 5 \
      --worker-rss-gb <peak>
"""

import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int)
THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
if THREADS is None and not {"-h", "--help"} & set(sys.argv):
    raise SystemExit("--threads is required (it is applied before numpy is imported)")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_var] = str(THREADS)

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from experiments.run_adversarial import FittedZoo  # noqa: E402
from experiments.run_consistent_relay_attack import (  # noqa: E402
    _peak_rss_gb,
    memory_guard,
    run_pool_all,
)
from experiments.run_evasion_per_class import per_class  # noqa: E402
from experiments.run_square_blackbox import CONTROL_FRACTION, FAMILIES, square_attack  # noqa: E402
from qgridbench.attacks.evasion import pgd  # noqa: E402
from qgridbench.attacks.gradients import GradProvider  # noqa: E402
from qgridbench.attacks.qsvm_whitebox import QSVMWhiteBox, assert_matches_reference  # noqa: E402
from qgridbench.models.classical.zoo import stratified_cap  # noqa: E402
from qgridbench.protocol import prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402
from qgridbench.utils.seeding import spawn_rng  # noqa: E402

log = get_logger(__name__)

VARIANT, QUBITS, ENCODING = "binary", 8, "zz"
EPS = 0.5  # sigma; formats to "0.5", V2-E2's stream and results key
RUNS = REPO_ROOT / "results" / "runs"
V2E2_RUN = RUNS / "20260901T105256Z_square_blackbox_binary_q8"
NS3A_RUN = RUNS / "20261116T072807Z_evasion_per_class_binary_q8"
RANDOM_CONTROL_SEED_OFFSET = 500  # V2-E2's random control: default_rng(500 + seed)
QSVM_GATE_ROWS = 40  # run_whitebox_qsvm.p5a_whitebox_qsvm's numeric-gate subset
WHITEBOX = ["rbf_svm", "qsvm"]
TREES = ["rf", "xgb", "lgbm"]
ANCHOR_TOL = 1e-6
HYPOTHESIS_GAIN = 0.10
KILL_BAND = 0.05
SMOKE_SQUARE = ["logreg", "mlp"]


# --------------------------------------------------------------------------- worker

_CTX: SimpleNamespace | None = None
_ZOO: dict = {}  # one fitted zoo per worker: {"seed", "zoo", "fit_s"}


def _context() -> SimpleNamespace:
    global _CTX
    if _CTX is None:
        cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
        cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
        best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
        reg = prepare_regime(VARIANT, "pca", n_components=QUBITS, seed=0)
        ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
        _CTX = SimpleNamespace(
            cfg_adv=cfg_adv,
            cfg_q=cfg_q,
            best=best,
            reg=reg,
            X=reg.X["test"][ev],
            y=reg.y["test"][ev],
            scale=reg.feature_scale,
        )
    return _CTX


def _zoo(seed: int) -> tuple[FittedZoo, float]:
    """V2-E2's zoo for this seed; one zoo served every family and eps there, so reuse matches."""
    if _ZOO.get("seed") != seed:
        _ZOO.clear()
        c = _context()
        t0 = time.perf_counter()
        zoo = FittedZoo(c.reg, ENCODING, QUBITS, seed, c.cfg_q["qkernel"]["subset_cap"], c.best)
        _ZOO.update(seed=seed, zoo=zoo, fit_s=time.perf_counter() - t0)
        return zoo, _ZOO["fit_s"]
    return _ZOO["zoo"], 0.0


def _linf_over_eps(Xa: np.ndarray, X: np.ndarray, scale: np.ndarray) -> float:
    r = float((np.abs(Xa - X) / scale).max() / EPS)
    if r > 1.0 + 1e-9:
        raise AssertionError(f"attack left the eps ball: linf/eps = {r}")
    return r


def _whitebox(zoo: FittedZoo, m: str, c: SimpleNamespace):
    """Gradient provider for the secondary arm, built as run_whitebox_qsvm builds it."""
    if m == "rbf_svm":  # fit on the PCA-space subset: same space the attack lives in
        return GradProvider(zoo.models["rbf_svm"], "rbf_svm"), None
    if m == "qsvm":
        svc = zoo.models["qsvm"]
        wb = QSVMWhiteBox(svc, zoo._states_tr, zoo._angle, ENCODING, c.best.get("qk_bw", 1.0))
        gi = stratified_cap(c.reg.X["test"], c.reg.y["test"], QSVM_GATE_ROWS, seed=0)
        gate = assert_matches_reference(wb, svc, zoo._states_tr, zoo._angle, c.reg.X["test"][gi])
        return wb, gate
    raise ValueError(m)


def run_unit(spec: dict) -> dict:
    t0 = time.perf_counter()
    c = _context()
    s, m, kind = spec["seed"], spec["family"], spec["kind"]
    zoo, fit_s = _zoo(s)
    X, y, scale = c.X, c.y, c.scale
    rec = {
        "unit": spec["unit"],
        "kind": kind,
        "seed": s,
        "family": m,
        "clean": per_class(y, zoo.clean_proba(m, X, y).argmax(1)),
    }
    t1 = time.perf_counter()
    if kind == "square":
        Xa, qps = square_attack(
            lambda Z: zoo.predict_proba(m, Z),
            X,
            y,
            EPS,
            scale,
            spawn_rng(s, f"square_{m}_{EPS}"),
            spec["max_iter"],
        )
        u = np.random.default_rng(RANDOM_CONTROL_SEED_OFFSET + s).uniform(-1, 1, X.shape)
        Xr = X + u * EPS * scale
        rec["square"] = per_class(y, zoo.predict_proba(m, Xa).argmax(1))
        rec["random"] = per_class(y, zoo.predict_proba(m, Xr).argmax(1))
        rec["queries_per_sample"] = qps
        rec["linf_over_eps"] = _linf_over_eps(Xa, X, scale)
    else:
        gp, gate = _whitebox(zoo, m, c)
        pgd_cfg = c.cfg_adv["evasion"]["pgd"]
        Xa = pgd(
            gp,
            X,
            y,
            EPS,
            scale,
            n_steps=pgd_cfg["n_steps"],
            step_frac=pgd_cfg["step_frac"],
            rng=spawn_rng(s, f"ns3b_wb_{m}"),
        )
        rec["pgd"] = per_class(y, zoo.predict_proba(m, Xa).argmax(1))
        rec["qsvm_gate"] = gate
        rec["linf_over_eps"] = _linf_over_eps(Xa, X, scale)
    attacked = rec["square"] if kind == "square" else rec["pgd"]
    rec["evasion"] = rec["clean"]["attack_recall"] - attacked["attack_recall"]
    rec["false_alarm_rise"] = attacked["false_alarm"] - rec["clean"]["false_alarm"]
    if kind == "square":
        rec["random_evasion"] = rec["clean"]["attack_recall"] - rec["random"]["attack_recall"]
        rec["random_false_alarm_rise"] = rec["random"]["false_alarm"] - rec["clean"]["false_alarm"]
    rec.update(
        zoo_fit_s=fit_s,
        attack_s=time.perf_counter() - t1,
        wall_s=time.perf_counter() - t0,
        peak_rss_gb=_peak_rss_gb(),
        pid=os.getpid(),
    )
    log.info(
        "%-28s recall %.3f -> %.3f (evasion %+.3f) | FA %+.3f | attack %.0f s, fit %.0f s, "
        "peak %.2f GB",
        spec["unit"],
        rec["clean"]["attack_recall"],
        attacked["attack_recall"],
        rec["evasion"],
        rec["false_alarm_rise"],
        rec["attack_s"],
        fit_s,
        rec["peak_rss_gb"],
    )
    return rec


# --------------------------------------------------------------------------- reading


def anchor(units: list[dict]) -> dict:
    """Clean, Square and random macro-F1 per (seed, family) against V2-E2 at "0.5"."""
    pub = {c["seed"]: c for c in json.loads((V2E2_RUN / "results.json").read_text())["per_seed"]}
    ek = str(EPS)
    diffs = {}
    for u in units:
        s, m = u["seed"], u["family"]
        diffs[f"{u['kind']}|{s}|{m}|clean"] = u["clean"]["macro_f1"] - pub[s]["clean"][m]
        if u["kind"] == "square":
            diffs[f"square|{s}|{m}|square"] = u["square"]["macro_f1"] - pub[s]["square"][ek][m]
            diffs[f"square|{s}|{m}|random"] = u["random"]["macro_f1"] - pub[s]["random"][ek][m]
    worst = max(diffs, key=lambda k: abs(diffs[k]))
    failed = sorted(k for k, v in diffs.items() if abs(v) > ANCHOR_TOL)
    return {
        "reference": V2E2_RUN.name,
        "epsilon_key": ek,
        "n_cells": len(diffs),
        "max_abs_diff": abs(diffs[worst]),
        "worst": worst,
        "failed_cells": {k: diffs[k] for k in failed},
        "failed_families": sorted({k.split("|")[2] for k in failed}),
        "tol": ANCHOR_TOL,
        "passed": not failed,
    }


def _mean(xs) -> float:
    return float(np.mean(xs))


def _seed_means(units, kind, families, states, deltas) -> dict:
    """Per family: seed means of each state's per-class figures and of each delta."""
    out = {}
    for m in families:
        us = sorted(
            (u for u in units if u["kind"] == kind and u["family"] == m), key=lambda u: u["seed"]
        )
        if not us:
            continue
        row = {"seeds": [u["seed"] for u in us]}
        for state in states:
            for k in ("macro_f1", "attack_recall", "false_alarm"):
                row[f"{state}_{k}"] = _mean([u[state][k] for u in us])
        for k in deltas:
            row[k] = _mean([u[k] for u in us])
            row[f"{k}_per_seed"] = [u[k] for u in us]
        out[m] = row
    return out


def summarize(units: list[dict], n_eval: int) -> dict:
    ns3a = json.loads((NS3A_RUN / "results.json").read_text())
    ns3a_fam = ns3a["summary"]["families"]
    sq = _seed_means(
        units,
        "square",
        FAMILIES,
        ("clean", "square", "random"),
        ("evasion", "false_alarm_rise", "random_evasion", "random_false_alarm_rise"),
    )
    wb = _seed_means(units, "whitebox", WHITEBOX, ("clean", "pgd"), ("evasion", "false_alarm_rise"))
    ns3a_wb = {
        m: {
            "whitebox_attack_recall": ns3a_fam[m]["whitebox_attack_recall"],
            "whitebox_false_alarm": ns3a_fam[m]["whitebox_false_alarm"],
            "evasion": ns3a_fam[m]["clean_attack_recall"] - ns3a_fam[m]["whitebox_attack_recall"],
            "false_alarm_rise": ns3a_fam[m]["whitebox_false_alarm"]
            - ns3a_fam[m]["clean_false_alarm"],
        }
        for m in ("mlp", "vqc")
    }

    # ---- controls
    counts = [
        u["clean"]["n_attack"] + u["clean"]["n_natural"] == n_eval
        and all(
            u[k]["n_attack"] + u[k]["n_natural"] == n_eval
            for k in ("square", "random", "pgd")
            if k in u
        )
        for u in units
    ]
    mlp_wb_evasion = ns3a_wb["mlp"]["evasion"]
    positive = None
    if "mlp" in sq:
        req = CONTROL_FRACTION * mlp_wb_evasion
        positive = {
            "mlp_square_evasion": sq["mlp"]["evasion"],
            "ns3a_mlp_whitebox_evasion": mlp_wb_evasion,
            "required": req,
            "pass": bool(sq["mlp"]["evasion"] >= req),
        }
    neg_fail = [m for m in sq if not sq[m]["random_evasion"] < sq[m]["evasion"]]
    controls = {
        "counts_sum_to_n_eval": {
            "n_eval": n_eval,
            "n_attack": sorted({u["clean"]["n_attack"] for u in units}),
            "n_natural": sorted({u["clean"]["n_natural"] for u in units}),
            "pass": bool(all(counts)),
        },
        "positive_mlp": positive,
        "negative_random_below_square": {
            "families": sorted(sq),
            "violations": neg_fail,
            "pass": not neg_fail,
        },
    }
    controls["pass"] = bool(
        controls["counts_sum_to_n_eval"]["pass"]
        and (positive is None or positive["pass"])
        and not neg_fail
    )

    # ---- statistic and branch (trees only; registered thresholds)
    transfer = {t: ns3a_fam[t]["evasion_rate_shared"] for t in TREES}
    matched = {
        t: _mean(
            [
                r["families"][t]["clean"]["attack_recall"]
                - r["families"][t]["shared"]["attack_recall"]
                for r in ns3a["rows"]
                if r["seed"] in sq.get(t, {}).get("seeds", [])
            ]
        )
        for t in TREES
        if t in sq
    }
    statistic = None
    if all(t in sq for t in TREES):
        gain = {t: sq[t]["evasion"] - transfer[t] for t in TREES}
        if any(g >= HYPOTHESIS_GAIN for g in gain.values()):
            branch = "HYPOTHESIS"
        elif all(abs(g) <= KILL_BAND for g in gain.values()):
            branch = "KILL"
        else:
            branch = "INTERMEDIATE"
        statistic = {
            "square_evasion": {t: sq[t]["evasion"] for t in TREES},
            "ns3a_transfer_evasion": transfer,
            "square_minus_transfer": gain,
            "hypothesis_gain": HYPOTHESIS_GAIN,
            "kill_band": KILL_BAND,
            "branch": branch,
            "supplementary_ns3a_transfer_evasion_matched_seeds": matched,
            "supplementary_square_minus_transfer_matched_seeds": {
                t: sq[t]["evasion"] - matched[t] for t in TREES
            },
        }

    ranking = None
    if all(m in sq for m in FAMILIES):
        sq_rec = [sq[m]["square_attack_recall"] for m in FAMILIES]
        ns_rec = [ns3a_fam[m]["shared_attack_recall"] for m in FAMILIES]
        sq_ev = [sq[m]["evasion"] for m in FAMILIES]
        ns_ev = [ns3a_fam[m]["evasion_rate_shared"] for m in FAMILIES]
        ranking = {
            "order_by_square_evasion": sorted(FAMILIES, key=lambda m: -sq[m]["evasion"]),
            "order_by_square_attacked_attack_recall": sorted(
                FAMILIES, key=lambda m: -sq[m]["square_attack_recall"]
            ),
            "ns3a_order_by_attacked_attack_recall": ns3a["summary"][
                "order_by_attacked_attack_recall"
            ],
            "spearman_attacked_recall_square_vs_ns3a": float(spearmanr(sq_rec, ns_rec).statistic),
            "spearman_evasion_square_vs_ns3a": float(spearmanr(sq_ev, ns_ev).statistic),
        }
    qsvm_vs_trees = None
    if "qsvm" in sq and all(t in sq for t in TREES):
        qsvm_vs_trees = {
            "qsvm_square_evasion": sq["qsvm"]["evasion"],
            "qsvm_minus_tree": {t: sq["qsvm"]["evasion"] - sq[t]["evasion"] for t in TREES},
        }
    return {
        "square": sq,
        "whitebox": wb,
        "whitebox_mlp_vqc_from_ns3a": ns3a_wb,
        "controls": controls,
        "statistic": statistic,
        "ranking": ranking,
        "qsvm_vs_trees": qsvm_vs_trees,
        "ns3a_artifact": NS3A_RUN.name,
    }


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, required=True, help="BLAS/OpenMP/torch per worker")
    ap.add_argument("--workers", type=int, required=True, help="parallel units")
    ap.add_argument(
        "--worker-rss-gb", type=float, required=True, help="per-worker peak RSS (smoke measures)"
    )
    ap.add_argument(
        "--smoke", action="store_true", help="seed 0: Square on logreg, mlp; white-box arm"
    )
    ap.add_argument("--resume", type=Path, help="results_partial.json of an interrupted run")
    args = ap.parse_args()
    if args.threads != THREADS:
        raise AssertionError(f"early thread parse {THREADS} != argparse {args.threads}")
    n_cpu = psutil.cpu_count(logical=True)
    if args.workers * args.threads > n_cpu:
        raise ValueError(f"workers x threads = {args.workers * args.threads} > {n_cpu} CPUs")

    v2e2 = yaml.safe_load((V2E2_RUN / "params.yaml").read_text())["config"]
    if EPS not in v2e2["epsilons"] or str(EPS) != "0.5":
        raise AssertionError(f"eps {EPS} not in V2-E2's grid {v2e2['epsilons']}")
    max_iter = v2e2["max_iter"]
    seeds = [0] if args.smoke else v2e2["seeds"]
    square_fams = SMOKE_SQUARE if args.smoke else FAMILIES
    n_eval = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")["eval_subset_adv"]

    specs = [
        {"kind": "square", "seed": s, "family": m, "max_iter": max_iter}
        for s in seeds
        for m in square_fams
    ] + [{"kind": "whitebox", "seed": s, "family": m} for s in seeds for m in WHITEBOX]
    for sp in specs:
        sp["unit"] = f"{sp['kind']}|{sp['seed']}|{sp['family']}"
    # longest first (QSVM Square ~12 min, QSVM white-box next); the rest seed-major so a
    # worker's cached zoo is reused
    specs.sort(key=lambda sp: (sp["family"] != "qsvm", sp["kind"] != "square", sp["seed"]))

    cfg = {
        "variant": VARIANT,
        "qubits": QUBITS,
        "encoding": ENCODING,
        "epsilon": EPS,
        "max_iter": max_iter,
        "seeds": seeds,
        "square_families": square_fams,
        "whitebox_families": WHITEBOX,
        "n_eval": n_eval,
        "threads": args.threads,
        "workers": args.workers,
        "smoke": args.smoke,
        "anchor_artifact": V2E2_RUN.name,
        "ns3a_artifact": NS3A_RUN.name,
        "resume": str(args.resume) if args.resume else None,
        "preregistration": "decision log 2026-11-16T09:19Z PRE-REGISTRATION, E-NS3b",
    }
    name = f"evasion_square_per_class_{VARIANT}_q{QUBITS}" + ("_smoke" if args.smoke else "")
    with RunDir(name, config=cfg, seeds=seeds) as run:
        guard = memory_guard(args.workers, args.worker_rss_gb, name)
        done = json.loads(args.resume.read_text())["units"] if args.resume else []
        have = {u["unit"] for u in done}
        todo = [sp for sp in specs if sp["unit"] not in have]
        log.info("units: %d to run, %d resumed", len(todo), len(done))

        def save(rec):
            done.append(rec)
            run.write_json("results_partial.json", {"units": done})

        # run_pool_all's initializer checks every worker's thread caps (env and torch) == T
        run_pool_all(run_unit, todo, SimpleNamespace(threads=args.threads), args.workers, save)
        units = sorted(done, key=lambda u: u["unit"])
        if {u["unit"] for u in units} != {sp["unit"] for sp in specs}:
            raise AssertionError("finished units do not match the planned units")
        anc = anchor(units)
        summary = summarize(units, n_eval)
        timing = {
            u["unit"]: {k: u[k] for k in ("zoo_fit_s", "attack_s", "wall_s", "peak_rss_gb", "pid")}
            for u in units
        }
        run.write_json(
            "results.json",
            {
                "config": cfg,
                "memory_guard": guard,
                "anchor": anc,
                "controls": summary["controls"],
                "summary": summary,
                "branch": summary["statistic"]["branch"] if summary["statistic"] else None,
                "timing": timing,
                "units": units,
            },
        )
        run.metrics.update(
            {
                "anchor_passed": anc["passed"],
                "anchor_max_abs_diff": anc["max_abs_diff"],
                "controls_passed": summary["controls"]["pass"],
                "branch": summary["statistic"]["branch"] if summary["statistic"] else None,
                "max_peak_rss_gb": max(u["peak_rss_gb"] for u in units),
            }
        )
        log.info("ANCHOR %s", json.dumps(anc))
        log.info("CONTROLS %s", json.dumps(summary["controls"]))
        log.info("STATISTIC %s", json.dumps(summary["statistic"]))
        print(f"-> {run.path}")
        if not anc["passed"]:
            raise AssertionError(
                f"anchor failed in {anc['failed_families']}: {anc['worst']} by "
                f"{anc['max_abs_diff']:.2e}"
            )
        if not summary["controls"]["pass"]:
            raise AssertionError(f"controls failed: {json.dumps(summary['controls'])}")


if __name__ == "__main__":
    main()
