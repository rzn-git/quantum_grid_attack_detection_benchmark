"""R4-22 H1 and H2 on the synthetic FDIA grids (fdia14_binary, fdia118_binary), read from runs.

Pre-registration: decision log 2026-11-16T12:17Z, "PRE-RUN AMENDMENT (R4-22", item 4 (frozen as
the builder proposed; the gas runners `run_gas_accuracy_verdict` and `run_gas_robustness_verdict`
are the templates, and each statistic below takes the gas definition it names).

h1  Phase-1 runs (run_classical PCA-8 and full, run_quantum QSVM and VQC). On PCA-8,
    d = best quantum mean test macro-F1 - best classical (the sign of the registered line):
      HOLDS         d <= +0.01
      KILL          d >= +0.02 and the best quantum model significantly above the best classical
                    one: a seeded model by paired Wilcoxon, Holm over every seeded model against
                    the best classical; a QSVM (deterministic) by a margin over 5 x the best
                    classical model's seed std, as in Table I and the gas runner
      INTERMEDIATE  otherwise
    The ceiling rule (study protocol §8, fixed in advance): the best classical macro-F1 at or
    above 0.99 in either regime is reported as ceiling_reached, which moves the headline weight
    off the accuracy contest whatever the branch.
h2  Phase-2 runs at PCA-8, eight qubits, eps 0.5 sigma. Four components, each with its own
    registered line; no aggregate branch was registered, so none is computed.
      fragility     F = QSVM white-box retention - RBF-SVM white-box retention (P5a and its RBF
                    control; the gas H-C1a B with the sign turned); REPLICATES F <= -0.10, paired
                    Wilcoxon over attack seeds reported
      availability  A = QSVM transfer retention (shared protocol) - QSVM white-box retention
                    (the gas H-C1a A); REPLICATES A >= 0.20
      asymmetry     B = (r - 1) / (r + 1), r = forward_over_reverse under the balanced MLP
                    surrogate (the gas H-C1c registered statistic); REPLICATES |B| <= 0.33, which
                    is r inside [0.5, 2]; the unweighted surrogate is reported beside it
      per-class     the per-class reading's Spearman rho against its 0.9 kill line
                    (run_evasion_per_class branch), reported with its anchor

Run:  uv run python -m experiments.run_fdia_verdicts h1 --variant fdia14_binary --runs <dirs>
      uv run python -m experiments.run_fdia_verdicts h2 --variant fdia14_binary --adv-run <dir> \
          --whitebox-run <dir> --surrogate-run <dir> --per-class-run <dir>
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from qgridbench.eval.stats import holm_bonferroni, paired_wilcoxon, robustness_stats
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger

log = get_logger(__name__)

VARIANTS = ("fdia14_binary", "fdia118_binary")
H1_HOLDS, H1_KILL, QSVM_MARGIN_STDS, ALPHA, CEILING = 0.01, 0.02, 5.0, 0.05, 0.99
F_MAX, A_MIN, B_ABS_MAX, EPS = -0.10, 0.20, 0.33, "0.5"
QUANTUM = ("qsvm_zz", "qsvm_angle_ry", "vqc")
CLASSICAL = ("logreg", "rbf_svm", "rf", "xgb", "lgbm", "mlp")


def _load(rd: str) -> dict:
    path = REPO_ROOT / rd
    status = json.loads((path / "metrics.json").read_text())["status"]
    if status != "ok":
        raise RuntimeError(f"{rd} finished with status {status}")
    return json.loads((path / "results.json").read_text())


def collect(variant: str, run_dirs: list[str]) -> dict[str, dict[str, list[float]]]:
    """{regime: {model: per-seed test macro-F1}} from Phase-1 run directories of one variant."""
    out: dict[str, dict[str, list[float]]] = {"pca8": {}, "full": {}}
    for rd in run_dirs:
        res, name = _load(rd), (REPO_ROOT / rd).name
        if f"_classical_{variant}_" in name:
            regime = "full" if name.endswith("_full") else "pca8"
            items = dict(res)
        elif f"_quantum_qkernel_{variant}_q8" in name:
            regime, items = "pca8", {f"qsvm_{e}": r for e, r in res.items()}
        elif f"_quantum_vqc_{variant}_q8" in name:
            regime, items = "pca8", {"vqc": res["vqc"]}
        else:
            raise ValueError(f"not a Phase-1 {variant} run: {rd}")
        for m, r in items.items():
            if m in out[regime]:
                raise ValueError(f"{m} ({regime}) appears in more than one run")
            out[regime][m] = [s["macro_f1"] for s in r["per_seed"]]
    return out


def h1_verdict(f1: dict[str, list[float]]) -> dict:
    mean = {m: float(np.mean(v)) for m, v in f1.items()}
    std = {m: float(np.std(v, ddof=1)) for m, v in f1.items()}
    best_c = max(CLASSICAL, key=lambda m: mean[m])
    best_q = max(QUANTUM, key=lambda m: mean[m])
    d = mean[best_q] - mean[best_c]
    seeded = [m for m in f1 if m != best_c and not m.startswith("qsvm")]
    tests = {m: paired_wilcoxon(np.array(f1[m]), np.array(f1[best_c])) for m in seeded}
    holm = holm_bonferroni({m: t["p_value"] for m, t in tests.items()})
    if best_q.startswith("qsvm"):
        above = bool(d > QSVM_MARGIN_STDS * std[best_c])
    else:
        above = bool(holm[best_q] < ALPHA and d > 0)
    if d <= H1_HOLDS:
        branch = "HOLDS"
    elif d >= H1_KILL and above:
        branch = "KILL"
    else:
        branch = "INTERMEDIATE"
    return {
        "mean": mean,
        "std": std,
        "best_classical": best_c,
        "best_quantum": best_q,
        "d_quantum_minus_classical": d,
        "wilcoxon_vs_best_classical": tests,
        "holm_p": holm,
        "best_quantum_significantly_above": above,
        "branch_H1": branch,
    }


def fragility(wb: dict) -> dict:
    p5 = wb["p5a_whitebox_qsvm"]
    q, r = p5["whitebox_retention_at_half"], p5["rbf_control"]["whitebox_retention_at_half"]
    q_seeds = np.array(p5["per_seed_pgd"][EPS]) / p5["clean_f1"]
    r_seeds = np.array(p5["rbf_control"]["per_seed_pgd"][EPS]) / p5["rbf_control"]["clean_f1"]
    f = q - r
    return {
        "qsvm_whitebox_retention": q,
        "rbf_whitebox_retention": r,
        "F": f,
        "paired_qsvm_vs_rbf": paired_wilcoxon(q_seeds, r_seeds),
        "branch": "REPLICATES" if f <= F_MAX else "NOT",
    }


def availability(adv_stats: dict, wb: dict) -> dict:
    q_transfer = adv_stats["per_family"]["qsvm"]["retention"]
    q_wb = wb["p5a_whitebox_qsvm"]["whitebox_retention_at_half"]
    a = q_transfer - q_wb
    return {
        "qsvm_transfer_retention": q_transfer,
        "qsvm_whitebox_retention": q_wb,
        "A": a,
        "branch": "REPLICATES" if a >= A_MIN else "NOT",
    }


def asymmetry_b(r: float) -> float:
    return (r - 1.0) / (r + 1.0)


def asymmetry(sur: dict) -> dict:
    s = sur["summary"]
    out = {}
    for surrogate in ("mlp_balanced", "mlp"):
        r = s[surrogate]["forward_over_reverse"]
        b = asymmetry_b(r)
        out[surrogate] = {"forward_over_reverse": r, "B": b, "within": bool(abs(b) <= B_ABS_MAX)}
    out["branch"] = "REPLICATES" if out["mlp_balanced"]["within"] else "NOT"
    return out


def per_class(pc: dict) -> dict:
    s = pc["summary"]
    return {
        "spearman_retention_vs_attack_recall": s["spearman_retention_vs_attack_recall"],
        "branch": s["branch"],
        "anchor": pc["anchor"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("h1")
    a1.add_argument("--variant", choices=VARIANTS, required=True)
    a1.add_argument("--runs", nargs="+", required=True)
    a2 = sub.add_parser("h2")
    a2.add_argument("--variant", choices=VARIANTS, required=True)
    for a in ("adv", "whitebox", "surrogate", "per-class"):
        a2.add_argument(f"--{a}-run", required=True)
    args = ap.parse_args()

    if args.cmd == "h1":
        f1 = collect(args.variant, args.runs)
        missing = [m for m in (*CLASSICAL, *QUANTUM) if m not in f1["pca8"]]
        missing += [f"{m} (full)" for m in CLASSICAL if m not in f1["full"]]
        if missing:
            raise SystemExit(f"Phase-1 runs missing: {missing}")
        v = h1_verdict(f1["pca8"])
        full = {m: float(np.mean(s)) for m, s in f1["full"].items()}
        best_c_any = max(max(full.values()), v["mean"][v["best_classical"]])
        out = {
            "pca8": v,
            "full_mean": full,
            "full_std": {m: float(np.std(s, ddof=1)) for m, s in f1["full"].items()},
            "ceiling_reached": bool(best_c_any >= CEILING),
            "per_seed": f1,
        }
        summary = {"branch_H1": v["branch_H1"], "d": v["d_quantum_minus_classical"]}
        summary["ceiling_reached"] = out["ceiling_reached"]
    else:
        wb, adv_stats = _load(args.whitebox_run), robustness_stats(REPO_ROOT / args.adv_run)
        out = {
            "fragility": fragility(wb),
            "availability": availability(adv_stats, wb),
            "asymmetry": asymmetry(_load(args.surrogate_run)),
            "per_class": per_class(_load(args.per_class_run)),
            "shared_protocol": adv_stats,
        }
        summary = {k: out[k]["branch"] for k in ("fragility", "availability", "asymmetry")}
        summary["per_class"] = out["per_class"]["branch"]
    with RunDir(f"fdia_{args.cmd}_verdict_{args.variant}", config=vars(args)) as run:
        run.write_json("results.json", out)
        run.metrics.update(summary)
        log.info("%s %s %s", args.cmd.upper(), args.variant, json.dumps(summary, default=str))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
