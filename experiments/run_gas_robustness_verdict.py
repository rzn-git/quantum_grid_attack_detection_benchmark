"""R4-12 robustness verdicts on the gas pipeline: H-C1a, H-C1b, H-C1c and checklist items 2-6.

Pre-registration: decision log 2026-11-16T11:43Z, "PRE-RUN AMENDMENT (R4-12", ruling 6. Reads
the Phase-3 artifacts of one gas variant (`--variant`: `gas_binary`, the main-study pipeline, or
`gas_binary_slog`, pressure log-scaled; every input run must be of that variant), all at PCA-8,
eight qubits, eps 0.5 sigma:
  --adv-run        run_adversarial --only evasion noise poison (10 seeds)
  --random-run     run_adversarial --only random
  --whitebox-run   run_whitebox_qsvm --only p5a (QSVM white-box and the RBF-SVM control)
  --surrogate-run  run_surrogate_control
  --hsj-runs       run_hopskipjump_restarts (one or more, 5 restarts per family)
  --square-run     run_square_blackbox
  --per-class-run  run_evasion_per_class (reported, no branch)

H-C1a  A = QSVM transfer retention (shared protocol) - QSVM white-box retention;
       B = RBF-SVM white-box retention - QSVM white-box retention, paired over attack seeds.
       REPLICATES A >= 0.30 and B >= 0.20 with p < 0.05; KILL B <= 0; else INTERMEDIATE.
H-C1b  the 12 quantum-classical pairs under the shared protocol, Holm-corrected
       (eval.stats.robustness_stats). REPLICATES: significant pairs in both directions;
       KILL: >= 10 of 12 significant and all one direction; else INTERMEDIATE.
H-C1c  the transfer asymmetry under the balanced MLP surrogate. REPLICATES within [0.5, 2];
       NEW FINDING (escalate) outside [0.33, 3]; else INTERMEDIATE. The registered
       statistic is forward_over_reverse, the paper's transfer asymmetry (10.2 -> 0.8 on the
       grid; R4-12 ruling 5, decision log 2026-11-16T13:05Z);
       damage_ratio_vqc_over_classical is reported as secondary under the same lines.
Items  2: the asymmetry moves by a factor >= 2 between the unweighted and balanced
          surrogate; 3: A >= 0.30; 4: the QSVM is top-2 by HSJ distortion and top-2 by Square
          damage; 5: QSVM HSJ coefficient of variation >= 0.15 over restarts, or the family
          ranking changes across restarts; 6: QSVM random-flip rate at 0.5 sigma >= 1.5 x the
          RBF-SVM's.

Run:  uv run python -m experiments.run_gas_robustness_verdict --variant gas_binary \
          --adv-run ... --random-run ... --whitebox-run ... --surrogate-run ... --hsj-runs ... \
          --square-run ... --per-class-run ...
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import yaml

from qgridbench.eval.stats import paired_wilcoxon, robustness_stats
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger

log = get_logger(__name__)

EPS = "0.5"
A_MIN, B_MIN, ALPHA = 0.30, 0.20, 0.05
ASYM_OK, ASYM_NEW = (0.5, 2.0), (1 / 3, 3.0)
ITEM2_FACTOR, ITEM5_CV, ITEM6_RATIO = 2.0, 0.15, 1.5


def assert_variant(rd: str, variant: str) -> None:
    """Every input run must be of the variant the verdict is for: a verdict over runs of two
    pipelines would mix their models."""
    params = yaml.safe_load((REPO_ROOT / rd / "params.yaml").read_text())
    found = params.get("config", params).get("variant")
    if found != variant:
        raise ValueError(f"{rd} is a run of variant {found!r}, not {variant!r}")


def _load(rd: str) -> dict:
    path = REPO_ROOT / rd
    status = json.loads((path / "metrics.json").read_text())["status"]
    if status != "ok":
        raise RuntimeError(f"{rd} finished with status {status}")
    return json.loads((path / "results.json").read_text())


def h_c1a(adv_stats: dict, wb: dict) -> dict:
    p5 = wb["p5a_whitebox_qsvm"]
    q_transfer = adv_stats["per_family"]["qsvm"]["retention"]
    q_wb = p5["whitebox_retention_at_half"]
    r_wb = p5["rbf_control"]["whitebox_retention_at_half"]
    q_seeds = np.array(p5["per_seed_pgd"][EPS]) / p5["clean_f1"]
    r_seeds = np.array(p5["rbf_control"]["per_seed_pgd"][EPS]) / p5["rbf_control"]["clean_f1"]
    test = paired_wilcoxon(r_seeds, q_seeds)
    a, b = q_transfer - q_wb, r_wb - q_wb
    if b <= 0:
        branch = "KILL"
    elif a >= A_MIN and b >= B_MIN and test["p_value"] < ALPHA:
        branch = "REPLICATES"
    else:
        branch = "INTERMEDIATE"
    return {
        "qsvm_transfer_retention": q_transfer,
        "qsvm_whitebox_retention": q_wb,
        "rbf_whitebox_retention": r_wb,
        "A": a,
        "B": b,
        "paired_rbf_vs_qsvm": test,
        "branch": branch,
    }


def h_c1b(adv_stats: dict) -> dict:
    pairs = adv_stats["quantum_vs_classical"]
    sig = {k: v["mean_diff"] for k, v in pairs.items() if v["significant_holm_0.05"]}
    up, down = sum(d > 0 for d in sig.values()), sum(d < 0 for d in sig.values())
    if up and down:
        branch = "REPLICATES"
    elif len(sig) >= 10 and (up == 0 or down == 0):
        branch = "KILL"
    else:
        branch = "INTERMEDIATE"
    return {
        "n_pairs": len(pairs),
        "n_significant": len(sig),
        "quantum_ahead": up,
        "classical_ahead": down,
        "branch": branch,
    }


def _asym_branch(r: float) -> str:
    if ASYM_OK[0] <= r <= ASYM_OK[1]:
        return "REPLICATES"
    if r < ASYM_NEW[0] or r > ASYM_NEW[1]:
        return "NEW_FINDING_escalate"
    return "INTERMEDIATE"


REGISTERED_ASYMMETRY = "forward_over_reverse"
SECONDARY_ASYMMETRY = "damage_ratio_vqc_over_classical"


def h_c1c(sur: dict) -> dict:
    s = sur["summary"]
    out: dict = {"registered": REGISTERED_ASYMMETRY}
    for stat in (REGISTERED_ASYMMETRY, SECONDARY_ASYMMETRY):
        unw, bal = s["mlp"][stat], s["mlp_balanced"][stat]
        factor = max(unw / bal, bal / unw) if unw > 0 and bal > 0 else float("inf")
        out[stat] = {
            "unweighted": unw,
            "balanced": bal,
            "branch_H_C1c": _asym_branch(bal),
            "item2_factor": factor,
            "item2_reproduces": bool(factor >= ITEM2_FACTOR),
        }
    out["branch"] = out[REGISTERED_ASYMMETRY]["branch_H_C1c"]
    return out


def hsj_summary(runs: list[dict]) -> dict:
    per_family: dict[str, list[float]] = {}
    for r in runs:
        for fam, rows in r["per_family"].items():
            if fam in per_family:
                raise ValueError(f"family {fam} appears in more than one HSJ run")
            per_family[fam] = [
                x["median_linf_sigma"] for x in sorted(rows, key=lambda x: x["restart"])
            ]
    n = {len(v) for v in per_family.values()}
    if len(n) != 1:
        raise ValueError(f"unequal restart counts {n}")
    mean = {f: float(np.mean(v)) for f, v in per_family.items()}
    cv = {
        f: float(np.std(v, ddof=1) / np.mean(v)) if np.mean(v) > 0 else None
        for f, v in per_family.items()
    }
    orders = [
        tuple(sorted(per_family, key=lambda f, i=i: -per_family[f][i])) for i in range(n.pop())
    ]
    return {"per_restart": per_family, "mean": mean, "cv": cv, "orders": orders}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for a in ("adv", "random", "whitebox", "surrogate", "square", "per-class"):
        ap.add_argument(f"--{a}-run", required=True)
    ap.add_argument("--hsj-runs", nargs="+", required=True)
    ap.add_argument("--variant", default="gas_binary", help="the gas variant of every input run")
    args = ap.parse_args()
    inputs = [args.adv_run, args.random_run, args.whitebox_run, args.surrogate_run]
    for rd in [*inputs, args.square_run, args.per_class_run, *args.hsj_runs]:
        assert_variant(rd, args.variant)

    adv_stats = robustness_stats(REPO_ROOT / args.adv_run)
    wb, sur = _load(args.whitebox_run), _load(args.surrogate_run)
    rnd = _load(args.random_run)["per_seed"][0]["random_control"]
    sq = _load(args.square_run)["verdict"]
    pc = _load(args.per_class_run)
    hsj = hsj_summary([_load(r) for r in args.hsj_runs])

    c1a, c1b, c1c = h_c1a(adv_stats, wb), h_c1b(adv_stats), h_c1c(sur)
    hsj_rank = sorted(hsj["mean"], key=lambda f: -hsj["mean"][f])
    sq_rank = sq["ordering_by_square_damage"]
    flip = {f: rnd[f]["by_radius"][EPS]["flip_rate"] for f in rnd}
    items = {
        "2_surrogate_specification": c1c[REGISTERED_ASYMMETRY]["item2_reproduces"],
        "3_attack_availability": bool(c1a["A"] >= A_MIN),
        "4_attack_objective": bool(hsj_rank.index("qsvm") < 2 and sq_rank.index("qsvm") < 2),
        "5_starting_point": bool(
            (hsj["cv"]["qsvm"] or 0.0) >= ITEM5_CV or len(set(hsj["orders"])) > 1
        ),
        "6_measurement_geometry": bool(flip["qsvm"] >= ITEM6_RATIO * flip["rbf_svm"]),
    }
    out = {
        "H_C1a": c1a,
        "H_C1b": c1b,
        "H_C1c": c1c,
        "items": items,
        "item_detail": {
            "hsj_rank_most_distortion_first": hsj_rank,
            "square_rank_most_damage_first": sq_rank,
            "hsj": hsj,
            "random_flip_rate_at_0.5": flip,
            "square_controls": sq["controls"],
        },
        "shared_protocol": adv_stats,
        "per_class": pc["summary"],
        "per_class_anchor": pc["anchor"],
    }
    with RunDir(f"gas_robustness_verdict_{args.variant}", config=vars(args)) as run:
        run.write_json("results.json", out)
        run.metrics.update(
            {"H_C1a": c1a["branch"], "H_C1b": c1b["branch"], "H_C1c": c1c["branch"], "items": items}
        )
        log.info(
            "VERDICT %s",
            json.dumps({k: out[k] for k in ("H_C1a", "H_C1b", "H_C1c", "items")}, default=str),
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
