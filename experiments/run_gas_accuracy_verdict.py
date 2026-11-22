"""R4-12 H-C2: the gas-pipeline accuracy contest at matched dimensionality, read from Phase 1.

Pre-registration: decision log 2026-11-16T11:43Z, "PRE-RUN AMENDMENT (R4-12", ruling 6.
Reads the Phase-1 run directories (run_classical PCA-8 and full, run_quantum QSVM and VQC on
`gas_binary`), and on the PCA-8 regime computes
  delta = best classical mean test macro-F1 - best quantum (max of QSVM-ZZ, QSVM-RY, VQC)
  REPLICATES  delta >= 0.02, the VQC significantly below the best classical model (paired
              Wilcoxon over the 10 seeds, Holm over every seeded model against the best),
              and the QSVM margin above 5 x the best classical model's seed std (the QSVM
              is deterministic and is compared by margin, as in Table I)
  KILL        delta <= 0
  PARITY      0 < delta < 0.02
  INTERMEDIATE delta >= 0.02 without both significance conditions
The full-feature regime is reported beside it, with the ceiling check (macro-F1 >= 0.99).

Run:  uv run python -m experiments.run_gas_accuracy_verdict --runs results/runs/<stamp>_... ...
      (--variant gas_binary_slog reads that derived variant's Phase-1 runs; R5-G1 option B)
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

from qgridbench.eval.stats import holm_bonferroni, paired_wilcoxon
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger

log = get_logger(__name__)

DELTA_REPLICATES = 0.02
QSVM_MARGIN_STDS = 5.0
ALPHA = 0.05
CEILING = 0.99
QUANTUM = ("qsvm_zz", "qsvm_angle_ry", "vqc")
CLASSICAL = ("logreg", "rbf_svm", "rf", "xgb", "lgbm", "mlp")


def collect(run_dirs: list[str], variant: str = "gas_binary") -> dict[str, dict[str, list[float]]]:
    """{regime: {model: per-seed test macro-F1}} from one variant's Phase-1 run directories."""
    out: dict[str, dict[str, list[float]]] = {"pca8": {}, "full": {}}
    for rd in run_dirs:
        path = REPO_ROOT / rd
        status = json.loads((path / "metrics.json").read_text())["status"]
        if status != "ok":
            raise RuntimeError(f"{rd} finished with status {status}")
        res = json.loads((path / "results.json").read_text())
        name = re.sub(r"_\d{1,2}$", "", path.name)  # RunDir's same-second suffix, _2.._99
        if name.endswith(f"_classical_{variant}_full"):
            regime, items = "full", dict(res)
        elif name.endswith(f"_classical_{variant}_pca8"):
            regime, items = "pca8", dict(res)
        elif name.endswith(f"_quantum_qkernel_{variant}_q8"):
            regime, items = "pca8", {f"qsvm_{e}": r for e, r in res.items()}
        elif name.endswith(f"_quantum_vqc_{variant}_q8"):
            regime, items = "pca8", {"vqc": res["vqc"]}
        else:
            raise ValueError(f"not a Phase-1 run of {variant}: {rd}")
        for m, r in items.items():
            if m in out[regime]:
                raise ValueError(f"{m} ({regime}) appears in more than one run")
            out[regime][m] = [s["macro_f1"] for s in r["per_seed"]]
    return out


def verdict(f1: dict[str, list[float]]) -> dict:
    mean = {m: float(np.mean(v)) for m, v in f1.items()}
    std = {m: float(np.std(v, ddof=1)) for m, v in f1.items()}
    best_c = max(CLASSICAL, key=lambda m: mean[m])
    best_q = max(QUANTUM, key=lambda m: mean[m])
    delta = mean[best_c] - mean[best_q]
    seeded = [m for m in f1 if m != best_c and not m.startswith("qsvm")]
    tests = {m: paired_wilcoxon(np.array(f1[best_c]), np.array(f1[m])) for m in seeded}
    holm = holm_bonferroni({m: t["p_value"] for m, t in tests.items()})
    vqc_sig = bool(holm["vqc"] < ALPHA and mean["vqc"] < mean[best_c])
    qsvm_best = max(("qsvm_zz", "qsvm_angle_ry"), key=lambda m: mean[m])
    qsvm_margin = mean[best_c] - mean[qsvm_best]
    qsvm_ok = bool(qsvm_margin > QSVM_MARGIN_STDS * std[best_c])
    if delta <= 0:
        branch = "KILL"
    elif delta < DELTA_REPLICATES:
        branch = "PARITY"
    elif vqc_sig and qsvm_ok:
        branch = "REPLICATES"
    else:
        branch = "INTERMEDIATE"
    return {
        "mean": mean,
        "std": std,
        "best_classical": best_c,
        "best_quantum": best_q,
        "delta": delta,
        "wilcoxon_vs_best_classical": tests,
        "holm_p": holm,
        "vqc_significantly_below": vqc_sig,
        "qsvm_best_encoding": qsvm_best,
        "qsvm_margin": qsvm_margin,
        "qsvm_margin_in_best_classical_stds": qsvm_margin / std[best_c] if std[best_c] else None,
        "qsvm_margin_condition": qsvm_ok,
        "branch_H_C2": branch,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--variant", default="gas_binary", help="gas variant, base or derived")
    args = ap.parse_args()
    f1 = collect(args.runs, args.variant)
    missing = [m for m in (*CLASSICAL, *QUANTUM) if m not in f1["pca8"]]
    missing += [f"{m} (full)" for m in CLASSICAL if m not in f1["full"]]
    if missing:
        raise SystemExit(f"Phase-1 runs missing: {missing}")
    with RunDir(f"gas_accuracy_verdict_{args.variant}", config=vars(args)) as run:
        v = verdict(f1["pca8"])
        full = {m: float(np.mean(s)) for m, s in f1["full"].items()}
        out = {
            "pca8": v,
            "full_mean": full,
            "full_std": {m: float(np.std(s, ddof=1)) for m, s in f1["full"].items()},
            "ceiling_reached": bool(max(full.values()) >= CEILING),
            "per_seed": f1,
        }
        run.write_json("results.json", out)
        run.metrics.update({"branch_H_C2": v["branch_H_C2"], "delta": v["delta"]})
        log.info("H-C2 %s", json.dumps({k: v[k] for k in ("mean", "delta", "branch_H_C2")}))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
