"""Print the P19 tables from a run_review_claims run directory, plus the P19c re-read.

Reads one finished `experiments.run_review_claims` run directory and renders four blocks:
P19d the surrogate fix (class weighting) and the C1 asymmetry it corrects, P19a the
soft-vote ensembles, P19b the operating point at a validation-fitted false-positive rate,
and P19c a floor-relative re-read of the published robustness table. It computes nothing
new -- every number comes from the run directory or from paper/tables/ -- so it is safe to
re-run at any time.

Run:  python -m experiments.analyze_review_claims results/runs/<stamp>_probe_review_claims_binary_q8

Graduated from tmp/ because the appendix reports its numbers: the floor-relative reading of
the black-box table and the three-surrogate asymmetry comparison are produced here, so the
producer has to be a file the repository carries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from qgridbench.utils.run_tracking import REPO_ROOT

CLASSICAL_TARGETS = ["logreg", "rbf_svm", "rf", "xgb", "lgbm"]
SURROGATES = ["mlp", "mlp_torch_unweighted", "mlp_weighted"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "run_dir",
        type=Path,
        help="a finished run_review_claims run directory (must contain results.json)",
    )
    args = ap.parse_args()
    results = args.run_dir / "results.json"
    if not results.is_file():
        raise SystemExit(f"no results.json in {args.run_dir} -- is this a run_review_claims run?")

    d = json.loads(results.read_text())
    ps = d["per_seed"]
    FL = d["floors"]["evasion_n500"]["stratified_random"]
    FL_BB = d["floors"]["blackbox_n100"]["stratified_random"]
    models = list(ps[0]["f1"]["clean"])
    sets = list(ps[0]["f1"])

    def agg(path):
        v = np.array([path(p) for p in ps], dtype=float)
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)

    print(
        f"floors: evasion n=500 stratified-random macro-F1 = {FL:.4f}; "
        f"black-box n=100 = {FL_BB:.4f}\n"
    )

    # ----------------------------------------------------------------- P19d
    print("=== P19d: the surrogate fix (class weighting) — clean macro-F1 vs the floor ===")
    for m in SURROGATES:
        mu, sd = agg(lambda p, m=m: p["f1"]["clean"][m])
        print(f"  {m:22s} clean {mu:.3f}+-{sd:.3f}   floor-relative {mu - FL:+.3f}")
    mu, sd = agg(lambda p: p["weighted_mlp_val_f1"])
    print(
        f"  weighted MLP val macro-F1 {mu:.4f}+-{sd:.4f} "
        "(sklearn MLP best of 50 Optuna trials: 0.4690)\n"
    )

    print("=== P19d: transfer, three surrogates. drop = own clean - adv ===")
    for src in ["pgd_mlp", "pgd_mlp_torch_unweighted", "pgd_mlp_weighted", "pgd_vqc"]:
        print(f"\n  source: {src}")
        print(f"    {'target':24s}{'adv F1':>16s}{'drop':>16s}{'adv-floor':>12s}")
        for t in models:
            a, asd = agg(lambda p, s=src, t=t: p["f1"][s][t])
            c, _ = agg(lambda p, t=t: p["f1"]["clean"][t])
            print(f"    {t:24s}{a:.3f}+-{asd:.3f}   {c - a:+.3f}          {a - FL:+.3f}")

    print("\n=== P19d: the C1 asymmetry, three ways (VQC vs classical mean) ===")
    for src, self_name in [
        ("pgd_mlp", "mlp"),
        ("pgd_mlp_torch_unweighted", "mlp_torch_unweighted"),
        ("pgd_mlp_weighted", "mlp_weighted"),
    ]:
        vq, _ = agg(lambda p, s=src: p["f1"][s]["vqc"])
        vqc_clean, _ = agg(lambda p: p["f1"]["clean"]["vqc"])
        cls_adv = np.mean([agg(lambda p, s=src, t=t: p["f1"][s][t])[0] for t in CLASSICAL_TARGETS])
        cls_cln = np.mean([agg(lambda p, t=t: p["f1"]["clean"][t])[0] for t in CLASSICAL_TARGETS])
        srcv, _ = agg(lambda p, s=src, n=self_name: p["f1"][s][n])
        # reverse direction: VQC-crafted attack on this surrogate
        rev, _ = agg(lambda p, n=self_name: p["f1"]["pgd_vqc"][n])
        rev_cln, _ = agg(lambda p, n=self_name: p["f1"]["clean"][n])
        print(f"\n  surrogate {self_name}:")
        print(
            f"    VQC adv {vq:.3f} vs classical-mean adv {cls_adv:.3f}   "
            f"(VQC lower by {cls_adv - vq:+.3f})"
        )
        print(
            f"    VQC drop {vqc_clean - vq:.3f} vs classical-mean drop {cls_cln - cls_adv:.3f}  "
            f"(ratio {(vqc_clean - vq) / (cls_cln - cls_adv):.2f}x)"
        )
        print(
            f"    VQC adv {vq:.3f} vs surrogate's own adv {srcv:.3f} "
            f"({'BELOW' if vq < srcv else 'ABOVE'} the surrogate)"
        )
        print(
            f"    forward drop (surrogate->VQC) {vqc_clean - vq:.3f} vs "
            f"reverse (VQC->surrogate) {rev_cln - rev:.3f}  "
            f"-> asymmetry {(vqc_clean - vq) / max(rev_cln - rev, 1e-9):.2f}x"
        )

    # ----------------------------------------------------------------- P19a
    print("\n\n=== P19a: soft-vote ensembles (does a VQC member help?) ===")
    ens = list(ps[0]["ensembles"]["clean"])
    print(f"  {'ensemble':14s}" + "".join(f"{s:>22s}" for s in sets))
    for e in ens:
        row = f"  {e:14s}"
        for s in sets:
            mu, sd = agg(lambda p, s=s, e=e: p["ensembles"][s][e])
            row += f"{mu:>15.3f}+-{sd:.3f}"
        print(row)
    print("\n  paired deltas (with VQC minus without), per attack set:")
    for base, plus in [
        ("cls3", "cls3+vqc"),
        ("cls5", "cls5+vqc"),
        ("cls3", "cls3+qsvm"),
        ("cls5", "cls5+qsvm"),
    ]:
        row = f"    {plus:12s} - {base:6s}: "
        for s in sets:
            dv = np.array([p["ensembles"][s][plus] - p["ensembles"][s][base] for p in ps])
            row += f"{s}={dv.mean():+.3f}+-{dv.std(ddof=1):.3f}  "
        print(row)

    # ----------------------------------------------------------------- P19b
    print("\n\n=== P19b: operating point — TPR and realised FPR at a val-fitted 1% FPR ===")
    for t in ["0.01", "0.05"]:
        print(f"\n  target FPR {t}")
        print(f"    {'model':22s}" + "".join(f"{s:>26s}" for s in sets))
        for m in models:
            row = f"    {m:22s}"
            for s in sets:
                tpr, _ = agg(lambda p, m=m, t=t, s=s: p["operating_point"][m][t][s]["tpr"])
                fpr, _ = agg(lambda p, m=m, t=t, s=s: p["operating_point"][m][t][s]["fpr"])
                row += f"{'TPR ' + format(tpr, '.3f') + ' FPR ' + format(fpr, '.3f'):>26s}"
            print(row)

    # ----------------------------------------------------------------- P19c
    print("\n\n=== P19c: Table II re-read floor-relative (published robustness CSV) ===")
    rows = list(csv.DictReader((REPO_ROOT / "paper/tables/robustness_binary_q8.csv").open()))
    print(
        f"  {'model':10s}{'clean':>8s}{'adv':>8s}{'retention':>11s}"
        f"{'clean-fl':>10s}{'adv-fl':>9s}{'above-floor ret':>17s}{'wb adv-fl':>11s}"
    )
    tab = []
    for r in rows:
        c, a = float(r["clean_f1"]), float(r["adv_f1"])
        wb = r["whitebox_retention"]
        wb_abs = float(wb) * c if wb else None
        tab.append(
            (
                r["model"],
                c,
                a,
                float(r["retention"]),
                c - FL,
                a - FL,
                (a - FL) / (c - FL) if c - FL > 0 else float("nan"),
                (wb_abs - FL) if wb_abs is not None else None,
            )
        )
    for m, c, a, ret, cf, af, afr, wbf in sorted(tab, key=lambda x: -x[3]):
        wbs = f"{wbf:+.3f}" if wbf is not None else "     -"
        print(f"  {m:10s}{c:8.3f}{a:8.3f}{ret:11.3f}{cf:+10.3f}{af:+9.3f}{afr:17.3f}{wbs:>11s}")
    print("\n  ranking by retention      :", [m for m, *_ in sorted(tab, key=lambda x: -x[3])])
    print("  ranking by adv F1 - floor :", [m for m, *_ in sorted(tab, key=lambda x: -x[5])])


if __name__ == "__main__":
    main()
