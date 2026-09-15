"""What does the 2,000-sample kernel cap cost the PCA-regime comparison?

Discrepancy that triggered this: an ad-hoc probe scored XGBoost at PCA-8 = 0.7631
using the FULL train split, while the reported PCA-8 table says 0.5968 — same
features, same hyperparameters. The only remaining difference is the training-set
size: the reported runs train on the 2,000-sample stratified subset that exists so
the classical comparison is fair against the O(N^2) quantum kernel.

If the cap is responsible, then the matched-dimensionality regime is depressed by
TWO handicaps (projection + training-set size), the second of which is a protocol
choice rather than a property of the data, and both must be stated when the ~0.60
numbers are reported.

Hypothesis: training-set size dominates; the cap costs more than the PCA projection
does at matched dimensionality.

Kill criterion: if capped and uncapped land within ~0.02, the cap is immaterial and
the discrepancy has another cause that must be found before anything is reported.

Run:  python -m experiments.run_kernel_cap_cost

Graduated from tmp/ because results.tex reports its numbers: the 2,000-sample cap costs
-0.197 macro-F1, and the PCA-8 XGBoost learning curve is still climbing at the full split
(0.595 at n=2,000 to 0.763 at n=47,020). It writes a RunDir under results/runs/ like every
other runner, so those numbers carry provenance -- git sha, lock hash, resolved config, wall
time -- instead of landing in gitignored scratch.
"""

from __future__ import annotations

import json

from qgridbench.eval.metrics import compute_all
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.protocol import kernel_subset, prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)


def score(model, Xtr, ytr, Xte, yte, params, seed=0):
    set_all_seeds(seed)
    est = build(model, params, seed)
    fit_model(model, est, Xtr, ytr)
    return compute_all(yte, est.predict_proba(Xte))["macro_f1"]


def main() -> None:
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    cap = cfg_q["qkernel"]["subset_cap"]
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())

    dims = (8, 12, 16)
    models = ("xgb", "lgbm")
    curve_sizes = (500, 1000, 2000, 5000, 10000, 20000)

    with RunDir(
        "probe_kernel_cap_cost_binary",
        config={
            "variant": "binary",
            "subset_cap": cap,
            "pca_dims": list(dims),
            "models": list(models),
            "curve_sizes": [*curve_sizes, "full"],
        },
        seeds=[0],
    ) as run:
        out: dict = {}
        print(f"{'regime':>8} {'model':>7} {'capped(2k)':>12} {'full-train':>12} {'cap cost':>10}")
        for d in dims:
            reg = prepare_regime("binary", "pca", n_components=d, seed=0)
            sub = kernel_subset(reg, cap, seed=0)
            Xte, yte = reg.X["test"], reg.y["test"]
            for model in models:
                p = best.get(model, {})
                capped = score(model, reg.X["train"][sub], reg.y["train"][sub], Xte, yte, p)
                full = score(model, reg.X["train"], reg.y["train"], Xte, yte, p)
                out[f"pca{d}_{model}"] = {
                    "capped": capped,
                    "full_train": full,
                    "cap_cost": full - capped,
                    "n_capped": int(len(sub)),
                    "n_full": int(len(reg.y["train"])),
                }
                run.write_json("results_partial.json", out)
                log.info(
                    "pca%d %s: capped %.4f, full %.4f, cap cost %+.4f",
                    d,
                    model,
                    capped,
                    full,
                    full - capped,
                )
                print(
                    f"{'pca' + str(d):>8} {model:>7} {capped:>12.4f} {full:>12.4f} "
                    f"{full - capped:>+10.4f}"
                )

        # how fast does it recover with size? locates where the cap bites
        print("\ntrain-size sweep (xgb, PCA-8):")
        reg = prepare_regime("binary", "pca", n_components=8, seed=0)
        curve = {}
        for n in (*curve_sizes, len(reg.y["train"])):
            idx = stratified_cap(reg.X["train"], reg.y["train"], n, seed=0)
            s = score(
                "xgb",
                reg.X["train"][idx],
                reg.y["train"][idx],
                reg.X["test"],
                reg.y["test"],
                best.get("xgb", {}),
            )
            curve[str(len(idx))] = s
            run.write_json("results_partial.json", {**out, "train_size_curve_pca8_xgb": curve})
            print(f"  n={len(idx):>6}  macro-F1 {s:.4f}")
        out["train_size_curve_pca8_xgb"] = curve

        run.metrics = {
            "cap_cost_pca8_xgb": out["pca8_xgb"]["cap_cost"],
            "capped_pca8_xgb": out["pca8_xgb"]["capped"],
            "full_train_pca8_xgb": out["pca8_xgb"]["full_train"],
        }
        run.write_json("results.json", out)
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
