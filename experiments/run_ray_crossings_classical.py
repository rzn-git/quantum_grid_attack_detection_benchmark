"""P17e-H3: the MATCHED CLASSICAL CONTROL that the first P17e run pre-registered and skipped.

Without it, "the fidelity kernel's decision regions interleave at fine scale" is uncontrolled:
any kernel machine at a wide length scale might cross many boundaries along a ray, in which
case the crossing counts describe kernel machines, not this feature map.

Identical protocol to `experiments/run_ray_crossings.py`: same evaluation subset, same rng seed and
radius so the RAYS ARE THE SAME, same dense sampling, same statistic. Only the model changes --
tuned RBF-SVM in the same PCA-8 space, swept over multipliers of its tuned gamma, mirroring the
quantum bandwidth sweep (the same pairing `bandwidth_fragility_classical` uses).

The headline comparison is at each family's OWN accuracy optimum, since that is where the paper
claims the fidelity kernel is fragile.

Run:  python -m experiments.run_ray_crossings_classical

Graduated from tmp/ because make_report consumes its run dir for the paper's
ray_crossings_binary_q8 table; the RunDir name keeps its original `probe_` prefix so
already-recorded results still resolve.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score

from experiments.run_ablations import kernel_subset
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

N_SAMPLES, RADIUS = 200, 0.5
GAMMA_MULTS = (0.01, 0.1, 1.0, 10.0, 100.0)

cfg = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")["ablations"]["bandwidth_fragility"]
best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
q = cfg["cell"][1]

reg = prepare_regime("binary", "pca", n_components=q, seed=0)
sub = kernel_subset(reg, 2000, seed=0)
Xtr, ytr = reg.X["train"][sub], reg.y["train"][sub]
scale = reg.feature_scale
ev = stratified_cap(reg.X["test"], reg.y["test"], cfg["eval_subset"], seed=0)
X_ev, y_ev = reg.X["test"][ev], reg.y["test"][ev]

rng = np.random.default_rng(0)  # SAME seed as the quantum arm -> identical rays
delta = rng.uniform(-1.0, 1.0, X_ev.shape) * RADIUS * scale
ts = np.linspace(0.0, 1.0, N_SAMPLES)

with RunDir(
    "probe_ray_crossings_classical_binary_q8", config={"gamma_mults": list(GAMMA_MULTS)}
) as run:
    rows = []
    base_gamma = None
    for m in GAMMA_MULTS:
        p = dict(best["rbf_svm"])
        set_all_seeds(0)
        est = build("rbf_svm", p, 0)
        g = est.get_params()["gamma"]
        if isinstance(g, str):  # 'scale' -> resolve numerically so the sweep is meaningful
            g = 1.0 / (Xtr.shape[1] * Xtr.var())
        base_gamma = base_gamma or float(g)
        est.set_params(gamma=base_gamma * m)
        fit_model("rbf_svm", est, Xtr, ytr)

        F = np.stack([est.decision_function(X_ev + t * delta) for t in ts])
        s = np.sign(F)
        s[s == 0] = 1.0
        crossings = (np.diff(s, axis=0) != 0).sum(axis=0)
        rows.append(
            {
                "gamma_mult": float(m),
                "gamma": float(base_gamma * m),
                "clean_macro_f1": float(f1_score(y_ev, (F[0] > 0).astype(int), average="macro")),
                "crossings_mean": float(crossings.mean()),
                "crossings_max": int(crossings.max()),
                "frac_rays_multi_crossing": float(np.mean(crossings > 1)),
            }
        )
        log.info(
            "gamma x%.2g (=%.4g): F1 %.3f | crossings mean %.2f (max %d, >1 on %.0f%%)",
            m,
            base_gamma * m,
            rows[-1]["clean_macro_f1"],
            rows[-1]["crossings_mean"],
            rows[-1]["crossings_max"],
            100 * rows[-1]["frac_rays_multi_crossing"],
        )
    peak = max(rows, key=lambda r: r["clean_macro_f1"])
    run.write_json("results.json", {"rows": rows, "accuracy_peak": peak})
    log.info("CLASSICAL PEAK\n%s", json.dumps(peak, indent=2))
    print(f"-> {run.path}")
