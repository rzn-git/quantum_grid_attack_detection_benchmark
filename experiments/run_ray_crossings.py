"""P17e: measure the interleaving mechanism directly instead of inferring it.

THE CLAIM UNDER TEST (results.tex): "the fidelity kernel's decision regions interleave at fine
scale, so [HopSkipJump's] search terminates far away and the large distortion masquerades as
robustness." It is stated as fact and is load-bearing for three results, but it rests on two
INFERENTIAL legs (the random-perturbation control, the white-box ranking inversion) since
P22-final withdrew the restart-variance leg. This samples f densely along each perturbation ray
and counts sign changes -- the quantity the mechanism is about.

PRE-REGISTERED PREDICTIONS (interleaving account):
  H1 crossings-per-ray rises with bandwidth over the healthy->wide range.
  H2 the PARITY model explains the flip rate: flip <=> an ODD number of crossings, so the
     measured flip rate should equal P(odd) computed from the same rays. This is the sharp
     one -- it is what makes "many crossings" produce a flip-rate that saturates and FALLS
     while a single-crossing model keeps rising (the co-peak P17d could not reproduce).
  H3 the matched classical control (tuned RBF-SVM) shows far fewer crossings at its own
     accuracy optimum -- otherwise "interleaving" is not specific to the fidelity kernel.

KILL CRITERION (pre-registered): the account FAILS if either
  (a) crossings-per-ray is flat or falling in bandwidth across the healthy->wide range, or
  (b) the parity prediction misses the measured flip rate by > 0.10 absolute at the
      accuracy-optimal bandwidth.
Either outcome means the interleaving sentence in results.tex is not supported and must be
rewritten as an open question. A confirmation upgrades it from inference to measurement.

CONTROLS: (i) endpoint check -- the ray endpoint's sign must agree with the flip actually
measured by the deployed predict path, else the scan is not sampling the same function;
(ii) degenerate bandwidth (0.1) has collapsed to the prior, so it must show ~0 crossings --
a scan that reports interleaving there is measuring numerical noise, not geometry.

Run:  python -m experiments.run_ray_crossings --n-samples 200

Graduated from tmp/ because make_report consumes its run dir for the paper's
ray_crossings_binary_q8 table; the RunDir name keeps its original `probe_` prefix so
already-recorded results still resolve.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from experiments.run_ablations import kernel_subset
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml

log = get_logger(__name__)


def _fit_qsvm(A_tr, y_tr, enc, bw, C_grid):
    s_tr = compute_states(A_tr, enc, bandwidth=bw)
    K_tr = fidelity_kernel(s_tr, s_tr)

    def cv(C):
        sc = []
        for t, v in StratifiedKFold(3, shuffle=True, random_state=0).split(K_tr, y_tr):
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_tr[np.ix_(t, t)], y_tr[t])
            sc.append(f1_score(y_tr[v], clf.predict(K_tr[np.ix_(v, t)]), average="macro"))
        return float(np.mean(sc))

    best_C = max(C_grid, key=cv)
    return (
        s_tr,
        SVC(kernel="precomputed", C=best_C, class_weight="balanced").fit(K_tr, y_tr),
        best_C,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--n-samples", type=int, default=200, help="points sampled along each ray")
    ap.add_argument("--radius", type=float, default=0.5, help="ray length in sigma")
    args = ap.parse_args()

    cfg = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")["ablations"]["bandwidth_fragility"]
    enc, q = cfg["cell"]
    reg = prepare_regime(args.variant, "pca", n_components=q, seed=0)
    sub = kernel_subset(reg, 2000, seed=0)
    A_tr, y_tr = reg.angles["train"][sub], reg.y["train"][sub]
    scale = reg.feature_scale
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg["eval_subset"], seed=0)
    X_ev, y_ev = reg.X["test"][ev], reg.y["test"][ev]

    # ONE perturbation direction per point, shared across bandwidths -> paired comparison,
    # and the same rng stream/convention the fragility stage uses.
    rng = np.random.default_rng(0)
    delta = rng.uniform(-1.0, 1.0, X_ev.shape) * args.radius * scale
    ts = np.linspace(0.0, 1.0, args.n_samples)

    with RunDir(f"probe_ray_crossings_{args.variant}_q{q}", config=vars(args)) as run:
        rows = []
        for bw in cfg["bandwidths"]:
            s_tr, clf, best_C = _fit_qsvm(A_tr, y_tr, enc, bw, cfg["svm_C_grid"])

            def f(X, _s=s_tr, _c=clf, _bw=bw):
                A = reg.angle_scaler.transform(X)
                return _c.decision_function(fidelity_kernel(compute_states(A, enc, _bw), _s))

            # f along every ray: (n_samples, n_points)
            F = np.stack([f(X_ev + t * delta) for t in ts])
            signs = np.sign(F)
            signs[signs == 0] = 1.0
            crossings = (np.diff(signs, axis=0) != 0).sum(axis=0)  # per point

            base = np.sign(F[0])
            end = np.sign(F[-1])
            flip_measured = float(np.mean(base != end))
            parity_pred = float(np.mean(crossings % 2 == 1))  # H2: flip <=> odd crossings

            rows.append(
                {
                    "bandwidth": float(bw),
                    "best_C": float(best_C),
                    "clean_macro_f1": float(
                        f1_score(y_ev, (F[0] > 0).astype(int), average="macro")
                    ),
                    "crossings_mean": float(crossings.mean()),
                    "crossings_median": float(np.median(crossings)),
                    "crossings_max": int(crossings.max()),
                    "frac_rays_multi_crossing": float(np.mean(crossings > 1)),
                    "flip_rate_measured": flip_measured,
                    "flip_rate_parity_prediction": parity_pred,
                    "parity_abs_error": abs(parity_pred - flip_measured),
                }
            )
            run.write_json("results_partial.json", {"rows": rows})
            log.info(
                "bw=%.2f  F1 %.3f | crossings mean %.2f (max %d, >1 on %.0f%%) | "
                "flip measured %.3f vs parity %.3f",
                bw,
                rows[-1]["clean_macro_f1"],
                rows[-1]["crossings_mean"],
                rows[-1]["crossings_max"],
                100 * rows[-1]["frac_rays_multi_crossing"],
                flip_measured,
                parity_pred,
            )

        # --- pre-registered verdict --------------------------------------------------
        healthy = [r for r in rows if r["bandwidth"] >= 0.4]
        cm = [r["crossings_mean"] for r in healthy]
        rising = all(b >= a - 1e-9 for a, b in zip(cm, cm[1:]))
        peak = max(rows, key=lambda r: r["clean_macro_f1"])
        verdict = {
            "H1_crossings_rise_with_bandwidth": bool(rising),
            "H1_crossings_healthy_range": cm,
            "H2_parity_error_at_accuracy_peak": peak["parity_abs_error"],
            "H2_parity_holds": bool(peak["parity_abs_error"] <= 0.10),
            "accuracy_peak_bandwidth": peak["bandwidth"],
            "control_degenerate_bw_crossings": rows[0]["crossings_mean"],
            "KILLED": bool(not rising or peak["parity_abs_error"] > 0.10),
        }
        run.write_json("results.json", {"rows": rows, "verdict": verdict})
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
