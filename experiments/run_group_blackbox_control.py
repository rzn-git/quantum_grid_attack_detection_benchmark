"""E8 — does the black-box measurement-geometry instance survive leakage-free training?

PRE-REGISTRATION (pass-4 workplan item E8; frozen before the run).
Seven of the paper's eight evaluator-choice instances are demonstrated only on detectors
trained under the row-level protocol the paper itself deprecates, and models that partly
memorize near-duplicates may have unrepresentative decision geometry. P24 already replicated
the WHITE-BOX kernel contrast under whole-file training. This run does the same for the
BLACK-BOX instance: HopSkipJump reports the fidelity kernel as the hardest target while plain
random noise flips it more than any other family, and that inversion is what licenses the
"any minimum-distortion comparison needs a random-perturbation control" deliverable.

THREE PRE-REGISTERED PREDICTIONS (>= 2 of 3 -> "the inversion replicates leakage-free"):
  (i)   HopSkipJump median L-inf distortion: QSVM >= 2x RBF
  (ii)  random-perturbation flip rate at 0.5 sigma: QSVM >= 1.5x RBF
  (iii) restart coefficient of variation is larger for the QSVM than for the RBF
If fewer than two hold, the instance gains a protocol-scope qualifier in the checklist —
which is itself a reportable finding, not a failure.

CONTROL: a degenerate-model check — any arm whose CLEAN macro-F1 sits within 0.05 of its own
stratified-random floor is flagged, because distortion and flip rates read off a model that
never learned anything are uninformative (the paper's own standing precondition).

HopSkipJump is NOT seeded (ART draws its initial point from an unseeded generator), so every
distortion figure is a restart mean +/- std, never a single draw.

Refit path is P24's, line for line: same partition stream (default_rng(0)), same tuned
parameters, same 2,000-row cap, same train-only transforms.

Run:  python -m experiments.run_group_blackbox_control
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.svm import SVC

from experiments.run_adversarial import _f1, _svc_proba
from experiments.run_group_split_control import group_splits, reconstruct_groups
from qgridbench.attacks.blackbox import hopskipjump_attack
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import AngleScaler, build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
ENCODING = "zz"
EPS_RANDOM = 0.5

# The RBF's HopSkipJump distortion reads exactly 0.000 (boundary-adjacent degeneracy, which
# the paper already documents), so a QSVM/RBF distortion RATIO divides by zero and returns a
# huge number that reads like a strong effect. Any ratio whose denominator falls below this
# floor is emitted with a `vacuous` flag and counts as NOT holding (see decision log
# 2026-08-22T09:45Z, where this bit three separate runs in one day).
HSJ_DENOM_FLOOR = 0.01  # sigma


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, nargs="*", default=[0, 2])
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--random-draws", type=int, default=3)
    args = ap.parse_args()

    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    bb = cfg_adv["blackbox"]

    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # same stream as the shipped controls
    parts = []
    for p in range(max(args.partitions) + 1):
        sp, held = group_splits(gid, y, rng_part)
        parts.append((p, sp, held[2]))

    cfg = {
        "variant": VARIANT,
        "encoding": ENCODING,
        "partitions": args.partitions,
        "restarts": args.restarts,
        "random_draws": args.random_draws,
        "eps_random_sigma": EPS_RANDOM,
        "blackbox_budget": bb,
        "qk_bw": best["qk_bw"],
        "qk_C": best["qk_C"],
        "rbf": best["rbf_svm"],
    }

    with RunDir(f"group_blackbox_control_{VARIANT}_q8", config=cfg) as run:
        results = []
        for p, sp, held in parts:
            if p not in args.partitions:
                continue
            pipe = build_preprocessor("pca", n_components=8, seed=0)
            Z = fit_transform_splits(pipe, X, sp)
            ytr, yte_full = y[sp["train"]], y[sp["test"]]
            sub = stratified_cap(Z["train"], ytr, 2000, seed=0)
            scaler = AngleScaler().fit(Z["train"])
            scale = Z["train"].std(axis=0, ddof=0)

            s_tr = compute_states(
                scaler.transform(Z["train"][sub]), ENCODING, bandwidth=best["qk_bw"]
            )
            set_all_seeds(0)
            qk = SVC(kernel="precomputed", C=best["qk_C"], class_weight="balanced")
            qk.fit(fidelity_kernel(s_tr, s_tr), ytr[sub])
            set_all_seeds(0)
            rbf = build("rbf_svm", best["rbf_svm"], 0)
            fit_model("rbf_svm", rbf, Z["train"][sub], ytr[sub])

            ev = stratified_cap(Z["test"], yte_full, bb["eval_subset"], seed=0)
            Xte, yte = Z["test"][ev], yte_full[ev]
            floor = trivial_baselines(ytr, yte)["baseline_stratified_random"]["aggregate"][
                "macro_f1"
            ]["mean"]

            def q_proba(Zq, s=s_tr, m=qk):
                K = fidelity_kernel(
                    compute_states(scaler.transform(Zq), ENCODING, bandwidth=best["qk_bw"]), s
                )
                return _svc_proba(m, K, 2)

            def r_proba(Zr, m=rbf):
                return _svc_proba(m, Zr, 2)

            models = {"qsvm": q_proba, "rbf_svm": r_proba}
            cell = {"partition": p, "test_groups": held, "floor": floor, "per_model": {}}

            for name, proba in models.items():
                clean_pred = proba(Xte).argmax(1)
                clean_f1 = _f1(yte, proba(Xte))
                # HopSkipJump over restarts (unseeded by construction)
                meds = []
                for _ in range(args.restarts):
                    Xa, budget = hopskipjump_attack(
                        proba,
                        Xte,
                        2,
                        max_iter=bb["max_iter"],
                        max_eval=bb["max_eval"],
                        init_eval=bb["init_eval"],
                    )
                    meds.append(float(np.median((np.abs(Xa - Xte) / scale).max(axis=1))))
                # random-perturbation control at matched radius
                flips = []
                for d in range(args.random_draws):
                    rng = np.random.default_rng(100 + d)
                    Xr = Xte + rng.uniform(-1, 1, Xte.shape) * EPS_RANDOM * scale
                    flips.append(float(np.mean(proba(Xr).argmax(1) != clean_pred)))
                cell["per_model"][name] = {
                    "clean_f1": clean_f1,
                    "clean_minus_floor": float(clean_f1 - floor),
                    "degenerate": bool(clean_f1 - floor < 0.05),
                    "hsj_median_linf_mean": float(np.mean(meds)),
                    "hsj_median_linf_std": float(np.std(meds, ddof=1)),
                    "hsj_cv": float(np.std(meds, ddof=1) / max(np.mean(meds), 1e-12)),
                    "hsj_per_restart": meds,
                    "random_flip_rate_mean": float(np.mean(flips)),
                    "random_flip_rate_std": float(np.std(flips, ddof=1)),
                    "queries_per_sample": budget["queries_per_sample"],
                }
                c = cell["per_model"][name]
                log.info(
                    "p%d %-8s clean %.4f (floor %+.4f%s) | HSJ %.3f +/- %.3f (CV %.2f) | "
                    "random flip %.3f",
                    p,
                    name,
                    c["clean_f1"],
                    c["clean_minus_floor"],
                    ", DEGENERATE" if c["degenerate"] else "",
                    c["hsj_median_linf_mean"],
                    c["hsj_median_linf_std"],
                    c["hsj_cv"],
                    c["random_flip_rate_mean"],
                )

            q, r = cell["per_model"]["qsvm"], cell["per_model"]["rbf_svm"]
            i_vacuous = bool(r["hsj_median_linf_mean"] < HSJ_DENOM_FLOOR)
            cell["predictions"] = {
                "i_hsj_ratio": float(
                    q["hsj_median_linf_mean"] / max(r["hsj_median_linf_mean"], 1e-12)
                ),
                # a ratio over a zero denominator measures nothing: vacuous => does NOT hold
                "i_vacuous": i_vacuous,
                "i_holds": bool(
                    q["hsj_median_linf_mean"] >= 2.0 * r["hsj_median_linf_mean"] and not i_vacuous
                ),
                "ii_flip_ratio": float(
                    q["random_flip_rate_mean"] / max(r["random_flip_rate_mean"], 1e-12)
                ),
                "ii_holds": bool(q["random_flip_rate_mean"] >= 1.5 * r["random_flip_rate_mean"]),
                "iii_holds": bool(q["hsj_cv"] > r["hsj_cv"]),
            }
            cell["predictions"]["n_holding"] = int(
                sum(cell["predictions"][k] for k in ("i_holds", "ii_holds", "iii_holds"))
            )
            cell["predictions"]["replicates"] = bool(cell["predictions"]["n_holding"] >= 2)
            log.info("p%d predictions: %s", p, cell["predictions"])
            results.append(cell)
            run.write_json("results_partial.json", {"results": results})

        n_rep = sum(c["predictions"]["replicates"] for c in results)
        verdict = {
            "partitions": [c["partition"] for c in results],
            "n_partitions_replicating": n_rep,
            "per_partition": {str(c["partition"]): c["predictions"] for c in results},
            "any_degenerate_arm": bool(
                any(m["degenerate"] for c in results for m in c["per_model"].values())
            ),
            "branch": (
                "inversion_replicates_leakage_free"
                if n_rep == len(results)
                else ("mixed_scope_qualifier" if n_rep else "does_not_replicate_scope_qualifier")
            ),
            "rule": ">=2 of 3 predictions per partition -> replicates",
        }
        run.write_json("results.json", {"results": results, "verdict": verdict})
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
