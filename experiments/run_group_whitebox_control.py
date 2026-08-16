"""P24 control: the white-box kernel contrast under leakage-free (whole-file) training.

The paper's strongest surviving robustness result — the identical white-box attack
leaves the tuned RBF kernel far above the fidelity kernel — was measured on models
trained under the row-level split, which P12a showed scores memorization of
near-duplicates. This control refits BOTH tuned kernels on whole-file partitions and
runs the identical attack, closing the "your headline contrast describes memorizing
models" objection with a measurement. Pre-registered 2026-08-16 (P24) before the code
was written; graduated from tmp/probe_group_whitebox.py after passing its criterion.

Reuses the shipped pieces exactly: group reconstruction + partition stream from
experiments.run_group_split_control (same default_rng(0) draws), tuned params from
adv_best_params, the QSVMWhiteBox graph with its numeric gate, the RBF GradProvider,
and the published PGD config. Only the split indices differ from the published P5a run.
Readouts are F1-minus-floor and end-state form, not retention: clean models on
whole-file splits sit near the floor, where retention ratios are uninformative.

Run:  python -m experiments.run_group_whitebox_control
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from experiments.run_adversarial import _f1, _svc_proba
from experiments.run_group_split_control import group_splits, reconstruct_groups
from qgridbench.attacks.evasion import pgd
from qgridbench.attacks.gradients import GradProvider
from qgridbench.attacks.qsvm_whitebox import QSVMWhiteBox, assert_matches_reference
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import AngleScaler, build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)

VARIANT = "binary"
ENCODING = "zz"
PARTITIONS = (0, 2)  # p1 (the contrast sign flip) added if time permits
ATTACK_SEEDS = tuple(range(10))
EPS = 0.5


def main():
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    pgd_cfg = cfg_adv["evasion"]["pgd"]
    n_eval = cfg_adv["eval_subset_adv"]

    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # same stream as the shipped control
    all_parts = []
    for p in range(max(PARTITIONS) + 1):
        sp, held = group_splits(gid, y, rng_part)
        all_parts.append((p, sp, held[2]))

    cfg = {
        "partitions": list(PARTITIONS),
        "attack_seeds": list(ATTACK_SEEDS),
        "eps": EPS,
        "qk_bw": best["qk_bw"],
        "qk_C": best["qk_C"],
        "rbf": best["rbf_svm"],
        "n_eval": n_eval,
    }
    with RunDir(f"group_whitebox_control_{VARIANT}_q8", config=cfg) as run:
        results = []
        for p, sp, held in all_parts:
            if p not in PARTITIONS:
                continue
            pipe = build_preprocessor("pca", n_components=8, seed=0)
            Z = fit_transform_splits(pipe, X, sp)
            ytr, yte_full = y[sp["train"]], y[sp["test"]]
            sub = stratified_cap(Z["train"], ytr, 2000, seed=0)
            scaler = AngleScaler().fit(Z["train"])
            scale = Z["train"].std(axis=0, ddof=0)  # feature_scale contract (protocol.py)

            # tuned QSVM on the capped subset, exactly as the shipped control
            s_tr = compute_states(
                scaler.transform(Z["train"][sub]), ENCODING, bandwidth=best["qk_bw"]
            )
            set_all_seeds(0)
            qk = SVC(kernel="precomputed", C=best["qk_C"], class_weight="balanced")
            qk.fit(fidelity_kernel(s_tr, s_tr), ytr[sub])

            # tuned RBF on the same subset
            set_all_seeds(0)
            rbf = build("rbf_svm", best["rbf_svm"], 0)
            fit_model("rbf_svm", rbf, Z["train"][sub], ytr[sub])

            # white-box graph + NUMERIC GATE before any reported number (per partition)
            wb = QSVMWhiteBox(qk, s_tr, scaler, ENCODING, best["qk_bw"])
            gate_idx = stratified_cap(Z["test"], yte_full, 40, seed=0)
            gate = assert_matches_reference(wb, qk, s_tr, scaler, Z["test"][gate_idx])
            log.info(
                "p%d gate: forward %.2e deployed %.2e grad %.2e",
                p,
                gate["forward_rel_err"],
                gate["deployed_float32_rel_err"],
                gate["grad_rel_err"],
            )

            ev = stratified_cap(Z["test"], yte_full, n_eval, seed=0)
            Xte, yte = Z["test"][ev], yte_full[ev]
            floor = trivial_baselines(ytr, yte)["baseline_stratified_random"]["aggregate"][
                "macro_f1"
            ]["mean"]

            def qsvm_f1(Xq):
                K = fidelity_kernel(
                    compute_states(scaler.transform(Xq), ENCODING, bandwidth=best["qk_bw"]), s_tr
                )
                return _f1(yte, _svc_proba(qk, K, 2))

            def rbf_f1(Xr):
                # argmax of the zoo's margin-sigmoid proba == sign of the margin == predict
                return float(f1_score(yte, rbf.predict(Xr), average="macro"))

            gp_rbf = GradProvider(rbf, "rbf_svm")
            clean_q, clean_r = qsvm_f1(Xte), rbf_f1(Xte)

            adv_q, adv_r = [], []
            for a in ATTACK_SEEDS:
                rng = spawn_rng(a, "p5a")  # same stream family as the published P5a
                Xq = pgd(
                    wb,
                    Xte,
                    yte,
                    EPS,
                    scale,
                    n_steps=pgd_cfg["n_steps"],
                    step_frac=pgd_cfg["step_frac"],
                    rng=rng,
                )
                adv_q.append(qsvm_f1(Xq))
                Xr = pgd(
                    gp_rbf,
                    Xte,
                    yte,
                    EPS,
                    scale,
                    n_steps=pgd_cfg["n_steps"],
                    step_frac=pgd_cfg["step_frac"],
                    rng=rng,
                )
                adv_r.append(rbf_f1(Xr))
                log.info(
                    "p%d seed %d | qsvm %.4f (floor %+.4f) | rbf %.4f (floor %+.4f)",
                    p,
                    a,
                    adv_q[-1],
                    adv_q[-1] - floor,
                    adv_r[-1],
                    adv_r[-1] - floor,
                )

            res = {
                "partition": p,
                "held_out_files": held,
                "floor": floor,
                "clean_qsvm": clean_q,
                "clean_rbf": clean_r,
                "adv_qsvm_mean": float(np.mean(adv_q)),
                "adv_qsvm_std": float(np.std(adv_q)),
                "adv_rbf_mean": float(np.mean(adv_r)),
                "adv_rbf_std": float(np.std(adv_r)),
                "qsvm_f1_minus_floor_attacked": float(np.mean(adv_q) - floor),
                "rbf_f1_minus_floor_attacked": float(np.mean(adv_r) - floor),
                "retention_qsvm": float(np.mean(adv_q) / clean_q) if clean_q else None,
                "retention_rbf": float(np.mean(adv_r) / clean_r) if clean_r else None,
                "paired_qsvm_below_rbf_count": int(
                    sum(q < r for q, r in zip(adv_q, adv_r, strict=True))
                ),
                "per_seed_adv_qsvm": adv_q,
                "per_seed_adv_rbf": adv_r,
            }
            results.append(res)
            run.write_json("results_partial.json", {"partitions": results})
            log.info(
                "p%d SUMMARY | clean q %.4f r %.4f floor %.4f | adv q %.4f+-%.4f r %.4f+-%.4f "
                "| q-floor %+.4f r-floor %+.4f | paired q<r %d/10",
                p,
                clean_q,
                clean_r,
                floor,
                res["adv_qsvm_mean"],
                res["adv_qsvm_std"],
                res["adv_rbf_mean"],
                res["adv_rbf_std"],
                res["qsvm_f1_minus_floor_attacked"],
                res["rbf_f1_minus_floor_attacked"],
                res["paired_qsvm_below_rbf_count"],
            )

        inverted = [r["qsvm_f1_minus_floor_attacked"] < -0.05 for r in results]
        rbf_at_floor = [abs(r["rbf_f1_minus_floor_attacked"]) <= 0.05 for r in results]
        verdict = {
            "H_qsvm_inverted_per_partition": inverted,
            "H_rbf_within_0.05_of_floor_per_partition": rbf_at_floor,
            "KILL_fires": not all(inverted),
        }
        run.write_json("results.json", {"partitions": results, "verdict": verdict})
        log.info("VERDICT %s", json.dumps(verdict))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
