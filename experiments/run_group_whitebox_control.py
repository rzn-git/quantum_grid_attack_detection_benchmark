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

The attack runs at every radius of the evasion grid up to the budget (WF-R, decision log
2026-11-20T10:33Z): a white-box result is read at the worst radius inside the budget, and the
first run of this control stored the budget alone. Each (attack seed, radius) draws its random
start from its own stream; the budget's stream is the first run's, so that reading reproduces
it. The run stores the per-radius, per-seed scores, every prediction and every attacked input.

Run:  python -m experiments.run_group_whitebox_control [--partitions 0 2] [--run-tag TAG]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from experiments.run_adversarial import _svc_proba
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
EPS = 0.5  # the budget; each partition's top-level record is the reading at it
TUNED_FILE = "results/adv_best_params_binary_q8.json"
ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def stream_name(eps: float) -> str:
    """The random-start stream of one radius: the published P5a family at the budget, as in the
    first run of this control, and a stream of its own at every other radius."""
    return "p5a" if abs(eps - EPS) < 1e-12 else f"p5a_eps{eps:g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--partitions", type=int, nargs="+", default=list(PARTITIONS))
    ap.add_argument(
        "--eps-grid",
        type=float,
        nargs="+",
        default=None,
        help="radii in sigma (default: the evasion grid of configs/adversarial.yaml up to 0.5)",
    )
    ap.add_argument("--run-tag", default=None, help="suffix of the run name")
    args = ap.parse_args()

    best = json.loads((REPO_ROOT / TUNED_FILE).read_text())
    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    pgd_cfg = cfg_adv["evasion"]["pgd"]
    n_eval = cfg_adv["eval_subset_adv"]
    grid = args.eps_grid or [e for e in cfg_adv["evasion"]["epsilons"] if e <= EPS + 1e-12]
    eps_grid = sorted(float(e) for e in grid)
    if not any(abs(e - EPS) < 1e-12 for e in eps_grid):
        raise ValueError(f"the radius grid {eps_grid} must contain the budget {EPS}")

    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # same stream as the shipped control
    all_parts = []
    for p in range(max(args.partitions) + 1):
        sp, held = group_splits(gid, y, rng_part)
        all_parts.append((p, sp, held[2]))

    cfg = {
        "partitions": list(args.partitions),
        "attack_seeds": list(ATTACK_SEEDS),
        "eps": EPS,
        "eps_grid": eps_grid,
        "best_params": TUNED_FILE,
        "qk_bw": best["qk_bw"],
        "qk_C": best["qk_C"],
        "rbf": best["rbf_svm"],
        "n_eval": n_eval,
    }
    name = f"group_whitebox_control_{VARIANT}_q8" + (f"_{args.run_tag}" if args.run_tag else "")
    with RunDir(name, config=cfg, seeds=list(ATTACK_SEEDS)) as run:
        results = []
        metric_rows, pred_frames, crafted = [], [], {}
        for p, sp, held in all_parts:
            if p not in args.partitions:
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

            row_ids = sp["test"][ev]  # index into the rows reconstruct_groups() returns

            def scored(model, condition, seed, labels, Xa=None, p=p, yte=yte, row_ids=row_ids):
                """Macro-F1 of `labels`, recorded with the predictions and the attacked input."""
                f = float(f1_score(yte, labels, average="macro"))
                metric_rows.append(
                    {
                        "partition": p,
                        "model": model,
                        "condition": condition,
                        "attack_seed": seed,
                        "macro_f1": f,
                    }
                )
                pred_frames.append(
                    pd.DataFrame(
                        {
                            "partition": p,
                            "model": model,
                            "condition": condition,
                            "attack_seed": seed,
                            "row_id": row_ids,
                            "y_true": yte,
                            "label": np.asarray(labels),
                        }
                    )
                )
                if Xa is not None:
                    key = f"p{p}|{model}|{condition}|{seed}"
                    crafted[key] = np.asarray(Xa, dtype=np.float64)
                return f

            def qsvm_labels(Xq, scaler=scaler, s_tr=s_tr, qk=qk):
                K = fidelity_kernel(
                    compute_states(scaler.transform(Xq), ENCODING, bandwidth=best["qk_bw"]), s_tr
                )
                return _svc_proba(qk, K, 2).argmax(1)

            gp_rbf = GradProvider(rbf, "rbf_svm")
            # argmax of the zoo's margin-sigmoid proba == sign of the margin == predict
            clean_q = scored("qsvm", "clean", -1, qsvm_labels(Xte))
            clean_r = scored("rbf_svm", "clean", -1, rbf.predict(Xte))

            per_q = {str(e): [] for e in eps_grid}
            per_r = {str(e): [] for e in eps_grid}
            for a in ATTACK_SEEDS:
                for e in eps_grid:
                    rng = spawn_rng(a, stream_name(e))
                    cond = f"pgd@{e:g}"
                    Xq = pgd(
                        wb,
                        Xte,
                        yte,
                        e,
                        scale,
                        n_steps=pgd_cfg["n_steps"],
                        step_frac=pgd_cfg["step_frac"],
                        rng=rng,
                    )
                    per_q[str(e)].append(scored("qsvm", cond, a, qsvm_labels(Xq), Xq))
                    Xr = pgd(
                        gp_rbf,
                        Xte,
                        yte,
                        e,
                        scale,
                        n_steps=pgd_cfg["n_steps"],
                        step_frac=pgd_cfg["step_frac"],
                        rng=rng,
                    )
                    per_r[str(e)].append(scored("rbf_svm", cond, a, rbf.predict(Xr), Xr))
                log.info(
                    "p%d seed %d | qsvm %s | rbf %s (floor %.4f)",
                    p,
                    a,
                    " ".join(f"{per_q[str(e)][-1]:.3f}" for e in eps_grid),
                    " ".join(f"{per_r[str(e)][-1]:.3f}" for e in eps_grid),
                    floor,
                )
            at_budget = next(str(e) for e in eps_grid if abs(e - EPS) < 1e-12)
            adv_q, adv_r = per_q[at_budget], per_r[at_budget]

            res = {
                "partition": p,
                "held_out_files": held,
                # the white-box graph against the reference implementation, before any
                # reported number
                "numeric_gate": gate,
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
                "curve": {"qsvm": per_q, "rbf_svm": per_r},
            }
            results.append(res)
            run.write_json("results_partial.json", {"partitions": results})
            # exports after every partition, so a killed run keeps what it finished
            pd.DataFrame(metric_rows).to_csv(Path(run.path, "per_seed.csv"), index=False)
            pd.concat(pred_frames, ignore_index=True).to_parquet(
                Path(run.path, "predictions.parquet")
            )
            np.savez_compressed(Path(run.path, "crafted_inputs.npz"), **crafted)
            run.write_json(
                "exports_provenance.json",
                {
                    "env": {k: os.environ.get(k) for k in ENV_KEYS},
                    "random_streams": sorted({stream_name(e) for e in eps_grid}),
                    "tuned_file": {
                        "path": TUNED_FILE,
                        "sha256": hashlib.sha256((REPO_ROOT / TUNED_FILE).read_bytes()).hexdigest(),
                    },
                    "row_id": "index into the rows experiments.run_group_split_control."
                    "reconstruct_groups() returns; attack_seed -1 is the clean reading",
                    "crafted_key": "p<partition>|<model>|pgd@<radius>|<attack seed>",
                },
            )
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
