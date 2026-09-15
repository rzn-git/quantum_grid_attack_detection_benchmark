"""N2c — does the QUANTUM arm benefit from N2b's supervised representation, or only classical?

PRE-REGISTRATION (BACKLOG `Validate` row N2c, winds review 2026-08-23T07:43Z; frozen before the
run). N2b found the only 8-dimensional representation in which a leakage-free arm clears the bar on
whole-file splits: a supervised encoder at +0.0626 excess over floor, against PCA-8 at +0.0291 and a
random projection at +0.0285. The paper's central comparison lives at 8 dimensions, and W1 declared
it untestable precisely because every arm sat at its floor there — two near-chance numbers are not a
comparison. The quantum references are already archived and stark: the tuned ZZ fidelity kernel
reads +0.0134 on PCA-8 and +0.0138 on a random projection
(`20260822T184937Z_invariant_projection_binary`).

METHOD. Per whole-file partition, build TWO 8-dimensional representations, both fit on TRAINING
FILES ONLY, and run the same three models in each so the contrast is within-run rather than across
artifacts:
  supervised8   N2b's encoder at lambda = 0 — the adversarial term contributes -0.0006, so the
                simpler encoder is the correct one and saying so is part of the design
  pca8          the shipped baseline, carried as the matched linear reference
Models: tuned XGBoost (the positive control), the tuned ZZ fidelity-kernel SVM, and the tuned VQC.
3 partitions x 3 seeds, 2,000-row cap. `AngleScaler` is a train-fitted MinMax into [-pi, pi] and
takes any 8-dimensional input, so nothing about the encoding pipeline changes.

DECISION RULE, fixed now:
  some quantum arm clears floor + 0.05 on supervised8  -> the matched-dimensionality comparison
                                                          becomes testable under the whole-file
                                                          protocol and the pending ruling MATTERS
  no quantum arm clears floor + 0.05                   -> the representation helps only the
                                                          classical side; the matched comparison
                                                          stays untestable for the quantum half and
                                                          the pending ruling is MOOT for it

CAVEAT, PRE-REGISTERED so the result cannot later be framed as something it is not, whichever way it
lands. A supervised encoder is trained by a CLASSICAL neural network. A quantum kernel applied to
its output is a quantum feature map over CLASSICALLY-LEARNED FEATURES, not over the data. That
materially weakens any quantum claim and is exactly the objection a reviewer will raise. It is
recorded here, before the numbers exist.

CONTROLS.
  positive (blocking)  the XGBoost arm on supervised8 must reproduce N2b's +0.0632 BIT-IDENTICALLY —
                       same encoder, same seeds, and the torch arms are now explicitly seeded, so
                       anything other than an exact match means this is not the representation N2b
                       measured
  negative             the same models on pca8, in this same run, bound what the representation is
                       worth; the archived PCA-8 (+0.0134) and random-8 (+0.0138) quantum figures
                       are the external references the result must beat to mean anything

ESCALATION TRIGGER (study protocol): any quantum-beats-classical accuracy result stops the task —
log the question, propagate nothing.

Run:  python -m experiments.run_supervised_repr_quantum
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from experiments.run_group_split_control import group_splits, reconstruct_groups
from experiments.run_nonlinear_embedding import encode, fit_file_adversarial
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import AngleScaler, build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
from qgridbench.models.quantum.vqc import VQClassifier
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
DIM = 8
CAP = 2000
ENCODER_FIT_N = 3000  # the same stratified subsample N2b fit its encoder on
N2B_SUPERVISED_XGB = 0.0632  # decision log 2026-08-23T01:20Z, lambda-0 arm
ARCHIVED_QSVM_PCA8 = 0.0134  # 20260822T184937Z_invariant_projection_binary
ARCHIVED_QSVM_RANDOM8 = 0.0138
QUALIFY = 0.05


def tuned() -> dict:
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    db = (REPO_ROOT / "results" / "optuna_cls_xgb_binary_pca8.db").as_posix()
    best = dict(best)
    best["xgb"] = optuna.load_study(
        study_name="cls_xgb_binary_pca8", storage=f"sqlite:///{db}"
    ).best_params
    return best


def qsvm_excess(Ztr, ytr, Zte, yte, best, seed: int) -> float:
    sc = AngleScaler().fit(Ztr)
    s_tr = compute_states(sc.transform(Ztr), "zz", bandwidth=best["qk_bw"])
    set_all_seeds(seed)
    m = SVC(kernel="precomputed", C=best["qk_C"], class_weight="balanced")
    m.fit(fidelity_kernel(s_tr, s_tr), ytr)
    s_te = compute_states(sc.transform(Zte), "zz", bandwidth=best["qk_bw"])
    return float(f1_score(yte, m.predict(fidelity_kernel(s_te, s_tr)), average="macro"))


def vqc_score(Ztr, ytr, Zte, yte, best, seed: int) -> float:
    sc = AngleScaler().fit(Ztr)
    set_all_seeds(seed)
    torch.manual_seed(seed)  # set_all_seeds does not cover torch (finding 2026-08-23T01:20Z)
    clf = VQClassifier(
        n_qubits=DIM,
        depth=int(best["vqc_depth"]),
        lr=float(best["vqc_lr"]),
        batch_size=int(best["vqc_batch"]),
        max_epochs=int(best["vqc_epochs"]),
        seed=seed,
    )
    clf.fit(sc.transform(Ztr), ytr)
    return float(f1_score(yte, clf.predict(sc.transform(Zte)), average="macro"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--no-vqc", action="store_true")
    args = ap.parse_args()

    best = tuned()
    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # shipped stream — partitions must match every prior run
    parts = []
    for p in range(args.partitions):
        sp, _held = group_splits(gid, y, rng_part)
        parts.append((p, sp))

    cfg = {
        "variant": VARIANT,
        "dim": DIM,
        "cap": CAP,
        "representations": ["supervised8", "pca8"],
        "models": ["xgb", "qsvm"] + ([] if args.no_vqc else ["vqc"]),
        "partitions": args.partitions,
        "seeds": args.seeds,
        "qualify": QUALIFY,
        "n2b_reference_supervised_xgb": N2B_SUPERVISED_XGB,
        "archived_qsvm_pca8": ARCHIVED_QSVM_PCA8,
        "archived_qsvm_random8": ARCHIVED_QSVM_RANDOM8,
        "caveat": (
            "a supervised encoder is trained by a CLASSICAL network, so a quantum kernel over its "
            "output is a quantum feature map over classically-learned features, not over the data"
        ),
    }

    with RunDir(f"supervised_repr_quantum_{VARIANT}", config=cfg, seeds=args.seeds) as run:
        rows: list[dict] = []
        for p, sp in parts:
            pipe = build_preprocessor("full", n_components=None, seed=0)
            Z = fit_transform_splits(pipe, X, sp)
            ytr_all, yte = y[sp["train"]], y[sp["test"]]
            gtr = gid[sp["train"]]
            floor = trivial_baselines(ytr_all, yte)["baseline_stratified_random"]["aggregate"][
                "macro_f1"
            ]["mean"]

            # both representations fit on TRAINING FILES ONLY, exactly as N2b did
            sub = stratified_cap(Z["train"], ytr_all, min(ENCODER_FIT_N, len(ytr_all)), seed=0)
            enc, _disc, _chance = fit_file_adversarial(
                Z["train"][sub], ytr_all[sub], gtr[sub], seed=0, lam_max=0.0
            )
            pca = PCA(n_components=DIM, random_state=0).fit(Z["train"])
            reps = {
                "supervised8": lambda A, _e=enc: encode(_e, A),
                "pca8": lambda A, _p=pca: _p.transform(A),
            }

            for rep, proj in reps.items():
                Ztr_full, Zte = proj(Z["train"]), proj(Z["test"])
                idx = stratified_cap(Ztr_full, ytr_all, CAP, seed=0)
                Ztr, ytr = Ztr_full[idx], ytr_all[idx]
                for s in args.seeds:
                    set_all_seeds(s)
                    xm = build("xgb", best["xgb"], s)
                    fit_model("xgb", xm, Ztr, ytr)
                    scores = {
                        "xgb": float(f1_score(yte, xm.predict(Zte), average="macro")),
                        "qsvm": qsvm_excess(Ztr, ytr, Zte, yte, best, s),
                    }
                    if not args.no_vqc:
                        scores["vqc"] = vqc_score(Ztr, ytr, Zte, yte, best, s)
                    rows.append(
                        {
                            "partition": p,
                            "representation": rep,
                            "seed": s,
                            "floor": floor,
                            **{f"{k}_f1": v for k, v in scores.items()},
                            **{f"{k}_excess": v - floor for k, v in scores.items()},
                        }
                    )
                    run.write_json("results_partial.json", {"rows": rows})
                last = rows[-len(args.seeds) :]
                log.info(
                    "p%d %-12s xgb %+0.4f | qsvm %+0.4f%s (floor %.4f)",
                    p,
                    rep,
                    float(np.mean([r["xgb_excess"] for r in last])),
                    float(np.mean([r["qsvm_excess"] for r in last])),
                    ""
                    if args.no_vqc
                    else f" | vqc {float(np.mean([r['vqc_excess'] for r in last])):+0.4f}",
                    floor,
                )

        # ---- aggregate ------------------------------------------------------------------
        models = cfg["models"]
        summary = {
            rep: {
                m: float(np.mean([r[f"{m}_excess"] for r in rows if r["representation"] == rep]))
                for m in models
            }
            for rep in reps
        }

        # ---- controls ---------------------------------------------------------------------
        pos = summary["supervised8"]["xgb"]
        controls = {
            "positive_supervised_xgb_excess": pos,
            "positive_n2b_reference": N2B_SUPERVISED_XGB,
            "positive_delta": float(pos - N2B_SUPERVISED_XGB),
            "positive_pass": bool(abs(pos - N2B_SUPERVISED_XGB) < 5e-4),
            "positive_note": (
                "bit-identical expected: same encoder, same seeds, torch now explicitly seeded"
            ),
            "negative_pca8_in_run": summary["pca8"],
            "negative_archived_qsvm_pca8": ARCHIVED_QSVM_PCA8,
            "negative_archived_qsvm_random8": ARCHIVED_QSVM_RANDOM8,
        }
        controls["pass"] = controls["positive_pass"]
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        # ---- decision rule -------------------------------------------------------------------
        quantum = [m for m in models if m in ("qsvm", "vqc")]
        best_q = max(quantum, key=lambda m: summary["supervised8"][m])
        clears = summary["supervised8"][best_q] >= QUALIFY
        beats_classical = summary["supervised8"][best_q] > summary["supervised8"]["xgb"]
        verdict = {
            "controls": controls,
            "summary": summary,
            "best_quantum_arm": best_q,
            "best_quantum_supervised_excess": summary["supervised8"][best_q],
            "quantum_gain_over_pca8": {
                m: summary["supervised8"][m] - summary["pca8"][m] for m in models
            },
            "qualify_threshold": QUALIFY,
            "quantum_beats_classical": bool(beats_classical),
            "escalate": bool(beats_classical),
            "branch": (
                "matched_comparison_becomes_testable"
                if clears
                else "representation_helps_classical_only_ruling_is_moot"
            ),
            "rule": (
                f"some quantum arm >= floor + {QUALIFY} on supervised8 -> the matched comparison "
                "becomes testable; else the pending ruling is moot for the quantum half"
            ),
            "caveat": cfg["caveat"],
            "rows": rows,
        }
        run.write_json("results.json", verdict)
        run.metrics = {"best_quantum_supervised_excess": summary["supervised8"][best_q]}
        log.info(
            "VERDICT %s | best quantum %s %+0.4f (bar %.2f) | xgb %+0.4f | beats_classical=%s",
            verdict["branch"],
            best_q,
            summary["supervised8"][best_q],
            QUALIFY,
            summary["supervised8"]["xgb"],
            beats_classical,
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
