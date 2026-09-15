"""N2 — does a domain-INVARIANT projection clear the floor at 8 dimensions?

PRE-REGISTRATION (BACKLOG `Validate` row N2, winds review 2026-08-23T00:35Z; frozen before the
run). W1 concluded that no quantum-reachable dimensionality is testable leakage-free, because on
whole-file splits every surrogate sits below its floor at 8, 12 and 16 PCA dimensions. X2 then
showed the transferable signal is redundant and spread — 48 voltage phasors carry 56% of the
headroom yet do no better than 48 random features. **Both facts are about PCA**, which is fit to
maximize POOLED variance and has no reason whatever to preserve what survives a file boundary.
If a projection chosen for cross-file INVARIANCE clears the floor at 8 dimensions, then the
matched-dimensionality comparison the quantum half of the paper rests on becomes testable
leakage-free for the first time, and W1's verdict narrows from "any quantum-reachable
dimensionality" to "under PCA".

NON-TRANSDUCTIVE BY CONSTRUCTION, and this is the load-bearing design choice. Every projection is
fit on the TRAINING files only. Invariance is estimated from the group structure WITHIN the
training split — how much a direction varies from one training file to another — and never from
the held-out files. A projection that peeked at the test files would answer a different and much
easier question.

FOUR ARMS, all 8-dimensional, all fit on train only:
  pca8                the shipped baseline (maximize pooled variance)
  file_centered_pca8  remove each TRAINING file's own mean, then PCA — directions that carry
                      variance AFTER file-level offsets are gone
  invariant_gev8      generalized eigenproblem inside the top-40 PCA subspace: the directions
                      MINIMIZING between-file scatter per unit total scatter
  random8             3 random 8-dimensional projections — the negative control

DECISION RULE (kill criterion, fixed in BACKLOG before the run): if no invariant 8-dimensional
projection lifts the BEST CLASSICAL arm to floor + 0.05, PCA is exonerated and W1's verdict
stands as the general one. If one does, W1's verdict narrows to PCA and the matched comparison
becomes testable — which is an escalation, not an edit.

CONTROLS.
  positive (blocking)  the pca8 arm must reproduce W1's PCA-8 XGBoost figure (+0.0282
                       excess-over-floor, whole-file partitions 0-2, seeds 0-2) within 0.005
  negative             the random8 arm bounds what "an 8-dimensional projection" buys for free;
                       an invariant projection that merely matches random has learned nothing
                       about invariance, and its excess is read against random, never zero

Run:  python -m experiments.run_invariant_projection --partitions 3 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from experiments.run_group_split_control import group_splits, reconstruct_groups
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import AngleScaler, build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
DIM = 8
CAP = 2000
SUBSPACE = 40  # drop near-null directions before the generalized eigenproblem (stability)
RIDGE = 1e-6
W1_PCA8_XGB_EXCESS = 0.0282  # decision log 2026-08-22T15:14Z
W1_TOL = 0.005
QUALIFY = 0.05  # excess-over-floor the best classical arm must clear
RANDOM_DRAWS = 3


def between_file_scatter(Z: np.ndarray, gid: np.ndarray) -> np.ndarray:
    """Size-weighted scatter of the per-file means about the grand mean."""
    mu = Z.mean(axis=0)
    S = np.zeros((Z.shape[1], Z.shape[1]))
    for f in np.unique(gid):
        m = gid == f
        d = (Z[m].mean(axis=0) - mu)[:, None]
        S += m.sum() * (d @ d.T)
    return S / len(Z)


def file_centered(Z: np.ndarray, gid: np.ndarray) -> np.ndarray:
    """Remove each file's own mean — the dominant, and cheapest, form of covariate shift."""
    out = Z.copy()
    for f in np.unique(gid):
        m = gid == f
        out[m] -= Z[m].mean(axis=0)
    return out


def invariant_directions(Z: np.ndarray, gid: np.ndarray, dim: int) -> np.ndarray:
    """Directions minimizing between-file scatter per unit total scatter, inside a PCA subspace."""
    p = PCA(n_components=min(SUBSPACE, Z.shape[1]), random_state=0).fit(Z)
    Y = p.transform(Z)
    S_f = between_file_scatter(Y, gid)
    S_t = np.cov(Y, rowvar=False) + RIDGE * np.eye(Y.shape[1])
    vals, vecs = eigh(S_f, S_t)  # ascending: smallest between-file variability first
    W = vecs[:, :dim]  # (subspace, dim)
    return p, W, vals[:dim]


def score_arm(Ztr, ytr, Zte, yte, best, seed) -> dict:
    """Tuned XGBoost and the tuned ZZ fidelity kernel on an 8-dimensional view."""
    set_all_seeds(seed)
    xgb = build("xgb", best["xgb"], seed)
    fit_model("xgb", xgb, Ztr, ytr)
    cls = float(f1_score(yte, xgb.predict(Zte), average="macro"))

    sc = AngleScaler().fit(Ztr)
    s_tr = compute_states(sc.transform(Ztr), "zz", bandwidth=best["qk_bw"])
    set_all_seeds(seed)
    qk = SVC(kernel="precomputed", C=best["qk_C"], class_weight="balanced")
    qk.fit(fidelity_kernel(s_tr, s_tr), ytr)
    s_te = compute_states(sc.transform(Zte), "zz", bandwidth=best["qk_bw"])
    qnt = float(f1_score(yte, qk.predict(fidelity_kernel(s_te, s_tr)), average="macro"))
    return {"xgb": cls, "qsvm": qnt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    db = (REPO_ROOT / "results" / "optuna_cls_xgb_binary_pca8.db").as_posix()
    best = dict(best)
    best["xgb"] = optuna.load_study(
        study_name="cls_xgb_binary_pca8", storage=f"sqlite:///{db}"
    ).best_params  # the regime-tuned parameters W1's control used

    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # shipped stream
    parts = []
    for p in range(args.partitions):
        sp, held = group_splits(gid, y, rng_part)
        parts.append((p, sp, held[2]))

    cfg = {
        "variant": VARIANT,
        "dim": DIM,
        "cap": CAP,
        "subspace": SUBSPACE,
        "arms": ["pca8", "file_centered_pca8", "invariant_gev8", "random8"],
        "partitions": args.partitions,
        "seeds": args.seeds,
        "qualify": QUALIFY,
        "w1_reference": W1_PCA8_XGB_EXCESS,
        "transductive": False,
        "note": "every projection is fit on TRAINING files only; test files are never seen",
    }

    with RunDir(f"invariant_projection_{VARIANT}", config=cfg, seeds=args.seeds) as run:
        rows: list[dict] = []
        for p, sp, held in parts:
            pipe = build_preprocessor("full", n_components=None, seed=0)
            Z = fit_transform_splits(pipe, X, sp)
            ytr_all, yte = y[sp["train"]], y[sp["test"]]
            gtr = gid[sp["train"]]
            floor = trivial_baselines(ytr_all, yte)["baseline_stratified_random"]["aggregate"][
                "macro_f1"
            ]["mean"]

            projections = {}
            pca = PCA(n_components=DIM, random_state=0).fit(Z["train"])
            projections["pca8"] = lambda A, _p=pca: _p.transform(A)

            pca_fc = PCA(n_components=DIM, random_state=0).fit(file_centered(Z["train"], gtr))
            projections["file_centered_pca8"] = lambda A, _p=pca_fc: _p.transform(A)

            sub, W, gev = invariant_directions(Z["train"], gtr, DIM)
            projections["invariant_gev8"] = lambda A, _s=sub, _W=W: _s.transform(A) @ _W

            for d in range(RANDOM_DRAWS):
                r = np.random.default_rng(7000 + d)
                R = r.standard_normal((Z["train"].shape[1], DIM)) / np.sqrt(Z["train"].shape[1])
                projections[f"random8_{d}"] = lambda A, _R=R: A @ _R

            for arm, proj in projections.items():
                Ztr_full, Zte = proj(Z["train"]), proj(Z["test"])
                sub_idx = stratified_cap(Ztr_full, ytr_all, CAP, seed=0)
                Ztr, ytr = Ztr_full[sub_idx], ytr_all[sub_idx]
                for s in args.seeds:
                    sc = score_arm(Ztr, ytr, Zte, yte, best, s)
                    rows.append(
                        {
                            "partition": p,
                            "arm": arm,
                            "seed": s,
                            "floor": floor,
                            **{f"{k}_f1": v for k, v in sc.items()},
                            **{f"{k}_excess": v - floor for k, v in sc.items()},
                        }
                    )
                m = np.mean([r_["xgb_excess"] for r_ in rows[-len(args.seeds) :]])
                q = np.mean([r_["qsvm_excess"] for r_ in rows[-len(args.seeds) :]])
                log.info("p%d %-20s xgb %+0.4f | qsvm %+0.4f (floor %.4f)", p, arm, m, q, floor)
                run.write_json("results_partial.json", {"rows": rows})
            rows[-1]["gev_smallest_eigenvalues"] = [float(v) for v in gev]

        # ---- aggregate ----------------------------------------------------------------
        def mean_of(arm_pred, key):
            v = [r["".join(key)] for r in rows if arm_pred(r["arm"])]
            return float(np.mean(v)) if v else float("nan")

        arms = ["pca8", "file_centered_pca8", "invariant_gev8"]
        summary = {
            a: {
                "xgb_excess": mean_of(lambda x, a=a: x == a, "xgb_excess"),
                "qsvm_excess": mean_of(lambda x, a=a: x == a, "qsvm_excess"),
            }
            for a in arms
        }
        summary["random8"] = {
            "xgb_excess": mean_of(lambda x: x.startswith("random8"), "xgb_excess"),
            "qsvm_excess": mean_of(lambda x: x.startswith("random8"), "qsvm_excess"),
        }

        # ---- controls -------------------------------------------------------------------
        pos_delta = summary["pca8"]["xgb_excess"] - W1_PCA8_XGB_EXCESS
        controls = {
            "positive_pca8_xgb_excess": summary["pca8"]["xgb_excess"],
            "positive_delta_vs_w1": float(pos_delta),
            "positive_pass": bool(abs(pos_delta) <= W1_TOL),
            "negative_random8_xgb_excess": summary["random8"]["xgb_excess"],
            "negative_random8_qsvm_excess": summary["random8"]["qsvm_excess"],
            "negative_note": (
                "an invariant projection that merely matches a random 8-dimensional one has "
                "learned nothing about invariance"
            ),
        }
        controls["pass"] = controls["positive_pass"]
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        # ---- decision rule ---------------------------------------------------------------
        invariant_arms = ["file_centered_pca8", "invariant_gev8"]
        best_inv = max(invariant_arms, key=lambda a: summary[a]["xgb_excess"])
        qualifies = summary[best_inv]["xgb_excess"] >= QUALIFY
        verdict = {
            "controls": controls,
            "summary": summary,
            "best_invariant_arm": best_inv,
            "best_invariant_xgb_excess": summary[best_inv]["xgb_excess"],
            "beats_pca_by": summary[best_inv]["xgb_excess"] - summary["pca8"]["xgb_excess"],
            "beats_random_by": summary[best_inv]["xgb_excess"] - summary["random8"]["xgb_excess"],
            "qualify_threshold": QUALIFY,
            "branch": (
                "ESCALATE_w1_verdict_narrows_to_pca"
                if qualifies
                else "pca_exonerated_w1_verdict_stands_generally"
            ),
            "rule": (
                f"best invariant arm's classical excess >= {QUALIFY} -> the matched comparison "
                "becomes testable leakage-free and W1 narrows to PCA; else PCA is exonerated"
            ),
        }
        run.write_json("results.json", {"rows": rows, "verdict": verdict})
        run.metrics = {"best_invariant_xgb_excess": summary[best_inv]["xgb_excess"]}
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
