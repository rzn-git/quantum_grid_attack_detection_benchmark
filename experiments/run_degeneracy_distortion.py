"""W2 — does large minimum-distortion co-occur with model degeneracy? (supersedes E10)

PRE-REGISTRATION (pass-4 workplan item W2; frozen before the run).
E8 read HopSkipJump at 2.854 sigma on a degenerate arm and 0.068 sigma on the one
non-degenerate arm and claimed nothing at n = 2. If that co-occurrence holds across
partitions it is a SECOND mechanism for the black-box artifact, and a cheaper one than the
query-budget explanation E10 was chasing — E8 already showed the magnitude is partition-
unstable by 42x, so refining it by budget alone was refining an unstable quantity. E10 is
superseded by this run.

HYPOTHESIS. Minimum-distortion magnitude is INVERSELY related to model health: HopSkipJump
reports a large distortion precisely when the model has not learned, because a model sitting
on its own floor has no coherent decision surface for a line search to find.

DECISION RULE (primary, accuracy axis). Spearman rho between excess-over-floor and median
L-inf distortion for the QSVM across the partitions:
  rho <= -0.5 and p < 0.05  -> confirmed
  |rho| < 0.3               -> refuted (distortion is partition noise, unrelated to health)
  otherwise                 -> inconclusive, and logged as inconclusive
SECONDARY AXIS (triangulation, accuracy-independent). The same correlation against the train
Gram's EFFECTIVE RANK (exp of the von Neumann entropy of its normalized eigenvalues) and its
kernel-target alignment. A confirmed accuracy correlation NOT echoed by the spectral axis is
reported as accuracy-only, never as a spectral mechanism (research discipline rule 6:
mechanism before northstar).

CONTROLS.
  positive (blocking)  partitions 0 and 2 must reproduce E8's DETERMINISTIC quantities —
                       QSVM excess-over-floor -0.010 and +0.068 — within 0.01. HopSkipJump
                       itself is unseeded with a measured restart CV of 0.31-0.55, so the
                       distortion magnitudes are deliberately NOT used as a control; that is
                       stated rather than worked around.
  negative             the RBF arm's distortion must stay at its documented 0.000
                       (boundary-adjacent degeneracy). A nonzero RBF reading means the attack
                       path changed, not that the RBF became attackable.

Refit path is P24's / E8's, line for line: same partition stream, tuned parameters, 2,000-row
cap, train-only transforms.

Run:  python -m experiments.run_degeneracy_distortion --partitions 10 --restarts 3
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.svm import SVC

from experiments.run_adversarial import _f1, _svc_proba
from experiments.run_group_split_control import group_splits, reconstruct_groups
from qgridbench.attacks.blackbox import hopskipjump_attack
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import AngleScaler, build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.models.quantum.qkernel import (
    compute_states,
    fidelity_kernel,
    kernel_target_alignment,
)
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
ENCODING = "zz"
EPS_RANDOM = 0.5
DEGENERATE_BAND = 0.05  # the paper's standing precondition: within this of floor = degenerate

# E8's deterministic anchors (decision log 2026-08-22T09:35Z). Distortion is NOT anchored:
# HopSkipJump is unseeded and its restart CV was measured at 0.31-0.55.
E8_QSVM_EXCESS = {0: -0.010, 2: 0.068}
E8_TOL = 0.01

RHO_CONFIRM = -0.5
RHO_REFUTE = 0.3
P_CONFIRM = 0.05


def spectral(K: np.ndarray, y: np.ndarray) -> dict:
    """Accuracy-independent degeneracy axis read off the training Gram."""
    ev = np.linalg.eigvalsh(K)
    ev = np.clip(ev, 0.0, None)
    total = float(ev.sum())
    p = ev / max(total, 1e-30)
    nz = p[p > 1e-15]
    entropy = float(-(nz * np.log(nz)).sum())
    return {
        "effective_rank": float(np.exp(entropy)),
        "top_eigenvalue_share": float(ev[-1] / max(total, 1e-30)),
        "eigenvalue_sum": total,
        "kta": float(kernel_target_alignment(K, y)),
        "n": int(K.shape[0]),
    }


def corr(x: list[float], yv: list[float]) -> dict:
    r = spearmanr(x, yv)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(x)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=10)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--random-draws", type=int, default=3)
    ap.add_argument(
        "--resume-from",
        default=None,
        help="run dir holding results_partial.json; completed partitions are reused, not refit",
    )
    args = ap.parse_args()

    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    bb = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")["blackbox"]

    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # shipped stream
    parts = []
    for p in range(args.partitions):
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
        "e8_anchors": E8_QSVM_EXCESS,
        "rule": "spearman(excess_over_floor, median_linf) <= -0.5 & p<0.05 -> confirmed",
    }

    with RunDir(f"degeneracy_distortion_{VARIANT}_q8", config=cfg) as run:
        cells = []
        if args.resume_from:
            src = REPO_ROOT / args.resume_from / "results_partial.json"
            cells = [c for c in json.loads(src.read_text())["cells"] if len(c["per_model"]) == 2]
            log.info("resuming: %d complete cells reused from %s", len(cells), src)
        done = {c["partition"] for c in cells}

        for p, sp, held in parts:
            if p in done:
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
            K_tr = fidelity_kernel(s_tr, s_tr)
            set_all_seeds(0)
            qk = SVC(kernel="precomputed", C=best["qk_C"], class_weight="balanced")
            qk.fit(K_tr, ytr[sub])
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

            cell = {
                "partition": p,
                "test_groups": held,
                "floor": floor,
                "spectral": spectral(K_tr, ytr[sub]),
                "per_model": {},
            }
            # The cell enters `cells` BEFORE its models run and the partial is rewritten after
            # EACH model, so a kill loses at most one model rather than a whole partition — the
            # defect that stopped E10 (decision log 2026-08-22T11:10Z), fixed here rather than
            # repeated. Half-finished cells are filtered on resume.
            cells.append(cell)
            for name, proba in (("qsvm", q_proba), ("rbf_svm", r_proba)):
                t_model = time.perf_counter()
                clean_pred = proba(Xte).argmax(1)
                clean_f1 = _f1(yte, proba(Xte))
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
                flips = []
                for d in range(args.random_draws):
                    rngd = np.random.default_rng(100 + d)
                    Xr = Xte + rngd.uniform(-1, 1, Xte.shape) * EPS_RANDOM * scale
                    flips.append(float(np.mean(proba(Xr).argmax(1) != clean_pred)))
                cell["per_model"][name] = {
                    "clean_f1": clean_f1,
                    "excess_over_floor": float(clean_f1 - floor),
                    "degenerate": bool(clean_f1 - floor < DEGENERATE_BAND),
                    "hsj_median_linf_mean": float(np.mean(meds)),
                    "hsj_median_linf_std": float(np.std(meds, ddof=1)),
                    "hsj_cv": float(np.std(meds, ddof=1) / max(np.mean(meds), 1e-12)),
                    "hsj_per_restart": meds,
                    "random_flip_rate_mean": float(np.mean(flips)),
                    "queries_per_sample": budget["queries_per_sample"],
                    "wall_seconds": float(time.perf_counter() - t_model),
                }
                run.write_json("results_partial.json", {"cells": cells})
                log.info(
                    "p%02d %-8s done in %.1f min",
                    p,
                    name,
                    cell["per_model"][name]["wall_seconds"] / 60,
                )
            q = cell["per_model"]["qsvm"]
            log.info(
                "p%02d QSVM excess %+0.4f%s | HSJ %.3f +/- %.3f | eff-rank %.1f | KTA %.4f | "
                "flip %.3f",
                p,
                q["excess_over_floor"],
                " DEGENERATE" if q["degenerate"] else "",
                q["hsj_median_linf_mean"],
                q["hsj_median_linf_std"],
                cell["spectral"]["effective_rank"],
                cell["spectral"]["kta"],
                q["random_flip_rate_mean"],
            )
            run.write_json("results_partial.json", {"cells": cells})

        cells.sort(key=lambda c: c["partition"])

        # ---- controls -------------------------------------------------------------
        by_p = {c["partition"]: c for c in cells}
        pos = {
            str(p): {
                "measured": by_p[p]["per_model"]["qsvm"]["excess_over_floor"],
                "e8": v,
                "delta": by_p[p]["per_model"]["qsvm"]["excess_over_floor"] - v,
            }
            for p, v in E8_QSVM_EXCESS.items()
            if p in by_p
        }
        pos_ok = all(abs(c["delta"]) <= E8_TOL for c in pos.values())
        rbf_max = max(c["per_model"]["rbf_svm"]["hsj_median_linf_mean"] for c in cells)
        controls = {
            "positive_e8_excess_reproduction": pos,
            "positive_pass": bool(pos_ok),
            "negative_rbf_distortion_max": float(rbf_max),
            "negative_pass": bool(rbf_max < 0.01),
            "distortion_not_anchored_note": (
                "HopSkipJump is unseeded (ART draws its initial point from an unseeded "
                "generator) with a measured restart CV of 0.31-0.55, so E8's distortion "
                "magnitudes are deliberately not used as a reproduction control"
            ),
            "pass": bool(pos_ok and rbf_max < 0.01),
        }
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        # ---- correlations ---------------------------------------------------------
        dist = [c["per_model"]["qsvm"]["hsj_median_linf_mean"] for c in cells]
        excess = [c["per_model"]["qsvm"]["excess_over_floor"] for c in cells]
        eff = [c["spectral"]["effective_rank"] for c in cells]
        kta = [c["spectral"]["kta"] for c in cells]
        flip = [c["per_model"]["qsvm"]["random_flip_rate_mean"] for c in cells]

        cors = {
            "accuracy_axis_excess_vs_distortion": corr(excess, dist),
            "spectral_axis_effective_rank_vs_distortion": corr(eff, dist),
            "spectral_axis_kta_vs_distortion": corr(kta, dist),
            # sanity companions, not decision inputs
            "excess_vs_effective_rank": corr(excess, eff),
            "excess_vs_random_flip_rate": corr(excess, flip),
        }
        primary = cors["accuracy_axis_excess_vs_distortion"]
        if primary["rho"] <= RHO_CONFIRM and primary["p"] < P_CONFIRM:
            branch = "confirmed_distortion_tracks_degeneracy"
        elif abs(primary["rho"]) < RHO_REFUTE:
            branch = "refuted_distortion_is_partition_noise"
        else:
            branch = "inconclusive"

        spec = cors["spectral_axis_effective_rank_vs_distortion"]
        echo = bool(spec["rho"] <= RHO_CONFIRM and spec["p"] < P_CONFIRM)
        verdict = {
            "controls": controls,
            "correlations": cors,
            "branch": branch,
            "spectral_echo": echo,
            "mechanism_status": (
                "mechanism (accuracy + spectral agree)"
                if branch.startswith("confirmed") and echo
                else (
                    "accuracy-only association, NOT a spectral mechanism"
                    if branch.startswith("confirmed")
                    else "no association"
                )
            ),
            "n_degenerate_qsvm_arms": int(sum(c["per_model"]["qsvm"]["degenerate"] for c in cells)),
            "distortion_range": [float(min(dist)), float(max(dist))],
            "excess_range": [float(min(excess)), float(max(excess))],
            "rule": cfg["rule"],
        }
        run.write_json("results.json", {"cells": cells, "verdict": verdict})
        run.metrics = {
            "rho_excess_vs_distortion": primary["rho"],
            "p_excess_vs_distortion": primary["p"],
        }
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
