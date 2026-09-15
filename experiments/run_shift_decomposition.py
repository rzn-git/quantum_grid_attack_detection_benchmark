"""W3 — covariate vs concept shift inside the 62% cross-file component of the split drop.

PRE-REGISTRATION (pass-4 workplan item W3; frozen before the run).
The split ladder attributes 0.193 of the 0.311 drop to cross-file shift but does not say what
KIND. E1 measured cross-file 1-NN label agreement at 59.4% against a 71% base rate — local
geometry across files is worse than ignoring the neighbour, which points at the class boundary
moving (concept shift), not only at the inputs moving (covariate shift). This run separates
them and, more importantly, asks whether the movement is a property of the PARTICULAR held-out
files or a generic property of file boundaries.

FOUR MEASUREMENTS per partition, all in standardized full-feature space (transform fit on the
partition's train split — leakage rule):
  (a) support overlap        fraction of test rows within radius 0.1 of a train row
  (b) domain-classifier AUC  tuned LightGBM separating train-domain from test-domain rows with
                             LABELS EXCLUDED; 0.5 = no covariate shift, 1.0 = disjoint
  (c) in-support disagreement  delta_nn among test rows inside the radius, against TWO
                             within-source references: a random ROW split (near-duplicate-rich,
                             the pooled protocol's own reference) and a leave-one-source-file-out
                             split (STRUCTURE-MATCHED, and the one the decision rule uses —
                             comparing a cross-file number to a row-level number is a test that
                             cannot fail, which this project has already been bitten by)
  (d) target-domain cross-file learnability  leave-one-test-file-out INSIDE the target domain,
                             plus a SIZE-MATCHED source->target arm on the same held-out file,
                             because the target-domain training set is much smaller and an
                             unmatched comparison would confound shift with training-set size

DECISION RULES, fixed now.
  concept axis   source->target delta_nn >= row-level reference + 0.10 AND within 0.10 of the
                 cross-file reference -> "generic across files"; exceeding the cross-file
                 reference by >= 0.10 -> "target-specific"; within 0.10 of the row-level
                 reference -> "no concept shift"
  covariate axis (b) >= 0.9 or (a) <= 0.10 -> covariate shift present
  framing        |target-domain LOFO macro-F1 - size-matched source->target macro-F1| <= 0.05
                 -> GENERIC FILE IDIOSYNCRASY: the gap is not a property of the particular
                 held-out files, and the paper's wording should say so. This is the decisive
                 control on framing and is escalated to the operator if it fires.

CONTROLS.
  positive (blocking)  (a) at partition 0 must reproduce E1's archived 0.047207 within 0.005
  negative             a domain classifier separating two RANDOM halves of the TRAIN domain
                       must read AUC <= 0.6 — otherwise (b) measures model capacity, not shift

Run:  python -m experiments.run_shift_decomposition --partitions 5
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from experiments.run_group_split_control import group_splits, reconstruct_groups, tuned_params
from experiments.run_label_ambiguity_audit import RADIUS, neighbour_label_agreement
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.utils.run_tracking import RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
E1_GROUP_P0_SUPPORT = 0.047206875383670965  # results/runs/20260822T074041Z_label_ambiguity_binary
E1_TOL = 0.005
DOMAIN_CAP = 20000  # rows per side for the domain classifier (cost control, stratified)
CONCEPT_MARGIN = 0.10
COVARIATE_AUC = 0.9
COVARIATE_SUPPORT = 0.10
FRAMING_TOL = 0.05


def domain_auc(Za: np.ndarray, Zb: np.ndarray, params: dict, seed: int = 0) -> float:
    """AUC of a tuned LightGBM asked to tell domain A rows from domain B rows (labels unused)."""
    ia = stratified_cap(Za, np.zeros(len(Za), dtype=int), DOMAIN_CAP, seed=seed)
    ib = stratified_cap(Zb, np.zeros(len(Zb), dtype=int), DOMAIN_CAP, seed=seed)
    Z = np.vstack([Za[ia], Zb[ib]])
    d = np.r_[np.zeros(len(ia), dtype=int), np.ones(len(ib), dtype=int)]
    itr, ite = train_test_split(np.arange(len(d)), test_size=0.3, stratify=d, random_state=seed)
    set_all_seeds(seed)
    est = build("lgbm", params, seed)
    fit_model("lgbm", est, Z[itr], d[itr])
    return float(roc_auc_score(d[ite], est.predict_proba(Z[ite])[:, 1]))


def fit_score(params: dict, Ztr, ytr, Zte, yte, seed: int = 0) -> float:
    set_all_seeds(seed)
    est = build("lgbm", params, seed)
    fit_model("lgbm", est, Ztr, ytr)
    return float(f1_score(yte, est.predict(Zte), average="macro"))


def floor_of(ytr, yte) -> float:
    return trivial_baselines(ytr, yte)["baseline_stratified_random"]["aggregate"]["macro_f1"][
        "mean"
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lgbm_params, _best = tuned_params()
    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # shipped stream
    parts = []
    for p in range(args.partitions):
        sp, held = group_splits(gid, y, rng_part)
        parts.append((p, sp, held[2]))

    cfg = {
        "variant": VARIANT,
        "partitions": args.partitions,
        "radius": RADIUS,
        "domain_cap": DOMAIN_CAP,
        "seed": args.seed,
        "concept_margin": CONCEPT_MARGIN,
        "framing_tol": FRAMING_TOL,
        "e1_support_anchor": E1_GROUP_P0_SUPPORT,
    }

    with RunDir(f"shift_decomposition_{VARIANT}", config=cfg, seeds=[args.seed]) as run:
        cells = []
        for p, sp, held in parts:
            pipe = build_preprocessor("full", n_components=None, seed=0)
            Z = fit_transform_splits(pipe, X, sp)
            ytr, yte = y[sp["train"]], y[sp["test"]]
            gtr, gte = gid[sp["train"]], gid[sp["test"]]

            # (a) + (c) source -> target
            s2t = neighbour_label_agreement(Z["train"], ytr, Z["test"], yte)

            # (c) reference 1: random ROW split inside the source domain
            rng = np.random.default_rng(1000 + p)
            perm = rng.permutation(len(ytr))
            half = len(perm) // 2
            ref_row = neighbour_label_agreement(
                Z["train"][perm[:half]], ytr[perm[:half]], Z["train"][perm[half:]], ytr[perm[half:]]
            )

            # (c) reference 2: leave-one-SOURCE-file-out (structure-matched)
            src_files = np.unique(gtr)
            ref_cf = []
            for f in src_files:
                m = gtr == f
                if len(np.unique(ytr[~m])) < 2 or m.sum() == 0:
                    continue
                r = neighbour_label_agreement(Z["train"][~m], ytr[~m], Z["train"][m], ytr[m])
                ref_cf.append({"file": int(f), **r})
            cf_delta = float(np.nanmean([r["delta_nn"] for r in ref_cf]))
            cf_support = float(np.mean([r["nn_frac_within_0.1"] for r in ref_cf]))

            # (b) covariate axis + its negative control
            auc = domain_auc(Z["train"], Z["test"], lgbm_params, seed=args.seed)
            neg_auc = domain_auc(
                Z["train"][perm[:half]], Z["train"][perm[half:]], lgbm_params, seed=args.seed
            )

            # (d) target-domain cross-file learnability, size-matched against source->target
            folds = []
            for f in np.unique(gte):
                m = gte == f
                if len(np.unique(yte[~m])) < 2 or len(np.unique(yte[m])) < 2:
                    log.info("p%d target file %d skipped (single-class fold)", p, int(f))
                    continue
                n_tgt = int((~m).sum())
                tgt = fit_score(lgbm_params, Z["test"][~m], yte[~m], Z["test"][m], yte[m])
                sub = stratified_cap(Z["train"], ytr, n_tgt, seed=0)  # SIZE-MATCHED source arm
                src_matched = fit_score(
                    lgbm_params, Z["train"][sub], ytr[sub], Z["test"][m], yte[m]
                )
                src_full = fit_score(lgbm_params, Z["train"], ytr, Z["test"][m], yte[m])
                folds.append(
                    {
                        "held_file": int(f),
                        "n_target_train": n_tgt,
                        "target_crossfile_f1": tgt,
                        "source_size_matched_f1": src_matched,
                        "source_full_f1": src_full,
                        "floor": floor_of(yte[~m], yte[m]),
                    }
                )
                log.info(
                    "p%d file %d | target-domain %.4f | source(size-matched) %.4f | "
                    "source(full) %.4f | floor %.4f",
                    p,
                    int(f),
                    tgt,
                    src_matched,
                    src_full,
                    folds[-1]["floor"],
                )

            cell = {
                "partition": p,
                "test_groups": held,
                "support_overlap": s2t["nn_frac_within_0.1"],
                "delta_nn_source_to_target": s2t["delta_nn"],
                "n_near_source_to_target": s2t["n_near"],
                "reference_row_level": {
                    "delta_nn": ref_row["delta_nn"],
                    "support": ref_row["nn_frac_within_0.1"],
                },
                "reference_cross_file_source": {
                    "delta_nn": cf_delta,
                    "support": cf_support,
                    "per_file": ref_cf,
                },
                "domain_auc": auc,
                "domain_auc_negative_control": neg_auc,
                "target_domain_folds": folds,
                "target_crossfile_mean": float(
                    np.mean([f["target_crossfile_f1"] for f in folds]) if folds else np.nan
                ),
                "source_size_matched_mean": float(
                    np.mean([f["source_size_matched_f1"] for f in folds]) if folds else np.nan
                ),
                "source_full_mean": float(
                    np.mean([f["source_full_f1"] for f in folds]) if folds else np.nan
                ),
            }
            log.info(
                "p%02d support %.4f | delta_nn s->t %.4f (row ref %.4f, cross-file ref %.4f) | "
                "domain AUC %.4f (neg %.4f)",
                p,
                cell["support_overlap"],
                cell["delta_nn_source_to_target"],
                ref_row["delta_nn"],
                cf_delta,
                auc,
                neg_auc,
            )
            cells.append(cell)
            run.write_json("results_partial.json", {"cells": cells})

        # ---- controls -------------------------------------------------------------
        p0 = next(c for c in cells if c["partition"] == 0)
        pos_delta = p0["support_overlap"] - E1_GROUP_P0_SUPPORT
        neg_max = max(c["domain_auc_negative_control"] for c in cells)
        controls = {
            "positive_support_reproduction_delta": float(pos_delta),
            "positive_pass": bool(abs(pos_delta) <= E1_TOL),
            "negative_random_half_auc_max": float(neg_max),
            "negative_pass": bool(neg_max <= 0.6),
            "pass": bool(abs(pos_delta) <= E1_TOL and neg_max <= 0.6),
        }
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        # ---- decomposition --------------------------------------------------------
        m_s2t = float(np.mean([c["delta_nn_source_to_target"] for c in cells]))
        m_row = float(np.mean([c["reference_row_level"]["delta_nn"] for c in cells]))
        m_cf = float(np.mean([c["reference_cross_file_source"]["delta_nn"] for c in cells]))
        m_sup = float(np.mean([c["support_overlap"] for c in cells]))
        m_auc = float(np.mean([c["domain_auc"] for c in cells]))
        m_tgt = float(np.nanmean([c["target_crossfile_mean"] for c in cells]))
        m_src_m = float(np.nanmean([c["source_size_matched_mean"] for c in cells]))
        m_src_f = float(np.nanmean([c["source_full_mean"] for c in cells]))

        if m_s2t >= m_row + CONCEPT_MARGIN and abs(m_s2t - m_cf) <= CONCEPT_MARGIN:
            concept = "generic_across_files"
        elif m_s2t >= m_cf + CONCEPT_MARGIN:
            concept = "target_specific"
        elif abs(m_s2t - m_row) < CONCEPT_MARGIN:
            concept = "no_concept_shift"
        else:
            concept = "inconclusive"
        covariate = bool(m_auc >= COVARIATE_AUC or m_sup <= COVARIATE_SUPPORT)
        generic_framing = bool(abs(m_tgt - m_src_m) <= FRAMING_TOL)

        verdict = {
            "controls": controls,
            "concept_axis": {
                "delta_nn_source_to_target": m_s2t,
                "reference_row_level": m_row,
                "reference_cross_file_source": m_cf,
                "branch": concept,
            },
            "covariate_axis": {
                "support_overlap": m_sup,
                "domain_auc": m_auc,
                "present": covariate,
            },
            "framing_control": {
                "target_domain_crossfile_f1": m_tgt,
                "source_size_matched_f1": m_src_m,
                "source_full_f1": m_src_f,
                "delta_target_vs_size_matched": float(m_tgt - m_src_m),
                "generic_file_idiosyncrasy": generic_framing,
                "escalate_to_operator": generic_framing,
            },
            "branch": (
                "both"
                if (covariate and concept in ("generic_across_files", "target_specific"))
                else (
                    "covariate_dominant"
                    if covariate
                    else (
                        "concept_dominant"
                        if concept in ("generic_across_files", "target_specific")
                        else "neither"
                    )
                )
            ),
            "rule": (
                "concept: s->t vs ROW ref (+0.10) and vs CROSS-FILE ref (+/-0.10); "
                "covariate: AUC>=0.9 or support<=0.10; framing: |target - size-matched|<=0.05"
            ),
        }
        run.write_json("results.json", {"cells": cells, "verdict": verdict})
        run.metrics = {
            "delta_nn_source_to_target": m_s2t,
            "reference_cross_file_source": m_cf,
            "support_overlap": m_sup,
            "domain_auc": m_auc,
        }
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
