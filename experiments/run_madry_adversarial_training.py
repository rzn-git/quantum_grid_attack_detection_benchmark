"""E12b — does MADRY MIN-MAX adversarial training change the cross-family picture, where
single-round augmentation did not?

PRE-REGISTRATION (BACKLOG `Validate` row "Adversarial training on MLP + VQC", opened by external
review 2026-08-15, confirmed NOT superseded by E12 at 2026-08-23T14:55Z; frozen before the run).
TIER A under the study protocol §11 experiment gate: it answers a named reviewer objection the paper
currently cannot — "every model in the study is undefended, while QUPID ships WITH a DP-noise
defense". Run authorised by the operator to have the answer in hand before the objection is raised.

WHY E12 DOES NOT CLOSE THIS. E12 (finding 2026-08-22T18:05Z, artifact
`20260822T175459Z_adversarial_augmentation_binary_q8`) ran single-round adversarial data
AUGMENTATION — craft PGD once against the CLEAN model, append, refit — and returned a null:
robustness gain -0.253 for the MLP and +0.039 for the VQC against a 0.05 bar. Its own scope line is
why this row survived: a null there licenses "augmentation does not change the picture" and never
"no defense changes the picture". The defect augmentation has is structural, not incidental: the
adversarial rows are crafted against a model that ceases to exist the moment the refit begins.

WHAT THIS RUN DOES DIFFERENTLY. Round-based min-max: at every round the adversarial set is
regenerated against the CURRENT defended model, then the model is refit on clean + adversarial. That
is the practical approximation to Madry's inner-maximisation / outer-minimisation loop, and the one
thing augmentation structurally cannot do. R rounds; R = 1 reduces EXACTLY to E12, which is what
makes the anchor below possible.

DECISION RULES, fixed now and deliberately IDENTICAL to E12's so the two are directly comparable:
  (1) is the defense real?   defended minus undefended macro-F1 under white-box PGD at 0.5 sigma
                             >= 0.05 for at least one family -> the defense does something. If
                             neither moves, this is a null DEFENSE and (2)-(3) are UNTESTED, not
                             answered — the same discipline E12 applied to itself.
  (2) does it change the cross-family picture?  the VQC's damage normalized by mean classical damage
                             moves by >= 0.10 between undefended and defended -> the cross-family
                             reading is defense-dependent and the paper must say so.
  (3) what does it cost?     clean macro-F1 delta, reported either way, no threshold.

EVALUATION DISCIPLINE (the classic adversarial-training reporting flaw, avoided by construction, as
in E12): each defended model is attacked with examples crafted AGAINST THE DEFENDED MODEL. Re-using
the undefended model's adversarial examples manufactures a robustness gain out of gradient mismatch
alone.

CONTROLS.
  positive A (blocking)  the UNDEFENDED arm must reproduce the published 10-seed figures within
                         0.03 — MLP clean 0.4653 and 0.2890 under PGD at 0.5 sigma, VQC clean
                         0.4754 (artifact `20260813T183448Z_adversarial_binary_q8`). Same bar E12
                         used.
  positive B (reported)  the ROUND-1 state is E12's construction exactly, so its defended clean
                         macro-F1 is compared against E12's archived 0.4634 (MLP) and 0.4836 (VQC).
                         Reported rather than blocking: E12 ran three seeds under a different RNG
                         stream name, so exact identity is not expected — a gross divergence would
                         still mean this is not the same procedure.
  negative               every defended model at eps = 0 must return its own clean macro-F1; an
                         attack path that fires at zero radius is broken.
  reported               the per-round robustness trace, so a defense that is still improving when
                         the round budget runs out is VISIBLE rather than reported as a ceiling.

Run:  python -m experiments.run_madry_adversarial_training --seeds 0 1 2 --rounds 5
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.run_adversarial import CLASSICAL, FittedZoo, _f1
from qgridbench.attacks.evasion import pgd
from qgridbench.attacks.gradients import GradProvider
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.models.quantum.vqc import VQClassifier
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)

VARIANT = "binary"
CLASSICAL_TARGETS = [c for c in CLASSICAL if c != "mlp"]
PUBLISHED = {"mlp_clean": 0.4653, "mlp_pgd_0.5": 0.2890, "vqc_clean": 0.4754}
E12_DEFENDED = {"mlp_clean": 0.4634123850843890, "vqc_clean": 0.4835829491332153}
CONTROL_TOL = 0.03
DEFENSE_BAR = 0.05
PICTURE_BAR = 0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="zz")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=5, help="min-max rounds; R=1 reduces to E12")
    ap.add_argument("--train-eps", type=float, default=0.5)
    ap.add_argument("--eval-eps", type=float, nargs="*", default=[0.1, 0.3, 0.5])
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    pgd_cfg = cfg_adv["evasion"]["pgd"]

    cfg = {
        "variant": args.variant,
        "qubits": args.qubits,
        "defense": "round-based Madry min-max adversarial training (R rounds, adversarial set "
        "regenerated against the CURRENT model each round); R=1 reduces to E12's augmentation",
        "rounds": args.rounds,
        "train_eps": args.train_eps,
        "eval_eps": args.eval_eps,
        "seeds": args.seeds,
        "published_reference": PUBLISHED,
        "e12_defended_reference": E12_DEFENDED,
        "defense_bar": DEFENSE_BAR,
        "picture_bar": PICTURE_BAR,
    }

    with RunDir(
        f"madry_adversarial_training_{args.variant}_q{args.qubits}", config=cfg, seeds=args.seeds
    ) as run:
        reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
        scale = reg.feature_scale
        ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
        Xte, yte = reg.X["test"][ev], reg.y["test"][ev]

        per_seed = []
        for s in args.seeds:
            zoo = FittedZoo(
                reg, args.encoding, args.qubits, s, cfg_q["qkernel"]["subset_cap"], best
            )
            Xtr, ytr = zoo.Xtr, zoo.ytr
            va = stratified_cap(reg.X["val"], reg.y["val"], 2000, seed=0)
            Xva, yva = reg.X["val"][va], reg.y["val"][va]
            y_aug = np.concatenate([ytr, ytr])

            # ---- the min-max loop ---------------------------------------------------
            # round r crafts against the model produced by round r-1; round 1 crafts against
            # the CLEAN model, which is exactly E12's single-round construction.
            mlp_def, vqc_def, trace = None, None, []

            def mlp_grad(m):
                return GradProvider(m, "mlp")

            def vqc_grad(v, _z=zoo):
                return lambda Z, yy: (
                    GradProvider(v, "vqc")(_z._to_angles(Z), yy) * _z._angle._mm.scale_
                )

            for r in range(1, args.rounds + 1):
                rng = spawn_rng(s, f"madry_round{r}")
                gp_mlp = zoo.grad_provider("mlp") if mlp_def is None else mlp_grad(mlp_def)
                gp_vqc = zoo.grad_provider("vqc") if vqc_def is None else vqc_grad(vqc_def)

                adv_mlp = pgd(
                    gp_mlp,
                    Xtr,
                    ytr,
                    args.train_eps,
                    scale,
                    n_steps=pgd_cfg["n_steps"],
                    step_frac=pgd_cfg["step_frac"],
                    rng=rng,
                )
                adv_vqc = pgd(
                    gp_vqc,
                    Xtr,
                    ytr,
                    args.train_eps,
                    scale,
                    n_steps=pgd_cfg["n_steps"],
                    step_frac=pgd_cfg["step_frac"],
                    rng=rng,
                )

                set_all_seeds(s)
                mlp_def = build("mlp", best["mlp"], s)
                fit_model("mlp", mlp_def, np.vstack([Xtr, adv_mlp]), y_aug)

                set_all_seeds(s)
                vqc_def = VQClassifier(
                    args.qubits,
                    depth=best.get("vqc_depth", 4),
                    n_classes=2,
                    lr=best.get("vqc_lr", 0.05),
                    batch_size=best.get("vqc_batch", 64),
                    max_epochs=best.get("vqc_epochs", 40),
                    patience=8,
                    seed=s,
                )
                vqc_def.fit(
                    zoo._to_angles(np.vstack([Xtr, adv_vqc])),
                    y_aug,
                    zoo._to_angles(Xva),
                    yva,
                )

                def proba(name, Z, _m=mlp_def, _v=vqc_def, _z=zoo):
                    return (
                        _m.predict_proba(Z) if name == "mlp" else _v.predict_proba(_z._to_angles(Z))
                    )

                rt = {"round": r}
                for fam, gp in (("mlp", mlp_grad(mlp_def)), ("vqc", vqc_grad(vqc_def))):
                    a = pgd(
                        gp,
                        Xte,
                        yte,
                        args.train_eps,
                        scale,
                        n_steps=pgd_cfg["n_steps"],
                        step_frac=pgd_cfg["step_frac"],
                        rng=spawn_rng(s, f"tr_{fam}_{r}"),
                    )
                    rt[f"{fam}_clean"] = _f1(yte, proba(fam, Xte))
                    rt[f"{fam}_attacked"] = _f1(yte, proba(fam, a))
                trace.append(rt)
                log.info(
                    "s%d round %d/%d | mlp clean %.4f att %.4f | vqc clean %.4f att %.4f",
                    s,
                    r,
                    args.rounds,
                    rt["mlp_clean"],
                    rt["mlp_attacked"],
                    rt["vqc_clean"],
                    rt["vqc_attacked"],
                )
                run.write_json("results_partial.json", {"per_seed": per_seed, "in_flight": trace})

            gp_def = {"mlp": mlp_grad(mlp_def), "vqc": vqc_grad(vqc_def)}

            def proba_def(name, Z, _m=mlp_def, _v=vqc_def, _z=zoo):
                return _m.predict_proba(Z) if name == "mlp" else _v.predict_proba(_z._to_angles(Z))

            cell = {"seed": s, "rounds": trace, "undefended": {}, "defended": {}, "classical": {}}
            for fam in ("mlp", "vqc"):
                cell["undefended"][fam] = {"clean": _f1(yte, zoo.predict_proba(fam, Xte))}
                cell["defended"][fam] = {"clean": _f1(yte, proba_def(fam, Xte))}
                z0 = pgd(
                    gp_def[fam],
                    Xte,
                    yte,
                    0.0,
                    scale,
                    n_steps=pgd_cfg["n_steps"],
                    step_frac=pgd_cfg["step_frac"],
                    rng=spawn_rng(s, f"zero_{fam}"),
                )
                cell["defended"][fam]["zero_radius"] = _f1(yte, proba_def(fam, z0))

            for e in args.eval_eps:
                ek = str(e)
                for fam in ("mlp", "vqc"):
                    und = pgd(
                        zoo.grad_provider(fam),
                        Xte,
                        yte,
                        e,
                        scale,
                        n_steps=pgd_cfg["n_steps"],
                        step_frac=pgd_cfg["step_frac"],
                        rng=spawn_rng(s, f"und_{fam}_{e}"),
                    )
                    dfn = pgd(
                        gp_def[fam],
                        Xte,
                        yte,
                        e,
                        scale,
                        n_steps=pgd_cfg["n_steps"],
                        step_frac=pgd_cfg["step_frac"],
                        rng=spawn_rng(s, f"def_{fam}_{e}"),
                    )
                    cell["undefended"][fam][ek] = _f1(yte, zoo.predict_proba(fam, und))
                    cell["defended"][fam][ek] = _f1(yte, proba_def(fam, dfn))
                # classical families are undefended throughout — they set the normalisation
                for c in CLASSICAL_TARGETS:
                    sur = pgd(
                        zoo.grad_provider("mlp"),
                        Xte,
                        yte,
                        e,
                        scale,
                        n_steps=pgd_cfg["n_steps"],
                        step_frac=pgd_cfg["step_frac"],
                        rng=spawn_rng(s, f"sur_{c}_{e}"),
                    )
                    cell["classical"].setdefault(c, {})["clean"] = _f1(
                        yte, zoo.predict_proba(c, Xte)
                    )
                    cell["classical"][c][ek] = _f1(yte, zoo.predict_proba(c, sur))
            per_seed.append(cell)
            run.write_json("results_partial.json", {"per_seed": per_seed})

        # ---- aggregate ----------------------------------------------------------------
        E = str(args.eval_eps[-1])

        def mean(f):
            return float(np.mean([f(c) for c in per_seed]))

        summary = {}
        for fam in ("mlp", "vqc"):
            summary[fam] = {
                "clean_undefended": mean(lambda c, f=fam: c["undefended"][f]["clean"]),
                "clean_defended": mean(lambda c, f=fam: c["defended"][f]["clean"]),
                "attacked_undefended": mean(lambda c, f=fam: c["undefended"][f][E]),
                "attacked_defended": mean(lambda c, f=fam: c["defended"][f][E]),
                "zero_radius_equals_clean": bool(
                    all(
                        abs(c["defended"][fam]["zero_radius"] - c["defended"][fam]["clean"]) < 1e-9
                        for c in per_seed
                    )
                ),
            }
            summary[fam]["clean_cost"] = (
                summary[fam]["clean_undefended"] - summary[fam]["clean_defended"]
            )
            summary[fam]["robustness_gain"] = (
                summary[fam]["attacked_defended"] - summary[fam]["attacked_undefended"]
            )

        cls_damage = float(
            np.mean(
                [
                    c["classical"][k]["clean"] - c["classical"][k][E]
                    for c in per_seed
                    for k in CLASSICAL_TARGETS
                ]
            )
        )
        nd_und = (summary["vqc"]["clean_undefended"] - summary["vqc"]["attacked_undefended"]) / (
            cls_damage if cls_damage else float("nan")
        )
        nd_def = (summary["vqc"]["clean_defended"] - summary["vqc"]["attacked_defended"]) / (
            cls_damage if cls_damage else float("nan")
        )

        controls = {
            "positive_A_mlp_clean": {
                "measured": summary["mlp"]["clean_undefended"],
                "published": PUBLISHED["mlp_clean"],
            },
            "positive_A_mlp_pgd": {
                "measured": summary["mlp"]["attacked_undefended"],
                "published": PUBLISHED["mlp_pgd_0.5"],
            },
            "positive_A_vqc_clean": {
                "measured": summary["vqc"]["clean_undefended"],
                "published": PUBLISHED["vqc_clean"],
            },
            "positive_B_round1_vs_e12": {
                "mlp_round1_clean": float(np.mean([c["rounds"][0]["mlp_clean"] for c in per_seed])),
                "mlp_e12_defended": E12_DEFENDED["mlp_clean"],
                "vqc_round1_clean": float(np.mean([c["rounds"][0]["vqc_clean"] for c in per_seed])),
                "vqc_e12_defended": E12_DEFENDED["vqc_clean"],
                "note": "reported, not blocking — R=1 is E12's construction under a different RNG "
                "stream name, so agreement in level is expected and exact identity is not",
            },
            "negative_zero_radius_no_op": {
                f: summary[f]["zero_radius_equals_clean"] for f in ("mlp", "vqc")
            },
        }
        controls["positive_A_pass"] = bool(
            all(
                abs(controls[k]["measured"] - controls[k]["published"]) <= CONTROL_TOL
                for k in ("positive_A_mlp_clean", "positive_A_mlp_pgd", "positive_A_vqc_clean")
            )
        )
        controls["negative_pass"] = bool(all(controls["negative_zero_radius_no_op"].values()))
        controls["pass"] = controls["positive_A_pass"] and controls["negative_pass"]

        defense_real = bool(
            max(summary["mlp"]["robustness_gain"], summary["vqc"]["robustness_gain"]) >= DEFENSE_BAR
        )
        picture_moved = bool(abs(nd_def - nd_und) >= PICTURE_BAR)
        verdict = {
            "controls": controls,
            "summary": summary,
            "round_trace_mean": [
                {
                    "round": r + 1,
                    **{
                        k: float(np.mean([c["rounds"][r][k] for c in per_seed]))
                        for k in ("mlp_clean", "mlp_attacked", "vqc_clean", "vqc_attacked")
                    },
                }
                for r in range(args.rounds)
            ],
            "mean_classical_damage": cls_damage,
            "normalized_vqc_damage_undefended": nd_und,
            "normalized_vqc_damage_defended": nd_def,
            "q1_defense_is_real": defense_real,
            "q2_picture_changed": picture_moved if defense_real else None,
            "q2_status": "read" if defense_real else "UNTESTED — the defense itself is a null",
            "q3_clean_cost": {f: summary[f]["clean_cost"] for f in ("mlp", "vqc")},
            "defense_bar": DEFENSE_BAR,
            "picture_bar": PICTURE_BAR,
            "branch": ("madry_defense_is_real" if defense_real else "madry_defense_is_a_NULL"),
            "per_seed": per_seed,
        }
        run.write_json("results.json", verdict)
        run.metrics = {
            "mlp_robustness_gain": summary["mlp"]["robustness_gain"],
            "vqc_robustness_gain": summary["vqc"]["robustness_gain"],
        }
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2, default=str))
        log.info(
            "VERDICT %s | gains mlp %+0.4f vqc %+0.4f (bar %.2f) | nd %.3f -> %.3f",
            verdict["branch"],
            summary["mlp"]["robustness_gain"],
            summary["vqc"]["robustness_gain"],
            DEFENSE_BAR,
            nd_und,
            nd_def,
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
