"""E12 — does a defense change the cross-family picture? (adversarial data augmentation)

PRE-REGISTRATION (pass-4 workplan item E12; frozen before the run).
The paper ships no defended model, and a reviewer will ask whether the cross-family readings
survive one. This run adds the cheapest honest defense to the two DIFFERENTIABLE families
(MLP, VQC) and re-reads the same quantities.

SCOPE, stated up front and never softened later: this is single-round adversarial data
AUGMENTATION — craft PGD at a training radius against the clean model, append those rows to the
training set, refit — NOT full Madry min-max adversarial training. Augmentation is the weaker
defense. A null result here therefore licenses "augmentation does not change the picture", never
"no defense changes the picture", and the log entry must say so.

EVALUATION DISCIPLINE (the classic adversarial-training reporting flaw, avoided by
construction): the defended model is attacked with examples crafted AGAINST THE DEFENDED MODEL.
Re-using the undefended model's adversarial examples would manufacture a robustness gain out of
gradient mismatch alone.

THREE PRE-REGISTERED QUESTIONS, rules fixed now.
  (1) does the defense do anything?  defended minus undefended macro-F1 under white-box PGD at
      0.5 sigma >= 0.05 for at least one family -> the defense is real; if neither moves, the
      run reports a null DEFENSE, not a null cross-family finding, and questions (2)-(3) are
      read as untested rather than answered
  (2) does it change the cross-family picture?  the VQC's damage normalized by mean classical
      damage moves by >= 0.10 between the undefended and defended fits -> the cross-family
      reading is defense-dependent and the paper says so
  (3) what does it cost?  clean macro-F1 delta, reported either way, no threshold

CONTROLS.
  positive (blocking)  the UNDEFENDED arm must reproduce the published 10-seed figures within
                       0.03: MLP clean 0.4653 and 0.2890 under PGD at 0.5 sigma, VQC clean
                       0.4754 (artifact `20260813T183448Z_adversarial_binary_q8`)
  negative             every defended model at eps = 0 must return its own clean macro-F1
                       exactly; a nonzero-radius attack path that fires at zero radius is broken

Run:  python -m experiments.run_adversarial_augmentation --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.run_adversarial import CLASSICAL, FittedZoo, _f1
from qgridbench.attacks.evasion import pgd
from qgridbench.attacks.gradients import GradProvider
from qgridbench.models.classical.zoo import build, fit_model
from qgridbench.models.quantum.vqc import VQClassifier
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)

VARIANT = "binary"
CLASSICAL_TARGETS = [c for c in CLASSICAL if c != "mlp"]
PUBLISHED = {"mlp_clean": 0.4653, "mlp_pgd_0.5": 0.2890, "vqc_clean": 0.4754}
CONTROL_TOL = 0.03
DEFENSE_BAR = 0.05  # (1) minimum robustness gain for the defense to count as real
PICTURE_BAR = 0.10  # (2) minimum move in normalized VQC damage to call the picture changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="zz")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--train-eps", type=float, default=0.5, help="radius the defense trains at")
    ap.add_argument("--eval-eps", type=float, nargs="*", default=[0.1, 0.3, 0.5])
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    pgd_cfg = cfg_adv["evasion"]["pgd"]

    cfg = {
        "variant": args.variant,
        "qubits": args.qubits,
        "defense": "single-round adversarial data augmentation (NOT Madry min-max)",
        "train_eps": args.train_eps,
        "eval_eps": args.eval_eps,
        "seeds": args.seeds,
        "published_reference": PUBLISHED,
        "defense_bar": DEFENSE_BAR,
        "picture_bar": PICTURE_BAR,
    }

    with RunDir(
        f"adversarial_augmentation_{args.variant}_q{args.qubits}", config=cfg, seeds=args.seeds
    ) as run:
        reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
        scale = reg.feature_scale
        from qgridbench.models.classical.zoo import stratified_cap

        ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
        Xte, yte = reg.X["test"][ev], reg.y["test"][ev]

        per_seed = []
        for s in args.seeds:
            zoo = FittedZoo(
                reg, args.encoding, args.qubits, s, cfg_q["qkernel"]["subset_cap"], best
            )
            Xtr, ytr = zoo.Xtr, zoo.ytr
            rng = spawn_rng(s, "augment")

            # ---- craft the augmentation set against the CLEAN models -----------------
            aug = {}
            for fam, gp in (("mlp", zoo.grad_provider("mlp")), ("vqc", zoo.grad_provider("vqc"))):
                aug[fam] = pgd(
                    gp,
                    Xtr,
                    ytr,
                    args.train_eps,
                    scale,
                    n_steps=pgd_cfg["n_steps"],
                    step_frac=pgd_cfg["step_frac"],
                    rng=rng,
                )
                log.info("s%d crafted %s augmentation set (%d rows)", s, fam, len(aug[fam]))

            # ---- refit each family on clean + adversarial ----------------------------
            X_aug_mlp = np.vstack([Xtr, aug["mlp"]])
            y_aug = np.concatenate([ytr, ytr])
            set_all_seeds(s)
            mlp_def = build("mlp", best["mlp"], s)
            fit_model("mlp", mlp_def, X_aug_mlp, y_aug)

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
            A_aug = zoo._to_angles(np.vstack([Xtr, aug["vqc"]]))
            va = stratified_cap(reg.X["val"], reg.y["val"], 2000, seed=0)
            vqc_def.fit(A_aug, y_aug, zoo._to_angles(reg.X["val"][va]), reg.y["val"][va])
            log.info("s%d refit both defended models", s)

            def proba_def(name, Z):
                if name == "mlp":
                    return mlp_def.predict_proba(Z)
                return vqc_def.predict_proba(zoo._to_angles(Z))

            gp_def = {
                "mlp": GradProvider(mlp_def, "mlp"),
                "vqc": (
                    lambda Z, yy, _v=vqc_def, _z=zoo: (
                        GradProvider(_v, "vqc")(_z._to_angles(Z), yy) * _z._angle._mm.scale_
                    )
                ),
            }

            cell = {"seed": s, "undefended": {}, "defended": {}, "cross_family": {}}
            for fam in ("mlp", "vqc"):
                cell["undefended"][fam] = {"clean": _f1(yte, zoo.predict_proba(fam, Xte))}
                cell["defended"][fam] = {"clean": _f1(yte, proba_def(fam, Xte))}
            # negative control: the attack path at zero radius must be a no-op
            for fam in ("mlp", "vqc"):
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
                    # each model is attacked with examples crafted against ITSELF
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
                    if fam == "vqc":
                        # cross-family: classical damage under the SAME crafted sets
                        cell["cross_family"].setdefault(ek, {})
                        for tag, Xa in (("undefended", und), ("defended", dfn)):
                            cls = float(
                                np.mean(
                                    [
                                        _f1(yte, zoo.predict_proba(t, Xte))
                                        - _f1(yte, zoo.predict_proba(t, Xa))
                                        for t in CLASSICAL_TARGETS
                                    ]
                                )
                            )
                            vq = (
                                cell["undefended"]["vqc"]["clean"] - cell["undefended"]["vqc"][ek]
                                if tag == "undefended"
                                else cell["defended"]["vqc"]["clean"] - cell["defended"]["vqc"][ek]
                            )
                            cell["cross_family"][ek][tag] = {
                                "classical_damage": cls,
                                "vqc_damage": vq,
                                "normalized_vqc_damage": float(vq / max(cls, 1e-9)),
                                "classical_damage_below_floor": bool(cls < 0.02),
                            }
                log.info(
                    "s%d eps %-4s | MLP %.4f -> %.4f | VQC %.4f -> %.4f (undefended -> defended)",
                    s,
                    ek,
                    cell["undefended"]["mlp"][ek],
                    cell["defended"]["mlp"][ek],
                    cell["undefended"]["vqc"][ek],
                    cell["defended"]["vqc"][ek],
                )
            per_seed.append(cell)
            run.write_json("results_partial.json", {"per_seed": per_seed})

        # ---- aggregate ---------------------------------------------------------------
        def m(path):
            out = []
            for c in per_seed:
                v = c
                for k in path:
                    v = v[k]
                out.append(v)
            return float(np.mean(out))

        head = str(max(args.eval_eps))
        summary = {
            fam: {
                "clean_undefended": m(["undefended", fam, "clean"]),
                "clean_defended": m(["defended", fam, "clean"]),
                "clean_cost": m(["undefended", fam, "clean"]) - m(["defended", fam, "clean"]),
                "attacked_undefended": m(["undefended", fam, head]),
                "attacked_defended": m(["defended", fam, head]),
                "robustness_gain": m(["defended", fam, head]) - m(["undefended", fam, head]),
                "zero_radius_equals_clean": bool(
                    abs(m(["defended", fam, "zero_radius"]) - m(["defended", fam, "clean"])) < 1e-9
                ),
            }
            for fam in ("mlp", "vqc")
        }
        norm_und = m(["cross_family", head, "undefended", "normalized_vqc_damage"])
        norm_def = m(["cross_family", head, "defended", "normalized_vqc_damage"])

        controls = {
            "positive_mlp_clean": {
                "measured": summary["mlp"]["clean_undefended"],
                "published": PUBLISHED["mlp_clean"],
            },
            "positive_mlp_pgd": {
                "measured": summary["mlp"]["attacked_undefended"],
                "published": PUBLISHED["mlp_pgd_0.5"],
            },
            "positive_vqc_clean": {
                "measured": summary["vqc"]["clean_undefended"],
                "published": PUBLISHED["vqc_clean"],
            },
            "negative_zero_radius_no_op": {
                fam: summary[fam]["zero_radius_equals_clean"] for fam in ("mlp", "vqc")
            },
        }
        controls["positive_pass"] = bool(
            abs(summary["mlp"]["clean_undefended"] - PUBLISHED["mlp_clean"]) <= CONTROL_TOL
            and abs(summary["mlp"]["attacked_undefended"] - PUBLISHED["mlp_pgd_0.5"]) <= CONTROL_TOL
            and abs(summary["vqc"]["clean_undefended"] - PUBLISHED["vqc_clean"]) <= CONTROL_TOL
        )
        controls["negative_pass"] = bool(all(controls["negative_zero_radius_no_op"].values()))
        controls["pass"] = bool(controls["positive_pass"] and controls["negative_pass"])
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        defense_real = [f for f in ("mlp", "vqc") if summary[f]["robustness_gain"] >= DEFENSE_BAR]
        picture_moved = abs(norm_def - norm_und) >= PICTURE_BAR
        verdict = {
            "controls": controls,
            "summary": summary,
            "normalized_vqc_damage_undefended": norm_und,
            "normalized_vqc_damage_defended": norm_def,
            "normalized_move": float(norm_def - norm_und),
            "q1_defense_is_real_for": defense_real,
            "q2_cross_family_picture_moved": bool(picture_moved and defense_real),
            "q3_clean_cost": {f: summary[f]["clean_cost"] for f in ("mlp", "vqc")},
            "branch": (
                "no_defense_effect_questions_2_3_untested"
                if not defense_real
                else (
                    "picture_is_defense_dependent"
                    if picture_moved
                    else "picture_survives_the_defense"
                )
            ),
            "scope": (
                "single-round adversarial data augmentation, NOT Madry min-max training; a null "
                "here licenses 'augmentation does not change the picture', never 'no defense does'"
            ),
        }
        run.write_json("results.json", {"per_seed": per_seed, "verdict": verdict})
        run.metrics = {
            "vqc_robustness_gain": summary["vqc"]["robustness_gain"],
            "mlp_robustness_gain": summary["mlp"]["robustness_gain"],
        }
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
