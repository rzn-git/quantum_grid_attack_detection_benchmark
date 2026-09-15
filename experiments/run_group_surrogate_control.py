"""E7 — does the surrogate-specification instance survive leakage-free training?

PRE-REGISTRATION (pass-4 workplan item E7; frozen before the run).
The paper's second-largest evaluator-choice instance is the surrogate: an MLP fitted without
the imbalance policy its estimator carries no argument for manufactures a 10.2x
forward/reverse transfer asymmetry, and refitting it with balanced sample weights collapses
that to 0.8x while leaving classical damage unchanged. Like six of the other seven instances,
it was measured only on detectors trained under the row-level protocol the paper deprecates.
Models that partly memorize near-duplicate rows may have unrepresentative gradient structure,
so the instance needs replication on whole-file-trained models.

THREE PRE-REGISTERED PREDICTIONS (all three must hold for "protocol-independent"):
  (i)   the UNWEIGHTED surrogate's forward/reverse ratio is >= 3x
  (ii)  the BALANCED surrogate's ratio lands in [0.5, 1.5]
  (iii) classical damage is flat within 0.03 between the two fits (so the corrected surrogate
        is not merely a weaker attack — the load-bearing half of the published claim)
If any prediction fails, the instance gains a protocol-scope qualifier in the checklist. That
is a reportable finding, not a failure.

CONTROL (blocking, and this is exactly the check P28 failed on 2026-08-20): the identical code
path on the SHIPPED row-level split must reproduce the published ratios (10.16x unweighted,
0.78x balanced) within 20% before any group-split number is read.

STATISTIC NOTE (E11, 2026-08-22T08:12Z): a forward/reverse ratio is unstable when the reverse
drop approaches zero. This runner therefore uses the PUBLISHED aggregation — ratio of
seed-MEAN drops, exactly as run_surrogate_control computes it, not a mean of per-seed ratios —
and additionally reports damage normalized by classical damage, which is the bounded statistic
the paper already prefers in tab:surrogate.

ARM ORDER IS LOAD-BEARING: the four crafting calls share one rng stream, so the order below is
the published one and a new arm may only be appended LAST.

Run:  python -m experiments.run_group_surrogate_control --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.run_adversarial import CLASSICAL, FittedZoo, _f1
from experiments.run_group_split_control import group_splits, reconstruct_groups
from qgridbench.attacks.evasion import pgd
from qgridbench.attacks.gradients import GradProvider
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import AngleScaler, build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.protocol import Regime, prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)

CLASSICAL_TARGETS = [c for c in CLASSICAL if c != "mlp"]
PUBLISHED = {"mlp": 10.16, "mlp_balanced": 0.78}
CONTROL_TOL = 0.20  # relative

# A forward/reverse ratio is not a measurement when its denominator is at zero: it returns a
# large number that reads like a strong effect. This bit the project three times on
# 2026-08-22 (E11 diagnosis 08:12Z, E8 prediction (i), E7 prediction (i)), so any ratio whose
# denominator falls below this floor is emitted with a `vacuous` flag instead of being trusted.
REVERSE_DAMAGE_FLOOR = 0.02  # macro-F1; below this the reverse drop is indistinguishable from 0


def prepare_group_regime(partition: int, n_components: int = 8) -> Regime:
    """A Regime built on a WHOLE-FILE split — prepare_regime's body, group indices."""
    X, y, gid = reconstruct_groups()
    rng = np.random.default_rng(0)  # same partition stream as every group control
    sp = None
    for _ in range(partition + 1):
        sp, _held = group_splits(gid, y, rng)
    pipe = build_preprocessor("pca", n_components=n_components, seed=0)
    Z = fit_transform_splits(pipe, X, sp)
    ys = {s: y[sp[s]] for s in ("train", "val", "test")}
    scaler = AngleScaler().fit(Z["train"])
    angles = {s: scaler.transform(Z[s]) for s in ("train", "val", "test")}
    return Regime("pca", n_components, Z, ys, angles, scaler)


def run_arms(reg, encoding, qubits, seeds, subset_cap, best, cfg_adv, tag):
    """The published surrogate-control flow, unchanged, on whatever Regime it is handed."""
    eps = max(cfg_adv["evasion"]["epsilons"])
    pgd_cfg = cfg_adv["evasion"]["pgd"]
    scale = reg.feature_scale
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    Xte, yte = reg.X["test"][ev], reg.y["test"][ev]
    floor = trivial_baselines(reg.y["train"], yte)["baseline_stratified_random"]["aggregate"][
        "macro_f1"
    ]["mean"]
    families = CLASSICAL + ["qsvm", "vqc"]

    per_seed = []
    for s in seeds:
        rng = spawn_rng(s, "adv")  # same stream family as the published run
        zoo = FittedZoo(reg, encoding, qubits, s, subset_cap, best)
        set_all_seeds(s)
        fixed = build("mlp", best["mlp"], s)
        fit_model("mlp", fixed, zoo.Xtr, zoo.ytr, balance_via_sample_weight=True)

        def proba(name, X, _f=fixed, _z=zoo):
            return _f.predict_proba(X) if name == "mlp_balanced" else _z.predict_proba(name, X)

        models = families + ["mlp_balanced"]
        sets = {"clean": Xte}
        for src, gp in [
            ("mlp", zoo.grad_provider("mlp")),
            ("mlp_balanced", GradProvider(fixed, "mlp")),
            ("vqc", zoo.grad_provider("vqc")),
            ("rbf_svm", GradProvider(zoo.models["rbf_svm"], "rbf_svm")),
        ]:
            sets[f"pgd_{src}"] = pgd(
                gp,
                Xte,
                yte,
                eps,
                scale,
                n_steps=pgd_cfg["n_steps"],
                step_frac=pgd_cfg["step_frac"],
                rng=rng,
            )
        f1 = {k: {m: _f1(yte, proba(m, X)) for m in models} for k, X in sets.items()}
        pos = {m: float(proba(m, Xte).argmax(1).mean()) for m in ("mlp", "mlp_balanced")}
        per_seed.append({"seed": s, "f1": f1, "surrogate_positive_rate": pos})
        log.info(
            "[%s] seed %d | surrogate clean %.3f -> %.3f | VQC under it %.3f -> %.3f",
            tag,
            s,
            f1["clean"]["mlp"],
            f1["clean"]["mlp_balanced"],
            f1["pgd_mlp"]["vqc"],
            f1["pgd_mlp_balanced"]["vqc"],
        )
    return per_seed, floor, families


def summarize(per_seed, floor, families):
    """The published aggregation: ratio of seed-MEAN drops (never a mean of ratios)."""

    def mn(fn):
        return float(np.mean([fn(p) for p in per_seed]))

    out = {"floor": floor}
    for src, name in (
        ("pgd_mlp", "mlp"),
        ("pgd_mlp_balanced", "mlp_balanced"),
        ("pgd_rbf_svm", "rbf_svm"),
    ):
        vq_c = mn(lambda p: p["f1"]["clean"]["vqc"])
        vq_a = mn(lambda p, s=src: p["f1"][s]["vqc"])
        cls_c = float(np.mean([mn(lambda p, t=t: p["f1"]["clean"][t]) for t in CLASSICAL_TARGETS]))
        cls_a = float(np.mean([mn(lambda p, s=src, t=t: p["f1"][s][t]) for t in CLASSICAL_TARGETS]))
        rev_c = mn(lambda p, n=name: p["f1"]["clean"][n])
        rev_a = mn(lambda p, n=name: p["f1"]["pgd_vqc"][n])
        out[name] = {
            "surrogate_clean": rev_c,
            "surrogate_clean_minus_floor": rev_c - floor,
            "vqc_damage": vq_c - vq_a,
            "classical_damage": cls_c - cls_a,
            "damage_ratio_vqc_over_classical": (vq_c - vq_a) / max(cls_c - cls_a, 1e-9),
            "reverse_damage": rev_c - rev_a,
            "forward_over_reverse": (vq_c - vq_a) / max(rev_c - rev_a, 1e-9),
            # never read the ratio when this is True — the denominator is at zero
            "forward_over_reverse_vacuous": bool((rev_c - rev_a) < REVERSE_DAMAGE_FLOOR),
            "reverse_damage_floor": REVERSE_DAMAGE_FLOOR,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="zz")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--partition", type=int, default=0)
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    cap = cfg_q["qkernel"]["subset_cap"]
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())

    cfg = {
        "variant": args.variant,
        "qubits": args.qubits,
        "encoding": args.encoding,
        "seeds": args.seeds,
        "partition": args.partition,
        "published_reference": PUBLISHED,
        "control_tolerance": CONTROL_TOL,
    }

    with RunDir(f"group_surrogate_control_{args.variant}_q8", config=cfg, seeds=args.seeds) as run:
        # ---- CONTROL first: row-level split must reproduce the published ratios ------
        reg_row = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
        ps_row, floor_row, fams = run_arms(
            reg_row, args.encoding, args.qubits, args.seeds, cap, best, cfg_adv, "row-level"
        )
        sum_row = summarize(ps_row, floor_row, fams)
        run.write_json("results_partial.json", {"row_level": sum_row})
        ctrl = {
            k: {
                "measured": sum_row[k]["forward_over_reverse"],
                "published": v,
                "rel_error": abs(sum_row[k]["forward_over_reverse"] - v) / v,
            }
            for k, v in PUBLISHED.items()
        }
        ctrl["pass"] = bool(
            all(c["rel_error"] <= CONTROL_TOL for c in ctrl.values() if isinstance(c, dict))
        )
        log.info("CONTROL (row-level reproduction): %s", json.dumps(ctrl, indent=2))

        # ---- the group-split arm ------------------------------------------------------
        reg_grp = prepare_group_regime(args.partition, args.qubits)
        ps_grp, floor_grp, _ = run_arms(
            reg_grp,
            args.encoding,
            args.qubits,
            args.seeds,
            cap,
            best,
            cfg_adv,
            f"group-p{args.partition}",
        )
        sum_grp = summarize(ps_grp, floor_grp, fams)

        u = sum_grp["mlp"]
        b = sum_grp["mlp_balanced"]
        preds = {
            # (i) is only admissible when its denominator clears the floor; a vacuous cell
            # counts as NOT holding, because a ratio at a zero denominator measures nothing.
            "i_unweighted_ge_3x": bool(
                u["forward_over_reverse"] >= 3.0 and not u["forward_over_reverse_vacuous"]
            ),
            "i_vacuous": u["forward_over_reverse_vacuous"],
            "i_value": u["forward_over_reverse"],
            "ii_balanced_in_band": bool(0.5 <= b["forward_over_reverse"] <= 1.5),
            "ii_value": b["forward_over_reverse"],
            "iii_classical_damage_flat": bool(
                abs(u["classical_damage"] - b["classical_damage"]) <= 0.03
            ),
            "iii_delta": float(u["classical_damage"] - b["classical_damage"]),
        }
        preds["n_holding"] = int(
            preds["i_unweighted_ge_3x"]
            + preds["ii_balanced_in_band"]
            + preds["iii_classical_damage_flat"]
        )
        verdict = {
            "control": ctrl,
            "predictions": preds,
            "branch": (
                "instance_replicates_leakage_free"
                if preds["n_holding"] == 3
                else "scope_qualifier_needed"
            ),
            "normalized_damage_row": {
                k: sum_row[k]["damage_ratio_vqc_over_classical"] for k in PUBLISHED
            },
            "normalized_damage_group": {
                k: sum_grp[k]["damage_ratio_vqc_over_classical"] for k in PUBLISHED
            },
            "rule": "all 3 predictions -> protocol-independent; else scope qualifier",
        }
        run.write_json(
            "results.json",
            {
                "row_level": {"summary": sum_row, "per_seed": ps_row},
                "group": {"summary": sum_grp, "per_seed": ps_grp},
                "verdict": verdict,
            },
        )
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
