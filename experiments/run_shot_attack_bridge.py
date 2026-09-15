"""E4 — shot/noise attack bridge: does the white-box kernel collapse survive shot noise?

PRE-REGISTRATION (pass-4 workplan item E4; frozen before the run).
The Discussion says "the shot-noise and depolarizing ablations are the bridge: both say the
conclusions are insensitive to realistic noise". Those ablations measured ACCURACY and KTA,
never an attack outcome. A shot-estimated kernel is stochastic at inference, and stochastic
decision paths are the mechanism by which noise acts as a defense (the paper itself cites
du2021quantum, where depolarizing noise provably improves a robustness bound). So the
headline white-box result — retention 0.064 — may be a property of exact simulation.

This run evaluates the SAME crafted white-box attack sets on noisy kernel decision paths.
The model is held fixed (trained on the exact Gram, exactly as published); only the
EVALUATION kernel is noisy. That isolates inference-time stochasticity, which is the channel
a smoothing defense would act through.

DECISION RULE on retention at eps=0.5 sigma with 1024 shots:
  retention_1024 <= 0.2   -> the claim survives; the bridge sentence upgrades to a measurement
  retention_1024 >= 0.4   -> the white-box result is scoped to exact simulation, and shot
                             noise is reported as an implicit defense (a finding either way)
  0.2 < r < 0.4           -> report as measured
Clean macro-F1 is recorded per arm: if the CLEAN score degrades > 0.05 under shots, that is
stated, because the comparison is RETENTION, not level.

CONTROLS (must pass before the rule is read):
  analytic   the shots=None arm must reproduce the published exact retention (0.064; attacked
             macro-F1 0.036 +/- 0.009) before any noisy arm is read
  numeric    assert_matches_reference gates the differentiable path, as in P5a

DEVIATION (logged): the published crafted sets were not archived as arrays, so they are
RE-CRAFTED here through the same code path with the same rng key (spawn_rng(seed, "p5a")),
which reproduces them exactly. Attack seeds default to 3 rather than the published 10 to
bound wall-clock; the analytic control checks the mean against the published figure.

Run:  python -m experiments.run_shot_attack_bridge --attack-seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.run_adversarial import FittedZoo, _f1, _svc_proba
from qgridbench.attacks.evasion import pgd
from qgridbench.attacks.qsvm_whitebox import QSVMWhiteBox, assert_matches_reference
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.models.quantum.qkernel import (
    compute_density_matrices,
    compute_states,
    fidelity_kernel,
    noisy_fidelity_kernel,
    shot_noise_kernel,
)
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import spawn_rng

log = get_logger(__name__)

# Published exact-path anchors (paper results sec. rq2 / P5a artifact).
PUBLISHED_ATTACKED_F1 = 0.036
PUBLISHED_RETENTION = 0.064
SHOT_REPEATS = 3  # sampling repetitions per shot arm (device-seed spread)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="zz")  # the deployed QSVM of Table II
    ap.add_argument("--attack-seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--shots", type=int, nargs="*", default=[1024, 4096])
    ap.add_argument("--p-depol", type=float, default=0.05)
    ap.add_argument("--skip-depol", action="store_true")
    ap.add_argument("--depol-train-cap", type=int, default=600)
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    subset_cap = cfg_q["qkernel"]["subset_cap"]
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    bw = float(best["qk_bw"])
    eps = 0.5 if 0.5 in cfg_adv["evasion"]["epsilons"] else max(cfg_adv["evasion"]["epsilons"])
    pgd_cfg = cfg_adv["evasion"]["pgd"]

    reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
    zoo = FittedZoo(reg, args.encoding, args.qubits, 0, subset_cap, best)
    svc = zoo.models["qsvm"]
    states_tr = np.asarray(zoo._states_tr)
    angle = zoo._angle
    wb = QSVMWhiteBox(svc, states_tr, angle, args.encoding, bw)

    gate_idx = stratified_cap(reg.X["test"], reg.y["test"], 40, seed=0)
    gate = assert_matches_reference(wb, svc, states_tr, angle, reg.X["test"][gate_idx])
    log.info("numeric gate: %s", {k: float(v) for k, v in gate.items() if "err" in k})

    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    X, y = reg.X["test"][ev], reg.y["test"][ev]
    scale = reg.feature_scale

    def exact_gram(Z):
        return fidelity_kernel(compute_states(angle.transform(Z), args.encoding, bw), states_tr)

    def f1_from_gram(K):
        return _f1(y, _svc_proba(svc, K, 2))

    # ---- craft the attack sets once (identical rng key -> the published sets) ---------
    K_clean = exact_gram(X)
    crafted = {}
    for a in args.attack_seeds:
        Xa = pgd(
            wb,
            X,
            y,
            eps,
            scale,
            n_steps=pgd_cfg["n_steps"],
            step_frac=pgd_cfg["step_frac"],
            rng=spawn_rng(a, "p5a"),
        )
        crafted[a] = exact_gram(Xa)
        log.info("crafted attack seed %d (exact-path F1 %.4f)", a, f1_from_gram(crafted[a]))

    cfg = {
        "variant": args.variant,
        "qubits": args.qubits,
        "encoding": args.encoding,
        "epsilon_sigma": eps,
        "attack_seeds": args.attack_seeds,
        "shots": args.shots,
        "p_depol": args.p_depol,
        "shot_repeats": SHOT_REPEATS,
        "n_eval": int(len(ev)),
        "published_anchor": {
            "attacked_f1": PUBLISHED_ATTACKED_F1,
            "retention": PUBLISHED_RETENTION,
        },
        "note": "model fixed (exact-Gram fit); only the EVALUATION kernel is noisy",
    }

    with RunDir(f"shot_attack_bridge_{args.variant}_q{args.qubits}", config=cfg) as run:
        arms = {}

        # ---- ARM 0: exact (analytic control) -----------------------------------------
        clean_exact = f1_from_gram(K_clean)
        att_exact = [f1_from_gram(crafted[a]) for a in args.attack_seeds]
        arms["exact"] = {
            "clean_f1_mean": clean_exact,
            "clean_f1_std": 0.0,
            "attacked_f1_mean": float(np.mean(att_exact)),
            "attacked_f1_std": float(np.std(att_exact, ddof=1)) if len(att_exact) > 1 else 0.0,
            "retention": float(np.mean(att_exact) / clean_exact),
            "per_seed": [float(v) for v in att_exact],
        }
        log.info("ARM exact: %s", arms["exact"])

        control_delta = arms["exact"]["attacked_f1_mean"] - PUBLISHED_ATTACKED_F1
        control_pass = abs(control_delta) <= 0.02  # published spread is +/-0.009
        log.info(
            "ANALYTIC CONTROL: attacked F1 %.4f vs published %.3f (delta %+.4f) -> pass=%s",
            arms["exact"]["attacked_f1_mean"],
            PUBLISHED_ATTACKED_F1,
            control_delta,
            control_pass,
        )
        run.write_json("results_partial.json", {"arms": arms})

        # ---- ARMS 1..n: shot-sampled kernels at inference -----------------------------
        for shots in args.shots:
            cleans, atts = [], []
            for rep in range(SHOT_REPEATS):
                rng = np.random.default_rng(1000 + rep)
                cleans.append(f1_from_gram(shot_noise_kernel(K_clean, shots, rng)))
                atts.extend(
                    f1_from_gram(shot_noise_kernel(crafted[a], shots, rng))
                    for a in args.attack_seeds
                )
            arms[f"shots_{shots}"] = {
                "clean_f1_mean": float(np.mean(cleans)),
                "clean_f1_std": float(np.std(cleans, ddof=1)),
                "attacked_f1_mean": float(np.mean(atts)),
                "attacked_f1_std": float(np.std(atts, ddof=1)),
                "retention": float(np.mean(atts) / np.mean(cleans)),
                "n_repeats": SHOT_REPEATS,
            }
            log.info("ARM shots=%d: %s", shots, arms[f"shots_{shots}"])
            run.write_json("results_partial.json", {"arms": arms})

        # ---- ARM depol: mixed-state kernel at inference -------------------------------
        if not args.skip_depol:
            # 4**d complex per sample: reduce the train reference to bound memory, and use
            # the SAME reduced reference for clean and attacked so retention stays matched.
            sub = np.sort(
                np.random.default_rng(0).choice(
                    len(states_tr), size=min(args.depol_train_cap, len(states_tr)), replace=False
                )
            )
            log.info("depolarizing arm: %d train references, p=%.3f", len(sub), args.p_depol)
            rho_tr = compute_density_matrices(zoo.A_tr[sub], args.encoding, args.p_depol, bw)

            def depol_f1(Z, rt=rho_tr, s=sub):
                rho = compute_density_matrices(angle.transform(Z), args.encoding, args.p_depol, bw)
                Kn = noisy_fidelity_kernel(rho, rt)
                # decision through the reduced reference set: renormalize the dual weights
                # to the retained columns (documented approximation, reported as such)
                return _f1(y, _svc_proba_reduced(svc, Kn, s))

            arms["depolarizing"] = {
                "p_depol": args.p_depol,
                "n_train_reference": int(len(sub)),
                "clean_f1": depol_f1(X),
                "approximation": "reduced train reference; dual weights restricted to it",
            }
            att_d = []
            for a in args.attack_seeds:
                Xa = pgd(
                    wb,
                    X,
                    y,
                    eps,
                    scale,
                    n_steps=pgd_cfg["n_steps"],
                    step_frac=pgd_cfg["step_frac"],
                    rng=spawn_rng(a, "p5a"),
                )
                att_d.append(depol_f1(Xa))
            arms["depolarizing"]["attacked_f1_mean"] = float(np.mean(att_d))
            arms["depolarizing"]["retention"] = float(
                np.mean(att_d) / arms["depolarizing"]["clean_f1"]
            )
            log.info("ARM depolarizing: %s", arms["depolarizing"])

        # ---- verdict ------------------------------------------------------------------
        r1024 = arms.get("shots_1024", {}).get("retention", float("nan"))
        clean_drop = clean_exact - arms.get("shots_1024", {}).get("clean_f1_mean", clean_exact)
        branch = (
            "claim_survives"
            if r1024 <= 0.2
            else ("scope_to_exact_simulation" if r1024 >= 0.4 else "report_as_measured")
        )
        verdict = {
            "analytic_control_delta": float(control_delta),
            "analytic_control_pass": bool(control_pass),
            "retention_exact": arms["exact"]["retention"],
            "retention_1024": r1024,
            "retention_4096": arms.get("shots_4096", {}).get("retention"),
            "clean_f1_drop_under_1024_shots": float(clean_drop),
            "clean_degraded_over_0.05": bool(clean_drop > 0.05),
            "branch": branch,
            "rule": "r<=0.2 survives; r>=0.4 scope to exact simulation; else report as measured",
        }
        run.write_json("results.json", {"arms": arms, "verdict": verdict, "numeric_gate": gate})
        run.metrics = {"retention_exact": arms["exact"]["retention"], "retention_1024": r1024}
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


def _svc_proba_reduced(svc, K_reduced: np.ndarray, sub: np.ndarray):
    """Decision through a REDUCED train reference set.

    The fitted SVC carries duals over its own support indices; restricting the Gram to a
    subset of training columns means only the support vectors inside `sub` contribute. This
    is an approximation, declared in the artifact — it exists because a full 2,000-column
    mixed-state Gram at 8 qubits is 4**8 complex per sample.
    """
    duals = np.zeros(len(sub))
    pos = {int(t): i for i, t in enumerate(sub)}
    for d, s_idx in zip(svc.dual_coef_[0], svc.support_):
        if int(s_idx) in pos:
            duals[pos[int(s_idx)]] = d
    dec = K_reduced @ duals + svc.intercept_[0]
    p1 = 1.0 / (1.0 + np.exp(-dec))
    return np.column_stack([1 - p1, p1])


if __name__ == "__main__":
    main()
