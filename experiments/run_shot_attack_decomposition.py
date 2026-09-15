"""E4b — WHY does shot noise change the white-box picture? Boring causes first.

CONTEXT. E4 (artifact 20260822T075249Z_shot_attack_bridge_binary_q8) measured, on the
DEPLOYED tuned QSVM (bandwidth 0.793, C = 835.2): exact retention 0.069 (reproducing the
published 0.064), but retention 0.900 at 1024 shots — while CLEAN macro-F1 fell 0.563 ->
0.482, i.e. below the ~0.50 stratified-random floor. A retention ratio computed between two
collapsed numbers is exactly the trap the paper's own Table II caption warns about, so the
E4 verdict may not be read until the cause is identified.

PRE-REGISTRATION (frozen before this run). Three candidate causes, two of them boring:
  (B1) TRAIN/TEST MISMATCH. E4 sampled only the EVALUATION Gram while the model was fit on
       the exact Gram. The published shot ablation (run_ablations.shot_study) samples BOTH.
       Prediction if B1 is the cause: sampling both restores clean macro-F1 to near its exact
       value, and retention returns toward the exact figure.
  (B2) REGULARIZATION AMPLIFICATION. The deployed fit has C = 835.2, so duals are pinned at a
       bound ~835x larger than the published ablation's C = 1.0 and residual kernel error is
       amplified into the decision (the mechanism qkernel.fidelity_kernel's docstring already
       records for bandwidth, and that P11d measured for C). Prediction if B2 is the cause:
       clean degradation under shots falls monotonically as C falls, and at C <= 10 the
       shot-sampled model retains its exact clean accuracy.
  (M)  MECHANISM. Neither: the fidelity kernel is intrinsically unusable at these shot counts
       regardless of fit, which would make the exact-simulation scoping unavoidable.

DECISION RULE (pre-committed):
  * If B1 alone explains it (clean drop under train+inference sampling <= 0.02 at the deployed
    C), E4's inference-only arm is a mis-specified deployment model; the reportable quantity
    is the train+inference retention, and E4's verdict is recomputed from it.
  * If B2 holds (clean drop at C <= 10 is <= 0.02 while at C = 835 it exceeds 0.05), the
    finding is that ACCURACY TUNING'S CHOICE OF C, not the encoding, decides shot-robustness —
    a second consequence of the knob P11d already indicts, and a new instance for the
    selection-protocol half of the checklist.
  * If neither, the exact-simulation scoping stands as E4 recorded it.

CONTROL (must pass): the published ablation's own configuration (bandwidth 1.0, C = 1.0, both
Grams sampled) must reproduce its reported insensitivity — clean macro-F1 within 0.02 of its
analytic value — or this runner, not the deployed model, is what differs.

Run:  python -m experiments.run_shot_attack_decomposition
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.run_adversarial import FittedZoo, _f1, _svc_on_kernel, _svc_proba
from qgridbench.attacks.evasion import pgd
from qgridbench.attacks.qsvm_whitebox import QSVMWhiteBox, assert_matches_reference
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel, shot_noise_kernel
from qgridbench.protocol import kernel_subset, prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import spawn_rng

log = get_logger(__name__)
SHOT_REPEATS = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="zz")
    ap.add_argument("--attack-seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--shots", type=int, nargs="*", default=[1024, 4096])
    ap.add_argument("--c-grid", type=float, nargs="*", default=[1.0, 10.0, 100.0])
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    subset_cap = cfg_q["qkernel"]["subset_cap"]
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    bw, C_tuned = float(best["qk_bw"]), float(best["qk_C"])
    eps = 0.5 if 0.5 in cfg_adv["evasion"]["epsilons"] else max(cfg_adv["evasion"]["epsilons"])
    pgd_cfg = cfg_adv["evasion"]["pgd"]

    reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
    zoo = FittedZoo(reg, args.encoding, args.qubits, 0, subset_cap, best)
    svc_dep, states_tr, angle = zoo.models["qsvm"], np.asarray(zoo._states_tr), zoo._angle
    wb = QSVMWhiteBox(svc_dep, states_tr, angle, args.encoding, bw)
    gate_idx = stratified_cap(reg.X["test"], reg.y["test"], 40, seed=0)
    assert_matches_reference(wb, svc_dep, states_tr, angle, reg.X["test"][gate_idx])

    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    X, y = reg.X["test"][ev], reg.y["test"][ev]
    ytr = zoo.ytr
    scale = reg.feature_scale

    maj = int(np.round(y.mean()))
    floor = _f1(y, np.column_stack([1 - np.full(len(y), maj), np.full(len(y), maj)]).astype(float))

    K_tr = fidelity_kernel(states_tr, states_tr)
    K_te_clean = fidelity_kernel(compute_states(angle.transform(X), args.encoding, bw), states_tr)
    K_te_adv = []
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
        K_te_adv.append(
            fidelity_kernel(compute_states(angle.transform(Xa), args.encoding, bw), states_tr)
        )
    log.info(
        "Grams built: train %s, test %s, %d attack sets",
        K_tr.shape,
        K_te_clean.shape,
        len(K_te_adv),
    )

    # ---- per-C WHITE-BOX attack sets --------------------------------------------------
    # CONFOUND THIS FIXES: reusing the deployed model's crafted set for every C turns the
    # low-C cells into TRANSFERRED (strictly weaker) attacks, which alone would produce
    # "retention rises as C falls". Each C must be attacked through its own decision
    # function for a retention comparison across C to mean anything.
    adv_by_C: dict[float, list[np.ndarray]] = {}

    def craft_for(C: float) -> list[np.ndarray]:
        if C in adv_by_C:
            return adv_by_C[C]
        svc_c = _svc_on_kernel(2, C).fit(K_tr, ytr)
        wb_c = QSVMWhiteBox(svc_c, states_tr, angle, args.encoding, bw)
        sets = []
        for a in args.attack_seeds:
            Xa = pgd(
                wb_c,
                X,
                y,
                eps,
                scale,
                n_steps=pgd_cfg["n_steps"],
                step_frac=pgd_cfg["step_frac"],
                rng=spawn_rng(a, "p5a"),
            )
            sets.append(
                fidelity_kernel(compute_states(angle.transform(Xa), args.encoding, bw), states_tr)
            )
        adv_by_C[C] = sets
        log.info("crafted per-C white-box sets for C=%g", C)
        return sets

    def evaluate(C: float, shots: int | None, sample_train: bool, whitebox: bool = True) -> dict:
        """One cell: fit at C (optionally on a sampled train Gram), evaluate clean + attacked.

        `whitebox=True` attacks each C through its OWN decision function; False reuses the
        deployed model's set, which is the transferred attack kept for contrast.
        """
        adv_sets = craft_for(C) if whitebox else K_te_adv
        cleans, atts = [], []
        reps = SHOT_REPEATS if shots else 1
        for rep in range(reps):
            rng = np.random.default_rng(2000 + rep)
            Ktr = shot_noise_kernel(K_tr, shots, rng) if (shots and sample_train) else K_tr
            svc = _svc_on_kernel(2, C).fit(Ktr, ytr)
            kc = shot_noise_kernel(K_te_clean, shots, rng, symmetric=False) if shots else K_te_clean
            cleans.append(_f1(y, _svc_proba(svc, kc, 2)))
            for Ka in adv_sets:
                ka = shot_noise_kernel(Ka, shots, rng, symmetric=False) if shots else Ka
                atts.append(_f1(y, _svc_proba(svc, ka, 2)))
        return {
            "clean_f1": float(np.mean(cleans)),
            "clean_f1_std": float(np.std(cleans, ddof=1)) if len(cleans) > 1 else 0.0,
            "attacked_f1": float(np.mean(atts)),
            "retention": float(np.mean(atts) / np.mean(cleans)),
            "clean_minus_floor": float(np.mean(cleans) - floor),
            "attacked_minus_floor": float(np.mean(atts) - floor),
            "attack": "whitebox_per_C" if whitebox else "transferred_from_deployed_C",
        }

    cfg = {
        "variant": args.variant,
        "qubits": args.qubits,
        "encoding": args.encoding,
        "bandwidth": bw,
        "C_tuned": C_tuned,
        "c_grid": args.c_grid,
        "shots": args.shots,
        "epsilon_sigma": eps,
        "attack_seeds": args.attack_seeds,
        "floor": floor,
    }

    with RunDir(f"shot_attack_decomposition_{args.variant}_q{args.qubits}", config=cfg) as run:
        grid = {}
        for C in sorted({*args.c_grid, C_tuned}):
            for shots in [None, *args.shots]:
                for sample_train in [False] if shots is None else [False, True]:
                    key = f"C={C:g}|shots={shots or 'exact'}|train_sampled={sample_train}"
                    grid[key] = evaluate(C, shots, sample_train, whitebox=True)
                    log.info(
                        "%-44s clean %.4f (floor %+.4f) attacked %.4f retention %.3f",
                        key,
                        grid[key]["clean_f1"],
                        grid[key]["clean_minus_floor"],
                        grid[key]["attacked_f1"],
                        grid[key]["retention"],
                    )
            # quantify the confound the first pass had: same C, TRANSFERRED attack
            tkey = f"C={C:g}|shots=exact|TRANSFERRED"
            grid[tkey] = evaluate(C, None, False, whitebox=False)
            log.info(
                "%-44s clean %.4f attacked %.4f retention %.3f  <- transferred contrast",
                tkey,
                grid[tkey]["clean_f1"],
                grid[tkey]["attacked_f1"],
                grid[tkey]["retention"],
            )
            run.write_json("results_partial.json", {"grid": grid})

        # ---- CONTROL: the published ablation's own configuration ----------------------
        sub = kernel_subset(reg, subset_cap, seed=0)
        A_tr = angle.transform(reg.X["train"][sub])
        st1 = compute_states(A_tr, args.encoding)  # bandwidth default 1.0, as published
        Ktr1 = fidelity_kernel(st1, st1)
        Kte1 = fidelity_kernel(compute_states(angle.transform(X), args.encoding), st1)
        ctrl = {}
        for shots in [None, *args.shots]:
            vals = []
            for rep in range(SHOT_REPEATS if shots else 1):
                rng = spawn_rng(rep, f"shots_{shots}")
                Ka = shot_noise_kernel(Ktr1, shots, rng) if shots else Ktr1
                Kb = shot_noise_kernel(Kte1, shots, rng, symmetric=False) if shots else Kte1
                s = _svc_on_kernel(2, 1.0).fit(Ka, ytr)
                vals.append(_f1(y, _svc_proba(s, Kb, 2)))
            ctrl[str(shots or "analytic")] = float(np.mean(vals))
        ctrl_drop = ctrl["analytic"] - ctrl[str(args.shots[0])]
        ctrl_pass = bool(abs(ctrl_drop) <= 0.02)
        log.info(
            "PUBLISHED-CONFIG CONTROL (bw 1.0, C 1.0, both Grams): %s -> drop %.4f pass=%s",
            ctrl,
            ctrl_drop,
            ctrl_pass,
        )

        # ---- read the pre-registered causes ------------------------------------------
        s0 = args.shots[0]
        exact_dep = grid[f"C={C_tuned:g}|shots=exact|train_sampled=False"]
        infonly_dep = grid[f"C={C_tuned:g}|shots={s0}|train_sampled=False"]
        both_dep = grid[f"C={C_tuned:g}|shots={s0}|train_sampled=True"]
        lowC = min(args.c_grid)
        exact_low = grid[f"C={lowC:g}|shots=exact|train_sampled=False"]
        both_low = grid[f"C={lowC:g}|shots={s0}|train_sampled=True"]

        b1_drop = exact_dep["clean_f1"] - both_dep["clean_f1"]
        b2_drop_low = exact_low["clean_f1"] - both_low["clean_f1"]
        # the C-vs-retention curve, white-box per C (the confound-free version)
        ret_by_C = {
            f"{C:g}": grid[f"C={C:g}|shots=exact|train_sampled=False"]["retention"]
            for C in sorted({*args.c_grid, C_tuned})
        }
        clean_by_C = {
            f"{C:g}": grid[f"C={C:g}|shots=exact|train_sampled=False"]["clean_f1"]
            for C in sorted({*args.c_grid, C_tuned})
        }
        transferred_by_C = {
            f"{C:g}": grid[f"C={C:g}|shots=exact|TRANSFERRED"]["retention"]
            for C in sorted({*args.c_grid, C_tuned})
        }
        verdict = {
            "whitebox_retention_by_C": ret_by_C,
            "clean_f1_by_C": clean_by_C,
            "transferred_retention_by_C": transferred_by_C,
            "clean_f1_span_across_C": float(max(clean_by_C.values()) - min(clean_by_C.values())),
            "retention_span_across_C": float(max(ret_by_C.values()) - min(ret_by_C.values())),
            "floor": floor,
            "deployed_C": C_tuned,
            "exact_clean_deployed": exact_dep["clean_f1"],
            "exact_retention_deployed": exact_dep["retention"],
            "inference_only_clean_drop": exact_dep["clean_f1"] - infonly_dep["clean_f1"],
            "train_and_inference_clean_drop": b1_drop,
            "B1_train_test_mismatch_explains": bool(b1_drop <= 0.02),
            "low_C_clean_drop": b2_drop_low,
            "B2_regularization_amplifies": bool(b1_drop > 0.05 and b2_drop_low <= 0.02),
            "retention_deployed_shots_both": both_dep["retention"],
            "retention_lowC_shots_both": both_low["retention"],
            "published_config_control": {"values": ctrl, "drop": ctrl_drop, "pass": ctrl_pass},
        }
        run.write_json("results.json", {"grid": grid, "control": ctrl, "verdict": verdict})
        run.metrics = {"train_and_inference_clean_drop": b1_drop, "low_C_clean_drop": b2_drop_low}
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
