"""E2 — worst-case attack aggregation: does the published mean understate every family?

PRE-REGISTRATION (pass-4 workplan item E2; frozen before the run).
Methodology states PGD runs "from one random start inside the ball, with no restarts", and
the reported figure averages over seeds. The strongest-attack convention in the canon this
paper cites (carlini2019evaluating) is PER-POINT WORST CASE over restarts: a point counts as
attacked if ANY restart flips it. The mean therefore understates attack strength for every
family, and the size of that understatement is unmeasured.

DESIGN. The published 10 "seeds" refit the models as well as re-drawing the PGD start, so a
per-point maximum over them would mix model variation into an attack statistic. The clean
form of the question holds the MODEL fixed (seed 0) and takes 10 PGD RESTARTS, which is what
the convention actually prescribes. Attack assignment mirrors the shipped protocol exactly:
white-box for the MLP and the VQC, MLP-surrogate transfer for every other family.

DECISION RULE: if no cross-family ordering at eps = 0.5 sigma flips between the mean and the
worst-case aggregation, one v2 sentence reports both and nothing else changes. If ANY
ordering flips, that is SURPRISING: re-derive through an independent aggregation
implementation before logging, then escalate — a benchmark whose family ranking depends on
the aggregation convention is a finding, not a footnote.

CONTROL (must pass before the rule is read): the mean-over-restarts figure at seed 0 must sit
inside the published across-seed spread for each family (Table II, eps = 0.5 sigma column).
Exact reproduction is NOT expected and is not the bar — the published protocol varies the
model too.

Both aggregations are computed from the SAME hard predictions so the comparison cannot be a
probability-versus-argmax artifact.

Run:  python -m experiments.run_evasion_max_aggregation
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import f1_score

from experiments.run_adversarial import CLASSICAL, FittedZoo
from qgridbench.attacks.evasion import pgd
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import spawn_rng

log = get_logger(__name__)

FAMILIES = CLASSICAL + ["qsvm", "vqc"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="angle_ry")  # shipped adversarial-suite encoding
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--model-seed", type=int, default=0)
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    subset_cap = cfg_q["qkernel"]["subset_cap"]
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    eps = 0.5 if 0.5 in cfg_adv["evasion"]["epsilons"] else max(cfg_adv["evasion"]["epsilons"])
    pgd_cfg = cfg_adv["evasion"]["pgd"]
    sources = cfg_adv["evasion"]["white_box_models"]  # [mlp, vqc]
    surrogate = cfg_adv["evasion"]["surrogate_model"]  # mlp

    reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
    zoo = FittedZoo(reg, args.encoding, args.qubits, args.model_seed, subset_cap, best)
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    X, y = reg.X["test"][ev], reg.y["test"][ev]
    scale = reg.feature_scale

    def preds(name, Z):
        return zoo.predict_proba(name, Z).argmax(axis=1)

    clean_pred = {f: preds(f, X) for f in FAMILIES}
    clean_f1 = {f: float(f1_score(y, clean_pred[f], average="macro")) for f in FAMILIES}
    log.info("clean macro-F1: %s", {k: round(v, 4) for k, v in clean_f1.items()})

    cfg = {
        "variant": args.variant,
        "qubits": args.qubits,
        "encoding": args.encoding,
        "epsilon_sigma": eps,
        "restarts": args.restarts,
        "model_seed": args.model_seed,
        "n_eval": int(len(ev)),
        "protocol": "white-box for mlp/vqc; MLP-surrogate transfer for the rest (as shipped)",
    }

    with RunDir(f"evasion_max_agg_{args.variant}_q{args.qubits}", config=cfg) as run:
        # per family: predictions under each restart
        per_restart: dict[str, list[np.ndarray]] = {f: [] for f in FAMILIES}
        for r in range(args.restarts):
            rng = spawn_rng(r, "e2_restart")
            # white-box sources
            adv_sets = {}
            for src in sources:
                gp = zoo.grad_provider(src)
                adv_sets[src] = pgd(
                    gp,
                    X,
                    y,
                    eps,
                    scale,
                    n_steps=pgd_cfg["n_steps"],
                    step_frac=pgd_cfg["step_frac"],
                    rng=rng,
                )
            # surrogate transfer for everyone else (one crafted set, as shipped)
            gp_s = zoo.grad_provider(surrogate)
            X_tr = pgd(
                gp_s,
                X,
                y,
                eps,
                scale,
                n_steps=pgd_cfg["n_steps"],
                step_frac=pgd_cfg["step_frac"],
                rng=rng,
            )
            for f in FAMILIES:
                Z = adv_sets[f] if f in sources else X_tr
                per_restart[f].append(preds(f, Z))
            log.info("restart %d/%d done", r + 1, args.restarts)
            run.write_json("results_partial.json", {"restarts_done": r + 1})

        # ---- the two aggregations, from identical hard predictions --------------------
        out = {}
        for f in FAMILIES:
            P = np.stack(per_restart[f])  # (restarts, n)
            per_restart_f1 = [float(f1_score(y, p, average="macro")) for p in P]
            wrong = P != y[None, :]
            # worst case: first wrong prediction per point, else restart 0's prediction
            worst = P[0].copy()
            any_wrong = wrong.any(axis=0)
            first_wrong = wrong.argmax(axis=0)
            worst[any_wrong] = P[first_wrong[any_wrong], np.flatnonzero(any_wrong)]
            out[f] = {
                "clean_f1": clean_f1[f],
                "mean_f1": float(np.mean(per_restart_f1)),
                "std_f1": float(np.std(per_restart_f1, ddof=1)),
                "worstcase_f1": float(f1_score(y, worst, average="macro")),
                "mean_retention": float(np.mean(per_restart_f1) / clean_f1[f]),
                "worstcase_retention": float(f1_score(y, worst, average="macro") / clean_f1[f]),
                "per_restart_f1": per_restart_f1,
                "attack": "white-box" if f in sources else "mlp-surrogate transfer",
            }
            log.info(
                "%-8s clean %.4f | mean %.4f +/- %.4f (ret %.3f) | worst-case %.4f (ret %.3f)",
                f,
                clean_f1[f],
                out[f]["mean_f1"],
                out[f]["std_f1"],
                out[f]["mean_retention"],
                out[f]["worstcase_f1"],
                out[f]["worstcase_retention"],
            )

        order_mean = sorted(FAMILIES, key=lambda f: out[f]["mean_retention"], reverse=True)
        order_worst = sorted(FAMILIES, key=lambda f: out[f]["worstcase_retention"], reverse=True)
        flips = [
            (a, b)
            for i, a in enumerate(order_mean)
            for b in order_mean[i + 1 :]
            if order_worst.index(a) > order_worst.index(b)
        ]
        verdict = {
            "ordering_mean": order_mean,
            "ordering_worstcase": order_worst,
            "pairwise_flips": [list(p) for p in flips],
            "n_pairwise_flips": len(flips),
            "any_ordering_flip": bool(flips),
            "mean_minus_worstcase_f1": {
                f: float(out[f]["mean_f1"] - out[f]["worstcase_f1"]) for f in FAMILIES
            },
            "worstcase_lower_for_all": bool(
                all(out[f]["worstcase_f1"] <= out[f]["mean_f1"] + 1e-12 for f in FAMILIES)
            ),
            "branch": ("report_both_one_sentence" if not flips else "SURPRISING_escalate"),
            "rule": "no ordering flip -> one v2 sentence; any flip -> re-derive and escalate",
        }
        run.write_json("results.json", {"per_family": out, "verdict": verdict})
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
