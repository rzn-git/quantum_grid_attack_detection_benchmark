"""E14 — score-based (NES) black-box attack: are the tree retention figures upper bounds?

PRE-REGISTRATION (pass-4 workplan item E14; frozen before the run).
After P5a the tree ensembles are the only families still attacked by TRANSFER alone, so their
retention figures are upper bounds for exactly the reason the QSVM's was before it got a
white-box path. HopSkipJump closes part of the gap but is DECISION-based: it sees only the
predicted label. A score-based attack sees the probability and is strictly stronger, and it is
the attacker class the paper has never run. The paper states this gap rather than closing it.

THE ATTACK. NES gradient estimation (Ilyas et al. 2018) with antithetic sampling: at each step,
probe the model at n_samples/2 mirrored Gaussian offsets, form the smoothed gradient estimate of
the cross-entropy loss, take a signed step, project back into the L-inf ball. Queries are
counted and reported per sample. EQUAL BUDGET is the comparability contract, so the default
20 steps x 30 samples = 600 queries/sample is matched to the shipped HopSkipJump budget
(~610 queries/sample measured), not chosen to favour this attack.

DECISION RULE, fixed now, on macro-F1 damage at eps = 0.5 sigma against the published
transfer figures (10-seed means, artifact `20260813T183448Z_adversarial_binary_q8`:
rf 0.2487, xgb 0.2799, lgbm 0.2605):
  any tree family with (NES damage - transfer damage) >= 0.05
    -> the transfer figures ARE upper bounds; the gap is quantified and the caveat becomes a
       measured statement
  all three within 0.02
    -> the trees are genuinely hard for this attacker class at this budget, which retires the
       caveat rather than quantifying it
  in between -> reported as inconclusive at this budget, with the budget stated

CONTROLS.
  positive (blocking)  NES against the MLP must reach >= 70% of the MLP's own WHITE-BOX PGD
                       damage at the same radius (published 0.1763 at 0.5 sigma, so >= 0.123).
                       An attack that cannot hurt a model we know is attackable is broken, and
                       no tree number from it would mean anything.
  negative             a random perturbation at the matched radius must do LESS damage than NES
                       for every model. If noise matches the estimator, the estimator is noise.

Run:  python -m experiments.run_score_based_blackbox --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.run_adversarial import FittedZoo, _f1
from qgridbench.models.classical.zoo import stratified_cap
from qgridbench.protocol import prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import spawn_rng

log = get_logger(__name__)

VARIANT = "binary"
TREES = ["rf", "xgb", "lgbm"]
CONTROL_MODEL = "mlp"
# Published 10-seed means at eps = 0.5 sigma (artifact 20260813T183448Z_adversarial_binary_q8).
PUBLISHED_TRANSFER_DAMAGE = {"rf": 0.2487, "xgb": 0.2799, "lgbm": 0.2605}
PUBLISHED_MLP_WHITEBOX_DAMAGE = 0.1763
CONTROL_FRACTION = 0.70
UPPER_BOUND_BAR = 0.05
HARD_BAR = 0.02


def nes_attack(
    proba_fn,
    X: np.ndarray,
    y: np.ndarray,
    eps: float,
    scale: np.ndarray,
    rng: np.random.Generator,
    n_steps: int,
    n_samples: int,
    sigma_frac: float,
    step_frac: float,
) -> tuple[np.ndarray, dict]:
    """L-inf NES with antithetic sampling; returns (X_adv, budget_info)."""
    assert n_samples % 2 == 0, "antithetic sampling needs an even sample count"
    sigma = sigma_frac * scale
    alpha, lo, hi = step_frac * eps * scale, X - eps * scale, X + eps * scale
    Xa = np.clip(X + rng.uniform(-1.0, 1.0, X.shape) * eps * scale, lo, hi)
    n, d = X.shape
    queries = 0

    def loss(Z):  # cross-entropy of the TRUE label, the quantity the attacker maximizes
        p = np.clip(proba_fn(Z), 1e-9, 1 - 1e-9)
        return -np.log(p[np.arange(len(Z)), np.tile(y, len(Z) // n)])

    for _ in range(n_steps):
        g = np.zeros_like(Xa)
        for _ in range(n_samples // 2):
            u = rng.standard_normal((n, d))
            lp = loss(Xa + sigma * u)
            lm = loss(Xa - sigma * u)
            queries += 2 * n
            g += (lp - lm)[:, None] * u
        g /= n_samples * np.maximum(sigma, 1e-12)
        Xa = np.clip(Xa + alpha * np.sign(g), lo, hi)
    return Xa, {
        "queries_total": int(queries),
        "queries_per_sample": float(queries / n),
        "n_steps": n_steps,
        "n_samples": n_samples,
        "sigma_frac": sigma_frac,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="zz")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epsilons", type=float, nargs="*", default=[0.1, 0.3, 0.5])
    ap.add_argument("--nes-steps", type=int, default=20)
    ap.add_argument("--nes-samples", type=int, default=30)
    ap.add_argument("--nes-sigma-frac", type=float, default=0.05)
    ap.add_argument(
        "--nes-step-frac",
        type=float,
        default=None,
        help="override the shipped PGD step fraction — the CONVERGENCE control. A signed step "
        "of step_frac*eps grows with eps, so a noisy estimated gradient can do LESS damage at a "
        "larger radius; that is an attack-tuning artifact, not robustness.",
    )
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    step_frac = args.nes_step_frac or cfg_adv["evasion"]["pgd"]["step_frac"]
    models = TREES + [CONTROL_MODEL]

    cfg = {
        "variant": args.variant,
        "qubits": args.qubits,
        "models": models,
        "epsilons": args.epsilons,
        "nes": {
            "steps": args.nes_steps,
            "samples": args.nes_samples,
            "sigma_frac": args.nes_sigma_frac,
            "step_frac": step_frac,
            "budget_note": (
                "steps x samples = queries/sample, matched to the shipped HopSkipJump budget "
                "(~610 measured) so the comparison is equal-budget"
            ),
        },
        "published_transfer_damage": PUBLISHED_TRANSFER_DAMAGE,
        "published_mlp_whitebox_damage": PUBLISHED_MLP_WHITEBOX_DAMAGE,
        "seeds": args.seeds,
    }

    with RunDir(
        f"score_based_blackbox_{args.variant}_q{args.qubits}", config=cfg, seeds=args.seeds
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
            cell = {"seed": s, "clean": {}, "nes": {}, "random": {}, "budget": {}}
            for m in models:
                cell["clean"][m] = _f1(yte, zoo.predict_proba(m, Xte))
            for e in args.epsilons:
                ek = str(e)
                cell["nes"][ek], cell["random"][ek] = {}, {}
                for m in models:
                    # NEVER hash() here: Python salts string hashing per process, so the
                    # attack stream would not reproduce across runs (determinism rule).
                    rng = spawn_rng(s, f"nes_{m}_{e}")
                    Xa, budget = nes_attack(
                        lambda Z, _m=m, _z=zoo: _z.predict_proba(_m, Z),
                        Xte,
                        yte,
                        e,
                        scale,
                        rng,
                        args.nes_steps,
                        args.nes_samples,
                        args.nes_sigma_frac,
                        step_frac,
                    )
                    Xr = Xte + np.random.default_rng(500 + s).uniform(-1, 1, Xte.shape) * e * scale
                    cell["nes"][ek][m] = _f1(yte, zoo.predict_proba(m, Xa))
                    cell["random"][ek][m] = _f1(yte, zoo.predict_proba(m, Xr))
                    cell["budget"][m] = budget
                    log.info(
                        "s%d eps %-4s %-6s clean %.4f -> NES %.4f (damage %.4f) | random %.4f "
                        "| %.0f q/sample",
                        s,
                        ek,
                        m,
                        cell["clean"][m],
                        cell["nes"][ek][m],
                        cell["clean"][m] - cell["nes"][ek][m],
                        cell["random"][ek][m],
                        budget["queries_per_sample"],
                    )
                run.write_json("results_partial.json", {"per_seed": per_seed + [cell]})
            per_seed.append(cell)
            run.write_json("results_partial.json", {"per_seed": per_seed})

        # ---- aggregate --------------------------------------------------------------
        def mean_over_seeds(d_key, ek, m):
            return float(np.mean([c[d_key][ek][m] for c in per_seed]))

        def clean_mean(m):
            return float(np.mean([c["clean"][m] for c in per_seed]))

        headline = str(max(args.epsilons))
        damage = {
            m: {
                "clean": clean_mean(m),
                "nes_f1": mean_over_seeds("nes", headline, m),
                "nes_damage": clean_mean(m) - mean_over_seeds("nes", headline, m),
                "random_f1": mean_over_seeds("random", headline, m),
                "random_damage": clean_mean(m) - mean_over_seeds("random", headline, m),
            }
            for m in models
        }

        # ---- controls ----------------------------------------------------------------
        mlp_nes = damage[CONTROL_MODEL]["nes_damage"]
        pos_ok = mlp_nes >= CONTROL_FRACTION * PUBLISHED_MLP_WHITEBOX_DAMAGE
        neg_violations = [
            m for m in models if damage[m]["random_damage"] >= damage[m]["nes_damage"]
        ]
        controls = {
            "positive_mlp_nes_damage": mlp_nes,
            "positive_required": CONTROL_FRACTION * PUBLISHED_MLP_WHITEBOX_DAMAGE,
            "positive_fraction_of_whitebox": float(mlp_nes / PUBLISHED_MLP_WHITEBOX_DAMAGE),
            "positive_pass": bool(pos_ok),
            "negative_random_matches_nes_for": neg_violations,
            "negative_pass": bool(not neg_violations),
            "pass": bool(pos_ok and not neg_violations),
        }
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        # ---- decision rule -------------------------------------------------------------
        gaps = {m: damage[m]["nes_damage"] - PUBLISHED_TRANSFER_DAMAGE[m] for m in TREES}
        upper_bound_confirmed = [m for m, g in gaps.items() if g >= UPPER_BOUND_BAR]
        all_hard = all(abs(g) <= HARD_BAR for g in gaps.values())
        verdict = {
            "controls": controls,
            "headline_epsilon": headline,
            "damage": damage,
            "nes_minus_transfer": {m: float(v) for m, v in gaps.items()},
            "branch": (
                "transfer_figures_are_upper_bounds"
                if upper_bound_confirmed
                else (
                    "trees_hard_for_this_attacker_class" if all_hard else "inconclusive_at_budget"
                )
            ),
            "families_where_nes_wins": upper_bound_confirmed,
            "queries_per_sample": per_seed[0]["budget"][TREES[0]]["queries_per_sample"],
            "rule": (
                f"any tree with NES - transfer >= {UPPER_BOUND_BAR} -> upper bounds confirmed; "
                f"all within {HARD_BAR} -> trees genuinely hard at this budget"
            ),
        }
        run.write_json("results.json", {"per_seed": per_seed, "verdict": verdict})
        run.metrics = {"max_nes_minus_transfer": float(max(gaps.values()))}
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
