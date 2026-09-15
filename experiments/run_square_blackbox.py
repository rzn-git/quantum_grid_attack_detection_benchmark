"""V2-E2: Square attack (field-standard score-based black-box) at the matched query budget.

PRE-REGISTERED 2026-09-01T11:05Z. E14 established with NES that the tree transfer figures are
upper bounds; Square (Andriushchenko et al. 2020, ART implementation) is the attack the field
actually reports for score-based black-box evaluation. Question: is E14's finding
attack-implementation-robust, and what does the standard attack do to every family at the
same ~600-queries/sample budget as HopSkipJump and NES?

Decision rule (fixed before the run): for the tree families, |Square damage - NES damage|
<= 0.05 at 0.5 sigma -> E14 is method-robust and v2 cites Square as confirmation; Square -
NES >= 0.05 anywhere -> Square becomes the binding number for that cell. NES comparators are
read from the E14 artifact (families NES covered: rf, xgb, lgbm, mlp); the other families'
Square damages are reported alongside their published decision-based figures.

Controls copied from E14 verbatim: positive (blocking) - Square on the MLP must reach >= 70%
of the published white-box PGD damage (>= 0.123); negative - a random perturbation at the
matched radius must do less damage than Square for every family.

INSTRUMENT AMENDMENT (decision log 2026-09-01T12:20Z, before any result): ART's SquareAttack
is image-only in source (its tile sampler hard-fails on any dimension <= tile), so this runner
implements the Square SCHEDULE faithfully for tabular input — see square_attack below. It
takes an explicit spawn_rng Generator per (seed, model, eps), so the ART global-RandomState
exception does not apply. Write-ups name it "Square-schedule random search (tabular
analogue)", never bare "Square (ART)".

Run:  python -m experiments.run_square_blackbox --seeds 0 1 2
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
FAMILIES = ["logreg", "rbf_svm", "rf", "xgb", "lgbm", "mlp", "qsvm", "vqc"]
CONTROL_MODEL = "mlp"
E14_ARTIFACT = REPO_ROOT / "results" / "runs" / "20260822T173155Z_score_based_blackbox_binary_q8"
PUBLISHED_MLP_WHITEBOX_DAMAGE = 0.1763  # 0.5 sigma, artifact 20260813T183448Z_adversarial_binary_q8
CONTROL_FRACTION = 0.70
METHOD_ROBUST_BAR = 0.05


P_INIT = 0.8  # Andriushchenko et al. 2020 default
# The paper's staged p-halving schedule, as fractions of the total iteration budget.
P_SCHEDULE_FRACS = [0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


def _p_at(i: int, n_iter: int) -> float:
    p = P_INIT
    for frac in P_SCHEDULE_FRACS:
        if i >= frac * n_iter:
            p /= 2.0
    return p


def square_attack(proba_fn, X, y, eps, scale, rng, max_iter):
    """Square-schedule random search, tabular analogue (AMENDMENT, decision log
    2026-09-01T12:20Z): ART's SquareAttack is image-only in source, so this implements the
    published schedule faithfully for feature vectors — stripe init becomes a random ±eps
    sign per feature; the square patch becomes a random subset of ceil(p*d) features driven
    to an eps-ball corner; greedy per-sample accept on the margin loss; one counted query
    per sample per batched model call. Takes an explicit Generator (so ART's documented
    global-RandomState exception does not apply here). Returns (X_adv, queries/sample)."""
    n, d = X.shape
    lo, hi = X - eps * scale, X + eps * scale
    counter = {"q": 0}

    def margin(Z):
        counter["q"] += 1  # every call queries the model once per sample (batched)
        p = np.clip(proba_fn(Z), 1e-12, 1.0)
        py = p[np.arange(len(Z)), y]
        other = np.max(np.where(np.eye(p.shape[1])[y] == 1, -np.inf, p), axis=1)
        return py - other  # < 0 means misclassified: the attacker minimizes this

    best = margin(X)
    init = np.clip(X + eps * scale * rng.choice([-1.0, 1.0], size=(n, d)), lo, hi)
    m0 = margin(init)
    Xa = np.where((m0 < best)[:, None], init, X)
    best = np.minimum(best, m0)
    for i in range(max_iter):
        k = max(1, int(round(_p_at(i, max_iter) * d)))
        cols = np.argsort(rng.random((n, d)), axis=1)[:, :k]  # per-sample feature subset
        signs = rng.choice([-1.0, 1.0], size=(n, k))
        prop = Xa.copy()
        corner = np.take_along_axis(X, cols, axis=1) + eps * scale[cols] * signs
        np.put_along_axis(prop, cols, corner, axis=1)
        prop = np.clip(prop, lo, hi)
        m = margin(prop)
        better = m < best
        Xa[better], best[better] = prop[better], m[better]
    return Xa, float(counter["q"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--encoding", default="zz")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epsilons", type=float, nargs="*", default=[0.1, 0.3, 0.5])
    ap.add_argument("--max-iter", type=int, default=600, help="~queries/sample; HSJ-matched")
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    e14 = json.loads((E14_ARTIFACT / "results.json").read_text())["verdict"]["damage"]
    nes_damage = {m: e14[m]["nes_damage"] for m in e14}  # rf, xgb, lgbm, mlp

    cfg = {
        "variant": VARIANT,
        "qubits": args.qubits,
        "families": FAMILIES,
        "epsilons": args.epsilons,
        "max_iter": args.max_iter,
        "budget_note": "max_iter ~= queries/sample, matched to HSJ (~610) and NES (600)",
        "nes_comparators": nes_damage,
        "nes_artifact": E14_ARTIFACT.name,
        "seeds": args.seeds,
        "preregistration": "decision log 2026-09-01T11:05Z (V2-E2)",
    }
    with RunDir(f"square_blackbox_{VARIANT}_q{args.qubits}", config=cfg, seeds=args.seeds) as run:
        reg = prepare_regime(VARIANT, "pca", n_components=args.qubits, seed=0)
        scale = reg.feature_scale
        ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
        Xte, yte = reg.X["test"][ev], reg.y["test"][ev]

        per_seed = []
        for s in args.seeds:
            zoo = FittedZoo(
                reg, args.encoding, args.qubits, s, cfg_q["qkernel"]["subset_cap"], best
            )
            cell = {"seed": s, "clean": {}, "square": {}, "random": {}, "queries": {}}
            for m in FAMILIES:
                cell["clean"][m] = _f1(yte, zoo.predict_proba(m, Xte))
            for e in args.epsilons:
                ek = str(e)
                cell["square"][ek], cell["random"][ek] = {}, {}
                for m in FAMILIES:
                    Xa, qps = square_attack(
                        lambda Z, _m=m, _z=zoo: _z.predict_proba(_m, Z),
                        Xte,
                        yte,
                        e,
                        scale,
                        spawn_rng(s, f"square_{m}_{e}"),
                        args.max_iter,
                    )
                    Xr = Xte + np.random.default_rng(500 + s).uniform(-1, 1, Xte.shape) * e * scale
                    cell["square"][ek][m] = _f1(yte, zoo.predict_proba(m, Xa))
                    cell["random"][ek][m] = _f1(yte, zoo.predict_proba(m, Xr))
                    cell["queries"][m] = qps
                    log.info(
                        "s%d eps %-4s %-7s clean %.4f -> square %.4f (damage %.4f) | random "
                        "%.4f | %.0f q/sample",
                        s,
                        ek,
                        m,
                        cell["clean"][m],
                        cell["square"][ek][m],
                        cell["clean"][m] - cell["square"][ek][m],
                        cell["random"][ek][m],
                        qps,
                    )
                    run.write_json("results_partial.json", {"per_seed": per_seed + [cell]})
            per_seed.append(cell)

        # ---- aggregate + controls + decision rule ------------------------------------------
        headline = str(max(args.epsilons))

        def mean_over(dk, ek, m):
            return float(np.mean([c[dk][ek][m] for c in per_seed]))

        damage = {}
        for m in FAMILIES:
            clean = float(np.mean([c["clean"][m] for c in per_seed]))
            damage[m] = {
                "clean": clean,
                "square_f1": mean_over("square", headline, m),
                "square_damage": clean - mean_over("square", headline, m),
                "random_damage": clean - mean_over("random", headline, m),
            }

        mlp_sq = damage[CONTROL_MODEL]["square_damage"]
        pos_ok = mlp_sq >= CONTROL_FRACTION * PUBLISHED_MLP_WHITEBOX_DAMAGE
        neg_violations = [
            m for m in FAMILIES if damage[m]["random_damage"] >= damage[m]["square_damage"]
        ]
        square_vs_nes = {
            m: damage[m]["square_damage"] - nes_damage[m] for m in nes_damage if m in damage
        }
        binding = [m for m, g in square_vs_nes.items() if g >= METHOD_ROBUST_BAR]
        verdict = {
            "controls": {
                "positive_mlp_square_damage": mlp_sq,
                "positive_required": CONTROL_FRACTION * PUBLISHED_MLP_WHITEBOX_DAMAGE,
                "positive_pass": bool(pos_ok),
                "negative_random_matches_square_for": neg_violations,
                "negative_pass": bool(not neg_violations),
                "pass": bool(pos_ok and not neg_violations),
            },
            "headline_epsilon": headline,
            "damage": damage,
            "square_minus_nes": square_vs_nes,
            "branch": (
                "square_becomes_binding_for_" + ",".join(binding)
                if binding
                else "e14_method_robust"
            ),
            "ordering_by_square_damage": sorted(
                FAMILIES, key=lambda m: -damage[m]["square_damage"]
            ),
        }
        run.metrics = {
            "mlp_square_damage": mlp_sq,
            "max_square_minus_nes": float(max(square_vs_nes.values())),
        }
        run.write_json("results.json", {"per_seed": per_seed, "verdict": verdict})
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
