"""Logistic regression under its own white-box attack (experiment `logreg_whitebox`, registered
in the decision log on 2026-11-21): the shared protocol reaches it only by transfer from the
unweighted MLP, so its retention there is an upper bound.

The model is the shared protocol's: tuned C, class-weighted, fitted on the kernel subset of the
matched regime. It does not vary with the seed, so one fitted model is attacked from ten PGD
starts at every radius of the grid, on the 500-row evaluation subset every other attack uses,
as the two kernel machines' own attack was run (run_whitebox_qsvm.py, P5a). Beside each
white-box score the run records the transferred PGD on the same rows, crafted on the unweighted
MLP fitted as the shared protocol fits it at seed 0.

Two checks stop the run before any number is written: the analytic input gradient must match
central finite differences on 40 test rows, and the fitted model must reproduce the clean score
the robustness table prints for logistic regression (`--expect-clean`, read from the table by
the caller).

Run:  python -m experiments.run_whitebox_logreg --variant binary --qubits 8 --expect-clean 0.5141
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.exports import Exports
from experiments.run_adversarial import _f1
from experiments.tuned_file import run_name
from qgridbench.attacks.evasion import pgd
from qgridbench.attacks.gradients import GradProvider
from qgridbench.models.classical.zoo import build, fit_model, predict_proba_timed, stratified_cap
from qgridbench.protocol import best_params_path, kernel_subset, prepare_regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds, spawn_rng

log = get_logger(__name__)

GATE_ROWS = 40
GATE_TOL = 1e-5  # relative error of the analytic gradient against finite differences
CLEAN_TOL = 5e-5  # the robustness table prints four decimals
STREAM = "logreg_whitebox"


def gradient_gate(clf, X: np.ndarray, y: np.ndarray) -> dict:
    """Largest relative error of the analytic input gradient against central differences."""
    gp = GradProvider(clf, "logreg")
    ana = gp(X, y)

    def loss(Xb):
        p = np.clip(clf.predict_proba(Xb)[:, 1], 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    h = 1e-5
    num = np.zeros_like(X)
    for j in range(X.shape[1]):
        step = np.zeros(X.shape[1])
        step[j] = h
        num[:, j] = (loss(X + step) - loss(X - step)) / (2 * h)
    rel = float(np.abs(ana - num).max() / np.abs(num).max())
    if not rel < GATE_TOL:
        raise RuntimeError(f"logistic regression gradient gate failed: relative error {rel:.3e}")
    return {"max_rel_err": rel, "n_rows": int(len(X)), "tolerance": GATE_TOL}


def curve(per_seed: dict[str, list[float]]) -> dict:
    return {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
        for k, v in per_seed.items()
    }


def run(reg, cfg_adv, best_params, subset_cap, attack_seeds, expect_clean, ex: Exports) -> dict:
    sub = kernel_subset(reg, subset_cap, seed=0)
    Xtr, ytr = reg.X["train"][sub], reg.y["train"][sub]
    set_all_seeds(0)
    models = {}
    for name in ("logreg", "mlp"):  # the shared protocol's fits at seed 0
        models[name] = build(name, best_params.get(name, {}), 0)
        fit_model(name, models[name], Xtr, ytr)
    clf = models["logreg"]

    def proba(X):
        return predict_proba_timed(clf, X)[0]

    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    Xte, yte = reg.X["test"][ev], reg.y["test"][ev]
    gate_idx = stratified_cap(reg.X["test"], reg.y["test"], GATE_ROWS, seed=0)
    gate = gradient_gate(clf, reg.X["test"][gate_idx], reg.y["test"][gate_idx])
    log.info("gradient gate: relative error %.2e on %d rows", gate["max_rel_err"], gate["n_rows"])

    def scored(seed_key, cond, source, X):
        p = proba(X)
        f = _f1(yte, p)
        if cond != "clean":
            ex.crafted_input(seed_key, cond, source, X)
        ex.predictions(seed_key, cond, "logreg", p, yte, ev)
        ex.metric(seed_key, "logreg", cond, "macro_f1", f)
        return f

    clean = scored(0, "clean", "", Xte)
    if expect_clean is not None and abs(clean - expect_clean) > CLEAN_TOL:
        raise RuntimeError(
            f"clean macro-F1 {clean:.6f} is not the robustness table's {expect_clean}: this is "
            "not the shared protocol's logistic regression"
        )

    scale = reg.feature_scale
    eps_grid = cfg_adv["evasion"]["epsilons"]
    pgd_cfg = cfg_adv["evasion"]["pgd"]
    gp_own, gp_mlp = GradProvider(clf, "logreg"), GradProvider(models["mlp"], "mlp")
    own = {str(e): [] for e in eps_grid}
    transfer = {str(e): [] for e in eps_grid}
    for a_seed in attack_seeds:
        rng = spawn_rng(a_seed, ex.stream(STREAM))
        for e in eps_grid:  # per radius: the own attack, then the transferred one
            for gp, source, cond, into in (
                (gp_own, "logreg", f"pgd_own@{e}", own),
                (gp_mlp, "mlp", f"pgd_transfer_mlp@{e}", transfer),
            ):
                Xa = pgd(
                    gp,
                    Xte,
                    yte,
                    e,
                    scale,
                    n_steps=pgd_cfg["n_steps"],
                    step_frac=pgd_cfg["step_frac"],
                    rng=rng,
                )
                into[str(e)].append(scored(a_seed, cond, source, Xa))

    own_curve, transfer_curve = curve(own), curve(transfer)
    budget = 0.5 if 0.5 in eps_grid else max(eps_grid)
    worst = min((e for e in eps_grid if e <= budget), key=lambda e: own_curve[str(e)]["mean"])
    excess = {str(e): own_curve[str(e)]["mean"] - transfer_curve[str(e)]["mean"] for e in eps_grid}

    def reading(c, e):
        return {"radius": e, "f1": c[str(e)]["mean"], "retention": c[str(e)]["mean"] / clean}

    out = {
        "model": "logreg",
        "C": float(clf.C),
        "n_eval": int(len(ev)),
        "n_train": int(len(sub)),
        "clean_f1": clean,
        "budget": budget,
        "eps_grid": list(eps_grid),
        "attack_seeds": list(attack_seeds),
        "model_is_deterministic": True,
        "whitebox": {"per_seed": own, "curve": own_curve},
        "transfer_mlp": {"per_seed": transfer, "curve": transfer_curve},
        "worst": reading(own_curve, worst),
        "at_budget": reading(own_curve, budget),
        "transfer_at_budget": reading(transfer_curve, budget),
        "numeric_gate": gate,
        # a linear model's loss ascent is the optimal bounded attack on it, so the white-box
        # score may not exceed the transferred one on the same rows
        "control_whitebox_not_weaker": {
            "excess_by_radius": excess,
            "tolerance": 0.01,
            "passed": bool(max(excess.values()) <= 0.01),
        },
    }
    log.info(
        "logistic regression: clean %.4f; own attack %.4f at %.2f sigma (retention %.3f), "
        "%.4f at the %.2f sigma budget; transferred %.4f at the budget (retention %.3f)",
        clean,
        out["worst"]["f1"],
        worst,
        out["worst"]["retention"],
        out["at_budget"]["f1"],
        budget,
        out["transfer_at_budget"]["f1"],
        out["transfer_at_budget"]["retention"],
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--best-params", default=None, help="tuned file (default: the regime's)")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)), help="PGD starts")
    ap.add_argument(
        "--expect-clean",
        type=float,
        default=None,
        help="clean macro-F1 the robustness table prints for logistic regression; the run "
        "stops if the fitted model does not reproduce it",
    )
    ap.add_argument("--run-tag", default=None, help="suffix for the run directory name")
    args = ap.parse_args()

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    subset_cap = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")["qkernel"]["subset_cap"]
    args.best_params = (
        args.best_params
        or best_params_path(args.variant, args.qubits).relative_to(REPO_ROOT).as_posix()
    )
    best_params = json.loads((REPO_ROOT / args.best_params).read_text())

    tag = f"{args.variant}_q{args.qubits}"
    with RunDir(
        run_name(f"whitebox_logreg_{tag}", args.run_tag), config=vars(args), seeds=args.seeds
    ) as run_dir:
        reg = prepare_regime(args.variant, "pca", n_components=args.qubits, seed=0)
        ex = Exports(args.variant, args.best_params)
        ex.extra["stream_note"] = (
            f"spawn_rng(attack seed, '{STREAM}'); per radius the own attack draws its start "
            "first, then the transferred attack"
        )
        try:
            out = run(reg, cfg_adv, best_params, subset_cap, args.seeds, args.expect_clean, ex)
        finally:
            ex.write(run_dir.path)  # a failed run still writes what it scored
        run_dir.write_json("results.json", out)
        print(f"-> {run_dir.path}")


if __name__ == "__main__":
    main()
