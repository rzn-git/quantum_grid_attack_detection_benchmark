"""X2 — which raw features carry the cross-file-transferable signal?

PRE-REGISTRATION (pass-4 workplan item X2; frozen before the run).
W1 measured that +0.028 of the 0.097 whole-file headroom comes from raw features beyond 16
principal components, but not WHICH features. This run names them, or shows the signal is
diffuse. It is the only route in this dataset to a statement a grid engineer can act on.

THE BLOCKS are MECE over all 128 features and were verified to cover them exactly:
  voltage_phasor 48 | current_phasor 48 | frequency 8 | impedance 8 | status_flag 4
  control_panel_log 4 | relay_log 4 | snort_log 4

TWO READINGS per block, both reported:
  block_only[b]   excess-over-floor using ONLY that block
  drop[b]         full-feature excess minus the leave-one-block-out excess (what the block
                  contributes that nothing else supplies)

DECISION RULE (kill criterion, fixed in BACKLOG before the run): if max_b drop[b] < 0.01 the
cross-file-transferable signal is DIFFUSE and the paper says so instead of naming features.
Otherwise the blocks clearing 0.01 are named, in order.

CONTROLS.
  positive (blocking)  the full-feature cell must reproduce W1's XGBoost figure (+0.0693 excess
                       at the matched 2,000-row cap, whole-file partitions 0-2, seeds 0-2)
                       within 0.005 — same protocol, same stream, so it must land
  negative             RANDOM feature subsets at each block size (4, 8, 48), 3 draws each. A
                       4-feature block's excess is read against random-4, never against zero;
                       otherwise "this block carries signal" is unfalsifiable for any block

HYPERPARAMETER NOTE, stated rather than hidden: every arm uses the hyperparameters tuned on the
FULL feature set. They are off-regime for a 4-feature subset, but they are off-regime IDENTICALLY
for every block and for the random controls, so the contrast is fair even though the absolute
levels are not each block's best achievable.

Run:  python -m experiments.run_feature_block_ablation --partitions 3 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
from sklearn.metrics import f1_score

from experiments.run_group_split_control import group_splits, reconstruct_groups, tuned_params
from qgridbench.data.validate import load_variant_frames
from qgridbench.eval.baselines import trivial_baselines
from qgridbench.features.reduce import build_preprocessor, fit_transform_splits
from qgridbench.models.classical.zoo import build, fit_model, stratified_cap
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

VARIANT = "binary"
CAP = 2000
MODELS = ["xgb", "lgbm"]
W1_FULL_XGB_EXCESS = 0.0693  # decision log 2026-08-22T15:14Z
W1_TOL = 0.005
DIFFUSE_BAR = 0.01  # max_b drop[b] below this -> signal is diffuse
RANDOM_DRAWS = 3


def block_of(col: str) -> str:
    if re.match(r"R\d-PA\d+:VH|R\d-PM\d+:V", col):
        return "voltage_phasor"
    if re.match(r"R\d-PA\d+:IH|R\d-PM\d+:I", col):
        return "current_phasor"
    if re.match(r"R\d:F$|R\d:DF$", col):
        return "frequency"
    if re.match(r"R\d-PA:Z", col):
        return "impedance"
    if re.match(r"R\d:S$", col):
        return "status_flag"
    if col.startswith("control_panel_log"):
        return "control_panel_log"
    if col.startswith("relay") and col.endswith("_log"):
        return "relay_log"
    if col.startswith("snort_log"):
        return "snort_log"
    raise ValueError(f"unassigned column {col!r} — the block map must stay MECE")


def full_regime_params(model: str) -> dict:
    """Hyperparameters tuned on the FULL feature set — the regime every arm here runs in."""
    import optuna

    name = f"cls_{model}_{VARIANT}_full"
    db = (REPO_ROOT / "results" / f"optuna_{name}.db").as_posix()
    return optuna.load_study(study_name=name, storage=f"sqlite:///{db}").best_params


def build_blocks() -> dict[str, np.ndarray]:
    label_col = load_yaml(REPO_ROOT / "configs" / "data.yaml")["schema"]["label_column"]
    cols = [c for c in load_variant_frames(VARIANT)[0].columns if c != label_col]
    idx: dict[str, list[int]] = {}
    for i, c in enumerate(cols):
        idx.setdefault(block_of(c), []).append(i)
    blocks = {k: np.array(v, dtype=int) for k, v in sorted(idx.items())}
    assert sum(len(v) for v in blocks.values()) == len(cols), "block map is not a partition"
    return blocks


def score(Xs, y, sp, params_by_model, seed) -> dict[str, float]:
    pipe = build_preprocessor("full", n_components=None, seed=0)
    Z = fit_transform_splits(pipe, Xs, sp)
    ytr_all, yte = y[sp["train"]], y[sp["test"]]
    sub = stratified_cap(Z["train"], ytr_all, CAP, seed=0)
    out = {}
    for m in MODELS:
        set_all_seeds(seed)
        est = build(m, params_by_model[m], seed)
        fit_model(m, est, Z["train"][sub], ytr_all[sub])
        out[m] = float(f1_score(yte, est.predict(Z["test"]), average="macro"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    # BOTH models take their FULL-REGIME tuned parameters. A first version fed XGBoost the
    # q8 parameters from adv_best_params while the docstring promised full-regime ones, and
    # the W1 reproduction control caught it (+0.0069 against a 0.005 tolerance).
    lgbm_params, _ = tuned_params()
    params_by_model = {"xgb": full_regime_params("xgb"), "lgbm": lgbm_params}
    blocks = build_blocks()
    X, y, gid = reconstruct_groups()
    rng_part = np.random.default_rng(0)  # shipped stream
    parts = []
    for p in range(args.partitions):
        sp, held = group_splits(gid, y, rng_part)
        parts.append((p, sp, held[2]))

    sizes = sorted({len(v) for v in blocks.values()})
    cfg = {
        "variant": VARIANT,
        "cap": CAP,
        "models": MODELS,
        "blocks": {k: int(len(v)) for k, v in blocks.items()},
        "random_control_sizes": sizes,
        "random_draws": RANDOM_DRAWS,
        "partitions": args.partitions,
        "seeds": args.seeds,
        "diffuse_bar": DIFFUSE_BAR,
        "w1_reference": W1_FULL_XGB_EXCESS,
        "hyperparameters": "tuned on the FULL feature set, identical for every arm (stated)",
    }

    with RunDir(f"feature_block_ablation_{VARIANT}", config=cfg, seeds=args.seeds) as run:
        rows: list[dict] = []
        all_idx = np.arange(X.shape[1])

        for p, sp, held in parts:
            floor = trivial_baselines(y[sp["train"]], y[sp["test"]])["baseline_stratified_random"][
                "aggregate"
            ]["macro_f1"]["mean"]

            def add(arm: str, cols: np.ndarray, seed: int, extra: dict | None = None):
                f1 = score(X[:, cols], y, sp, params_by_model, seed)
                for m, v in f1.items():
                    rows.append(
                        {
                            "partition": p,
                            "arm": arm,
                            "model": m,
                            "seed": seed,
                            "n_features": int(len(cols)),
                            "clean_f1": v,
                            "floor": floor,
                            "excess": v - floor,
                            **(extra or {}),
                        }
                    )

            for s in args.seeds:
                add("full", all_idx, s)
                for b, cols in blocks.items():
                    add(f"only:{b}", cols, s)
                    add(f"loo:{b}", np.setdiff1d(all_idx, cols), s)
                for k in sizes:
                    for d in range(RANDOM_DRAWS):
                        rng = np.random.default_rng(9000 + 100 * k + d)
                        add(
                            f"random{k}",
                            np.sort(rng.choice(all_idx, k, replace=False)),
                            s,
                            {"draw": d},
                        )
                run.write_json("results_partial.json", {"rows": rows})
                log.info("p%d seed %d done (%d rows)", p, s, len(rows))

        # ---- aggregate -------------------------------------------------------------
        def mean_excess(arm: str, model: str) -> float:
            v = [r["excess"] for r in rows if r["arm"] == arm and r["model"] == model]
            return float(np.mean(v)) if v else float("nan")

        summary = {}
        for m in MODELS:
            full = mean_excess("full", m)
            rnd = {f"random{k}": mean_excess(f"random{k}", m) for k in sizes}
            per_block = {}
            for b, cols in blocks.items():
                k = len(cols)
                only = mean_excess(f"only:{b}", m)
                loo = mean_excess(f"loo:{b}", m)
                per_block[b] = {
                    "n_features": int(k),
                    "block_only_excess": only,
                    "random_same_size_excess": rnd[f"random{k}"],
                    "block_only_minus_random": only - rnd[f"random{k}"],
                    "loo_excess": loo,
                    "drop": full - loo,
                }
            summary[m] = {"full_excess": full, "random_controls": rnd, "per_block": per_block}

        # ---- controls ---------------------------------------------------------------
        pos_delta = summary["xgb"]["full_excess"] - W1_FULL_XGB_EXCESS
        controls = {
            "positive_full_vs_w1_delta": float(pos_delta),
            "positive_pass": bool(abs(pos_delta) <= W1_TOL),
            "negative_random_controls": {m: summary[m]["random_controls"] for m in MODELS},
            "negative_note": (
                "a block's excess is read against a random subset of the SAME size, never "
                "against zero"
            ),
            "pass": bool(abs(pos_delta) <= W1_TOL),
        }
        log.info("CONTROLS\n%s", json.dumps(controls, indent=2))

        # ---- decision rule -----------------------------------------------------------
        drops = {b: max(summary[m]["per_block"][b]["drop"] for m in MODELS) for b in blocks}
        named = sorted([b for b, d in drops.items() if d >= DIFFUSE_BAR], key=lambda b: -drops[b])
        # a block is "self-sufficient" if alone it beats a random subset of its own size
        self_sufficient = sorted(
            [
                b
                for b in blocks
                if max(summary[m]["per_block"][b]["block_only_minus_random"] for m in MODELS)
                >= DIFFUSE_BAR
            ],
            key=lambda b: (
                -max(summary[m]["per_block"][b]["block_only_minus_random"] for m in MODELS)
            ),
        )
        verdict = {
            "controls": controls,
            "summary": summary,
            "max_drop": float(max(drops.values())),
            "drops_by_block": {b: float(v) for b, v in sorted(drops.items(), key=lambda t: -t[1])},
            "blocks_clearing_bar_by_drop": named,
            "blocks_beating_random_alone": self_sufficient,
            "branch": ("signal_localized" if named else "signal_diffuse"),
            "rule": f"max_b drop[b] < {DIFFUSE_BAR} -> diffuse; else name the blocks clearing it",
        }
        run.write_json("results.json", {"rows": rows, "verdict": verdict})
        run.metrics = {"max_drop": float(max(drops.values()))}
        log.info("VERDICT\n%s", json.dumps(verdict, indent=2))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
