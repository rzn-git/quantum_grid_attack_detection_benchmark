"""N-DECAY-GAS: does a second ICS benchmark carry a leak length? The leak decay curve on the
Morris gas-pipeline Modbus dataset (274,628 time-stamped rows, binary attack label).

PRE-REGISTRATION (decision log 2026-09-15T19:16Z, N-DECAY-GAS; Tier B). N-DECAY-2b gave the grid
benchmark's leak a length (tau = 2.5 released rows). This replicates the construction on
the gas-pipeline release: rows in time order (the `time` column is monotone), blocks are
fixed windows of WINDOW consecutive rows, and the arms are those of N-DECAY-2b -- A
middle-60% band, B shuffled within window (negative control), E interleaved (two-sided gap
1), D arm A with XGBoost. Models at library defaults with the class-weight policy, no
tuning: this replicates a shape, not a benchmark. Prior, measured before the design was
fixed: adjacent rows are NOT nearer than random rows here (lag-1 / random distance ratio
1.17 against 0.32 on the grid release), so the expected outcome is the KILL branch.

Data: `data/raw/gas_pipeline/IanArffDataset.arff` (Morris ICS datasets page, gas pipeline,
ARFF with 17 features + 3 labels). Features exclude `time` and the two other labels;
payload fields are 77% missing by design and stay NaN.

CONTROLS. Negative: arm B, gaps 1..20, |Spearman| <= 0.5, permutation-null p >= 0.05, bin
mean within 0.02 of arm B's overall macro-F1 -- VOID if it fails. Positive: arm B overall
exceeds arm A overall by >= 0.02 (a leak exists); if it fails the run is NOT void, the KILL
branch is read directly.
H1: F(1) - asymptote(gaps 15..20) >= 0.05, Spearman <= -0.9 over gaps 1..10, tau < 20.
KILL: F(1) - asymptote < 0.02.
H3 / H4 (only if H1): F_E(1) - F(1) >= 0.02; tau_xgb / tau_lgbm within [0.5, 2].

Run:  python -m experiments.run_leak_decay_gas_pipeline [--smoke]
"""

from __future__ import annotations

import argparse
import io

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from experiments.run_leak_decay_curve import (
    GAP_FIT,
    GAP_H1,
    band_split,
    curve_summary,
    fit_decay,
    gaps_to_train,
    interleave_split,
    per_gap_curve,
    permutation_null_range,
)
from qgridbench.models.classical.zoo import build, fit_model
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger
from qgridbench.utils.seeding import set_all_seeds

log = get_logger(__name__)

ARFF = REPO_ROOT / "data" / "raw" / "gas_pipeline" / "IanArffDataset.arff"
COLS = [
    "address",
    "function",
    "length",
    "setpoint",
    "gain",
    "reset_rate",
    "deadband",
    "cycle_time",
    "rate",
    "system_mode",
    "control_scheme",
    "pump",
    "solenoid",
    "pressure",
    "crc_rate",
    "command_response",
    "time",
    "binary",
    "categorized",
    "specific",
]
FEATURES = COLS[:16]
WINDOW = 200
SEED = 0
FAMILIES = ["lgbm", "xgb"]


def load_gas() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows in time order; returns X (NaN kept), y (binary), block (window id), t (seconds)."""
    lines = ARFF.read_text(encoding="latin-1").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
    body = "\n".join(line for line in lines[start:] if line.strip() and not line.startswith("%"))
    df = pd.read_csv(io.StringIO(body), header=None, names=COLS, na_values="?", quotechar="'")
    t = df["time"].to_numpy(dtype=np.float64)
    if not np.all(np.diff(t) >= 0):
        raise SystemExit("time column is not monotone; sort before splitting")
    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = df["binary"].astype(int).to_numpy()
    block = np.arange(len(y)) // WINDOW
    return X, y, block, t


def fit_predict(family: str, train_mask: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    set_all_seeds(SEED)
    est = build(family, {}, SEED)
    fit_model(family, est, X[train_mask], y[train_mask])
    pred = np.full(len(y), -1, dtype=np.int64)
    pred[~train_mask] = est.predict(X[~train_mask])
    return pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    X, y, block, t = load_gas()
    n = len(y)
    unit = np.zeros(n, dtype=np.int64)  # one sequence: gaps measured over the whole stream
    pos_true = np.arange(n, dtype=np.int64)
    dt = np.diff(t)
    median_dt = float(np.median(dt))
    tr_a, pos_a, side_a = band_split(block, np.random.default_rng(SEED), shuffled=False)
    tr_b, pos_b, _ = band_split(block, np.random.default_rng(SEED), shuffled=True)
    tr_e, te_e = interleave_split(block)
    gap_a = gaps_to_train(tr_a, unit, pos_a)
    gap_b = gaps_to_train(tr_b, unit, pos_b)
    gap_e = gaps_to_train(tr_e, unit, pos_true)
    te_a, te_b = ~tr_a, ~tr_b
    counts = {
        int(g): int(c)
        for g, c in zip(*np.unique(gap_a[te_a], return_counts=True))
        if np.isfinite(g)
    }
    log.info(
        "rows %d | windows %d of %d | attack rate %.3f | median dt %.3fs | "
        "arm A train %d test %d | "
        "gap-1 rows %d | arm E test %d",
        n,
        int(block.max()) + 1,
        WINDOW,
        y.mean(),
        median_dt,
        int(tr_a.sum()),
        int(te_a.sum()),
        counts.get(1, 0),
        int(te_e.sum()),
    )
    if args.smoke:
        print("SMOKE gap counts:", {g: counts[g] for g in sorted(counts)[:22]})
        return

    cfg = {
        "dataset": "morris_gas_pipeline_IanArffDataset",
        "n_rows": n,
        "window": WINDOW,
        "features": FEATURES,
        "families": FAMILIES,
        "model_params": "library defaults + class-weight policy (zoo.build with {})",
        "seed": SEED,
        "median_dt_s": median_dt,
        "preregistration": "decision log 2026-09-15T19:16Z (N-DECAY-GAS)",
    }
    with RunDir("leak_decay_gas_pipeline_binary", config=cfg, seeds=[SEED]) as run:
        out: dict = {"curves": {}, "fits": {}}
        preds = {}
        for fam in FAMILIES:
            preds[("A", fam)] = fit_predict(fam, tr_a, X, y)
            out["curves"][f"A_{fam}"] = per_gap_curve(
                preds[("A", fam)], y, gap_a, te_a, y[tr_a], GAP_FIT
            )
            out["fits"][f"A_{fam}"] = fit_decay(out["curves"][f"A_{fam}"])
            for p in out["curves"][f"A_{fam}"]:
                log.info(
                    "A %-4s gap %2d | n %5d | macro-F1 %.4f | floor %.3f",
                    fam,
                    p["gap"],
                    p["n"],
                    p["macro_f1"],
                    p["floor"],
                )
            log.info("A %-4s fit: %s", fam, out["fits"][f"A_{fam}"])
        preds[("B", "lgbm")] = fit_predict("lgbm", tr_b, X, y)
        out["curves"]["B_lgbm"] = per_gap_curve(
            preds[("B", "lgbm")], y, gap_b, te_b, y[tr_b], GAP_FIT
        )
        preds[("E", "lgbm")] = fit_predict("lgbm", tr_e, X, y)
        out["curves"]["E_lgbm"] = per_gap_curve(
            preds[("E", "lgbm")], y, gap_e, te_e, y[tr_e], GAP_H1
        )
        overall = {
            "A_lgbm": float(f1_score(y[te_a], preds[("A", "lgbm")][te_a], average="macro")),
            "A_xgb": float(f1_score(y[te_a], preds[("A", "xgb")][te_a], average="macro")),
            "B_lgbm": float(f1_score(y[te_b], preds[("B", "lgbm")][te_b], average="macro")),
            "E_lgbm": float(f1_score(y[te_e], preds[("E", "lgbm")][te_e], average="macro")),
        }
        null = permutation_null_range(
            preds[("B", "lgbm")], y, gap_b, te_b, np.random.default_rng(SEED)
        )
        log.info("overall %s | negative null %s", overall, null)

        fa, fb_pts, fe = (
            curve_summary(out["curves"]["A_lgbm"]),
            out["curves"]["B_lgbm"],
            curve_summary(out["curves"]["E_lgbm"]),
        )
        fb = np.array([p["macro_f1"] for p in fb_pts])
        rho_b = float(spearmanr([p["gap"] for p in fb_pts], fb)[0])
        ctrl_neg = bool(
            abs(rho_b) <= 0.5
            and null["p_value"] >= 0.05
            and abs(fb.mean() - overall["B_lgbm"]) <= 0.02
        )
        ctrl_pos = bool(overall["B_lgbm"] - overall["A_lgbm"] >= 0.02)
        fit_l, fit_x = out["fits"]["A_lgbm"], out["fits"]["A_xgb"]
        f1_1, fe_1 = fa.get(1), fe.get(1)
        amp = f1_1 - fit_l["asymptote_empirical"]
        h1_pts = [(g, f) for g, f in fa.items() if g <= GAP_H1]
        rho = float(spearmanr([g for g, _ in h1_pts], [f for _, f in h1_pts])[0])
        h1 = bool(amp >= 0.05 and rho <= -0.9 and fit_l["tau_released_rows"] < 20)
        kill = bool(amp < 0.02)
        tau_ratio = fit_x["tau_released_rows"] / fit_l["tau_released_rows"]
        verdict = {
            "controls_negative_pass": ctrl_neg,
            "control_negative": {
                "spearman_gap_f1": rho_b,
                "bin_mean": float(fb.mean()),
                "bin_range": [float(fb.min()), float(fb.max())],
                **null,
            },
            "control_positive_leak_exists": ctrl_pos,
            "leak_B_minus_A_overall": overall["B_lgbm"] - overall["A_lgbm"],
            "overall": overall,
            "f1_gap": {str(g): fa[g] for g in sorted(fa) if g in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20)},
            "f_e_interleaved_gap1": fe_1,
            "multiplicity_delta": None if fe_1 is None else fe_1 - f1_1,
            "amplitude_f1_minus_asymptote": amp,
            "spearman_gap_f1_le_10": rho,
            **{f"lgbm_{k}": v for k, v in fit_l.items()},
            **{f"xgb_{k}": v for k, v in fit_x.items()},
            "tau_ratio_xgb_over_lgbm": tau_ratio,
            "tau_lgbm_seconds": fit_l["tau_released_rows"] * median_dt,
            "median_dt_s": median_dt,
            "h1_decay_law": h1,
            "kill_no_leak_length": kill,
            "h3_two_sided_multiplicity": bool(h1 and fe_1 is not None and fe_1 - f1_1 >= 0.02),
            "h4_length_belongs_to_data": bool(h1 and 0.5 <= tau_ratio <= 2.0),
        }
        if not ctrl_neg:
            log.error("NEGATIVE CONTROL FAILED (run VOID): %s", verdict["control_negative"])
        run.metrics = verdict
        run.write_json(
            "results.json", {**out, "overall": overall, "negative_null": null, "verdict": verdict}
        )
        log.info(
            "VERDICT | F(1) %.4f asymptote %.4f amp %.4f | F_E(1) %s | "
            "tau %.2f rows = %.1f s (xgb ratio %.2f) | "
            "rho %.3f | leak B-A %.4f | neg=%s pos=%s H1=%s KILL=%s",
            f1_1,
            fit_l["asymptote_empirical"],
            amp,
            None if fe_1 is None else round(fe_1, 4),
            fit_l["tau_released_rows"],
            verdict["tau_lgbm_seconds"],
            tau_ratio,
            rho,
            verdict["leak_B_minus_A_overall"],
            ctrl_neg,
            ctrl_pos,
            h1,
            kill,
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
