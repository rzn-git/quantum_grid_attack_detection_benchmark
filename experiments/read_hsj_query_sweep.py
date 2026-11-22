"""Read the HopSkipJump query-budget sweep against its registered rules (archive-only).

Registered 2026-11-22 (experiment `hsj_query_sweep`, decision log): does the quantum kernel still
read as the hardest black-box target, at a distance that plain noise contradicts, when the attack
gets more than about 600 queries per point? The restart runs of each budget are named in
experiments/hsj_query_sweep.yaml; this pools each model's restarts per budget and applies the
rules, all read at the highest budget:

  BUDGET_ARTIFACT  the QSVM's mean median distance is under `kill_below_sigma`
  HOLDS            it is at least that, above the best classical comparator at the same budget,
                   and the QSVM's 30th-percentile distance is above `noise_radius_sigma`
  WEAKENED         every other outcome

Checklist item 6 (starting point) holds at a budget when the QSVM's coefficient of variation
over restarts is at least `cv_threshold`.

A restart index is a label, not a repeatable draw (the attack library draws its starting points
unseeded), so budgets are compared as samples of restarts from this sweep's own runs. Distances
are read over the rows the attack found (a row it could not flip comes back on the clean side),
with the found share beside them. Before any distance is read the runs are checked: the attack
settings each run recorded, its measured queries per point against the sweep file's range, and
its clean class-1 probabilities on the attacked rows against `clean_reference_run` (the same
fitted model in every run).

Read after the registered result, and not a registered statistic (decision log 2026-11-22,
14:40Z and 15:34Z): the registered median is taken over all found rows, and those are two groups,
the rows a model labels natural (they flip within hundredths of a sigma) and the rows it labels
attack. `by_direction` gives each group's median distance and its spread over restarts.
`noise_rows` sets the attack against the sweep file's `noise_control` run row by row: a row whose
label random noise of `noise_radius_sigma` changes has an adversarial point at that radius, so a
larger distance from the attack on that row is not the smallest one.

Run: python -m experiments.read_hsj_query_sweep [--sweep-file experiments/hsj_query_sweep.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml

log = get_logger(__name__)

RUNS = REPO_ROOT / "results" / "runs"
DEFAULT_SWEEP_FILE = "experiments/hsj_query_sweep.yaml"
MUST_MATCH = ("variant", "best_params", "start", "eval_subset")
BUDGET_KEYS = ("max_iter", "init_eval", "max_eval")
SEED_FIT = 0  # the fitted model's seed in the restart runner (its exports are keyed by it)
CLEAN_TOL = 1e-6  # class-1 probabilities of one fitted model, between two runs
GROUPS = {"0": "labelled_natural", "1": "labelled_attack"}  # by the model's clean label
ATTACK_LABEL = 1
RADIUS_TOL = 1e-9  # a stored noise point's distance from its row, against the radius


def _spread(values: np.ndarray) -> dict:
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
        "cv": float(std / mean) if mean else float("nan"),
    }


def pool_direction(records: list[dict], cls: str) -> dict:
    """One model's restarts at one budget, over the found rows of one clean label: the spread of
    the per-restart median distance."""
    by_restart = sorted(records, key=lambda r: r["restart"])
    cells = [r["by_clean_prediction"][cls] for r in by_restart]
    rows = {c["n"] for c in cells}
    if len(rows) != 1:
        raise ValueError(f"the restarts disagree on the rows of clean label {cls}: {sorted(rows)}")
    out = {"rows": int(rows.pop()), "found": [int(c["found"]) for c in cells]}
    if out["rows"] == 0:
        return out
    if any(c["median_linf_sigma_found"] is None for c in cells):
        raise ValueError(f"a restart found no row of clean label {cls}")
    med = np.array([c["median_linf_sigma_found"] for c in cells], dtype=float)
    return {**out, "medians_found": [float(m) for m in med], **_spread(med)}


def noise_row_counts(
    dist: np.ndarray,
    found: np.ndarray,
    attack_labelled: np.ndarray,
    noise_flips: np.ndarray,
    radius: float,
    at_least: int,
) -> dict:
    """One restart, over the rows the model labels attack that the attack flipped: how many of
    them the noise flips, and how many of those the attack still places beyond the noise radius.

    `noise_flips` is draws by rows: whether that noise draw changes the row's label. A row counts
    as flipped by the noise under two rules, in at least `at_least` draws and in any draw.
    """
    group = attack_labelled & found
    if not group.any():
        raise ValueError("the attack flipped no row the model labels attack")
    n_flips = noise_flips.sum(axis=0)
    beyond = dist > radius
    out = {"rows": int(group.sum()), "attack_within_share": float((~beyond)[group].mean())}
    for name, flipped in (("", n_flips >= at_least), ("_any_draw", n_flips >= 1)):
        out[f"noise_flips{name}"] = int((group & flipped).sum())
        out[f"noise_flips{name}_attack_beyond"] = int((group & flipped & beyond).sum())
    return out


def pool(records: list[dict]) -> dict:
    """One model's restarts at one budget: the spread of the per-restart median distance over
    the rows the attack found."""
    restarts = sorted(r["restart"] for r in records)
    if len(set(restarts)) != len(restarts):
        raise ValueError(f"a restart index appears twice: {restarts}")
    by_restart = sorted(records, key=lambda r: r["restart"])
    med = np.array([r["median_linf_sigma_found"] for r in by_restart])
    mean = float(med.mean())
    std = float(med.std(ddof=1)) if len(med) > 1 else 0.0
    out = {
        "restarts": restarts,
        "medians_found": [float(m) for m in med],
        "mean": mean,
        "std": std,
        "standard_error": float(std / np.sqrt(len(med))),
        "min": float(med.min()),
        "max": float(med.max()),
        "cv": float(std / mean) if mean else float("nan"),
        "queries_per_sample": {
            "min": float(min(r["queries_per_sample"] for r in by_restart)),
            "max": float(max(r["queries_per_sample"] for r in by_restart)),
        },
        "found_share_min": float(min(r["found_share"] for r in by_restart)),
        "stalled_steps": int(sum(r["stalled_steps"] for r in by_restart)),
    }
    if all("p30_found" in r for r in by_restart):
        out["p30_mean"] = float(np.mean([r["p30_found"] for r in by_restart]))
    return out


def branch(qsvm: dict, best_classical_mean: float, kill_below: float, noise_radius: float) -> str:
    """The registered reading at the highest budget."""
    if qsvm["mean"] < kill_below:
        return "BUDGET_ARTIFACT"
    if qsvm["mean"] > best_classical_mean and qsvm["p30_mean"] > noise_radius:
        return "HOLDS"
    return "WEAKENED"


def check_queries(name: str, family: str, pooled: dict, allowed: list[float]) -> None:
    q = pooled["queries_per_sample"]
    if not (allowed[0] <= q["min"] and q["max"] <= allowed[1]):
        raise ValueError(
            f"{name}/{family}: {q['min']:.0f} to {q['max']:.0f} queries per point, "
            f"the sweep file allows {allowed[0]:.0f} to {allowed[1]:.0f}"
        )


def attacked_rows(run: str, family: str, restart: int, X_test, scale) -> dict:
    """One restart's rows from the run's stored predictions and crafted rows: the test positions,
    the model's clean label, whether the attack flipped the row, and the distance it reports."""
    folder = RUNS / run
    pred = pd.read_parquet(folder / "predictions.parquet")
    pred = pred[pred["model"] == family]
    cond = f"hsj_restart{restart}"
    clean = pred[pred["condition"] == "clean"].sort_values("test_pos")
    adv = pred[pred["condition"] == cond].sort_values("test_pos")
    if len(clean) == 0 or len(adv) != len(clean):
        raise ValueError(f"{run}: no predictions for {family} under {cond}")
    pos = clean["test_pos"].to_numpy()
    if not np.array_equal(pos, adv["test_pos"].to_numpy()):
        raise ValueError(f"{run}: clean and attacked rows of {family} differ")
    label = clean["label"].to_numpy()
    with np.load(folder / "crafted_inputs.npz") as crafted:
        Xa = crafted[f"{SEED_FIT}|{cond}|{family}"]
    return {
        "test_pos": pos,
        "label": label,
        "found": label != adv["label"].to_numpy(),
        "dist": (np.abs(Xa - X_test[pos]) / scale).max(axis=1),
    }


def found_percentile(run: str, family: str, restart: int, q: float, X_test, scale) -> float:
    """The q-th percentile of the per-point distance over the rows the attack flipped."""
    rows = attacked_rows(run, family, restart, X_test, scale)
    if not rows["found"].any():
        raise ValueError(f"{run}: the attack found no row for {family} under restart {restart}")
    return float(np.percentile(rows["dist"][rows["found"]], q))


def noise_control(run: str, family: str, radius: float, draws: int, X_test, scale) -> dict:
    """The random-noise run on the attacked rows: the model's label without noise, per draw
    whether noise of `radius` changes it, and how many label-changing noise points lie outside
    the box the attack searches (the library clips to the smallest and largest value of the
    attacked batch, in single precision)."""
    folder = RUNS / run
    pred = pd.read_parquet(folder / "predictions.parquet")
    pred = pred[pred["model"] == family]
    base = pred[pred["condition"] == "random_base"].sort_values("test_pos")
    if len(base) == 0:
        raise ValueError(f"{run}: no noise-free predictions for {family}")
    pos, label = base["test_pos"].to_numpy(), base["label"].to_numpy()
    batch = X_test[pos].astype(np.float32)
    low, high = batch.min(), batch.max()
    flips, outside = [], 0
    with np.load(folder / "crafted_inputs.npz") as crafted:
        for d in range(draws):
            cond = f"random@{radius}|draw{d}"
            drawn = pred[pred["condition"] == cond].sort_values("test_pos")
            if not np.array_equal(drawn["test_pos"].to_numpy(), pos):
                raise ValueError(f"{run}: {cond} does not cover the noise-free rows of {family}")
            points = crafted[f"{SEED_FIT}|{cond}|random"]
            at = (np.abs(points - X_test[pos]) / scale).max(axis=1)
            if not np.allclose(at, radius, atol=RADIUS_TOL):
                raise ValueError(
                    f"{run}: the stored points of {cond} are {at.min():.6f} to {at.max():.6f} "
                    f"sigma from their rows, not {radius}"
                )
            flipped = drawn["label"].to_numpy() != label
            inside = ((points >= low) & (points <= high)).all(axis=1)
            outside += int((flipped & ~inside).sum())
            flips.append(flipped)
    return {
        "test_pos": pos,
        "label": label,
        "proba_1": base["proba_1"].to_numpy(),
        "flips": np.array(flips),
        "flipping_points_outside_search_box": outside,
    }


def noise_rows(
    records: list[dict], control: dict, radius: float, at_least: int, X_test, scale
) -> dict:
    """One model at one budget: `noise_row_counts` per restart, and the mean of each count.

    `records` are (run, family, restart). The control run must have scored the fitted model the
    attack ran against: the same rows with the same clean labels.
    """
    per_restart = []
    for run, family, restart in sorted(records, key=lambda r: r[2]):
        rows = attacked_rows(run, family, restart, X_test, scale)
        if not np.array_equal(rows["test_pos"], control["test_pos"]):
            raise ValueError(f"{run}: the noise control ran on other rows")
        if not np.array_equal(rows["label"], control["label"]):
            raise ValueError(f"{run}: {family}'s clean labels differ from the noise control's")
        counts = noise_row_counts(
            rows["dist"],
            rows["found"],
            rows["label"] == ATTACK_LABEL,
            control["flips"],
            radius,
            at_least,
        )
        per_restart.append({"restart": restart, "run": run, **counts})
    keys = [k for k in per_restart[0] if k not in ("restart", "run")]
    return {
        "per_restart": per_restart,
        "mean": {k: float(np.mean([r[k] for r in per_restart])) for k in keys},
    }


def clean_difference(
    run: str, reference: str, family: str, reference_condition: str = "clean"
) -> float:
    """Largest difference in clean class-1 probability on the attacked rows between two runs.
    `reference_condition` names the reference run's unperturbed predictions."""

    def clean(r, condition):
        p = pd.read_parquet(RUNS / r / "predictions.parquet")
        p = p[(p["model"] == family) & (p["condition"] == condition)].sort_values("test_pos")
        if len(p) == 0:
            raise ValueError(f"{r}: no {condition} predictions for {family}")
        return p["test_pos"].to_numpy(), p["proba_1"].to_numpy()

    pos_a, a = clean(run, "clean")
    pos_b, b = clean(reference, reference_condition)
    if not np.array_equal(pos_a, pos_b):
        raise ValueError(f"{run} and {reference} attacked different rows")
    return float(np.abs(a - b).max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep-file", default=DEFAULT_SWEEP_FILE)
    args = ap.parse_args()
    path = REPO_ROOT / args.sweep_file
    spec = load_yaml(path)
    order = list(spec["budgets"])  # lowest to highest, as written
    comparators = spec["comparators"]

    from qgridbench.protocol import prepare_regime

    reg = prepare_regime(spec["variant"], "pca", n_components=spec["qubits"], seed=0)
    X_test, scale = reg.X["test"], reg.feature_scale

    noise = spec["noise_control"]
    radius = spec["noise_radius_sigma"]
    families = ["qsvm", *comparators]
    controls = {
        f: noise_control(noise["run"], f, radius, noise["draws"], X_test, scale) for f in families
    }

    pooled: dict = {}
    by_direction: dict = {}
    by_row: dict = {}
    shared: dict = {}
    clean_diff: dict = {}
    noise_diff: dict = {}
    for name in order:
        budget = spec["budgets"][name]
        pooled[name] = {"settings": {k: budget[k] for k in BUDGET_KEYS}}
        by_direction[name], by_row[name] = {}, {}
        for family, runs in budget["families"].items():
            records = []
            restarts = []
            for run in runs:
                cfg = load_yaml(RUNS / run / "params.yaml")["config"]
                if cfg["variant"] != spec["variant"]:
                    raise ValueError(f"{run}: variant {cfg['variant']}")
                for key in BUDGET_KEYS:
                    if cfg[key] != budget[key]:
                        raise ValueError(
                            f"{run}: {key} {cfg[key]}, the sweep file says {budget[key]}"
                        )
                for key in MUST_MATCH:
                    if shared.setdefault(key, cfg[key]) != cfg[key]:
                        raise ValueError(f"{run}: {key} {cfg[key]} differs from {shared[key]}")
                d = clean_difference(run, spec["clean_reference_run"], family)
                clean_diff[f"{run}|{family}"] = d
                noise_diff[f"{run}|{family}"] = clean_difference(
                    run, noise["run"], family, "random_base"
                )
                if d > CLEAN_TOL:
                    raise ValueError(
                        f"{run}: {family}'s clean probabilities differ from "
                        f"{spec['clean_reference_run']} by {d:.2e}: another fitted model"
                    )
                results = json.loads((RUNS / run / "results.json").read_text())
                for rec in results["per_family"][family]:
                    rec = dict(rec)
                    rec["p30_found"] = found_percentile(
                        run, family, rec["restart"], 30, X_test, scale
                    )
                    records.append(rec)
                    restarts.append((run, family, rec["restart"]))
            if len(records) != spec["restarts_per_cell"]:
                raise ValueError(
                    f"{name}/{family}: {len(records)} restarts, "
                    f"the sweep file says {spec['restarts_per_cell']}"
                )
            pooled[name][family] = pool(records)
            check_queries(name, family, pooled[name][family], budget["queries_per_point"])
            by_direction[name][family] = {
                group: pool_direction(records, cls) for cls, group in GROUPS.items()
            }
            by_row[name][family] = noise_rows(
                restarts, controls[family], radius, noise["flipped_in_at_least"], X_test, scale
            )

    low, high = order[0], order[-1]
    qsvm_high = pooled[high]["qsvm"]
    best_classical = max(pooled[high][f]["mean"] for f in comparators)
    reading = branch(
        qsvm_high, best_classical, spec["kill_below_sigma"], spec["noise_radius_sigma"]
    )
    item6 = {
        name: {
            "cv": pooled[name]["qsvm"]["cv"],
            "holds": bool(pooled[name]["qsvm"]["cv"] >= spec["cv_threshold"]),
        }
        for name in order
    }
    # control C4: a classical model's mean distance does not rise by more than its restart
    # spread from the lowest to the highest budget
    monotone = {
        f: bool(pooled[high][f]["mean"] <= pooled[low][f]["mean"] + pooled[low][f]["std"])
        for f in comparators
    }
    after_the_result = {
        "note": (
            "read after the registered result; not a registered statistic "
            "(decision log 2026-11-22, 14:40Z and 15:34Z)"
        ),
        "by_direction": by_direction,
        "noise_rows": by_row,
        "noise_control": {
            **noise,
            "radius_sigma": radius,
            "rows_labelled_attack": {
                f: int((controls[f]["label"] == ATTACK_LABEL).sum()) for f in families
            },
            "rows_labelled_attack_noise_flips": {
                f: int(
                    (
                        (controls[f]["label"] == ATTACK_LABEL)
                        & (controls[f]["flips"].sum(axis=0) >= noise["flipped_in_at_least"])
                    ).sum()
                )
                for f in families
            },
            "rows_labelled_attack_noise_flips_any_draw": {
                f: int(
                    (
                        (controls[f]["label"] == ATTACK_LABEL)
                        & (controls[f]["flips"].sum(axis=0) >= 1)
                    ).sum()
                )
                for f in families
            },
            "flip_rate": {f: float(controls[f]["flips"].mean()) for f in families},
            "flipping_points_outside_search_box": {
                f: controls[f]["flipping_points_outside_search_box"] for f in families
            },
            "clean_probability_max_difference": noise_diff,
        },
    }
    out = {
        "branch": reading,
        "read_at": high,
        "qsvm_mean_high": qsvm_high["mean"],
        "best_classical_mean_high": best_classical,
        "qsvm_p30_mean_high": qsvm_high["p30_mean"],
        "item6_starting_point": item6,
        "controls": {
            "found_share_min": {
                n: {f: pooled[n][f]["found_share_min"] for f in families} for n in order
            },
            "stalled_steps": {
                n: {f: pooled[n][f]["stalled_steps"] for f in families} for n in order
            },
            "classical_not_rising": monotone,
            "clean_probability_max_difference": clean_diff,
            "clean_reference_run": spec["clean_reference_run"],
        },
        "per_budget": pooled,
        "rules": {k: spec[k] for k in ("kill_below_sigma", "noise_radius_sigma", "cv_threshold")},
        "shared_settings": shared,
        "after_the_result": after_the_result,
    }
    with RunDir(
        f"hsj_query_sweep_verdict_{spec['variant']}_q{spec['qubits']}",
        config={
            "sweep_file": args.sweep_file,
            "sweep_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sources": {n: spec["budgets"][n]["families"] for n in order},
        },
        seeds=[],
    ) as run:
        run.metrics = {
            "qsvm_mean_high": qsvm_high["mean"],
            "best_classical_mean_high": best_classical,
        }
        run.write_json("results.json", out)
        for name in order:
            q = pooled[name]["qsvm"]
            log.info(
                "%s: QSVM %.3f +- %.3f (cv %.2f, p30 %.3f, %s queries), classical %s",
                name,
                q["mean"],
                q["std"],
                q["cv"],
                q["p30_mean"],
                q["queries_per_sample"],
                {f: round(pooled[name][f]["mean"], 3) for f in comparators},
            )
        log.info("branch: %s (read at %s)", reading, high)
        for name in order:
            for f in families:
                d = by_direction[name][f]["labelled_attack"]
                n = by_row[name][f]["mean"]
                log.info(
                    "%s %s, %d rows labelled attack: %.3f +- %.3f (cv %.2f); the noise flips "
                    "%.1f, the attack puts %.1f of them beyond %.1f sigma",
                    name,
                    f,
                    d["rows"],
                    d["mean"],
                    d["std"],
                    d["cv"],
                    n["noise_flips"],
                    n["noise_flips_attack_beyond"],
                    radius,
                )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
