"""Read stored white-box curves at the worst grid radius inside the budget (archive-only).

An attacker bounded by a radius may use any smaller perturbation, so the reading at the budget
is not the attacker's best when a stored curve is not monotone in the radius (Carlini et al.,
"On Evaluating Adversarial Robustness": attack success must not fall as the budget grows).
Operator ruling (h), decision log 2026-11-20T10:17Z: a white-box result is read at the grid
radius, at or below the budget, that leaves the lowest mean attacked macro-F1 over the attack
starts; a tie goes to the smaller radius. The reading is the mean and standard deviation over
the starts at that radius. It bounds the attacker from above: one who picks the radius per
input can only do better.

This reads the per-radius, per-start scores that the source runs named in
experiments/worst_radius_sources.yaml stored, attacks nothing, and writes one run with both
readings (at the budget, at the worst radius) per source and model, the paired test of the
kernel pair at each model's own worst radius, and the replication statistics re-read against
the bars registered for each source. The reading at the budget must reproduce the stored
figures of the source and verdict runs, or the run fails. Three disclosures ride beside the
readings: for the whole-file control, the score left by an attacker who picks the radius per
input (from the stored predictions); for the end-state sweeps, the number of radii at which each
setting reads inverted; for a replication source, the availability statistic with the
transferred attack also read at its own worst radius.

Run: python -m experiments.read_worst_radius
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from qgridbench.eval.stats import paired_wilcoxon
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml

log = get_logger(__name__)

RUNS = REPO_ROOT / "results" / "runs"
DEFAULT_SOURCES_FILE = "experiments/worst_radius_sources.yaml"
ANCHOR_TOL = 1e-9
# the accuracy half of the end-state rule (qgridbench.eval.tables._regime; rule B of
# experiments/run_taxonomy_scope.py): macro-F1 against the constant-predictor floor
DISABLED_WITHIN, INVERTED_BELOW = 0.05, -0.2


def _spread(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def end_state(f1: float, floor: float) -> str:
    """The end-state label that macro-F1 alone gives against the constant-predictor floor."""
    d = f1 - floor
    if abs(d) <= DISABLED_WITHIN:
        return "disabled"
    if d <= INVERTED_BELOW:
        return "inverted"
    return "partial"


def worst_radius(per_start: dict[str, list[float]], budget: float) -> str:
    """The grid radius at or below `budget` with the lowest mean score over the starts.

    A tie goes to the smaller radius. `per_start` maps a radius (as the run stored it, a
    string) to one score per attack start.
    """
    inside = sorted((r for r in per_start if float(r) <= budget + 1e-12), key=float)
    if not inside:
        raise ValueError(f"no stored radius at or below the budget {budget}: {sorted(per_start)}")
    means = [float(np.mean(per_start[r])) for r in inside]
    return inside[int(np.argmin(means))]  # argmin returns the first minimum: the smaller radius


def read_model(
    clean: float, per_start: dict[str, list[float]], budget: float, floor: float | None
) -> dict:
    """Both readings of one model's stored white-box curve: at the budget and at the worst radius.

    `clean` is the clean macro-F1 the retention is taken against (a mean over fits where the
    starts are refits). Retention is the mean attacked score over the clean score, as the source
    runs compute it.
    """
    at = next((r for r in per_start if abs(float(r) - budget) < 1e-12), None)
    if at is None:
        raise ValueError(f"the budget {budget} is not a stored radius: {sorted(per_start)}")
    n = {len(v) for v in per_start.values()}
    if len(n) != 1:
        raise ValueError(f"radii carry different numbers of attack starts: {sorted(n)}")
    if not clean > 0:
        raise ValueError(f"clean macro-F1 must be positive, got {clean}")
    radii = sorted(per_start, key=float)
    worst = worst_radius(per_start, budget)

    def reading(r: str) -> dict:
        v = np.asarray(per_start[r], dtype=float)
        out = {"radius": float(r), **_spread(v), "retention": float(v.mean() / clean)}
        if floor is not None:
            out["f1_minus_floor"] = float(v.mean() - floor)
            out["end_state"] = end_state(float(v.mean()), floor)
        return out

    budget_reading, worst_reading = reading(at), reading(worst)
    below = np.asarray(per_start[worst]) <= np.asarray(per_start[at])
    worst_reading["starts_at_or_below_the_budget_reading"] = int(below.sum())
    inside = [r for r in radii if float(r) <= budget + 1e-12]
    means = [float(np.mean(per_start[r])) for r in inside]
    return {
        "clean": float(clean),
        "n_starts": n.pop(),
        "radii": [float(r) for r in radii],
        "mean_curve": {r: float(np.mean(per_start[r])) for r in radii},
        "per_start": {r: [float(x) for x in per_start[r]] for r in radii},
        "at_budget": budget_reading,
        "worst": worst_reading,
        "d_retention": budget_reading["retention"] - worst_reading["retention"],
        "monotone_inside_budget": bool(all(a >= b for a, b in zip(means, means[1:]))),
    }


def reading_radius_verdict(d_rbf: float, yes_at: float, partial_at: float) -> str:
    """The new checklist choice 7 on a replication source, by the bars registered for it."""
    if d_rbf >= yes_at:
        return "yes"
    if d_rbf >= partial_at:
        return "partial"
    return "no"


def registered_verdicts(rule: dict, a: float, q_minus_r: float, p_value: float) -> dict:
    """The verdicts each source registered for the availability and kernel-contrast statistics.

    Gas (H-C1a, decision log 2026-11-16T11:43Z): B = RBF-SVM minus QSVM white-box retention;
    KILL at B <= 0, REPLICATES at A >= a_min and B >= b_min with p < alpha, else INTERMEDIATE;
    checklist item 3 holds at A >= a_min. Grids (R4-22 H2, 2026-11-16T12:17Z): availability
    REPLICATES at A >= a_min, fragility at F = QSVM minus RBF-SVM <= f_max.
    """
    if rule["rule"] == "gas":
        b = -q_minus_r
        if b <= 0:
            contrast = "KILL"
        elif a >= rule["a_min"] and b >= rule["b_min"] and p_value < rule["alpha"]:
            contrast = "REPLICATES"
        else:
            contrast = "INTERMEDIATE"
        return {"availability": "REPLICATES" if a >= rule["a_min"] else "NOT", "contrast": contrast}
    if rule["rule"] == "grid":
        return {
            "availability": "REPLICATES" if a >= rule["a_min"] else "NOT",
            "contrast": "REPLICATES" if q_minus_r <= rule["f_max"] else "NOT",
        }
    raise ValueError(f"unknown registered rule {rule['rule']!r}")


def constant_predictor_floor(labels_file: Path, n_eval: int, clean_qsvm: float) -> dict:
    """Macro-F1 of predicting the majority class on the evaluated rows, from their stored labels.

    The file must hold the kernel run's rows: `n_eval` distinct rows, and the QSVM's clean
    macro-F1 recomputed from it must equal the kernel run's.
    """
    p = pd.read_parquet(labels_file)
    clean = p[(p["condition"] == "clean") & (p["model"] == "qsvm")]
    clean = clean[clean["seed"] == clean["seed"].min()].drop_duplicates("row_id")
    if len(clean) != n_eval:
        raise ValueError(
            f"{labels_file}: {len(clean)} clean QSVM rows, the kernel run has {n_eval}"
        )
    y = clean["y_true"].to_numpy()
    f1 = float(f1_score(y, clean["label"].to_numpy(), average="macro"))
    if abs(f1 - clean_qsvm) > ANCHOR_TOL:
        raise ValueError(
            f"{labels_file}: the QSVM's clean macro-F1 is {f1!r}, the kernel run's is "
            f"{clean_qsvm!r}; these are not the kernel run's rows or model"
        )
    majority = int(round(float(y.mean())))
    floor = float(f1_score(y, np.full(len(y), majority), average="macro"))
    return {"floor": floor, "majority_share": float(max(y.mean(), 1 - y.mean())), "n_rows": len(y)}


def _json(path: Path, sources: dict) -> dict:
    sources[path.relative_to(REPO_ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def _anchor(anchors: list, what: str, got: float, stored: float, tol: float = ANCHOR_TOL) -> None:
    ok = abs(got - stored) <= tol
    anchors.append({"what": what, "recomputed": got, "stored": stored, "tol": tol, "ok": bool(ok)})
    if not ok:
        raise ValueError(f"anchor failed: {what}: recomputed {got!r}, stored {stored!r}")


def read_sweep(block: dict, axis: str, budget: float, min_radii: int) -> dict:
    """An end-state sweep (one model per kernel width or bandwidth) read at the full budget and
    at the worst radius inside it, as macro-F1 minus the constant-predictor floor.

    `block[axis]` maps a setting to its `curve`: radius -> {"f1_minus_floor": mean over the
    attack starts}. Radius 0 is the clean score and is not an attack reading.

    The worst-radius label is a single-radius label. The paper's end states hold across a
    run of radii, so each setting also carries its label at every radius and the number
    of radii that read inverted; `n_inverted_over_min_radii` counts the settings inverted
    at `min_radii` radii or more.
    """
    floor = float(block["constant_predictor_floor"])
    settings = {}
    for setting, v in block[axis].items():
        curve = {
            r: float(c["f1_minus_floor"])
            for r, c in v["curve"].items()
            if 0 < float(r) <= budget + 1e-12
        }
        at = next((r for r in curve if abs(float(r) - budget) < 1e-12), None)
        if at is None:
            raise ValueError(f"{axis} {setting}: the budget {budget} is not a stored radius")
        radii = sorted(curve, key=float)
        worst = radii[int(np.argmin([curve[r] for r in radii]))]
        settings[setting] = {
            which: {
                "radius": float(r),
                "f1_minus_floor": curve[r],
                "end_state": end_state(curve[r] + floor, floor),
            }
            for which, r in (("at_budget", at), ("worst", worst))
        }
        by_radius = {r: end_state(curve[r] + floor, floor) for r in radii}
        settings[setting]["end_state_by_radius"] = by_radius
        settings[setting]["f1_minus_floor_by_radius"] = {r: curve[r] for r in radii}
        settings[setting]["n_radii_inverted"] = sum(v == "inverted" for v in by_radius.values())
    worst_values = [s["worst"]["f1_minus_floor"] for s in settings.values()]
    return {
        "constant_predictor_floor": floor,
        "budget": budget,
        "settings": settings,
        "n_settings": len(settings),
        "worst_lowest": min(worst_values),
        "worst_highest": max(worst_values),
        "n_inverted_at_budget": sum(
            s["at_budget"]["end_state"] == "inverted" for s in settings.values()
        ),
        "n_inverted_at_worst": sum(
            s["worst"]["end_state"] == "inverted" for s in settings.values()
        ),
        "persistence_min_radii": min_radii,
        "n_inverted_over_min_radii": sum(
            s["n_radii_inverted"] >= min_radii for s in settings.values()
        ),
        "settings_inverted_at_fewer_radii": sorted(
            (k for k, s in settings.items() if s["n_radii_inverted"] < min_radii), key=float
        ),
    }


def whole_file_branch(passes: list[bool]) -> str:
    """The registered branch of the whole-file kernel contrast (decision log 2026-11-20T10:33Z)."""
    if all(passes):
        return "REPLICATES UNDER R"
    return "MIXED" if any(passes) else "DOES NOT"


def per_input_union(predictions: pd.DataFrame, model: str, budget: float) -> dict:
    """Macro-F1 when the attacker picks the radius per input, over the stored radii.

    A row counts as evaded when the attacked prediction is wrong at any stored radius at
    or below the budget, for the same attack seed. The worst single radius leaves a
    higher score than this, so the worst-radius reading bounds the attacker from above.
    `predictions` holds one row per (model, condition, attack seed, row id) with the
    true label and the predicted label; conditions are `pgd@<radius>`.
    """
    d = predictions[
        (predictions["model"] == model) & predictions["condition"].str.startswith("pgd@")
    ]
    d = d[d["condition"].str.slice(4).astype(float) <= budget + 1e-12]
    if d.empty:
        raise ValueError(f"no attacked predictions for {model} at or below {budget}")
    scores = []
    for _, g in d.groupby("attack_seed"):
        wide = g.pivot(index="row_id", columns="condition", values="label")
        if wide.isna().any().any():
            raise ValueError(f"{model}: a row is missing at one of the stored radii")
        y = g.drop_duplicates("row_id").set_index("row_id")["y_true"].loc[wide.index]
        y = y.to_numpy()
        evaded = (wide.to_numpy() != y[:, None]).any(axis=1)
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError(f"{model}: the union reading is defined for two classes")
        other = np.where(y == labels[0], labels[1], labels[0])
        scores.append(float(f1_score(y, np.where(evaded, other, y), average="macro")))
    return {
        **_spread(np.asarray(scores)),
        "per_start": scores,
        "radii": sorted(float(c[4:]) for c in d["condition"].unique()),
    }


def read_whole_file(spec: dict, budget: float, sources: dict, anchors: list) -> dict:
    """The kernel contrast with whole files held out, each model at its worst radius.

    Per partition: macro-F1 minus the partition's stratified-random floor, the gap RBF-SVM minus
    QSVM, and the attack seeds in which the QSVM is the lower. A partition passes when the gap
    is at least `gap_min` in the mean and in at least `seeds_min` of the attack seeds, as
    registered (decision log 2026-11-20T10:33Z). The reading at the budget must reproduce the
    first run of the control (`anchor_run`) within `anchor_tol`. Each model also carries the
    reading of an attacker who picks the radius per input (`per_input_union`).
    """
    first = _json(RUNS / spec["anchor_run"] / "results.json", sources)["partitions"]
    first = {p["partition"]: p for p in first}
    partitions = {}
    for run in spec["runs"]:
        pred_file = RUNS / run / "predictions.parquet"
        predictions = pd.read_parquet(pred_file)
        sources[pred_file.relative_to(REPO_ROOT).as_posix()] = hashlib.sha256(
            pred_file.read_bytes()
        ).hexdigest()
        for part in _json(RUNS / run / "results.json", sources)["partitions"]:
            p = part["partition"]
            of_part = predictions[predictions["partition"] == p]
            floor = float(part["floor"])
            models = {
                "qsvm": read_model(part["clean_qsvm"], part["curve"]["qsvm"], budget, None),
                "rbf_svm": read_model(part["clean_rbf"], part["curve"]["rbf_svm"], budget, None),
            }
            for m, key in (("qsvm", "adv_qsvm_mean"), ("rbf_svm", "adv_rbf_mean")):
                _anchor(
                    anchors,
                    f"whole-file p{p} {m} attacked macro-F1 at the budget",
                    models[m]["at_budget"]["mean"],
                    first[p][key],
                    tol=spec["anchor_tol"],
                )
                for which in ("at_budget", "worst"):
                    models[m][which]["f1_minus_stratified_floor"] = models[m][which]["mean"] - floor
                models[m]["per_input_union"] = per_input_union(of_part, m, budget)
                if models[m]["per_input_union"]["mean"] > models[m]["worst"]["mean"] + 1e-12:
                    raise ValueError(
                        f"whole-file p{p} {m}: the per-input reading is above the worst radius; "
                        "the predictions are not this run's"
                    )

            def at_worst(m: str, models=models) -> np.ndarray:
                k = next(
                    r for r in models[m]["per_start"] if float(r) == models[m]["worst"]["radius"]
                )
                return np.asarray(models[m]["per_start"][k])

            gap = models["rbf_svm"]["worst"]["mean"] - models["qsvm"]["worst"]["mean"]
            per_seed_gap = at_worst("rbf_svm") - at_worst("qsvm")
            below = int((per_seed_gap > 0).sum())
            clear = int((per_seed_gap >= spec["gap_min"]).sum())
            partitions[str(p)] = {
                "run": run,
                "held_out_files": part["held_out_files"],
                "stratified_random_floor": floor,
                "models": models,
                "gap_rbf_minus_qsvm_at_budget": models["rbf_svm"]["at_budget"]["mean"]
                - models["qsvm"]["at_budget"]["mean"],
                "gap_rbf_minus_qsvm_worst": gap,
                "seeds_qsvm_below_rbf_worst": below,
                "seeds_gap_at_least_gap_min": clear,
                "smallest_per_seed_gap": float(per_seed_gap.min()),
                "passes": bool(gap >= spec["gap_min"] and clear >= spec["seeds_min"]),
            }
    return {
        "partitions": partitions,
        "gap_min": spec["gap_min"],
        "seeds_min": spec["seeds_min"],
        "branch": whole_file_branch([v["passes"] for v in partitions.values()]),
    }


def read_source(name: str, spec: dict, cfg: dict, sources: dict, anchors: list) -> dict:
    budget = float(cfg["budget"])
    p5 = _json(RUNS / spec["kernel_run"] / "results.json", sources)
    full = p5.get(spec.get("full_feature_mlp", ""), None)
    p5 = p5["p5a_whitebox_qsvm"]
    if abs(float(p5["eps_half"]) - budget) > 1e-12:
        raise ValueError(f"{name}: the kernel run read {p5['eps_half']}, the budget is {budget}")
    labels_file = REPO_ROOT / spec["labels_file"]
    floor = constant_predictor_floor(labels_file, int(p5["n_eval"]), float(p5["clean_f1"]))
    sources[spec["labels_file"]] = hashlib.sha256(labels_file.read_bytes()).hexdigest()

    models = {
        "qsvm": read_model(p5["clean_f1"], p5["per_seed_pgd"], budget, floor["floor"]),
        "rbf_svm": read_model(
            p5["rbf_control"]["clean_f1"], p5["rbf_control"]["per_seed_pgd"], budget, floor["floor"]
        ),
    }
    _anchor(
        anchors,
        f"{name} QSVM white-box retention at the budget",
        models["qsvm"]["at_budget"]["retention"],
        p5["whitebox_retention_at_half"],
    )
    _anchor(
        anchors,
        f"{name} RBF-SVM white-box retention at the budget",
        models["rbf_svm"]["at_budget"]["retention"],
        p5["rbf_control"]["whitebox_retention_at_half"],
    )
    if "full_feature_mlp" in spec:
        # ten refits on all features; other evaluation rows, so no floor from this label file
        models["mlp_full"] = read_model(
            full["clean_f1"]["mean"], full["per_seed_pgd"], budget, None
        )
        _anchor(
            anchors,
            f"{name} full-feature MLP retention at the budget",
            models["mlp_full"]["at_budget"]["retention"],
            full["retention_at_half"],
        )
    if "shared_run" in spec:
        per_seed = _json(RUNS / spec["shared_run"] / "results.json", sources)["per_seed"]
        shared = {}
        for fam in per_seed[0]["evasion"]["curves"]:
            curves = [s["evasion"]["curves"][fam]["pgd"] for s in per_seed]
            clean = float(np.mean([s["evasion"]["clean"][fam] for s in per_seed]))
            per_start = {r: [c[r] for c in curves] for r in curves[0]}
            shared[fam] = read_model(clean, per_start, budget, floor["floor"])
        for fam in spec.get("shared_white_box", []):
            models[fam] = shared[fam]
        # the shared protocol's curve of every family (white-box for the two above, transferred
        # for the rest): reported so the text can say how far its two readings are apart; the
        # exhibits keep the shared protocol at the budget
        shared_protocol = {
            fam: {k: m[k] for k in ("clean", "mean_curve", "at_budget", "worst", "d_retention")}
            for fam, m in shared.items()
        }

    def per_start_retention(m: str, which: str) -> np.ndarray:
        r = next(k for k in models[m]["per_start"] if float(k) == models[m][which]["radius"])
        return np.asarray(models[m]["per_start"][r]) / models[m]["clean"]

    q, r = models["qsvm"], models["rbf_svm"]
    paired = {
        w: paired_wilcoxon(per_start_retention("qsvm", w), per_start_retention("rbf_svm", w))
        for w in ("at_budget", "worst")
    }
    contrast = {
        "qsvm_minus_rbf_at_budget": q["at_budget"]["retention"] - r["at_budget"]["retention"],
        "qsvm_minus_rbf_worst": q["worst"]["retention"] - r["worst"]["retention"],
        "paired_at_budget": paired["at_budget"],
        "paired_worst": paired["worst"],
        "qsvm_below_rbf_at_every_radius": bool(
            all(q["mean_curve"][k] < r["mean_curve"][k] for k in q["mean_curve"])
        ),
    }
    out: dict = {
        "kernel_run": spec["kernel_run"],
        "n_eval": int(p5["n_eval"]),
        "constant_predictor_floor": floor,
        "models": models,
        "contrast": contrast,
    }
    if "shared_run" in spec:
        out["shared_protocol"] = shared_protocol
        out["shared_protocol_largest_d_retention"] = max(
            m["d_retention"] for m in shared_protocol.values()
        )

    if "verdict_run" in spec:
        v = _json(RUNS / spec["verdict_run"] / "results.json", sources)
        rule = spec["registered"]
        if rule["rule"] == "gas":
            block = v["H_C1a"]
            stored_a, stored_q_minus_r = block["A"], -block["B"]
            transfer = block["qsvm_transfer_retention"]
        else:
            stored_a, stored_q_minus_r = v["availability"]["A"], v["fragility"]["F"]
            transfer = v["availability"]["qsvm_transfer_retention"]
        a_budget = transfer - q["at_budget"]["retention"]
        _anchor(anchors, f"{name} registered availability A", a_budget, stored_a)
        _anchor(
            anchors,
            f"{name} registered QSVM minus RBF-SVM white-box retention",
            contrast["qsvm_minus_rbf_at_budget"],
            stored_q_minus_r,
        )
        a_worst = transfer - q["worst"]["retention"]
        both_worst = None
        if "shared_run" in spec:
            # the transferred attack read at its own worst radius too: disclosed beside the
            # registered statistic, which keeps the transfer at the budget
            t = shared["qsvm"]
            _anchor(
                anchors,
                f"{name} QSVM transferred retention at the budget",
                t["at_budget"]["retention"],
                transfer,
            )
            both_worst = {
                "qsvm_transfer_worst_retention": t["worst"]["retention"],
                "qsvm_transfer_worst_radius": t["worst"]["radius"],
                "A": t["worst"]["retention"] - q["worst"]["retention"],
            }
        out["registered"] = {
            "rule": rule,
            "at_budget": registered_verdicts(
                rule, a_budget, contrast["qsvm_minus_rbf_at_budget"], paired["at_budget"]["p_value"]
            ),
            "worst": registered_verdicts(
                rule, a_worst, contrast["qsvm_minus_rbf_worst"], paired["worst"]["p_value"]
            ),
        }
        d = r["d_retention"]
        reading = reading_radius_verdict(
            d, cfg["reading_radius"]["yes_at"], cfg["reading_radius"]["partial_at"]
        )
    else:
        # where the finding was made: the transfer retention is the kernel run's own
        transfer = p5["transferred_retention_at_half"]
        a_budget = transfer - q["at_budget"]["retention"]
        a_worst = transfer - q["worst"]["retention"]
        reading = "origin"
        both_worst = None  # the kernel run stored its transferred attack at the budget only
    out["availability"] = {
        "qsvm_transfer_retention": float(transfer),
        "A_at_budget": float(a_budget),
        "A_worst": float(a_worst),
    }
    if both_worst is not None:
        out["availability"]["transfer_also_at_its_worst_radius"] = both_worst
    out["reading_radius"] = {
        "D_rbf": r["d_retention"],
        "D_qsvm": q["d_retention"],
        "reading": reading,
        "end_state_differs_rbf": r["at_budget"]["end_state"] != r["worst"]["end_state"],
        "end_state_differs_qsvm": q["at_budget"]["end_state"] != q["worst"]["end_state"],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources-file", default=DEFAULT_SOURCES_FILE)
    args = ap.parse_args()
    cfg = load_yaml(REPO_ROOT / args.sources_file)
    sources: dict = {}
    anchors: list = []
    per_source = {
        name: read_source(name, spec, cfg, sources, anchors)
        for name, spec in cfg["sources"].items()
    }
    out = {
        "rule": (
            "a white-box result is read at the grid radius, at or below the budget, with the "
            "lowest mean attacked macro-F1 over the attack starts; a tie goes to the smaller radius"
        ),
        "budget": float(cfg["budget"]),
        "sources": per_source,
    }
    sweeps = cfg.get("end_state_sweeps")
    if sweeps:
        out["end_state_sweeps"] = {
            name: read_sweep(
                _json(RUNS / s["run"] / "results.json", sources)[s["block"]],
                s["axis"],
                float(sweeps["budget"]),
                int(sweeps["persistence_min_radii"]),
            )
            | {"run": s["run"]}
            for name, s in sweeps["sweeps"].items()
        }
    if cfg.get("whole_file"):
        out["whole_file"] = read_whole_file(
            cfg["whole_file"], float(cfg["budget"]), sources, anchors
        )
    out |= {"anchors": anchors, "source_files_sha256": sources}
    with RunDir("whitebox_worst_radius", config={"sources_file": args.sources_file, **cfg}) as run:
        rows = ["source,model,radius,start,attacked_macro_f1"]
        for name, s in per_source.items():
            for model, m in s["models"].items():
                for radius, values in m["per_start"].items():
                    rows += [f"{name},{model},{radius},{i},{v!r}" for i, v in enumerate(values)]
        Path(run.path, "per_start.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        run.write_json("results.json", out)
        for name, s in per_source.items():
            for model, m in s["models"].items():
                log.info(
                    "%s %s: %.4f at %.2f sigma, %.4f at the worst radius %.2f sigma",
                    name,
                    model,
                    m["at_budget"]["retention"],
                    m["at_budget"]["radius"],
                    m["worst"]["retention"],
                    m["worst"]["radius"],
                )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
