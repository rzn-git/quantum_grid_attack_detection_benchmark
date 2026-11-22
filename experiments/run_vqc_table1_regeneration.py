"""R4-24: every paper figure that rests on the depth-4 VQC, at depth 4 and at Table I's VQC.

Analysis only (no model is fit). Reads, per source experiment, three result sets named in a run
map: the archived run (depth 4, the grid file), its depth-4 replay (the anchor run), and its run
at Table I's configuration (depth 6). Writes one row per paper figure to `per_figure.csv` with
the registered branch (decision log 2026-11-16T13:33Z, R4-24 PRE-REGISTRATION):
  test "abs"    KEEP when |depth6 - depth4| < 0.05, else REWRITE
  test "ratio"  a fold figure (forward/reverse, normalized damage, readout mass ratio):
                KEEP when depth6 / depth4 lies in [0.5, 2], else REWRITE (R4-23's reading)
  test "claim"  a verdict (an ordering, a label, a Holm count, a bar): KEEP when it holds as at
                depth 4, else REWRITE with VERDICT-CHANGE
A figure whose sign flips between the two configurations is also flagged VERDICT-CHANGE.
`depth4_reproduced` is the anchor run's value where the anchor run covers the figure.

Anchors (blocking, per source; `anchors.json`): every shared leaf of the anchor run equals the
archive's within 1e-6, and every non-VQC leaf of the depth-6 run equals the archive's within
1e-6. Leaves matching a source's `exclude` tokens (XGBoost, whose fit depends on the thread
count, and the ensembles that contain it) are reported, not gated. A source with no non-VQC leaf
reports its depth-6 anchor as not applicable. A source whose run-map entry sets
`"anchor_rule": "within_archive_restart_range"` (HopSkipJump, not replayable from its seed) is
anchored by every anchor restart lying inside the archive's restart range instead.

Run:  uv run python -m experiments.run_vqc_table1_regeneration --run-map tmp/r4_24/run_map.json
      The run map is {source: {"archive": DIR, "anchor": DIR, "depth6": DIR, "exclude": [...]}}
      with repo-relative run dirs; `DIR#arm` reads arm `arm` of a run_vqc_config_check run. The
      run map and the inventory are copied into the output run directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

from qgridbench.eval.tables import _regime
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger

log = get_logger(__name__)

ANCHOR_TOL = 1e-6
MOVE_TOL = 0.05
RATIO_BAND = (0.5, 2.0)
TIMING_KEYS = {"seconds", "wall_s", "fit_s", "peak_rss_gb"}
ARMS = ("archive", "anchor", "depth6")
CLS5 = ["logreg", "rbf_svm", "rf", "xgb", "lgbm"]
FAMILIES = CLS5 + ["mlp", "qsvm", "vqc"]
RUNS = REPO_ROOT / "results" / "runs"


# --------------------------------------------------------------------------- #
# loading and anchors
# --------------------------------------------------------------------------- #
def result_file(spec: str) -> Path:
    """`DIR` -> DIR/results.json; `DIR/x.json` -> that file (the data-efficiency stage)."""
    path = REPO_ROOT / spec.partition("#")[0]
    return path if path.suffix == ".json" else path / "results.json"


def load(spec: str) -> dict:
    """A run's results file and seeds; `DIR#arm` reads one arm of a run_vqc_config_check run."""
    arm = spec.partition("#")[2]
    f = result_file(spec)
    d = f.parent
    res = json.loads(f.read_text())
    params = yaml.safe_load((d / "params.yaml").read_text()) if (d / "params.yaml").exists() else {}
    if arm:
        res = {"surrogate_summary": res["arms"][arm]["surrogate_summary"]}
    return {"res": res, "seeds": params.get("seeds") or [0], "spec": spec}


def file_sha256(spec: str) -> str:
    return hashlib.sha256(result_file(spec).read_bytes()).hexdigest()


def leaves(obj, path: str = "") -> dict[str, float]:
    """Every numeric leaf (bools and timing keys skipped) as {path: value}."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k not in TIMING_KEYS:
                out.update(leaves(v, f"{path}/{k}"))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(leaves(v, f"{path}/{i}"))
        return out
    if isinstance(obj, bool) or not isinstance(obj, int | float):
        return {}
    return {path: float(obj)}


def is_vqc_path(p: str) -> bool:
    """A leaf involves the VQC when a path component names it (vqc, pgd_vqc, cls3+vqc, the
    deployed_ arm): the VQC is then its target, its source or an ensemble member."""
    return any("vqc" in c or c.startswith("deployed_") for c in p.split("/"))


def compare(ours, ref, exclude=(), only_non_vqc: bool = False) -> dict:
    """Leaf differences on the shared leaves; leaves matching an exclude token reported apart."""
    a, b = leaves(ours), leaves(ref)
    shared = sorted(set(a) & set(b))
    if only_non_vqc:
        shared = [p for p in shared if not is_vqc_path(p)]
        if not shared:
            return {"n_leaves": 0, "passed": None, "note": "no non-VQC leaf in this source"}
    if not shared:
        raise ValueError("no shared leaves to anchor")
    diffs, excl = {}, {}
    for p in shared:
        d = 0.0 if math.isnan(a[p]) and math.isnan(b[p]) else abs(a[p] - b[p])
        hit = any(tok in c for tok in exclude for c in p.split("/"))
        (excl if hit else diffs)[p] = d
    worst = max(diffs, key=diffs.get) if diffs else None
    return {
        "n_leaves": len(diffs),
        "n_vqc_leaves": sum(is_vqc_path(p) for p in diffs),
        "max_abs_diff": diffs[worst] if worst else 0.0,
        "worst": worst,
        "tol": ANCHOR_TOL,
        "passed": bool(diffs[worst] <= ANCHOR_TOL) if worst else None,
        "excluded_tokens": list(exclude),
        "excluded_n_leaves": len(excl),
        "excluded_max_abs_diff": max(excl.values()) if excl else None,
    }


def by_seed(run: dict, sub) -> dict:
    """{seed: subtree} for a per-seed list, {"all": subtree} otherwise."""
    if isinstance(sub, list):
        return {str(s): x for s, x in zip(run["seeds"], sub, strict=True)}
    return {"all": sub}


def tax_cells(r: dict) -> dict:
    return {
        "by_family": r["disagreements_by_family"],
        "n_cells": r["n_cells"],
        "cells": {
            f"{c['family']}|{c['source']}|{c['cell']}": {
                "mass_ratio": c["mass_ratio"],
                "f1_minus_floor": c["f1_minus_floor"],
            }
            for c in r["disagreement_cells"]
        },
    }


# the subtree each source anchors on; keys name the VQC wherever a leaf involves it
ROOTS = {
    "t3np": lambda r: [{"sensor_noise": p["sensor_noise"]} for p in r["per_seed"]],
    "t3np_poison": lambda r: r["poisoning"],
    "rnd": lambda r: [p["random_control"] for p in r["per_seed"]],
    "hsj": lambda r: r["per_family"],
    "rc": lambda r: r["per_seed"],
    "sw": lambda r: r["per_seed"],
    "cs": lambda r: r["per_seed"],
    "p15a": lambda r: {k: r["p15a_transfer_alignment"][k] for k in ("cells", "clean_f1")},
    "p7b8": lambda r: r["p7b_kernel_exhaustion"],
    "p7b12": lambda r: r["p7b_kernel_exhaustion"],
    "e13": lambda r: {
        "vqc_curve": r["curve"],
        "vqc_clean_mass": r["verdict"]["clean_mass_abs"],
        "floor": r["verdict"]["floor"],
    },
    "e12": lambda r: [
        {k: p[k] for k in ("undefended", "defended")} | {"vqc_crafted": p["cross_family"]}
        for p in r["per_seed"]
    ],
    "sq": lambda r: [{k: p[k] for k in ("clean", "square", "random")} for p in r["per_seed"]],
    "ru": lambda r: r["per_seed"],
    "ph": lambda r: r["per_seed"],
    "de": lambda r: r,
    "tax": tax_cells,
}


# the ruling names the published statistic only; with 5 archive restarts a fresh draw lands
# outside their range with probability 2/6 per family by exchangeability, so this is a sanity
# check on the distortion, not a test, and other metrics would only add false failures
RESTART_METRICS = ("median_linf_sigma",)


def anchor_within_restart_range(anchor_res: dict, archive_res: dict) -> dict:
    """Distributional anchor for an attack that cannot be replayed from its seed (HopSkipJump:
    ART draws its initial point from an unseeded RandomState). Every anchor restart's value lies
    inside the archive's [min, max] over its restarts, per family and metric (ruling, decision
    log 2026-11-18T15:40Z)."""
    out, worst = [], None
    for fam, rows in anchor_res["per_family"].items():
        ref = archive_res["per_family"].get(fam)
        if ref is None:
            continue
        for m in RESTART_METRICS:
            lo, hi = min(r[m] for r in ref), max(r[m] for r in ref)
            for r in rows:
                inside = lo <= r[m] <= hi
                out.append(inside)
                if not inside and worst is None:
                    worst = f"/{fam}/{r['restart']}/{m}: {r[m]} outside [{lo}, {hi}]"
    return {
        "rule": "within_archive_restart_range",
        "n_leaves": len(out),
        "n_outside": out.count(False),
        "worst": worst,
        "passed": bool(out) and all(out),
    }


def anchor_source(runs: dict, rmap: dict, src: str) -> dict:
    key = "t3np" if src == "t3np_poison" else src
    excl = tuple(rmap[key].get("exclude", []))
    ref = by_seed(runs[key]["archive"], ROOTS[src](runs[key]["archive"]["res"]))
    out = {}
    for arm in ("anchor", "depth6"):
        if arm not in runs[key]:
            continue
        if arm == "anchor" and rmap[key].get("anchor_rule") == "within_archive_restart_range":
            out[arm] = anchor_within_restart_range(
                runs[key][arm]["res"], runs[key]["archive"]["res"]
            )
            continue
        ours = by_seed(runs[key][arm], ROOTS[src](runs[key][arm]["res"]))
        seeds = sorted(set(ours) & set(ref))
        out[arm] = compare(
            {s: ours[s] for s in seeds},
            {s: ref[s] for s in seeds},
            excl,
            only_non_vqc=arm == "depth6",
        )
        out[arm]["seeds_compared"] = seeds
    return out


# --------------------------------------------------------------------------- #
# figure helpers
# --------------------------------------------------------------------------- #
def seed_mean(ps: list[dict], fn) -> float:
    return float(np.mean([fn(p) for p in ps]))


def clean(ps, f):
    return seed_mean(ps, lambda p: p["evasion"]["clean"][f])


def pgd_at(ps, f, e="0.5"):
    return seed_mean(ps, lambda p: p["evasion"]["curves"][f]["pgd"][e])


def retention(ps, src, tgt):
    return seed_mean(ps, lambda p: p["evasion"]["transfer_matrix"][src][tgt]) / clean(ps, tgt)


def fgsm_pgd(ps) -> dict:
    """|PGD - FGSM| of the seed means over every model x epsilon cell (decision log
    2026-08-15, review item 10: 48 cells, mean 0.0074, max 0.058, Spearman 0.993)."""
    curves = ps[0]["evasion"]["curves"]
    pg, fg = [], []
    for f in curves:
        for e in curves[f]["pgd"]:
            pg.append(pgd_at(ps, f, e))
            fg.append(seed_mean(ps, lambda p, f=f, e=e: p["evasion"]["curves"][f]["fgsm"][e]))
    d = np.abs(np.array(pg) - np.array(fg))
    return {"mean": float(d.mean()), "max": float(d.max()), "rho": float(spearmanr(pg, fg)[0])}


def sw_artifact(res) -> dict:
    """P23 step 1: artefact damage per target (sample-weight minus as-tuned MLP-crafted
    macro-F1) and its Spearman with the target's clean distance from the stratified floor."""
    ps, fl = res["per_seed"], res["summary"]["floor"]
    tg = CLS5 + ["qsvm", "vqc"]
    art = [
        seed_mean(ps, lambda p, t=t: p["f1"]["pgd_mlp_sample_weight"][t] - p["f1"]["pgd_mlp"][t])
        for t in tg
    ]
    dist = [seed_mean(ps, lambda p, t=t: p["f1"]["clean"][t]) - fl for t in tg]
    r = spearmanr(art, dist)
    return {"vqc": art[-1], "rho": float(r[0]), "p": float(r[1]), "art": art}


def ens_delta(res, pair: str, key: str) -> float:
    base = pair.split("+")[0]
    return seed_mean(
        res["per_seed"], lambda p: p["ensembles"][key][pair] - p["ensembles"][key][base]
    )


def p7b(res) -> dict:
    return res["p7b_kernel_exhaustion"]["per_model"]["vqc"]


def end_state(res) -> str:
    c = p7b(res)["2.0"]
    return _regime(c["kernel_mass_ratio"], c["f1_minus_floor"])


def sm(r) -> dict:
    return r.get("summary") or r["surrogate_summary"]


def deployed(r) -> dict:
    (k,) = [k for k in r["summary"] if k.startswith("deployed_")]
    return r["summary"][k]


def flip(r, f: str) -> float:
    return r["per_seed"][0]["random_control"][f]["by_radius"]["0.5"]["flip_rate"]


def p15a_cell(r, s: str, t: str) -> dict:
    return r["p15a_transfer_alignment"]["cells"][s][t]


def archived_hsj_cv() -> dict:
    """The HSJ column's coefficients of variation as make_report merged them when this
    regeneration ran (every archived restart run, later runs overriding earlier ones).
    make_report now reads the pooled run of experiments/pool_hsj_restarts.py instead."""
    out = {}
    for d in sorted(RUNS.glob("*probe_hsj_restarts_binary_q8")):
        if (d / "results.json").exists():
            out.update(json.loads((d / "results.json").read_text())["summary"])
    return {f: v["cv"] for f, v in out.items()}


# --------------------------------------------------------------------------- #
# the figures: (inventory ids, figure, source, test, extractor on results.json)
# --------------------------------------------------------------------------- #
def figures(ctx: dict) -> list[tuple]:
    F: list[tuple] = []

    def add(ids, name, src, test, fn):
        F.append((ids, name, src, test, fn))

    arms = (
        ("mlp", "as tuned", "1 3 4 5 7 10 42 45 59", "8 58"),
        ("mlp_balanced", "balanced", "4 5 10 15 42 45 61", "9 60"),
        ("rbf_svm", "RBF-SVM", "12 15 63", "62"),
    )
    for arm, tag, fr_ids, dmg_ids in arms:
        add(
            fr_ids,
            f"forward/reverse, {tag} surrogate",
            "sur",
            "ratio",
            lambda r, a=arm: sm(r)[a]["forward_over_reverse"],
        )
        add(
            dmg_ids,
            f"VQC damage, {tag} surrogate",
            "sur",
            "abs",
            lambda r, a=arm: sm(r)[a]["vqc_damage"],
        )
        add(
            "13 14 64 65",
            f"VQC damage over classical damage, {tag} surrogate",
            "sur",
            "ratio",
            lambda r, a=arm: sm(r)[a]["damage_ratio_vqc_over_classical"],
        )
    add(
        "6",
        "VQC macro-F1 under the as-tuned MLP surrogate",
        "sur",
        "abs",
        lambda r: sm(r)["mlp"]["per_target_adv_f1"]["vqc"],
    )
    add(
        "9",
        "claim: balanced-surrogate VQC damage below the classical mean",
        "sur",
        "claim",
        lambda r: sm(r)["mlp_balanced"]["vqc_damage"] < sm(r)["mlp_balanced"]["classical_damage"],
    )

    add("18", "Table III VQC clean macro-F1", "t3", "abs", lambda r: clean(r["per_seed"], "vqc"))
    add(
        "19",
        "Table III VQC PGD macro-F1 at 0.5 sigma",
        "t3",
        "abs",
        lambda r: pgd_at(r["per_seed"], "vqc"),
    )
    add(
        "20",
        "Table III VQC PGD macro-F1 minus floor",
        "t3",
        "abs",
        lambda r: pgd_at(r["per_seed"], "vqc") - ctx["t3_floor"],
    )
    add(
        "16 17 21 22",
        "Table III VQC retention (white-box)",
        "t3",
        "abs",
        lambda r: pgd_at(r["per_seed"], "vqc") / clean(r["per_seed"], "vqc"),
    )
    add(
        "17",
        "claim: VQC white-box retention below the MLP's 0.621 and the RBF-SVM's 0.685",
        "t3",
        "claim",
        lambda r: pgd_at(r["per_seed"], "vqc") / clean(r["per_seed"], "vqc") < 0.621,
    )
    add(
        "2 25 44",
        "Holm-significant quantum-classical pairs (of 12)",
        "t3stats",
        "abs",
        lambda r: r["n_significant_holm"],
    )
    add(
        "2 25 44",
        "claim: all 12 quantum-classical pairs Holm-significant",
        "t3stats",
        "claim",
        lambda r: r["n_significant_holm"] == r["n_comparisons"] == 12,
    )
    for src in ("mlp", "mlp_surrogate"):
        add(
            "66",
            f"transfer retention {src} -> vqc",
            "t3",
            "abs",
            lambda r, s=src: retention(r["per_seed"], s, "vqc"),
        )
    for tgt in FAMILIES:
        add(
            "66",
            f"transfer retention vqc -> {tgt}",
            "t3",
            "abs",
            lambda r, t=tgt: retention(r["per_seed"], "vqc", t),
        )
    for e in ("0.01", "0.05", "0.1", "0.2", "0.3", "0.5"):
        add(
            "81",
            f"robustness_pgd VQC PGD macro-F1 at {e} sigma",
            "t3",
            "abs",
            lambda r, e=e: pgd_at(r["per_seed"], "vqc", e),
        )
    for k in ("mean", "max", "rho"):
        add(
            "82",
            f"FGSM/PGD agreement over 48 cells: {k}",
            "t3",
            "abs",
            lambda r, k=k: fgsm_pgd(r["per_seed"])[k],
        )

    for snr in ("30", "20", "10"):
        add(
            "83",
            f"poisoning_sensor VQC macro-F1 under sensor noise, SNR {snr} dB",
            "t3np",
            "abs",
            lambda r, x=snr: seed_mean(r["per_seed"], lambda p: p["sensor_noise"]["vqc"][x]),
        )
    for rate in ("0.0", "0.05", "0.1", "0.2"):
        add(
            "83",
            f"poisoning_sensor VQC macro-F1 at label-flip rate {rate} (seed 0)",
            "t3np",
            "abs",
            lambda r, x=rate: r["poisoning"][x]["vqc"],
        )

    add(
        "23",
        "VQC random-perturbation flip rate at 0.5 sigma",
        "rnd",
        "abs",
        lambda r: flip(r, "vqc"),
    )
    add(
        "26",
        "claim: the QSVM's 45% flip rate is the highest of all eight",
        "rnd",
        "claim",
        lambda r: flip(r, "vqc") < ctx["qsvm_flip"],
    )

    add(
        "24",
        "VQC HopSkipJump median distortion, mean over 5 restarts",
        "hsj",
        "abs",
        lambda r: r["summary"]["vqc"]["mean"],
    )
    add(
        "24",
        "VQC HopSkipJump median distortion, std over 5 restarts",
        "hsj",
        "abs",
        lambda r: r["summary"]["vqc"]["std"],
    )
    add(
        "29",
        "VQC HopSkipJump restart coefficient of variation",
        "hsj",
        "abs",
        lambda r: r["summary"]["vqc"]["cv"],
    )
    add(
        "29",
        "claim: CV order QSVM > XGBoost > VQC > LightGBM",
        "hsj",
        "claim",
        lambda r: ctx["hsj_cv"]["lgbm"] < r["summary"]["vqc"]["cv"] < ctx["hsj_cv"]["xgb"],
    )

    add(
        "27 28",
        "VQC Square damage at 0.5 sigma",
        "sq",
        "abs",
        lambda r: r["verdict"]["damage"]["vqc"]["square_damage"],
    )
    add(
        "27 28",
        "claim: the QSVM's Square damage (0.556) is the largest of eight",
        "sq",
        "claim",
        lambda r: r["verdict"]["damage"]["vqc"]["square_damage"] < ctx["qsvm_square"],
    )

    add(
        "30 79",
        "E13 minimum |<Z>| ratio under magnitude descent",
        "e13",
        "ratio",
        lambda r: r["verdict"]["min_mass_ratio_descent"],
    )
    add(
        "30 79",
        "claim: descent never reaches the disabled bar",
        "e13",
        "claim",
        lambda r: not r["verdict"]["disabled_regime_reached_at"],
    )
    add(
        "31 79",
        "E13 |<Z>| ratio under loss ascent at 0.1 sigma",
        "e13",
        "ratio",
        lambda r: r["curve"]["0.1"]["loss_objective"]["mass_ratio"],
    )
    add(
        "32 80",
        "E13 macro-F1 under loss ascent at 2 sigma",
        "e13",
        "abs",
        lambda r: r["curve"]["2.0"]["loss_objective"]["macro_f1"],
    )
    add(
        "32 80",
        "claim: loss ascent inverts at 2 sigma (F1 - floor <= -0.2)",
        "e13",
        "claim",
        lambda r: r["curve"]["2.0"]["loss_objective"]["f1_minus_floor"] <= -0.2,
    )
    add(
        "33 80",
        "E13 macro-F1 under magnitude descent at 0.5 sigma",
        "e13",
        "abs",
        lambda r: r["curve"]["0.5"]["mass_descent_objective"]["macro_f1"],
    )
    add(
        "33 80",
        "claim: descent parks at the floor at 0.5 sigma (|F1 - floor| <= 0.05)",
        "e13",
        "claim",
        lambda r: abs(r["curve"]["0.5"]["mass_descent_objective"]["f1_minus_floor"]) <= 0.05,
    )
    add(
        "52",
        "clean mean |<Z>| on the P7c probe's 200 points (E13)",
        "e13",
        "abs",
        lambda r: r["verdict"]["clean_mass_abs"],
    )

    for rad in ("0.0", "0.01", "0.05", "0.1", "0.2", "0.3", "0.5", "1.0", "2.0"):
        add(
            "50",
            f"exhaustion table VQC m/m0 at {rad} sigma",
            "p7b8",
            "ratio",
            lambda r, x=rad: p7b(r)[x]["kernel_mass_ratio"],
        )
        add(
            "51",
            f"exhaustion table VQC F1 - constant at {rad} sigma",
            "p7b8",
            "abs",
            lambda r, x=rad: p7b(r)[x]["f1_minus_floor"],
        )
    add(
        "31 48",
        "VQC |<Z>| ratio at 0.1 sigma (P7c)",
        "p7b8",
        "ratio",
        lambda r: p7b(r)["0.1"]["kernel_mass_ratio"],
    )
    add(
        "48",
        "claim: |<Z>| dips below clean, then rises above it by 2 sigma",
        "p7b8",
        "claim",
        lambda r: (
            min(c["kernel_mass_ratio"] for c in p7b(r).values()) < 1.0
            and p7b(r)["2.0"]["kernel_mass_ratio"] > 1.0
        ),
    )
    add(
        "49",
        "VQC predicted-probability spread at 2 sigma",
        "p7b8",
        "abs",
        lambda r: p7b(r)["2.0"]["proba_std"],
    )
    add(
        "53",
        "VQC |<Z>| growth at 1 sigma",
        "p7b8",
        "ratio",
        lambda r: p7b(r)["1.0"]["kernel_mass_ratio"],
    )
    add(
        "53 56",
        "VQC |<Z>| growth at 2 sigma (m/m0)",
        "p7b8",
        "ratio",
        lambda r: p7b(r)["2.0"]["kernel_mass_ratio"],
    )
    for q, ids in (("p7b8", "56"), ("p7b12", "57")):
        add(
            ids,
            f"end-state row ({q}): clean macro-F1",
            q,
            "abs",
            lambda r: p7b(r)["0.0"]["macro_f1"],
        )
        add(
            ids,
            f"end-state row ({q}): F1 - constant at 2 sigma",
            q,
            "abs",
            lambda r: p7b(r)["2.0"]["f1_minus_floor"],
        )
        add(
            "55 56" if q == "p7b8" else ids,
            f"claim: the VQC ends Inverted ({q})",
            q,
            "claim",
            lambda r: end_state(r) == "inverted",
        )
    add(
        "57",
        "end-state row (p7b12): m/m0 at 2 sigma",
        "p7b12",
        "ratio",
        lambda r: p7b(r)["2.0"]["kernel_mass_ratio"],
    )

    add(
        "34",
        "taxonomy: configurations whose single-radius label differs",
        "tax",
        "abs",
        lambda r: r["n_disagreements"],
    )
    add("34", "taxonomy: recorded configurations", "tax", "abs", lambda r: r["n_cells"])
    add(
        "34",
        "claim: the persistence rule separates all of them (refined branch unchanged)",
        "tax",
        "claim",
        lambda r: r["refined_branch"] == ctx["tax_refined"],
    )

    for size in ("250", "500", "1000", "2000"):
        add(
            "47",
            f"data_efficiency VQC macro-F1 at {size}",
            "de",
            "abs",
            lambda r, x=size: r["vqc"][x]["mean"],
        )
    for size in ("250", "2000"):
        add(
            "35 46",
            f"XGBoost minus VQC at {size}",
            "de",
            "abs",
            lambda r, x=size: r["xgb_full"][x]["mean"] - r["vqc"][x]["mean"],
        )
    add(
        "47",
        "claim: XGBoost leads the VQC at every size, by more at 2000 than at 250",
        "de",
        "claim",
        lambda r: (
            all(r["xgb_full"][x]["mean"] > r["vqc"][x]["mean"] for x in r["vqc"])
            and r["xgb_full"]["2000"]["mean"] - r["vqc"]["2000"]["mean"]
            > r["xgb_full"]["250"]["mean"] - r["vqc"]["250"]["mean"]
        ),
    )

    add(
        "36 37",
        "VQC robustness gain under single-round augmentation",
        "e12",
        "abs",
        lambda r: r["verdict"]["summary"]["vqc"]["robustness_gain"],
    )
    add(
        "36 37",
        "claim: the VQC's gain is short of the 0.05 bar",
        "e12",
        "claim",
        lambda r: r["verdict"]["summary"]["vqc"]["robustness_gain"] < 0.05,
    )
    add(
        "38 39",
        "VQC clean cost of augmentation (the MLP's is 0.0042)",
        "e12",
        "abs",
        lambda r: r["verdict"]["summary"]["vqc"]["clean_cost"],
    )

    add(
        "43",
        "rounding cost, max over seeds and families",
        "ph",
        "abs",
        lambda r: max(
            abs(p["f1"]["pgd_rounded"][f] - p["f1"]["pgd"][f])
            for p in r["per_seed"]
            for f in p["f1"]["pgd"]
        ),
    )
    add(
        "43",
        "rounding cost, VQC max over seeds",
        "ph",
        "abs",
        lambda r: max(
            abs(p["f1"]["pgd_rounded"]["vqc"] - p["f1"]["pgd"]["vqc"]) for p in r["per_seed"]
        ),
    )

    add(
        "84",
        "deployed VQC white-box retention, mean over 3 seeds",
        "ru",
        "abs",
        lambda r: deployed(r)["retention_mean"],
    )
    add(
        "84",
        "deployed VQC white-box retention, std over 3 seeds",
        "ru",
        "abs",
        lambda r: deployed(r)["retention_std"],
    )
    add(
        "85",
        "re-uploading arm retention, mean (hard-coded arm)",
        "ru",
        "abs",
        lambda r: r["summary"]["reupload_d4_lr0.05"]["retention_mean"],
    )

    add(
        "68",
        "sample-weight MLP forward/reverse",
        "sw",
        "ratio",
        lambda r: r["summary"]["mlp_sample_weight"]["forward_over_reverse"],
    )
    add(
        "11 15 67",
        "torch weighted MLP forward/reverse (the P19f run)",
        "sw",
        "ratio",
        lambda r: r["summary"]["mlp_torch_weighted"]["forward_over_reverse"],
    )
    add("69", "artefact damage of the VQC", "sw", "abs", lambda r: sw_artifact(r)["vqc"])
    add(
        "69",
        "claim: the VQC carries the largest artefact damage of the seven",
        "sw",
        "claim",
        lambda r: sw_artifact(r)["vqc"] == max(sw_artifact(r)["art"]),
    )
    add(
        "70",
        "Spearman, artefact damage vs distance from the floor (7 targets)",
        "sw",
        "abs",
        lambda r: sw_artifact(r)["rho"],
    )
    add("70", "Spearman p-value", "sw", "abs", lambda r: sw_artifact(r)["p"])
    add(
        "70",
        "claim: artefact damage does not track the floor distance (p >= 0.05)",
        "sw",
        "claim",
        lambda r: sw_artifact(r)["p"] >= 0.05,
    )

    add(
        "71",
        "VQC damage from one constant displacement",
        "cs",
        "abs",
        lambda r: r["summary"]["vqc"]["damage_constant_prior"],
    )
    add(
        "71",
        "VQC damage from the point-specific residual",
        "cs",
        "abs",
        lambda r: r["summary"]["vqc"]["damage_perpoint_residual"],
    )
    add(
        "72",
        "claim: a constant displacement moves the VQC least of all eight",
        "cs",
        "claim",
        lambda r: (
            min(r["summary"], key=lambda f: abs(r["summary"][f]["damage_constant_prior"])) == "vqc"
        ),
    )

    pv = "p15a_transfer_alignment"
    add(
        "73",
        "Spearman, alignment vs damage",
        "p15a",
        "abs",
        lambda r: r[pv]["verdict"]["spearman_alignment_vs_drop"]["rho"],
    )
    add(
        "73",
        "Spearman p-value",
        "p15a",
        "abs",
        lambda r: r[pv]["verdict"]["spearman_alignment_vs_drop"]["p"],
    )
    for s, t, ids in (
        ("mlp", "vqc", "74"),
        ("vqc", "mlp", "74"),
        ("mlp", "mlp", "75"),
        ("vqc", "vqc", "75"),
        ("vqc", "rbf_svm", "76"),
        ("vqc", "qsvm", "76"),
    ):
        add(
            ids,
            f"alignment {s} -> {t}",
            "p15a",
            "abs",
            lambda r, s=s, t=t: p15a_cell(r, s, t)["alignment"],
        )
    add(
        "74",
        "claim: alignment runs opposite to damage (MLP->VQC below VQC->MLP)",
        "p15a",
        "claim",
        lambda r: (
            p15a_cell(r, "mlp", "vqc")["alignment"] < p15a_cell(r, "vqc", "mlp")["alignment"]
            and p15a_cell(r, "mlp", "vqc")["drop"] > p15a_cell(r, "vqc", "mlp")["drop"]
        ),
    )
    add(
        "75",
        "claim: self-transfer is the maximum-alignment cell",
        "p15a",
        "claim",
        lambda r: r[pv]["controls"]["self_transfer_is_max_alignment"],
    )
    add(
        "76",
        "claim: VQC-source alignment to both kernel machines within 0.05 of zero",
        "p15a",
        "claim",
        lambda r: all(
            abs(p15a_cell(r, "vqc", t)["alignment"]) <= 0.05 for t in ("rbf_svm", "qsvm")
        ),
    )

    vqc_pairs = ("cls3+vqc", "cls5+vqc")
    sets = ("clean", "pgd_mlp", "pgd_vqc", "pgd_mlp_weighted")
    for pair in vqc_pairs:
        for k in sets:
            add(
                "77",
                f"ensemble {pair} minus base, {k}",
                "rc",
                "abs",
                lambda r, p=pair, k=k: ens_delta(r, p, k),
            )
    for pair in ("cls3+qsvm", "cls5+qsvm"):
        add(
            "78",
            f"ensemble {pair} minus base, pgd_vqc",
            "rc",
            "abs",
            lambda r, p=pair: ens_delta(r, p, "pgd_vqc"),
        )
    add(
        "77",
        "claim: adding the VQC never helps (every delta <= 0)",
        "rc",
        "claim",
        lambda r: all(ens_delta(r, p, k) <= 0 for p in vqc_pairs for k in sets),
    )
    add(
        "78",
        "claim: adding the QSVM helps against VQC-crafted attacks",
        "rc",
        "claim",
        lambda r: all(ens_delta(r, p, "pgd_vqc") > 0 for p in ("cls3+qsvm", "cls5+qsvm")),
    )
    return F


# --------------------------------------------------------------------------- #
# the branch
# --------------------------------------------------------------------------- #
def branch(test: str, d4, d6) -> dict:
    if test == "claim":
        same = bool(d4) == bool(d6)
        return {
            "abs_change": None,
            "ratio": None,
            "ratio_check": None,
            "branch": "KEEP" if same else "REWRITE",
            "verdict_change": not same,
        }
    change = abs(d6 - d4)
    ratio = d6 / d4 if d4 else float("nan")
    in_band = RATIO_BAND[0] <= ratio <= RATIO_BAND[1]
    keep = in_band if test == "ratio" else change < MOVE_TOL
    return {
        "abs_change": change,
        "ratio": ratio,
        "ratio_check": in_band if test == "ratio" else None,
        "branch": "KEEP" if keep else "REWRITE",
        "verdict_change": bool(d4 and d6 and (d4 > 0) != (d6 > 0)),
    }


def row(inv_row: dict, **kw) -> dict:
    base = {
        "inventory_id": inv_row["id"],
        "file": inv_row["file"],
        "line": inv_row["line"],
        "quote": inv_row["quote"],
        "figure": None,
        "test": None,
        "depth4_archived": None,
        "depth4_reproduced": None,
        "depth6": None,
        "abs_change": None,
        "ratio_depth6_over_depth4": None,
        "ratio_check": None,
        "branch": None,
        "verdict_change": None,
        "source_run_depth4": None,
        "source_run_depth6": None,
    }
    return base | kw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-map", required=True, type=Path)
    ap.add_argument("--inventory", default="tmp/r4_24/vqc_inventory.csv", type=Path)
    args = ap.parse_args()

    from qgridbench.eval.stats import robustness_stats

    rmap = json.loads((REPO_ROOT / args.run_map).read_text())
    runs = {s: {a: load(v[a]) for a in ARMS if v.get(a)} for s, v in rmap.items()}
    runs["t3stats"] = {
        a: {"res": robustness_stats(REPO_ROOT / rmap["t3"][a]), "seeds": [0], "spec": rmap["t3"][a]}
        for a in ARMS
        if rmap["t3"].get(a)
    }
    inv = {r["id"]: r for r in csv.DictReader((REPO_ROOT / args.inventory).open(encoding="utf-8"))}

    arch = {s: runs[s]["archive"]["res"] for s in runs}
    t3_rows = list(csv.DictReader((REPO_ROOT / "paper/tables/robustness_binary_q8.csv").open()))
    (vqc_row,) = [r for r in t3_rows if r["model"] == "vqc"]
    ctx = {
        "t3_floor": float(vqc_row["adv_f1"]) - float(vqc_row["adv_f1_minus_floor"]),
        "qsvm_flip": flip(arch["rnd"], "qsvm"),
        "qsvm_square": arch["sq"]["verdict"]["damage"]["qsvm"]["square_damage"],
        "hsj_cv": archived_hsj_cv(),
        "tax_refined": arch["tax"]["refined_branch"],
    }

    anchors = {src: anchor_source(runs, rmap, src) for src in ROOTS}
    # the depth-6 noise/poison run replays the evasion stage first (the noise stage reads the
    # rng after it); its evasion must equal R4-23's depth-6 arm, which the figures use
    r6 = runs["t3np"]["depth6"]["res"]["per_seed"]
    ref6 = runs["t3"]["depth6"]["res"]["per_seed"]
    anchors["t3np_evasion_vs_r4_23_depth6"] = {
        "depth6": compare(
            [{"evasion": p["evasion"]} for p in r6],
            [{"evasion": p["evasion"]} for p in ref6],
            tuple(rmap["t3np"].get("exclude", [])),
        )
    }

    rows = []
    for ids, name, src, test, fn in figures(ctx):
        vals = {a: fn(runs[src][a]["res"]) for a in ARMS if a in runs[src]}
        b = branch(test, vals["archive"], vals["depth6"])
        for i in ids.split():
            rows.append(
                row(
                    inv[i],
                    figure=name,
                    test=test,
                    depth4_archived=vals["archive"],
                    depth4_reproduced=vals.get("anchor"),
                    depth6=vals["depth6"],
                    abs_change=b["abs_change"],
                    ratio_depth6_over_depth4=b["ratio"],
                    ratio_check=b["ratio_check"],
                    branch=b["branch"],
                    verdict_change=b["verdict_change"],
                    source_run_depth4=runs[src]["archive"]["spec"],
                    source_run_depth6=runs[src]["depth6"]["spec"],
                )
            )
    # row 54: clean |<Z>| (E13) times the 2-sigma growth (P7c), two sources
    v = {
        a: runs["e13"][a]["res"]["verdict"]["clean_mass_abs"]
        * p7b(runs["p7b8"][a]["res"])["2.0"]["kernel_mass_ratio"]
        for a in ARMS
        if a in runs["e13"] and a in runs["p7b8"]
    }
    b = branch("abs", v["archive"], v["depth6"])
    rows.append(
        row(
            inv["54"],
            figure="clean |<Z>| times its 2-sigma growth",
            test="abs",
            depth4_archived=v["archive"],
            depth4_reproduced=v.get("anchor"),
            depth6=v["depth6"],
            abs_change=b["abs_change"],
            ratio_depth6_over_depth4=b["ratio"],
            branch=b["branch"],
            verdict_change=b["verdict_change"],
            source_run_depth4=f"{rmap['e13']['archive']} x {rmap['p7b8']['archive']}",
            source_run_depth6=f"{rmap['e13']['depth6']} x {rmap['p7b8']['depth6']}",
        )
    )
    for i in ("40", "41"):  # E12b Madry: R4-08 reruns it at the Table I configuration
        rows.append(
            row(
                inv[i],
                figure=inv[i]["what"],
                branch="OUT-OF-SCOPE (R4-08)",
                source_run_depth4=inv[i]["source_run"],
            )
        )
    missing = sorted(set(inv) - {r["inventory_id"] for r in rows}, key=int)
    if missing:
        raise KeyError(f"inventory rows without a figure: {missing}")
    rows.sort(key=lambda r: int(r["inventory_id"]))

    scored = [r for r in rows if r["branch"] in ("KEEP", "REWRITE")]
    counts = {
        "figure_rows": len(scored),
        "KEEP": sum(r["branch"] == "KEEP" for r in scored),
        "REWRITE": sum(r["branch"] == "REWRITE" for r in scored),
        "VERDICT-CHANGE": sum(bool(r["verdict_change"]) for r in scored),
        "inventory_rows": len(inv),
        "inventory_rows_with_a_REWRITE": len(
            {r["inventory_id"] for r in scored if r["branch"] == "REWRITE"}
        ),
        "inventory_rows_out_of_scope": 2,
    }
    failed = sorted(
        f"{src}/{arm}"
        for src, rec in anchors.items()
        for arm, x in rec.items()
        if x.get("passed") is False
    )
    with RunDir(
        "vqc_table1_regeneration_binary",
        config={"run_map": rmap, "inventory": args.inventory.as_posix()},
        seeds=sorted({s for v in runs.values() if "depth6" in v for s in v["depth6"]["seeds"]}),
    ) as run:
        with (run.path / "per_figure.csv").open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        shutil.copy(REPO_ROOT / args.inventory, run.path / "inventory.csv")
        shutil.copy(REPO_ROOT / args.run_map, run.path / "run_map.json")
        run.write_json("anchors.json", anchors)
        run.write_json(
            "input_hashes.json",
            {f"{s}/{a}": file_sha256(rmap[s][a]) for s in rmap for a in ARMS if rmap[s].get(a)},
        )
        run.write_json(
            "results.json",
            {
                "counts": counts,
                "anchors_failed": failed,
                "verdict_changes": [r for r in scored if r["verdict_change"]],
                "context": ctx,
            },
        )
        run.metrics.update(counts | {"anchors_failed": len(failed)})
        log.info("COUNTS %s | anchors failed: %s", counts, failed)
        print(f"-> {run.path}")
        if failed:
            raise AssertionError(f"anchor failed: {failed}")


if __name__ == "__main__":
    main()
