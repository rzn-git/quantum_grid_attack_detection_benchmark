"""R4-03 (NS10): does the kernel length scale choose a fail-safe or a fail-silent detector?

A per-class replay of P8a (the RBF-SVM's gamma sweep, 8 multipliers, artifact
`20260814T111131Z_whitebox_qsvm_binary_q8`, key `p8a_lengthscale_control`, attack seeds 0-4)
and P9a (the fidelity-kernel QSVM's bandwidth sweep, 10 bandwidths, artifact
`20260814T180008Z_whitebox_qsvm_binary_q8`, key `p9a_quantum_bandwidth_regime`, attack seeds
0-2, evaluated through the exact complex128 Gram). Each unit is one (axis, gamma multiplier or
bandwidth) and calls `run_whitebox_qsvm.p8a_lengthscale_control` or
`run_whitebox_qsvm.p9a_quantum_bandwidth_regime` unchanged, restricted to that one value, so the
subset, attack seeds, C, white-box PGD settings, RNG streams (`spawn_rng(seed, "p8a_<mult>_<eps>")`,
`"p9a_<bw>_<eps>"`) and numeric gate are the archived ones. Those functions score every
prediction through the module's `_f1`; the replay wraps it (delegating unchanged) to keep each
scored prediction and reads it per class at eps 0.5 and 2.0 sigma: macro-F1, attack-class
recall, natural-class false-alarm rate, evasion (clean attack recall minus attacked) and the
false-alarm rise.

Pre-registration: decision log "2026-11-16T11:07Z — PRE-REGISTRATION (round 4", item R4-03.
Anchor (blocking): every returned cell's seed-mean macro-F1 (clean and every radius) reproduces
the archived value within 1e-6, and the constant-predictor floor matches.
End states are the archived configuration labels at 2 sigma: `qgridbench.eval.tables._regime`
applied to the archived 2-sigma kernel-mass ratio and F1 - floor (the rule behind the paper's
regime exhibits), with a `partial` cell whose |F1 - floor| is inside `_regime`'s floor band
labelled `at_floor` (P9a's finding's "at floor", which `_regime` does not name). Statistic, per
axis at 2 sigma: mean evasion over the cells labelled `inverted` minus mean evasion over the
cells labelled `disabled` or `at_floor` (`partial` cells belong to neither end). HYPOTHESIS if
it is at least 0.30 on both axes, KILL if under 0.30 on both, SPLIT otherwise. The axis
endpoints (the outermost cell of each end) are reported beside it as a sensitivity.

--smoke runs the four axis endpoints (gamma x1 and x0.001; bandwidth 0.05 and 3.0), anchored
like the full run; the statistic needs every labelled cell, so the smoke reports none.

Run (smoke, then full; --worker-rss-gb is the smoke's measured per-worker peak, 0.88 GB, so two
workers fit a 2.5 GB budget; a fidelity unit takes 130-225 s at 3 threads, an RBF unit ~30 s):
  uv run python -m experiments.run_length_scale_per_class --smoke --workers 2 --threads 3 \
      --worker-rss-gb 0.88
  uv run python -m experiments.run_length_scale_per_class --workers 2 --threads 3 \
      --worker-rss-gb 0.88
"""

import argparse
import os
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int)
THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
if THREADS is not None:
    for _var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    ):
        os.environ[_var] = str(THREADS)
elif __name__ == "__main__" and not {"-h", "--help"} & set(sys.argv):
    raise SystemExit("--threads is required (it is applied before numpy is imported)")

import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest import mock  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import yaml  # noqa: E402

from experiments import run_whitebox_qsvm as wq  # noqa: E402
from experiments.run_evasion_per_class import per_class  # noqa: E402
from experiments.runner_guard import assert_clean_predictor  # noqa: E402
from qgridbench.eval.tables import _regime  # noqa: E402
from qgridbench.protocol import prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402

log = get_logger(__name__)

RUNS = REPO_ROOT / "results" / "runs"
ARCHIVE = {
    "rbf": (RUNS / "20260814T111131Z_whitebox_qsvm_binary_q8", "p8a_lengthscale_control"),
    "fidelity": (RUNS / "20260814T180008Z_whitebox_qsvm_binary_q8", "p9a_quantum_bandwidth_regime"),
}
PER_VALUE = {"rbf": "per_width", "fidelity": "per_bandwidth"}
CELL_EPS = ("0.5", "2.0")  # the registered cells; archived keys of both runs
STAT_EPS = "2.0"
ANCHOR_TOL = 1e-6
THRESHOLD = 0.30
INVERTED_END = ("inverted",)
FLOOR_END = ("disabled", "at_floor")
SMOKE_VALUES = {"rbf": ("1.0", "0.001"), "fidelity": ("0.05", "3.0")}


# --------------------------------------------------------------------------- pure reading


def end_state(mass: float, f1_minus_floor: float) -> str:
    """`_regime`'s label, with a `partial` cell at the floor named `at_floor`.

    `_regime(0.0, d)` evaluates only the F1 half of `_regime`'s disabled test (|F1 - floor|
    inside its band), so the floor band is `_regime`'s own and is not restated here.
    """
    label = _regime(mass, f1_minus_floor)
    if label == "partial" and _regime(0.0, f1_minus_floor) == "disabled":
        return "at_floor"
    return label


def split_scored_calls(calls: list, curve_keys: list[str], n_seeds: int) -> dict:
    """Map the `_f1` calls of one P8a/P9a value to their cells.

    Both functions score, in order: the constant-predictor floor, then for each key of the
    returned curve (insertion order "0.0", then the radii ascending) the clean rows once and
    each attacked radius once per attack seed.
    """
    radii = [k for k in curve_keys if k != "0.0"]
    if curve_keys[0] != "0.0" or radii != sorted(radii, key=float):
        raise ValueError(f"unexpected curve key order {curve_keys}")
    want = 2 + len(radii) * n_seeds
    if len(calls) != want:
        raise ValueError(f"{len(calls)} scored calls, expected {want} for {curve_keys}")
    out, i = {"floor": calls[0], "0.0": [calls[1]]}, 2
    for k in radii:
        out[k] = calls[i : i + n_seeds]
        i += n_seeds
    return out


def _ms(xs) -> dict:
    a = np.asarray(xs, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "per_seed": a.tolist(),
    }


def read_cells(split: dict, curve: dict, floor: float, eps_keys=CELL_EPS) -> dict:
    """Per-class figures at clean and each eps key; each cell's seed-mean macro-F1 from the
    kept predictions must equal the macro-F1 the replayed function returned for it, exactly."""
    y0, p0 = split["0.0"][0]
    clean = per_class(y0, p0.argmax(1))
    y_f, p_f = split["floor"]
    floor_f1 = per_class(y_f, p_f.argmax(1))["macro_f1"]
    if floor_f1 != floor:
        raise AssertionError(f"first scored call is not the floor: {floor_f1} != {floor}")
    out = {"clean": clean}
    for k in ("0.0", *eps_keys):
        f1s = [per_class(y, p.argmax(1))["macro_f1"] for y, p in split[k]]
        if float(np.mean(f1s)) != curve[k]["macro_f1"]:
            raise AssertionError(
                f"cell {k}: kept predictions give {np.mean(f1s)}, function returned "
                f"{curve[k]['macro_f1']} (scored calls mapped to the wrong cell)"
            )
    for k in eps_keys:
        reads = [per_class(y, p.argmax(1)) for y, p in split[k]]
        if any(
            r["n_attack"] != clean["n_attack"] or r["n_natural"] != clean["n_natural"]
            for r in reads
        ):
            raise AssertionError(f"cell {k}: per-class counts differ from the clean rows")
        out[k] = {
            "macro_f1": _ms([r["macro_f1"] for r in reads]),
            "attack_recall": _ms([r["attack_recall"] for r in reads]),
            "false_alarm": _ms([r["false_alarm"] for r in reads]),
            "evasion": _ms([clean["attack_recall"] - r["attack_recall"] for r in reads]),
            "false_alarm_rise": _ms([r["false_alarm"] - clean["false_alarm"] for r in reads]),
        }
    return out


def axis_statistic(cells: list[dict], eps_key: str = STAT_EPS) -> dict:
    """Evasion at the inverted end minus evasion at the disabled / at-floor end, one axis.

    `cells`: one dict per value on the axis with "value" (float), "end_state" (the 2-sigma
    configuration label) and "cells"[eps_key]["evasion"]["mean"]. The two ends must occupy
    disjoint ranges of the axis; interleaved labels have no "end" and fail loud.
    """
    inv = [c for c in cells if c["end_state"] in INVERTED_END]
    flo = [c for c in cells if c["end_state"] in FLOOR_END]
    if not inv or not flo:
        raise ValueError(f"axis lacks an end: {len(inv)} inverted, {len(flo)} disabled/at-floor")
    vi, vf = [c["value"] for c in inv], [c["value"] for c in flo]
    if max(vi) < min(vf):
        inv_tip, flo_tip = min(inv, key=lambda c: c["value"]), max(flo, key=lambda c: c["value"])
    elif min(vi) > max(vf):
        inv_tip, flo_tip = max(inv, key=lambda c: c["value"]), min(flo, key=lambda c: c["value"])
    else:
        raise ValueError(
            f"inverted values {sorted(vi)} interleave the floor-end values {sorted(vf)}"
        )

    def ev(c):
        return c["cells"][eps_key]["evasion"]["mean"]

    inv_e, flo_e = float(np.mean([ev(c) for c in inv])), float(np.mean([ev(c) for c in flo]))
    return {
        "eps": eps_key,
        "inverted_end_values": sorted(vi),
        "floor_end_values": sorted(vf),
        "excluded_partial_values": sorted(
            c["value"] for c in cells if c["end_state"] not in INVERTED_END + FLOOR_END
        ),
        "inverted_end_evasion": inv_e,
        "floor_end_evasion": flo_e,
        "difference": inv_e - flo_e,
        "sensitivity_endpoints": {
            "inverted_value": inv_tip["value"],
            "floor_value": flo_tip["value"],
            "difference": ev(inv_tip) - ev(flo_tip),
        },
    }


def branch(differences: dict[str, float], threshold: float = THRESHOLD) -> str:
    hits = [a for a, d in differences.items() if d >= threshold]
    if len(hits) == len(differences):
        return "HYPOTHESIS"
    if not hits:
        return "KILL"
    return "SPLIT"


# --------------------------------------------------------------------------- worker

_CTX: SimpleNamespace | None = None


def _archive(axis: str) -> tuple[dict, dict]:
    run, key = ARCHIVE[axis]
    params = yaml.safe_load((run / "params.yaml").read_text())["config"]
    return params, json.loads((run / "results.json").read_text())[key]


def _context() -> SimpleNamespace:
    """Inputs as run_whitebox_qsvm.main built them, from the archived runs' own params."""
    global _CTX
    if _CTX is None:
        p8, _ = _archive("rbf")
        p9, _ = _archive("fidelity")
        for k in ("variant", "qubits", "encoding", "best_params"):
            if p8[k] != p9[k]:
                raise AssertionError(f"P8a and P9a differ in {k}: {p8[k]} vs {p9[k]}")
        cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
        _CTX = SimpleNamespace(
            params=p9,
            cfg_adv=load_yaml(REPO_ROOT / "configs" / "adversarial.yaml"),
            cfg_q=cfg_q,
            subset_cap=cfg_q["qkernel"]["subset_cap"],
            best=json.loads((REPO_ROOT / p9["best_params"]).read_text()),
            reg=prepare_regime(p9["variant"], "pca", n_components=p9["qubits"], seed=0),
        )
    return _CTX


def _replay(spec: dict) -> tuple[dict, dict, list]:
    """Call the archived function for one value, keeping every prediction it scores."""
    c = _context()
    calls = []
    real_f1 = wq._f1

    def keep(y, proba):
        calls.append((np.array(y, copy=True), np.array(proba, copy=True)))
        return real_f1(y, proba)

    with mock.patch.object(wq, "_f1", keep):
        if spec["axis"] == "rbf":
            out = wq.p8a_lengthscale_control(
                c.reg,
                c.cfg_adv,
                c.best,
                c.subset_cap,
                seed=0,
                gamma_mults=[float(spec["value_key"])],
                attack_seeds=spec["seeds"],
            )
        else:
            # P9a sweeps sorted(config bandwidths | {qk_bw}); a one-value grid and qk_bw set to
            # the same value restrict it to this bandwidth. qk_bw feeds nothing else in P9a
            # (C is qk_C), so the fitted model and its streams are the archived ones.
            bw = float(spec["value_key"])
            cfg_q = copy.deepcopy(c.cfg_q)
            cfg_q["ablations"]["bandwidth_curve"]["bandwidths"] = [bw]
            out = wq.p9a_quantum_bandwidth_regime(
                c.reg,
                SimpleNamespace(encoding=c.params["encoding"]),
                c.cfg_adv,
                cfg_q,
                {**c.best, "qk_bw": bw},
                c.subset_cap,
                seed=0,
                attack_seeds=spec["seeds"],
            )
    per = out[PER_VALUE[spec["axis"]]]
    if list(per) != [spec["value_key"]]:
        raise AssertionError(f"{spec['unit']}: replay returned values {list(per)}")
    return out, per[spec["value_key"]], calls


def run_unit(spec: dict) -> dict:
    from experiments.run_consistent_relay_attack import _peak_rss_gb

    t0 = time.perf_counter()
    out, blk, calls = _replay(spec)
    curve, floor = blk["curve"], out["constant_predictor_floor"]
    split = split_scored_calls(calls, list(curve), len(spec["seeds"]))
    y0, p0 = split["0.0"][0]
    assert_clean_predictor(f"{spec['axis']}@{spec['value_key']}", y0, p0, p0.argmax(1))
    cells = read_cells(split, curve, floor)
    rec = {
        "unit": spec["unit"],
        "axis": spec["axis"],
        "value_key": spec["value_key"],
        "value": float(spec["value_key"]),
        "seeds": spec["seeds"],
        "constant_predictor_floor": floor,
        "n_eval": out["n_eval"],
        "replay_curve": curve,
        "cells": cells,
        "wall_s": time.perf_counter() - t0,
        "peak_rss_gb": _peak_rss_gb(),
        "pid": os.getpid(),
    }
    if spec["axis"] == "rbf":
        rec["gamma"] = blk["gamma"]
    else:
        rec.update(degenerate=blk["degenerate"], numeric_gate=blk["numeric_gate"])
    e = cells[STAT_EPS]
    log.info(
        "%-16s clean recall %.3f | 2 sigma: recall %.3f (evasion %+.3f), FA rise %+.3f, "
        "F1 %.3f | %.0f s, peak %.2f GB",
        spec["unit"],
        cells["clean"]["attack_recall"],
        e["attack_recall"]["mean"],
        e["evasion"]["mean"],
        e["false_alarm_rise"]["mean"],
        e["macro_f1"]["mean"],
        rec["wall_s"],
        rec["peak_rss_gb"],
    )
    return rec


# --------------------------------------------------------------------------- run reading


def plan(smoke: bool) -> list[dict]:
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    specs = []
    for axis in ("fidelity", "rbf"):  # fidelity units are the slow ones; start them first
        params, arch = _archive(axis)
        keys = list(arch[PER_VALUE[axis]])
        if axis == "rbf" and keys != [str(m) for m in params["gamma_mults"]]:
            raise AssertionError(f"P8a archive values {keys} != its gamma_mults")
        if axis == "fidelity":
            grid = sorted(
                set(cfg_q["ablations"]["bandwidth_curve"]["bandwidths"]) | {arch["tuned_bandwidth"]}
            )
            if keys != [str(b) for b in grid]:
                raise AssertionError(f"P9a archive bandwidths {keys} != the config grid {grid}")
        for k in keys:
            if smoke and k not in SMOKE_VALUES[axis]:
                continue
            specs.append(
                {
                    "axis": axis,
                    "value_key": k,
                    "seeds": list(params["seeds"]),
                    "unit": f"{axis}|{k}",
                }
            )
    if smoke and len(specs) != sum(len(v) for v in SMOKE_VALUES.values()):
        raise AssertionError(f"smoke values {SMOKE_VALUES} not all in the archives: {specs}")
    return specs


def anchor(units: list[dict]) -> dict:
    diffs = {}
    for u in units:
        _, arch = _archive(u["axis"])
        diffs[f"{u['unit']}|floor"] = (
            u["constant_predictor_floor"] - arch["constant_predictor_floor"]
        )
        a_curve = arch[PER_VALUE[u["axis"]]][u["value_key"]]["curve"]
        if list(a_curve) != list(u["replay_curve"]):
            raise AssertionError(f"{u['unit']}: radii {list(u['replay_curve'])} != {list(a_curve)}")
        for k, cell in a_curve.items():
            diffs[f"{u['unit']}|{k}"] = u["replay_curve"][k]["macro_f1"] - cell["macro_f1"]
    worst = max(diffs, key=lambda k: abs(diffs[k]))
    failed = {k: v for k, v in diffs.items() if abs(v) > ANCHOR_TOL}
    return {
        "references": {a: r.name for a, (r, _) in ARCHIVE.items()},
        "compared": "seed-mean macro-F1 of every archived cell (clean, every radius) and the floor",
        "n_cells": len(diffs),
        "max_abs_diff": abs(diffs[worst]),
        "worst": worst,
        "failed_cells": failed,
        "tol": ANCHOR_TOL,
        "passed": not failed,
    }


def label(units: list[dict]) -> None:
    """Attach the archived 2-sigma configuration label and each cell's own archived label."""
    for u in units:
        _, arch = _archive(u["axis"])
        a_curve = arch[PER_VALUE[u["axis"]]][u["value_key"]]["curve"]
        top = max(a_curve, key=float)
        if top != STAT_EPS:
            raise AssertionError(f"{u['unit']}: archived top radius {top} != {STAT_EPS}")
        u["end_state"] = end_state(
            a_curve[top]["kernel_mass_ratio"], a_curve[top]["f1_minus_floor"]
        )
        for k in CELL_EPS:
            u["cells"][k]["archived_end_state_at_eps"] = end_state(
                a_curve[k]["kernel_mass_ratio"], a_curve[k]["f1_minus_floor"]
            )


def summarize(units: list[dict]) -> dict:
    table = [
        {
            "axis": u["axis"],
            "value": u["value"],
            "end_state_2sigma": u["end_state"],
            "clean_macro_f1": u["cells"]["clean"]["macro_f1"],
            "clean_attack_recall": u["cells"]["clean"]["attack_recall"],
            "clean_false_alarm": u["cells"]["clean"]["false_alarm"],
            **{
                f"eps{k}_{m}": u["cells"][k][m]["mean"]
                for k in CELL_EPS
                for m in ("macro_f1", "attack_recall", "false_alarm", "evasion", "false_alarm_rise")
            },
            **{f"eps{k}_evasion_std": u["cells"][k]["evasion"]["std"] for k in CELL_EPS},
        }
        for u in units
    ]
    return {"cells": table}


def statistic(units: list[dict]) -> dict:
    by_axis = {a: [u for u in units if u["axis"] == a] for a in ARCHIVE}
    stats = {a: {k: axis_statistic(us, k) for k in CELL_EPS} for a, us in by_axis.items()}
    diffs = {a: stats[a][STAT_EPS]["difference"] for a in stats}
    tips = {a: stats[a][STAT_EPS]["sensitivity_endpoints"]["difference"] for a in stats}
    return {
        "per_axis": stats,
        "differences_at_2sigma": diffs,
        "threshold": THRESHOLD,
        "branch": branch(diffs),
        "sensitivity_endpoints_branch": branch(tips),
        "end_definition": "archived 2-sigma labels: qgridbench.eval.tables._regime on the "
        "archived mass ratio and F1 - floor, partial-at-floor named at_floor (P9a's 'at floor')",
    }


# --------------------------------------------------------------------------- main


def main() -> None:
    from experiments.run_consistent_relay_attack import memory_guard, run_pool_all

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, required=True, help="BLAS/OpenMP/torch per worker")
    ap.add_argument("--workers", type=int, required=True, help="parallel units")
    ap.add_argument(
        "--worker-rss-gb", type=float, required=True, help="per-worker peak RSS (smoke measures)"
    )
    ap.add_argument("--smoke", action="store_true", help="the four axis endpoints only")
    ap.add_argument("--resume", type=Path, help="results_partial.json of an interrupted run")
    args = ap.parse_args()
    if args.threads != THREADS:
        raise AssertionError(f"early thread parse {THREADS} != argparse {args.threads}")
    n_cpu = psutil.cpu_count(logical=True)
    if args.workers * args.threads > n_cpu:
        raise ValueError(f"workers x threads = {args.workers * args.threads} > {n_cpu} CPUs")

    specs = plan(args.smoke)
    cfg = {
        "archives": {a: f"{r.name}:{k}" for a, (r, k) in ARCHIVE.items()},
        "units": [sp["unit"] for sp in specs],
        "cell_eps": list(CELL_EPS),
        "statistic_eps": STAT_EPS,
        "threshold": THRESHOLD,
        "threads": args.threads,
        "workers": args.workers,
        "smoke": args.smoke,
        "resume": str(args.resume) if args.resume else None,
        "preregistration": "decision log 2026-11-16T11:07Z PRE-REGISTRATION (round 4), R4-03",
    }
    name = "length_scale_per_class_binary_q8" + ("_smoke" if args.smoke else "")
    with RunDir(name, config=cfg, seeds=sorted({s for sp in specs for s in sp["seeds"]})) as run:
        guard = memory_guard(args.workers, args.worker_rss_gb, name)
        done = json.loads(args.resume.read_text())["units"] if args.resume else []
        have = {u["unit"] for u in done}
        todo = [sp for sp in specs if sp["unit"] not in have]
        log.info("units: %d to run, %d resumed", len(todo), len(done))

        def save(rec):
            done.append(rec)
            run.write_json("results_partial.json", {"units": done})

        # run_pool_all's initializer checks every worker's thread caps (env and torch) == T
        run_pool_all(run_unit, todo, SimpleNamespace(threads=args.threads), args.workers, save)
        order = {sp["unit"]: i for i, sp in enumerate(specs)}
        units = sorted(done, key=lambda u: order[u["unit"]])
        if [u["unit"] for u in units] != [sp["unit"] for sp in specs]:
            raise AssertionError("finished units do not match the planned units")
        anc = anchor(units)
        label(units)
        summary = summarize(units)
        stat = None if args.smoke else statistic(units)
        timing = {u["unit"]: {k: u[k] for k in ("wall_s", "peak_rss_gb", "pid")} for u in units}
        run.write_json(
            "results.json",
            {
                "config": cfg,
                "memory_guard": guard,
                "anchor": anc,
                "summary": summary,
                "statistic": stat,
                "branch": stat["branch"] if stat else None,
                "timing": timing,
                "units": units,
            },
        )
        run.metrics.update(
            {
                "anchor_passed": anc["passed"],
                "anchor_max_abs_diff": anc["max_abs_diff"],
                "branch": stat["branch"] if stat else None,
                "max_peak_rss_gb": max(u["peak_rss_gb"] for u in units),
                "max_unit_wall_s": max(u["wall_s"] for u in units),
            }
        )
        log.info("ANCHOR %s", json.dumps({k: v for k, v in anc.items() if k != "failed_cells"}))
        log.info("STATISTIC %s", json.dumps(stat))
        print(f"-> {run.path}")
        if not anc["passed"]:
            raise AssertionError(f"anchor failed: {anc['worst']} by {anc['max_abs_diff']:.2e}")


if __name__ == "__main__":
    main()
