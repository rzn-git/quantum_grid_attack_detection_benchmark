"""R4-13 supplementary arm A0': the deployed angle-encoded VQC at Table I's configuration (depth 6,
lr 0.017467604901736582), with R4-13's attacks, seeds and readouts, and the encoding comparison
read against both A0 (depth 4, the robustness suite's file) and A0'.

Ruling: decision log 2026-11-16T13:33Z (R4-23 MATERIAL, ruling 2, the R4-13 bullet). A0' reuses
`run_vqc_encoding_axis.final_unit` unchanged (the A0 path: `VQClassifier`, FittedZoo's 40 epochs and
patience 8, the published `spawn_rng(seed, "adv")` stream), with the parameters of
`results/adv_best_params_binary_q8_table1.json`, which must differ from the grid file only in
`vqc_depth` and `vqc_lr`.

Run (after the main R4-13 run has finished; 10 seeds; peak about 0.95 GB per worker):
  uv run python -m experiments.run_vqc_encoding_a0prime --workers 2 --threads 3 \
      --worker-rss-gb 0.95 --budget-gb 2.05 \
      --main-run results/runs/<stamp>_vqc_encoding_axis_binary_q8
Smoke (seed 0 only; smoke numbers are not results): add --smoke.
Resume: add --resume results/runs/<stamp>_vqc_encoding_a0prime_binary_q8.

Anchor (blocking, 1e-6, every seed): A0' clean F1 on the 500-row subset and white-box PGD F1 at
0.5 sigma equal R4-23's depth-6 arm (`20261116T122057Z_vqc_config_check_binary_q8`, run at 3
threads). Reported: the MLP-surrogate transfer to the VQC against the same arm.
Negative control (blocking): no PGD score above clean (ruling 2026-11-16T12:18Z).
Comparison (reported; the registered branch is unchanged):
  - per-arm summaries (`arm_summary`) for A0, A0', A1, A2, A3;
  - the branch over {A0, A1, A2, A3} (must equal the main run's) and over {A0', A1, A2, A3};
  - per encoding arm, the change in clean F1, PGD F1 at 0.5 sigma, retention and both transfers
    against A0 and against A0', and the 2-sigma end-state labels beside each reference's;
  - paired Wilcoxon over seeds on PGD F1 at 0.5 sigma: A1-A3 against A0 and against A0' (Holm over
    6), and A2 and A3 against A1 (the design note's pair, Holm over 2).
"""

import argparse
import os
import sys

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int)
_THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
if _THREADS is not None:
    for _var in THREAD_VARS:
        os.environ[_var] = str(_THREADS)
elif __name__ == "__main__" and not {"-h", "--help"} & set(sys.argv):
    raise SystemExit("--threads is required (it is applied before numpy is imported)")

import json  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from experiments import run_vqc_encoding_axis as ax  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger  # noqa: E402

log = get_logger(__name__)

ARM = "A0p_table1"
REFS = ("A0_deployed", ARM)
ENCODING_ARMS = ("A1_angle", "A2_zz", "A3_reupload")
GRID_FILE = REPO_ROOT / "results" / "adv_best_params_binary_q8.json"
TABLE1_FILE = REPO_ROOT / "results" / "adv_best_params_binary_q8_table1.json"
R423_RUN = REPO_ROOT / "results" / "runs" / "20261116T122057Z_vqc_config_check_binary_q8"
R423_ARM = "depth6"
TABLE1_VQC = {"vqc_depth": 6, "vqc_lr": 0.017467604901736582}
ANCHOR_TOL = 1e-6
DELTA_KEYS = (
    ("clean500", "mean"),
    ("pgd_0.5", "mean"),
    ("retention_0.5", None),
    ("transfer_mlp_to_arm_0.5", "mean"),
    ("transfer_arm_to_mlp_0.5", "mean"),
)


def table1_params(grid: dict, table1: dict) -> dict:
    """A0's final parameters at Table I's configuration; fails loud on any other difference."""
    diff = {k for k in set(grid) | set(table1) if grid.get(k) != table1.get(k)}
    if diff != set(TABLE1_VQC):
        raise AssertionError(f"table1 file differs from the grid file in {sorted(diff)}")
    for k, v in TABLE1_VQC.items():
        if table1[k] != v:
            raise AssertionError(f"table1 {k} = {table1[k]}, expected {v} (R4-23 depth-6 arm)")
    if table1.get("vqc_epochs", ax.FINAL_EPOCHS) != ax.FINAL_EPOCHS:
        raise AssertionError(f"table1 epochs {table1['vqc_epochs']} != {ax.FINAL_EPOCHS}")
    return {"lr": table1["vqc_lr"], "depth": table1["vqc_depth"], "batch_size": table1["vqc_batch"]}


def a0p_unit(spec: dict) -> dict:
    """`final_unit` on its A0 path (plain `VQClassifier`), relabelled A0'."""
    u = ax.final_unit({**spec, "arm": "A0_deployed"})
    u["arm"] = ARM
    return u


def anchors(units: list[dict], r423_units: list[dict]) -> dict:
    ref = {x["seed"]: x["evasion"] for x in r423_units if x["arm"] == R423_ARM}
    rows = []
    for u in units:
        e = ref[u["seed"]]
        for cell, got, want, blocking in (
            ("clean500", u["clean"]["eval500"], e["clean"]["vqc"], True),
            ("pgd_0.5", u["r2"]["pgd"]["0.5"], e["curves"]["vqc"]["pgd"]["0.5"], True),
            (
                "mlp_to_vqc_0.5",
                u["r4"]["mlp_to_arm"],
                e["transfer_matrix"]["mlp_surrogate"]["vqc"],
                False,
            ),
        ):
            rows.append(
                {
                    "seed": u["seed"],
                    "cell": cell,
                    "got": got,
                    "archived": want,
                    "blocking": blocking,
                    "pass": abs(got - want) <= ANCHOR_TOL,
                }
            )
    return {
        "rows": rows,
        "pass": bool(rows) and all(r["pass"] for r in rows if r["blocking"]),
        "reported_transfer_all_exact": all(r["pass"] for r in rows if not r["blocking"]),
    }


def _get(summary: dict, key: str, sub: str | None) -> float:
    return summary[key] if sub is None else summary[key][sub]


def deltas(summaries: dict[str, dict]) -> dict:
    """Encoding arm minus each reference, on every comparison figure, with the end states."""
    out = {}
    for arm in ENCODING_ARMS:
        s = summaries[arm]
        out[arm] = {}
        for ref in REFS:
            r = summaries[ref]
            out[arm][f"minus_{ref}"] = {
                key: _get(s, key, sub) - _get(r, key, sub) for key, sub in DELTA_KEYS
            }
            out[arm][f"end_state_vs_{ref}"] = {
                obj: {
                    "arm": s["end_state_2sigma"][obj]["label"],
                    "ref": r["end_state_2sigma"][obj]["label"],
                }
                for obj in ("loss", "distance")
            }
    return out


def wilcoxon_tests(units: list[dict]) -> dict:
    """Paired over seeds on PGD F1 at 0.5 sigma; Holm within each family of comparisons."""
    from qgridbench.eval.stats import holm_bonferroni, paired_wilcoxon

    by = {}
    for u in units:
        by.setdefault(u["arm"], {})[u["seed"]] = u["r2"]["pgd"]["0.5"]
    seeds = sorted(by[REFS[0]])
    if any(sorted(v) != seeds for v in by.values()):
        raise AssertionError("arms do not share the same seeds")

    def vec(arm):
        return np.array([by[arm][s] for s in seeds])

    fams = {
        "vs_references": [(a, r) for a in ENCODING_ARMS for r in REFS],
        "design_note_vs_A1": [("A2_zz", "A1_angle"), ("A3_reupload", "A1_angle")],
    }
    out = {}
    for fam, pairs in fams.items():
        res = {f"{a}-{b}": paired_wilcoxon(vec(a), vec(b)) for a, b in pairs}
        holm = holm_bonferroni({k: v["p_value"] for k, v in res.items()})
        for k in res:
            res[k]["p_holm"] = holm[k]
        out[fam] = res
    return {"n_seeds": len(seeds), "metric": "PGD F1 at 0.5 sigma", **out}


def compare(main_units: list[dict], a0p_units: list[dict], main_branch: dict | None) -> dict:
    units = main_units + a0p_units
    arms = sorted({u["arm"] for u in units})
    summaries = {a: ax.arm_summary([u for u in units if u["arm"] == a]) for a in arms}
    main_arms = {a: summaries[a] for a in ("A0_deployed", *ENCODING_ARMS)}
    with_a0p = {a: summaries[a] for a in (ARM, *ENCODING_ARMS)}
    b_main = ax.branch(main_arms)
    if main_branch is not None and b_main["verdict"] != main_branch["verdict"]:
        raise AssertionError(f"branch over the main arms {b_main} != the main run's {main_branch}")
    full = all(s["n_seeds"] == len(ax.SEEDS) for s in summaries.values())
    return {
        "summaries": summaries,
        "branch_main_arms": b_main,
        "branch_with_a0prime": ax.branch(with_a0p),
        "deltas": deltas(summaries),
        "wilcoxon": wilcoxon_tests(units) if full else "skipped: not every arm has 10 seeds",
    }


def main() -> None:
    from experiments.run_xgb_thread_sensitivity import _peak_rss_gb, memory_guard

    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--worker-rss-gb", type=float, required=True)
    ap.add_argument("--budget-gb", type=float, default=2.0)
    ap.add_argument("--chunk", type=int, default=512, help="QGB_VQC_CHUNK per worker")
    ap.add_argument("--main-run", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()

    mem = memory_guard(
        args.workers, args.worker_rss_gb, args.budget_gb - _peak_rss_gb(), "R4-13 A0' workers"
    )
    main_res = json.loads((args.main_run / "results.json").read_text())
    if not args.smoke and "branch" not in main_res:
        raise AssertionError(f"{args.main_run} has no branch: the main R4-13 run is not complete")
    grid = json.loads(GRID_FILE.read_text())
    params = table1_params(grid, json.loads(TABLE1_FILE.read_text()))
    r423 = json.loads((R423_RUN / "results.json").read_text())["units"]
    seeds = ax.SEEDS[:1] if args.smoke else ax.SEEDS
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    cfg = {
        "smoke": args.smoke,
        "workers": args.workers,
        "threads": args.threads,
        "chunk": args.chunk,
        "arm": ARM,
        "params": params,
        "main_run": str(args.main_run),
        "anchor_run": str(R423_RUN.relative_to(REPO_ROOT)),
        "ruling": "decision log 2026-11-16T13:33Z (R4-23, ruling 2, R4-13 bullet)",
        "memory": mem,
    }
    name = "vqc_encoding_a0prime_binary_q8" + ("_smoke" if args.smoke else "")
    with RunDir(name, config=cfg, seeds=list(seeds)) as run:
        (run.path / "git_status_at_launch.txt").write_text(porcelain)
        done: dict[str, dict] = {}
        if args.resume:
            done = json.loads((args.resume / "results_partial.json").read_text())["final"]

        def save(u):
            done[f"{ARM}_s{u['seed']}"] = u
            run.write_json("results_partial.json", {"final": done})
            log.info(
                "%s seed %d done in %.0fs, peak %.2f GB",
                ARM,
                u["seed"],
                u["unit_wall_s"],
                u["peak_rss_gb"],
            )

        specs = [
            {"arm": ARM, "seed": s, "params": params, "mlp_params": grid["mlp"]}
            for s in seeds
            if f"{ARM}_s{s}" not in done
        ]
        ax._pool(a0p_unit, specs, args.workers, args.threads, args.chunk, save)

        units = list(done.values())
        anc = anchors(units, r423)
        neg_ok = all(not u["negative_control"]["pgd_above_clean"] for u in units)
        out = {"anchors": anc, "negative_controls_pass": neg_ok, "final": done}
        if not anc["pass"] or not neg_ok:
            run.write_json("results.json", out)
            raise AssertionError(f"anchor or negative control failed: {anc['rows']}")
        main_units = list(main_res["final"].values())
        out["comparison"] = compare(main_units, units, main_res.get("branch"))
        peak = max(u["peak_rss_gb"] for u in units)
        out["peak_rss_gb"] = {"worker_max": peak, "main": _peak_rss_gb()}
        run.write_json("results.json", out)
        run.metrics.update(
            {
                "anchors_pass": anc["pass"],
                "worker_peak_rss_gb": peak,
                "branch_main_arms": out["comparison"]["branch_main_arms"]["verdict"],
                "branch_with_a0prime": out["comparison"]["branch_with_a0prime"]["verdict"],
            }
        )
        log.info("anchors %s | with A0': %s", anc["pass"], out["comparison"]["branch_with_a0prime"])
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
