"""R4-13 attack-strength control: does A2's (the kernel ZZ map's) white-box retention survive a
stronger attack, or is it an under-attack artifact of the periodic map?

Run (A1 and A2, seeds 0-9; refits each VQC with the main run's seed and settings):
  uv run python -m experiments.run_vqc_encoding_attack_control --workers 1 --threads 3 \
      --worker-rss-gb 1.05 --budget-gb 2.05 \
      --main-run results/runs/20261116T122156Z_vqc_encoding_axis_binary_q8
Smoke (seed 0, 10-step PGD x 2 restarts, 20 Square iterations; numbers are not results): --smoke.
Resume: add --resume results/runs/<stamp>_vqc_encoding_attack_control_binary_q8.

Pre-registration: decision log 2026-11-16T14:58Z (R4-13 PROVISIONAL, the control). Per seed, at
0.5 sigma on the main run's 500 test rows, three attacks on each of A1 and A2:
  pgd20    the original 20-step PGD (step 0.25 eps), replayed on the main run's exact stream
           (`spawn_rng(seed, "adv")`, the arm's set at 0.5 sigma is the stream's 12th draw)
  pgd100x5 100-step PGD, 5 random restarts, worst case per row over restarts (the restart with
           the highest BCE loss); step 0.025 eps, Madry et al.'s 2.5 eps / steps rule
  square   the shipped Square-schedule random search (`run_square_blackbox.square_attack`) at its
           shipped budget, 600 iterations on the margin loss
Retention per (arm, attack) = seed-mean attacked F1 / seed-mean clean F1 (R4-13's definition).
The strongest attack for an arm is the one with its lowest seed-mean attacked F1.
  HOLDS      A2's retention under its strongest attack > 0.30, AND A2 minus A1 (each under its
             strongest attack) >= 0.10
  COLLAPSES  A2's retention under its strongest attack is within 0.05 of A1's
  INTERMEDIATE  otherwise (amendment 2026-11-16, "R4-13 attack-strength control")
Reported: the per-row worst case over all three attacks, as a sensitivity.
Anchors (blocking, 1e-6, every seed): the refit model's clean F1 on the 500 rows, and pgd20,
equal the main run's per-seed values.
Negative control (blocking): no PGD score above clean; Square above clean is reported.
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
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from experiments import run_vqc_encoding_axis as ax  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger  # noqa: E402

log = get_logger(__name__)

ARMS = ("A1_angle", "A2_zz")
EPS = 0.5
PGD100 = {"n_steps": 100, "step_frac": 0.025, "restarts": 5}
SQUARE_ITER = 600  # run_square_blackbox's shipped --max-iter
SMOKE = {"n_steps": 10, "step_frac": 0.25, "restarts": 2, "square_iter": 20}
HOLD_RETENTION = 0.30
HOLD_GAP = 0.10
COLLAPSE_BAND = 0.05
ANCHOR_TOL = 1e-6
ATTACKS = ("pgd20", "pgd100x5", "square")


# --------------------------------------------------------------------------- pure reading


def stream_draws_before(eps_grid: list[float], eps: float) -> int:
    """Uniform draws the published stream makes before the arm's own PGD set at `eps`:
    run_adversarial.evasion_and_transfer crafts mlp then the arm at each epsilon in order,
    one draw per PGD call."""
    return 2 * list(eps_grid).index(eps) + 1


def worst_over_restarts(losses: np.ndarray) -> np.ndarray:
    """Index of the restart with the highest loss, per row; losses is (restarts, n)."""
    return np.argmax(losses, axis=0)


def strongest(cells: dict[str, dict]) -> str:
    """The attack with the lowest seed-mean attacked F1 for one arm."""
    return min(ATTACKS, key=lambda a: cells[a]["adv_mean"])


def verdict(retention: dict[str, dict[str, float]], strong: dict[str, str]) -> dict:
    a2 = retention["A2_zz"][strong["A2_zz"]]
    a1 = retention["A1_angle"][strong["A1_angle"]]
    gap = a2 - a1
    if a2 > HOLD_RETENTION and gap >= HOLD_GAP:
        branch = "HOLDS"
    elif abs(gap) <= COLLAPSE_BAND:
        branch = "COLLAPSES"
    else:
        branch = "INTERMEDIATE"
    return {
        "a2_retention_strongest": a2,
        "a1_retention_strongest": a1,
        "gap_a2_minus_a1": gap,
        "strongest_attack": strong,
        "branch": branch,
        "rule": f"HOLDS: A2 > {HOLD_RETENTION} and gap >= {HOLD_GAP}; "
        f"COLLAPSES: |gap| <= {COLLAPSE_BAND}",
    }


# --------------------------------------------------------------------------- worker


def control_unit(spec: dict) -> dict:
    from experiments.run_adversarial import _f1
    from experiments.run_square_blackbox import square_attack
    from experiments.run_xgb_thread_sensitivity import _peak_rss_gb
    from experiments.runner_guard import assert_clean_predictor
    from qgridbench.attacks.evasion import pgd
    from qgridbench.utils.seeding import set_all_seeds, spawn_rng

    c, arm, seed = ax._CTX, spec["arm"], spec["seed"]
    t0 = time.perf_counter()
    sc, ang = c["scale"], c["reg"].angle_scaler
    pgd_cfg = c["cfg_adv"]["evasion"]["pgd"]
    eps_grid = c["cfg_adv"]["evasion"]["epsilons"]
    sm = spec["smoke"]

    set_all_seeds(seed)  # final_unit's order: the MLP fit leaves global RNG state untouched
    m = ax.make_vqc(arm, spec["params"], seed, ax.FINAL_EPOCHS, ax.FINAL_PATIENCE)
    m.fit(c["A_tr"], c["ytr"], c["A_va"], c["y_va"])

    def proba(X):
        return m.predict_proba(ang.transform(X))

    def grad(X, y):
        return m.loss_input_grad(ang.transform(X), y) * ang._mm.scale_

    X, y = c["X500"], c["y500"]
    p0 = proba(X)
    assert_clean_predictor(f"{arm} seed {seed}", y, p0, p0.argmax(1))
    clean = _f1(y, p0)

    # pgd20 on the main run's stream: skip the draws made before the arm's 0.5-sigma set
    rng = spawn_rng(seed, "adv")
    for _ in range(stream_draws_before(eps_grid, EPS)):
        rng.uniform(-1.0, 1.0, size=X.shape)
    X20 = pgd(
        grad, X, y, EPS, sc, n_steps=pgd_cfg["n_steps"], step_frac=pgd_cfg["step_frac"], rng=rng
    )

    steps = SMOKE["n_steps"] if sm else PGD100["n_steps"]
    frac = SMOKE["step_frac"] if sm else PGD100["step_frac"]
    restarts = SMOKE["restarts"] if sm else PGD100["restarts"]
    sets, losses = [], []
    for r in range(restarts):
        Xr = pgd(
            grad,
            X,
            y,
            EPS,
            sc,
            n_steps=steps,
            step_frac=frac,
            rng=spawn_rng(seed, f"r4_13_pgd100_{arm}_r{r}"),
        )
        p = np.clip(proba(Xr)[np.arange(len(y)), y], 1e-12, 1.0)
        sets.append(Xr)
        losses.append(-np.log(p))
    pick = worst_over_restarts(np.array(losses))
    X100 = np.array(sets)[pick, np.arange(len(y))]

    n_sq = SMOKE["square_iter"] if sm else SQUARE_ITER
    Xsq, queries = square_attack(proba, X, y, EPS, sc, spawn_rng(seed, f"r4_13_square_{arm}"), n_sq)

    adv = {"pgd20": X20, "pgd100x5": X100, "square": Xsq}
    probs = {k: proba(v) for k, v in adv.items()}
    f1 = {k: _f1(y, p) for k, p in probs.items()}
    # per-row worst case over the three attacks: the set with the lowest true-class probability
    ptrue = np.array([probs[k][np.arange(len(y)), y] for k in ATTACKS])
    worst = np.array([adv[k] for k in ATTACKS])[np.argmin(ptrue, axis=0), np.arange(len(y))]
    return {
        "arm": arm,
        "seed": seed,
        "params": spec["params"],
        "clean": clean,
        "adv": f1,
        "worst_of_three": _f1(y, proba(worst)),
        "restart_pick_counts": np.bincount(pick, minlength=restarts).tolist(),
        "square_queries_per_row": queries,
        "pgd_above_clean": [k for k in ("pgd20", "pgd100x5") if f1[k] > clean + 1e-9],
        "square_above_clean": bool(f1["square"] > clean + 1e-9),
        "unit_wall_s": time.perf_counter() - t0,
        "peak_rss_gb": _peak_rss_gb(),
    }


# --------------------------------------------------------------------------- assembly


def anchors(units: list[dict], main_final: dict) -> dict:
    rows = []
    for u in units:
        ref = main_final[f"{u['arm']}_s{u['seed']}"]
        for cell, got, want in (
            ("clean500", u["clean"], ref["clean"]["eval500"]),
            ("pgd20", u["adv"]["pgd20"], ref["r2"]["pgd"]["0.5"]),
        ):
            rows.append(
                {
                    "arm": u["arm"],
                    "seed": u["seed"],
                    "cell": cell,
                    "got": got,
                    "archived": want,
                    "pass": abs(got - want) <= ANCHOR_TOL,
                }
            )
    return {"rows": rows, "pass": bool(rows) and all(r["pass"] for r in rows)}


def summarize(units: list[dict]) -> dict:
    out = {"per_arm": {}, "retention": {}, "strongest": {}}
    for arm in ARMS:
        us = sorted((u for u in units if u["arm"] == arm), key=lambda u: u["seed"])
        clean = float(np.mean([u["clean"] for u in us]))
        cells = {}
        for a in (*ATTACKS, "worst_of_three"):
            vals = [u["adv"][a] if a in ATTACKS else u["worst_of_three"] for u in us]
            cells[a] = {
                "adv_mean": float(np.mean(vals)),
                "adv_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "retention": float(np.mean(vals)) / clean,
            }
        out["per_arm"][arm] = {"n_seeds": len(us), "clean_mean": clean, "attacks": cells}
        out["retention"][arm] = {a: cells[a]["retention"] for a in cells}
        out["strongest"][arm] = strongest(cells)
    out["verdict"] = verdict(out["retention"], out["strongest"])
    return out


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
        args.workers, args.worker_rss_gb, args.budget_gb - _peak_rss_gb(), "R4-13 control workers"
    )
    main_res = json.loads((args.main_run / "results.json").read_text())
    main_final = main_res["final"]
    seeds = ax.SEEDS[:1] if args.smoke else ax.SEEDS
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    cfg = {
        "smoke": args.smoke,
        "workers": args.workers,
        "threads": args.threads,
        "arms": ARMS,
        "eps": EPS,
        "pgd100": SMOKE if args.smoke else PGD100,
        "square_iter": SMOKE["square_iter"] if args.smoke else SQUARE_ITER,
        "main_run": str(args.main_run),
        "prereg": "decision log 2026-11-16T14:58Z (R4-13 control)",
        "memory": mem,
    }
    name = "vqc_encoding_attack_control_binary_q8" + ("_smoke" if args.smoke else "")
    with RunDir(name, config=cfg, seeds=list(seeds)) as run:
        (run.path / "git_status_at_launch.txt").write_text(porcelain)
        done: dict[str, dict] = {}
        if args.resume:
            done = json.loads((args.resume / "results_partial.json").read_text())["units"]

        def save(u):
            done[f"{u['arm']}_s{u['seed']}"] = u
            run.write_json("results_partial.json", {"units": done})
            log.info(
                "%s seed %d | clean %.4f | %s | worst3 %.4f | %.0fs %.2fGB",
                u["arm"],
                u["seed"],
                u["clean"],
                {k: round(v, 4) for k, v in u["adv"].items()},
                u["worst_of_three"],
                u["unit_wall_s"],
                u["peak_rss_gb"],
            )

        specs = [
            {"arm": a, "seed": s, "params": main_final[f"{a}_s{s}"]["params"], "smoke": args.smoke}
            for s in seeds
            for a in ARMS
            if f"{a}_s{s}" not in done
        ]
        ax._pool(control_unit, specs, args.workers, args.threads, args.chunk, save)

        units = list(done.values())
        anc = anchors(units, main_final)
        neg_ok = all(not u["pgd_above_clean"] for u in units)
        out = {"anchors": anc, "negative_control_pass": neg_ok, "units": done}
        if not anc["pass"] or not neg_ok:
            run.write_json("results.json", out)
            raise AssertionError(f"anchor or negative control failed: {anc['rows']}")
        out["summary"] = summarize(units)
        peak = max(u["peak_rss_gb"] for u in units)
        out["peak_rss_gb"] = {"worker_max": peak, "main": _peak_rss_gb()}
        run.write_json("results.json", out)
        run.metrics.update(
            {
                "anchors_pass": True,
                "worker_peak_rss_gb": peak,
                **(
                    {"smoke": True}
                    if args.smoke
                    else {"branch": out["summary"]["verdict"]["branch"]}
                ),
            }
        )
        log.info("anchors ok | %s", out["summary"]["verdict"])
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
