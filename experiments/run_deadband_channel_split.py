"""R4-02 (NS9): does unconstrained MLP-surrogate transfer to the trees ride on the derived channels?

Recreates Stage 1's A1 MLP-surrogate PGD sets (`run_channel_restricted_attack`, artifact
`20261116T052503Z_channel_restricted_attack_binary_q8`: 112 physical channels, standardized raw
space u, eps in {0.1, 0.25, 0.5}, seeds {0, 1, 2}, 500 eval rows, 20 steps) by replaying Stage
1's per-seed stream `spawn_rng(seed, "m1a")`: one (n, 128) uniform draw per pgd call, in Stage
1's order (arms A0, A1, A2-R1..R4; eps; sources mlp, qsvm, rbf_svm, vqc). Each perturbation
delta = U - U0 is split into
    controlled: delta on the 56 channels E-M1c controls (per relay PM1..PM6, PA1..PA6, F, DF;
                `relay_columns`, the source of RelayMap.ctrl_names), zero elsewhere;
    derived:    delta on the other 56 A1 channels (per relay PM7..PM12, PA7..PA12, PA:Z,
                PA:ZH, asserted equal to RelayMap.der_names), zero elsewhere;
and the full delta and each part alone go through Stage 1's map Z = Z0 + delta @ W.T to every
family. The 1e-12-scaled residue Stage 1's pgd leaves on blocked channels stays in the full
delta (so the anchor is exact) and is in neither part.

Random controls (kind "random"): the full delta's is Stage 1's own A1 random control, replayed
from `spawn_rng(seed, "m1a_random")` (one (n, 128) sign draw per (arm, eps)); each part's is
random +-eps signs on the part's channels in the same unit (u, standardized), from
`spawn_rng(seed, "ns9_random_<part>")`, one (n, 128) sign draw per eps in Stage 1's eps order.

Statistic (decision log 2026-11-16T11:07Z, R4-02): E(part) = seed mean of (random F1 - transfer
F1) on the part's channels; share(part) = E(part) / E(A1, full), per tree at eps 0.5.
HYPOTHESIS: share(derived) >= 0.5 and share(controlled) <= 0.3 for >= 2 of 3 trees; KILL:
share(controlled) >= 0.5 for >= 2 of 3 trees; INTERMEDIATE otherwise. Every family and every
eps is reported; the branch reads the trees at eps 0.5.

Anchor (blocking, raised after results.json is written): every full-delta cell (transfer_mlp
and random) reproduces Stage 1's A1 clean F1, attacked F1, radius_mean and radius_max EXACTLY,
at Stage 1's thread count (T = 10, XGBoost's fit depends on it; the runner refuses another T).
Other controls (each raises): Stage 1's eps ball and blocked-channel bound on the replayed
delta; the parts are disjoint, cover A1, and sum to the full delta on A1 exactly.

Extra readout, no branch: per class (`run_evasion_per_class.per_class`): evasion (clean minus
attacked attack recall), false-alarm rise, and the evasion share of the misclassified-row change
(evasion rows / (evasion rows + false-alarm rows), reported only when both are >= 0); and per
set, the rows where delta moves a deadband-zero sequence channel (clean raw PMk == 0, k 7..12)
off zero (|delta| > 1e-9 sigma on PMk or PAk).

Units are seeds (each fits its own zoo; its streams depend on the seed only). Thread budget is
W x T; --threads is read before numpy loads, in every worker.
Run (smoke = seed 0 at full size, which is the seed-0 anchor; then the full run):
    uv run python -m experiments.run_deadband_channel_split --smoke --threads 10
    uv run python -m experiments.run_deadband_channel_split --workers 1 --threads 10 \
        --worker-rss-gb <smoke peak>
"""

from __future__ import annotations

import os
import sys


def _early_threads(argv: list[str]) -> str | None:
    """--threads, read BEFORE numpy/torch load so the BLAS/OpenMP pools obey it."""
    for i, a in enumerate(argv):
        if a == "--threads":
            return argv[i + 1]
        if a.startswith("--threads="):
            return a.split("=", 1)[1]
    return None


_THREADS = _early_threads(sys.argv)
if _THREADS is None and not {"-h", "--help"} & set(sys.argv):
    raise SystemExit("--threads is required (it is applied before numpy is imported)")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
for _var in THREAD_VARS:
    os.environ[_var] = _THREADS or "1"

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

import experiments.run_consistent_relay_attack as crr  # noqa: E402
from experiments.run_adversarial import FittedZoo, _f1  # noqa: E402
from experiments.run_channel_restricted_attack import (  # noqa: E402
    BLOCK_SCALE,
    ENCODING,
    EPS_GRID,
    FAMILIES,
    QUBITS,
    RELAYS,
    SEEDS,
    VARIANT,
    _ms,
    build_context,
    u_space_grad,
)
from experiments.run_consistent_relay_attack import (  # noqa: E402
    STAGE1_ARMS,
    STAGE1_RESULTS,
    STAGE1_SOURCES,
    TREES,
    Shared,
    _peak_rss_gb,
    memory_guard,
    relay_columns,
    run_pool_all,
)
from experiments.run_evasion_per_class import per_class  # noqa: E402
from qgridbench.attacks.evasion import pgd  # noqa: E402
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.features.reduce import build_preprocessor  # noqa: E402
from qgridbench.models.classical.zoo import stratified_cap  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger  # noqa: E402
from qgridbench.utils.seeding import spawn_rng  # noqa: E402

log = get_logger(__name__)

ARM = "A1"
PARTS = ("full", "controlled", "derived")
SPLIT_PARTS = ("controlled", "derived")
KINDS = ("transfer_mlp", "random")
EPS_TOP = 0.5
# pre-registered branch thresholds (decision log 2026-11-16T11:07Z, R4-02)
HYP_DERIVED_MIN, HYP_CONTROLLED_MAX, KILL_CONTROLLED_MIN, MIN_TREES = 0.5, 0.3, 0.5, 2
MOVE_TOL = 1e-9  # sigma units: a delta above this moves a channel
ANCHOR_FIELDS = ("clean_f1", "attacked_f1", "radius_mean", "radius_max")


# --------------------------------------------------------------------------- pure helpers


def part_masks(columns: list[str], a1: np.ndarray, relays=RELAYS) -> tuple[dict, dict]:
    """Boolean (128,) masks of the controlled and derived parts, and the sequence-channel indices.

    Raises unless the parts are disjoint, each 14 x relays, and together exactly A1."""
    ctrl = np.zeros(len(columns), bool)
    der = np.zeros(len(columns), bool)
    seq = {"pm": [], "pa": [], "names": []}
    for r in relays:
        _, der_names, ci, di = relay_columns(columns, r)
        ctrl[ci] = True
        der[di] = True
        seq["pm"].extend(di[:6].tolist())
        seq["pa"].extend(di[6:12].tolist())
        seq["names"].extend(n.split("-")[1].split(":")[0] for n in der_names[:6])
    if (ctrl & der).any():
        raise AssertionError("controlled and derived parts overlap")
    n = 14 * len(relays)
    if ctrl.sum() != n or der.sum() != n:
        raise AssertionError(f"parts have {ctrl.sum()} and {der.sum()} channels, expected {n}")
    if not np.array_equal(ctrl | der, a1):
        raise AssertionError("controlled + derived is not A1")
    seq = {"pm": np.array(seq["pm"]), "pa": np.array(seq["pa"]), "names": seq["names"]}
    return {"controlled": ctrl, "derived": der}, seq


def stage1_draw(seed: int, stream: str, n_before: int, draw):
    """Stage 1's stream after n_before draws of the same kind; returns the generator."""
    rng = spawn_rng(seed, stream)
    for _ in range(n_before):
        draw(rng)
    return rng


def stage1_pgd_skip(e_idx: int, arm: str = ARM, source: str = "mlp") -> int:
    """pgd calls Stage 1 made on spawn_rng(seed, "m1a") before (arm, eps index, source)."""
    return (
        STAGE1_ARMS.index(arm) * len(EPS_GRID) * len(STAGE1_SOURCES)
        + e_idx * len(STAGE1_SOURCES)
        + STAGE1_SOURCES.index(source)
    )


def stage1_random_skip(e_idx: int, arm: str = ARM) -> int:
    """Random-sign draws Stage 1 made on spawn_rng(seed, "m1a_random") before (arm, eps)."""
    return STAGE1_ARMS.index(arm) * len(EPS_GRID) + e_idx


def deadband_moves(x0raw: np.ndarray, d: np.ndarray, seq: dict, tol: float = MOVE_TOL) -> dict:
    """Rows where d moves a deadband-zero sequence channel (clean raw PMk == 0) off zero."""
    zero = x0raw[:, seq["pm"]] == 0
    moved = (np.abs(d[:, seq["pm"]]) > tol) | (np.abs(d[:, seq["pa"]]) > tol)
    off = zero & moved
    by = {}
    for j, name in enumerate(seq["names"]):
        by[name] = by.get(name, 0) + int(off[:, j].sum())
    return {
        "n_rows": int(len(x0raw)),
        "n_rows_with_deadband_zero": int(zero.any(axis=1).sum()),
        "n_rows_moved_off_zero": int(off.any(axis=1).sum()),
        "n_channel_cells_moved_off_zero": int(off.sum()),
        "by_channel": by,
    }


def class_split(clean: dict, att: dict) -> dict:
    """Evasion and false-alarm rise of one attacked set against clean (rates, seed means), and
    the evasion share of the changed rows (both components >= 0, else None)."""
    ev = clean["attack_recall"] - att["attack_recall"]
    fa = att["false_alarm"] - clean["false_alarm"]
    return {"evasion": ev, "false_alarm_rise": fa} | _share(ev, fa, att)


def _share(ev: float, fa: float, counts: dict) -> dict:
    ev_rows, fa_rows = ev * counts["n_attack"], fa * counts["n_natural"]
    ok = ev_rows >= 0 and fa_rows >= 0 and ev_rows + fa_rows > 0
    return {"evasion_share_rows": ev_rows / (ev_rows + fa_rows) if ok else None}


def excess(f1: dict, seeds, fam: str, eps: float, part: str) -> dict:
    """E(part): random F1 minus transfer F1, per seed and as the seed mean."""
    per = [
        f1[(s, eps, fam, part, "random")] - f1[(s, eps, fam, part, "transfer_mlp")] for s in seeds
    ]
    return _ms(per) | {"per_seed": per}


def branch(shares: dict) -> dict:
    """shares[tree] = {"controlled": s_c, "derived": s_d} at eps 0.5 -> the registered verdict."""
    hyp = [
        t
        for t in TREES
        if shares[t]["derived"] >= HYP_DERIVED_MIN and shares[t]["controlled"] <= HYP_CONTROLLED_MAX
    ]
    kill = [t for t in TREES if shares[t]["controlled"] >= KILL_CONTROLLED_MIN]
    if len(hyp) >= MIN_TREES:
        verdict = "HYPOTHESIS"
    elif len(kill) >= MIN_TREES:
        verdict = "KILL"
    else:
        verdict = "INTERMEDIATE"
    return {"verdict": verdict, "trees_hypothesis": hyp, "trees_kill": kill}


# --------------------------------------------------------------------------- worker


def run_unit(spec: dict) -> dict:
    """One seed: fit the zoo, replay Stage 1's A1 MLP PGD per eps, score every set per class."""
    sh = crr._SHARED
    ctx, seed = sh.ctx, spec["seed"]
    t_unit = time.perf_counter()
    zoo = FittedZoo(ctx.reg, ENCODING, QUBITS, seed, ctx.subset_cap, ctx.best_params)
    t_fit = time.perf_counter() - t_unit
    y, Z0, U0, W = ctx.y, ctx.Z0, ctx.U0, ctx.W
    scale_z = ctx.reg.feature_scale
    a1 = ctx.arm_masks[ARM]
    masks, seq = part_masks(ctx.columns, a1)
    masks["full"] = a1
    clean = {}
    for f in FAMILIES:
        proba = zoo.clean_proba(f, Z0, y)
        clean[f] = {"f1": _f1(y, proba), **per_class(y, proba.argmax(1))}
    gu = u_space_grad(zoo.grad_provider("mlp"), ctx)
    scale_u = np.where(a1, 1.0, BLOCK_SCALE)  # Stage 1's per-channel pgd scale for A1
    rng_part = {p: spawn_rng(seed, f"ns9_random_{p}") for p in SPLIT_PARTS}

    def unif(g):
        return g.uniform(-1.0, 1.0, size=U0.shape)

    def signs(g):
        return g.choice([-1.0, 1.0], size=U0.shape)

    records, sets, split_ctl, t_pgd = [], [], [], 0.0
    for e_idx, eps in enumerate(EPS_GRID):
        t0 = time.perf_counter()
        rng = stage1_draw(seed, "m1a", stage1_pgd_skip(e_idx), unif)
        U = pgd(gu, U0, y, eps, scale_u, n_steps=ctx.n_steps, step_frac=ctx.step_frac, rng=rng)
        t_pgd += time.perf_counter() - t0
        d = U - U0
        if np.abs(d[:, ~a1]).max() >= 1e-9:  # Stage 1's control 3
            raise AssertionError(f"seed {seed} eps {eps}: a blocked channel moved")
        if np.abs(d[:, a1]).max() > eps + 1e-9:
            raise AssertionError(f"seed {seed} eps {eps}: eps ball exceeded")
        trans = {"full": d} | {p: d * masks[p] for p in SPLIT_PARTS}
        resid = d - trans["controlled"] - trans["derived"]
        if np.any(resid[:, a1] != 0):
            raise AssertionError(f"seed {seed} eps {eps}: the parts do not sum to delta on A1")
        split_ctl.append(
            {"eps": eps, "max_abs_outside_a1": float(np.abs(resid[:, ~a1]).max()), "on_a1": 0.0}
        )
        r1 = stage1_draw(seed, "m1a_random", stage1_random_skip(e_idx), signs)
        rand = {"full": signs(r1) * eps * a1}
        rand |= {p: signs(rng_part[p]) * eps * masks[p] for p in SPLIT_PARTS}
        for part in PARTS:
            for kind, dd in (("transfer_mlp", trans[part]), ("random", rand[part])):
                dZ = dd @ W.T
                Zadv = Z0 + dZ
                radius = (np.abs(dZ) / scale_z).max(axis=1)
                sets.append(
                    {"eps": eps, "part": part, "kind": kind} | deadband_moves(sh.X0raw, dd, seq)
                )
                for fam in FAMILIES:
                    proba = zoo.predict_proba(fam, Zadv)
                    pc = per_class(y, proba.argmax(1))
                    records.append(
                        {
                            "seed": seed,
                            "eps": eps,
                            "part": part,
                            "kind": kind,
                            "family": fam,
                            "n_channels": int(masks[part].sum()),
                            "clean_f1": clean[fam]["f1"],
                            "attacked_f1": _f1(y, proba),
                            "radius_mean": float(radius.mean()),
                            "radius_max": float(radius.max()),
                            "attack_recall": pc["attack_recall"],
                            "false_alarm": pc["false_alarm"],
                            "n_attack": pc["n_attack"],
                            "n_natural": pc["n_natural"],
                        }
                    )
        log.info("seed %d | eps %.2f done (%.0f s)", seed, eps, time.perf_counter() - t_unit)
    return {
        "unit": f"s{seed}",
        "seed": seed,
        "clean": clean,
        "records": records,
        "deadband_sets": sets,
        "split_control": split_ctl,
        "timing_s": {"zoo_fit": round(t_fit, 1), "mlp_pgd": round(t_pgd, 2)},
        "wall_s": round(time.perf_counter() - t_unit, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "pid": os.getpid(),
    }


# --------------------------------------------------------------------------- parent


def load_stage1(threads: int, ctx) -> tuple[dict, dict]:
    """Stage 1's config (checked against this run) and its A1 transfer_mlp / random records."""
    res = json.loads((STAGE1_RESULTS / "results.json").read_text())
    cfg = res["config"]
    want = {
        "threads": str(threads),
        "arms": STAGE1_ARMS,
        "eps_grid": EPS_GRID,
        "n_eval": int(len(ctx.y)),
        "n_steps": ctx.n_steps,
        "step_frac": ctx.step_frac,
        "block_scale": BLOCK_SCALE,
    }
    bad = {k: (cfg[k], v) for k, v in want.items() if cfg[k] != v}
    if bad:
        raise SystemExit(f"Stage 1 config differs from this replay (Stage 1, here): {bad}")
    recs = {
        (x["seed"], x["eps"], x["family"], x["kind"]): x
        for p in res["per_seed"]
        for x in p["records"]
        if x["arm"] == ARM and x["kind"] in KINDS
    }
    return cfg, recs


def anchor(records: list[dict], stage1: dict) -> dict:
    """Full-delta cells against Stage 1's A1 cells: every field must be equal."""
    full = [x for x in records if x["part"] == "full"]
    diffs, missing = [], []
    for x in full:
        ref = stage1.get((x["seed"], x["eps"], x["family"], x["kind"]))
        if ref is None:
            missing.append((x["seed"], x["eps"], x["family"], x["kind"]))
            continue
        for k in ANCHOR_FIELDS:
            if x[k] != ref[k]:
                diffs.append(
                    {
                        "key": [x["seed"], x["eps"], x["family"], x["kind"], k],
                        "here": x[k],
                        "stage1": ref[k],
                        "diff": x[k] - ref[k],
                    }
                )
    return {
        "n_cells": len(full),
        "fields": list(ANCHOR_FIELDS),
        "n_differ": len(diffs),
        "differ": diffs,
        "missing": missing,
        "tolerance": "exact (==)",
        "passed": not diffs and not missing and bool(full),
    }


def _mean(xs, key):
    return float(np.mean([x[key] for x in xs]))


def summarize(units: list[dict]) -> dict:
    units = sorted(units, key=lambda u: u["seed"])
    seeds = [u["seed"] for u in units]
    records = [x for u in units for x in u["records"]]
    f1 = {
        (x["seed"], x["eps"], x["family"], x["part"], x["kind"]): x["attacked_f1"] for x in records
    }
    counts = {(x["n_attack"], x["n_natural"]) for x in records}
    if len(counts) != 1:
        raise AssertionError(f"class counts differ across records: {counts}")
    n_att, n_nat = counts.pop()
    counts = {"n_attack": n_att, "n_natural": n_nat}
    cells, E, shares = [], {}, {}
    for eps in EPS_GRID:
        for fam in FAMILIES:
            cl = {
                "attack_recall": float(np.mean([u["clean"][fam]["attack_recall"] for u in units])),
                "false_alarm": float(np.mean([u["clean"][fam]["false_alarm"] for u in units])),
            }
            for part in PARTS:
                E[(fam, eps, part)] = excess(f1, seeds, fam, eps, part)
                by_kind = {}
                for kind in KINDS:
                    xs = [
                        x
                        for x in records
                        if (x["eps"], x["family"], x["part"], x["kind"]) == (eps, fam, part, kind)
                    ]
                    att = {
                        "attack_recall": _mean(xs, "attack_recall"),
                        "false_alarm": _mean(xs, "false_alarm"),
                    } | counts
                    by_kind[kind] = {
                        "attacked_f1": _ms([x["attacked_f1"] for x in xs]),
                        "clean_f1": _ms([x["clean_f1"] for x in xs]),
                        "attack_recall": att["attack_recall"],
                        "false_alarm": att["false_alarm"],
                    } | class_split(cl, att)
                t, r = by_kind["transfer_mlp"], by_kind["random"]
                ev_x = t["evasion"] - r["evasion"]
                fa_x = t["false_alarm_rise"] - r["false_alarm_rise"]
                cells.append(
                    {
                        "eps": eps,
                        "family": fam,
                        "part": part,
                        "E": E[(fam, eps, part)],
                        "transfer_mlp": t,
                        "random": r,
                        "excess_evasion": ev_x,
                        "excess_false_alarm_rise": fa_x,
                        "excess_evasion_share_rows": _share(ev_x, fa_x, counts)[
                            "evasion_share_rows"
                        ],
                    }
                )
            e_full = E[(fam, eps, "full")]["mean"]
            shares[(fam, eps)] = {
                p: (E[(fam, eps, p)]["mean"] / e_full if e_full > 0 else None) for p in SPLIT_PARTS
            } | {
                "E_full": e_full,
                "per_seed": {
                    p: [
                        a / b if b > 0 else None
                        for a, b in zip(
                            E[(fam, eps, p)]["per_seed"],
                            E[(fam, eps, "full")]["per_seed"],
                            strict=True,
                        )
                    ]
                    for p in SPLIT_PARTS
                },
            }
    top = {t: shares[(t, EPS_TOP)] for t in TREES}
    undefined = [t for t, s in top.items() if s["E_full"] <= 0]
    if undefined:
        raise ValueError(f"E(A1, full) <= 0 at eps {EPS_TOP} for {undefined}: share undefined")
    return {
        "seeds": seeds,
        "statistic": "share(part) = E(part) / E(A1, full); E = seed mean of random F1 - "
        "transfer_mlp F1 on the part's channels",
        "branch": branch(top)
        | {"eps": EPS_TOP, "shares": {t: {p: top[t][p] for p in SPLIT_PARTS} for t in TREES}},
        "shares": {f"{f}|{e}": v for (f, e), v in shares.items()},
        "cells": cells,
    }


def deadband_summary(units: list[dict]) -> list[dict]:
    """Per (eps, part, kind): the deadband-zero counts summed over seeds."""
    out = {}
    for u in units:
        for s in u["deadband_sets"]:
            k = (s["eps"], s["part"], s["kind"])
            acc = out.setdefault(
                k,
                {
                    "eps": s["eps"],
                    "part": s["part"],
                    "kind": s["kind"],
                    "n_row_seeds": 0,
                    "n_row_seeds_with_deadband_zero": 0,
                    "n_row_seeds_moved_off_zero": 0,
                    "by_channel": {},
                },
            )
            acc["n_row_seeds"] += s["n_rows"]
            acc["n_row_seeds_with_deadband_zero"] += s["n_rows_with_deadband_zero"]
            acc["n_row_seeds_moved_off_zero"] += s["n_rows_moved_off_zero"]
            for c, n in s["by_channel"].items():
                acc["by_channel"][c] = acc["by_channel"].get(c, 0) + n
    return list(out.values())


def build_shared(threads: int) -> tuple[Shared, dict]:
    """Stage 1's context plus the raw eval rows (for the deadband readout), checked equal."""
    ctx = build_context(SimpleNamespace(smoke=False))
    Xraw, _, splits = load_processed(VARIANT)
    pipe = build_preprocessor("pca", n_components=QUBITS, seed=0).fit(Xraw[splits["train"]])
    if not np.array_equal(pipe.named_steps["pca"].components_.astype(np.float64), ctx.W):
        raise AssertionError("refitted pipeline differs from Stage 1's context")
    ev = stratified_cap(ctx.reg.X["test"], ctx.reg.y["test"], len(ctx.y), seed=0)
    X0raw = Xraw[splits["test"]][ev].astype(np.float64)
    X0imp = pipe.named_steps["impute"].transform(X0raw)
    if not np.array_equal(pipe.named_steps["scale"].transform(X0imp), ctx.U0):
        raise AssertionError("eval rows differ from Stage 1's context")
    masks, seq = part_masks(ctx.columns, ctx.arm_masks[ARM])
    sh = Shared(
        ctx=ctx,
        X0raw=X0raw,
        X0imp=X0imp,
        s=pipe.named_steps["scale"].scale_.astype(np.float64),
        maps={},
        stage1=None,
        threads=threads,
        angle_unit="marginal",
    )
    info = {
        "controlled_channels": [
            c for c, m in zip(ctx.columns, masks["controlled"], strict=True) if m
        ],
        "derived_channels": [c for c, m in zip(ctx.columns, masks["derived"], strict=True) if m],
        "n_eval_rows_with_deadband_zero": int((X0raw[:, seq["pm"]] == 0).any(axis=1).sum()),
    }
    return sh, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, required=True, help="BLAS/OpenMP/torch threads/worker")
    ap.add_argument("--workers", type=int, default=1, help="parallel seed units")
    ap.add_argument("--smoke", action="store_true", help="seed 0 at full size (the seed-0 anchor)")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--worker-rss-gb", type=float, help="per-worker peak RSS measured by smoke")
    ap.add_argument("--resume", type=Path, help="results_partial.json of an interrupted run")
    args = ap.parse_args()
    if str(args.threads) != _THREADS:
        raise AssertionError(f"early thread parse {_THREADS} != argparse {args.threads}")
    T = args.threads
    W = 1 if args.smoke else args.workers
    n_cpu = psutil.cpu_count(logical=True)
    print(
        f"workers W={W} x threads T={T} = {W * T} of {n_cpu} logical; "
        f"{psutil.virtual_memory().available / 2**30:.1f} GB available",
        flush=True,
    )
    if W * T > n_cpu:
        raise ValueError(f"W x T = {W * T} exceeds the {n_cpu} logical CPUs")
    sh, info = build_shared(T)
    s1_cfg, stage1 = load_stage1(T, sh.ctx)
    seeds = [0] if args.smoke else args.seeds
    if not set(seeds) <= set(s1_cfg["seeds"]):
        raise SystemExit(f"seeds {seeds} are not all Stage 1's {s1_cfg['seeds']}")
    name = f"deadband_channel_split_{VARIANT}_q{QUBITS}" + ("_smoke" if args.smoke else "")
    cfg = {
        "experiment": "R4-02 (NS9)",
        "threads": T,
        "workers": W,
        "smoke": args.smoke,
        "seeds": seeds,
        "eps_grid": EPS_GRID,
        "eps_branch": EPS_TOP,
        "n_eval": int(len(sh.ctx.y)),
        "n_steps": sh.ctx.n_steps,
        "step_frac": sh.ctx.step_frac,
        "stage1_results": str(STAGE1_RESULTS.relative_to(REPO_ROOT)),
        "streams": {
            "pgd": 'spawn_rng(seed, "m1a"), Stage 1 replay',
            "random_full": 'spawn_rng(seed, "m1a_random"), Stage 1 replay',
            "random_part": 'spawn_rng(seed, "ns9_random_<part>"), one draw per eps',
        },
        "thresholds": {
            "hyp_derived_min": HYP_DERIVED_MIN,
            "hyp_controlled_max": HYP_CONTROLLED_MAX,
            "kill_controlled_min": KILL_CONTROLLED_MIN,
            "min_trees": MIN_TREES,
        },
        "resume": str(args.resume) if args.resume else None,
    }
    with RunDir(name, config=cfg, seeds=seeds) as run:
        extra = {}
        if not args.smoke:
            if args.worker_rss_gb is None:
                raise SystemExit("--worker-rss-gb is required (smoke prints the measured value)")
            extra["memory_guard"] = memory_guard(W, args.worker_rss_gb, "full run")
        done = json.loads(args.resume.read_text())["units"] if args.resume else []
        have = {u["seed"] for u in done}
        todo = [{"seed": s} for s in seeds if s not in have]

        def save(res):
            done.append(res)
            run.write_json("results_partial.json", {"units": done})

        run_pool_all(run_unit, todo, sh, W, on_done=save)
        units = sorted(done, key=lambda u: u["seed"])
        records = [x for u in units for x in u["records"]]
        anc = anchor(records, stage1)
        summary = summarize(units)
        peak = max(u["peak_rss_gb"] for u in units)
        results = {
            "config": cfg,
            "partition": info,
            **extra,
            "anchor": anc,
            "branch": summary["branch"],
            "shares": summary["shares"],
            "cells": summary["cells"],
            "deadband": deadband_summary(units),
            "split_control": {u["unit"]: u["split_control"] for u in units},
            "units_meta": [
                {k: u[k] for k in ("unit", "timing_s", "wall_s", "peak_rss_gb", "pid")}
                for u in units
            ],
            "parent_peak_rss_gb": round(_peak_rss_gb(), 3),
            "records": records,
        }
        run.write_json("results.json", results)
        run.metrics.update(
            {
                "anchor_passed": anc["passed"],
                "verdict": summary["branch"]["verdict"],
                "worker_peak_rss_gb": peak,
            }
        )
        log.info("anchor: %d cells, %d differ", anc["n_cells"], anc["n_differ"])
        log.info("branch: %s", json.dumps(summary["branch"]))
        print(f"measured peak RSS of one worker: {peak:.2f} GB", flush=True)
        print(f"-> {run.path}")
        if not anc["passed"]:
            raise AssertionError(
                f"anchor failed: {anc['n_differ']} differ, {len(anc['missing'])} missing; "
                f"first {anc['differ'][:3]}"
            )


if __name__ == "__main__":
    main()
