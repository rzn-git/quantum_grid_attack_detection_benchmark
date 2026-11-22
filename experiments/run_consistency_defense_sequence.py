"""R4-11b: a relay-local consistency check over the sequence channels only, as a defense.

Scores each row by how far each relay's reported sequence channels (PM7..PM12, PA7..PA12; 48
channels over the 4 relays) sit from what the relay's own phase phasors imply (the Stage 2 map),
with a +/-2% deadband band, and measures how well a 1%-false-alarm threshold separates clean test
rows from the Stage 1 mask-only attacks and the Stage 2 frame rows. PA:Z and PA:ZH are out of the
score: NS4 as registered was closed on a measured bound (decision log 2026-11-16T11:39Z, "FINDING
(R4-11, NS4 as registered"), and R4-11b is the gated amendment logged there.

Run (thread budget W x T; --threads is applied before numpy loads, in every worker):
    python -m experiments.run_consistency_defense_sequence --smoke --threads 2
    python -m experiments.run_consistency_defense_sequence --workers 3 --threads 2 \
        --worker-rss-gb 1.15

Score (primary, "sequence_band"). Per row: max over relays and the 12 sequence channels of
|reported - map| / s_c, s_c E-NS4's amendment-2 scale (1.4826 x MAD of the clean train residual on
rows where the relay reports the channel; imported). Deadband band: where the fitted deadband state
(|S| > t) disagrees with the reported 0/nonzero state and | |S| - t | <= 2% of t, the mismatch is
indeterminate and both the magnitude and its angle score 0. Outside the band a mismatch keeps its
residual. Beside it, E-NS4's amendment-2 14-channel score ("ens4_amended_14ch") is computed and
reported, not gated.

Precondition gates (blocking, clean train and validation only, before the pool starts):
  P1 per relay x sequence channel, train agreement outside the band >= 0.99;
  P2 validation rows with an out-of-band state disagreement anywhere <= 0.1%;
  P3 the clean-validation 1% threshold <= 20 noise units;
  P4 the analytic C2 bound >= 0.995: each channel's 1-train-std offset (both signs) added to its
     real clean validation residual, against the threshold, averaged over channels.
The PA:Z impedance regression (per relay, OLS of log PA:Z on log|V1| and log|I1|, train) is fitted
and its validation residual tail reported; it is not in the score.

Hypothesis, kill and controls are E-NS4's (decision log 2026-11-16T07:22Z and amendments), on this
score: kill if AUROC < 0.9 or detection < 0.9 for any Stage 1 white-box arm at eps 0.5;
hypothesis if AUROC >= 0.99 and detection >= 0.9 for every arm at every eps; else intermediate.
C1 Stage 2 control-1 medians reproduced; C2 positive control, a 1-train-std random-sign offset on
one random finite sequence channel per clean test row caught on >= 99% of rows (curve 0.25-2 std);
C3 Stage 2 frame rows flagged at most the clean test false-alarm rate + 0.01; C4 anchors (full
run) to Stage 1's and Stage 2 frame's recorded attacked macro-F1 within 0.01; C5 Stage 2 controls 3
and 4 on every regenerated set; C6 (smoke) 1 and 2 workers give identical scores. Added negative
control (reported): per arm and eps, white-box detection against random-sign detection, and the
AUROC of white-box scores against random-sign scores.
"""

from __future__ import annotations

# imported first: the Stage 2 runner applies --threads to the BLAS/OpenMP/torch pools before
# numpy loads, in the parent and in every spawned worker
from experiments import run_consistent_relay_attack as s2

_THREADS = s2._THREADS

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from experiments import run_consistency_defense as cd  # noqa: E402
from experiments.run_channel_restricted_attack import (  # noqa: E402
    EPS_GRID,
    QUBITS,
    RELAYS,
    SEEDS,
    VARIANT,
)
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger  # noqa: E402
from qgridbench.utils.seeding import spawn_rng  # noqa: E402

log = get_logger(__name__)

N_SEQ = cd.N_SEQ  # 6 sequence channels; magnitudes then angles = 12 scored channels per relay
N_SCORED = 2 * N_SEQ
BAND = 0.02  # half-width of the indeterminate deadband band, fraction of the threshold
GROSS = 0.5  # a state mismatch farther than this fraction of t from it is "gross" (reported)
SCORES = ("sequence_band", "ens4_amended_14ch")  # the first is primary
P1_MIN, P2_MAX, P3_MAX, P4_MIN = 0.99, 0.001, 20.0, 0.995
FA_TARGET, C2_OFFSET, POS_CONTROL_MIN = cd.FA_TARGET, cd.C2_OFFSET, cd.POS_CONTROL_MIN


# --------------------------------------------------------------------------- score


def band_state(X: np.ndarray, rm: s2.RelayMap) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n, 12) per scored channel (magnitude, then its angle): state mismatch, inside the band,
    gross mismatch. The state is the map's (|S| > t) against the reported value != 0."""
    xc = X[:, rm.ctrl_idx]
    sv, si = s2.sequences_np(xc)
    dis = np.zeros((len(X), N_SCORED), bool)
    band, gross = dis.copy(), dis.copy()
    for j, k in enumerate(s2.SEQ_CHANNELS):
        S = np.abs((sv if k <= 9 else si)[rm.assign[k]])
        t = rm.deadband[k]
        d = (S > t) != (X[:, rm.der_idx[j]] != 0)
        dist = np.abs(S - t)
        for col in (j, N_SEQ + j):
            dis[:, col] = d
            band[:, col] = dist <= BAND * t
            gross[:, col] = d & (dist > GROSS * t)
    return dis, band, gross


def seq_residuals(X: np.ndarray, maps, s_c, nan_rule: bool = True) -> np.ndarray:
    """(n, n_relays, 12) |reported - map| / s_c on the sequence channels, band mismatches 0."""
    out = []
    for r in RELAYS:
        res = cd.relay_residual(X, maps[r], s_c[r], nan_rule)[:, :N_SCORED]
        dis, band, _ = band_state(X, maps[r])
        out.append(np.where(dis & band, 0.0, res))
    return np.stack(out, axis=1)


def all_scores(X: np.ndarray, maps, s_c) -> dict:
    """Row scores of both scores on raw rows."""
    return {
        SCORES[0]: cd.row_score(seq_residuals(X, maps, s_c)),
        SCORES[1]: cd.row_score(cd.residuals(X, maps, s_c)),
    }


def threshold(s: np.ndarray) -> float:
    return float(np.quantile(s, 1.0 - FA_TARGET, method="higher"))


def seq_names(maps) -> list[str]:
    return [n for r in RELAYS for n in maps[r].der_names[:N_SCORED]]


# --------------------------------------------------------------------------- preconditions


def impedance_regression(Xtr: np.ndarray, Xva: np.ndarray, rm: s2.RelayMap) -> dict:
    """Per-relay OLS of log PA:Z on [1, log|V1|, log|I1|] (train); the validation residual tail in
    units of the train residual's 1.4826 x MAD, beside the map's constant-ratio residual."""

    def rows(X):
        xc = X[:, rm.ctrl_idx]
        fin = np.isfinite(xc).all(axis=1)
        sv, si = s2.sequences_np(np.where(fin[:, None], xc, 0.0))
        v1, i1 = np.abs(sv["pos"]), np.abs(si["pos"])
        z = X[:, rm.der_idx[cd.Z_COL]]
        ok = fin & np.isfinite(z) & (z > 0) & (i1 > 0) & (v1 > 0)
        F = np.column_stack([np.ones(ok.sum()), np.log(v1[ok]), np.log(i1[ok])])
        return F, np.log(z[ok])

    def mad(x):
        return 1.4826 * float(np.median(np.abs(x - np.median(x))))

    Ftr, ytr = rows(Xtr)
    beta, *_ = np.linalg.lstsq(Ftr, ytr, rcond=None)
    Fva, yva = rows(Xva)
    out = {"beta_1_logV1_logI1": beta.tolist(), "n_train": len(ytr), "n_val": len(yva)}
    for name, rtr, rva in (
        ("M1_regression", ytr - Ftr @ beta, yva - Fva @ beta),
        (
            "M0_constant_ratio",
            ytr - (rm.log_ratio + Ftr[:, 1] - Ftr[:, 2]),
            yva - (rm.log_ratio + Fva[:, 1] - Fva[:, 2]),
        ),
    ):
        s = mad(rtr)
        out[name] = {
            "train_mad_log": s,
            "val_p99_abs_over_train_mad": float(
                np.percentile(np.abs(rva - np.median(rtr)), 99) / s
            ),
            "val_r2": float(1 - rva.var() / yva.var()),
        }
    return out


def c2_bound(Xva, maps, s_c, rec, res_va, thr) -> tuple[float, dict]:
    """P4: per channel, the 1-train-std offset (both signs) added to its real clean validation
    residual is caught if it or the row's other channels exceed thr; mean over channels."""
    per = {}
    for i, r in enumerate(RELAYS):
        signed, _ = cd.relay_signed(Xva, maps[r], nan_rule=True)
        dis, band, _ = band_state(Xva, maps[r])
        for j, nm in enumerate(maps[r].der_names[:N_SCORED]):
            off = C2_OFFSET * rec[nm]["train_std"] / rec[nm]["s_c"]
            base = np.where(dis[:, j] & band[:, j], 0.0, signed[:, j] / s_c[r][j])
            other = res_va.copy()
            other[:, i, j] = 0.0
            o = cd.row_score(other)
            fin = np.isfinite(base)
            hit = [((np.abs(base + sg * off) > thr) | (o > thr))[fin] for sg in (-1.0, 1.0)]
            per[nm] = {
                "offset_noise_units": float(off),
                "caught": float(np.concatenate(hit).mean()),
            }
    return float(np.mean([v["caught"] for v in per.values()])), per


def preconditions(Xtr, Xva, maps, s_c, rec) -> dict:
    agree, union_va = {}, np.zeros(len(Xva), bool)
    for r in RELAYS:
        dis, band, _ = band_state(Xtr, maps[r])
        for j, k in enumerate(s2.SEQ_CHANNELS):
            agree[f"R{r}-PM{k}"] = float(1.0 - np.mean(dis[:, j] & ~band[:, j]))
        dv, bv, _ = band_state(Xva, maps[r])
        union_va |= (dv & ~bv).any(axis=1)
    res_va = seq_residuals(Xva, maps, s_c)
    thr = threshold(cd.row_score(res_va))
    bound, per = c2_bound(Xva, maps, s_c, rec, res_va, thr)
    gates = {
        "P1_train_agreement_outside_band_min": (min(agree.values()), P1_MIN, ">="),
        "P2_val_union_out_of_band_rate": (float(union_va.mean()), P2_MAX, "<="),
        "P3_val_threshold_noise_units": (thr, P3_MAX, "<="),
        "P4_analytic_c2_bound": (bound, P4_MIN, ">="),
    }
    out = {
        k: {"value": v, "gate": g, "pass": bool(v >= g if op == ">=" else v <= g)}
        for k, (v, g, op) in gates.items()
    }
    out["P1_per_relay_channel"] = agree
    out["P4_worst_channels"] = dict(sorted(per.items(), key=lambda kv: kv[1]["caught"])[:6])
    out["impedance_regression"] = {f"R{r}": impedance_regression(Xtr, Xva, maps[r]) for r in RELAYS}
    out["all_pass"] = all(out[k]["pass"] for k in gates)
    return out


# --------------------------------------------------------------------------- controls


def positive_control(Xte, maps, s_c, train_std, thrs) -> dict:
    """C2 and its curve: a random-sign offset of k train std on one random finite sequence channel
    per clean test row (the same rows, channels and signs at every k), on both scores."""
    rng = spawn_rng(0, "ns4b_positive_control")
    seq = np.concatenate([maps[r].der_idx[:N_SCORED] for r in RELAYS])
    sig = np.concatenate([train_std[r][:N_SCORED] for r in RELAYS])
    n = len(Xte)
    pick = np.where(np.isfinite(Xte[:, seq]), rng.random((n, len(seq))), -1.0).argmax(axis=1)
    sign = rng.choice([-1.0, 1.0], size=n)
    names = seq_names(maps)
    out = {k: {"curve": {}, "threshold": thrs[k]} for k in SCORES}
    for k in sorted(set(cd.POS_CURVE_OFFSETS) | {C2_OFFSET}):
        X = Xte.copy()
        X[np.arange(n), seq[pick]] += k * sign * sig[pick]
        sc = all_scores(X, maps, s_c)
        for name in SCORES:
            det = sc[name] > thrs[name]
            out[name]["curve"][str(k)] = float(det.mean())
            if k == C2_OFFSET:
                out[name]["by_channel_type_at_c2_offset"] = {
                    names[j].split("-", 1)[1]: float(det[pick % N_SCORED == j].mean())
                    for j in range(N_SCORED)
                }
    for name in SCORES:
        rate = out[name]["curve"][str(C2_OFFSET)]
        out[name] |= {"c2_detection": rate, "pass": bool(rate >= POS_CONTROL_MIN)}
    return {"offset_unit": "train std of the channel", "gated_score": SCORES[0], **out}


def random_contrast(cells, name) -> list[dict]:
    """Added negative control: per (Stage 1 arm, eps, seed), white-box detection against
    random-sign detection, and the AUROC of white-box scores against random-sign scores."""
    by = {}
    for c in cells:
        if c["kind"] == "stage1":
            by.setdefault((c["label"], c["eps"], c["seed"]), {})[c["source"]] = c
    out = []
    for (label, eps, seed), g in sorted(by.items()):
        rnd = np.array(g["random"]["scores"][name])
        thr = g["random"]["threshold"][name]
        for src in cd.SOURCES:
            wb = np.array(g[src]["scores"][name])
            out.append(
                {
                    "label": label,
                    "eps": eps,
                    "seed": seed,
                    "source": src,
                    "detection_whitebox": float(np.mean(wb > thr)),
                    "detection_random": float(np.mean(rnd > thr)),
                    "auroc_whitebox_vs_random": cd.auroc(wb, rnd),
                }
            )
    return out


# --------------------------------------------------------------------------- worker


def run_unit(spec: dict) -> dict:
    """One (seed, Stage 1 arm) or (seed, Stage 2 frame relay): regenerate, score, return.
    The RNG consumption and row writing follow run_consistency_defense.run_unit exactly."""
    sh = s2._SHARED
    ctx, seed = sh.ctx, spec["seed"]
    n, width = len(ctx.y), sh.X0imp.shape[1]
    t_unit = time.perf_counter()
    zoo, wbs, gate, t_fit = cd._zoo(seed)
    s_c, thrs = spec["s_c"], spec["thresholds"]
    names = seq_names(sh.maps)
    space, written, rng, rng_rand = cd.make_space(spec, sh)
    label = cd.unit_label(spec)
    cells = []

    def score(v, eps, source, families):
        X, _ = space.rows(v)
        ctl = space.check(v, X, eps, f"{label} s{seed} {source} eps {eps}")  # Stage 2 C3/C4
        raw = sh.X0raw.copy()
        a = space.att
        raw[np.ix_(a, written)] = X[np.ix_(a, written)]
        res_seq = seq_residuals(raw, sh.maps, s_c)
        sc = all_scores(raw, sh.maps, s_c)
        cell = {
            "label": label,
            "kind": spec["kind"],
            "seed": seed,
            "eps": eps,
            "source": source,
            "threshold": thrs,
            "scores": {k: v.tolist() for k, v in sc.items()},
            "attacked_f1": {f: s2._f1_of(zoo, f, ctx.y, space.Z(X)) for f in families},
            "top_channels_sequence_band": Counter(
                names[i] for i in res_seq.reshape(n, -1).argmax(axis=1)
            ).most_common(5),
            "n_attacked_rows": int(a.sum()),
            "stage2_checks": {k: ctl[k] for k in ("untouched_max_abs", "ctrl_max_over_eps")},
        }
        if spec["kind"] == "frame":
            j = RELAYS.index(spec["relay"])
            res14 = cd.residuals(raw, sh.maps, s_c)
            cell["attacked_relay_max_residual_on_attacked_rows"] = {
                SCORES[0]: float(res_seq[a, j].max()),
                SCORES[1]: float(res14[a, j].max()),
            }
        cells.append(cell)

    for eps in EPS_GRID:  # the recorded runs' consumption order, skipped where not scored
        for src in s2.STAGE1_SOURCES:
            if eps in spec["eps_grid"] and src in cd.SOURCES:
                v = space.pgd(wbs[src], ctx.y, eps, ctx.n_steps, ctx.step_frac, rng)
                score(v, eps, src, [src])
            else:
                rng.uniform(-1.0, 1.0, size=(n, width))
        if eps in spec["eps_grid"]:
            score(space.random_sign(eps, rng_rand), eps, "random", cd.ANCHOR_FAMILIES)
        else:
            rng_rand.choice([-1.0, 1.0], size=(n, width))
        log.info(
            "%s seed %d eps %.2f done (%.0f s)", label, seed, eps, time.perf_counter() - t_unit
        )
    return {
        "unit": cd.unit_name(spec),
        "cells": cells,
        "qsvm_gate": gate,
        "zoo_fit_s": t_fit,
        "wall_s": round(time.perf_counter() - t_unit, 1),
        "peak_rss_gb": round(s2._peak_rss_gb(), 3),
        "pid": os.getpid(),
    }


# --------------------------------------------------------------------------- parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, required=True, help="BLAS/OpenMP/torch per worker")
    ap.add_argument("--workers", type=int, default=1, help="parallel (seed, arm) units")
    ap.add_argument("--smoke", action="store_true", help="40 rows, seed 0, A1+A2-R1+frame R1")
    ap.add_argument("--worker-rss-gb", type=float, help="per-worker peak RSS measured by smoke")
    ap.add_argument("--resume", type=Path, help="results_partial.json of an interrupted run")
    args = ap.parse_args()
    if str(args.threads) != _THREADS:
        raise AssertionError(f"early thread parse {_THREADS} != argparse {args.threads}")
    T, W = args.threads, 2 if args.smoke else args.workers
    n_cpu = psutil.cpu_count(logical=True)
    print(f"workers W={W} x threads T={T} = {W * T} of {n_cpu} logical threads", flush=True)
    if W * T > n_cpu:
        raise ValueError(f"W x T = {W * T} exceeds the {n_cpu} logical CPUs")

    shared_args = SimpleNamespace(smoke=args.smoke, angle_unit="frame", magnitude_unit="scale")
    sh, map_diag = s2.build_shared(shared_args, T, RELAYS)
    if list(sh.ctx.arm_masks) != s2.STAGE1_ARMS:
        raise AssertionError(f"arm order {list(sh.ctx.arm_masks)} != Stage 1's {s2.STAGE1_ARMS}")

    Xraw, _, splits = load_processed(VARIANT)
    Xtr, Xva, Xte = (Xraw[splits[k]].astype(np.float64) for k in ("train", "val", "test"))
    nanstd = np.nanstd(Xtr, axis=0)
    train_std = {r: nanstd[sh.maps[r].der_idx] for r in RELAYS}
    s_c, s_c_record = cd.robust_scale(Xtr, sh.maps, nanstd)

    pre = preconditions(Xtr, Xva, sh.maps, s_c, s_c_record)
    log.info(
        "preconditions: %s", {k: v for k, v in pre.items() if k.startswith("P") and "pass" in v}
    )

    sc_va, sc_te, sc_ev = (all_scores(X, sh.maps, s_c) for X in (Xva, Xte, sh.X0raw))
    thrs = {k: threshold(sc_va[k]) for k in SCORES}
    ens4_ok = abs(thrs[SCORES[1]] - ENS4_SMOKE_THRESHOLD) <= 1e-9 * ENS4_SMOKE_THRESHOLD
    c1, c1_ok = cd.control_c1(Xtr, sh.maps, train_std)
    clean = {}
    for k in SCORES:
        clean[k] = {
            "threshold": thrs[k],
            "false_alarm_val": float(np.mean(sc_va[k] > thrs[k])),
            "false_alarm_test": float(np.mean(sc_te[k] > thrs[k])),
            "val": cd.describe(sc_va[k]),
            "test": cd.describe(sc_te[k]),
            "eval_subset": cd.describe(sc_ev[k]),
        }
    gross = {}
    for part, X, sc in (("val", Xva, sc_va), ("test", Xte, sc_te)):
        g = np.zeros(len(X), bool)
        for r in RELAYS:
            g |= band_state(X, sh.maps[r])[2].any(axis=1)
        gross[part] = {
            "n_rows_gross_mismatch": int(g.sum()),
            "n_flagged": int(np.sum(sc[SCORES[0]][g] > thrs[SCORES[0]])),
        }
    pos = positive_control(Xte, sh.maps, s_c, train_std, thrs)
    log.info("thresholds %s; C2 curve %s", thrs, pos[SCORES[0]]["curve"])

    seeds = [0] if args.smoke else SEEDS
    arms = ["A1", "A2-R1"] if args.smoke else cd.ARMS
    frame_relays = [1] if args.smoke else RELAYS
    eps_grid = [0.5] if args.smoke else EPS_GRID
    common = {"eps_grid": eps_grid, "s_c": s_c, "thresholds": thrs}
    specs = [{"kind": "stage1", "seed": s, "arm": a} | common for s in seeds for a in arms]
    specs += [
        {"kind": "frame", "seed": s, "relay": r} | common for s in seeds for r in frame_relays
    ]
    cfg = {
        "threads": T,
        "workers": W,
        "smoke": args.smoke,
        "seeds": seeds,
        "arms": arms,
        "frame_relays": frame_relays,
        "eps_grid": eps_grid,
        "sources": cd.SOURCES + ["random"],
        "n_eval": int(len(sh.ctx.y)),
        "n_steps": sh.ctx.n_steps,
        "step_frac": sh.ctx.step_frac,
        "fa_target": FA_TARGET,
        "band_fraction_of_threshold": BAND,
        "scores": {
            SCORES[0]: "max over 4 relays x 12 sequence channels of |residual| / s_c (E-NS4 "
            "amendment-2 scale), state mismatches within +/-2% of the deadband threshold score 0",
            SCORES[1]: "E-NS4 amendment-2 score over all 14 derived channels (reported)",
        },
        "preregistration": "decision log 2026-11-16T11:39Z, amendment R4-11b",
        "stage1_results": str(s2.STAGE1_RESULTS.relative_to(REPO_ROOT)),
        "stage2_frame_results": str(cd.STAGE2_FRAME_RESULTS.relative_to(REPO_ROOT)),
        "resume": str(args.resume) if args.resume else None,
    }
    name = f"consistency_defense_sequence_{VARIANT}_q{QUBITS}" + ("_smoke" if args.smoke else "")
    with RunDir(name, config=cfg, seeds=seeds) as run:
        run.write_json("controls_map.json", map_diag)
        run.write_json("precondition.json", pre)
        if not pre["all_pass"] or not ens4_ok:
            raise AssertionError(
                f"precondition gate failed (P1-P4 {pre['all_pass']}, E-NS4 threshold "
                f"reproduced {ens4_ok}: {thrs[SCORES[1]]}); no attack unit is run"
            )
        extra = {}
        if args.smoke:  # C6: one worker, then two, sized on the first pass's peak RSS
            one = s2.run_pool(run_unit, specs, sh, 1)
            peak = max(u["peak_rss_gb"] for u in one)
            print(f"measured peak RSS of one worker: {peak:.2f} GB", flush=True)
            extra["memory_guard"] = s2.memory_guard(2, peak, "smoke pass 2")
            units = s2.run_pool(run_unit, specs, sh, 2)
            a, b = cd.cell_scores(one), cd.cell_scores(units)
            differ = [str(k) for k in a if a[k] != b.get(k)]
            extra["c6_parallel_invariance"] = {"n_cells": len(a), "n_differ": len(differ)}
            c6_ok = not differ and len(a) == len(b)
        else:
            if args.worker_rss_gb is None:
                raise SystemExit("--worker-rss-gb is required (smoke prints the measured value)")
            extra["memory_guard"] = s2.memory_guard(W, args.worker_rss_gb, "full run")
            done = json.loads(args.resume.read_text())["units"] if args.resume else []
            have = {u["unit"] for u in done}
            todo = [sp for sp in specs if cd.unit_name(sp) not in have]

            def save(res):
                done.append(res)
                run.write_json("results_partial.json", {"units": done})

            s2.run_pool(run_unit, todo, sh, W, on_done=save)
            units, c6_ok = done, True
        cells = [c for u in units for c in u["cells"]]
        n = len(sh.ctx.y)
        per_score, c3 = {}, {}
        for k in SCORES:
            table, frame, c3[k] = cd.metrics(
                cells, k, sc_te[k], sc_ev[k], thrs[k], clean[k]["test"]
            )
            per_score[f"score_{k}"] = {
                "clean": clean[k],
                "branch": cd.branch(table, eps_grid),
                "detection_table": table,
                "stage2_frame_rows": frame,
                "c3_frame_rows_flagged_within_fa_plus_margin": c3[k],
                "negative_control_whitebox_vs_random": random_contrast(cells, k),
            }
        anc = cd.anchors(
            cells, s2.load_stage1(n, sh.ctx.n_steps), cd.load_stage2_frame(n, sh.ctx.n_steps)
        )
        br = per_score[f"score_{SCORES[0]}"]["branch"]
        controls = {
            "P1_P4_preconditions": pre["all_pass"],
            "ens4_14ch_threshold_reproduced": ens4_ok,
            "C1_control1_reproduced_train": c1_ok,
            "C2_positive_control": pos[SCORES[0]]["pass"],
            "C3_stage2_rows_flagged_within_fa_plus_0.01": c3[SCORES[0]],
            "C4_anchors": anc["pass"] if anc["applied"] else "not applied (smoke)",
            "C6_parallel_invariance": c6_ok if args.smoke else "smoke only",
        }
        results = {
            "config": cfg,
            "controls": controls,
            "primary_score": SCORES[0],
            "branch_sequence_band": br,
            "branch_ens4_amended_14ch": per_score[f"score_{SCORES[1]}"]["branch"],
            "preconditions": pre,
            "c1_control1_reproduction": c1,
            "c2_positive_control": pos,
            "gross_state_mismatch_rows": gross,
            **per_score,
            "c4_anchors": anc,
            **extra,
            "units_meta": [
                {k: u[k] for k in ("unit", "zoo_fit_s", "wall_s", "peak_rss_gb", "pid")}
                for u in sorted(units, key=lambda u: u["unit"])
            ],
            "qsvm_gate": {u["unit"]: u["qsvm_gate"] for u in units},
            "cells_meta": [{k: v for k, v in c.items() if k != "scores"} for c in cells],
        }
        run.write_json("results.json", results)
        np.savez_compressed(
            run.path / "scores.npz",
            **{
                f"clean_{p}|{k}": v[k] for p, v in (("val", sc_va), ("test", sc_te)) for k in SCORES
            },
            **{
                f"{c['label']}|{c['eps']}|{c['source']}|s{c['seed']}|{k}": c["scores"][k]
                for c in cells
                for k in SCORES
            },
        )
        run.metrics.update(
            {
                "thresholds": thrs,
                "branch_sequence_band": br["all_arms"]["verdict"],
                "c2_detection": pos[SCORES[0]]["c2_detection"],
                "controls": controls,
            }
        )
        log.info("controls: %s", json.dumps(controls))
        log.info("branch: %s", json.dumps(br))
        print(f"-> {run.path}")
        failed = [k for k, v in controls.items() if v is False]
        if failed:
            raise AssertionError(f"blocking controls failed: {failed}")


ENS4_SMOKE_THRESHOLD = 261.27386539745424  # E-NS4 amendment-2 clean validation threshold


if __name__ == "__main__":
    main()
