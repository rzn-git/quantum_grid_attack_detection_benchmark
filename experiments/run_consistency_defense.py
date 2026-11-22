"""E-NS4: a relay-local consistency check as a defense against the Stage 1 channel attacks.

Scores every row by how far each relay's reported derived channels sit from what the relay's
own phase phasors imply, fits an alarm threshold at 1% false alarms on clean validation rows,
and measures how well it separates clean test rows from the Stage 1 mask-only attacks (which
move derived channels independently of the phasors) and from the Stage 2 frame-arm rows
(physically consistent by construction). Pre-registered in experiment_decision_log.md
(2026-11-16T07:22Z, E-NS4).

Run (thread budget W x T; --threads is applied before numpy loads, in every worker):
    python -m experiments.run_consistency_defense --smoke --threads 2
    python -m experiments.run_consistency_defense --workers 3 --threads 3 --worker-rss-gb 1.2

Score. Per row: max over the 4 relays and each relay's 14 derived channels (PM7..PM12,
PA7..PA12, PA:Z, PA:ZH) of |X_derived - f(X_controlled)| / scale_c, f the Stage 2 derived map
fit on TRAIN (sequence assignment, deadband, impedance ratio and angle offset; imported, not
copied). Angle differences are wrapped (degrees for PA7..PA12, radians for PA:ZH). Two scales,
both computed for every row (pre-run amendment after the smoke's C2 failure: the train-std
scale let the impedance channels' fit noise set the threshold at 3.37 std):
  amended (primary): s_c = 1.4826 x MAD of channel c's signed clean TRAIN residual on the rows
      where the relay reports the channel (value finite and nonzero, i.e. above the deadband; an
      angle uses its magnitude partner's state, PA:ZH uses PA:Z's), on all rows for a channel
      never reported on train, floored at 1e-6 x the channel's train std (amendment 2, decision
      log 07:46Z; amendment 1's MAD over all rows collapsed to its floor on the deadband
      channels); s_c and the fit used are recorded per channel;
  preregistered_train_std: the derived channel's raw TRAIN std (np.nanstd, the denominator of
      Stage 2's control 1).
Each score has its own threshold, AUROC and detection; the branch is read on the amended score
and the pre-registered score's branch is recorded beside it. Per score, the clean validation and
test rows flagged only because the fitted deadband disagrees with the reported 0/nonzero state
(the row drops to the threshold or below once those channels are zeroed) are counted per relay
and channel, with their share of the 1% false-alarm budget.

Representation: RAW, before imputation. The check sits at the relay, which reports PA:Z as NaN
when I1 = 0; the map's own rule is "PA:Z is NaN exactly when I1 = 0" (Stage 2 control 1), so a
NaN PA:Z with I1 = 0 is consistent (residual 0) and a NaN/finite status that disagrees with
I1 is inconsistent (residual +inf, ranked above every finite score). An attacked row is the
relay's clean report with the attacker's written channels (its controlled set, plus the
recomputed derived set under Stage 2) replaced on its attackable rows. The imputed-matrix
score (what the models see; the map's z_impute in place of NaN) is computed beside it and the
rows where the two differ are counted.

Positives. Stage 1 rows regenerated through the Stage 2 runner's a2_mask AttackSpace path
(arms A2-R1..R4; A1 through the same path with Stage 1's A1 mask), Stage 1's RNG streams
replayed (spawn_rng(seed, "m1a") and "m1a_random", consumed arm by arm, eps by eps, source by
source), so the rows are Stage 1's; QSVM and RBF-SVM white-box PGD plus random sign, eps
{0.1, 0.25, 0.5}, seeds {0, 1, 2}. Stage 2 frame-arm rows regenerated the same way through the
consistent frame path (spawn_rng(seed, "m1a2_consistent_R<r>")). Each (seed, arm) unit runs in
its own process; a worker keeps the last fitted zoo, so a seed's zoo is fitted once per worker.

Metrics per arm, eps, source (seed mean, std, min): AUROC (positives against all clean test
rows), detection rate at the threshold (the 99th percentile, method "higher", of clean
validation scores; a row is flagged when its score exceeds it), realised false-alarm rate on
clean test rows. Branch (seed means, white-box sources): kill if AUROC < 0.9 or detection
< 0.9 for any arm at eps 0.5; hypothesis if AUROC >= 0.99 and detection >= 0.9 for every arm
at every eps; else intermediate.

Blocking controls (each raises after results are written): C1 the score reproduces Stage 2's
control 1 per-channel train medians within 1e-9 + 1e-6 relative, and on clean test each
sequence magnitude's median residual is within control 1's gate (both on the train-std scale);
C2 positive control, a random-sign offset of 1 train std on one randomly chosen finite derived
channel per clean test row is flagged on >= 99% of rows by the amended score (the curve at
0.25, 0.5, 1 and 2 train std is reported for both scores); C3 negative reading, the Stage 2
frame rows of every (relay, eps, source) cell are flagged by the amended score at no more than
the clean test false-alarm rate + 0.01 (1 - AUROC is reported as a "too-consistent"
diagnostic, not gated; the old clean-IQR reading is kept as a record); C4 anchors (full run),
the regenerated rows reproduce Stage 1's and Stage 2's recorded attacked macro-F1 within 0.01;
C5 Stage 2's own controls 3 and 4 on every regenerated set; C6 (smoke) 1 and 2 workers give
identical scores.
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
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from experiments.run_adversarial import FittedZoo  # noqa: E402
from experiments.run_channel_restricted_attack import (  # noqa: E402
    ENCODING,
    EPS_GRID,
    QUBITS,
    RELAYS,
    SEEDS,
    VARIANT,
    make_whitebox,
)
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.features.reduce import build_preprocessor  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger  # noqa: E402
from qgridbench.utils.seeding import spawn_rng  # noqa: E402

log = get_logger(__name__)

STAGE2_FRAME_RESULTS = (
    REPO_ROOT / "results" / "runs" / "20261116T065642Z_consistent_relay_attack_binary_q8_frame"
)
ARMS = ["A1", "A2-R1", "A2-R2", "A2-R3", "A2-R4"]
SOURCES = ["qsvm", "rbf_svm"]  # white-box PGD sources; "random" is the random-sign set
ANCHOR_FAMILIES = ["qsvm", "rbf_svm"]
FA_TARGET = 0.01
N_SEQ = len(s2.SEQ_CHANNELS)
ANG_COLS = slice(N_SEQ, 2 * N_SEQ)  # PA7..PA12, degrees (s2.derived_np's output layout)
Z_COL, ZH_COL = 2 * N_SEQ, 2 * N_SEQ + 1  # PA:Z, PA:ZH (radians)
C1_ABS_TOL, C1_REL_TOL = 1e-9, 1e-6
POS_CONTROL_MIN = 0.99
POS_CURVE_OFFSETS = (0.25, 0.5, 1.0, 2.0)  # train-std units
C2_OFFSET = 1.0
C3_MARGIN = 0.01  # Stage 2 rows flagged at most the clean test false-alarm rate + this
MAD_CONSISTENCY, MAD_FLOOR = 1.4826, 1e-6
SCORES = ("amended", "preregistered_train_std")  # the amended score is primary
HYP_AUROC, HYP_DET, KILL_AUROC, KILL_DET, KILL_EPS = 0.99, 0.90, 0.90, 0.90, 0.5


# --------------------------------------------------------------------------- score


def relay_signed(X: np.ndarray, rm: s2.RelayMap, nan_rule: bool):
    """(n, 14) signed reported - map for one relay, angles wrapped; NaN where PA:Z is NaN.

    Also returns the (n,) mask of rows where I1 == 0 (the map's PA:Z NaN condition).
    """
    xc, xd = X[:, rm.ctrl_idx], X[:, rm.der_idx]
    if not np.isfinite(xc).all():
        raise AssertionError(f"R{rm.relay}: a controlled channel is not finite")
    nan_d = np.isnan(xd)
    other = np.ones(xd.shape[1], bool)
    other[Z_COL] = False
    if nan_d[:, other].any() or (nan_d.any() and not nan_rule):
        raise AssertionError(f"R{rm.relay}: NaN in a derived channel the rule does not cover")
    d = xd - s2.derived_np(xc, rm)
    d[:, ANG_COLS] = s2.wrap_deg(d[:, ANG_COLS])
    d[:, ZH_COL] = s2.wrap_rad(d[:, ZH_COL])
    _, si = s2.sequences_np(xc)
    return d, np.abs(si["pos"]) == 0


def relay_residual(X: np.ndarray, rm: s2.RelayMap, sig: np.ndarray, nan_rule: bool):
    """(n, 14) |reported - map| / scale for one relay; nan_rule applies the raw PA:Z rule."""
    d, i1_zero = relay_signed(X, rm, nan_rule)
    r = np.abs(d) / sig
    if nan_rule:  # the map: PA:Z is NaN exactly when I1 == 0
        z_nan = np.isnan(d[:, Z_COL])
        r[:, Z_COL] = np.where(z_nan & i1_zero, 0.0, np.where(z_nan | i1_zero, np.inf, r[:, Z_COL]))
    return r


def residuals(X, maps, sigma, nan_rule=True) -> np.ndarray:
    """(n, n_relays, 14) residuals, relays in RELAYS order."""
    return np.stack([relay_residual(X, maps[r], sigma[r], nan_rule) for r in RELAYS], axis=1)


def reported(X: np.ndarray, rm: s2.RelayMap) -> np.ndarray:
    """(n, 14) the relay reports the channel: its value (an angle: its magnitude partner's;
    PA:ZH: PA:Z's) is finite and nonzero, i.e. above the deadband for a sequence channel."""
    xd = X[:, rm.der_idx]
    on = np.isfinite(xd) & (xd != 0)
    out = on.copy()
    out[:, ANG_COLS] = on[:, :N_SEQ]
    out[:, ZH_COL] = on[:, Z_COL]
    return out


def robust_scale(Xtr: np.ndarray, maps, nanstd: np.ndarray) -> tuple[dict, dict]:
    """Amended scale (amendment 2) s_c: 1.4826 x MAD of each channel's signed clean TRAIN
    residual (angles wrapped) on the rows where the relay reports the channel; all finite rows
    for a channel never reported on train; floored at MAD_FLOOR x the channel's train std."""
    scale, rec = {}, {}
    for r in RELAYS:
        rm = maps[r]
        d, _ = relay_signed(Xtr, rm, nan_rule=True)
        rep = reported(Xtr, rm)
        s = np.empty(d.shape[1])
        for j, name in enumerate(rm.der_names):
            fin = np.isfinite(d[:, j])
            fit = "reported_rows" if (rep[:, j] & fin).any() else "all_rows"
            x = d[fin & rep[:, j], j] if fit == "reported_rows" else d[fin, j]
            mad = MAD_CONSISTENCY * float(np.median(np.abs(x - np.median(x))))
            floor = MAD_FLOOR * float(nanstd[rm.der_idx[j]])
            s[j] = max(mad, floor)
            rec[name] = {
                "s_c": s[j],
                "fit": fit,
                "mad_scaled": mad,
                "floor": floor,
                "floored": bool(mad < floor),
                "train_std": float(nanstd[rm.der_idx[j]]),
                "n_train_rows_fit": int(len(x)),
            }
        scale[r] = s
    return scale, rec


def deadband_disagreement(X: np.ndarray, maps) -> np.ndarray:
    """(n, n_relays, 14) the fitted deadband's on/off state disagrees with the reported
    0/nonzero state of a sequence channel (marked on the magnitude and its angle)."""
    out = np.zeros((len(X), len(RELAYS), 2 * N_SEQ + 2), bool)
    for i, r in enumerate(RELAYS):
        rm = maps[r]
        sv, si = s2.sequences_np(X[:, rm.ctrl_idx])
        for j, k in enumerate(s2.SEQ_CHANNELS):
            on = np.abs((sv if k <= 9 else si)[rm.assign[k]]) > rm.deadband[k]
            dis = on != (X[:, rm.der_idx[j]] != 0)
            out[:, i, j] = out[:, i, N_SEQ + j] = dis
    return out


def flagged_by_deadband_only(res, dis, thr, names) -> dict:
    """Rows flagged (score > thr) that drop to <= thr once the disagreeing channels are zeroed,
    attributed to their (relay, magnitude channel), or "multiple" when more than one."""
    flagged = row_score(res) > thr
    only = flagged & (row_score(np.where(dis, 0.0, res)) <= thr)
    keys = Counter()
    mag = np.zeros(dis.shape[2], bool)
    mag[:N_SEQ] = True
    for i in np.flatnonzero(only):
        hit = [names[f] for f in np.flatnonzero((dis[i] & mag).reshape(-1))]
        keys[hit[0] if len(hit) == 1 else "multiple"] += 1
    return {
        "n_rows": int(len(res)),
        "n_flagged": int(flagged.sum()),
        "n_flagged_only_by_deadband_disagreement": int(only.sum()),
        "share_of_flagged": float(only.sum() / max(flagged.sum(), 1)),
        "share_of_1pct_budget": float(only.sum() / (FA_TARGET * len(res))),
        "by_relay_channel": dict(keys.most_common()),
        "n_rows_with_any_disagreement": int(dis.reshape(len(dis), -1).any(axis=1).sum()),
    }


def row_score(res: np.ndarray) -> np.ndarray:
    return res.reshape(len(res), -1).max(axis=1)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based (inf-safe; AUROC depends on ranks only)."""
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(lab, rankdata(np.r_[pos, neg])))


def describe(s: np.ndarray) -> dict:
    fin = s[np.isfinite(s)]
    q = np.percentile(fin, [25, 50, 75, 99])
    return {
        "n": int(len(s)),
        "n_inf": int(np.isinf(s).sum()),
        "q25": float(q[0]),
        "median": float(q[1]),
        "q75": float(q[2]),
        "p99": float(q[3]),
        "max_finite": float(fin.max()),
    }


def channel_names(maps) -> list[str]:
    return [n for r in RELAYS for n in maps[r].der_names]


# --------------------------------------------------------------------------- attack spaces


class MaskSpace(s2.AttackSpace):
    """The Stage 2 runner's a2_mask path (Stage 1's mask-only attacker) on any Stage 1 arm mask.

    The parent builds the a2_mask space of one relay, then swaps in the arm's channel set; box,
    rows, Z, grad, pgd, random_sign and check read only ctrl, sc and x0c in this mode.
    """

    def __init__(self, arm: str, sh: s2.Shared, device: str = "cpu"):
        super().__init__("a2_mask", sh.maps[RELAYS[0]], sh, device)
        self.ctrl = np.flatnonzero(sh.ctx.arm_masks[arm])
        self.sc = sh.s[self.ctrl].copy()
        self.x0c = sh.X0imp[:, self.ctrl]


def make_space(spec: dict, sh: s2.Shared):
    """(space, written channels, Stage 1/2 RNG streams already advanced to this unit)."""
    seed, n, width = spec["seed"], len(sh.ctx.y), sh.X0imp.shape[1]
    if spec["kind"] == "stage1":
        arm = spec["arm"]
        if arm.startswith("A2-R"):
            space = s2.AttackSpace("a2_mask", sh.maps[int(arm[4:])], sh, "cpu")
        else:
            space = MaskSpace(arm, sh)
        rng, rng_rand = spawn_rng(seed, "m1a"), spawn_rng(seed, "m1a_random")
        before = s2.STAGE1_ARMS.index(arm) * len(EPS_GRID)  # Stage 1 ran arms in this order
        for _ in range(before * len(s2.STAGE1_SOURCES)):
            rng.uniform(-1.0, 1.0, size=(n, width))
        for _ in range(before):
            rng_rand.choice([-1.0, 1.0], size=(n, width))
        return space, space.ctrl, rng, rng_rand
    r = spec["relay"]
    rm = sh.maps[r]
    space = s2.AttackSpace("consistent", rm, sh, "cpu")
    if not space.frame:
        raise AssertionError("Stage 2 rows must come from the frame arm")
    rng = spawn_rng(seed, f"m1a2_consistent_R{r}")
    rng_rand = spawn_rng(seed, f"m1a2_consistent_random_R{r}")
    return space, np.concatenate([rm.ctrl_idx, rm.der_idx]), rng, rng_rand


# --------------------------------------------------------------------------- worker

_ZOO: dict = {}


def _zoo(seed: int):
    """The zoo of one seed, kept per worker (a refit is seeded and identical; this saves time)."""
    sh = s2._SHARED
    if seed not in _ZOO:
        _ZOO.clear()
        t0 = time.perf_counter()
        ctx = sh.ctx
        zoo = FittedZoo(ctx.reg, ENCODING, QUBITS, seed, ctx.subset_cap, ctx.best_params)
        wbs, gate = make_whitebox(zoo, ctx)
        _ZOO[seed] = (zoo, wbs, gate, round(time.perf_counter() - t0, 1))
        return _ZOO[seed]
    zoo, wbs, gate, _ = _ZOO[seed]
    return zoo, wbs, gate, 0.0


def run_unit(spec: dict) -> dict:
    """One (seed, Stage 1 arm) or (seed, Stage 2 frame relay): regenerate, score, return."""
    sh = s2._SHARED
    ctx, seed = sh.ctx, spec["seed"]
    n, width = len(ctx.y), sh.X0imp.shape[1]
    t_unit = time.perf_counter()
    zoo, wbs, gate, t_fit = _zoo(seed)
    scales = spec["scales"]
    names = channel_names(sh.maps)
    mag_idx = np.concatenate([sh.maps[r].ctrl_idx[:6] for r in RELAYS])
    space, written, rng, rng_rand = make_space(spec, sh)
    label = unit_label(spec)
    cells = []

    def score(v, eps, source, families):
        X, _ = space.rows(v)
        ctl = space.check(v, X, eps, f"{label} s{seed} {source} eps {eps}")  # Stage 2 C3/C4
        raw = sh.X0raw.copy()
        a = space.att
        raw[np.ix_(a, written)] = X[np.ix_(a, written)]
        res = {k: residuals(raw, sh.maps, scales[k]) for k in SCORES}
        s_imp = row_score(residuals(X, sh.maps, scales[SCORES[0]], nan_rule=False))
        Z = space.Z(X)
        cell = {
            "label": label,
            "kind": spec["kind"],
            "seed": seed,
            "eps": eps,
            "source": source,
            "scores": {k: row_score(res[k]).tolist() for k in SCORES},
            "scores_imputed": s_imp.tolist(),
            "attacked_f1": {f: s2._f1_of(zoo, f, ctx.y, Z) for f in families},
            "top_channels": {
                k: Counter(names[i] for i in res[k].reshape(n, -1).argmax(axis=1)).most_common(5)
                for k in SCORES
            },
            "n_rows_magnitude_below_min0_clean": int(
                (raw[:, mag_idx] < np.minimum(sh.X0raw[:, mag_idx], 0.0)).any(axis=1).sum()
            ),
            "n_attacked_rows": int(a.sum()),
            "stage2_checks": {k: ctl[k] for k in ("untouched_max_abs", "ctrl_max_over_eps")},
        }
        if spec["kind"] == "frame":
            j = RELAYS.index(spec["relay"])
            cell["attacked_relay_max_residual_on_attacked_rows"] = {
                k: float(res[k][a, j].max()) for k in SCORES
            }
        cells.append(cell)

    for eps in EPS_GRID:  # the recorded runs' consumption order, skipped where not scored
        for src in s2.STAGE1_SOURCES:
            if eps in spec["eps_grid"] and src in SOURCES:
                v = space.pgd(wbs[src], ctx.y, eps, ctx.n_steps, ctx.step_frac, rng)
                score(v, eps, src, [src])
            else:
                rng.uniform(-1.0, 1.0, size=(n, width))
        if eps in spec["eps_grid"]:
            score(space.random_sign(eps, rng_rand), eps, "random", ANCHOR_FAMILIES)
        else:
            rng_rand.choice([-1.0, 1.0], size=(n, width))
        log.info(
            "%s seed %d eps %.2f done (%.0f s)", label, seed, eps, time.perf_counter() - t_unit
        )
    return {
        "unit": unit_name(spec),
        "cells": cells,
        "qsvm_gate": gate,
        "zoo_fit_s": t_fit,
        "wall_s": round(time.perf_counter() - t_unit, 1),
        "peak_rss_gb": round(s2._peak_rss_gb(), 3),
        "pid": os.getpid(),
    }


# --------------------------------------------------------------------------- parent


def unit_label(spec: dict) -> str:
    return spec["arm"] if spec["kind"] == "stage1" else f"frame-R{spec['relay']}"


def unit_name(spec: dict) -> str:
    return f"{unit_label(spec)}_s{spec['seed']}"


def cell_scores(units: list[dict]) -> dict:
    """Scores per (label, seed, eps, source): C6's comparison key."""
    return {
        (c["label"], c["seed"], c["eps"], c["source"]): c["scores"]
        for u in units
        for c in u["cells"]
    }


def control_c1(Xtr, maps, sigma) -> tuple[dict, bool]:
    """C1 (train): this score reproduces Stage 2's control-1 per-channel medians."""
    out, ok = {}, True
    for r in RELAYS:
        rm = maps[r]
        res = relay_residual(Xtr, rm, sigma[r], nan_rule=True)
        _, si = s2.sequences_np(Xtr[:, rm.ctrl_idx])
        i1 = np.abs(si["pos"]) > 0
        z, zh = Xtr[:, rm.der_idx[Z_COL]], Xtr[:, rm.der_idx[ZH_COL]]
        pairs = []
        for j, k in enumerate(s2.SEQ_CHANNELS):
            c1 = rm.control1[f"PM{k}/PA{k}"]
            pairs.append((f"PM{k}", np.median(res[:, j]), c1["mag_median_err_over_std"]))
            pairs.append((f"PA{k}", np.median(res[:, N_SEQ + j]), c1["angle_median_err_over_std"]))
        okz = np.isfinite(z) & (z > 0) & i1
        okh = np.isfinite(zh) & i1
        pairs.append(
            ("PA:Z", np.median(res[okz, Z_COL]), rm.control1["PA:Z"]["median_err_over_std"])
        )
        pairs.append(
            ("PA:ZH", np.median(res[okh, ZH_COL]), rm.control1["PA:ZH"]["median_err_over_std"])
        )
        rows = {}
        for name, mine, ref in pairs:
            diff = abs(float(mine) - ref)
            good = diff <= C1_ABS_TOL + C1_REL_TOL * abs(ref)
            ok &= good
            rows[name] = {"score_median": float(mine), "control1_median": ref, "pass": good}
        out[f"R{r}"] = rows
    return out, bool(ok)


def clean_channel_table(res: np.ndarray, maps) -> tuple[dict, bool]:
    """Per-channel clean-test median and p99 residual; magnitudes against control 1's gate."""
    out, ok = {}, True
    for i, r in enumerate(RELAYS):
        rm = maps[r]
        for j, name in enumerate(rm.der_names):
            x = res[:, i, j]
            x = x[np.isfinite(x)]
            row = {"median": float(np.median(x)), "p99": float(np.percentile(x, 99))}
            if j < N_SEQ:
                k = s2.SEQ_CHANNELS[j]
                row["control1_gate"] = s2.SEQ_MAG_TOL[k]
                row["control1_train_p99"] = rm.control1[f"PM{k}/PA{k}"]["mag_p99_err_over_std"]
                row["pass"] = bool(row["median"] <= s2.SEQ_MAG_TOL[k])
                ok &= row["pass"]
            out[name] = row
    return out, bool(ok)


def positive_control(Xte, maps, train_std, scales, thrs) -> dict:
    """C2 and its curve: a random-sign offset of k train std on one random finite derived
    channel per clean test row (the same rows, channels and signs at every k), per score."""
    rng = spawn_rng(0, "ns4_positive_control")
    der = np.concatenate([maps[r].der_idx for r in RELAYS])
    sig = np.concatenate([train_std[r] for r in RELAYS])
    n, n_ch = len(Xte), len(maps[RELAYS[0]].der_names)
    pick = np.where(np.isfinite(Xte[:, der]), rng.random((n, len(der))), -1.0).argmax(axis=1)
    sign = rng.choice([-1.0, 1.0], size=n)
    names = channel_names(maps)
    out = {name: {"curve": {}, "threshold": thrs[name]} for name in SCORES}
    for k in sorted(set(POS_CURVE_OFFSETS) | {C2_OFFSET}):
        X = Xte.copy()
        X[np.arange(n), der[pick]] += k * sign * sig[pick]
        for name in SCORES:
            s = row_score(residuals(X, maps, scales[name]))
            det = s > thrs[name]
            out[name]["curve"][str(k)] = float(det.mean())
            if k == C2_OFFSET:
                out[name]["by_channel_type_at_c2_offset"] = {
                    names[j].split("-", 1)[1]: float(det[pick % n_ch == j].mean())
                    for j in range(n_ch)
                }
                out[name]["score_at_c2_offset"] = describe(s)
    for name in SCORES:
        rate = out[name]["curve"][str(C2_OFFSET)]
        out[name] |= {"c2_detection": rate, "pass": bool(rate >= POS_CONTROL_MIN)}
    return {
        "offset_unit": "train std of the channel",
        "c2_offset": C2_OFFSET,
        "gate": POS_CONTROL_MIN,
        "gated_score": SCORES[0],
        "n_rows": n,
        **out,
    }


def load_stage2_frame(n_eval: int, n_steps: int):
    res = json.loads((STAGE2_FRAME_RESULTS / "results.json").read_text())
    cfg = res["config"]
    if cfg["n_eval"] != n_eval or cfg["n_steps"] != n_steps or cfg["angle_unit"] != "frame":
        return None  # smoke: different rows and steps
    return {
        (x["seed"], x["relay"], x["eps"], x["family"], x["kind"]): x["attacked_f1"]
        for x in res["records"]
        if x["mode"] == "consistent"
    }


def anchors(cells, stage1, stage2) -> dict:
    """C4: regenerated rows reproduce the recorded attacked macro-F1 of both runs."""
    if stage1 is None or stage2 is None:
        return {"applied": False, "reason": "smoke rows and steps differ from the recorded runs"}
    rows = []
    for c in cells:
        for fam, f1 in c["attacked_f1"].items():
            kind = "random" if c["source"] == "random" else "whitebox"
            if c["kind"] == "stage1":
                ref = stage1[(c["seed"], c["label"], c["eps"], fam, kind)]
            else:
                ref = stage2[(c["seed"], int(c["label"][7:]), c["eps"], fam, kind)]
            rows.append(
                {
                    "label": c["label"],
                    "seed": c["seed"],
                    "eps": c["eps"],
                    "source": c["source"],
                    "family": fam,
                    "attacked_f1": f1,
                    "recorded": ref,
                    "abs_diff": abs(f1 - ref),
                }
            )
    worst = max(r["abs_diff"] for r in rows)
    return {
        "applied": True,
        "tol": s2.ANCHOR_TOL,
        "max_abs_diff": worst,
        "n_cells": len(rows),
        "pass": bool(worst <= s2.ANCHOR_TOL),
        "cells": rows,
    }


def _msm(v) -> dict:
    v = np.asarray(v, float)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "min": float(v.min()),
    }


def metrics(cells, name, neg_test, neg_eval, thr, clean_test_desc) -> tuple[list, list, bool]:
    """Per (label, eps, source) on score `name`: Stage 1 detection table, Stage 2 frame
    table and C3 (frame rows flagged at most the clean test false-alarm rate + C3_MARGIN)."""
    fa_test, fa_eval = float(np.mean(neg_test > thr)), float(np.mean(neg_eval > thr))
    groups: dict = {}
    for c in cells:
        groups.setdefault((c["kind"], c["label"], c["eps"], c["source"]), []).append(c)
    table, frame, c3_ok = [], [], True
    for (kind, label, eps, src), cs in sorted(groups.items()):
        cs = sorted(cs, key=lambda c: c["seed"])
        per = []
        for c in cs:
            s = np.array(c["scores"][name])
            p = {
                "seed": c["seed"],
                "auroc": auroc(s, neg_test),
                "detection": float(np.mean(s > thr)),
            }
            if name == SCORES[0]:  # the imputed-representation diagnostic, amended score
                si = np.array(c["scores_imputed"])
                p |= {"auroc_imputed_scores": auroc(si, neg_test)}
                p |= {"n_rows_raw_ne_imputed": int(np.sum(s != si))}
            per.append(p)
        row = {
            "label": label,
            "eps": eps,
            "source": src,
            "seeds": [c["seed"] for c in cs],
            "auroc": _msm([p["auroc"] for p in per]),
            "detection": _msm([p["detection"] for p in per]),
            "false_alarm_test": fa_test,
            "false_alarm_eval_subset_clean": fa_eval,
            "per_seed": per,
        }
        if kind == "stage1":
            table.append(row)
        else:
            pooled = np.concatenate([c["scores"][name] for c in cs])
            d = describe(pooled)
            flagged = float(np.mean(pooled > thr))
            ok = flagged <= fa_test + C3_MARGIN
            c3_ok &= ok
            row |= {
                "score": d,
                "flagged_pooled_seeds": flagged,
                "c3_limit": fa_test + C3_MARGIN,
                "c3_pass": bool(ok),
                "too_consistent_1_minus_auroc": 1.0 - auroc(pooled, neg_test),
                "clean_test_iqr": [clean_test_desc["q25"], clean_test_desc["q75"]],
                "clean_test_median": clean_test_desc["median"],
                "median_inside_clean_iqr": bool(
                    clean_test_desc["q25"] <= d["median"] <= clean_test_desc["q75"]
                ),
                "attacked_relay_max_residual": max(
                    c["attacked_relay_max_residual_on_attacked_rows"][name] for c in cs
                ),
            }
            frame.append(row)
    return table, frame, bool(c3_ok)


def branch(table, eps_grid) -> dict:
    wb = [t for t in table if t["source"] in SOURCES]

    def verdict(rows, stat):
        top = [t for t in rows if t["eps"] == KILL_EPS]
        killed = [
            (t["label"], t["source"])
            for t in top
            if t["auroc"][stat] < KILL_AUROC or t["detection"][stat] < KILL_DET
        ]
        hyp = all(t["auroc"][stat] >= HYP_AUROC and t["detection"][stat] >= HYP_DET for t in rows)
        v = "kill" if killed else "hypothesis" if hyp else "intermediate"
        return {"verdict": v, "killed_cells_at_eps_0.5": killed}

    return {
        "rule": "seed means of the white-box sources; kill: AUROC < 0.9 or detection < 0.9 for "
        "any arm at eps 0.5; hypothesis: AUROC >= 0.99 and detection >= 0.9 for every arm at "
        "every eps; else intermediate",
        "all_arms": verdict(wb, "mean"),
        "a2_arms_only": verdict([t for t in wb if t["label"].startswith("A2-")], "mean"),
        "sensitivity_worst_seed": verdict(wb, "min"),
        "complete_grid": len({t["label"] for t in wb}) == len(ARMS) and len(eps_grid) == 3,
    }


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

    sh, map_diag = s2.build_shared(SimpleNamespace(smoke=args.smoke, angle_unit="frame"), T, RELAYS)
    if list(sh.ctx.arm_masks) != s2.STAGE1_ARMS:
        raise AssertionError(f"arm order {list(sh.ctx.arm_masks)} != Stage 1's {s2.STAGE1_ARMS}")
    union = np.sort(np.concatenate([np.r_[m.ctrl_idx, m.der_idx] for m in sh.maps.values()]))
    if not np.array_equal(np.flatnonzero(sh.ctx.arm_masks["A1"]), union):
        raise AssertionError("A1 is not the union of the four relays' map channels")

    Xraw, _, splits = load_processed(VARIANT)
    Xtr, Xva, Xte = (Xraw[splits[k]].astype(np.float64) for k in ("train", "val", "test"))
    imputer = build_preprocessor("pca", n_components=QUBITS, seed=0).fit(Xraw[splits["train"]])
    imputer = imputer.named_steps["impute"]
    nanstd = np.nanstd(Xtr, axis=0)
    sigma = {r: nanstd[sh.maps[r].der_idx] for r in RELAYS}
    s_c, s_c_record = robust_scale(Xtr, sh.maps, nanstd)
    scales = {SCORES[0]: s_c, SCORES[1]: sigma}

    c1, c1_ok = control_c1(Xtr, sh.maps, sigma)
    chan, chan_ok = clean_channel_table(residuals(Xte, sh.maps, sigma), sh.maps)
    names = channel_names(sh.maps)
    Xte_imp = imputer.transform(Xte)
    dis = {"val": deadband_disagreement(Xva, sh.maps), "test": deadband_disagreement(Xte, sh.maps)}
    clean, s_clean = {}, {}
    for k in SCORES:
        res_va, res_te = residuals(Xva, sh.maps, scales[k]), residuals(Xte, sh.maps, scales[k])
        s_va, s_te = row_score(res_va), row_score(res_te)
        s_te_imp = row_score(residuals(Xte_imp, sh.maps, scales[k], nan_rule=False))
        s_ev = row_score(residuals(sh.X0raw, sh.maps, scales[k]))
        thr = float(np.quantile(s_va, 1.0 - FA_TARGET, method="higher"))
        over = s_va > thr
        s_clean[k] = {"val": s_va, "test": s_te, "test_imputed": s_te_imp, "eval": s_ev}
        clean[k] = {
            "threshold": thr,
            "false_alarm_val": float(np.mean(over)),
            "false_alarm_test": float(np.mean(s_te > thr)),
            "val": describe(s_va),
            "test": describe(s_te),
            "eval_subset": describe(s_ev),
            "test_imputed_representation": describe(s_te_imp),
            "test_rows_raw_ne_imputed": int(np.sum(s_te != s_te_imp)),
            "test_top_channels": Counter(
                names[i] for i in res_te.reshape(len(s_te), -1).argmax(axis=1)
            ).most_common(8),
            "val_rows_over_threshold_top_channels": Counter(
                names[i] for i in res_va[over].reshape(over.sum(), -1).argmax(axis=1)
            ).most_common(8),
            "flagged_only_by_deadband_disagreement": {
                "val": flagged_by_deadband_only(res_va, dis["val"], thr, names),
                "test": flagged_by_deadband_only(res_te, dis["test"], thr, names),
            },
        }
    clean[SCORES[1]]["test_per_channel"] = chan
    thrs = {k: clean[k]["threshold"] for k in SCORES}
    pos = positive_control(Xte, sh.maps, sigma, scales, thrs)
    for k in SCORES:
        log.info(
            "clean %s: threshold %.4f, test median %.4f p99 %.4f, FA val %.4f test %.4f; "
            "positive control curve %s",
            k,
            thrs[k],
            clean[k]["test"]["median"],
            clean[k]["test"]["p99"],
            clean[k]["false_alarm_val"],
            clean[k]["false_alarm_test"],
            pos[k]["curve"],
        )
    log.info("C1 %s, clean channels %s", c1_ok, chan_ok)

    seeds = [0] if args.smoke else SEEDS
    arms = ["A1", "A2-R1"] if args.smoke else ARMS
    frame_relays = [1] if args.smoke else RELAYS
    eps_grid = [0.5] if args.smoke else EPS_GRID
    common = {"eps_grid": eps_grid, "scales": scales}
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
        "sources": SOURCES + ["random"],
        "n_eval": int(len(sh.ctx.y)),
        "n_steps": sh.ctx.n_steps,
        "step_frac": sh.ctx.step_frac,
        "fa_target": FA_TARGET,
        "representation": "raw (before imputation), PA:Z NaN rule of the map",
        "scores": {
            SCORES[0]: f"|residual| / s_c, s_c = {MAD_CONSISTENCY} x MAD of the clean train "
            "residual on rows where the relay reports the channel (all rows if never "
            f"reported), floored at {MAD_FLOOR} x train std (primary; amendment 2)",
            SCORES[1]: "|residual| / np.nanstd of the raw train column (pre-registered)",
        },
        "c2_offset_train_std": C2_OFFSET,
        "positive_curve_offsets_train_std": list(POS_CURVE_OFFSETS),
        "c3_margin": C3_MARGIN,
        "stage1_results": str(s2.STAGE1_RESULTS.relative_to(REPO_ROOT)),
        "stage2_frame_results": str(STAGE2_FRAME_RESULTS.relative_to(REPO_ROOT)),
        "resume": str(args.resume) if args.resume else None,
    }
    name = f"consistency_defense_{VARIANT}_q{QUBITS}" + ("_smoke" if args.smoke else "")
    with RunDir(name, config=cfg, seeds=seeds) as run:
        run.write_json("controls_map.json", map_diag)
        extra = {}
        if args.smoke:  # C6: one worker, then two, sized on the first pass's peak RSS
            one = s2.run_pool(run_unit, specs, sh, 1)
            peak = max(u["peak_rss_gb"] for u in one)
            print(f"measured peak RSS of one worker: {peak:.2f} GB", flush=True)
            extra["memory_guard"] = s2.memory_guard(2, peak, "smoke pass 2")
            units = s2.run_pool(run_unit, specs, sh, 2)
            a, b = cell_scores(one), cell_scores(units)
            differ = [str(k) for k in a if a[k] != b.get(k)]
            extra["c6_parallel_invariance"] = {
                "n_cells": len(a),
                "n_differ": len(differ),
                "differ": differ,
                "pids_pass1": sorted({u["pid"] for u in one}),
                "pids_pass2": sorted({u["pid"] for u in units}),
                "peak_rss_gb_pass1": [u["peak_rss_gb"] for u in one],
                "peak_rss_gb_pass2": [u["peak_rss_gb"] for u in units],
            }
            c6_ok = not differ and len(a) == len(b)
        else:
            if args.worker_rss_gb is None:
                raise SystemExit("--worker-rss-gb is required (smoke prints the measured value)")
            extra["memory_guard"] = s2.memory_guard(W, args.worker_rss_gb, "full run")
            done = json.loads(args.resume.read_text())["units"] if args.resume else []
            have = {u["unit"] for u in done}
            todo = [sp for sp in specs if unit_name(sp) not in have]

            def save(res):
                done.append(res)
                run.write_json("results_partial.json", {"units": done})

            s2.run_pool(run_unit, todo, sh, W, on_done=save)
            units, c6_ok = done, True
        cells = [c for u in units for c in u["cells"]]
        n = len(sh.ctx.y)
        per_score, c3 = {}, {}
        for k in SCORES:
            sc = s_clean[k]
            table, frame, c3[k] = metrics(
                cells, k, sc["test"], sc["eval"], thrs[k], clean[k]["test"]
            )
            per_score[f"score_{k}"] = {
                "clean": clean[k],
                "branch": branch(table, eps_grid),
                "detection_table": table,
                "stage2_frame_rows": frame,
                "c3_frame_rows_flagged_within_fa_plus_margin": c3[k],
            }
        anc = anchors(
            cells,
            s2.load_stage1(n, sh.ctx.n_steps),
            load_stage2_frame(n, sh.ctx.n_steps),
        )
        br = per_score[f"score_{SCORES[0]}"]["branch"]
        controls = {
            "C1_control1_reproduced_train": c1_ok,
            "C1_clean_test_sequence_magnitudes_within_control1_gate": chan_ok,
            "C2_positive_control_amended_score": pos[SCORES[0]]["pass"],
            "C3_stage2_rows_flagged_within_fa_plus_0.01_amended_score": c3[SCORES[0]],
            "C4_anchors": anc["pass"] if anc["applied"] else "not applied (smoke)",
            "C6_parallel_invariance": c6_ok if args.smoke else "smoke only",
        }
        results = {
            "config": cfg,
            "controls": controls,
            "primary_score": SCORES[0],
            "s_c_amended_scale": s_c_record,
            "c1_control1_reproduction": c1,
            "c2_positive_control": pos,
            "branch_amended": br,
            "branch_preregistered_train_std": per_score[f"score_{SCORES[1]}"]["branch"],
            **per_score,
            "c4_anchors": anc,
            **extra,
            "units_meta": [
                {k: u[k] for k in ("unit", "zoo_fit_s", "wall_s", "peak_rss_gb", "pid")}
                for u in sorted(units, key=lambda u: u["unit"])
            ],
            "qsvm_gate": {u["unit"]: u["qsvm_gate"] for u in units},
            "cells_meta": [
                {k: v for k, v in c.items() if k not in ("scores", "scores_imputed")} for c in cells
            ],
        }
        run.write_json("results.json", results)
        np.savez_compressed(
            run.path / "scores.npz",
            **{f"clean_{part}|{k}": v for k in SCORES for part, v in s_clean[k].items()},
            **{
                f"{c['label']}|{c['eps']}|{c['source']}|s{c['seed']}|{k}": c["scores"][k]
                for c in cells
                for k in SCORES
            },
        )
        run.metrics = {
            "thresholds": thrs,
            "branch_amended": br["all_arms"]["verdict"],
            "branch_preregistered_train_std": results["branch_preregistered_train_std"]["all_arms"][
                "verdict"
            ],
            "controls": controls,
        }
        log.info("controls: %s", json.dumps(controls))
        log.info("branch: %s", json.dumps(br))
        print(f"-> {run.path}")
        failed = [k for k, v in controls.items() if v is False]
        if failed:
            raise AssertionError(f"blocking controls failed: {failed}")


if __name__ == "__main__":
    main()
