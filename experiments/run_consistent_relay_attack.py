"""E-M1a Stage 2 and E-M1c: physically consistent relay attacks on the binary q8 detectors.

Stage 1 (experiments/run_channel_restricted_attack.py) let a one-relay attacker move every
physical channel of that relay independently. Stage 2 keeps the attacker's own measurements
and makes everything the relay computes from them follow: the attacker controls the 12 phase
quantities (|Va|, |Vb|, |Vc|, |Ia|, |Ib|, |Ic| and their 6 angles, PM1..PM6 / PA1..PA6) plus
the frequency channels F and DF, and every derived channel of that relay is RECOMPUTED from
the perturbed phasors each step; logs, status flags and the other relays stay fixed.

Derived map (fit on TRAIN only, per relay; control 1 measures it against the dataset):
    a = exp(2j pi / 3); S0 = (A+B+C)/3, S1 = (A + aB + a^2 C)/3, S2 = (A + a^2 B + a C)/3
    PM7..PM12 = |S|, PA7..PA12 = angle(S) in degrees, each the sequence chosen on train and
        reported as 0 (magnitude AND angle) when |S| is at or under a fitted deadband t
    PA:Z  = exp(c_r) |V1 / I1|           (NaN when I1 == 0, which the pipeline imputes)
    PA:ZH = wrap(angle(V1 / I1) + o_r)   (RADIANS; 0 when I1 == 0)
The assignment of each sequence channel is chosen on the rows where the dataset reports it
(magnitude relative error, the angle must agree); scoring over all rows would be decided by
the deadband zeros, which are 98% of PM8/PM9. Rows whose relay channels carry a NaN (exactly
the rows with I1 == 0, where the dataset's PA:Z is NaN) cannot be attacked consistently: the
relay is left untouched on them and the count is reported.

Attack: L-inf PGD over the controlled channels in units of each channel's raw TRAIN std
(scaler.scale_, the unit Stage 1 used), eps in {0.1, 0.25, 0.5}; a controlled magnitude is
kept >= min(0, clean value). Each step maps controlled -> derived -> full raw row -> the frozen
pipeline (median impute -> StandardScaler -> PCA(8), fit on train), whose delta map is exact:
    Z = Z0 + ((x - x0) / s) @ W.T,     g_x = g_Z @ W / s
and the gradient on the controlled channels is g_x[ctrl] + J^T g_x[derived], the vector-
Jacobian product of the derived map taken with torch autograd (float64). Models, seeds,
eval subset, PGD steps/step_frac, white-box assignment and the MLP-surrogate transfer are
Stage 1's, imported from it. Each (relay, seed) unit fits its own zoo and runs in its own
process; its RNG streams depend on (seed, relay, mode) only.

Blocking controls (each raises): 1 derived map vs the dataset on train (sequence magnitudes
median |err| / std <= 0.01, PM8/PM9 <= 0.15; impedance log-ratio IQR <= 0.06; angles recorded);
2 Jacobian (the PGD's own VJP path, unit cotangents) vs central finite differences of an
independent numpy implementation, <= 1e-4 relative, 3 rows x 14 channels; 3 untouched
channels move < 1e-9 raw, controlled channels stay in their box; 4 every perturbed row
satisfies the numpy map (< 1e-9 in std units); 5 random signs in the same consistent box do
less damage than PGD per white-box family (seed means, checked after results are written);
6a the map-off path runs (smoke); 6b with Stage 1's A2 mask and Stage 1's RNG stream replayed,
Stage 1's seed-0 eps-0.5 QSVM and RBF-SVM attacked F1 is reproduced within 0.01 (anchor);
7 smoke under 1 and 2 workers gives identical attacked F1 in every cell.

Angle unit (pre-run amendment, decision log 2026-11-16T06:24Z and addendum 06:33Z):
--angle-unit marginal (the registered arm) budgets each phase angle in its marginal train std
(about 100 deg). --angle-unit frame budgets the six phase angles of the relay in the train std
of wrap(PA_k - angle(V1)) after removing its circular mean, V1 the clean positive-sequence
voltage, over all train rows (the definition and values logged in the amendment); magnitudes,
F and DF keep scaler.scale_. Under frame a controlled angle is reported as wrap(theta0 + delta)
in [-180, 180), as a PMU reports it; the box stays on delta, the gradient passes the wrap as
the identity and control 3 measures the box on wrap(x - x0). Every cell also counts the rows
whose recomputed sequence quantities cross a fitted deadband relative to the clean row.

Joint mode (E-M1c, decision log 2026-11-16T07:22Z): --joint attacks every relay in --relays at
once (default all four, 4 x 14 = 56 controlled channels), under either angle unit. Each
relay's block of v is that relay's own AttackSpace, so its derived channels follow its own map
with the arm's angle unit (and the wrap under frame), floor and deadband, and logs and status
flags stay fixed; the maps are independent, so the Jacobian is block-diagonal and the gradient
is each relay's VJP on its block. A unit is one seed. Every white-box source's PGD set is also
scored on every family (kind transfer_<src>). Read: control 5 per (family, eps) cell; per
tree family (rf, xgb, lgbm) and eps, transfer damage minus random damage (= random F1 -
transfer F1, seed means) -> noise (< 0.05 everywhere) / survives (>= 0.05 for 2 of 3 at eps
0.5) / intermediate, for the MLP source (the published protocol) and for each source; the
MLP->VQC against VQC->MLP drop in run_surrogate_control's published form
(forward_over_reverse) and bounded, (F - R) / (|F| + |R|); Stage 1's A1 arm at matched eps.
Control 2j checks the joint chain-rule gradient against central differences of the RBF-SVM
loss in v (through its kernel sum, see control_joint_gradient). --joint-anchor runs the mode
restricted to R1 at the single-relay arm's seeds and T (ARM_RESULTS: the post-fix frame
rerun, or the post-fix registered marginal rerun) and requires its R1 cells bit-identical.
A failed unit no longer discards finished ones: they are in results_partial.json first.

E-NS8(a) (decision log 2026-11-16T09:19Z): --joint --angle-unit marginal is the factorial's
new cell (map enforced, angles in their marginal train std). Its full run also writes
factorial.json: E = random F1 - MLP-transfer F1 (seed means) per tree and eps for Stage 1's
A1, this run and E-M1c's frame joint run; the consistency effect E(A1) - E(cons, marginal),
the unit effect E(cons, marginal) - E(cons, frame), and the branch at eps 0.5 over the trees
with E(A1) >= 0.05: PHYSICS (E(cons, marginal) < 0.05 for 2) / UNIT (>= 0.05 for 2) / MIXED.

R4-06 (decision log 2026-11-16T11:07Z, part A): --joint --angle-unit frame --magnitude-unit
physical budgets each relay's phase-voltage magnitudes PM1..PM3 in their train std over the rows
where all three of that relay's phase voltages exceed its fitted PM7 deadband (outage rows
leave the unit); currents, F and DF keep scaler.scale_, angles the frame unit. The units are
logged in volts beside scaler.scale_ and saved in the run config. The run is read as E-M1c is
(control 5 per cell, MLP-transfer minus random per tree, the asymmetry) and writes
branch_magnitude_unit: SURVIVES when the QSVM's white-box F1 is under the registered floor
0.415 at every eps, BUDGET-BOUND otherwise, beside E-M1c's cells. --magnitude-unit scale-equal
with --joint-anchor sets the unit to scaler.scale_ through the same path; its R1 cells must
reproduce the frame arm exactly.

Run (thread budget is W x T; --threads is read before numpy loads, in every worker):
    python -m experiments.run_consistent_relay_attack --joint --smoke --angle-unit frame --threads 2
    python -m experiments.run_consistent_relay_attack --joint-anchor --angle-unit frame \
        --threads 4 --worker-rss-gb 1.2
    python -m experiments.run_consistent_relay_attack --joint --angle-unit frame --workers 3 \
        --threads 4 --worker-rss-gb 1.4
    python -m experiments.run_consistent_relay_attack --joint --smoke --angle-unit marginal \
        --threads 2
    python -m experiments.run_consistent_relay_attack --joint-anchor --angle-unit marginal \
        --threads 4 --worker-rss-gb 1.2
    python -m experiments.run_consistent_relay_attack --joint --angle-unit marginal --workers 2 \
        --threads 4 --worker-rss-gb 1.2
    python -m experiments.run_consistent_relay_attack --joint --smoke --angle-unit frame \
        --magnitude-unit physical --threads 4
    python -m experiments.run_consistent_relay_attack --joint-anchor --angle-unit frame \
        --magnitude-unit scale-equal --workers 2 --threads 4 --worker-rss-gb 1.2
    python -m experiments.run_consistent_relay_attack --joint --angle-unit frame \
        --magnitude-unit physical --workers 2 --threads 4 --worker-rss-gb 1.2
    python -m experiments.run_consistent_relay_attack --smoke --threads 4
    python -m experiments.run_consistent_relay_attack --smoke --angle-unit frame --threads 2
    python -m experiments.run_consistent_relay_attack --anchor-only --threads 8
    python -m experiments.run_consistent_relay_attack --workers 6 --threads 4 --worker-rss-gb 1.5
    python -m experiments.run_consistent_relay_attack --angle-unit frame --workers 6 --threads 4 \
        --worker-rss-gb 1.2
"""

from __future__ import annotations

import os
import sys


def _early_threads(argv: list[str]) -> str | None:
    """--threads, read BEFORE numpy/torch load so the BLAS/OpenMP pools obey it.

    A spawned worker re-imports this module with the parent's argv, so it gets the same T.
    """
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
import copy  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from experiments.run_adversarial import FittedZoo, _f1  # noqa: E402
from experiments.run_channel_restricted_attack import (  # noqa: E402
    ENCODING,
    EPS_GRID,
    FAMILIES,
    QUBITS,
    RELAYS,
    SEEDS,
    VARIANT,
    WHITEBOX,
    Context,
    _ms,
    aggregate,
    build_context,
    control_gradient,
    make_whitebox,
)
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.features.reduce import build_preprocessor  # noqa: E402
from qgridbench.models.classical.zoo import stratified_cap  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger  # noqa: E402
from qgridbench.utils.seeding import spawn_rng  # noqa: E402

log = get_logger(__name__)

STAGE1_RESULTS = (
    REPO_ROOT / "results" / "runs" / "20261116T052503Z_channel_restricted_attack_binary_q8"
)
STAGE1_ARMS = ["A0", "A1", "A2-R1", "A2-R2", "A2-R3", "A2-R4"]  # Stage 1's RNG consumption order
STAGE1_SOURCES = ["mlp", "qsvm", "rbf_svm", "vqc"]
A = np.exp(2j * np.pi / 3)
SEQ_CHANNELS = (7, 8, 9, 10, 11, 12)  # PM/PA k: 7-9 voltage, 10-12 current
SEQ_NAMES = ("zero", "pos", "neg")
# control 1 gates (pre-registration and brief): median |err| / train std, impedance log IQR
SEQ_MAG_TOL = {7: 0.01, 8: 0.15, 9: 0.15, 10: 0.01, 11: 0.01, 12: 0.01}
Z_LOG_IQR_TOL = 0.06
JAC_TOL, FD_REL_H = 1e-4, 1e-6
MOVE_TOL, MAP_TOL = 1e-9, 1e-9
ANCHOR_EPS, ANCHOR_TOL = 0.5, 0.01
GPU_GATE_TOL = 1e-6  # max |v_cuda - v_cpu| in units of the eps box
I1_RESIDUE_A = 1e-9  # a nonzero |I1| below this is float residue, not a current
FLOOR_SNAP_V = 1e-12  # v within this (std units) of a magnitude's floor is on it (rounding)
MEM_FRACTION = 0.8
MODES = ("consistent", "map_off", "a2_mask")
ANGLE_UNITS = ("marginal", "frame")
# PM1..PM3 budget unit: scaler.scale_ (every earlier arm), the R4-06 physical unit, or
# scaler.scale_ written through the physical arm's path (its anchor)
MAGNITUDE_UNITS = ("scale", "physical", "scale-equal")
MAG_UNIT_FLOOR = 0.415  # R4-06 as registered: SURVIVES iff the QSVM is under it at every eps
# frame units (deg) from the lead's independent probe (tmp/review_shihav_20261116/
# probe_stage2_review.py, rounded to 0.01): PA1..PA6 per relay; the fit must match within 1%
FRAME_UNIT_REFERENCE = {
    1: [0.31, 4.07, 4.06, 22.23, 23.33, 21.51],
    2: [2.04, 10.24, 10.24, 26.14, 19.53, 21.00],
    3: [2.18, 10.27, 10.27, 26.05, 19.55, 21.55],
    4: [1.38, 3.01, 3.13, 21.91, 23.92, 21.34],
}
FRAME_REL_TOL, FRAME_ROUND = 0.01, 0.005  # the reference is rounded to 2 decimals


# --------------------------------------------------------------------------- derived map


@dataclass
class RelayMap:
    """Column layout and train-fitted constants of one relay's derived map."""

    relay: int
    ctrl_names: list[str]  # PM1..PM6 magnitudes, PA1..PA6 angles (deg), F, DF
    der_names: list[str]  # PM7..PM12, PA7..PA12, PA:Z, PA:ZH
    ctrl_idx: np.ndarray
    der_idx: np.ndarray
    assign: dict[int, str]  # sequence channel k -> zero | pos | neg
    deadband: dict[int, float]  # |S| <= t is reported as 0
    log_ratio: float  # c_r: PA:Z = exp(c_r) |V1 / I1|
    zh_offset: float  # o_r, radians
    z_impute: float  # the frozen imputer's PA:Z median (what a NaN becomes)
    control1: dict = field(default_factory=dict)
    frame_unit: np.ndarray | None = None  # (6,) deg, PA1..PA6 budget unit under frame
    mag_unit: np.ndarray | None = None  # (3,) V, PM1..PM3 budget unit unless scaler.scale_


def fit_magnitude_unit(
    Xtr: np.ndarray, rm: RelayMap, s: np.ndarray, kind: str
) -> tuple[np.ndarray, dict]:
    """PM1..PM3 budget unit in volts, on TRAIN.

    physical (R4-06): the std of each phase-voltage magnitude over the rows where all three
    phase voltages exceed the relay's fitted PM7 deadband, so outage rows leave the unit.
    scale-equal: scaler.scale_, written through the same path (the anchor's mode)."""
    if kind not in ("physical", "scale-equal"):
        raise ValueError(f"magnitude unit {kind!r} has no fitted unit")
    default = s[rm.ctrl_idx[:3]].astype(np.float64)
    v = Xtr[:, rm.ctrl_idx[:3]]
    t = rm.deadband[7]
    rows = (v > t).all(axis=1)  # a NaN voltage compares False, so its row is out as well
    if rows.sum() < 2:
        raise AssertionError(f"R{rm.relay}: {rows.sum()} train rows have every voltage over {t}")
    phys = v[rows].std(axis=0)
    if not np.all(phys > 0):
        raise AssertionError(f"R{rm.relay}: a physical voltage unit is not positive: {phys}")
    nan_any = np.isnan(v).any(axis=1)
    diag = {
        "kind": kind,
        "definition": "std (ddof 0) of PM1..PM3 over train rows where all three exceed the "
        "relay's PM7 deadband",
        "pm7_assigned": rm.assign[7],
        "pm7_deadband_V": t,
        "physical_unit_V": phys.tolist(),
        "scaler_scale_V": default.tolist(),
        "physical_over_scale": (phys / default).tolist(),
        "nanstd_all_rows_V": np.nanstd(v, axis=0).tolist(),
        "n_train_rows": int(len(v)),
        "n_rows_used": int(rows.sum()),
        "n_rows_any_voltage_nan": int(nan_any.sum()),
        "n_rows_all_voltages_at_or_under_deadband": int((~nan_any & (v <= t).all(axis=1)).sum()),
        "n_rows_some_voltages_at_or_under_deadband": int(
            (~nan_any & ~rows & ~(v <= t).all(axis=1)).sum()
        ),
    }
    return (phys if kind == "physical" else default), diag


def fit_frame_unit(Xtr: np.ndarray, rm: RelayMap) -> tuple[np.ndarray, dict]:
    """Std of each phase angle's wrapped, circular-centred deviation from angle(V1), on TRAIN.

    All train rows, as the amendment defines and the reference values were measured. The
    subset with every relay channel finite and |V1| > 0 is recorded beside it, not used.
    """
    xc = Xtr[:, rm.ctrl_idx]
    if not np.isfinite(xc[:, :12]).all():
        raise AssertionError(f"R{rm.relay}: NaN in a train phase quantity")
    sv, _ = sequences_np(xc)
    a1 = np.rad2deg(np.angle(sv["pos"]))

    def unit(rows):
        out = []
        for k in range(6):
            dev = wrap_deg(xc[rows, 6 + k] - a1[rows])
            cm = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(dev)))))
            out.append(float(np.std(wrap_deg(dev - cm))))
        return np.array(out)

    u = unit(np.ones(len(xc), bool))
    rel_ok = np.isfinite(Xtr[:, np.concatenate([rm.ctrl_idx, rm.der_idx])]).all(axis=1)
    sub = rel_ok & (np.abs(sv["pos"]) > 0)
    ref = np.array(FRAME_UNIT_REFERENCE[rm.relay])
    diag = {
        "frame_unit_deg": u.tolist(),
        "reference_deg": ref.tolist(),
        "max_abs_diff_vs_reference": float(np.abs(u - ref).max()),
        "n_train_rows": int(len(xc)),
        "subset_finite_relay_and_V1_nonzero_deg": unit(sub).tolist(),
        "n_subset_rows": int(sub.sum()),
    }
    if np.any(np.abs(u - ref) > np.maximum(FRAME_REL_TOL * ref, FRAME_ROUND)):
        raise AssertionError(f"R{rm.relay}: frame unit {u} does not match reference {ref}")
    return u, diag


def relay_columns(columns: list[str], r: int):
    ctrl = [f"R{r}-PM{k}:{'V' if k <= 3 else 'I'}" for k in range(1, 7)]
    ctrl += [f"R{r}-PA{k}:{'V' if k <= 3 else 'I'}H" for k in range(1, 7)]
    ctrl += [f"R{r}:F", f"R{r}:DF"]
    der = [f"R{r}-PM{k}:{'V' if k <= 9 else 'I'}" for k in SEQ_CHANNELS]
    der += [f"R{r}-PA{k}:{'V' if k <= 9 else 'I'}H" for k in SEQ_CHANNELS]
    der += [f"R{r}-PA:Z", f"R{r}-PA:ZH"]
    pos = {c: i for i, c in enumerate(columns)}
    missing = [c for c in ctrl + der if c not in pos]
    if missing:
        raise KeyError(f"relay R{r}: columns not in the feature list: {missing}")
    rest = [c for c in columns if c.startswith((f"R{r}-", f"R{r}:")) and c not in ctrl + der]
    if rest != [f"R{r}:S"]:
        raise AssertionError(f"relay R{r}: unexpected channels outside the map: {rest}")
    return ctrl, der, np.array([pos[c] for c in ctrl]), np.array([pos[c] for c in der])


def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def wrap_rad(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def sequences_np(xc: np.ndarray):
    """Symmetrical components of the voltage and current phasor triples (numpy, complex)."""
    P = xc[:, 0:6] * np.exp(1j * np.deg2rad(xc[:, 6:12]))

    def seq(a_, b_, c_):
        return {
            "zero": (a_ + b_ + c_) / 3,
            "pos": (a_ + A * b_ + A * A * c_) / 3,
            "neg": (a_ + A * A * b_ + A * c_) / 3,
        }

    return seq(*P[:, 0:3].T), seq(*P[:, 3:6].T)


def derived_np(xc: np.ndarray, rm: RelayMap) -> np.ndarray:
    """Reference implementation (numpy complex arithmetic); controls 2 and 4 check torch on it."""
    sv, si = sequences_np(xc)
    mags, angs = [], []
    for k in SEQ_CHANNELS:
        S = (sv if k <= 9 else si)[rm.assign[k]]
        on = np.abs(S) > rm.deadband[k]
        mags.append(np.where(on, np.abs(S), 0.0))
        angs.append(np.where(on, np.rad2deg(np.angle(S)), 0.0))
    v1, i1 = sv["pos"], si["pos"]
    zero = np.abs(i1) == 0
    ratio = v1 / np.where(zero, 1.0, i1)
    z = np.where(zero, rm.z_impute, np.exp(rm.log_ratio) * np.abs(ratio))
    zh = np.where(zero, 0.0, wrap_rad(np.angle(ratio) + rm.zh_offset))
    return np.column_stack(mags + angs + [z, zh])


def derived_torch(xc: torch.Tensor, rm: RelayMap) -> torch.Tensor:
    """The same map in real torch arithmetic, differentiable (the PGD's path)."""
    th = torch.deg2rad(xc[:, 6:12])
    re, im = xc[:, 0:6] * torch.cos(th), xc[:, 0:6] * torch.sin(th)
    ar, ai = -0.5, np.sqrt(3.0) / 2  # a = exp(2j pi / 3); a^2 = conj(a)

    def rot(r_, i_, conj):  # multiply by a (conj False) or a^2 (conj True)
        s = -ai if conj else ai
        return r_ * ar - i_ * s, r_ * s + i_ * ar

    def seq(o):
        (r0, i0), (r1, i1), (r2, i2) = [(re[:, o + j], im[:, o + j]) for j in range(3)]
        b1, c1 = rot(r1, i1, False), rot(r2, i2, True)
        b2, c2 = rot(r1, i1, True), rot(r2, i2, False)
        return {
            "zero": ((r0 + r1 + r2) / 3, (i0 + i1 + i2) / 3),
            "pos": ((r0 + b1[0] + c1[0]) / 3, (i0 + b1[1] + c1[1]) / 3),
            "neg": ((r0 + b2[0] + c2[0]) / 3, (i0 + b2[1] + c2[1]) / 3),
        }

    sv, si = seq(0), seq(3)
    mags, angs = [], []
    for k in SEQ_CHANNELS:
        sr, s_i = (sv if k <= 9 else si)[rm.assign[k]]
        m2 = sr * sr + s_i * s_i
        on = m2 > rm.deadband[k] ** 2
        sr_s, si_s = torch.where(on, sr, 1.0), torch.where(on, s_i, 0.0)  # NaN-free grads
        mags.append(torch.where(on, torch.sqrt(sr_s * sr_s + si_s * si_s), 0.0))
        angs.append(torch.where(on, torch.rad2deg(torch.atan2(si_s, sr_s)), 0.0))
    (vr, vi), (ir, ii) = sv["pos"], si["pos"]
    i2 = ir * ir + ii * ii
    zero = i2 == 0
    num_r, num_i = vr * ir + vi * ii, vi * ir - vr * ii  # V1 * conj(I1)
    den = torch.where(zero, 1.0, i2)
    num_r_s, num_i_s = torch.where(zero, 1.0, num_r), torch.where(zero, 0.0, num_i)
    mag = torch.sqrt(num_r_s * num_r_s + num_i_s * num_i_s) / den  # |V1 / I1|
    z = torch.where(zero, rm.z_impute, np.exp(rm.log_ratio) * mag)
    zh = torch.atan2(num_i_s, num_r_s) + rm.zh_offset
    zh = torch.where(zero, 0.0, torch.remainder(zh + np.pi, 2 * np.pi) - np.pi)
    return torch.stack(mags + angs + [z, zh], dim=1)


def derived_vjp(xc: np.ndarray, cot: np.ndarray, rm: RelayMap, device: str):
    """(map(xc), J(xc)^T cot) via one autograd pass; numpy in, numpy out, float64."""
    xt = torch.tensor(xc, dtype=torch.float64, device=device, requires_grad=True)
    out = derived_torch(xt, rm)
    (g,) = torch.autograd.grad(out, xt, grad_outputs=torch.as_tensor(cot, device=device))
    return out.detach().cpu().numpy(), g.cpu().numpy()


def fit_deadband(m: np.ndarray, on: np.ndarray):
    """Threshold t minimizing train disagreements between (|S| > t) and (reported != 0)."""
    o = np.argsort(m)
    ms, ons = m[o], on[o]
    below_on = np.concatenate([[0], np.cumsum(ons)])  # reported rows predicted 0
    above_off = (~ons).sum() - np.concatenate([[0], np.cumsum(~ons)])  # zero rows predicted on
    err = below_on + above_off
    i = int(np.argmin(err))
    if i == 0:
        t = 0.0
    elif i == len(ms):
        t = float(ms[-1])
    else:
        t = float((ms[i - 1] + ms[i]) / 2)
    return t, int(err[i])


def fit_relay_map(Xtr: np.ndarray, columns: list[str], r: int, z_impute: float) -> RelayMap:
    """Fit assignment, deadbands, impedance ratio and angle offset on TRAIN; control 1."""
    ctrl, der, ci, di = relay_columns(columns, r)
    xc = Xtr[:, ci]
    finite_c = np.isfinite(xc).all(axis=1)
    sv, si = sequences_np(np.where(finite_c[:, None], xc, 0.0))
    assign, deadband, c1 = {}, {}, {"n_train_rows_phase_finite": int(finite_c.sum())}
    for j, k in enumerate(SEQ_CHANNELS):
        cands = sv if k <= 9 else si
        pm, pa = Xtr[:, di[j]], Xtr[:, di[6 + j]]
        ok = finite_c & np.isfinite(pm) & np.isfinite(pa)
        rep = ok & (pm > 0)
        mag_err = {
            n: float(np.median(np.abs(pm[rep] - np.abs(cands[n][rep])) / pm[rep]))
            for n in SEQ_NAMES
        }
        ang_err = {
            n: float(np.median(np.abs(wrap_deg(pa[rep] - np.rad2deg(np.angle(cands[n][rep]))))))
            for n in SEQ_NAMES
        }
        best = min(mag_err, key=mag_err.get)
        if min(ang_err, key=ang_err.get) != best:
            raise AssertionError(
                f"control 1: R{r} PM{k}/PA{k} magnitude picks {best}, angle picks "
                f"{min(ang_err, key=ang_err.get)}: {mag_err} {ang_err}"
            )
        S = cands[best]
        t, n_mis = fit_deadband(np.abs(S[ok]), pm[ok] > 0)
        assign[k], deadband[k] = best, t
        on = np.abs(S) > t
        d_pm = np.where(on, np.abs(S), 0.0)
        d_pa = np.where(on, np.rad2deg(np.angle(S)), 0.0)
        sd_pm, sd_pa = float(np.nanstd(pm)), float(np.nanstd(pa))
        e_pm = np.abs(pm[ok] - d_pm[ok]) / sd_pm
        e_pa = np.abs(wrap_deg(pa[ok] - d_pa[ok]))
        # the literal all-rows rule, for the record: decided by the deadband zeros
        all_rows = {
            n: float(np.median(np.abs(np.abs(cands[n][ok]) - pm[ok])) / sd_pm) for n in SEQ_NAMES
        }
        c1[f"PM{k}/PA{k}"] = {
            "assigned": best,
            "reported_rows_mag_rel_err": mag_err,
            "reported_rows_angle_err_deg": ang_err,
            "all_rows_rule_choice": min(all_rows, key=all_rows.get),
            "all_rows_rule_err_over_std": all_rows,
            "deadband": t,
            "deadband_disagreements": n_mis,
            "deadband_fit": "midpoint of the |S| gap minimizing train disagreements between "
            "|S| > t and a reported value > 0",
            "n_reported_zero": int((pm[ok] == 0).sum()),
            "n_reported_negative": int((pm[ok] < 0).sum()),
            "max_abs_S_reported_zero": float(np.abs(S[ok][pm[ok] == 0]).max(initial=0.0)),
            "min_abs_S_reported_positive": float(np.abs(S[ok][pm[ok] > 0]).min()),
            "smallest_abs_S_reported_positive": np.sort(np.abs(S[ok][pm[ok] > 0]))[:3].tolist(),
            "frac_reported_nonzero": float(np.mean(pm[ok] > 0)),
            "mag_median_err_over_std": float(np.median(e_pm)),
            "mag_p99_err_over_std": float(np.percentile(e_pm, 99)),
            "mag_tol": SEQ_MAG_TOL[k],
            "angle_median_err_deg": float(np.median(e_pa)),
            "angle_p99_err_deg": float(np.percentile(e_pa, 99)),
            "angle_median_err_over_std": float(np.median(e_pa) / sd_pa),
        }
        if np.median(e_pm) > SEQ_MAG_TOL[k]:
            raise AssertionError(
                f"control 1: R{r} PM{k} median |err|/std {np.median(e_pm):.4f} > {SEQ_MAG_TOL[k]}"
            )
    z, zh = Xtr[:, di[12]], Xtr[:, di[13]]
    v1, i1 = sv["pos"], si["pos"]
    okz = finite_c & np.isfinite(z) & (z > 0) & (np.abs(i1) > 0)
    lr = np.log(z[okz]) - np.log(np.abs(v1[okz] / i1[okz]))
    q25, q50, q75 = np.percentile(lr, [25, 50, 75])
    okh = finite_c & np.isfinite(zh) & (np.abs(i1) > 0)
    dh = wrap_rad(zh[okh] - np.angle(v1[okh] / i1[okh]))
    off = float(np.median(dh))
    e_zh = np.abs(wrap_rad(dh - off))
    d_z = np.exp(q50) * np.abs(v1[okz] / i1[okz])
    c1["PA:Z"] = {
        "log_ratio_median": float(q50),
        "ratio": float(np.exp(q50)),
        "log_ratio_iqr": float(q75 - q25),
        "iqr_tol": Z_LOG_IQR_TOL,
        "median_err_over_std": float(np.median(np.abs(z[okz] - d_z)) / np.nanstd(z)),
        "p99_err_over_std": float(np.percentile(np.abs(z[okz] - d_z), 99) / np.nanstd(z)),
        "n_nan_train": int(np.isnan(z).sum()),
        "n_nan_with_I1_zero": int((np.isnan(z) & (np.abs(i1) == 0)).sum()),
    }
    c1["PA:ZH"] = {
        "units": "radians (PA:ZH spans [-pi, pi])",
        "offset_rad": off,
        "median_err_rad": float(np.median(e_zh)),
        "p99_err_rad": float(np.percentile(e_zh, 99)),
        "median_err_over_std": float(np.median(e_zh) / np.nanstd(zh)),
        "value_on_I1_zero_rows": sorted({float(v) for v in zh[np.abs(i1) == 0]}),
    }
    if q75 - q25 > Z_LOG_IQR_TOL:
        raise AssertionError(f"control 1: R{r} PA:Z log-ratio IQR {q75 - q25:.4f} > tol")
    if c1["PA:Z"]["n_nan_train"] != c1["PA:Z"]["n_nan_with_I1_zero"]:
        raise AssertionError(f"control 1: R{r} PA:Z is NaN on rows where I1 != 0: {c1['PA:Z']}")
    return RelayMap(r, ctrl, der, ci, di, assign, deadband, float(q50), off, z_impute, c1)


def control_jacobian(rm: RelayMap, xc_rows: np.ndarray, s_ctrl: np.ndarray, n_rows=3) -> dict:
    """Control 2: the PGD's VJP path (unit cotangents) against central FD of the numpy map."""
    keep = []
    for i, x in enumerate(xc_rows):
        sv, si = sequences_np(x[None])
        clear = all(
            abs(abs((sv if k <= 9 else si)[rm.assign[k]][0]) - rm.deadband[k])
            > 1e-3 * max(rm.deadband[k], 1.0)
            for k in SEQ_CHANNELS
        )
        d = derived_np(x[None], rm)[0]
        clear &= np.all(np.abs(np.abs(d[6:12]) - 180.0) > 1e-3) and abs(abs(d[13]) - np.pi) > 1e-5
        clear &= abs(si["pos"][0]) > 0
        if clear:
            keep.append(i)
        if len(keep) == n_rows:
            break
    if len(keep) < n_rows:
        raise AssertionError(f"control 2: R{rm.relay} only {len(keep)} rows clear of breakpoints")
    worst, per_row = 0.0, []
    nd = len(rm.der_names)
    for i in keep:
        x = xc_rows[i : i + 1]
        J = np.stack([derived_vjp(x, np.eye(nd)[o][None], rm, "cpu")[1][0] for o in range(nd)])
        FD = np.empty_like(J)
        for j in range(x.shape[1]):
            h = FD_REL_H * s_ctrl[j]
            up, dn = x.copy(), x.copy()
            up[0, j] += h
            dn[0, j] -= h
            FD[:, j] = (derived_np(up, rm)[0] - derived_np(dn, rm)[0]) / (2 * h)
        scale = np.maximum(np.abs(FD).max(axis=1, keepdims=True), 1e-300)
        live = np.abs(FD).max(axis=1) > 0  # outputs in the deadband are identically 0
        rel = (np.abs(J - FD) / scale)[live]
        if np.any(np.abs(J[~live]) > 0):
            raise AssertionError(f"control 2: R{rm.relay} nonzero gradient on a deadband output")
        per_row.append({"row": int(i), "max_rel_err": float(rel.max()), "n_live": int(live.sum())})
        worst = max(worst, float(rel.max()))
    out = {"max_rel_err": worst, "rows": per_row, "h_over_std": FD_REL_H, "tol": JAC_TOL}
    if worst > JAC_TOL:
        raise AssertionError(f"control 2: R{rm.relay} Jacobian off by {worst:.2e} > {JAC_TOL}")
    return out


# --------------------------------------------------------------------------- attack space


@dataclass
class Shared:
    """Everything a worker needs, pickled once per worker."""

    ctx: Context
    X0raw: np.ndarray  # (n, 128) raw eval rows
    X0imp: np.ndarray  # (n, 128) after the frozen imputer
    s: np.ndarray  # (128,) raw train std (scaler.scale_)
    maps: dict[int, RelayMap]
    stage1: dict | None  # Stage 1 records keyed by (seed, arm, eps, family, kind)
    threads: int
    angle_unit: str  # marginal (registered) | frame (amendment)


class AttackSpace:
    """One attacker: which raw channels it moves (in std units v), and whether derived follow."""

    def __init__(self, mode: str, rm: RelayMap, sh: Shared, device: str):
        self.mode, self.rm, self.sh, self.device = mode, rm, sh, device
        if mode == "a2_mask":
            self.ctrl = np.flatnonzero(sh.ctx.arm_masks[f"A2-R{rm.relay}"])
            expect = np.sort(np.concatenate([rm.ctrl_idx, rm.der_idx]))
            if not np.array_equal(self.ctrl, expect):
                raise AssertionError(f"A2-R{rm.relay} mask is not the relay's 28 map channels")
        else:
            self.ctrl = rm.ctrl_idx
        self.derive = mode == "consistent"
        n = len(sh.X0raw)
        relay_raw = sh.X0raw[:, np.concatenate([rm.ctrl_idx, rm.der_idx])]
        self.n_nan_phase = int(np.isnan(sh.X0raw[:, rm.ctrl_idx[:12]]).any(axis=1).sum())
        self.n_nan_relay = int(np.isnan(relay_raw).any(axis=1).sum())
        # a consistent attack needs every relay channel; elsewhere Stage 1's rule (all rows)
        self.att = np.isfinite(relay_raw).all(axis=1) if self.derive else np.ones(n, bool)
        self.sc = sh.s[self.ctrl].copy()
        if rm.mag_unit is not None and mode != "a2_mask":  # R4-06: PM1..PM3 budget unit (V)
            self.sc[:3] = rm.mag_unit
        self.x0c = sh.X0imp[:, self.ctrl]
        self.mag_cols = np.arange(6) if mode != "a2_mask" else np.array([], int)
        # magnitude floor min(0, clean): in v units, and the exact raw value written there
        # (x0 + v * s at v = -x0 / s leaves a rounding residue, e.g. 7e-15 A, which made
        # |I1| ~ 3e-15 A and PA:Z ~ 1e18 ohm; the floor is written exactly instead)
        m = self.mag_cols
        self.floor_v = -np.maximum(self.x0c[:, m], 0.0) / self.sc[m]
        self.floor_x = np.minimum(self.x0c[:, m], 0.0)
        # frame arm: the six phase angles are budgeted in the positive-sequence frame unit and
        # reported wrapped into [-180, 180); the anchor's A2 mask path stays Stage 1's
        self.frame = sh.angle_unit == "frame" and mode != "a2_mask"
        self.ang_cols = np.arange(6, 12)
        if self.frame:
            self.sc[self.ang_cols] = rm.frame_unit
            a0 = self.x0c[:, self.ang_cols]
            if np.any(a0 < -180.0) or np.any(a0 >= 180.0):
                raise AssertionError(f"R{rm.relay}: a clean phase angle lies outside [-180, 180)")
        if self.derive:  # clean on/off state of each sequence channel, for the crossing count
            self.on0 = self._deadband_state(self.x0c[self.att])

    def _deadband_state(self, xc: np.ndarray) -> np.ndarray:
        sv, si = sequences_np(xc)
        return np.column_stack(
            [
                np.abs((sv if k <= 9 else si)[self.rm.assign[k]]) > self.rm.deadband[k]
                for k in SEQ_CHANNELS
            ]
        )

    def on_floor(self, v: np.ndarray) -> np.ndarray:
        """Magnitudes on their floor. A PGD step back from the floor, (f + d) - d, can land a
        few ulps above f (E-M1c seed 0: v - floor_v = 1.4e-17, written 2.7e-15 A, |I1| 8.9e-16
        A, PA:Z 4e18 ohm); no step is under 0.025 std, so within FLOOR_SNAP_V is the floor."""
        return v[:, self.mag_cols] <= self.floor_v + FLOOR_SNAP_V

    def box(self, eps: float):
        n, c = self.x0c.shape
        lo, hi = np.full((n, c), -eps), np.full((n, c), eps)
        if self.derive:  # a magnitude cannot go below zero (or below a negative clean value)
            m = self.mag_cols
            lo[:, m] = np.maximum(lo[:, m], self.floor_v)
        lo[~self.att], hi[~self.att] = 0.0, 0.0
        return lo, hi

    def rows(self, v: np.ndarray, cot_der: np.ndarray | None = None):
        """Full imputed raw rows for v; with cot_der also J^T cot on the controlled channels."""
        X = self.sh.X0imp.copy()
        X[:, self.ctrl] = self.x0c + v * self.sc
        if self.derive:  # a magnitude on its floor is the floor, exactly
            m = self.mag_cols
            cm = self.ctrl[m]
            X[:, cm] = np.where(self.on_floor(v), self.floor_x, X[:, cm])
        if self.frame:  # reported as a PMU reports it; d wrap / d theta = 1 for the gradient
            ac = self.ctrl[self.ang_cols]
            th = X[:, ac]
            X[:, ac] = np.where((th < -180.0) | (th >= 180.0), wrap_deg(th), th)  # bit-exact else
        vjp = None
        if self.derive:
            a, xc = self.att, X[self.att][:, self.ctrl]
            if cot_der is None:
                with torch.no_grad():
                    xt = torch.as_tensor(xc, dtype=torch.float64, device=self.device)
                    d = derived_torch(xt, self.rm).cpu().numpy()
            else:
                d, g = derived_vjp(xc, cot_der[a], self.rm, self.device)
                vjp = np.zeros_like(v)
                vjp[a] = g
            X[np.ix_(a, self.rm.der_idx)] = d
        return X, vjp

    def Z(self, X: np.ndarray) -> np.ndarray:
        ctx = self.sh.ctx
        return ctx.Z0 + ((X - self.sh.X0imp) / self.sh.s) @ ctx.W.T

    def grad(self, v: np.ndarray, wb, y: np.ndarray) -> np.ndarray:
        """d loss / d v: chain rule through the frozen pipeline and the derived map."""
        ctx, s = self.sh.ctx, self.sh.s
        X, _ = self.rows(v)
        gx = (wb(self.Z(X), y) @ ctx.W) / s  # d loss / d raw x, all 128 channels
        g = gx[:, self.ctrl]
        if self.derive:
            g = g + self.rows(v, cot_der=gx[:, self.rm.der_idx])[1]
        return g * self.sc

    def pgd(self, wb, y, eps, n_steps, step_frac, rng):
        """evasion.pgd's recipe on v: random start (drawn 128 wide, Stage 1's draw), sign steps,
        projection on the box after every step."""
        lo, hi = self.box(eps)
        start = rng.uniform(-1.0, 1.0, size=(len(y), self.sh.X0imp.shape[1]))[:, self.ctrl]
        v = np.clip(start * eps, lo, hi)
        for _ in range(n_steps):
            v = np.clip(v + step_frac * eps * np.sign(self.grad(v, wb, y)), lo, hi)
        return v

    def random_sign(self, eps, rng):
        lo, hi = self.box(eps)
        d = rng.choice([-1.0, 1.0], size=(len(self.x0c), self.sh.X0imp.shape[1]))
        return np.clip(d[:, self.ctrl] * eps, lo, hi)

    def check(self, v: np.ndarray, X: np.ndarray, eps: float, tag: str) -> dict:
        """Controls 3 and 4 on one perturbed set."""
        moved = np.ones(X.shape[1], bool)
        moved[self.ctrl] = False
        if self.derive:
            moved[self.rm.der_idx] = False
        d_other = float(np.abs(X[:, moved] - self.sh.X0imp[:, moved]).max())
        if d_other >= MOVE_TOL:
            raise AssertionError(f"control 3 ({tag}): an untouched channel moved by {d_other:.2e}")
        dx = X[:, self.ctrl] - self.x0c
        if self.frame:  # a wrapped value is not a 360-degree move
            dx[:, self.ang_cols] = wrap_deg(dx[:, self.ang_cols])
        dc = np.abs(dx) / self.sc
        if dc.max() > eps * (1 + 1e-9):
            raise AssertionError(f"control 3 ({tag}): box exceeded, {dc.max():.6f} > {eps}")
        lo, hi = self.box(eps)
        if np.any(v < lo - 1e-12) or np.any(v > hi + 1e-12):
            raise AssertionError(f"control 3 ({tag}): v outside its projected box")
        out = {"untouched_max_abs": d_other, "ctrl_max_over_eps": float(dc.max() / eps)}
        if self.derive:  # regression pin for the R1 s2 control-4 failure (rounding floor)
            m = self.mag_cols
            at = self.on_floor(v)
            if np.any(X[:, self.ctrl[m]][at] != self.floor_x[at]):
                raise AssertionError(f"control 3 ({tag}): a floored magnitude is not exact")
            _, si = sequences_np(X[self.att][:, self.ctrl])
            i1 = np.abs(si["pos"])
            if np.any((i1 > 0) & (i1 < I1_RESIDUE_A)):
                raise AssertionError(
                    f"control 4 ({tag}): |I1| = {i1[(i1 > 0)].min():.2e} A is a rounding residue"
                )
            moved = at & (self.x0c[:, m] != self.floor_x) & self.att[:, None]  # driven there
            out.update(
                {
                    "n_rows_all_currents_driven_to_floor": int(moved[:, 3:6].all(axis=1).sum()),
                    "n_rows_all_voltages_driven_to_floor": int(moved[:, 0:3].all(axis=1).sum()),
                    "n_magnitudes_driven_to_floor": int(moved.sum()),
                    "min_nonzero_abs_I1": float(i1[i1 > 0].min()) if (i1 > 0).any() else None,
                }
            )
        if self.mode != "a2_mask":
            ang = X[:, self.ctrl[6:12]]
            out["ctrl_angle_values_outside_180"] = int((np.abs(ang) > 180.0).sum())
        if self.frame:
            raw = self.x0c[:, self.ang_cols] + v[:, self.ang_cols] * self.sc[self.ang_cols]
            out["ctrl_angle_values_wrapped"] = int((ang != raw).sum())
            out["ctrl_angle_values_outside_m180_180"] = int(((ang < -180) | (ang >= 180)).sum())
            if out["ctrl_angle_values_outside_m180_180"]:
                raise AssertionError(f"control 3 ({tag}): a reported angle is outside [-180, 180)")
        if self.derive:
            a, di = self.att, self.rm.der_idx
            on = self._deadband_state(X[a][:, self.ctrl])
            cross = on != self.on0
            out["deadband_crossing"] = {
                "n_rows_any": int(cross.any(axis=1).sum()),
                "n_attackable_rows": int(a.sum()),
                "by_channel_off_to_on": {
                    f"PM{k}": int((cross[:, j] & on[:, j]).sum())
                    for j, k in enumerate(SEQ_CHANNELS)
                },
                "by_channel_on_to_off": {
                    f"PM{k}": int((cross[:, j] & ~on[:, j]).sum())
                    for j, k in enumerate(SEQ_CHANNELS)
                },
            }
            ref = derived_np(X[a][:, self.ctrl], self.rm)
            err = float((np.abs(X[a][:, di] - ref) / self.sh.s[di]).max())
            if not err < MAP_TOL:
                raise AssertionError(f"control 4 ({tag}): map violated by {err:.2e} std")
            du = np.abs(X[:, di] - self.sh.X0imp[:, di]) / self.sh.s[di]
            out.update(
                {
                    "map_max_err_over_std": err,
                    "derived_du_max": float(du.max()),
                    "derived_du_p99": float(np.percentile(du, 99)),
                    "derived_du_max_by_channel": {
                        n: float(x) for n, x in zip(self.rm.der_names, du.max(axis=0), strict=True)
                    },
                }
            )
        return out


# --------------------------------------------------------------------------- worker


_SHARED: Shared | None = None


def _peak_rss_gb() -> float:
    mi = psutil.Process().memory_info()
    if sys.platform == "win32":
        return mi.peak_wset / 2**30
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _init_worker(sh: Shared):
    global _SHARED
    want = str(sh.threads)
    got = {v: os.environ.get(v) for v in THREAD_VARS}
    if any(x != want for x in got.values()) or torch.get_num_threads() != sh.threads:
        raise AssertionError(
            f"worker {os.getpid()}: thread caps {got}, torch {torch.get_num_threads()}, "
            f"expected {want}"
        )
    _SHARED = sh


def _f1_of(zoo, fam, y, Z):
    return _f1(y, zoo.predict_proba(fam, Z))


def run_unit(spec: dict) -> dict:
    """One (relay, seed): fit the zoo, run every mode in spec['modes']; returns its records."""
    sh = _SHARED
    ctx, r, seed = sh.ctx, spec["relay"], spec["seed"]
    t_unit = time.perf_counter()
    zoo = FittedZoo(ctx.reg, ENCODING, QUBITS, seed, ctx.subset_cap, ctx.best_params)
    t_fit = time.perf_counter() - t_unit
    wbs, gate = make_whitebox(zoo, ctx)
    grad_ctl = control_gradient(zoo, ctx)
    y, scale_z = ctx.y, ctx.reg.feature_scale
    clean = {f: _f1(y, zoo.clean_proba(f, ctx.Z0, y)) for f in FAMILIES}
    if min(clean.values()) <= 0:
        raise ValueError(f"a clean macro-F1 is not positive: {clean}")
    records, checks, timing, spaces = [], [], {}, {}

    def record(mode, eps, fam, kind, Z, space, extra=None):
        f1 = _f1_of(zoo, fam, y, Z)
        dZ = Z - ctx.Z0
        radius = (np.abs(dZ) / scale_z).max(axis=1)
        rec = {
            "seed": seed,
            "relay": r,
            "mode": mode,
            "arm": f"{mode}-R{r}",
            "eps": eps,
            "family": fam,
            "kind": kind,
            "n_channels": int(len(space.ctrl)),
            "n_attackable_rows": int(space.att.sum()),
            "clean_f1": clean[fam],
            "attacked_f1": f1,
            "retention": f1 / clean[fam],
            "f1_minus_floor": f1 - ctx.floor,
            "radius_mean": float(radius.mean()),
            "radius_max": float(radius.max()),
        }
        if sh.stage1 is not None and kind != "projection_only" and mode == "consistent":
            ref = sh.stage1.get((seed, f"A2-R{r}", eps, fam, kind))
            rec["stage1_a2_attacked_f1"] = ref
            rec["minus_stage1_a2"] = None if ref is None else f1 - ref
        records.append(rec | (extra or {}))

    for mode in spec["modes"]:
        space = AttackSpace(mode, sh.maps[r], sh, spec["device"])
        spaces[mode] = {
            "n_channels": int(len(space.ctrl)),
            "n_attackable_rows": int(space.att.sum()),
            "n_rows_nan_phase": space.n_nan_phase,
            "n_rows_nan_any_relay_channel": space.n_nan_relay,
        }
        rng = spawn_rng(seed, f"m1a2_{mode}_R{r}")
        rng_rand = spawn_rng(seed, f"m1a2_{mode}_random_R{r}")
        if spec["device"] == "cuda" and mode == "consistent":
            spaces[mode]["gpu_gate"] = gpu_gate(space, wbs["rbf_svm"], zoo, rng, spec, sh)
        if space.derive:  # v = 0 with the derived channels recomputed: the map's own shift
            X, _ = space.rows(np.zeros_like(space.x0c))
            c = space.check(np.zeros_like(space.x0c), X, 1.0, f"R{r} s{seed} projection")
            checks.append({"mode": mode, "eps": 0.0, "src": "projection_only"} | c)
            for fam in FAMILIES:
                record(mode, 0.0, fam, "projection_only", space.Z(X), space)
        for eps in spec["eps_grid"]:
            for src in STAGE1_SOURCES:
                t0 = time.perf_counter()
                v = space.pgd(wbs[src], y, eps, ctx.n_steps, ctx.step_frac, rng)
                timing[f"{mode}_{src}"] = timing.get(f"{mode}_{src}", 0.0) + (
                    time.perf_counter() - t0
                )
                X, _ = space.rows(v)
                c = space.check(v, X, eps, f"R{r} s{seed} {mode} {src} eps {eps}")
                checks.append({"mode": mode, "eps": eps, "src": src} | c)
                Z = space.Z(X)
                record(mode, eps, src, "whitebox", Z, space)
                if src == "mlp":
                    for fam in FAMILIES:
                        record(mode, eps, fam, "transfer_mlp", Z, space)
            v = space.random_sign(eps, rng_rand)
            X, _ = space.rows(v)
            c = space.check(v, X, eps, f"R{r} s{seed} {mode} random eps {eps}")
            checks.append({"mode": mode, "eps": eps, "src": "random"} | c)
            for fam in FAMILIES:
                record(mode, eps, fam, "random", space.Z(X), space)
            log.info(
                "R%d seed %d | %s eps %.2f done (%.0f s)",
                r,
                seed,
                mode,
                eps,
                time.perf_counter() - t_unit,
            )
    return {
        "unit": f"R{r}_s{seed}",
        "relay": r,
        "seed": seed,
        "records": records,
        "checks": checks,
        "spaces": spaces,
        "qsvm_gate": gate,
        "stage1_control4_gradient": grad_ctl,
        "timing_s": {"zoo_fit": round(t_fit, 1), **{k: round(v, 2) for k, v in timing.items()}},
        "wall_s": round(time.perf_counter() - t_unit, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "pid": os.getpid(),
    }


def gpu_gate(space: AttackSpace, wb, zoo, rng, spec, sh: Shared) -> dict:
    """The cuda derived-map path against the float64 cpu path: same PGD, same RNG state."""
    eps, y = max(spec["eps_grid"]), sh.ctx.y
    cpu = AttackSpace(space.mode, space.rm, sh, "cpu")
    v_cpu = cpu.pgd(wb, y, eps, sh.ctx.n_steps, sh.ctx.step_frac, copy.deepcopy(rng))
    v_gpu = space.pgd(wb, y, eps, sh.ctx.n_steps, sh.ctx.step_frac, copy.deepcopy(rng))
    diff = float(np.abs(v_gpu - v_cpu).max() / eps)
    Zc, Zg = cpu.Z(cpu.rows(v_cpu)[0]), space.Z(space.rows(v_gpu)[0])
    same = {
        f: bool(
            np.array_equal(zoo.predict_proba(f, Zc).argmax(1), zoo.predict_proba(f, Zg).argmax(1))
        )
        for f in FAMILIES
    }
    out = {"max_abs_diff_over_eps": diff, "identical_predictions": same}
    if diff > GPU_GATE_TOL or not all(same.values()):
        raise AssertionError(f"GPU gate failed: {out}")
    return out


def run_anchor(spec: dict) -> dict:
    """Control 6b: Stage 1's A2 arm through this runner's path, Stage 1's RNG stream replayed."""
    sh = _SHARED
    ctx = sh.ctx
    if sh.stage1 is None:
        raise AssertionError("anchor needs Stage 1's records at the same eval size")
    t_unit = time.perf_counter()
    zoo = FittedZoo(ctx.reg, ENCODING, QUBITS, 0, ctx.subset_cap, ctx.best_params)
    wbs, gate = make_whitebox(zoo, ctx)
    y = ctx.y
    out = {"cells": [], "qsvm_gate": gate, "tol": ANCHOR_TOL}
    e_idx = EPS_GRID.index(ANCHOR_EPS)
    for r in spec["relays"]:
        space = AttackSpace("a2_mask", sh.maps[r], sh, "cpu")
        for fam in ("qsvm", "rbf_svm"):
            rng = spawn_rng(0, "m1a")  # Stage 1's stream: one (n, 128) draw per pgd call
            skip = STAGE1_ARMS.index(f"A2-R{r}") * len(EPS_GRID) * len(STAGE1_SOURCES)
            skip += e_idx * len(STAGE1_SOURCES) + STAGE1_SOURCES.index(fam)
            for _ in range(skip):
                rng.uniform(-1.0, 1.0, size=sh.X0imp.shape)
            v = space.pgd(wbs[fam], y, ANCHOR_EPS, ctx.n_steps, ctx.step_frac, rng)
            X, _ = space.rows(v)
            space.check(v, X, ANCHOR_EPS, f"anchor R{r} {fam}")
            f1 = _f1_of(zoo, fam, y, space.Z(X))
            ref = sh.stage1[(0, f"A2-R{r}", ANCHOR_EPS, fam, "whitebox")]
            clean = _f1_of(zoo, fam, y, ctx.Z0)
            ok = abs(f1 - ref) <= ANCHOR_TOL
            out["cells"].append(
                {
                    "relay": r,
                    "family": fam,
                    "attacked_f1": f1,
                    "stage1_attacked_f1": ref,
                    "abs_diff": abs(f1 - ref),
                    "clean_f1": clean,
                    "stage1_clean_f1": sh.stage1[("clean", fam)],
                    "pass": bool(ok),
                }
            )
            log.info("anchor R%d %s: %.4f vs Stage 1 %.4f", r, fam, f1, ref)
    out["passed"] = all(c["pass"] for c in out["cells"])
    out["wall_s"] = round(time.perf_counter() - t_unit, 1)
    out["peak_rss_gb"] = round(_peak_rss_gb(), 3)
    return out


# --------------------------------------------------------------------------- joint (E-M1c)

JOINT_MODES = {"consistent": "joint", "map_off": "joint_map_off"}
TREES = ("rf", "xgb", "lgbm")
NS6_MARGIN = 0.05  # pre-registered: transfer damage minus random damage, macro-F1
ASYM_SMALL_DEN = 0.01  # a reverse drop under this leaves the published ratio uninformative
STAGE1_KINDS = ("whitebox", "transfer_mlp", "random")
JOINT_H, JOINT_GRAD_EPS, JOINT_GRAD_ROWS = 1e-5, 0.1, 3  # control 2j
FRAME_RESULTS = (  # the post-fix frame rerun (5170397): one radius statistic differs from 065642Z
    REPO_ROOT / "results" / "runs" / "20261116T080633Z_consistent_relay_attack_binary_q8_frame"
)
MARGINAL_RESULTS = (  # the post-fix registered-arm rerun (T=4, 816 of 816 records = 065045Z)
    REPO_ROOT / "results" / "runs" / "20261116T081157Z_consistent_relay_attack_binary_q8"
)
ARM_RESULTS = {"frame": FRAME_RESULTS, "marginal": MARGINAL_RESULTS}  # the joint anchor's ref
FRAME_JOINT_RESULTS = (  # E-M1c: E(consistent, frame) of the E-NS8(a) factorial
    REPO_ROOT
    / "results"
    / "runs"
    / "20261116T081940Z_consistent_relay_attack_binary_q8_frame_joint"
)
SURROGATE_RESULTS = REPO_ROOT / "results" / "runs" / "20260815T195529Z_surrogate_control_binary_q8"
ANCHOR_FIELDS = (
    "n_channels",
    "n_attackable_rows",
    "clean_f1",
    "attacked_f1",
    "retention",
    "f1_minus_floor",
    "radius_mean",
    "radius_max",
)


def joint_tag(relays: list[int]) -> str:
    return "+".join(f"R{r}" for r in relays)


class JointSpace:
    """Several relays attacked at once. Each relay's block of v is that relay's own AttackSpace
    (its unit, wrap, floor, deadband and map); the relays' maps are independent, so the
    Jacobian is block-diagonal and the gradient is each relay's own VJP on its block. With one
    relay every array equals that relay's AttackSpace's, bit for bit (the E-M1c anchor)."""

    def __init__(self, mode: str, relays: list[int], sh: Shared, device: str):
        if mode not in JOINT_MODES:
            raise ValueError(f"joint mode {mode!r} not in {list(JOINT_MODES)}")
        self.mode, self.sh = mode, sh
        self.subs = [AttackSpace(mode, sh.maps[r], sh, device) for r in relays]
        self.ctrl = np.concatenate([s.ctrl for s in self.subs])
        self.sc = np.concatenate([s.sc for s in self.subs])
        self.x0c = sh.X0imp[:, self.ctrl]
        self.att = np.any([s.att for s in self.subs], axis=0)
        ends = np.cumsum([0] + [len(s.ctrl) for s in self.subs])
        self.blocks = [slice(int(a), int(b)) for a, b in zip(ends[:-1], ends[1:], strict=True)]
        # the raw columns each relay writes: its controlled channels, plus derived when mapped
        self.cols = [
            np.concatenate([s.ctrl, s.rm.der_idx]) if s.derive else s.ctrl for s in self.subs
        ]
        allc = np.concatenate(self.cols)
        if len(np.unique(allc)) != len(allc):
            raise AssertionError("joint space: two relays write the same raw column")

    # the PGD, random-sign and pipeline steps are AttackSpace's own, on the joint arrays
    pgd = AttackSpace.pgd
    random_sign = AttackSpace.random_sign
    Z = AttackSpace.Z

    def box(self, eps: float):
        b = [s.box(eps) for s in self.subs]
        return np.concatenate([x[0] for x in b], axis=1), np.concatenate([x[1] for x in b], axis=1)

    def rows(self, v: np.ndarray) -> np.ndarray:
        """Full imputed raw rows: each relay's columns from its own AttackSpace.rows."""
        X = self.sh.X0imp.copy()
        for s, b, c in zip(self.subs, self.blocks, self.cols, strict=True):
            X[:, c] = s.rows(v[:, b])[0][:, c]
        return X

    def grad(self, v: np.ndarray, wb, y: np.ndarray) -> np.ndarray:
        """d loss / d v: one pass through the frozen pipeline, then each relay's own chain rule."""
        gx = (wb(self.Z(self.rows(v)), y) @ self.sh.ctx.W) / self.sh.s
        out = []
        for s, b in zip(self.subs, self.blocks, strict=True):
            g = gx[:, s.ctrl]
            if s.derive:
                g = g + s.rows(v[:, b], cot_der=gx[:, s.rm.der_idx])[1]
            out.append(g * s.sc)
        return np.concatenate(out, axis=1)

    def check(self, v: np.ndarray, X: np.ndarray, eps: float, tag: str) -> dict:
        """Control 3 on the channels no relay writes; controls 3 and 4 per relay (its own check
        on a row set that carries only its columns)."""
        moved = np.ones(X.shape[1], bool)
        moved[np.concatenate(self.cols)] = False
        d_other = float(np.abs(X[:, moved] - self.sh.X0imp[:, moved]).max())
        if d_other >= MOVE_TOL:
            raise AssertionError(f"control 3 ({tag}): an untouched channel moved by {d_other:.2e}")
        out = {"untouched_max_abs": d_other, "by_relay": {}}
        for s, b, c in zip(self.subs, self.blocks, self.cols, strict=True):
            Xs = self.sh.X0imp.copy()
            Xs[:, c] = X[:, c]
            out["by_relay"][f"R{s.rm.relay}"] = s.check(v[:, b], Xs, eps, f"{tag} R{s.rm.relay}")
        return out


def _near_breakpoint(s: AttackSpace, xc: np.ndarray) -> bool:
    """A row whose relay sits at a deadband edge, an angle cut or I1 = 0 (control 2's rule)."""
    sv, si = sequences_np(xc[None])
    near = any(
        abs(abs((sv if k <= 9 else si)[s.rm.assign[k]][0]) - s.rm.deadband[k])
        <= 1e-3 * max(s.rm.deadband[k], 1.0)
        for k in SEQ_CHANNELS
    )
    d = derived_np(xc[None], s.rm)[0]
    near |= bool(np.any(np.abs(np.abs(d[6:12]) - 180.0) <= 1e-3))
    near |= bool(abs(abs(d[13]) - np.pi) <= 1e-5) or abs(si["pos"][0]) == 0
    near |= bool(np.any(np.abs(np.abs(xc[6:12]) - 180.0) <= 1e-3))  # the reported-angle wrap
    return near


def control_joint_gradient(space: JointSpace, wb, svc, y: np.ndarray) -> dict:
    """Control 2j: the joint chain-rule gradient (frozen pipeline, every relay's VJP) against
    central differences in v of the RBF-SVM's BCE loss, the two largest-gradient channels of
    every relay block on rows clear of breakpoints, at a random point of the eps-0.1 box.

    d loss / dv = (sigmoid(f) - y) d f / dv, and only the kernel sum k = f - intercept moves
    with v, so k is the function differenced. Differencing the loss itself loses the gradient to
    rounding once k is tiny against the intercept: under the marginal unit the eps-0.1 point
    leaves |g| ~ 1e-9 while the loss steps by ~1e-11 per ulp over 2h (E-NS8(a) smoke)."""
    sv, alpha = svc.support_vectors_, svc.dual_coef_[0]
    gamma = svc._gamma if hasattr(svc, "_gamma") else svc.gamma

    def kernel_sum(v, i):  # f - intercept, at full relative precision
        Z = space.Z(space.rows(v))[i]
        return float(np.exp(-gamma * ((Z[None, :] - sv) ** 2).sum(axis=1)) @ alpha)

    def dloss_df(v, i):  # BCE(sigmoid(f), y), the loss the RBF-SVM GradProvider differentiates
        f = kernel_sum(v, i) + svc.intercept_[0]
        return 1.0 / (1.0 + np.exp(-f)) - y[i]

    eps = JOINT_GRAD_EPS
    lo, hi = space.box(eps)
    v = lo + (hi - lo) * spawn_rng(0, "m1c_ctl2j").uniform(size=lo.shape)
    g = space.grad(v, wb, y)
    X = space.rows(v)
    every = np.all([s.att for s in space.subs], axis=0)
    order = np.argsort(-np.abs(g).max(axis=1))
    keep = [
        int(i)
        for i in order
        if every[i] and not any(_near_breakpoint(s, X[i, s.ctrl]) for s in space.subs)
    ][:JOINT_GRAD_ROWS]
    if len(keep) < JOINT_GRAD_ROWS:
        raise AssertionError(f"control 2j: only {len(keep)} rows clear of breakpoints")
    margin = 1e-3 * eps
    worst, per = 0.0, []
    for i in keep:
        scale = float(np.abs(g[i]).max())
        w = dloss_df(v, i)
        for s, b in zip(space.subs, space.blocks, strict=True):
            vb, lb, hb = v[i, b], lo[i, b], hi[i, b]
            inside = (vb > lb + margin) & (vb < hb - margin)
            cand = [int(j) for j in np.argsort(-np.abs(g[i, b])) if inside[j]][:2]
            if not cand:
                raise AssertionError(f"control 2j: R{s.rm.relay} row {i} has no interior channel")
            for j in cand:
                jj = b.start + j
                up, dn = v.copy(), v.copy()
                up[i, jj] += JOINT_H
                dn[i, jj] -= JOINT_H
                fd = w * (kernel_sum(up, i) - kernel_sum(dn, i)) / (2 * JOINT_H)
                rel = abs(fd - g[i, jj]) / max(scale, 1e-300)
                worst = max(worst, rel)
                per.append(
                    {
                        "row": i,
                        "relay": s.rm.relay,
                        "channel": s.rm.ctrl_names[j],
                        "grad": float(g[i, jj]),
                        "fd": fd,
                        "rel_err_over_row_max": rel,
                    }
                )
    out = {"max_rel_err": worst, "tol": JAC_TOL, "h": JOINT_H, "eps": eps, "checks": per}
    if worst > JAC_TOL:
        raise AssertionError(f"control 2j: joint gradient off by {worst:.2e} > {JAC_TOL}")
    return out


def run_joint_unit(spec: dict) -> dict:
    """One seed: fit the zoo, attack every relay in spec['relays'] jointly in each mode."""
    sh = _SHARED
    ctx, relays, seed = sh.ctx, spec["relays"], spec["seed"]
    tag = joint_tag(relays)
    t_unit = time.perf_counter()
    zoo = FittedZoo(ctx.reg, ENCODING, QUBITS, seed, ctx.subset_cap, ctx.best_params)
    t_fit = time.perf_counter() - t_unit
    wbs, gate = make_whitebox(zoo, ctx)
    grad_ctl = control_gradient(zoo, ctx)
    y, scale_z = ctx.y, ctx.reg.feature_scale
    clean = {f: _f1(y, zoo.clean_proba(f, ctx.Z0, y)) for f in FAMILIES}
    if min(clean.values()) <= 0:
        raise ValueError(f"a clean macro-F1 is not positive: {clean}")
    records, checks, timing, spaces, ctl2j = [], [], {}, {}, None
    with_stage1 = sh.stage1 is not None and list(relays) == RELAYS

    def record(mode, eps, fam, kind, Z, space):
        f1 = _f1_of(zoo, fam, y, Z)
        radius = (np.abs(Z - ctx.Z0) / scale_z).max(axis=1)
        rec = {
            "seed": seed,
            "relay": tag,
            "relays": list(relays),
            "mode": mode,
            "arm": f"{mode}-{tag}",
            "eps": eps,
            "family": fam,
            "kind": kind,
            "n_channels": int(len(space.ctrl)),
            "n_attackable_rows": int(space.att.sum()),
            "clean_f1": clean[fam],
            "attacked_f1": f1,
            "retention": f1 / clean[fam],
            "f1_minus_floor": f1 - ctx.floor,
            "radius_mean": float(radius.mean()),
            "radius_max": float(radius.max()),
        }
        if with_stage1 and mode == "joint" and kind in STAGE1_KINDS:
            ref = sh.stage1.get((seed, "A1", eps, fam, kind))
            rec["stage1_a1_attacked_f1"] = ref
            rec["minus_stage1_a1"] = None if ref is None else f1 - ref
        records.append(rec)

    for sub in spec["modes"]:
        mode = JOINT_MODES[sub]
        space = JointSpace(sub, relays, sh, "cpu")
        spaces[mode] = {
            "n_channels": int(len(space.ctrl)),
            "n_attackable_rows_any_relay": int(space.att.sum()),
            "by_relay": {
                f"R{s.rm.relay}": {
                    "n_attackable_rows": int(s.att.sum()),
                    "n_rows_nan_phase": s.n_nan_phase,
                    "n_rows_nan_any_relay_channel": s.n_nan_relay,
                }
                for s in space.subs
            },
        }
        # the stream names of run_unit's consistent mode with the relay set as the tag, so the
        # mode restricted to R1 replays the frame arm's R1 streams
        rng = spawn_rng(seed, f"m1a2_{sub}_{tag}")
        rng_rand = spawn_rng(seed, f"m1a2_{sub}_random_{tag}")
        if sub == "consistent":
            ctl2j = control_joint_gradient(space, wbs["rbf_svm"], zoo.models["rbf_svm"], y)
            log.info("%s seed %d: control 2j max rel err %.2e", tag, seed, ctl2j["max_rel_err"])
            X = space.rows(np.zeros_like(space.x0c))  # v = 0, derived recomputed
            c = space.check(np.zeros_like(space.x0c), X, 1.0, f"{tag} s{seed} projection")
            checks.append({"mode": mode, "eps": 0.0, "src": "projection_only"} | c)
            for fam in FAMILIES:
                record(mode, 0.0, fam, "projection_only", space.Z(X), space)
        for eps in spec["eps_grid"]:
            for src in STAGE1_SOURCES:
                t0 = time.perf_counter()
                v = space.pgd(wbs[src], y, eps, ctx.n_steps, ctx.step_frac, rng)
                timing[f"{mode}_{src}"] = timing.get(f"{mode}_{src}", 0.0) + (
                    time.perf_counter() - t0
                )
                X = space.rows(v)
                c = space.check(v, X, eps, f"{tag} s{seed} {mode} {src} eps {eps}")
                checks.append({"mode": mode, "eps": eps, "src": src} | c)
                Z = space.Z(X)
                record(mode, eps, src, "whitebox", Z, space)
                for fam in FAMILIES:  # inference only: this source's set against every family
                    record(mode, eps, fam, f"transfer_{src}", Z, space)
            v = space.random_sign(eps, rng_rand)
            X = space.rows(v)
            c = space.check(v, X, eps, f"{tag} s{seed} {mode} random eps {eps}")
            checks.append({"mode": mode, "eps": eps, "src": "random"} | c)
            for fam in FAMILIES:
                record(mode, eps, fam, "random", space.Z(X), space)
            log.info(
                "%s seed %d | %s eps %.2f done (%.0f s)",
                tag,
                seed,
                mode,
                eps,
                time.perf_counter() - t_unit,
            )
    return {
        "unit": f"{tag}_s{seed}",
        "relay": tag,
        "relays": list(relays),
        "seed": seed,
        "records": records,
        "checks": checks,
        "spaces": spaces,
        "control2j_joint_gradient": ctl2j,
        "qsvm_gate": gate,
        "stage1_control4_gradient": grad_ctl,
        "timing_s": {"zoo_fit": round(t_fit, 1), **{k: round(v, 2) for k, v in timing.items()}},
        "wall_s": round(time.perf_counter() - t_unit, 1),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "pid": os.getpid(),
    }


# --------------------------------------------------------------------------- parent


def load_stage1(n_eval: int, n_steps: int):
    res = json.loads((STAGE1_RESULTS / "results.json").read_text())
    cfg = res["config"]
    if cfg["n_eval"] != n_eval or cfg["n_steps"] != n_steps:
        return None  # smoke: different rows and steps, no comparison
    out = {}
    for p in res["per_seed"]:
        for x in p["records"]:
            out[(x["seed"], x["arm"], x["eps"], x["family"], x["kind"])] = x["attacked_f1"]
            if x["seed"] == 0:
                out[("clean", x["family"])] = x["clean_f1"]
    return out


def build_shared(args, threads: int, relays: list[int]) -> tuple[Shared, dict]:
    ctx = build_context(SimpleNamespace(smoke=args.smoke))
    Xraw, _, splits = load_processed(VARIANT)
    Xtr = Xraw[splits["train"]].astype(np.float64)
    pipe = build_preprocessor("pca", n_components=QUBITS, seed=0).fit(Xraw[splits["train"]])
    if not np.array_equal(pipe.named_steps["pca"].components_.astype(np.float64), ctx.W):
        raise AssertionError("refitted pipeline differs from Stage 1's context")
    ev = stratified_cap(ctx.reg.X["test"], ctx.reg.y["test"], len(ctx.y), seed=0)
    X0raw = Xraw[splits["test"]][ev].astype(np.float64)
    imputer, scaler = pipe.named_steps["impute"], pipe.named_steps["scale"]
    X0imp = imputer.transform(X0raw)
    if not np.array_equal(scaler.transform(X0imp), ctx.U0):
        raise AssertionError("eval rows differ from Stage 1's context")
    s = scaler.scale_.astype(np.float64)
    pos = {c: i for i, c in enumerate(ctx.columns)}
    maps, diag = {}, {"control1": {}, "control2_jacobian": {}}
    for r in relays:
        z_imp = float(imputer.statistics_[pos[f"R{r}-PA:Z"]])
        rm = fit_relay_map(Xtr, ctx.columns, r, z_imp)
        ok = np.isfinite(X0raw[:, np.concatenate([rm.ctrl_idx, rm.der_idx])]).all(axis=1)
        diag["control1"][f"R{r}"] = {
            "assign": rm.assign,
            "deadband": rm.deadband,
            "log_ratio": rm.log_ratio,
            "zh_offset_rad": rm.zh_offset,
            "z_impute": rm.z_impute,
            **rm.control1,
        }
        diag["control2_jacobian"][f"R{r}"] = control_jacobian(
            rm, X0raw[ok][:, rm.ctrl_idx], s[rm.ctrl_idx]
        )
        if args.angle_unit == "frame":
            rm.frame_unit, diag.setdefault("frame_unit", {})[f"R{r}"] = fit_frame_unit(Xtr, rm)
            log.info("R%d frame unit (deg) PA1..PA6: %s", r, np.round(rm.frame_unit, 3).tolist())
        if args.magnitude_unit != "scale":
            rm.mag_unit, d = fit_magnitude_unit(Xtr, rm, s, args.magnitude_unit)
            diag.setdefault("magnitude_unit", {})[f"R{r}"] = d
            log.info(
                "R%d magnitude unit (%s) PM1..PM3: %s V; physical %s V on %d of %d train rows; "
                "scaler.scale_ %s V",
                r,
                args.magnitude_unit,
                np.round(rm.mag_unit, 1).tolist(),
                np.round(d["physical_unit_V"], 1).tolist(),
                d["n_rows_used"],
                d["n_train_rows"],
                np.round(d["scaler_scale_V"], 1).tolist(),
            )
        maps[r] = rm
        log.info(
            "R%d map: assign %s deadband %s c %.4f o %.4f rad | Jacobian %.2e",
            r,
            rm.assign,
            {k: round(v, 2) for k, v in rm.deadband.items()},
            rm.log_ratio,
            rm.zh_offset,
            diag["control2_jacobian"][f"R{r}"]["max_rel_err"],
        )
    raw_std = np.nanstd(Xtr, axis=0)
    diag["unit_scale_vs_nanstd_max_rel"] = float(np.max(np.abs(s - raw_std) / raw_std))
    sh = Shared(
        ctx,
        X0raw,
        X0imp,
        s,
        maps,
        load_stage1(len(ctx.y), ctx.n_steps),
        threads,
        angle_unit=args.angle_unit,
    )
    return sh, diag


def run_pool(fn, specs, sh: Shared, workers: int, on_done=None) -> list[dict]:
    out = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(sh,),
    ) as ex:
        futs = {ex.submit(fn, sp): sp for sp in specs}
        for f in as_completed(futs):
            res = f.result()  # a failed unit raises here, with its traceback
            out.append(res)
            if on_done:
                on_done(res)
    return out


def cells(units: list[dict]) -> dict:
    """Attacked F1 per (mode, relay, seed, eps, family, kind): control 7's comparison key."""
    return {
        (x["mode"], x["relay"], x["seed"], x["eps"], x["family"], x["kind"]): x["attacked_f1"]
        for u in units
        for x in u["records"]
    }


def memory_guard(workers: int, rss_gb: float, what: str) -> dict:
    avail = psutil.virtual_memory().available / 2**30
    need = workers * rss_gb
    out = {"workers": workers, "worker_rss_gb": rss_gb, "need_gb": need, "available_gb": avail}
    log.info("memory (%s): %d x %.2f GB = %.2f GB vs %.2f GB available", what, *out.values())
    if need > MEM_FRACTION * avail:
        raise MemoryError(f"{what}: {need:.1f} GB > {MEM_FRACTION:.0%} of {avail:.1f} GB free")
    return out


def summarize(units: list[dict], floor: float, eps_grid: list[float]):
    units = sorted(units, key=lambda u: (u["relay"], u["seed"]))
    records = [x for u in units for x in u["records"]]
    summary = aggregate(records)
    idx = {(x["arm"], x["eps"], x["family"], x["kind"]): x for x in summary}
    relays = sorted({u["relay"] for u in units})
    ctl5 = []
    for r in relays:
        for eps in eps_grid:
            for fam in WHITEBOX:
                pg = idx[(f"consistent-R{r}", eps, fam, "whitebox")]["attacked_f1"]["mean"]
                rd = idx[(f"consistent-R{r}", eps, fam, "random")]["attacked_f1"]["mean"]
                ctl5.append(
                    {
                        "relay": r,
                        "eps": eps,
                        "family": fam,
                        "pgd_f1": pg,
                        "random_f1": rd,
                        "pass": bool(rd > pg),
                    }
                )
    # Stage 1 comparison at matched relay / eps / family / kind (seed means)
    cmp_rows = []
    groups: dict = {}
    for x in records:
        if x["mode"] == "consistent" and x.get("stage1_a2_attacked_f1") is not None:
            groups.setdefault((x["relay"], x["eps"], x["family"], x["kind"]), []).append(x)
    for (r, eps, fam, kind), xs in sorted(groups.items()):
        cmp_rows.append(
            {
                "relay": r,
                "eps": eps,
                "family": fam,
                "kind": kind,
                "stage2_attacked_f1": _ms([x["attacked_f1"] for x in xs]),
                "stage1_a2_attacked_f1": _ms([x["stage1_a2_attacked_f1"] for x in xs]),
                "stage2_minus_stage1": _ms([x["minus_stage1_a2"] for x in xs]),
            }
        )
    # pre-registered reading: QSVM and RBF-SVM white-box below the floor at the top eps
    top = max(eps_grid)
    wb = {
        (r, eps, fam): idx[(f"consistent-R{r}", eps, fam, "whitebox")]["attacked_f1"]["mean"]
        for r in relays
        for eps in eps_grid
        for fam in ("qsvm", "rbf_svm")
    }
    below = [r for r in relays if wb[(r, top, "qsvm")] < floor and wb[(r, top, "rbf_svm")] < floor]
    if len(below) >= 3:
        verdict = "hypothesis"
    elif all(v > floor for v in wb.values()):
        verdict = "kill"
    else:
        verdict = "intermediate"
    branch = {
        "floor": floor,
        "eps_top": top,
        "whitebox_f1_seed_mean": {f"R{r}|{e}|{f}": v for (r, e, f), v in wb.items()},
        "relays_both_below_floor_at_top_eps": below,
        "n_relays": len(relays),
        "verdict": verdict,
        "complete_grid": len(relays) == 4 and len(eps_grid) == 3,
    }
    return records, summary, ctl5, cmp_rows, branch


def run_pool_all(fn, specs, sh: Shared, workers: int, on_done) -> list[dict]:
    """run_pool, but one failed unit does not discard the others: every unit that finishes is
    handed to on_done (which writes it) and the first failure is raised after the pool drains."""
    out, failed = [], []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(sh,),
    ) as ex:
        futs = {ex.submit(fn, sp): sp for sp in specs}
        for f in as_completed(futs):
            exc = f.exception()
            if exc is not None:
                log.error("unit %s failed", futs[f], exc_info=exc)
                failed.append((futs[f], exc))
                continue
            out.append(f.result())
            on_done(out[-1])
    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(specs)} units failed (finished units are in "
            f"results_partial.json): {[sp for sp, _ in failed]}"
        ) from failed[0][1]
    return out


def _seed_ms(seeds, fn) -> dict:
    return _ms([fn(s) for s in seeds])


def asymmetry(fwd: float, rev: float) -> dict:
    """The published statistic (run_surrogate_control: forward_over_reverse, the VQC's drop
    under MLP-crafted examples over the MLP's drop under VQC-crafted ones, each the seed-mean
    clean minus attacked macro-F1, denominator floored at 1e-9) and a bounded form, the
    normalized difference (F - R) / (|F| + |R|) in [-1, 1]."""
    den = abs(fwd) + abs(rev)
    return {
        "forward_drop_mlp_to_vqc": fwd,
        "reverse_drop_vqc_to_mlp": rev,
        "published_ratio_forward_over_reverse": fwd / max(rev, 1e-9),
        "ratio_denominator_small": bool(rev < ASYM_SMALL_DEN),
        "bounded_index": None if den == 0 else (fwd - rev) / den,
        "difference": fwd - rev,
    }


def summarize_joint(units: list[dict], sh: Shared, eps_grid: list[float], relays) -> dict:
    """Control 5 per cell, the transfer matrix, the asymmetry, Stage 1 A1, and the branches."""
    units = sorted(units, key=lambda u: u["seed"])
    seeds = [u["seed"] for u in units]
    records = [x for u in units for x in u["records"]]
    floor, top = sh.ctx.floor, max(eps_grid)
    f1 = {
        (x["seed"], x["eps"], x["family"], x["kind"]): x["attacked_f1"]
        for x in records
        if x["mode"] == "joint"
    }
    clean = {(x["seed"], x["family"]): x["clean_f1"] for x in records if x["mode"] == "joint"}

    ctl5 = []
    for eps in eps_grid:
        for fam in WHITEBOX:
            pg = _seed_ms(seeds, lambda s, e=eps, f=fam: f1[(s, e, f, "whitebox")])["mean"]
            rd = _seed_ms(seeds, lambda s, e=eps, f=fam: f1[(s, e, f, "random")])["mean"]
            cl = _seed_ms(seeds, lambda s, f=fam: clean[(s, f)])["mean"]
            ctl5.append(
                {
                    "eps": eps,
                    "family": fam,
                    "clean_f1": cl,
                    "pgd_f1": pg,
                    "random_f1": rd,
                    "pgd_damage": cl - pg,
                    "random_damage": cl - rd,
                    "status": "admitted" if rd > pg else "random_saturated",
                    "pass": bool(rd > pg),
                }
            )

    matrix, excess = [], {}
    for src in STAGE1_SOURCES:
        for fam in FAMILIES:
            for eps in eps_grid:
                kind = f"transfer_{src}"
                row = {
                    "source": src,
                    "target": fam,
                    "eps": eps,
                    "clean_f1": _seed_ms(seeds, lambda s, f=fam: clean[(s, f)]),
                    "transfer_f1": _seed_ms(
                        seeds, lambda s, e=eps, f=fam, k=kind: f1[(s, e, f, k)]
                    ),
                    "random_f1": _seed_ms(seeds, lambda s, e=eps, f=fam: f1[(s, e, f, "random")]),
                    # transfer damage minus random damage = random F1 minus transfer F1
                    "transfer_minus_random_damage": _seed_ms(
                        seeds,
                        lambda s, e=eps, f=fam, k=kind: f1[(s, e, f, "random")] - f1[(s, e, f, k)],
                    ),
                }
                row["transfer_damage"] = row["clean_f1"]["mean"] - row["transfer_f1"]["mean"]
                row["random_damage"] = row["clean_f1"]["mean"] - row["random_f1"]["mean"]
                excess[(src, fam, eps)] = row["transfer_minus_random_damage"]["mean"]
                matrix.append(row)

    def drop(eps, src, tgt):
        return _seed_ms(seeds, lambda s: clean[(s, tgt)] - f1[(s, eps, tgt, f"transfer_{src}")])

    asym = {}
    for eps in eps_grid:
        a = asymmetry(drop(eps, "mlp", "vqc")["mean"], drop(eps, "vqc", "mlp")["mean"])
        a["forward_drop_seed_ms"], a["reverse_drop_seed_ms"] = (
            drop(eps, "mlp", "vqc"),
            drop(eps, "vqc", "mlp"),
        )
        a["forward_excess_over_random"] = excess[("mlp", "vqc", eps)]
        a["reverse_excess_over_random"] = excess[("vqc", "mlp", eps)]
        asym[str(eps)] = a
    pub = json.loads((SURROGATE_RESULTS / "results.json").read_text())["summary"]["mlp"]
    asym["published_reference"] = {
        "run": str(SURROGATE_RESULTS.relative_to(REPO_ROOT)),
        "eps_pca_space": 0.5,
        **asymmetry(pub["vqc_damage"], pub["reverse_damage"]),
    }

    def ns6(ex: dict) -> dict:
        noise = all(ex[(f, e)] < NS6_MARGIN for f in TREES for e in eps_grid)
        surv = [f for f in TREES if ex[(f, top)] >= NS6_MARGIN]
        return {
            "transfer_minus_random_damage": {
                f"{f}|{e}": ex[(f, e)] for f in TREES for e in eps_grid
            },
            "per_family_noise_at_every_eps": {
                f: all(ex[(f, e)] < NS6_MARGIN for e in eps_grid) for f in TREES
            },
            "families_at_least_margin_at_top_eps": surv,
            "verdict": "noise" if noise else "survives" if len(surv) >= 2 else "intermediate",
        }

    by_src = {
        src: ns6({(f, e): excess[(src, f, e)] for f in TREES for e in eps_grid})
        for src in STAGE1_SOURCES
    }
    strongest = ns6(
        {
            (f, e): max(excess[(src, f, e)] for src in STAGE1_SOURCES)
            for f in TREES
            for e in eps_grid
        }
    )
    branch_ns6 = {
        "margin": NS6_MARGIN,
        "eps_top": top,
        "primary_source": "mlp",
        "verdict": by_src["mlp"]["verdict"],
        "by_source": by_src,
        "strongest_source_per_cell": strongest,
        "complete_grid": list(relays) == RELAYS and list(eps_grid) == EPS_GRID,
    }

    cell5 = {(c["family"], c["eps"]): c for c in ctl5}
    kern = {}
    for fam in ("qsvm", "rbf_svm"):
        adm = [e for e in eps_grid if cell5[(fam, e)]["pass"]]
        kern[fam] = {
            "whitebox_f1": {str(e): cell5[(fam, e)]["pgd_f1"] for e in eps_grid},
            "admitted_eps": adm,
            "admitted_at_top_eps": top in adm,
            "below_floor_at_top_eps": cell5[(fam, top)]["pgd_f1"] < floor,
            "above_floor_every_admitted_eps": all(cell5[(fam, e)]["pgd_f1"] > floor for e in adm),
        }
    if not all(k["admitted_at_top_eps"] for k in kern.values()):
        kverdict = "inconclusive_at_top_eps"
    elif all(k["below_floor_at_top_eps"] for k in kern.values()):
        kverdict = "hypothesis"
    elif all(k["above_floor_every_admitted_eps"] for k in kern.values()):
        kverdict = "kill"
    else:
        kverdict = "intermediate"

    cmp_a1 = []
    if sh.stage1 is not None and list(relays) == RELAYS:
        for eps in eps_grid:
            for fam in FAMILIES:
                for kind in STAGE1_KINDS:
                    if kind == "whitebox" and fam not in WHITEBOX:
                        continue
                    j = _seed_ms(seeds, lambda s, e=eps, f=fam, k=kind: f1[(s, e, f, k)])
                    a1 = _seed_ms(
                        seeds, lambda s, e=eps, f=fam, k=kind: sh.stage1[(s, "A1", e, f, k)]
                    )
                    cmp_a1.append(
                        {
                            "eps": eps,
                            "family": fam,
                            "kind": kind,
                            "joint_attacked_f1": j,
                            "stage1_a1_attacked_f1": a1,
                            "joint_minus_a1": j["mean"] - a1["mean"],
                        }
                    )
    return {
        "records": records,
        "summary": aggregate(records),
        "control5_negative": ctl5,
        "transfer_matrix": matrix,
        "asymmetry": asym,
        "compare_stage1_a1": cmp_a1,
        "branch_ns6": branch_ns6,
        "branch_kernels": {"floor": floor, "eps_top": top, "per_kernel": kern, "verdict": kverdict},
    }


def compare_frame_arm(units: list[dict], ref_run: Path = FRAME_RESULTS) -> dict:
    """Joint anchor: every R1 cell of the single-relay arm run ref_run (the frame arm for
    E-M1c, the registered marginal arm for E-NS8(a)) at the run's seeds is reproduced exactly."""
    res = json.loads((ref_run / "results.json").read_text())
    unit = res["config"]["angle_unit"]
    seeds = {u["seed"] for u in units}
    ref = {
        (x["seed"], x["eps"], x["family"], x["kind"]): x
        for x in res["records"]
        if x["relay"] == 1 and x["seed"] in seeds
    }
    got = {
        (x["seed"], x["eps"], x["family"], x["kind"]): x
        for u in units
        for x in u["records"]
        if x["mode"] == "joint"
    }
    missing = [list(k) for k in ref if k not in got]
    cells, worst = [], 0.0
    for k, r in sorted(ref.items(), key=lambda kv: str(kv[0])):
        if k not in got:
            continue
        diff = {f: [r[f], got[k][f]] for f in ANCHOR_FIELDS if r[f] != got[k][f]}
        worst = max(worst, abs(r["attacked_f1"] - got[k]["attacked_f1"]))
        cells.append(
            {
                "seed": k[0],
                "eps": k[1],
                "family": k[2],
                "kind": k[3],
                f"{unit}_attacked_f1": r["attacked_f1"],
                "joint_attacked_f1": got[k]["attacked_f1"],
                "fields_differ": diff,
            }
        )
    n_bad = sum(bool(c["fields_differ"]) for c in cells)
    return {
        f"{unit}_arm_run": str(ref_run.relative_to(REPO_ROOT)),
        "fields_compared": list(ANCHOR_FIELDS),
        "rule": "exact equality",
        "n_reference_cells": len(ref),
        "n_compared": len(cells),
        "n_missing": len(missing),
        "missing": missing,
        "n_differ": n_bad,
        "max_abs_attacked_f1_diff": worst,
        "passed": bool(ref) and not missing and n_bad == 0,
        "cells": cells,
    }


FACTORIAL_KEYS = ("seeds", "eps_grid", "n_eval", "n_steps", "step_frac")


def _k(key: str) -> tuple:
    """'rf|0.5' -> ('rf', 0.5): E-M1c's branch_ns6 key."""
    f, e = key.split("|")
    return f, float(e)


def factorial_ns8a(records: list[dict], cfg: dict) -> dict:
    """E-NS8(a), the consistency-against-unit factorial (decision log 2026-11-16T09:19Z).

    E = MLP-surrogate transfer damage minus random damage = random F1 - transfer_mlp F1, seed
    means, per tree family and eps, in three cells: Stage 1's A1 (channels independent, marginal
    unit), this run (consistent, marginal) and E-M1c (consistent, frame). Consistency effect =
    E(A1) - E(consistent, marginal); unit effect = E(consistent, marginal) - E(consistent,
    frame). Branch at the top eps over the trees with E(A1) >= the margin: PHYSICS when E(cons,
    marginal) < margin for 2 of them, UNIT when >= margin for 2, MIXED otherwise. Everything
    besides the factor that could differ between the cells is asserted equal or recorded."""
    s1 = json.loads((STAGE1_RESULTS / "results.json").read_text())
    fj = json.loads((FRAME_JOINT_RESULTS / "results.json").read_text())
    src = {
        "A1": (
            STAGE1_RESULTS,
            s1["config"],
            [x for p in s1["per_seed"] for x in p["records"] if x["arm"] == "A1"],
        ),
        "consistent_marginal": (None, cfg, [x for x in records if x["mode"] == "joint"]),
        "consistent_frame": (
            FRAME_JOINT_RESULTS,
            fj["config"],
            [x for x in fj["records"] if x["mode"] == "joint"],
        ),
    }
    for k in FACTORIAL_KEYS:
        vals = {name: c[k] for name, (_, c, _) in src.items()}
        if len({json.dumps(v) for v in vals.values()}) != 1:
            raise AssertionError(f"factorial: {k} differs between the cells: {vals}")
    for name in ("consistent_marginal", "consistent_frame"):
        c = src[name][1]
        if c["relays"] != RELAYS or c["angle_unit"] != name.split("_")[1]:
            raise AssertionError(f"factorial: {name} is relays {c['relays']}, {c['angle_unit']}")
    seeds, eps_grid = cfg["seeds"], cfg["eps_grid"]
    top = max(eps_grid)

    f1, clean, E = {}, {}, {}
    for name, (_, _, recs) in src.items():
        f1[name] = {(x["seed"], x["eps"], x["family"], x["kind"]): x["attacked_f1"] for x in recs}
        clean[name] = {(x["seed"], x["family"]): x["clean_f1"] for x in recs}
        E[name] = {
            (f, e): _seed_ms(
                seeds,
                lambda s, n=name, f=f, e=e: (
                    f1[n][(s, e, f, "random")] - f1[n][(s, e, f, "transfer_mlp")]
                ),
            )
            for f in TREES
            for e in eps_grid
        }
    # reader control: the frame cell is E-M1c's own published quantity, bit for bit
    pub = fj["branch_ns6"]["by_source"]["mlp"]["transfer_minus_random_damage"]
    off = {k: (pub[k], E["consistent_frame"][_k(k)]["mean"]) for k in pub}
    bad = {k: v for k, v in off.items() if v[0] != v[1]}
    if bad:
        raise AssertionError(f"factorial: frame cell does not reproduce E-M1c's NS6 read: {bad}")

    mean = {n: {k: v["mean"] for k, v in E[n].items()} for n in E}
    a1, cm, cf = mean["A1"], mean["consistent_marginal"], mean["consistent_frame"]
    included = [f for f in TREES if a1[(f, top)] >= NS6_MARGIN]
    low = [f for f in included if cm[(f, top)] < NS6_MARGIN]
    high = [f for f in included if cm[(f, top)] >= NS6_MARGIN]
    verdict = "PHYSICS" if len(low) >= 2 else "UNIT" if len(high) >= 2 else "MIXED"

    fams = sorted({f for c in clean.values() for _, f in c})
    clean_diff = {
        name: {
            f: max(abs(clean[name][(s, f)] - clean["consistent_marginal"][(s, f)]) for s in seeds)
            for f in fams
        }
        for name in ("A1", "consistent_frame")
    }
    differ = {n: [f for f, d in v.items() if d != 0] for n, v in clean_diff.items()}
    if any(differ.values()):
        log.warning("factorial: clean F1 differs from this run's models in %s", differ)
    return {
        "experiment": "E-NS8(a)",
        "statistic": "seed mean of (random F1 - transfer_mlp F1), macro-F1",
        "margin": NS6_MARGIN,
        "eps_top": top,
        "cells": {
            name: {
                "run": "this run" if run is None else str(run.relative_to(REPO_ROOT)),
                "angle_unit": c.get("angle_unit", "marginal (Stage 1 standardized space)"),
                "threads": c["threads"],
                "n_channels": sorted({x["n_channels"] for x in recs}),
                "n_attacked_rows": sorted({x.get("n_attackable_rows", c["n_eval"]) for x in recs}),
                "E": {f"{f}|{e}": v for (f, e), v in E[name].items()},
            }
            for name, (run, c, recs) in src.items()
        },
        "consistency_effect": {
            f"{f}|{e}": a1[(f, e)] - cm[(f, e)] for f in TREES for e in eps_grid
        },
        "unit_effect": {f"{f}|{e}": cm[(f, e)] - cf[(f, e)] for f in TREES for e in eps_grid},
        "per_family_at_top_eps": {
            f: {
                "E_A1": a1[(f, top)],
                "E_consistent_marginal": cm[(f, top)],
                "E_consistent_frame": cf[(f, top)],
                "reading": "excluded" if f not in included else "physics" if f in low else "unit",
            }
            for f in TREES
        },
        "included_trees": included,
        "excluded_trees_E_A1_under_margin": [f for f in TREES if f not in included],
        "verdict": verdict,
        "identity": {
            "asserted_equal": list(FACTORIAL_KEYS) + ["source transfer_mlp", "relays (joint)"],
            "reader_control_frame_cell_equals_E_M1c_branch_ns6": True,
            "clean_f1_max_abs_diff_vs_consistent_marginal": clean_diff,
            "families_whose_fitted_model_differs": differ,
            "random_control": {
                "A1": "sign(+-1) x eps on every A1 channel of every row, no floor (Stage 1)",
                "consistent": "sign(+-1) x eps clipped to the consistent box (magnitude floor, "
                "rows with a NaN relay channel left untouched for that relay), derived "
                "channels recomputed",
            },
        },
    }


def magnitude_verdict(qsvm_whitebox_f1: dict, floor: float = MAG_UNIT_FLOOR) -> str:
    """R4-06's registered branch on the QSVM's seed-mean white-box F1 per eps."""
    if not qsvm_whitebox_f1:
        raise ValueError("no QSVM white-box cell to read")
    return "SURVIVES" if all(v < floor for v in qsvm_whitebox_f1.values()) else "BUDGET-BOUND"


def crossing_rows(checks: dict) -> dict:
    """Attacked rows whose recomputed sequence state crosses a deadband, per eps and source,
    summed over seeds and relays: '<eps>|<src>' -> [rows crossing, attackable rows]."""
    out: dict = {}
    for unit_checks in checks.values():
        for c in unit_checks:
            if c["mode"] != "joint" or c["src"] == "projection_only":
                continue
            n = out.setdefault(f"{c['eps']}|{c['src']}", [0, 0])
            for rel in c["by_relay"].values():
                n[0] += rel["deadband_crossing"]["n_rows_any"]
                n[1] += rel["deadband_crossing"]["n_attackable_rows"]
    return out


def read_magnitude_arm(out: dict, checks: dict, cfg: dict, ref: dict | None) -> dict:
    """R4-06 read as E-M1c is: white-box per family with control 5 per cell, MLP-transfer minus
    random per tree, the asymmetry and the deadband crossings, each beside E-M1c's (ref, the
    frame joint run's results.json; None in smoke, whose rows and steps differ); the branch."""
    eps_grid, top = cfg["eps_grid"], max(cfg["eps_grid"])
    if ref is not None:
        for k in FACTORIAL_KEYS:
            if ref["config"][k] != cfg[k]:
                raise AssertionError(f"R4-06: {k} differs from E-M1c: {ref['config'][k]} {cfg[k]}")
    cell = {(c["family"], c["eps"]): c for c in out["control5_negative"]}
    rcell = {} if ref is None else {(c["family"], c["eps"]): c for c in ref["control5_negative"]}
    q = {e: cell[("qsvm", e)]["pgd_f1"] for e in eps_grid}
    whitebox = {
        f"{fam}|{e}": {
            "pgd_f1": cell[(fam, e)]["pgd_f1"],
            "random_f1": cell[(fam, e)]["random_f1"],
            "status": cell[(fam, e)]["status"],
            "e_m1c_pgd_f1": rcell[(fam, e)]["pgd_f1"] if rcell else None,
            "e_m1c_random_f1": rcell[(fam, e)]["random_f1"] if rcell else None,
            "e_m1c_status": rcell[(fam, e)]["status"] if rcell else None,
        }
        for fam in WHITEBOX
        for e in eps_grid
    }
    ex = out["branch_ns6"]["by_source"]["mlp"]["transfer_minus_random_damage"]
    rex = (
        {} if ref is None else ref["branch_ns6"]["by_source"]["mlp"]["transfer_minus_random_damage"]
    )
    keys = ("published_ratio_forward_over_reverse", "forward_drop_mlp_to_vqc")
    keys += ("reverse_drop_vqc_to_mlp", "ratio_denominator_small", "bounded_index")
    asym = {
        str(e): {
            "this": {k: out["asymmetry"][str(e)][k] for k in keys},
            "e_m1c": None if ref is None else {k: ref["asymmetry"][str(e)][k] for k in keys},
        }
        for e in eps_grid
    }
    cross = crossing_rows(checks)
    rcross = {} if ref is None else crossing_rows(ref["checks"])
    q_top = q[top]
    r_top = rcell[("qsvm", top)]["pgd_f1"] if rcell else None
    return {
        "experiment": "R4-06",
        "registered_floor": MAG_UNIT_FLOOR,
        "rule": "SURVIVES if the QSVM's seed-mean white-box F1 < the floor at every eps, "
        "BUDGET-BOUND otherwise",
        "qsvm_whitebox_f1": {str(e): v for e, v in q.items()},
        "qsvm_control5_status": {str(e): cell[("qsvm", e)]["status"] for e in eps_grid},
        "verdict": magnitude_verdict(q),
        "complete_grid": list(cfg["relays"]) == RELAYS and list(eps_grid) == EPS_GRID,
        "statistic_qsvm_whitebox_f1_top_eps": {
            "eps": top,
            "this": q_top,
            "e_m1c": r_top,
            "this_minus_e_m1c": None if r_top is None else q_top - r_top,
        },
        "whitebox": whitebox,
        "mlp_transfer_minus_random_damage": {k: {"this": ex[k], "e_m1c": rex.get(k)} for k in ex},
        "asymmetry": asym,
        "deadband_crossing_rows_any": {
            k: {"this": v, "e_m1c": rcross.get(k)} for k, v in sorted(cross.items())
        },
        "e_m1c_run": str(FRAME_JOINT_RESULTS.relative_to(REPO_ROOT)) if ref else None,
    }


def main_joint(args, T: int, W: int):
    """E-M1c (--angle-unit frame) and E-NS8(a) (marginal): --joint (smoke or full) and
    --joint-anchor."""
    if args.device != "cpu":
        raise SystemExit("--joint runs the derived map on the cpu")
    ref_run = ARM_RESULTS[args.angle_unit]
    if args.joint_anchor:
        fcfg = json.loads((ref_run / "results.json").read_text())["config"]
        if T != fcfg["threads"] or fcfg["angle_unit"] != args.angle_unit:
            raise SystemExit(
                f"the anchor needs the {args.angle_unit} arm's T={fcfg['threads']} (XGBoost)"
            )
        relays, seeds, eps_grid = [1], fcfg["seeds"], fcfg["eps_grid"]
    elif args.smoke:
        relays, seeds, eps_grid = RELAYS, [0, 1], [0.5]  # two units for control 7
    else:
        relays, seeds, eps_grid = args.relays, args.seeds, EPS_GRID
    sh, diag = build_shared(args, T, relays)
    if args.joint_anchor and (len(sh.ctx.y) != fcfg["n_eval"] or sh.ctx.n_steps != fcfg["n_steps"]):
        raise AssertionError(
            f"anchor: eval rows or PGD steps differ from the {args.angle_unit} arm"
        )
    frame = args.angle_unit == "frame"
    mag = args.magnitude_unit
    tag = joint_tag(relays)
    suffix = "_smoke" if args.smoke else "_anchor" if args.joint_anchor else ""
    arm = "_frame" if frame else ""
    arm += {"scale": "", "physical": "_physmag", "scale-equal": "_scaleeqmag"}[mag]
    name = f"consistent_relay_attack_{VARIANT}_q{QUBITS}{arm}_joint{suffix}"
    experiment = "E-M1c" if frame else "E-NS8(a)"
    if mag != "scale":
        experiment = "R4-06" if mag == "physical" else "R4-06 anchor (scale-equal)"
    cfg = {
        "experiment": experiment,
        "joint": True,
        "joint_tag": tag,
        "threads": T,
        "workers": W,
        "device": args.device,
        "angle_unit": args.angle_unit,
        "frame_unit_deg_PA1_PA6": {
            f"R{r}": None if rm.frame_unit is None else rm.frame_unit.tolist()
            for r, rm in sh.maps.items()
        },
        "magnitude_unit": mag,
        "magnitude_unit_V_PM1_PM3": {
            f"R{r}": {
                "unit": None if rm.mag_unit is None else rm.mag_unit.tolist(),
                "scaler_scale": sh.s[rm.ctrl_idx[:3]].tolist(),
            }
            for r, rm in sh.maps.items()
        },
        "smoke": args.smoke,
        "joint_anchor": args.joint_anchor,
        "relays": relays,
        "seeds": seeds,
        "eps_grid": eps_grid,
        "n_steps": sh.ctx.n_steps,
        "step_frac": sh.ctx.step_frac,
        "n_eval": int(len(sh.ctx.y)),
        "stage1_results": str(STAGE1_RESULTS.relative_to(REPO_ROOT)),
        f"{args.angle_unit}_results": str(ref_run.relative_to(REPO_ROOT)),
        **(
            {}
            if frame
            else {"frame_joint_results": str(FRAME_JOINT_RESULTS.relative_to(REPO_ROOT))}
        ),
        "resume": str(args.resume) if args.resume else None,
    }
    modes = ["consistent", "map_off"] if args.smoke else ["consistent"]
    specs = [{"relays": relays, "seed": s, "modes": modes, "eps_grid": eps_grid} for s in seeds]
    with RunDir(name, config=cfg, seeds=seeds) as run:
        run.write_json("controls_map.json", diag)
        extra = {}
        if args.smoke:
            one = run_pool(run_joint_unit, specs, sh, 1)
            peak = max(u["peak_rss_gb"] for u in one)
            print(f"measured peak RSS of one worker: {peak:.2f} GB", flush=True)
            extra["memory_guard"] = memory_guard(2, peak, "joint smoke pass 2")
            units = run_pool(run_joint_unit, specs, sh, 2)
            a, b = cells(one), cells(units)
            diff = {str(k): (a[k], b.get(k)) for k in a if a[k] != b.get(k)}
            extra["control7_parallel_invariance"] = {
                "n_cells": len(a),
                "n_differ": len(diff),
                "differ": diff,
                "pids_pass1": sorted({u["pid"] for u in one}),
                "pids_pass2": sorted({u["pid"] for u in units}),
                "peak_rss_gb_pass1": [u["peak_rss_gb"] for u in one],
                "peak_rss_gb_pass2": [u["peak_rss_gb"] for u in units],
            }
            if diff or len(a) != len(b):
                raise AssertionError(f"control 7: {len(diff)} cells differ between 1 and 2 workers")
        else:
            if args.worker_rss_gb is None:
                raise SystemExit("--worker-rss-gb is required (smoke prints the measured value)")
            extra["memory_guard"] = memory_guard(W, args.worker_rss_gb, "joint run")
            done = []
            if args.resume:
                done = json.loads(args.resume.read_text())["units"]
                if any(u["relays"] != relays for u in done):
                    raise AssertionError(f"resume: units of another relay set than {relays}")
                log.info("resuming: units already done %s", [u["unit"] for u in done])
            have = {u["seed"] for u in done}
            todo = [sp for sp in specs if sp["seed"] not in have]

            def save(res):
                done.append(res)
                run.write_json("results_partial.json", {"units": done})

            run_pool_all(run_joint_unit, todo, sh, W, on_done=save)
            units = done
        out = summarize_joint(units, sh, eps_grid, relays)
        anchor = compare_frame_arm(units, ref_run) if args.joint_anchor else None
        if anchor is not None:
            run.write_json("anchor.json", anchor)
        results = {
            "config": cfg,
            "stratified_random_floor": sh.ctx.floor,
            "controls_map": diag,
            "stage1_control2_mapping": sh.ctx.map_control,
            **extra,
            f"anchor_{args.angle_unit}_arm": anchor,
            "branch_ns6": out["branch_ns6"],
            "branch_kernels": out["branch_kernels"],
            "asymmetry": out["asymmetry"],
            "control5_negative": out["control5_negative"],
            "transfer_matrix": out["transfer_matrix"],
            "compare_stage1_a1": out["compare_stage1_a1"],
            "units_meta": [
                {k: u[k] for k in ("unit", "spaces", "timing_s", "wall_s", "peak_rss_gb", "pid")}
                for u in sorted(units, key=lambda u: u["seed"])
            ],
            "control2j_joint_gradient": {u["unit"]: u["control2j_joint_gradient"] for u in units},
            "checks": {u["unit"]: u["checks"] for u in units},
            "qsvm_gate": {u["unit"]: u["qsvm_gate"] for u in units},
            "stage1_control4_gradient": {u["unit"]: u["stage1_control4_gradient"] for u in units},
            "summary": out["summary"],
            "records": out["records"],
        }
        if mag == "physical":
            ref = None
            if not args.smoke:
                ref = json.loads((FRAME_JOINT_RESULTS / "results.json").read_text())
            y = sh.ctx.y
            const = f1_score(y, np.full_like(y, np.bincount(y).argmax()), average="macro")
            results["branch_magnitude_unit"] = read_magnitude_arm(out, results["checks"], cfg, ref)
            results["branch_magnitude_unit"]["floors"] = {
                "registered": MAG_UNIT_FLOOR,
                "constant_majority_macro_f1": float(const),
                "stratified_random_floor": sh.ctx.floor,
            }
            log.info(
                "R4-06 branch: %s (QSVM white-box F1 %s, floor %.3f)",
                results["branch_magnitude_unit"]["verdict"],
                results["branch_magnitude_unit"]["qsvm_whitebox_f1"],
                MAG_UNIT_FLOOR,
            )
        results["parent_peak_rss_gb"] = round(_peak_rss_gb(), 3)
        run.write_json("results.json", results)
        log.info("NS6 branch: %s", out["branch_ns6"]["verdict"])
        log.info("kernel branch: %s", out["branch_kernels"]["verdict"])
        if not (frame or args.smoke or args.joint_anchor):  # E-NS8(a), after results.json
            fac = factorial_ns8a(out["records"], cfg)
            run.write_json("factorial.json", fac)
            log.info("E-NS8(a) factorial: %s", fac["verdict"])
        print(f"-> {run.path}")
        if anchor is not None and not anchor["passed"]:
            raise AssertionError(
                f"{cfg['experiment']} anchor failed: {anchor['n_differ']} cells differ, "
                f"{anchor['n_missing']} missing (anchor.json)"
            )
        failed = [c for c in out["control5_negative"] if not c["pass"]]
        if failed and (args.smoke or args.joint_anchor):
            print(f"control 5 (recorded, not blocking here): random >= PGD in {failed}")
        elif failed:
            raise AssertionError(f"control 5 failed (random >= PGD damage) in {failed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--threads", type=int, required=True, help="BLAS/OpenMP/torch threads per worker"
    )
    ap.add_argument("--workers", type=int, default=1, help="parallel (relay, seed) units")
    ap.add_argument("--smoke", action="store_true", help="40 rows, seed 0, R1+R2, eps 0.5, 3 steps")
    ap.add_argument("--anchor-only", action="store_true", help="control 6b only, then stop")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--relays", type=int, nargs="*", default=RELAYS)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="derived-map device")
    ap.add_argument(
        "--angle-unit",
        choices=ANGLE_UNITS,
        default="marginal",
        help="phase-angle budget unit: marginal (registered) or frame (amendment arm)",
    )
    ap.add_argument(
        "--magnitude-unit",
        choices=MAGNITUDE_UNITS,
        default="scale",
        help="PM1..PM3 budget unit: scale (scaler.scale_), physical (R4-06, with --joint "
        "--angle-unit frame) or scale-equal (scaler.scale_ through the physical path, with "
        "--joint-anchor --angle-unit frame)",
    )
    ap.add_argument("--worker-rss-gb", type=float, help="per-worker peak RSS measured by smoke")
    ap.add_argument("--resume", type=Path, help="results_partial.json of an interrupted run")
    ap.add_argument(
        "--joint", action="store_true", help="E-M1c: every relay in --relays attacked at once"
    )
    ap.add_argument(
        "--joint-anchor",
        action="store_true",
        help="joint anchor: the joint mode on R1 must reproduce the single-relay arm's R1 cells",
    )
    args = ap.parse_args()
    if str(args.threads) != _THREADS:
        raise AssertionError(f"early thread parse {_THREADS} != argparse {args.threads}")
    T = args.threads
    n_cpu = psutil.cpu_count(logical=True)
    W = 2 if args.smoke else 1 if args.anchor_only else args.workers
    print(
        f"workers W={W} x threads T={T} = {W * T} logical threads of {n_cpu}; "
        f"{psutil.virtual_memory().available / 2**30:.1f} GB available; device {args.device}",
        flush=True,
    )
    if W * T > n_cpu:
        raise ValueError(f"W x T = {W * T} exceeds the {n_cpu} logical CPUs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but torch sees no CUDA device")
    if args.magnitude_unit != "scale":
        if args.angle_unit != "frame":
            raise SystemExit(f"--magnitude-unit {args.magnitude_unit} needs --angle-unit frame")
        if args.magnitude_unit == "physical" and not (args.joint and not args.joint_anchor):
            raise SystemExit("--magnitude-unit physical runs with --joint only")
        if args.magnitude_unit == "scale-equal" and not args.joint_anchor:
            raise SystemExit("--magnitude-unit scale-equal is the anchor: --joint-anchor only")
    if args.joint or args.joint_anchor:
        if args.anchor_only or (args.joint_anchor and args.smoke):
            raise SystemExit("--joint/--joint-anchor do not combine with --anchor-only or smoke")
        return main_joint(args, T, W)

    relays = [1, 2] if args.smoke else args.relays
    seeds = [0] if args.smoke else args.seeds
    eps_grid = [0.5] if args.smoke else EPS_GRID
    sh, diag = build_shared(args, T, relays)
    suffix = "_smoke" if args.smoke else "_anchor" if args.anchor_only else ""
    suffix = f"_frame{suffix}" if args.angle_unit == "frame" else suffix
    name = f"consistent_relay_attack_{VARIANT}_q{QUBITS}{suffix}"
    frame_units = {
        f"R{r}": None if rm.frame_unit is None else rm.frame_unit.tolist()
        for r, rm in sh.maps.items()
    }
    cfg = {
        "threads": T,
        "workers": W,
        "device": args.device,
        "angle_unit": args.angle_unit,
        "frame_unit_deg_PA1_PA6": frame_units,
        "smoke": args.smoke,
        "anchor_only": args.anchor_only,
        "relays": relays,
        "seeds": seeds,
        "eps_grid": eps_grid,
        "n_steps": sh.ctx.n_steps,
        "step_frac": sh.ctx.step_frac,
        "n_eval": int(len(sh.ctx.y)),
        "stage1_results": str(STAGE1_RESULTS.relative_to(REPO_ROOT)),
        "resume": str(args.resume) if args.resume else None,
    }

    if args.anchor_only:
        with RunDir(name, config=cfg, seeds=[0]) as run:
            run.write_json("controls_map.json", diag)
            res = run_pool(run_anchor, [{"relays": relays}], sh, 1)[0]
            run.write_json("anchor.json", res)
            print(f"-> {run.path}")
            if not res["passed"]:
                raise AssertionError(f"control 6b (anchor) failed: {json.dumps(res['cells'])}")
        return

    modes = ["consistent", "map_off"] if args.smoke else ["consistent"]
    specs = [
        {"relay": r, "seed": s, "modes": modes, "eps_grid": eps_grid, "device": args.device}
        for r in relays
        for s in seeds
    ]
    with RunDir(name, config=cfg, seeds=seeds) as run:
        run.write_json("controls_map.json", diag)
        extra = {}
        if args.smoke:
            # control 7: one worker, then two; the second pass is sized on the first's peak RSS
            one = run_pool(run_unit, specs, sh, 1)
            peak = max(u["peak_rss_gb"] for u in one)
            print(f"measured peak RSS of one worker: {peak:.2f} GB", flush=True)
            extra["memory_guard"] = memory_guard(2, peak, "smoke pass 2")
            units = run_pool(run_unit, specs, sh, 2)
            a, b = cells(one), cells(units)
            diff = {str(k): (a[k], b.get(k)) for k in a if a[k] != b.get(k)}
            extra["control7_parallel_invariance"] = {
                "n_cells": len(a),
                "n_differ": len(diff),
                "differ": diff,
                "pids_pass1": sorted({u["pid"] for u in one}),
                "pids_pass2": sorted({u["pid"] for u in units}),
                "peak_rss_gb_pass1": [u["peak_rss_gb"] for u in one],
                "peak_rss_gb_pass2": [u["peak_rss_gb"] for u in units],
            }
            if diff or len(a) != len(b):
                raise AssertionError(f"control 7: {len(diff)} cells differ between 1 and 2 workers")
        else:
            if args.worker_rss_gb is None:
                raise SystemExit("--worker-rss-gb is required (smoke prints the measured value)")
            extra["memory_guard"] = memory_guard(W, args.worker_rss_gb, "full run")
            done = []
            if args.resume:
                done = json.loads(args.resume.read_text())["units"]
                log.info("resuming: units already done %s", [u["unit"] for u in done])
            have = {(u["relay"], u["seed"]) for u in done}
            todo = [sp for sp in specs if (sp["relay"], sp["seed"]) not in have]

            def save(res):
                done.append(res)
                run.write_json("results_partial.json", {"units": done})

            run_pool_all(run_unit, todo, sh, W, on_done=save)
            units = done
        records, summary, ctl5, cmp_rows, branch = summarize(units, sh.ctx.floor, eps_grid)
        results = {
            "config": cfg,
            "angle_unit": args.angle_unit,
            "frame_unit_deg_PA1_PA6": frame_units,
            "stratified_random_floor": sh.ctx.floor,
            "controls_map": diag,
            "stage1_control2_mapping": sh.ctx.map_control,
            **extra,
            "branch": branch,
            "control5_negative": ctl5,
            "compare_stage1_a2": cmp_rows,
            "units_meta": [
                {k: u[k] for k in ("unit", "spaces", "timing_s", "wall_s", "peak_rss_gb", "pid")}
                for u in sorted(units, key=lambda u: (u["relay"], u["seed"]))
            ],
            "checks": {u["unit"]: u["checks"] for u in units},
            "qsvm_gate": {u["unit"]: u["qsvm_gate"] for u in units},
            "stage1_control4_gradient": {u["unit"]: u["stage1_control4_gradient"] for u in units},
            "summary": summary,
            "records": records,
        }
        run.write_json("results.json", results)
        log.info("branch: %s", json.dumps(branch))
        print(f"-> {run.path}")
        failed = [c for c in ctl5 if not c["pass"]]
        if failed and args.smoke:  # control 5 blocks the full run; smoke records it
            print(f"control 5 (recorded, not blocking in smoke): random >= PGD in {failed}")
        elif failed:
            raise AssertionError(f"control 5 failed (random >= PGD damage) in {failed}")


if __name__ == "__main__":
    main()
