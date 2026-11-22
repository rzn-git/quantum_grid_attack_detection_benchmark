"""R4-13: is the VQC's white-box fragility and its 2-sigma end state a property of the variational
family, or of the RY angle encoding? Four encoding arms at equal tuning budget, ten seeds each.

Full run (search, then finals; the lead launches it on a clean tree):
  uv run python -m experiments.run_vqc_encoding_axis --workers 2 --threads 3 --worker-rss-gb <peak>
Smoke (2 search trials of 2 epochs per arm into the run directory, finals at seed 0 only; smoke
numbers are not results):
  uv run python -m experiments.run_vqc_encoding_axis --smoke --workers 2 --threads 1 \
      --worker-rss-gb 0.9
Resume: add --resume results/runs/<stamp>_vqc_encoding_axis_binary_q8 (finished final units are
skipped; the Optuna searches resume from their sqlite stores on their own).

Pre-registration: decision log "2026-11-16T11:37Z — PRE-RUN AMENDMENT (round 4 part B: R4-13", and
the design note `tmp/review_shihav_20261116/r4_13_vqc_encoding_design.md`.

Arms (binary, PCA-8, 8 qubits, the shared 2,000-row `kernel_subset`):
  A0 deployed  `VQClassifier` (AngleEmbedding RY + StronglyEntanglingLayers), the robustness
               table's hyperparameters (`results/adv_best_params_binary_q8.json`); no search
  A1 angle     RY(c x) + StronglyEntanglingLayers
  A2 zz        the fidelity kernel's ZZ map at bandwidth c + StronglyEntanglingLayers
  A3 reupload  RY(c x) re-uploaded before each layer (P21's circuit)
A1-A3 share one search: 50 trials, `TPESampler(seed=0)`, serial per arm, objective best-epoch val
macro-F1 on the 2,000-row val cap (`run_quantum.run_vqc`'s), over lr, depth and batch from
`configs/quantum.yaml` plus the bandwidth c on the QSVM's box [0.05, 2.0] (log). Finals use
`FittedZoo`'s VQC protocol (40 epochs, patience 8, val cap 2,000) at seeds 0-9.

Readouts per (arm, seed), all in the shared pre-encoding space:
  R1 clean macro-F1 on the full test split and on the 500- and 200-row attack subsets
  R2 white-box BCE PGD and FGSM at every epsilon of `configs/adversarial.yaml` on the 500-row
     subset, with the MLP surrogate attacked beside it on the SAME `spawn_rng(seed, "adv")` stream
     in `run_adversarial.evasion_and_transfer`'s order, so A0 replays the published cells;
     retention = PGD F1 at 0.5 sigma over clean
  R3 2-sigma end state on the 200-row subset under the loss objective (p7b's stream) and the
     distance objective (E13's |<Z>| mass descent): mass ratio, F1 - floor, `end_state` label
  R4 transfer at 0.5 sigma both ways: MLP-crafted to the arm, arm-crafted to the MLP
  R5 random +-eps signs at 0.5 sigma (the null)
Branches (seed means; an arm is unreadable when its clean F1 on the 500-row subset is within 0.05
of the constant-predictor floor):
  ENCODING-SPECIFIC  any readable arm has retention > 0.30, or is `disabled` at 2 sigma under the
                     loss objective
  ENCODING-GENERAL   otherwise
Controls:
  A0 anchors (blocking): per-seed clean and PGD-0.5 VQC F1, and the MLP-surrogate transfer to the
    VQC, equal the archived `20260813T183448Z_adversarial_binary_q8` within 1e-6 (Table II's
    0.5018 / 0.0361 / 0.0719 are their seed means); E13's loss-objective |<Z>| profile at seed 0
    within 0.02 (`20260814T100533Z_whitebox_qsvm_binary_q8`)
  A2's encoding state equals `compute_states(.., "zz", c)` to 1e-10; backprop equals
    parameter-shift input gradients on 3 rows within 1e-6 for A2 and A3 (blocking, per unit)
  P9b's negative control (blocking, PGD cells only, ruling 2026-11-16T12:18Z): no PGD score above
    clean; FGSM scores above clean are reported as overshoot; PGD at 0.5 sigma against the
    random-sign damage is reported per cell
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
import multiprocessing  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from functools import cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

# torch, pennylane and the VQC are imported inside the workers' functions: the main process only
# schedules, and importing them there costs ~0.5 GB of the run's memory budget
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402

log = get_logger(__name__)

VARIANT, QUBITS = "binary", 8
SEEDS = tuple(range(10))
ARMS = {
    "A0_deployed": {"encoding": None, "reupload": False},
    "A1_angle": {"encoding": "angle_ry", "reupload": False},
    "A2_zz": {"encoding": "zz", "reupload": False},
    "A3_reupload": {"encoding": "angle_ry", "reupload": True},
}
SEARCHED = ("A1_angle", "A2_zz", "A3_reupload")
BW_BOX = (0.05, 2.0)  # the QSVM's bandwidth box (configs/quantum.yaml qkernel.bandwidth)
FINAL_EPOCHS, FINAL_PATIENCE = 40, 8  # FittedZoo's VQC (run_adversarial.FittedZoo)
VAL_CAP = 2000
END_EPS = 2.0
RETENTION_LINE = 0.30
READABLE_MARGIN = 0.05
ANCHOR_RUN = REPO_ROOT / "results" / "runs" / "20260813T183448Z_adversarial_binary_q8"
E13_LOSS_MASS = {  # p7b / E13 positive control, artifact 20260814T100533Z_whitebox_qsvm_binary_q8
    "0.01": 0.9825,
    "0.05": 0.9248,
    "0.1": 0.8869,
    "0.2": 0.9805,
    "0.3": 1.2102,
    "0.5": 1.8585,
    "1.0": 2.7157,
    "2.0": 2.6684,
}
ANCHOR_TOL, E13_TOL, STATE_TOL, GRAD_TOL = 1e-6, 0.02, 1e-10, 1e-6


@cache
def encoded_vqc_class():
    """The `EncodedVQC` class, built on first use so the scheduler never imports torch."""
    import pennylane as qml

    from qgridbench.models.quantum.qkernel import feature_map
    from qgridbench.models.quantum.vqc import VQClassifier

    class EncodedVQC(VQClassifier):
        """`VQClassifier` with the encoding as a variable: `feature_map(c * x)` (the fidelity
        kernel's own map) before the StronglyEntanglingLayers, or re-uploaded before each layer.
        Only the circuit changes; weights, head, loss, optimiser and early stopping are the
        deployed ones."""

        def __init__(self, *args, encoding: str, bandwidth: float, reupload: bool, **kw):
            self.encoding, self.bandwidth, self.reupload = (
                encoding,
                float(bandwidth),
                bool(reupload),
            )
            super().__init__(*args, **kw)

        def _build_qnodes(self) -> None:
            n, enc, c = self.n_qubits, self.encoding, self.bandwidth

            def circuit(x, weights):
                if self.reupload:
                    for layer in range(self.depth):
                        feature_map(c * x, enc, n)
                        qml.StronglyEntanglingLayers(weights[layer : layer + 1], wires=range(n))
                else:
                    feature_map(c * x, enc, n)
                    qml.StronglyEntanglingLayers(weights, wires=range(n))
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_out)]

            self._circuit = circuit
            dev = qml.device(self.device_name, wires=n)
            self._qnode = qml.QNode(circuit, dev, interface="torch", diff_method=self.diff_method)
            if self.grad_method == self.diff_method:
                self._qnode_grad = self._qnode
            else:  # the parameter-shift reference, as VQClassifier builds it
                dev_g = qml.device("lightning.qubit", wires=n)
                self._qnode_grad = qml.QNode(
                    circuit, dev_g, interface="torch", diff_method=self.grad_method
                )

    return EncodedVQC


def make_vqc(arm: str, params: dict, seed: int, max_epochs: int, patience: int, **kw):
    common = dict(
        n_classes=2,
        lr=params["lr"],
        batch_size=params["batch_size"],
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        **kw,
    )
    if arm == "A0_deployed":
        from qgridbench.models.quantum.vqc import VQClassifier

        return VQClassifier(QUBITS, depth=params["depth"], **common)
    a = ARMS[arm]
    return encoded_vqc_class()(
        QUBITS,
        depth=params["depth"],
        encoding=a["encoding"],
        bandwidth=params["bandwidth"],
        reupload=a["reupload"],
        **common,
    )


# --------------------------------------------------------------------------- pure reading


def _ms(xs) -> dict:
    a = np.asarray(xs, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def negative_control(clean: float, r2: dict) -> dict:
    """P9b's hard control, as ruled for R4-13 (decision log 2026-11-16T12:18Z): an attacked PGD
    score above clean on the same subset blocks. FGSM is one step and overshoots on a periodic map,
    so an FGSM score above clean is reported as measured overshoot and does not block."""
    return {
        "pgd_above_clean": [e for e, v in r2["pgd"].items() if v > clean + 1e-9],
        "fgsm_overshoot_above_clean": [e for e, v in r2["fgsm"].items() if v > clean + 1e-9],
    }


def arm_summary(units: list[dict]) -> dict:
    clean = [u["clean"]["eval500"] for u in units]
    adv = [u["r2"]["pgd"]["0.5"] for u in units]
    floor = units[0]["floor500"]
    from experiments.run_length_scale_per_class import end_state

    end = {}
    for obj in ("loss", "distance"):
        mass = float(np.mean([u["r3"][obj]["mass_ratio"] for u in units]))
        d = float(np.mean([u["r3"][obj]["f1_minus_floor"] for u in units]))
        end[obj] = {"mass_ratio": mass, "f1_minus_floor": d, "label": end_state(mass, d)}
    c, a = float(np.mean(clean)), float(np.mean(adv))
    return {
        "n_seeds": len(units),
        "clean500": _ms(clean),
        "pgd_0.5": _ms(adv),
        "retention_0.5": a / c,
        "floor500": floor,
        "readable": bool(c - floor > READABLE_MARGIN),
        "end_state_2sigma": end,
        "transfer_mlp_to_arm_0.5": _ms([u["r4"]["mlp_to_arm"] for u in units]),
        "transfer_arm_to_mlp_0.5": _ms([u["r4"]["arm_to_mlp"] for u in units]),
        "random_0.5": _ms([u["r5"]["random_f1"] for u in units]),
        "clean_full_test": _ms([u["clean"]["full_test"] for u in units]),
        "fgsm_overshoot_cells": sum(
            len(u["negative_control"]["fgsm_overshoot_above_clean"]) for u in units
        ),
        "pgd_beats_random_seeds": sum(u["r5"]["pgd_beats_random"] for u in units),
    }


def branch(summaries: dict[str, dict]) -> dict:
    readable = {k: v for k, v in summaries.items() if v["readable"]}
    hits = {
        k: {
            "retention_above_line": bool(v["retention_0.5"] > RETENTION_LINE),
            "disabled_under_loss": v["end_state_2sigma"]["loss"]["label"] == "disabled",
        }
        for k, v in readable.items()
    }
    specific = [k for k, h in hits.items() if any(h.values())]
    return {
        "readable_arms": sorted(readable),
        "unreadable_arms": sorted(set(summaries) - set(readable)),
        "per_arm": hits,
        "verdict": "ENCODING-SPECIFIC" if specific else "ENCODING-GENERAL",
        "arms_driving_specific": specific,
        "rule": f"retention > {RETENTION_LINE} or disabled at 2 sigma under loss, readable arms",
    }


# --------------------------------------------------------------------------- context

_CTX = None


def _init_worker(threads: int, chunk: int):
    import torch

    want = str(threads)
    got = {v: os.environ.get(v) for v in THREAD_VARS}
    if any(x != want for x in got.values()):
        raise AssertionError(f"worker {os.getpid()}: thread caps {got}, expected {want}")
    torch.set_num_threads(threads)
    os.environ["QGB_VQC_CHUNK"] = str(chunk)
    global _CTX
    _CTX = _context()


def _context() -> dict:
    from qgridbench.models.classical.zoo import stratified_cap
    from qgridbench.protocol import kernel_subset, prepare_regime

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    reg = prepare_regime(VARIANT, "pca", n_components=QUBITS, seed=0)
    sub = kernel_subset(reg, cfg_q["qkernel"]["subset_cap"], seed=0)
    va = stratified_cap(reg.X["val"], reg.y["val"], VAL_CAP, seed=0)
    ev = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["eval_subset_adv"], seed=0)
    e2 = stratified_cap(reg.X["test"], reg.y["test"], cfg_adv["blackbox"]["eval_subset"] * 2, 0)
    A = reg.angle_scaler.transform
    return {
        "reg": reg,
        "cfg_adv": cfg_adv,
        "cfg_q": cfg_q,
        "Xtr": reg.X["train"][sub],
        "ytr": reg.y["train"][sub],
        "A_tr": A(reg.X["train"][sub]),
        "A_va": A(reg.X["val"][va]),
        "y_va": reg.y["val"][va],
        "X500": reg.X["test"][ev],
        "y500": reg.y["test"][ev],
        "X200": reg.X["test"][e2],
        "y200": reg.y["test"][e2],
        "scale": np.asarray(reg.feature_scale, dtype=np.float64),
    }


def _floor(y) -> float:
    from experiments.run_adversarial import _f1

    maj = int(np.round(y.mean()))
    return _f1(y, np.column_stack([1 - np.full(len(y), maj), np.full(len(y), maj)]).astype(float))


# --------------------------------------------------------------------------- stage 1: search


def search_unit(spec: dict) -> dict:
    """One arm's 50-trial search (smoke: 2 trials of 2 epochs), serial, resumable from sqlite."""
    import optuna

    from qgridbench.utils.seeding import set_all_seeds

    c, o = _CTX, _CTX["cfg_q"]["vqc"]["optuna"]
    arm = spec["arm"]
    depths = _CTX["cfg_q"]["vqc"]["depths"]
    epochs = spec["tuning_epochs"]
    patience = _CTX["cfg_q"]["vqc"]["patience"]

    def objective(trial):
        lr = trial.suggest_float("lr", o["lr"]["low"], o["lr"]["high"], log=o["lr"]["log"])
        depth = trial.suggest_categorical("depth", depths)
        bs = trial.suggest_categorical("batch_size", o["batch_size"])
        bw = trial.suggest_float("bandwidth", *BW_BOX, log=True)
        set_all_seeds(0)
        p = {"lr": lr, "depth": depth, "batch_size": bs, "bandwidth": bw}
        m = make_vqc(arm, p, 0, epochs, patience)
        m.fit(c["A_tr"], c["ytr"], c["A_va"], c["y_va"])
        return m.best_val_f1_

    study = optuna.create_study(
        direction="maximize",
        study_name=f"vqc_encoding_{arm}_{VARIANT}_q{QUBITS}",
        storage=spec["storage"],
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    t0 = time.perf_counter()
    n_done = len([t for t in study.trials if t.state.is_finished()])
    if spec["n_trials"] - n_done > 0:
        study.optimize(objective, n_trials=spec["n_trials"] - n_done, show_progress_bar=False)
    from experiments.run_xgb_thread_sensitivity import _peak_rss_gb

    vals = [t.value for t in study.trials if t.value is not None]
    return {
        "kind": "search",
        "arm": arm,
        "best_params": study.best_params,
        "best_val_macro_f1": study.best_value,
        "n_trials": len(vals),
        "trial_val_median": float(np.median(vals)),
        "trial_val_min": float(np.min(vals)),
        "wall_s": time.perf_counter() - t0,
        "peak_rss_gb": _peak_rss_gb(),
    }


# --------------------------------------------------------------------------- stage 2: finals


def gates(m, arm: str, c: dict) -> dict:
    """A2's state equals the fidelity kernel's; backprop equals parameter-shift input gradients."""
    import pennylane as qml
    import torch

    from qgridbench.models.quantum.qkernel import compute_states, feature_map

    out = {}
    A = np.asarray(c["A_tr"][:3], dtype=np.float64)  # the VQC runs in float64
    if arm == "A2_zz":
        dev = qml.device("default.qubit", wires=QUBITS)

        @qml.qnode(dev, interface="torch")
        def st(x):
            feature_map(m.bandwidth * x, "zz", QUBITS)
            return qml.state()

        mine = st(torch.as_tensor(A)).detach().numpy()
        ref = compute_states(A, "zz", m.bandwidth, dtype=np.complex128)
        out["state_max_abs_err"] = float(np.abs(mine - ref).max())
        if out["state_max_abs_err"] > STATE_TOL:
            raise AssertionError(f"A2 state gate: {out['state_max_abs_err']:.2e}")
    if arm in ("A2_zz", "A3_reupload"):
        ref = make_vqc(arm, m._params, m.seed, 1, 1, grad_method="parameter-shift")
        with torch.no_grad():
            for p_ref, p in zip(
                (ref.weights, ref.head_scale, ref.head_bias),
                (m.weights, m.head_scale, m.head_bias),
            ):
                p_ref.copy_(p)
        y = c["ytr"][:3]
        g_bp, g_ps = m.loss_input_grad(A, y), ref.loss_input_grad(A, y)
        out["grad_max_abs_err"] = float(np.abs(g_bp - g_ps).max())
        if out["grad_max_abs_err"] > GRAD_TOL:
            raise AssertionError(f"{arm} gradient gate: {out['grad_max_abs_err']:.2e}")
    return out


def final_unit(spec: dict) -> dict:
    import torch

    from experiments.run_adversarial import _f1
    from experiments.run_length_scale_per_class import end_state
    from experiments.run_vqc_objective_axis import vqc_mass_grad
    from experiments.run_whitebox_qsvm import _mass_descent
    from experiments.run_xgb_thread_sensitivity import _peak_rss_gb
    from experiments.runner_guard import assert_clean_predictor
    from qgridbench.attacks.evasion import fgsm, pgd
    from qgridbench.attacks.gradients import GradProvider
    from qgridbench.models.classical.zoo import build, fit_model, predict_proba_timed
    from qgridbench.utils.seeding import set_all_seeds, spawn_rng

    c, arm, seed = _CTX, spec["arm"], spec["seed"]
    t0 = time.perf_counter()
    sc, ang = c["scale"], c["reg"].angle_scaler
    pgd_cfg = c["cfg_adv"]["evasion"]["pgd"]
    eps_grid = c["cfg_adv"]["evasion"]["epsilons"]
    steps, frac = pgd_cfg["n_steps"], pgd_cfg["step_frac"]

    # the MLP surrogate exactly as FittedZoo fits it (random_state = seed)
    mlp = build("mlp", spec["mlp_params"], seed)
    fit_model("mlp", mlp, c["Xtr"], c["ytr"])
    gp_mlp = GradProvider(mlp, "mlp")

    set_all_seeds(seed)  # FittedZoo seeds immediately before the VQC
    m = make_vqc(arm, spec["params"], seed, FINAL_EPOCHS, FINAL_PATIENCE)
    m._params = spec["params"]
    m.fit(c["A_tr"], c["ytr"], c["A_va"], c["y_va"])
    gate = gates(m, arm, c)

    def proba(X):
        return m.predict_proba(ang.transform(X))

    def mlp_proba(X):
        return predict_proba_timed(mlp, X)[0]

    def gp_arm(X, y):
        return m.loss_input_grad(ang.transform(X), y) * ang._mm.scale_

    X, y = c["X500"], c["y500"]
    p_clean = proba(X)
    assert_clean_predictor(f"{arm} seed {seed}", y, p_clean, p_clean.argmax(1))
    clean500 = _f1(y, p_clean)
    p_full = proba(c["reg"].X["test"])
    clean = {
        "eval500": clean500,
        "full_test": _f1(c["reg"].y["test"], p_full),
        "eval200": _f1(c["y200"], proba(c["X200"])),
    }

    # R2 + R4 on run_adversarial.evasion_and_transfer's stream and order (mlp, then the arm)
    rng = spawn_rng(seed, "adv")
    r2 = {"pgd": {}, "fgsm": {}}
    max_eps = max(eps_grid)
    arm_to_mlp = None
    for eps in eps_grid:
        for src in ("mlp", "arm"):
            gp = gp_mlp if src == "mlp" else gp_arm
            # FGSM draws nothing from the stream, so only the arm's is computed
            Xp = pgd(gp, X, y, eps, sc, n_steps=steps, step_frac=frac, rng=rng)
            if src == "arm":
                r2["fgsm"][str(eps)] = _f1(y, proba(fgsm(gp, X, y, eps, sc)))
                r2["pgd"][str(eps)] = _f1(y, proba(Xp))
                if eps == max_eps:
                    arm_to_mlp = _f1(y, mlp_proba(Xp))
    for eps in eps_grid:  # the surrogate loop consumes the stream exactly as published
        pgd(gp_mlp, X, y, eps, sc, n_steps=steps, step_frac=frac, rng=rng)
    # the published transfer row crafts a FRESH max-eps set per target family, in the order
    # CLASSICAL + qsvm + vqc; each pgd call draws exactly one uniform start of X's shape, so the
    # targets before the VQC are consumed as draws and the VQC's own set is crafted on its draw
    from experiments.run_adversarial import CLASSICAL

    for _ in [*CLASSICAL, "qsvm"]:
        rng.uniform(-1.0, 1.0, size=X.shape)
    X_tr = pgd(gp_mlp, X, y, max_eps, sc, n_steps=steps, step_frac=frac, rng=rng)
    r4 = {"mlp_to_arm": _f1(y, proba(X_tr)), "arm_to_mlp": arm_to_mlp, "eps": max_eps}
    negative = negative_control(clean500, r2)

    # R5 random +-eps signs at 0.5 sigma
    r_rng = spawn_rng(seed, "r4_13_random")
    Xr = X + r_rng.choice([-1.0, 1.0], size=X.shape) * 0.5 * sc
    r5 = {
        "random_f1": _f1(y, proba(Xr)),
        "pgd_beats_random": bool(r2["pgd"]["0.5"] <= _f1(y, proba(Xr))),
    }

    # R3 end state on the 200-row subset: loss (p7b's stream) and distance (E13's descent)
    X2, y2 = c["X200"], c["y200"]
    floor200 = _floor(y2)

    def expv(Z):
        with torch.no_grad():
            return np.abs(m._expvals(torch.as_tensor(ang.transform(Z)), m.weights).numpy())

    m0 = float(expv(X2).mean())

    def read(Xa):
        f1 = _f1(y2, proba(Xa))
        return {
            "mass_ratio": float(expv(Xa).mean() / m0),
            "macro_f1": f1,
            "f1_minus_floor": f1 - floor200,
        }

    radii = sorted(set(eps_grid) | {1.0, END_EPS})
    loss_curve = {
        str(e): read(
            pgd(
                gp_arm,
                X2,
                y2,
                e,
                sc,
                n_steps=steps,
                step_frac=frac,
                rng=spawn_rng(seed, f"p7b_vqc_{e}"),
            )
        )
        for e in radii
    }
    Xd = _mass_descent(
        lambda Z: vqc_mass_grad(m, ang, Z, m.chunk),
        X2,
        END_EPS,
        sc,
        spawn_rng(seed, "r4_13_descent"),
        steps,
        frac,
    )
    r3 = {"loss": loss_curve[str(END_EPS)], "distance": read(Xd), "loss_curve": loss_curve}
    for obj in ("loss", "distance"):
        r3[obj]["label"] = end_state(r3[obj]["mass_ratio"], r3[obj]["f1_minus_floor"])

    return {
        "kind": "final",
        "arm": arm,
        "seed": seed,
        "params": spec["params"],
        "best_epoch": getattr(m, "best_epoch_", None),
        "clean": clean,
        "floor500": _floor(y),
        "floor200": floor200,
        "r2": r2,
        "r3": r3,
        "r4": r4,
        "r5": r5,
        "negative_control": negative,
        "gates": gate,
        "unit_wall_s": time.perf_counter() - t0,
        "peak_rss_gb": _peak_rss_gb(),
        "pid": os.getpid(),
    }


# --------------------------------------------------------------------------- anchors


def a0_anchors(units: list[dict]) -> dict:
    arch = json.loads((ANCHOR_RUN / "results.json").read_text())["per_seed"]
    rows = []
    for u in units:
        if u["arm"] != "A0_deployed":
            continue
        e = arch[u["seed"]]["evasion"]
        for name, got, want, tol in (
            ("clean500", u["clean"]["eval500"], e["clean"]["vqc"], ANCHOR_TOL),
            ("pgd_0.5", u["r2"]["pgd"]["0.5"], e["curves"]["vqc"]["pgd"]["0.5"], ANCHOR_TOL),
            (
                "mlp_to_vqc_0.5",
                u["r4"]["mlp_to_arm"],
                e["transfer_matrix"]["mlp_surrogate"]["vqc"],
                ANCHOR_TOL,
            ),
        ):
            rows.append(
                {
                    "seed": u["seed"],
                    "cell": name,
                    "got": got,
                    "archived": want,
                    "pass": abs(got - want) <= tol,
                }
            )
        if u["seed"] == 0:
            for e_key, want in E13_LOSS_MASS.items():
                got = u["r3"]["loss_curve"][e_key]["mass_ratio"]
                rows.append(
                    {
                        "seed": 0,
                        "cell": f"e13_loss_mass_{e_key}",
                        "got": got,
                        "archived": want,
                        "pass": abs(got - want) <= E13_TOL,
                    }
                )
    return {"rows": rows, "pass": bool(rows) and all(r["pass"] for r in rows)}


# --------------------------------------------------------------------------- main


def _pool(fn, specs, workers, threads, chunk, on_done):
    failed = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(threads, chunk),
    ) as ex:
        futs = {ex.submit(fn, s): s for s in specs}
        for f in as_completed(futs):
            exc = f.exception()
            if exc is not None:
                log.error("unit %s failed", futs[f], exc_info=exc)
                failed.append(futs[f])
                continue
            on_done(f.result())
    if failed:
        raise RuntimeError(f"{len(failed)} units failed (finished ones are saved): {failed}")


def _storage(store: Path, arm: str) -> str:
    db = store / f"optuna_vqc_encoding_{arm}_{VARIANT}_q{QUBITS}.db"
    return f"sqlite:///{db.as_posix()}"


def main() -> None:
    from experiments.run_xgb_thread_sensitivity import _peak_rss_gb, memory_guard

    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--worker-rss-gb", type=float, required=True)
    ap.add_argument("--budget-gb", type=float, default=2.0)
    ap.add_argument("--chunk", type=int, default=512, help="QGB_VQC_CHUNK per worker")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()

    mem = memory_guard(
        args.workers, args.worker_rss_gb, args.budget_gb - _peak_rss_gb(), "R4-13 workers"
    )
    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    best = json.loads((REPO_ROOT / "results" / "adv_best_params_binary_q8.json").read_text())
    seeds = SEEDS[:1] if args.smoke else SEEDS
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    cfg = {
        "smoke": args.smoke,
        "workers": args.workers,
        "threads": args.threads,
        "chunk": args.chunk,
        "arms": ARMS,
        "bandwidth_box": BW_BOX,
        "search_trials": 2 if args.smoke else cfg_q["vqc"]["optuna"]["n_trials"],
        "final": {"epochs": FINAL_EPOCHS, "patience": FINAL_PATIENCE, "val_cap": VAL_CAP},
        "anchor_run": str(ANCHOR_RUN.relative_to(REPO_ROOT)),
        "prereg": "decision log 2026-11-16T11:37Z, R4-13",
        "memory": mem,
    }
    name = f"vqc_encoding_axis_{VARIANT}_q{QUBITS}" + ("_smoke" if args.smoke else "")
    with RunDir(name, config=cfg, seeds=list(seeds)) as run:
        (run.path / "git_status_at_launch.txt").write_text(porcelain)
        state = {"search": {}, "final": {}}
        if args.resume:
            state = json.loads((args.resume / "results_partial.json").read_text())
            log.info("resume from %s", args.resume)

        def save(u):
            key = u["arm"] if u["kind"] == "search" else f"{u['arm']}_s{u['seed']}"
            state[u["kind"]][key] = u
            run.write_json("results_partial.json", state)
            log.info(
                "%s done in %.0fs, peak %.2f GB",
                key,
                u.get("wall_s", u.get("unit_wall_s")),
                u["peak_rss_gb"],
            )

        # stage 1: one serial search per arm, in parallel across arms
        store = run.path if args.smoke else REPO_ROOT / "results"
        s_specs = [
            {
                "arm": a,
                "storage": _storage(store, a),
                "n_trials": cfg["search_trials"],
                "tuning_epochs": 2 if args.smoke else cfg_q["vqc"]["tuning_max_epochs"],
            }
            for a in SEARCHED
            if a not in state["search"]
        ]
        if s_specs:
            _pool(search_unit, s_specs, args.workers, args.threads, args.chunk, save)

        # stage 2: (arm, seed) finals
        a0 = {
            "lr": best["vqc_lr"],
            "depth": best["vqc_depth"],
            "batch_size": best["vqc_batch"],
        }
        if best.get("vqc_epochs", FINAL_EPOCHS) != FINAL_EPOCHS:
            raise AssertionError(f"deployed epochs {best['vqc_epochs']} != {FINAL_EPOCHS}")
        params = {"A0_deployed": a0, **{a: state["search"][a]["best_params"] for a in SEARCHED}}
        f_specs = [
            {"arm": a, "seed": s, "params": params[a], "mlp_params": best["mlp"]}
            for s in seeds
            for a in ARMS
            if f"{a}_s{s}" not in state["final"]
        ]
        _pool(final_unit, f_specs, args.workers, args.threads, args.chunk, save)

        finals = list(state["final"].values())
        anchors = a0_anchors(finals)
        gates_ok = all(not u["negative_control"]["pgd_above_clean"] for u in finals)
        out = {"anchors": anchors, "negative_controls_pass": gates_ok, **state}
        if not anchors["pass"] or not gates_ok:
            run.write_json("results.json", out)
            raise AssertionError(f"anchors or negative controls failed: {anchors['rows']}")
        summaries = {a: arm_summary([u for u in finals if u["arm"] == a]) for a in ARMS}
        out["summaries"] = summaries
        if not args.smoke:
            out["branch"] = branch(summaries)
        peak = max(u["peak_rss_gb"] for u in finals + list(state["search"].values()))
        out["peak_rss_gb"] = {"worker_max": peak, "main": _peak_rss_gb()}
        run.write_json("results.json", out)
        run.metrics.update(
            {
                "anchors_pass": anchors["pass"],
                "worker_peak_rss_gb": peak,
                **({"verdict": out["branch"]["verdict"]} if not args.smoke else {"smoke": True}),
            }
        )
        log.info("anchors %s | %s", anchors["pass"], out.get("branch", "smoke: no verdict"))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
