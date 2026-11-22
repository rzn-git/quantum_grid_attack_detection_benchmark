"""R4-21 (N3): is clean-accuracy selection blind to security on the kernel hyperparameters?
Assembles P9b, P9a, P11d and P8a, and computes the cells they leave missing.

Run (full; about 10 minutes at 2 workers x 3 threads, peak about 0.9 GB per worker):
  uv run python -m experiments.run_n3_selection_assembly --workers 2 --threads 3 --worker-rss-gb 0.9
Smoke (one point per knob, one attack seed per objective; numbers are smoke, not results):
  uv run python -m experiments.run_n3_selection_assembly --smoke --workers 1 --threads 2 \
      --worker-rss-gb 0.9
Resume an interrupted run:  add --resume results/runs/<stamp>_n3_selection_assembly_binary_q8

Pre-registration: decision log "2026-11-16T11:37Z — PRE-RUN AMENDMENT (round 4 part B: R4-13",
item R4-21, and the design note `tmp/review_shihav_20261116/r4_21_n3_design.md`.

Knobs (binary, PCA-8, the shared 2,000-row train subset `stratified_cap(train, 2000, 0)`):
  K1 quantum bandwidth, ZZ fidelity kernel q8, C re-tuned per bandwidth on clean val (P9b's grid
     and C, archived in `20260823T210050Z_adversarial_bandwidth_selection_binary_q8`)
  K2 quantum C at the tuned bandwidth 0.7932, C in {1, 10, 100, 835.2} (P11d, part B of
     `20260814T200349Z_whitebox_qsvm_binary_q8`)
  K3 classical RBF gamma, P8a's 8 multipliers of the tuned 4.19 at C held at 403.3
     (`20260814T111131Z_whitebox_qsvm_binary_q8`)

Per point:
  clean   clean val macro-F1 on P9b's 2,000-row val sample (the accuracy protocol's own view)
  S1      adversarial macro-F1 under P9b's white-box margin PGD at 0.5 sigma (40 steps, step 0.1),
          val 100 and test 100, attack seeds 0-2 (`pgd_margin(..., rng_seed=s)`; seed 0 is P9b's)
  S2      end state at 2 sigma on P8a/P9a/P11d's 200 test rows (20 steps, step 0.25), attack seeds
          1-3 (`default_rng(s)`, P11d's), under three objectives: loss (BCE ascent), margin
          (P14a's non-saturating ascent) and mass (P11c's mass descent). Quantum decisions and
          kernel mass go through the exact complex128 Gram (P9a, P11d). Label: R4-03's
          `end_state` (`tables._regime` with `partial` at the floor named `at_floor`) on the
          seed-mean mass ratio and seed-mean F1 - floor.
  K3's loss-objective S2 is P8a's archived cell (5 attack seeds); K1's clean val is P9b's archived
  curve. K1 points are computed only where P9b's archived clean val is inside the tie band.

Claim (N3): for knob k, A_k = argmax clean val over the knob's grid, T_k = points with clean val
>= clean(A_k) - 0.02. Knob k HOLDS if within T_k the S1 test mean (3 seeds) spans >= 0.05, or
under one named objective T_k holds both an `inverted` and a `disabled`/`at_floor` S2 label.
N3 HOLDS on >= 2 of 3 knobs; KILL on <= 1. Riders (no branch): direction, clean cost, box reach.

Anchors (blocking, 1e-6): P9b bw 0.8 and 3.0 at C 1000, clean val and seed-0 adversarial val;
P11d clean test-200 at C 10 and 835.2; P8a clean test-200 at gamma 4.19 and 0.1257.
Reported anchor: K2 mass-descent per-seed F1 - floor and mass ratio against P11d's archive.
Negative controls (P9b's, blocking, every S1 cell): eps 0 is a no-op; no adversarial score exceeds
the clean score on the same subset; `assert_clean_predictor` on every clean decision.
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402

log = get_logger(__name__)

RUNS = REPO_ROOT / "results" / "runs"
P9B = RUNS / "20260823T210050Z_adversarial_bandwidth_selection_binary_q8"
P9A = RUNS / "20260814T180008Z_whitebox_qsvm_binary_q8"
P11D = RUNS / "20260814T200349Z_whitebox_qsvm_binary_q8"
P8A = RUNS / "20260814T111131Z_whitebox_qsvm_binary_q8"
BEST_PARAMS = REPO_ROOT / "results" / "adv_best_params_binary_q8.json"

VARIANT, QUBITS, ENCODING = "binary", 8, "zz"
TRAIN_CAP, CLEAN_CAP, S1_CAP = 2000, 2000, 100
S1_EPS, S2_EPS = 0.5, 2.0
S1_SEEDS, S2_SEEDS = (0, 1, 2), (1, 2, 3)
OBJECTIVES = ("loss", "margin", "mass")
TIE_BAND = 0.02
S1_BAR = 0.05
KNOBS_TO_HOLD = 2
ANCHOR_TOL = 1e-6
INVERTED = ("inverted",)
FLOOR_END = ("disabled", "at_floor")
# the accuracy protocol's search boxes (configs/quantum.yaml qkernel; zoo.py rbf_svm search)
BOX = {"K1": (0.05, 2.0), "K2": (0.01, 1000.0), "K3": (1e-4, 10.0)}


# --------------------------------------------------------------------------- pure reading


def _ms(xs) -> dict:
    a = np.asarray(xs, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def knob_verdict(points: list[dict], knob: str, tie_band: float = TIE_BAND) -> dict:
    """The N3 rule for one knob. Each point: {"value", "clean_val", "s1_test" (seed mean or
    None), "s2": {objective: label}}. Fails loud if a tie-set member lacks a reading."""
    if not points:
        raise ValueError(f"{knob}: no points")
    acc = max(points, key=lambda p: p["clean_val"])
    tie = [p for p in points if p["clean_val"] >= acc["clean_val"] - tie_band]
    missing = [p["value"] for p in tie if p["s1_test"] is None]
    if missing:
        raise ValueError(f"{knob}: tie-set points without S1 test: {missing}")
    s1 = [p["s1_test"] for p in tie]
    s1_spread = max(s1) - min(s1)
    s2_both = {}
    for obj in OBJECTIVES:
        labels = [p["s2"].get(obj) for p in tie]
        if any(lab is None for lab in labels):
            raise ValueError(f"{knob}: tie-set points without an S2 {obj} label")
        s2_both[obj] = bool(
            any(lab in INVERTED for lab in labels) and any(lab in FLOOR_END for lab in labels)
        )
    best = max(tie, key=lambda p: p["s1_test"])
    lo, hi = BOX[knob]
    return {
        "accuracy_selected": acc["value"],
        "accuracy_clean_val": acc["clean_val"],
        "tie_set": [p["value"] for p in tie],
        "s1_test_in_tie_set": dict(zip([str(p["value"]) for p in tie], s1)),
        "s1_spread": s1_spread,
        "s1_holds": bool(s1_spread >= S1_BAR),
        "s2_labels_in_tie_set": {str(p["value"]): p["s2"] for p in tie},
        "s2_both_ends_by_objective": s2_both,
        "s2_holds": bool(any(s2_both.values())),
        "holds": bool(s1_spread >= S1_BAR or any(s2_both.values())),
        "rider_direction_acc_minus_best_s1": acc["s1_test"] - best["s1_test"],
        "rider_accuracy_pick_less_secure": bool(best["s1_test"] - acc["s1_test"] >= S1_BAR),
        "rider_clean_cost_of_s1_best": acc["clean_val"] - best["clean_val"],
        "rider_s1_best_value": best["value"],
        "rider_s1_best_inside_search_box": bool(lo <= best["value"] <= hi),
    }


def n3_verdict(knobs: dict[str, dict]) -> dict:
    n = sum(v["holds"] for v in knobs.values())
    return {
        "knobs_holding": [k for k, v in knobs.items() if v["holds"]],
        "n_holding": int(n),
        "verdict": "N3 HOLDS" if n >= KNOBS_TO_HOLD else "KILL",
        "rule": f">= {KNOBS_TO_HOLD} of 3 knobs; tie band {TIE_BAND}; S1 bar {S1_BAR}",
    }


def s2_label(seed_cells: list[dict]) -> dict:
    from experiments.run_length_scale_per_class import end_state

    mass = float(np.mean([c["mass_ratio"] for c in seed_cells]))
    d = float(np.mean([c["f1_minus_floor"] for c in seed_cells]))
    return {"mass_ratio": mass, "f1_minus_floor": d, "label": end_state(mass, d)}


# --------------------------------------------------------------------------- archive


def _load(path: Path, key: str | None = None) -> dict:
    d = json.loads((path / "results.json").read_text())
    return d[key] if key else d


def archive() -> dict:
    p9b = _load(P9B)
    p11d = _load(P11D, "p11c_objective_and_regularization")["part_b_regularization"]
    p8a = _load(P8A, "p8a_lengthscale_control")
    p9a = _load(P9A, "p9a_quantum_bandwidth_regime")
    best = json.loads(BEST_PARAMS.read_text())
    return {"p9b": p9b, "p11d": p11d, "p8a": p8a, "p9a": p9a, "best": best}


def plan(arc: dict, smoke: bool) -> list[dict]:
    q = arc["p9b"]["quantum_curve"]
    top = max(p["clean_val"] for p in q)
    k1 = [p for p in q if p["clean_val"] >= top - TIE_BAND]
    bw_t = float(arc["best"]["qk_bw"])
    c_vals = sorted(float(c) for c in arc["p11d"])
    rbf = arc["best"]["rbf_svm"]
    specs = [{"knob": "K1", "value": p["bandwidth"], "bw": p["bandwidth"], "C": p["C"]} for p in k1]
    specs += [{"knob": "K2", "value": c, "bw": bw_t, "C": c} for c in c_vals]
    specs += [
        {"knob": "K3", "value": v["gamma"], "mult": float(m), "gamma": v["gamma"], "C": rbf["C"]}
        for m, v in arc["p8a"]["per_width"].items()
    ]
    if smoke:
        keep = {("K1", 3.0), ("K2", 10.0), ("K3", 0.03)}
        specs = [
            s for s in specs if (s["knob"], s["value"] if s["knob"] != "K3" else s["mult"]) in keep
        ]
    for s in specs:
        s["unit"] = f"{s['knob']}_{s['value']:.6g}"
    return specs


# --------------------------------------------------------------------------- worker

_CTX = None


def _init_worker(threads: int, smoke: bool):
    import torch

    want = str(threads)
    got = {v: os.environ.get(v) for v in THREAD_VARS}
    if any(x != want for x in got.values()):
        raise AssertionError(f"worker {os.getpid()}: thread caps {got}, expected {want}")
    torch.set_num_threads(threads)
    global _CTX
    from qgridbench.models.classical.zoo import stratified_cap
    from qgridbench.protocol import prepare_regime

    cfg_adv = load_yaml(REPO_ROOT / "configs" / "adversarial.yaml")
    reg = prepare_regime(VARIANT, "pca", n_components=QUBITS, seed=0)
    tr = stratified_cap(reg.X["train"], reg.y["train"], TRAIN_CAP, seed=0)
    vc = stratified_cap(reg.X["val"], reg.y["val"], CLEAN_CAP, seed=0)
    va = stratified_cap(reg.X["val"], reg.y["val"], S1_CAP, seed=0)
    te = stratified_cap(reg.X["test"], reg.y["test"], S1_CAP, seed=0)
    n2 = cfg_adv["blackbox"]["eval_subset"] * 2
    t2 = stratified_cap(reg.X["test"], reg.y["test"], n2, seed=0)
    y2 = reg.y["test"][t2]
    maj = int(np.round(y2.mean()))
    from experiments.run_adversarial import _f1

    floor = _f1(
        y2, np.column_stack([1 - np.full(len(y2), maj), np.full(len(y2), maj)]).astype(float)
    )
    _CTX = {
        "reg": reg,
        "Xtr": reg.X["train"][tr],
        "ytr": reg.y["train"][tr],
        "Xvc": reg.X["val"][vc],
        "yvc": reg.y["val"][vc],
        "Xva": reg.X["val"][va],
        "yva": reg.y["val"][va],
        "Xte": reg.X["test"][te],
        "yte": reg.y["test"][te],
        "X2": reg.X["test"][t2],
        "y2": y2,
        "floor": floor,
        "scale": np.asarray(reg.feature_scale, dtype=np.float64),
        "s2_steps": cfg_adv["evasion"]["pgd"]["n_steps"],
        "s2_step_frac": cfg_adv["evasion"]["pgd"]["step_frac"],
        "s1_seeds": S1_SEEDS[:1] if smoke else S1_SEEDS,
        "s2_seeds": S2_SEEDS[:1] if smoke else S2_SEEDS,
    }


def _s1_cells(wb, decision, c: dict) -> dict:
    """P9b's S1 on val 100 and test 100 with its negative controls."""
    from experiments.run_adversarial_bandwidth_selection import _f1 as f1_dec
    from experiments.run_adversarial_bandwidth_selection import pgd_margin
    from experiments.runner_guard import assert_clean_predictor

    out = {}
    for split, X, y in (("val", c["Xva"], c["yva"]), ("test", c["Xte"], c["yte"])):
        dec = decision(X)
        assert_clean_predictor(f"S1 clean {split}", y, dec, dec > 0)
        clean = f1_dec(y, dec)
        zero = f1_dec(y, decision(pgd_margin(wb, X, y, 0.0, c["scale"])))
        if abs(zero - clean) > 1e-12:
            raise AssertionError(f"eps 0 is not a no-op on {split}: {zero} vs {clean}")
        adv = []
        for s in c["s1_seeds"]:
            a = f1_dec(y, decision(pgd_margin(wb, X, y, S1_EPS, c["scale"], rng_seed=s)))
            if a > clean + 1e-9:
                raise AssertionError(f"adversarial {a} exceeds clean {clean} on {split}, seed {s}")
            adv.append(a)
        out[split] = {"clean": clean, "adv_per_seed": adv, "adv": _ms(adv)}
    return out


def _s2_cells(grads: dict, decision, mass_rows, c: dict) -> dict:
    """End state at 2 sigma on the 200 test rows, per objective and attack seed."""
    from experiments.run_adversarial import _f1
    from experiments.run_whitebox_qsvm import _mass_descent
    from qgridbench.attacks.evasion import pgd

    X, y, sc = c["X2"], c["y2"], c["scale"]
    m0 = mass_rows(X).mean()
    out = {}
    for obj in OBJECTIVES:
        cells = []
        for s in c["s2_seeds"]:
            rng = np.random.default_rng(s)
            if obj == "mass":
                Xa = _mass_descent(
                    grads["mass"], X, S2_EPS, sc, rng, c["s2_steps"], c["s2_step_frac"]
                )
            else:
                Xa = pgd(grads[obj], X, y, S2_EPS, sc, c["s2_steps"], c["s2_step_frac"], rng)
            d = decision(Xa)
            p1 = 1.0 / (1.0 + np.exp(-d))
            f1 = _f1(y, np.column_stack([1 - p1, p1]))
            cells.append(
                {
                    "seed": s,
                    "mass_ratio": float(mass_rows(Xa).mean() / m0),
                    "macro_f1": f1,
                    "f1_minus_floor": f1 - c["floor"],
                    "modal_frac": float(max((d > 0).mean(), (d <= 0).mean())),
                }
            )
        out[obj] = {"per_seed": cells, **s2_label(cells)}
    return out


def _quantum_unit(spec: dict, c: dict) -> dict:
    from sklearn.svm import SVC

    from experiments.run_adversarial_bandwidth_selection import _f1 as f1_dec
    from experiments.run_whitebox_qsvm import _qsvm_mass_grad
    from qgridbench.attacks.qsvm_whitebox import QSVMWhiteBox
    from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
    from qgridbench.utils.seeding import set_all_seeds

    reg, bw, C = c["reg"], spec["bw"], spec["C"]
    A = reg.angle_scaler.transform
    s_tr = compute_states(A(c["Xtr"]), ENCODING, bandwidth=bw)
    set_all_seeds(0)
    svc = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(
        fidelity_kernel(s_tr, s_tr), c["ytr"]
    )
    s_vc = compute_states(A(c["Xvc"]), ENCODING, bandwidth=bw)
    clean_val = f1_dec(c["yvc"], svc.decision_function(fidelity_kernel(s_vc, s_tr)))
    wb = QSVMWhiteBox(svc, s_tr, reg.angle_scaler, ENCODING, bw)

    def gram_exact(Z):
        return fidelity_kernel(compute_states(A(Z), ENCODING, bw), s_tr, exact=True)

    def dec_exact(Z):
        return svc.decision_function(gram_exact(Z))

    def mass_rows(Z):
        return gram_exact(Z)[:, svc.support_]

    clean2 = f1_dec(c["y2"], dec_exact(c["X2"]))
    s1 = _s1_cells(wb, wb.decision_function, c)
    grads = {
        "loss": wb.loss_input_grad,
        "margin": wb.margin_input_grad,
        "mass": lambda Z: _qsvm_mass_grad(wb, Z),
    }
    s2 = _s2_cells(grads, dec_exact, mass_rows, c)
    return {
        "clean_val": clean_val,
        "clean_test200": clean2,
        "n_support": int(len(svc.support_)),
        "s1": s1,
        "s2": s2,
    }


def _rbf_mass_grad(est, gamma):
    """d/dx of the mean RBF kernel to the support set (P11c's classical control)."""
    sv = est.support_vectors_

    def g(Z):
        d = Z[:, None, :] - sv[None, :, :]
        K = np.exp(-gamma * (d**2).sum(axis=2))
        return (K[:, :, None] * (-2.0 * gamma) * d).mean(axis=1)

    return g


def _classical_unit(spec: dict, c: dict) -> dict:
    from experiments.run_adversarial_bandwidth_selection import RBFWhiteBox
    from experiments.run_adversarial_bandwidth_selection import _f1 as f1_dec
    from qgridbench.attacks.gradients import GradProvider
    from qgridbench.models.classical.zoo import build, fit_model
    from qgridbench.utils.seeding import set_all_seeds

    gamma = spec["gamma"]
    set_all_seeds(0)
    est = build("rbf_svm", {"C": spec["C"], "gamma": gamma}, 0)
    fit_model("rbf_svm", est, c["Xtr"], c["ytr"])
    clean_val = f1_dec(c["yvc"], est.decision_function(c["Xvc"]))
    clean2 = f1_dec(c["y2"], est.decision_function(c["X2"]))
    wb = RBFWhiteBox(est, gamma)
    s1 = _s1_cells(wb, est.decision_function, c)
    sv = est.support_vectors_

    def mass_rows(Z):
        return np.exp(-gamma * ((Z[:, None, :] - sv[None, :, :]) ** 2).sum(axis=2))

    grads = {
        "loss": GradProvider(est, "rbf_svm", "bce"),
        "margin": GradProvider(est, "rbf_svm", "margin"),
        "mass": _rbf_mass_grad(est, gamma),
    }
    s2 = _s2_cells(grads, est.decision_function, mass_rows, c)
    return {
        "clean_val": clean_val,
        "clean_test200": clean2,
        "n_support": int(len(sv)),
        "s1": s1,
        "s2": s2,
    }


def run_unit(spec: dict) -> dict:
    from experiments.run_xgb_thread_sensitivity import _peak_rss_gb

    t0 = time.perf_counter()
    out = (_classical_unit if spec["knob"] == "K3" else _quantum_unit)(spec, _CTX)
    return {
        **spec,
        **out,
        "floor_test200": _CTX["floor"],
        "unit_wall_s": time.perf_counter() - t0,
        "peak_rss_gb": _peak_rss_gb(),
        "pid": os.getpid(),
    }


# --------------------------------------------------------------------------- assembly


def anchors(arc: dict, units: dict[str, dict], smoke: bool) -> dict:
    """Blocking exact anchors on the recomputed points that the archive also holds."""
    rows = []

    def check(name, got, want, tol=ANCHOR_TOL):
        rows.append({"anchor": name, "got": got, "archived": want, "pass": abs(got - want) <= tol})

    qc = {p["bandwidth"]: p for p in arc["p9b"]["quantum_curve"]}
    for bw in (0.8, 3.0):
        u = units.get(f"K1_{bw:.6g}")
        if u is None:
            continue
        check(f"P9b bw {bw} clean val", u["clean_val"], qc[bw]["clean_val"])
        check(
            f"P9b bw {bw} clean val-100", u["s1"]["val"]["clean"], qc[bw]["clean_val_attack_subset"]
        )
        check(f"P9b bw {bw} adv val seed 0", u["s1"]["val"]["adv_per_seed"][0], qc[bw]["adv_val"])
    for key, cv in arc["p11d"].items():
        u = units.get(f"K2_{float(key):.6g}")
        if u is not None and float(key) in (10.0, float(arc["best"]["qk_C"])):
            check(
                f"P11d C {float(key):.6g} clean test-200", u["clean_test200"], cv["clean_macro_f1"]
            )
    for m, v in arc["p8a"]["per_width"].items():
        u = units.get(f"K3_{v['gamma']:.6g}")
        if u is not None and float(m) in (1.0, 0.03):
            check(
                f"P8a gamma x{m} clean test-200", u["clean_test200"], v["curve"]["0.0"]["macro_f1"]
            )
    # reported, not blocking: K2's recomputed mass-descent cells against P11d's per-seed archive
    reported = []
    for key, cv in arc["p11d"].items():
        u = units.get(f"K2_{float(key):.6g}")
        if u is None:
            continue
        arch = {p["seed"]: p for p in cv["per_seed"]}
        for cell in u["s2"]["mass"]["per_seed"]:
            a = arch[cell["seed"]]
            reported.append(
                {
                    "C": float(key),
                    "seed": cell["seed"],
                    "d_f1_minus_floor": cell["f1_minus_floor"] - a["f1_minus_floor"],
                    "d_mass_ratio": cell["mass_ratio"] - a["mass_ratio"],
                }
            )
    expected = 5 if smoke else 10
    return {
        "blocking": rows,
        "n_blocking_expected_full": 10,
        "complete": len(rows) == expected,
        "pass": bool(rows) and all(r["pass"] for r in rows),
        "reported_k2_mass_vs_p11d": reported,
        "reported_k2_mass_max_abs_diff": max(
            (max(abs(r["d_f1_minus_floor"]), abs(r["d_mass_ratio"])) for r in reported),
            default=None,
        ),
    }


def assemble(arc: dict, units: dict[str, dict]) -> dict:
    """Points per knob from computed units plus archived cells, then the N3 rule."""
    from experiments.run_length_scale_per_class import end_state

    def s1t(u):
        return u["s1"]["test"]["adv"]["mean"]

    def labels(u, archived_loss=None):
        lab = {o: u["s2"][o]["label"] for o in OBJECTIVES}
        if archived_loss is not None:
            lab["loss"] = archived_loss
        return lab

    knobs: dict[str, list[dict]] = {"K1": [], "K2": [], "K3": []}
    by_unit = {u["unit"]: u for u in units.values()}
    for p in arc["p9b"]["quantum_curve"]:
        u = by_unit.get(f"K1_{p['bandwidth']:.6g}")
        knobs["K1"].append(
            {
                "value": p["bandwidth"],
                "clean_val": p["clean_val"],  # archived: the protocol's selection curve
                "s1_test": s1t(u) if u else None,
                "s2": labels(u) if u else {},
                "computed": u is not None,
            }
        )
    for key in arc["p11d"]:
        u = by_unit[f"K2_{float(key):.6g}"]
        knobs["K2"].append(
            {
                "value": float(key),
                "clean_val": u["clean_val"],
                "s1_test": s1t(u),
                "s2": labels(u),
                "computed": True,
            }
        )
    for m, v in arc["p8a"]["per_width"].items():
        u = by_unit[f"K3_{v['gamma']:.6g}"]
        c2 = v["curve"]["2.0"]
        knobs["K3"].append(
            {
                "value": v["gamma"],
                "mult": float(m),
                "clean_val": u["clean_val"],
                "s1_test": s1t(u),
                "s2": labels(u, end_state(c2["kernel_mass_ratio"], c2["f1_minus_floor"])),
                "s2_loss_source": "P8a archive (5 attack seeds)",
                "computed": True,
            }
        )
    per_knob = {k: knob_verdict(pts, k) for k, pts in knobs.items()}
    return {"points": knobs, "knobs": per_knob, "n3": n3_verdict(per_knob)}


# --------------------------------------------------------------------------- main


def main() -> None:
    from experiments.run_xgb_thread_sensitivity import _peak_rss_gb, memory_guard

    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--worker-rss-gb", type=float, required=True)
    ap.add_argument("--budget-gb", type=float, default=2.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()

    # the budget covers the main process too: workers get what the main process leaves
    mem = memory_guard(
        args.workers, args.worker_rss_gb, args.budget_gb - _peak_rss_gb(), "R4-21 workers"
    )
    arc = archive()
    specs = plan(arc, args.smoke)
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    cfg = {
        "smoke": args.smoke,
        "workers": args.workers,
        "threads": args.threads,
        "units": [s["unit"] for s in specs],
        "tie_band": TIE_BAND,
        "s1_bar": S1_BAR,
        "knobs_to_hold": KNOBS_TO_HOLD,
        "s1": f"margin PGD {S1_EPS} sigma, 40 steps, step 0.1, seeds {list(S1_SEEDS)}",
        "s2": f"{S2_EPS} sigma, objectives {list(OBJECTIVES)}, seeds {list(S2_SEEDS)}",
        "archive": [str(p.relative_to(REPO_ROOT)) for p in (P9B, P9A, P11D, P8A)],
        "prereg": "decision log 2026-11-16T11:37Z, R4-21",
        "memory": mem,
    }
    name = f"n3_selection_assembly_{VARIANT}_q{QUBITS}" + ("_smoke" if args.smoke else "")
    with RunDir(name, config=cfg, seeds=list(S1_SEEDS)) as run:
        (run.path / "git_status_at_launch.txt").write_text(porcelain)
        done: dict[str, dict] = {}
        if args.resume:
            prev = json.loads((args.resume / "results_partial.json").read_text())
            done = {u["unit"]: u for u in prev["units"]}
            log.info("resume: %d units from %s", len(done), args.resume)
        todo = [s for s in specs if s["unit"] not in done]
        # quantum units first (the slow ones), K2's large-C points first among them
        todo.sort(key=lambda s: (s["knob"] == "K3", -s["C"]))

        def save():
            run.write_json("results_partial.json", {"units": list(done.values())})

        failed = []
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(args.threads, args.smoke),
        ) as ex:
            futs = {ex.submit(run_unit, s): s for s in todo}
            for f in as_completed(futs):
                exc = f.exception()
                if exc is not None:
                    log.error("unit %s failed", futs[f]["unit"], exc_info=exc)
                    failed.append(futs[f]["unit"])
                    continue
                u = f.result()
                done[u["unit"]] = u
                save()
                log.info(
                    "%s | clean val %.4f | S1 test %.4f | S2 loss/margin/mass "
                    "%s/%s/%s | %.0fs %.2fGB",
                    u["unit"],
                    u["clean_val"],
                    u["s1"]["test"]["adv"]["mean"],
                    u["s2"]["loss"]["label"],
                    u["s2"]["margin"]["label"],
                    u["s2"]["mass"]["label"],
                    u["unit_wall_s"],
                    u["peak_rss_gb"],
                )
        if failed:
            raise RuntimeError(
                f"units failed (finished ones are in results_partial.json): {failed}"
            )

        anc = anchors(arc, done, args.smoke)
        out = {"anchors": anc, "units": list(done.values())}
        if not anc["pass"] or not anc["complete"]:
            run.write_json("results.json", out)
            raise AssertionError(f"anchor failed or incomplete: {anc['blocking']}")
        if not args.smoke:
            out.update(assemble(arc, done))
        out["peak_rss_gb"] = {
            "main": _peak_rss_gb(),
            "worker_max": max(u["peak_rss_gb"] for u in done.values()),
        }
        run.write_json("results.json", out)
        run.metrics.update(
            {
                "anchors_pass": anc["pass"],
                "worker_peak_rss_gb": out["peak_rss_gb"]["worker_max"],
                "main_peak_rss_gb": out["peak_rss_gb"]["main"],
                **(
                    {
                        "verdict": out["n3"]["verdict"],
                        "knobs_holding": out["n3"]["knobs_holding"],
                    }
                    if not args.smoke
                    else {"smoke": True}
                ),
            }
        )
        log.info("anchors %s | %s", anc["pass"], out.get("n3", "smoke: no verdict"))
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
