"""R4-09 (NS1(b)): refit the binary PCA-8 benchmark with phase angles in each relay's own frame.

Each phase angle PA1..PA12 of relay R<k> is replaced by wrap(PA - angle(V1)) in degrees, wrapped
to (-180, 180], where angle(V1) is R<k>-PA7:VH (dataset README: "PA7:VH - PA9:VH Pos. - Neg. -
Zero Voltage Phase Angle"; the 2026-11-16T06:55Z feasibility probe found PM7:V = |V1| exactly).
PA:ZH (impedance angle) is already relative and is untouched. The reference channel itself becomes
identically 0 and is KEPT (the 128-column layout is unchanged; StandardScaler gives a zero-variance
column scale 1, so it enters the PCA as a zero column). The transform is computed in float64 from
the loaded array and cast back to its dtype (float32 on the shipped row split, float64 on the
whole-file path), so the pipeline downstream sees the dtype its archived run saw.

Arms:
  absolute   the shipped features (the anchor arm)
  relative   the registered transform
  permuted   control: the same transform with each row's reference angle taken from a random
             other row of the same split part (train, val or test); one cyclic derangement per
             split, drawn from spawn_rng(0, "permuted|<split>")
Splits:
  row        the shipped stratified row split (data/processed), float32 as Table I
  wf0..wf4   the paper's five whole-file partitions (`run_group_split_control.group_splits`,
             default_rng(0) consumed in order, held-out files asserted against Table I's), float64
             from `reconstruct_groups` as the archived whole-file runs
Pipeline per (arm, split): median impute -> StandardScaler -> PCA(8), fit on train only; the
[-pi, pi] AngleScaler fit on train for the quantum models. Families and frozen hyperparameters are
Table I's PCA-8 runs (read from their results.json), on the shared 2,000-row stratified subset:
  logreg rbf_svm rf xgb lgbm mlp   20260813T064540Z_classical_binary_pca8 (run_classical's
                                   build_pinned / fit_model / predict_proba_timed / compute_all)
  qsvm                             20260813T071427Z_quantum_qkernel_binary_q8, ZZ map, deterministic
                                   (one fit per (arm, split), as run_quantum)
  vqc                              20260813T201901Z_quantum_vqc_binary_q8 (run_quantum's final loop)
Memory: every (arm, split) view is built once, by a throwaway process per protocol, and saved
under the run directory's cache/ (deleted at the end); the fitting workers load only the saved
8-dimensional views, so their peak is the models', not the raw arrays'.
Both splits use Table I's set: the archived whole-file PCA-8 run
(`20260816T023620Z_group_split_control_binary`) fits only XGBoost and the ZZ QSVM, with
parameters identical to Table I's.
Every clean prediction passes `runner_guard.assert_clean_predictor`.

Threads: every worker pins its pool's thread count before numpy. XGBoost's hist fit depends on its
thread count (decision log 2026-08-15T20:09Z); Table I's PCA-8 XGBoost reproduces at 4 threads
(R4-01, and this item's probe: seeds 0 and 1 exact, every other count 1..20 off), so EVERY XGBoost
unit, in every arm and split, fits with n_jobs=XGB_THREADS so the gain is thread-matched and the
absolute row arm is Table I's row itself. XGBoost's n_jobs is capped by the process's OpenMP
thread count (the first smoke fitted at 2 under OMP_NUM_THREADS=2), so the units run in three pools,
one after another, each a single process group whose thread variables equal its count:
XGBoost at XGB_THREADS, the whole-file XGBoost anchor at WF_XGB_ANCHOR_THREADS, and everything else
at --threads. The pool's count reaches the spawned workers through POOL_THREADS_ENV, read before
numpy. The other seven families reproduced Table I exactly at 3 threads (probe) and at 2 (smoke).

Statistic (decision log 2026-11-16T11:07Z, R4-09): per family, the mean over seeds of the paired
test macro-F1 gain (relative - absolute), per split; on the whole-file split the mean over the five
partitions x ten seeds. The mean over the eight families per split decides:
  HYPOTHESIS  whole-file mean gain >= 0.02
  KILL        mean gain < 0.02 under both splits
  ROW_ONLY    otherwise (the row-split gain alone), reported as registered
Reported, not branched on: the permuted arm's gain over absolute (it should not beat absolute
systematically), and the share of rows whose relay V1 magnitude (PM7) is at the deadband zero.

Anchors (blocking): absolute/row reproduces Table I's per-seed macro-F1 for all eight families
within 1e-6. Whole-file (reported): absolute/wf<k> QSVM against the archived matched_quantum, and
XGBoost seed 0 at the archive's unpinned thread count (24 logical CPUs) against matched_classical.

Run (W x T threads; W x worker peak + the parent must fit 80% of free RAM; QGB_VQC_CHUNK=256
bounds the VQC's state batch and is recorded in the run config):
  QGB_VQC_CHUNK=256 uv run python -m experiments.run_frame_relative_angles --smoke --workers 1 \
      --threads 2 --worker-rss-gb 1.3
  QGB_VQC_CHUNK=256 uv run python -m experiments.run_frame_relative_angles --workers 1 \
      --threads 2 --worker-rss-gb <smoke peak>
Resume: add --resume results/runs/<stamp>_frame_relative_angles_binary_q8 (finished units skip).
Test: uv run pytest -q tests/test_frame_relative_angles.py
"""

from __future__ import annotations

import argparse
import os
import sys

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int)
_THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
POOL_THREADS_ENV = "QGB_R409_POOL_THREADS"  # set by the parent for one pool's spawned workers
if os.environ.get(POOL_THREADS_ENV):
    _THREADS = int(os.environ[POOL_THREADS_ENV])
if _THREADS is not None:
    for _var in THREAD_VARS:
        os.environ[_var] = str(_THREADS)
elif __name__ == "__main__" and not {"-h", "--help"} & set(sys.argv):
    raise SystemExit("--threads is required (it is applied before numpy is imported)")

import json  # noqa: E402
import multiprocessing  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger  # noqa: E402

log = get_logger(__name__)

RUNS = REPO_ROOT / "results" / "runs"
TABLE_I_CLASSICAL = RUNS / "20260813T064540Z_classical_binary_pca8"
TABLE_I_QSVM = RUNS / "20260813T071427Z_quantum_qkernel_binary_q8"
TABLE_I_VQC = RUNS / "20260813T201901Z_quantum_vqc_binary_q8"
WHOLE_FILE_ARCHIVE = RUNS / "20260816T023620Z_group_split_control_binary"
VARIANT = "binary"
QUBITS = 8
ENCODING = "zz"
CLASSICAL = ("logreg", "rbf_svm", "rf", "xgb", "lgbm", "mlp")
FAMILIES = (*CLASSICAL, "qsvm", "vqc")
ARMS = ("absolute", "relative", "permuted")
SPLITS = ("row", "wf0", "wf1", "wf2", "wf3", "wf4")
SEEDS = tuple(range(10))
XGB_THREADS = 4  # Table I's PCA-8 XGBoost thread count (R4-01; this item's probe)
WF_XGB_ANCHOR_THREADS = 24  # the archive ran unpinned on 24 logical CPUs (its compute_env)
REF_CHANNEL = 7  # PA7:VH = angle(V1)
ANGLE_RE = re.compile(r"^R([1-4])-PA(\d+):(VH|IH)$")
PERM_SEED = 0
ANCHOR_TOL = 1e-6
GAIN_LINE = 0.02
MEM_FRACTION = 0.8
# long units first (longest-processing-time order), so the pool's tail is short
COST_ORDER = {"vqc": 0, "qsvm": 1, "xgb": 2, "lgbm": 3, "rf": 4, "rbf_svm": 5, "mlp": 6}


# ----------------------------------------------------------------------------- the transform


def angle_channels(names: list[str]) -> dict[int, dict[int, int]]:
    """{relay: {k: column index}} for PA1..PA12 of each relay; fails loud on a gap."""
    out: dict[int, dict[int, int]] = {}
    for j, n in enumerate(names):
        m = ANGLE_RE.match(n)
        if m:
            out.setdefault(int(m.group(1)), {})[int(m.group(2))] = j
    for r, chans in out.items():
        if sorted(chans) != list(range(1, 13)):
            raise ValueError(f"relay R{r}: phase-angle channels {sorted(chans)}, expected 1..12")
        if names[chans[REF_CHANNEL]] != f"R{r}-PA{REF_CHANNEL}:VH":
            raise ValueError(f"relay R{r}: reference is {names[chans[REF_CHANNEL]]}")
    if sorted(out) != [1, 2, 3, 4]:
        raise ValueError(f"relays found {sorted(out)}, expected 1..4")
    return out


def wrap_deg(d: np.ndarray) -> np.ndarray:
    """Wrap degrees to (-180, 180]."""
    return 180.0 - np.mod(180.0 - d, 360.0)


def frame_relative(
    X: np.ndarray, chans: dict[int, dict[int, int]], ref_rows: np.ndarray | None = None
) -> np.ndarray:
    """wrap(PA_k - PA7) per relay, in float64, cast back to X's dtype.

    `ref_rows[i]` is the row whose reference angle row i uses (the permuted control); None means
    the row's own. Every other column is returned unchanged. The cast can round a value just above
    -180 to -180 (float32), which is re-wrapped to +180, the same angle, so the output stays in
    (-180, 180] at its own precision. One float64 working copy (memory budget)."""
    out = X.astype(np.float64)
    refs = {
        r: (out[:, c[REF_CHANNEL]] if ref_rows is None else out[ref_rows, c[REF_CHANNEL]]).copy()
        for r, c in chans.items()
    }
    for r, c in chans.items():
        for j in c.values():
            out[:, j] = wrap_deg(out[:, j] - refs[r])
    out = out.astype(X.dtype, copy=False)
    for c in chans.values():
        for j in c.values():
            col = out[:, j]
            col[col == -180.0] = 180.0
    return out


def derangement_within(parts: dict[str, np.ndarray], n: int, rng) -> np.ndarray:
    """ref[i] = another row of i's own part: a cyclic shift of a random order, per part."""
    ref = np.full(n, -1, dtype=np.int64)
    for name in ("train", "val", "test"):
        idx = parts[name]
        if len(idx) < 2:
            raise ValueError(f"part {name} has {len(idx)} rows; a derangement needs 2")
        order = rng.permutation(idx)
        ref[order] = np.roll(order, -1)
    if (ref < 0).any():
        raise ValueError(f"{int((ref < 0).sum())} rows belong to no split part")
    if (ref == np.arange(n)).any():
        raise AssertionError("derangement has a fixed point")
    return ref


# ----------------------------------------------------------------------------- data, per worker

_CACHE: dict = {}


def _feature_names() -> list[str]:
    meta = json.loads((REPO_ROOT / "data" / "processed" / f"meta_{VARIANT}.json").read_text())
    return meta["feature_names"]


WF_CACHE_ENV = "QGB_R409_WF_CACHE"  # directory of the whole-file arrays, set by the parent


def prepare_wf_cache(cache: str) -> dict:
    """Write `reconstruct_groups`' float64 X, y and file ids once, as .npy (run in a throwaway
    process: the CSV pooling peaks at several hundred MB, which a long-lived worker would keep as
    its peak). Checked against the processed array before it is written."""
    from experiments.run_group_split_control import reconstruct_groups
    from qgridbench.data.preprocess import load_processed

    Xp, yp, _ = load_processed(VARIANT)
    X, y, gid = reconstruct_groups()
    same = (X.astype(np.float32) == Xp) | (np.isnan(Xp) & np.isnan(X))
    if X.shape != Xp.shape or not np.array_equal(y, yp) or not same.all():
        raise AssertionError("whole-file reconstruction does not match the processed array")
    d = Path(cache)
    d.mkdir(parents=True, exist_ok=True)
    for name, a in (("X", X), ("y", y), ("gid", gid)):
        np.save(d / f"wf_{name}.npy", a)
    return {"shape": list(X.shape), "dtype": str(X.dtype), "peak_rss_gb": _peak_rss_gb()}


def _raw(protocol: str):
    """(X, y, {split: parts}) for 'row' (float32, shipped) or 'wf' (float64, five partitions).

    A worker serves one protocol only (pools are split by protocol), so one raw array is held."""
    if protocol in _CACHE:
        return _CACHE[protocol]
    if protocol == "row":
        from qgridbench.data.preprocess import load_processed

        Xp, yp, sp = load_processed(VARIANT)
        out = (Xp, yp, {"row": sp})
    else:
        from experiments.run_group_split_control import group_splits
        from experiments.run_whole_file_retune import EXPECTED_TEST_GROUPS, assert_file_disjoint

        d = Path(os.environ[WF_CACHE_ENV])
        X = np.load(d / "wf_X.npy", mmap_mode="r")
        y, gid = np.load(d / "wf_y.npy"), np.load(d / "wf_gid.npy")
        rng = np.random.default_rng(0)  # the archived control's stream -> Table I's partitions
        parts = {}
        for p in range(5):
            splits, held = group_splits(gid, y, rng)
            if held[2] != EXPECTED_TEST_GROUPS[p]:
                raise AssertionError(
                    f"wf{p}: test files {held[2]}, expected {EXPECTED_TEST_GROUPS[p]}"
                )
            assert_file_disjoint(splits, gid, p)
            parts[f"wf{p}"] = splits
        out = (X, y, parts)
    _CACHE[protocol] = out
    return out


def arm_matrix(arm: str, split: str):
    """The raw feature matrix of one arm on one split, its labels and split indices."""
    X, y, parts = _raw("row" if split == "row" else "wf")
    sp = parts[split]
    if arm == "absolute":
        return X, y, sp
    chans = angle_channels(_feature_names())
    if arm == "relative":
        return frame_relative(X, chans), y, sp
    if arm == "permuted":
        from qgridbench.utils.seeding import spawn_rng

        ref = derangement_within(sp, len(y), spawn_rng(PERM_SEED, f"permuted|{split}"))
        return frame_relative(X, chans, ref), y, sp
    raise ValueError(arm)


def regime(arm: str, split: str):
    """Leakage-safe PCA-8 view (prepare_regime's steps on the arm's matrix); cached per process."""
    key = ("regime", arm, split)
    if key in _CACHE:
        return _CACHE[key]
    from qgridbench.features.reduce import AngleScaler, build_preprocessor, fit_transform_splits
    from qgridbench.protocol import Regime, prepare_regime

    X, y, sp = arm_matrix(arm, split)
    Z = fit_transform_splits(build_preprocessor("pca", n_components=QUBITS, seed=0), X, sp)
    for s, z in Z.items():
        if not np.isfinite(z).all():
            raise AssertionError(f"{arm}/{split}: non-finite PCA features in {s}")
    sc = AngleScaler().fit(Z["train"])
    reg = Regime(
        name="pca",
        n_components=QUBITS,
        X=Z,
        y={s: y[sp[s]] for s in ("train", "val", "test")},
        angles={s: sc.transform(Z[s]) for s in ("train", "val", "test")},
        angle_scaler=sc,
    )
    if arm == "absolute" and split == "row":  # control: the shipped regime, exactly
        ref = prepare_regime(VARIANT, "pca", n_components=QUBITS, seed=0)
        for s in ("train", "val", "test"):
            if not (
                np.array_equal(ref.X[s], reg.X[s]) and np.array_equal(ref.angles[s], reg.angles[s])
            ):
                raise AssertionError(f"absolute/row features differ from prepare_regime on {s}")
    if len([k for k in _CACHE if k[0] == "regime"]) >= 6:
        _CACHE.pop(next(k for k in _CACHE if k[0] == "regime"))
    _CACHE[key] = reg
    return reg


def prepare_features(protocol: str, cache: str, pairs: list[tuple[str, str]]) -> dict:
    """Build every (arm, split) view of one protocol, save it, and run the protocol's diagnostics.

    Runs in a throwaway process: building a view holds the raw array, its float64 working copy
    and the pipeline's intermediate copies, a few hundred MB that a long-lived fitting worker
    would otherwise carry as its peak. The fitting workers load only the saved 8-dimensional
    views (`load_regime`)."""
    t0 = time.perf_counter()
    d = Path(cache)
    for arm, split in pairs:
        reg = regime(arm, split)
        arrays = {}
        for part in ("train", "val", "test"):
            arrays[f"X_{part}"] = reg.X[part]
            arrays[f"y_{part}"] = reg.y[part]
            arrays[f"A_{part}"] = reg.angles[part]
        np.savez(d / f"feat_{arm}_{split}.npz", **arrays)
        _CACHE.pop(("regime", arm, split))
    diag = diagnostics(protocol)
    return {"diagnostics": diag, "wall_s": time.perf_counter() - t0, "peak_rss_gb": _peak_rss_gb()}


def load_regime(arm: str, split: str):
    """A saved view (`prepare_features`) as a Regime; the two most recent are kept."""
    key = ("loaded", arm, split)
    if key in _CACHE:
        return _CACHE[key]
    from qgridbench.protocol import Regime

    with np.load(Path(os.environ[WF_CACHE_ENV]) / f"feat_{arm}_{split}.npz") as f:
        parts = ("train", "val", "test")
        reg = Regime(
            name="pca",
            n_components=QUBITS,
            X={p: f[f"X_{p}"] for p in parts},
            y={p: f[f"y_{p}"] for p in parts},
            angles={p: f[f"A_{p}"] for p in parts},
        )
    loaded = [k for k in _CACHE if k[0] == "loaded"]
    if len(loaded) >= 2:
        _CACHE.pop(loaded[0])
    _CACHE[key] = reg
    return reg


# ----------------------------------------------------------------------------- units


def table_i() -> dict:
    """Table I's frozen PCA-8 configurations and per-seed macro-F1, from its own artifacts."""
    import yaml

    cls = json.loads((TABLE_I_CLASSICAL / "results.json").read_text())
    cap = yaml.safe_load((TABLE_I_CLASSICAL / "params.yaml").read_text())["config"]["subset_cap"]
    qk = json.loads((TABLE_I_QSVM / "results.json").read_text())[ENCODING]
    vq = json.loads((TABLE_I_VQC / "results.json").read_text())["vqc"]
    if qk["subset_size"] != cap or vq["subset_size"] != cap:
        raise AssertionError("Table I quantum subsets differ from the classical cap")
    return {
        "cap": cap,
        "params": {
            **{f: cls[f]["best_params"] for f in CLASSICAL},
            "qsvm": qk["best_params"],
            "vqc": vq["best_params"],
        },
        "per_seed": {
            **{f: [s["macro_f1"] for s in cls[f]["per_seed"]] for f in CLASSICAL},
            # the archived QSVM run predates per_seed; it is deterministic (std 1e-16)
            "qsvm": [qk["aggregate"]["macro_f1"]["mean"]] * len(SEEDS),
            "vqc": [s["macro_f1"] for s in vq["per_seed"]],
        },
    }


def _peak_rss_gb() -> float:
    import psutil

    if sys.platform == "win32":
        return psutil.Process().memory_info().peak_wset / 2**30
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _guarded(name, y, proba) -> dict:
    from experiments.runner_guard import assert_clean_predictor
    from qgridbench.eval.metrics import compute_all

    assert_clean_predictor(name, y, proba, proba.argmax(1))
    m = compute_all(y, proba)
    m["positive_rate"] = float((proba.argmax(1) == 1).mean())
    return m


def fit_classical(name, params, reg, seed, cap, n_jobs=None) -> dict:
    """run_classical.eval_model's per-seed body, with the clean-prediction guard."""
    import experiments.run_classical as rc
    from qgridbench.models.classical.zoo import fit_model, predict_proba_timed, stratified_cap
    from qgridbench.utils.seeding import set_all_seeds

    Xtr, ytr = reg.X["train"], reg.y["train"]
    sub = stratified_cap(Xtr, ytr, cap, seed=0)
    set_all_seeds(seed)
    est = rc.build_pinned(name, params, seed)
    if n_jobs is not None:
        est.set_params(n_jobs=n_jobs)
    fit_model(name, est, Xtr[sub], ytr[sub])
    proba, _ = predict_proba_timed(est, reg.X["test"])
    return _guarded(name, reg.y["test"], proba)


def fit_qsvm(params, reg, cap) -> dict:
    """run_quantum.run_qkernel's final evaluation (deterministic: one fit)."""
    from experiments.run_quantum import _svc_on_kernel, _svc_proba
    from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel
    from qgridbench.protocol import kernel_subset

    sub = kernel_subset(reg, cap, seed=0)
    A_tr, y_tr, A_te = reg.angles["train"][sub], reg.y["train"][sub], reg.angles["test"]
    bw, C = params["bandwidth"], params["C"]
    s_tr = compute_states(A_tr, ENCODING, bandwidth=bw)
    K_tr = fidelity_kernel(s_tr, s_tr)
    clf = _svc_on_kernel(2, C).fit(K_tr, y_tr)
    del K_tr
    proba = np.empty((len(A_te), 2))
    for s in range(0, len(A_te), 2000):  # row-independent; bounds memory (no full test kernel)
        K_te = fidelity_kernel(compute_states(A_te[s : s + 2000], ENCODING, bandwidth=bw), s_tr)
        proba[s : s + 2000] = _svc_proba(clf, K_te, y_tr, 2)
    return _guarded("qsvm", reg.y["test"], proba)


def fit_vqc(params, reg, seed, cap) -> dict:
    """run_quantum.run_vqc's final per-seed loop body."""
    from qgridbench.models.classical.zoo import stratified_cap
    from qgridbench.models.quantum.vqc import VQClassifier
    from qgridbench.protocol import kernel_subset
    from qgridbench.utils.run_tracking import load_yaml
    from qgridbench.utils.seeding import set_all_seeds

    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    sub = kernel_subset(reg, cap, seed=0)
    A_va, y_va = reg.angles["val"], reg.y["val"]
    va = stratified_cap(A_va, y_va, cfg_q["qkernel"]["tuning_val_subset"], seed=0)
    set_all_seeds(seed)
    m = VQClassifier(
        QUBITS,
        depth=params["depth"],
        n_classes=2,
        lr=params["lr"],
        batch_size=params["batch_size"],
        max_epochs=cfg_q["vqc"]["max_epochs"],
        patience=cfg_q["vqc"]["patience"],
        seed=seed,
    )
    m.fit(reg.angles["train"][sub], reg.y["train"][sub], A_va[va], y_va[va])
    out = _guarded("vqc", reg.y["test"], m.predict_proba(reg.angles["test"]))
    out["epochs_run"] = len(m.history["epoch_loss"])
    return out


def diagnostics(protocol: str) -> dict:
    """Column map, deadband-zero shares (PM7 == 0) and the transform's own checks, on real data."""
    names = _feature_names()
    chans = angle_channels(names)
    out = {
        "reference": {f"R{r}": names[c[REF_CHANNEL]] for r, c in chans.items()},
        "transformed": {f"R{r}": [names[j] for j in c.values()] for r, c in chans.items()},
        "untouched_angle_like": [n for n in names if "PA" in n and not ANGLE_RE.match(n)],
        "deadband_zero": {},
        "frame_std_deg": {},
    }
    splits = ["row"] if protocol == "row" else [f"wf{p}" for p in range(5)]
    for proto in (protocol,):
        X, _, parts = _raw(proto)
        rel = frame_relative(X, chans)
        moved = sorted(j for c in chans.values() for j in c.values())
        keep = np.setdiff1d(np.arange(X.shape[1]), moved)
        if not np.array_equal(rel[:, keep], X[:, keep], equal_nan=True):
            raise AssertionError("the transform moved a non-angle column")
        lo, hi = np.nanmin(rel[:, moved]), np.nanmax(rel[:, moved])
        if not (lo > -180.0 and hi <= 180.0):
            raise AssertionError(f"{proto}: relative angles span [{lo}, {hi}]")
        if any(np.nanmax(np.abs(rel[:, c[REF_CHANNEL]])) != 0 for c in chans.values()):
            raise AssertionError("reference channel is not identically 0")
        pm7 = {r: X[:, names.index(f"R{r}-PM7:V")] == 0 for r in chans}
        anyz = np.logical_or.reduce(list(pm7.values()))
        for s in splits:
            out["deadband_zero"][s] = {
                part: {
                    **{f"R{r}": float(z[idx].mean()) for r, z in pm7.items()},
                    "any_relay": float(anyz[idx].mean()),
                    "n_rows": len(idx),
                }
                for part, idx in parts[s].items()
            }
        if proto == "row":
            tr = parts["row"]["train"]
            out["frame_std_deg"] = {
                f"R{r}": {
                    "absolute_median": float(
                        np.median([X[tr, j].std() for k, j in c.items() if k != REF_CHANNEL])
                    ),
                    "relative_median": float(
                        np.median([rel[tr, j].std() for k, j in c.items() if k != REF_CHANNEL])
                    ),
                }
                for r, c in chans.items()
            }
            out["deadband_zero_all_rows"] = {
                **{f"R{r}": float(z.mean()) for r, z in pm7.items()},
                "any_relay": float(anyz.mean()),
            }
    return out


def _protocol(sp: dict) -> str:
    return "row" if sp["split"] == "row" else "wf"


def unit_key(sp: dict) -> str:
    return "|".join(str(sp[k]) for k in ("kind", "arm", "split", "family", "seed"))


def run_unit(sp: dict) -> dict:
    t0 = time.perf_counter()
    ti = sp["table_i"]
    reg = load_regime(sp["arm"], sp["split"])
    fam = sp["family"]
    if fam == "qsvm":
        res = fit_qsvm(ti["params"]["qsvm"], reg, ti["cap"])
    elif fam == "vqc":
        res = fit_vqc(ti["params"]["vqc"], reg, sp["seed"], ti["cap"])
    else:
        res = fit_classical(fam, ti["params"][fam], reg, sp["seed"], ti["cap"], sp["n_jobs"])
    out = {k: v for k, v in sp.items() if k != "table_i"}
    out.update(key=unit_key(sp), result=res)
    out.update(wall_s=time.perf_counter() - t0, peak_rss_gb=_peak_rss_gb())
    out["pool_threads"] = _THREADS
    log.info("unit %s done in %.1fs", out["key"], out["wall_s"])
    return out


def build_specs(smoke: bool, ti: dict) -> list[dict]:
    if smoke:
        splits, seeds = ("row", "wf0"), (0, 1)
        families = ("logreg", "xgb", "lgbm", "qsvm", "vqc")
    else:
        splits, seeds, families = SPLITS, SEEDS, FAMILIES
    specs = []
    for arm in ARMS:
        for split in splits:
            for fam in families:
                if fam == "qsvm":
                    specs.append(
                        {"kind": "main", "arm": arm, "split": split, "family": fam, "seed": "det"}
                    )
                    continue
                for s in seeds:
                    if smoke and fam == "vqc" and (s != 0 or split != "row"):
                        continue  # smoke: one VQC fit per arm (about 200 s each)
                    specs.append(
                        {
                            "kind": "main",
                            "arm": arm,
                            "split": split,
                            "family": fam,
                            "seed": s,
                            "n_jobs": XGB_THREADS if fam == "xgb" else None,
                        }
                    )
    for split in splits:
        if split != "row":  # the whole-file XGBoost anchor at the archive's thread count
            specs.append(
                {
                    "kind": "wf_anchor",
                    "arm": "absolute",
                    "split": split,
                    "family": "xgb",
                    "seed": 0,
                    "n_jobs": WF_XGB_ANCHOR_THREADS,
                }
            )
    for sp in specs:
        sp.setdefault("n_jobs", None)
        sp["table_i"] = ti
    return sorted(specs, key=lambda sp: (COST_ORDER.get(sp["family"], 9), sp["arm"], sp["split"]))


# ----------------------------------------------------------------------------- summary


def _ms(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "n": len(v),
    }


def cell_scores(units: list[dict], seeds) -> dict:
    """{(arm, split, family): {seed: macro-F1}}; the deterministic QSVM fills every seed."""
    cells: dict = {}
    for u in units:
        if u["kind"] != "main":
            continue
        k = (u["arm"], u["split"], u["family"])
        f1 = u["result"]["macro_f1"]
        if u["family"] == "qsvm":
            cells[k] = dict.fromkeys(seeds, f1)
        else:
            cells.setdefault(k, {})[u["seed"]] = f1
    return cells


def gains(cells: dict, arm: str, base: str = "absolute") -> dict:
    """Per split protocol and family: paired (arm - base) gains over seeds (and partitions)."""
    out: dict = {"row": {}, "whole_file": {}}
    fams = sorted({k[2] for k in cells}, key=FAMILIES.index)
    for fam in fams:
        if (arm, "row", fam) in cells and (base, "row", fam) in cells:
            a, b = cells[(arm, "row", fam)], cells[(base, "row", fam)]
            d = [a[s] - b[s] for s in sorted(a) if s in b]
            out["row"][fam] = {"gain": _ms(d), "per_seed": d}
        parts = [s for s in SPLITS[1:] if (arm, s, fam) in cells and (base, s, fam) in cells]
        if parts:
            per_part, all_d = {}, []
            for p in parts:
                a, b = cells[(arm, p, fam)], cells[(base, p, fam)]
                d = [a[s] - b[s] for s in sorted(a) if s in b]
                per_part[p] = _ms(d)
                all_d += d
            out["whole_file"][fam] = {
                "gain": _ms(all_d),
                "per_partition": per_part,
                "partition_means_std": _ms([v["mean"] for v in per_part.values()])["std"],
            }
    for proto in ("row", "whole_file"):
        g = [v["gain"]["mean"] for v in out[proto].values()]
        out[proto + "_mean_over_families"] = float(np.mean(g)) if g else None
        out[proto + "_n_families"] = len(g)
    return out


def branch(rel: dict) -> str:
    wf, row = rel["whole_file_mean_over_families"], rel["row_mean_over_families"]
    if wf is not None and wf >= GAIN_LINE:
        return "HYPOTHESIS"
    if wf is not None and row is not None and wf < GAIN_LINE and row < GAIN_LINE:
        return "KILL"
    return "ROW_ONLY"


def anchors(units: list[dict], cells: dict, ti: dict) -> dict:
    rows = {}
    for fam in FAMILIES:
        got = cells.get(("absolute", "row", fam))
        if not got:
            continue
        dev = {s: abs(v - ti["per_seed"][fam][s]) for s, v in got.items()}
        rows[fam] = {"max_dev": max(dev.values()), "n_seeds": len(dev), "per_seed_dev": dev}
    worst = max((r["max_dev"] for r in rows.values()), default=None)
    table_i_anchor = {
        "tol": ANCHOR_TOL,
        "families": rows,
        "max_dev": worst,
        "complete": len(rows) == len(FAMILIES),
        "passed": worst is not None and worst <= ANCHOR_TOL,
        "xgb_threads": XGB_THREADS,
    }
    arch = {
        r["partition"]: r
        for r in json.loads((WHOLE_FILE_ARCHIVE / "results.json").read_text())["rows"]
        if r["split"] == "group"
    }
    wf = {}
    for u in units:
        if u["kind"] == "diagnostics" or u["split"] == "row" or u["arm"] != "absolute":
            continue
        p = int(u["split"][2:])
        if u["kind"] == "main" and u["family"] == "qsvm":
            wf[f"{u['split']}|qsvm"] = u["result"]["macro_f1"] - arch[p]["matched_quantum"]
        if u["kind"] == "wf_anchor":
            wf[f"{u['split']}|xgb_t{u['n_jobs']}"] = (
                u["result"]["macro_f1"] - arch[p]["matched_classical"]
            )
    wf_worst = max((abs(v) for v in wf.values()), default=None)
    return {
        "table_i_row": table_i_anchor,
        "whole_file": {
            "archive": WHOLE_FILE_ARCHIVE.name,
            "deviation": wf,
            "max_abs_dev": wf_worst,
            "passed": wf_worst is not None and wf_worst <= ANCHOR_TOL,
            "blocking": False,
        },
    }


def summarize(units: list[dict], ti: dict, smoke: bool) -> dict:
    cells = cell_scores(units, SEEDS)
    rel = gains(cells, "relative")
    perm = gains(cells, "permuted")
    rel_vs_perm = gains(cells, "relative", base="permuted")
    means = {f"{a}|{s}|{f}": _ms(list(v.values())) for (a, s, f), v in sorted(cells.items())}
    diag = {u["split"]: u["result"]["diagnostics"] for u in units if u["kind"] == "diagnostics"}
    perm_beats = {
        proto: [f for f, v in perm[proto].items() if v["gain"]["mean"] > 0]
        for proto in ("row", "whole_file")
    }
    main = [u for u in units if u["kind"] != "diagnostics"]
    by_family = {}
    for f in FAMILIES:
        us = [u for u in main if u["family"] == f]
        if us:
            by_family[f] = {
                "n_units": len(us),
                "wall_s": _ms([u["wall_s"] for u in us]),
                "max_peak_rss_gb": max(u["peak_rss_gb"] for u in us),
            }
    anc = anchors(units, cells, ti)
    return {
        "smoke": smoke,
        "anchors": anc,
        "branch": "ANCHOR_FAILED" if not anc["table_i_row"]["passed"] else branch(rel),
        "gain_relative_minus_absolute": rel,
        "control_permuted_minus_absolute": perm,
        "control_permuted_beats_absolute_families": perm_beats,
        "relative_minus_permuted": rel_vs_perm,
        "cell_means": means,
        "diagnostics": diag,
        "cost_per_family": by_family,
    }


# ----------------------------------------------------------------------------- orchestration


def memory_guard(workers: int, worker_rss_gb: float) -> dict:
    """W x worker peak + this process's resident set must fit 80% of free RAM."""
    import psutil

    parent = psutil.Process().memory_info().rss / 2**30
    avail = psutil.virtual_memory().available / 2**30
    need = workers * worker_rss_gb + parent
    out = {
        "workers": workers,
        "worker_rss_gb": worker_rss_gb,
        "parent_rss_gb": round(parent, 3),
        "need_gb": round(need, 3),
        "available_gb": round(avail, 3),
    }
    log.info("memory %s", out)
    if need > MEM_FRACTION * avail:
        raise MemoryError(f"need {need:.2f} GB > 80% of {avail:.2f} GB free")
    return out


def _init_worker(threads: int):
    import psutil
    import torch

    got = {v: os.environ.get(v) for v in THREAD_VARS}
    want_torch = min(threads, psutil.cpu_count(logical=False))  # torch caps at physical cores
    if any(x != str(threads) for x in got.values()) or torch.get_num_threads() != want_torch:
        raise AssertionError(
            f"worker {os.getpid()}: thread caps {got}, torch {torch.get_num_threads()}, "
            f"expected {threads}"
        )
    # load the OpenMP users now, under the pool's thread variables: run_classical, imported later,
    # re-pins the variables to --threads, which a runtime loaded after it would read
    import lightgbm  # noqa: F401
    import xgboost  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--worker-rss-gb", type=float, required=True)
    ap.add_argument("--smoke", action="store_true", help="row + wf0, seeds 0-1, 5 families")
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()

    ti = table_i()
    done = []
    if args.resume:
        done = json.loads((args.resume / "results_partial.json").read_text())["units"]
        log.info("resume from %s: %d finished units", args.resume, len(done))
    have = {u["key"] for u in done}
    specs = [sp for sp in build_specs(args.smoke, ti) if unit_key(sp) not in have]
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    config = {
        **{k: v for k, v in vars(args).items() if k != "resume"},
        "resume": str(args.resume) if args.resume else None,
        "table_i_runs": [TABLE_I_CLASSICAL.name, TABLE_I_QSVM.name, TABLE_I_VQC.name],
        "whole_file_archive": WHOLE_FILE_ARCHIVE.name,
        "table_i_params": ti["params"],
        "cap": ti["cap"],
        "xgb_threads": XGB_THREADS,
        "wf_xgb_anchor_threads": WF_XGB_ANCHOR_THREADS,
        "perm_seed": PERM_SEED,
        # the documented VQC chunk override (vqc.resolve_chunk); a chunk >= the batch size (128)
        # leaves training unchanged and prediction is row-independent, so it sets memory only
        "QGB_VQC_CHUNK": os.environ.get("QGB_VQC_CHUNK"),
        "preregistration": "decision log 2026-11-16T11:07Z, R4-09",
        "n_units_total": len(specs) + len(done),
    }
    name = f"frame_relative_angles_{VARIANT}_q{QUBITS}" + ("_smoke" if args.smoke else "")
    with RunDir(name, config=config, seeds=list(SEEDS)) as run:
        (run.path / "git_status_at_launch.txt").write_text(git_status)
        guard = memory_guard(args.workers, args.worker_rss_gb)
        units, failed = list(done), []
        cache = run.path / "cache"
        cache.mkdir()
        os.environ[WF_CACHE_ENV] = str(cache)
        os.environ[POOL_THREADS_ENV] = str(args.threads)
        units = [u for u in units if u["kind"] != "diagnostics"]  # recomputed below
        try:  # each step in its own throwaway process, one after another (memory budget)
            for proto in sorted({_protocol(sp) for sp in specs} | {"row", "wf"}):
                steps = [prepare_features]
                if proto == "wf":
                    steps.insert(0, prepare_wf_cache)
                pairs = sorted(
                    {(sp["arm"], sp["split"]) for sp in specs if _protocol(sp) == proto}
                    | ({("absolute", "row")} if proto == "row" else set())
                )
                for step in steps:
                    args_ = (
                        (str(cache),) if step is prepare_wf_cache else (proto, str(cache), pairs)
                    )
                    # no initializer: these steps fit nothing, so torch and the tree libraries
                    # stay unloaded; the thread variables are pinned at import (POOL_THREADS_ENV)
                    with ProcessPoolExecutor(
                        1, mp_context=multiprocessing.get_context("spawn")
                    ) as ex:
                        res = ex.submit(step, *args_).result()
                    if step is prepare_wf_cache:
                        log.info("whole-file cache %s", res)
                    else:
                        units.append(
                            {
                                "kind": "diagnostics",
                                "arm": "-",
                                "split": proto,
                                "family": "-",
                                "seed": "-",
                                "key": f"diagnostics|-|{proto}|-|-",
                                "result": {"diagnostics": res["diagnostics"]},
                                "wall_s": res["wall_s"],
                                "peak_rss_gb": res["peak_rss_gb"],
                                "n_views": len(pairs),
                            }
                        )
                        log.info(
                            "%s views and diagnostics: peak %.3f GB", proto, res["peak_rss_gb"]
                        )
        finally:
            del os.environ[POOL_THREADS_ENV]
        # one pool per (thread count, protocol), in turn: XGBoost's n_jobs is capped by the
        # process's OpenMP count, so each pool's workers get thread variables equal to the count
        # they fit at
        pools = sorted({(sp["n_jobs"] or args.threads, _protocol(sp)) for sp in specs})
        for threads, proto in sorted(pools, key=lambda tp: (-tp[0], tp[1])):
            group = [
                sp
                for sp in specs
                if (sp["n_jobs"] or args.threads, _protocol(sp)) == (threads, proto)
            ]
            log.info("pool at %d threads, %s: %d units", threads, proto, len(group))
            os.environ[POOL_THREADS_ENV] = str(threads)
            try:
                with ProcessPoolExecutor(
                    max_workers=args.workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_worker,
                    initargs=(threads,),
                ) as ex:
                    futs = {ex.submit(run_unit, sp): unit_key(sp) for sp in group}
                    for f in as_completed(futs):
                        exc = f.exception()
                        if exc is not None:
                            log.error("unit %s failed", futs[f], exc_info=exc)
                            failed.append(futs[f])
                            continue
                        units.append(f.result())
                        run.write_json("results_partial.json", {"units": units})
            finally:
                del os.environ[POOL_THREADS_ENV]
        for f in cache.iterdir():  # derived data, rebuilt from the raw data on every launch
            f.unlink()
        cache.rmdir()
        if failed:
            raise RuntimeError(
                f"{len(failed)} units failed (others in results_partial.json): {failed}"
            )
        summary = summarize(units, ti, args.smoke)
        summary["memory_guard"] = guard
        run.write_json("results.json", {**summary, "units": units})
        rel = summary["gain_relative_minus_absolute"]
        run.metrics.update(
            {
                "branch": summary["branch"],
                "anchor_table_i_passed": summary["anchors"]["table_i_row"]["passed"],
                "anchor_table_i_max_dev": summary["anchors"]["table_i_row"]["max_dev"],
                "anchor_whole_file_max_abs_dev": summary["anchors"]["whole_file"]["max_abs_dev"],
                "gain_row_mean_over_families": rel["row_mean_over_families"],
                "gain_whole_file_mean_over_families": rel["whole_file_mean_over_families"],
                "max_unit_peak_rss_gb": max(u["peak_rss_gb"] for u in units),
                "max_unit_wall_s": max(u["wall_s"] for u in units),
            }
        )
        log.info("ANCHORS %s", json.dumps(summary["anchors"]["table_i_row"]["max_dev"]))
        log.info("BRANCH %s", summary["branch"])
        print(f"-> {run.path}")
        if not summary["anchors"]["table_i_row"]["passed"]:
            raise AssertionError(
                f"Table I anchor failed: max dev {summary['anchors']['table_i_row']['max_dev']}"
            )


if __name__ == "__main__":
    main()
