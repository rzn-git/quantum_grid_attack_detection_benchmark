"""R4-20: the multiclass appendix -- 37 x 37 confusion counts per family, arXiv build only.

Ruling 2026-11-16T13:05Z item 9 (option b): the six classical families and the QSVM
(sklearn one-vs-rest on the deployed ZZ kernel, 8 qubits) on the 37-scenario release, with the
THREE-CLASS tuned hyperparameters transferred (no multiclass search exists; disclosed in every
artifact's `notes`), 10 seeds, counts summed over seeds. The VQC is excluded: its logged
multiclass strategy reads one qubit per class, and 37 classes exceed 8 qubits. Generation only,
no branch.

Classical cells reuse the shipped three-class path (`run_ablations.confusion_three_class`) one
family per unit; the QSVM is the run_quantum/FittedZoo path (stratified 2,000-row kernel subset,
fidelity Gram, precomputed-kernel SVC one-vs-rest) with the triple ZZ study's bandwidth and C.
The analytic kernel and the SVM solve are deterministic, so its seed sum is 10 copies of one
fit, as run_quantum reports it.

Stages, in order:
  probe   one fit per family and regime on the multiclass release (seed 0), wall time and peak
          memory per unit, the projected full-run cost written to probe.json.
  anchor  the same per-family path on the three-class release, merged and tabulated: the CSV
          must equal paper/tables/confusion_three_class.csv byte for byte (blocking); per-family
          count matches against the archived 20260815T052319Z_ablations_triple_q8 are reported.
  full    the multiclass run of record; make_report then writes figures/confusion_multiclass_
          {full,pca}.pdf and paper/tables/confusion_multiclass*.csv from it.
Each unit's result is cached under models/multiclass_confusion/ and read back on a rerun.

Run (needs data/processed/multiclass.npz once: uv run python -m qgridbench.data.preprocess
multiclass):
  uv run python -m experiments.run_multiclass_confusion --stage probe --threads 4 --workers 1
  uv run python -m experiments.run_multiclass_confusion --stage anchor --threads 4 --workers 1
  uv run python -m experiments.run_multiclass_confusion --stage full --threads 4 --workers 1
  [--family-threads xgb=N] fits that family at N threads in its own process.
"""

from __future__ import annotations

import argparse
import os
import sys

from experiments.family_threads import THREAD_VARS

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=4)
THREADS = _pre.parse_known_args(sys.argv[1:])[0].threads
if __name__ == "__main__":  # pin before numpy loads; spawned children inherit the environment
    for _var in THREAD_VARS:
        os.environ[_var] = str(THREADS)

import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from scipy.special import softmax  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from experiments import run_ablations  # noqa: E402
from experiments.family_threads import (  # noqa: E402
    check_thread_env,
    parse_family_threads,
    run_grouped,
)
from experiments.run_ablations import confusion_three_class  # noqa: E402
from experiments.run_adversarial import CLASSICAL, _svc_on_kernel  # noqa: E402
from experiments.runner_guard import assert_clean_predictor  # noqa: E402
from qgridbench.data.preprocess import load_processed  # noqa: E402
from qgridbench.eval.metrics import compute_all  # noqa: E402
from qgridbench.eval.tables import (  # noqa: E402
    MULTICLASS_CONFUSION_FILES,
    three_class_confusion_table,
)
from qgridbench.models.classical.zoo import predict_proba_timed  # noqa: E402
from qgridbench.models.quantum.qkernel import compute_states, fidelity_kernel  # noqa: E402
from qgridbench.protocol import kernel_subset, prepare_regime  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger, load_yaml  # noqa: E402

log = get_logger(__name__)

STAGES = ("probe", "anchor", "full")
QUBITS = 8  # the deployed QSVM and the PCA-8 regime
ENCODING = "zz"  # the deployed QSVM's feature map
TUNED_RUNS = {  # three-class winners, transferred to the multiclass release
    "full": "results/runs/merged_accuracy_triple_full",
    "pca": "results/runs/merged_accuracy_triple_pca8",
}
QSVM_TUNED_RUN = "results/runs/20260813T175045Z_quantum_qkernel_triple_q8"  # zz bandwidth, C
ANCHOR_RUN = "results/runs/20260815T052319Z_ablations_triple_q8"
ANCHOR_CSV = REPO_ROOT / "paper" / "tables" / "confusion_three_class.csv"
ANCHOR_FILES = {"full": "confusion_triple_full.json", "pca": "confusion_triple_pca8.json"}
CACHE_DIR = REPO_ROOT / "models" / "multiclass_confusion"
MEM_FRACTION = 0.8
UNIT_RSS_GB = 1.3  # the R4-20 budget for all its processes; the probe measures the real peak
NOTES = (
    "Hyperparameters are the three-class tuned winners transferred to the 37-scenario release "
    "(no multiclass search was run). The VQC is excluded: its multiclass strategy reads one "
    "qubit per class and 37 classes exceed 8 qubits. The QSVM is deterministic (fixed 2,000-row "
    "kernel subset, analytic kernel), so its seed sum is 10 copies of one fit."
)


def _peak_rss_gb() -> float:
    mi = psutil.Process().memory_info()
    if sys.platform == "win32":
        return mi.peak_wset / 2**30
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def qsvm_params() -> dict:
    zz = json.loads((REPO_ROOT / QSVM_TUNED_RUN / "results.json").read_text())[ENCODING]
    return {"bandwidth": zz["best_params"]["bandwidth"], "C": zz["best_params"]["C"]}


def prf_summary(prf: np.ndarray, labels: list[int]) -> dict:
    """Per-class precision, recall, F1 over seeds, in confusion_three_class's shape."""
    std = (lambda v: float(v.std(ddof=1))) if prf.shape[0] > 1 else (lambda v: 0.0)
    return {
        str(c): {
            k: {"mean": float(prf[:, j, i].mean()), "std": std(prf[:, j, i])}
            for j, k in enumerate(("precision", "recall", "f1"))
        }
        for i, c in enumerate(labels)
    }


def qsvm_confusion(
    variant: str, seeds: list[int], cap: int, params: dict, rec: SeedRecorder
) -> dict:
    """The deployed QSVM on one release: one deterministic fit, counted once per seed.
    Probabilities are the softmax of the one-vs-rest margins, as run_quantum scores it."""
    reg = prepare_regime(variant, "pca", n_components=QUBITS, seed=0)
    sub = kernel_subset(reg, cap, seed=0)
    a_tr, y_tr = reg.angles["train"][sub], reg.y["train"][sub]
    a_te, y_te = reg.angles["test"], reg.y["test"]
    labels = sorted(np.unique(y_te).tolist())
    s_tr = compute_states(a_tr, ENCODING, bandwidth=params["bandwidth"])
    clf = _svc_on_kernel(len(labels), params["C"]).fit(fidelity_kernel(s_tr, s_tr), y_tr)
    chunk = 2000
    dec = np.vstack(
        [
            clf.decision_function(
                fidelity_kernel(
                    compute_states(a_te[s : s + chunk], ENCODING, bandwidth=params["bandwidth"]),
                    s_tr,
                )
            )
            for s in range(0, len(a_te), chunk)
        ]
    )
    pred = clf.classes_[dec.argmax(1)]
    assert_clean_predictor("qsvm", y_te, dec, pred)
    codes = np.searchsorted(clf.classes_, y_te)
    for sd in seeds:
        rec.add("qsvm", sd, codes, dec.argmax(1), softmax(dec, axis=1))
    cm = confusion_matrix(y_te, pred, labels=labels)
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, labels=labels, zero_division=0)
    prf = np.stack([np.stack([p, r, f])] * len(seeds))
    total = cm * len(seeds)
    rows = total.sum(axis=1, keepdims=True)
    return {
        "labels": labels,
        "class_support": {str(c): int((y_te == c).sum()) for c in labels},
        "per_model": {
            "qsvm": {
                "confusion_counts": total.tolist(),
                "confusion_rates": (total / np.maximum(rows, 1)).tolist(),
                "per_class": prf_summary(prf, labels),
                "deterministic_seed_copies": len(seeds),
                "encoding": ENCODING,
                "params": params,
                "subset_size": int(len(sub)),
            }
        },
    }


class SeedRecorder:
    """Per-seed exports of one unit (operator rule 2026-11-17T02:18Z, "Exports"): every metric
    per seed, and per test row the true class, the predicted class and the class probabilities.

    Fed label CODES (0..K-1, the order of `markers`); writes scenario markers. `test_rows` maps
    each test position to its row in the processed release (the splits file's test indices).
    """

    def __init__(self, regime: str, markers: np.ndarray, test_rows: np.ndarray):
        self.regime, self.markers, self.test_rows = regime, np.asarray(markers), test_rows
        self.per_seed: list[dict] = []
        self.frames: list[pd.DataFrame] = []

    def __call__(self, name, seed, est, x_test, y_test, pred) -> None:
        """confusion_three_class's on_seed hook."""
        proba, _ = predict_proba_timed(est, x_test)
        self.add(name, seed, y_test, pred, proba)

    def add(self, name: str, seed: int, y_code, pred_code, proba) -> None:
        y_code, pred_code = np.asarray(y_code), np.asarray(pred_code)
        if proba.shape != (len(y_code), len(self.markers)) or len(y_code) != len(self.test_rows):
            raise SystemExit(f"{name} seed {seed}: proba {proba.shape} against {len(y_code)} rows")
        metrics = compute_all(y_code, proba)
        metrics["macro_f1_predicted"] = float(f1_score(y_code, pred_code, average="macro"))
        self.per_seed.append({"regime": self.regime, "model": name, "seed": seed, **metrics})
        frame = pd.DataFrame(
            {
                "test_pos": np.arange(len(y_code)),
                "row": self.test_rows,
                "regime": self.regime,
                "model": name,
                "seed": seed,
                "y_true": self.markers[y_code],
                "y_pred": self.markers[pred_code],
            }
        )
        probs = pd.DataFrame(
            proba.astype(np.float32), columns=[f"p_{m}" for m in self.markers.tolist()]
        )
        self.frames.append(pd.concat([frame, probs], axis=1))

    def predictions(self) -> pd.DataFrame:
        return pd.concat(self.frames, ignore_index=True)


_W: dict = {}


def _init_worker(threads: int, seeds: list[int], cap: int) -> None:
    check_thread_env(threads)
    _W.update(seeds=seeds, cap=cap)


def scenario_markers(variant: str) -> np.ndarray:
    """The release's label values, sorted; position i is the code XGBoost is given."""
    return np.unique(load_processed(variant)[1])


def encode_regime(reg, markers: np.ndarray):
    """Labels as 0..K-1 codes. XGBoost refuses the 37 scenario markers (1..41 with gaps); the
    codes keep the markers' order, so the sklearn and LightGBM fits are unchanged by it."""
    for s, v in reg.y.items():
        if not np.isin(v, markers).all():
            raise SystemExit(f"split {s} holds a label outside the release's markers")
    reg.y = {s: np.searchsorted(markers, v) for s, v in reg.y.items()}
    return reg


def decode_cell(out: dict, markers: np.ndarray) -> dict:
    """Codes back to scenario markers in a confusion block; the matrix order is unchanged.
    Drops the natural/attack cells, which index the three-class map's classes 1 and 2."""
    if out["labels"] != list(range(len(markers))):
        raise SystemExit(f"test labels {out['labels']} are not the codes 0..{len(markers) - 1}")
    m = [int(x) for x in markers]
    out["labels"] = m
    out["class_support"] = {str(m[int(k)]): v for k, v in out["class_support"].items()}
    for cell in out["per_model"].values():
        cell["per_class"] = {str(m[int(k)]): v for k, v in cell["per_class"].items()}
        cell.pop("natural_called_attack")
        cell.pop("attack_called_natural")
    return out


def _unit(unit: tuple) -> dict:
    """One (variant, regime, family) cell: fit over the seeds, return its confusion block."""
    variant, regime, family, tuned_dir = unit
    t = time.perf_counter()
    markers = scenario_markers(variant)
    rec = SeedRecorder(regime, markers, load_processed(variant)[2]["test"])
    if family == "qsvm":
        out = qsvm_confusion(variant, _W["seeds"], _W["cap"], qsvm_params(), rec)
    elif variant == "triple":
        n_comp = QUBITS if regime == "pca" else None
        cap = _W["cap"] if regime == "pca" else None
        out = confusion_three_class(variant, regime, n_comp, tuned_dir, _W["seeds"], cap, rec)
    else:
        n_comp = QUBITS if regime == "pca" else None
        cap = _W["cap"] if regime == "pca" else None
        shipped = run_ablations.prepare_regime
        run_ablations.prepare_regime = lambda *a, **k: encode_regime(shipped(*a, **k), markers)
        try:
            out = confusion_three_class(variant, regime, n_comp, tuned_dir, _W["seeds"], cap, rec)
        finally:
            run_ablations.prepare_regime = shipped
        out = decode_cell(out, markers)
    out["per_seed"] = rec.per_seed
    out["_predictions"] = rec.predictions()
    out["wall_s"] = time.perf_counter() - t
    out["peak_rss_gb"] = _peak_rss_gb()
    return out


def write_exports(run_path: Path, units: list[tuple], results: dict, parts: dict) -> dict:
    """per_seed.csv and predictions.parquet (a parquet dataset directory, one file per unit,
    copied from the unit caches) in the run directory."""
    rows = [r for u in units for r in results[u]["per_seed"]]
    pd.DataFrame(rows).to_csv(run_path / "per_seed.csv", index=False)
    pq = run_path / "predictions.parquet"
    pq.mkdir()
    for u in units:
        shutil.copyfile(parts[u], pq / f"{u[1]}_{u[2]}.parquet")
    return {"per_seed_rows": len(rows), "prediction_parts": len(units)}


def filtered_tuned(regime: str, family: str) -> Path:
    """A results.json holding one family's tuned block, so confusion_three_class fits only it."""
    tuned = json.loads((REPO_ROOT / TUNED_RUNS[regime] / "results.json").read_text())
    d = CACHE_DIR / "tuned" / f"{regime}_{family}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps({family: tuned[family]}), encoding="utf-8")
    return d


def plan_units(variant: str, with_qsvm: bool) -> list[tuple]:
    """One unit per (regime, family), families in each tuned run's own key order, which is
    the row order of the shipped three-class table."""
    units = []
    for regime, run in TUNED_RUNS.items():
        tuned = json.loads((REPO_ROOT / run / "results.json").read_text())
        fams = [f for f, blk in tuned.items() if "best_params" in blk]
        if sorted(fams) != sorted(CLASSICAL):
            raise SystemExit(f"{run}: tuned families {fams} are not the six classical ones")
        units += [(variant, regime, fam, filtered_tuned(regime, fam)) for fam in fams]
    if with_qsvm:
        units.append((variant, "pca", "qsvm", None))
    return units


def exclude_units(units: list[tuple], exclude: list[str]) -> list[tuple]:
    """Drop REGIME/FAMILY units; an entry that names no planned unit fails loud."""
    names = {f"{u[1]}/{u[2]}" for u in units}
    unknown = sorted(set(exclude) - names)
    if unknown:
        raise SystemExit(f"--exclude names no planned unit: {unknown} (planned: {sorted(names)})")
    return [u for u in units if f"{u[1]}/{u[2]}" not in exclude]


def unit_cache_path(stage: str, unit: tuple, seeds: list[int], cap: int, threads: int) -> Path:
    variant, regime, family, tuned_dir = unit
    params = (
        qsvm_params()
        if family == "qsvm"
        else json.loads((tuned_dir / "results.json").read_text())[family]["best_params"]
    )
    key = json.dumps([variant, regime, family, params, seeds, cap, threads], sort_keys=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return CACHE_DIR / stage / f"{variant}_{regime}_{family}_{h}.json"


def merge(
    results: dict,
    units: list[tuple],
    regime: str,
    variant: str,
    seeds: list[int],
    excluded: list[str] = (),
) -> dict:
    """The per-family cells of one regime in confusion_three_class's artifact shape, families in
    the tuned run's order (the order the shipped table rows follow). Units left out of the run
    are named in `excluded_units` and in the notes, so the artifact discloses them."""
    cells = [results[u] for u in units if u[1] == regime]
    labels, support = cells[0]["labels"], cells[0]["class_support"]
    for c in cells:
        if c["labels"] != labels or c["class_support"] != support:
            raise SystemExit(f"{variant}/{regime}: units disagree on the test labels")
    return {
        "regime": regime,
        "n_components": QUBITS if regime == "pca" else None,
        "tuned_run": TUNED_RUNS[regime],
        "seeds": list(seeds),
        "labels": labels,
        "class_support": support,
        "notes": NOTES
        + (
            f" Excluded from this run (over this machine's memory budget): {', '.join(excluded)}."
            if excluded
            else ""
        ),
        "excluded_units": sorted(excluded),
        "per_model": {k: v for c in cells for k, v in c["per_model"].items()},
    }


def anchor_compare(run_path: Path) -> dict:
    """Blocking: the regenerated three-class CSV equals the shipped one byte for byte.

    Per family, the summed counts are compared with the archive; a family the run lacks is
    reported as "missing", never raised as a KeyError, so a partial anchor still says which
    families reproduce."""
    three_class_confusion_table(run_path, "confusion_three_class", out_dir=run_path)
    got = (run_path / "confusion_three_class.csv").read_bytes()
    want = ANCHOR_CSV.read_bytes()
    per_family = {}
    for regime, fname in ANCHOR_FILES.items():
        arch = json.loads((REPO_ROOT / ANCHOR_RUN / fname).read_text())["per_model"]
        new = json.loads((run_path / fname).read_text())["per_model"]
        per_family[regime] = {
            k: (arch[k]["confusion_counts"] == new[k]["confusion_counts"])
            if k in new
            else "missing"
            for k in arch
        }
    out = {"csv_exact": got == want, "counts_exact": per_family}
    log.info("ANCHOR %s", out)
    return out


def anchor_passes(anchor: dict, excluded: list[str]) -> bool:
    """The registered anchor is the shipped CSV byte for byte. Amended 2026-11-17 ("R4-20, the
    multiclass appendix anchor", option B): with units excluded for memory, every family that
    ran must equal the archived summed counts exactly, an excluded one must be the only kind
    reported missing, and the CSV comparison is reported only."""
    if not excluded:
        return bool(anchor["csv_exact"])
    for regime, fams in anchor["counts_exact"].items():
        for fam, ok in fams.items():
            if ok == "missing":
                if f"{regime}/{fam}" not in excluded:
                    return False
            elif ok is not True:
                return False
    return True


def memory_guard(workers: int, rss_gb: float) -> dict:
    avail = psutil.virtual_memory().available / 2**30
    out = {"workers": workers, "unit_rss_gb": rss_gb, "available_gb": avail}
    log.info("memory: %d x %.2f GB vs %.2f GB available", workers, rss_gb, avail)
    if workers * rss_gb > MEM_FRACTION * avail:
        raise MemoryError(f"{workers} x {rss_gb} GB > {MEM_FRACTION:.0%} of {avail:.1f} GB free")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--threads", type=int, default=4, help="BLAS/OpenMP threads per process")
    ap.add_argument("--workers", type=int, default=1, help="processes fitting units in parallel")
    ap.add_argument("--family-threads", nargs="*", default=[], help="FAMILY=N")
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="REGIME/FAMILY units left out of this launch (recorded in params.yaml), e.g. a unit "
        "whose fitted model cannot fit the memory budget",
    )
    args = ap.parse_args()
    family_threads = parse_family_threads(args.family_threads)
    t_start = time.perf_counter()

    cfg_q = load_yaml(REPO_ROOT / "configs" / "quantum.yaml")
    cfg_c = load_yaml(REPO_ROOT / "configs" / "classical.yaml")
    cap = cfg_q["qkernel"]["subset_cap"]
    all_seeds = cfg_c["evaluation"]["seeds"]
    variant = "triple" if args.stage == "anchor" else "multiclass"
    seeds = all_seeds[:1] if args.stage == "probe" else all_seeds
    units = plan_units(variant, with_qsvm=args.stage != "anchor")
    units = exclude_units(units, args.exclude)
    mem = memory_guard(args.workers, UNIT_RSS_GB)

    def threads_of(u: tuple) -> int:
        return family_threads.get(u[2], THREADS)

    cfg = {
        "stage": args.stage,
        "variant": variant,
        "seeds": seeds,
        "subset_cap": cap,
        "qubits": QUBITS,
        "encoding": ENCODING,
        "tuned_runs": TUNED_RUNS,
        "qsvm_tuned_run": QSVM_TUNED_RUN,
        "qsvm_params": qsvm_params(),
        "threads": THREADS,
        "family_threads": family_threads,
        "excluded_units": args.exclude,
        "workers": args.workers,
        "memory_guard": mem,
        "notes": NOTES,
        "ruling": "decision log 2026-11-16T13:05Z item 9 (R4-20, option b); exports per "
        "the 2026-11-17T02:18Z operator rule",
    }
    name = {"probe": "probe_multiclass", "anchor": "anchor_triple", "full": "multiclass"}
    with RunDir(f"multiclass_confusion_{name[args.stage]}", config=cfg, seeds=seeds) as run:
        results, todo, parts = {}, [], {}
        for u in units:
            f = unit_cache_path(args.stage, u, seeds, cap, threads_of(u))
            parts[u] = f.with_suffix(".parquet")
            if f.exists() and parts[u].exists():
                results[u] = json.loads(f.read_text())
                log.info("unit %s/%s/%s resumed from %s", *u[:3], f.name)
            else:
                todo.append(u)

        def keep(u: tuple, res: dict) -> None:
            f = unit_cache_path(args.stage, u, seeds, cap, threads_of(u))
            f.parent.mkdir(parents=True, exist_ok=True)
            res.pop("_predictions").to_parquet(parts[u], index=False)
            f.write_text(json.dumps(res), encoding="utf-8")
            results[u] = res
            log.info(
                "unit %s/%s/%s %.1f s | %d threads | peak RSS %.2f GB",
                *u[:3],
                res["wall_s"],
                threads_of(u),
                res["peak_rss_gb"],
            )

        run_grouped(
            todo, threads_of, _unit, _init_worker, (seeds, cap), args.workers, THREADS, keep
        )
        timing = {
            f"{u[1]}/{u[2]}": {
                "wall_s": results[u]["wall_s"],
                "peak_rss_gb": results[u]["peak_rss_gb"],
            }
            for u in units
        }
        run.metrics.update({"unit_timing": timing})
        run.metrics.update({"exports": write_exports(run.path, units, results, parts)})

        if args.stage == "probe":
            n = len(all_seeds)
            projected = {
                k: v["wall_s"] * (1 if k.endswith("/qsvm") else n) for k, v in timing.items()
            }
            probe = {
                "timing_one_seed": timing,
                "projected_full_s": projected,
                "projected_full_total_s": sum(projected.values()),
                "max_peak_rss_gb": max(v["peak_rss_gb"] for v in timing.values()),
            }
            run.write_json("probe.json", probe)
            run.metrics.update(probe)
            log.info("PROBE projected full run %.0f s", probe["projected_full_total_s"])
        else:
            files = ANCHOR_FILES if args.stage == "anchor" else MULTICLASS_CONFUSION_FILES
            for regime, fname in files.items():
                run.write_json(fname, merge(results, units, regime, variant, seeds, args.exclude))
            if args.stage == "anchor":
                anchor = anchor_compare(run.path)
                anchor["pass"] = anchor_passes(anchor, args.exclude)
                run.metrics.update({"anchor": anchor})
                if not anchor["pass"]:
                    raise SystemExit(f"ANCHOR FAILED: {anchor}")
        run.metrics["run_wall_s"] = time.perf_counter() - t_start
        run.metrics["run_peak_rss_gb"] = _peak_rss_gb()
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
