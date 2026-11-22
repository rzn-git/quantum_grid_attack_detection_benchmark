"""R4-19 (P12a): is a released source file one event, a distinct recording session, or a thinning
of one shared recording? A read of the 15 multiclass files; no model, no split.

Run (one process, about 1 minute, peak about 0.7 GB):
  uv run python -m experiments.run_file_event_identity --threads 2

Pre-registration: decision log "2026-11-16T11:37Z — PRE-RUN AMENDMENT (round 4 part B: R4-13",
item R4-19. The lines were fixed in the read-only probe (`tmp/r4_misc/probe_r4_19_file_event.py`)
before that probe ran, and its output was seen before this runner was written; this runner
reproduces it under the logged lines and writes it to `results/runs/` so a finding can cite it.

Measured on the 15 multiclass ARFFs (row-aligned with the binary CSVs, N-TEMP 2026-09-02):
  (1) coverage: distinct scenario ids per file, contiguous runs, scenarios split over > 1 run;
  (2) order: files whose run sequence is the modal one;
  (3) exact duplicate feature rows within a file, and distinct vectors present in > 1 file;
  (4) for every row, in pooled-standardized 128-feature space (inf -> NaN -> column median):
        d_within = distance to its nearer released-order neighbour in its own (file, scenario) run,
        d_cross  = distance to the nearest row of the same scenario in any OTHER file,
        same     = whether its nearest same-scenario row (all files, itself excluded) is its file's;
      R = median(d_cross) / median(d_within).

Verdicts (lines unchanged from the probe; the amendment's "no scenario is split" is added to
DISTINCT, so the verdict is the stricter of the two statements of the line set):
  file = one event        no: when every file carries more than one scenario
  DISTINCT SESSIONS       no scenario split AND cross-file duplicate share <= 0.001 AND R >= 2
                          AND same-file fraction >= 0.9
  SHARED RECORDING        cross-file duplicate share >= 0.005 OR R <= 1.2 OR same-file <= 0.2
  INCONCLUSIVE            otherwise
Positive control (blocking): each file's even and odd released positions become two groups (a
shared recording by construction); its R must be <= 1.2.
Anchors (blocking): 13 files carry all 37 scenarios (D2e, 2026-09-01T23:03Z) and 13 files share
the modal run sequence (N-TEMP: 78 = C(13, 2) of 105 pairs).
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

import subprocess  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from qgridbench.data.validate import load_variant_frames  # noqa: E402
from qgridbench.utils.run_tracking import REPO_ROOT, RunDir, get_logger  # noqa: E402

log = get_logger(__name__)

VARIANT = "multiclass"
LABEL = "marker"
DISTINCT_DUP_MAX = 0.001
DISTINCT_R_MIN = 2.0
DISTINCT_SAME_MIN = 0.9
SHARED_DUP_MIN = 0.005
SHARED_R_MAX = 1.2
SHARED_SAME_MAX = 0.2
CONTROL_R_MAX = 1.2
ANCHOR_FILES_ALL_37 = 13
ANCHOR_MODAL_SEQUENCE_FILES = 13
N_RANDOM_PAIRS = 500
NAN_TOKEN = -9.87654321e300  # NaN rows compare equal to each other in the duplicate count


def runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous (label, start, stop) runs in released order."""
    labels = np.asarray(labels)
    cut = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts, stops = np.r_[0, cut], np.r_[cut, len(labels)]
    return [(int(labels[a]), int(a), int(b)) for a, b in zip(starts, stops)]


def coverage(names: list[str], labels: list[np.ndarray]) -> tuple[dict, dict]:
    all_scen = sorted(set(np.concatenate(labels).tolist()))
    per_file, seqs = {}, {}
    for n, lab in zip(names, labels):
        rr = runs(lab)
        cnt = Counter(s for s, _, _ in rr)
        seqs[n] = tuple(s for s, _, _ in rr)
        lens = np.array([b - a for _, a, b in rr])
        present = set(lab.tolist())
        per_file[n] = {
            "n_rows": int(len(lab)),
            "n_scenarios": len(present),
            "missing_scenarios": [int(s) for s in all_scen if s not in present],
            "n_runs": len(rr),
            "scenarios_in_more_than_one_run": sorted(int(s) for s, c in cnt.items() if c > 1),
            "run_length_min_median_max": [int(lens.min()), float(np.median(lens)), int(lens.max())],
        }
    modal = Counter(seqs.values()).most_common(1)[0][0]
    order = {
        "n_files_with_modal_sequence": int(sum(s == modal for s in seqs.values())),
        "modal_sequence": list(modal),
    }
    return per_file, order


def duplicates(feats: list[np.ndarray]) -> dict:
    """Exact duplicate feature rows within a file, and distinct vectors found in > 1 file."""
    owner: dict[bytes, set[int]] = {}
    within = 0
    for fi, F in enumerate(feats):
        G = np.where(np.isnan(F), NAN_TOKEN, F)
        seen: set[bytes] = set()
        for row in G:
            k = row.tobytes()
            if k in seen:
                within += 1
            seen.add(k)
            owner.setdefault(k, set()).add(fi)
    n_total = int(sum(len(F) for F in feats))
    n_cross = sum(1 for s in owner.values() if len(s) > 1)
    return {
        "n_rows_total": n_total,
        "within_file_duplicate_rows": int(within),
        "distinct_vectors_in_more_than_one_file": int(n_cross),
        "cross_file_share_of_rows": n_cross / n_total,
    }


def standardize_pooled(feats: list[np.ndarray]) -> np.ndarray:
    P = np.vstack(feats).astype(np.float64)
    P[~np.isfinite(P)] = np.nan
    med = np.nanmedian(P, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    r, c = np.where(np.isnan(P))
    P[r, c] = med[c]
    sd = P.std(axis=0)
    sd[sd == 0] = 1.0
    return (P - P.mean(axis=0)) / sd


def block_stats(Z: np.ndarray, scen: np.ndarray, groups: np.ndarray, seed: int = 0) -> dict:
    """d_within, d_cross and the same-group nearest-neighbour fraction, `groups` as the file id.

    Rows of one group must be in released order inside Z (they are: files are stacked in order).
    """
    d_w, d_c, same, rand, per_scen = [], [], [], [], []
    rng = np.random.default_rng(seed)
    for s in sorted(set(scen.tolist())):
        idx = np.flatnonzero(scen == s)
        Zs, gs = Z[idx], groups[idx]
        _, j2 = NearestNeighbors(n_neighbors=2).fit(Zs).kneighbors(Zs)
        same_s = gs[j2[:, 1]] == gs
        same.append(same_s)
        dw_s, dc_s = [], []
        for g in np.unique(gs):
            m = gs == g
            Zb = Zs[m]
            if len(Zb) < 3 or m.all():
                continue
            step = np.linalg.norm(Zb[1:] - Zb[:-1], axis=1)
            dw_s.append(np.minimum(np.r_[np.inf, step], np.r_[step, np.inf]))
            dc_s.append(NearestNeighbors(n_neighbors=1).fit(Zs[~m]).kneighbors(Zb)[0][:, 0])
            a, b = rng.integers(0, len(Zb), (2, min(N_RANDOM_PAIRS, len(Zb))))
            k = a != b
            rand.append(np.linalg.norm(Zb[a[k]] - Zb[b[k]], axis=1))
        dw_s, dc_s = np.concatenate(dw_s), np.concatenate(dc_s)
        d_w.append(dw_s)
        d_c.append(dc_s)
        per_scen.append(
            {
                "scenario": int(s),
                "n_rows": int(len(idx)),
                "same_group_nn_fraction": float(same_s.mean()),
                "R_cross_over_within": float(np.median(dc_s) / np.median(dw_s)),
            }
        )
    d_w, d_c = np.concatenate(d_w), np.concatenate(d_c)
    rand, same = np.concatenate(rand), np.concatenate(same)
    return {
        "median_d_within_adjacent": float(np.median(d_w)),
        "median_d_cross_nearest": float(np.median(d_c)),
        "median_d_within_random_pair": float(np.median(rand)),
        "R_cross_over_within": float(np.median(d_c) / np.median(d_w)),
        "same_group_nn_fraction": float(same.mean()),
        "share_rows_d_cross_below_d_within": float(np.mean(d_c < d_w)),
        "per_scenario": per_scen,
    }


def verdict(per_file: dict, dup_share: float, R: float, same: float) -> dict:
    n_scen = [v["n_scenarios"] for v in per_file.values()]
    any_split = any(v["scenarios_in_more_than_one_run"] for v in per_file.values())
    distinct_lines = {
        "no_scenario_split": not any_split,
        f"cross_file_dup_share<={DISTINCT_DUP_MAX}": dup_share <= DISTINCT_DUP_MAX,
        f"R>={DISTINCT_R_MIN}": R >= DISTINCT_R_MIN,
        f"same_file_nn>={DISTINCT_SAME_MIN}": same >= DISTINCT_SAME_MIN,
    }
    shared_lines = {
        f"cross_file_dup_share>={SHARED_DUP_MIN}": dup_share >= SHARED_DUP_MIN,
        f"R<={SHARED_R_MAX}": R <= SHARED_R_MAX,
        f"same_file_nn<={SHARED_SAME_MAX}": same <= SHARED_SAME_MAX,
    }
    if all(distinct_lines.values()):
        session = "DISTINCT SESSIONS"
    elif any(shared_lines.values()):
        session = "SHARED RECORDING"
    else:
        session = "INCONCLUSIVE"
    return {
        "file_is_one_event": bool(max(n_scen) <= 1),
        "min_scenarios_per_file": int(min(n_scen)),
        "distinct_session_lines": distinct_lines,
        "shared_recording_lines": shared_lines,
        "session_verdict": session,
    }


def _peak_rss_gb() -> float:
    mi = psutil.Process().memory_info()
    return (mi.peak_wset if sys.platform == "win32" else mi.rss) / 2**30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    args = ap.parse_args()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    cfg = {
        "variant": VARIANT,
        "threads": args.threads,
        "lines": {
            "distinct": [DISTINCT_DUP_MAX, DISTINCT_R_MIN, DISTINCT_SAME_MIN, "no split"],
            "shared": [SHARED_DUP_MIN, SHARED_R_MAX, SHARED_SAME_MAX],
            "control_R_max": CONTROL_R_MAX,
        },
        "prereg": "decision log 2026-11-16T11:37Z, R4-19",
    }
    with RunDir(f"file_event_identity_{VARIANT}", config=cfg) as run:
        (run.path / "git_status_at_launch.txt").write_text(porcelain)
        frames = load_variant_frames(VARIANT)
        names = [f.attrs["source_file"].split(" ")[0] for f in frames]
        feats = [f.drop(columns=[LABEL]).to_numpy(dtype=np.float64) for f in frames]
        labels = [f[LABEL].astype(int).to_numpy() for f in frames]

        per_file, order = coverage(names, labels)
        n37 = sum(v["n_scenarios"] == 37 for v in per_file.values())
        anchors = {
            "files_with_all_37": n37,
            "files_with_all_37_expected": ANCHOR_FILES_ALL_37,
            "modal_sequence_files": order["n_files_with_modal_sequence"],
            "modal_sequence_files_expected": ANCHOR_MODAL_SEQUENCE_FILES,
        }
        anchors["pass"] = bool(
            n37 == ANCHOR_FILES_ALL_37
            and order["n_files_with_modal_sequence"] == ANCHOR_MODAL_SEQUENCE_FILES
        )
        if not anchors["pass"]:
            run.write_json("results_partial.json", {"anchors": anchors, "files": per_file})
            raise AssertionError(f"anchor failed: {anchors}")

        dup = duplicates(feats)
        Z = standardize_pooled(feats)
        file_id = np.concatenate([np.full(len(F), i) for i, F in enumerate(feats)])
        pos = np.concatenate([np.arange(len(F)) for F in feats])
        scen = np.concatenate(labels)
        files_view = block_stats(Z, scen, file_id)
        ctrl_view = block_stats(Z, scen, file_id * 2 + (pos % 2))
        control = {
            "R_cross_over_within": ctrl_view["R_cross_over_within"],
            "same_group_nn_fraction": ctrl_view["same_group_nn_fraction"],
            "pass": bool(ctrl_view["R_cross_over_within"] <= CONTROL_R_MAX),
        }
        if not control["pass"]:
            run.write_json("results_partial.json", {"positive_control": ctrl_view})
            raise AssertionError(f"positive control failed: {control}")

        v = verdict(
            per_file,
            dup["cross_file_share_of_rows"],
            files_view["R_cross_over_within"],
            files_view["same_group_nn_fraction"],
        )
        n_runs = sum(x["n_runs"] for x in per_file.values())
        out = {
            "verdict": v,
            "n_file_scenario_runs": int(n_runs),
            "anchors": anchors,
            "positive_control_even_odd": {**control, "detail": ctrl_view},
            "exact_duplicates": dup,
            "files_as_groups": files_view,
            "scenario_order": order,
            "files": per_file,
            "peak_rss_gb": _peak_rss_gb(),
        }
        run.write_json("results.json", out)
        run.metrics.update(
            {
                "session_verdict": v["session_verdict"],
                "file_is_one_event": v["file_is_one_event"],
                "n_file_scenario_runs": int(n_runs),
                "cross_file_dup_share": dup["cross_file_share_of_rows"],
                "R_cross_over_within": files_view["R_cross_over_within"],
                "same_file_nn_fraction": files_view["same_group_nn_fraction"],
                "control_R": control["R_cross_over_within"],
                "peak_rss_gb": out["peak_rss_gb"],
            }
        )
        log.info(
            "file=event %s | sessions %s | dup %.2e R %.2f same %.3f | control R %.3f | %d runs",
            v["file_is_one_event"],
            v["session_verdict"],
            dup["cross_file_share_of_rows"],
            files_view["R_cross_over_within"],
            files_view["same_group_nn_fraction"],
            control["R_cross_over_within"],
            n_runs,
        )
        print(f"-> {run.path}")


if __name__ == "__main__":
    main()
