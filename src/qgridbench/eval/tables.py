"""Generate LaTeX + CSV tables from results/runs JSON. No hand-typed numbers.

Every table function reads a run's results.json and emits a .tex (\\input-able by
paper/main.tex) plus a .csv sidecar. Significance marks come from paired Wilcoxon
+ Holm-Bonferroni across the headline family.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from qgridbench.eval.design_tokens import LATEX_COLOR_NAMES, QUANTUM_PREFIXES, family
from qgridbench.eval.stats import holm_bonferroni, paired_wilcoxon, robustness_stats
from qgridbench.utils.run_tracking import REPO_ROOT

TABLE_DIR = REPO_ROOT / "paper" / "tables"
# R4-20's two artifacts, one per feature regime (experiments/run_multiclass_confusion.py)
MULTICLASS_CONFUSION_FILES = {
    "full": "confusion_multiclass_full.json",
    "pca": "confusion_multiclass_pca8.json",
}
HEADLINE_METRICS = ["macro_f1", "balanced_accuracy", "auprc", "roc_auc", "brier", "ece"]
# exhaustion sweep column order: the two quantum families adjacent, classical last, so the
# exhibit reads left-to-right as "both quantum models invert, the classical kernel does not"
MODELS_EXHAUSTION = ["qsvm", "vqc", "rbf_svm"]


def _neg(text: str) -> str:
    """Typeset a leading hyphen-minus as a true minus sign (IEEE: a minus is not a hyphen)."""
    return "$-$" + text[1:] if text.startswith("-") else text


def _num(value: float, spec: str = ".3f") -> str:
    """A table number with a true minus; the spec may ask for a shown sign (`+.3f`)."""
    return _neg(format(value, spec))


def _count(value: float) -> str:
    """An integer count, IEEE style: four digits take no separator, five or more a thin space."""
    digits = str(abs(int(round(value))))
    if len(digits) >= 5:
        head = len(digits) % 3 or 3
        digits = "\\,".join(
            [digits[:head]] + [digits[i : i + 3] for i in range(head, len(digits), 3)]
        )
    return _neg(("-" if value < 0 else "") + digits)


def _fmt(mean: float, std: float) -> str:
    return f"{_num(mean)}\\,$\\pm$\\,{std:.3f}"


# Display names: the ONE mapping from code identifiers to the words a reader sees. The CSV
# sidecars and the results JSON keep the identifiers (so a name can be grepped across them);
# every typeset table and every figure label goes through `display_*`. An identifier missing
# from a mapping raises KeyError rather than printing a code name.
METRIC_DISPLAY = {
    "macro_f1": "Macro-F1",
    "balanced_accuracy": "Balanced accuracy",
    "auprc": "AUPRC",
    "roc_auc": "ROC-AUC",
    "brier": "Brier",
    "ece": "ECE",
}
MODEL_DISPLAY = {
    "logreg": "Logistic regression",
    "rbf_svm": "RBF-SVM",
    "rf": "Random forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "mlp": "MLP",
    "qsvm_angle_ry": "QSVM (RY-angle)",
    "qsvm_zz": "QSVM (ZZ)",
    "vqc": "VQC",
    "baseline_stratified_random": "Stratified random",
    "baseline_majority": "Majority class",
    # family named without an encoding (adversarial tables and figures)
    "qsvm": "QSVM",
    # data-efficiency series and transfer-matrix sources (figures)
    "xgb_full": "XGBoost (all features)",
    "qkernel_angle_ry": "QSVM (RY-angle)",
    "mlp_surrogate": "MLP surrogate",
}
ENCODING_DISPLAY = {"angle_ry": "RY-angle", "zz": "ZZ"}
TASK_DISPLAY = {"binary": "binary", "triple": "three-class"}

# Table footnote marks are letters (IEEE PES author kit, Part 4a). Within a table the first
# mark type is `a`, the second `b`.
MARK_A = "$^{a}$"
MARK_B = "$^{b}$"


def display_metric(key: str) -> str:
    return METRIC_DISPLAY[key]


def display_model(key: str) -> str:
    return MODEL_DISPLAY[key]


def display_encoding(key: str) -> str:
    return ENCODING_DISPLAY[key]


def display_cell(cell: str) -> str:
    """`zz_q8_binary` -> `ZZ, 8 qubits, binary` (the encoding x qubits x task cells)."""
    enc, rest = cell.split("_q", 1)
    qubits, task = rest.split("_", 1)
    return f"{display_encoding(enc)}, {qubits} qubits, {TASK_DISPLAY[task]}"


def _row_tint(model_key: str) -> str:
    """A `\\rowcolor` prefix that shades a quantum model's row (and a chance baseline's in grey)
    in its meaning color (design_tokens.MEANING, ruling 2026-11-19T16:18Z); classical rows stay
    white.
    """
    fam = family(model_key)
    return "" if fam == "classical" else f"\\rowcolor{{{LATEX_COLOR_NAMES[fam]}Tint}}"


def _holm_vs_best(results: dict, best_family: str | None = None) -> tuple[str | None, dict]:
    """The best model on mean macro-F1 over seeds, and each other model's Holm-corrected paired
    Wilcoxon p-value against it (models with the same number of seeds only)."""
    per_seed_f1 = {
        name: np.array([s["macro_f1"] for s in res["per_seed"]])
        for name, res in results.items()
        if "per_seed" in res
    }
    if best_family is None and per_seed_f1:
        best_family = max(per_seed_f1, key=lambda k: per_seed_f1[k].mean())
    pvals = {}
    for name, arr in per_seed_f1.items():
        if name != best_family and len(arr) == len(per_seed_f1[best_family]):
            pvals[name] = paired_wilcoxon(per_seed_f1[best_family], arr)["p_value"]
    return best_family, (holm_bonferroni(pvals) if pvals else {})


def accuracy_table(run_dir: Path, out_name: str, best_family: str | None = None) -> Path:
    """Main accuracy table: models x metrics (mean±std over seeds), significance vs best."""
    results = json.loads((run_dir / "results.json").read_text())
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    best_family, corrected = _holm_vs_best(results, best_family)

    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{l" + "r" * len(HEADLINE_METRICS) + "}",
        "\\toprule",
        "Model & " + " & ".join(display_metric(m) for m in HEADLINE_METRICS) + " \\\\",
        "\\midrule",
    ]
    rows_csv = [["model", *HEADLINE_METRICS, "holm_p_vs_best"]]
    for name, res in results.items():
        agg = res["aggregate"]
        cells = [_fmt(agg[m]["mean"], agg[m]["std"]) for m in HEADLINE_METRICS]
        mark = ""
        if name == best_family:
            mark = MARK_A  # a = best model on mean macro-F1
        elif name in corrected and corrected[name] < 0.05:
            mark = MARK_B  # b = significantly below the best (Holm-corrected p < 0.05)
        lines.append(
            f"{_row_tint(name)}{display_model(name)}{mark} & " + " & ".join(cells) + " \\\\"
        )
        rows_csv.append(
            [
                name,
                *[f"{agg[m]['mean']:.4f}" for m in HEADLINE_METRICS],
                f"{corrected.get(name, float('nan')):.4g}",
            ]
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "% a = best model on mean macro-F1;",
        "% b = significantly below the best (Holm-corrected p < 0.05).",
    ]

    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", rows_csv)
    return tex


def cost_table(run_dir: Path, out_name: str) -> Path:
    """Training wall-clock and per-sample inference latency for EVERY model.

    Required by study protocol §5: grid detection has real-time constraints, so a
    reviewer will ask what the quantum models cost to run. Quantum latencies are
    SIMULATOR latencies and are marked as such — they are a lower bound on a real
    device only in the sense that a device would add queue and shot overhead, and
    they are not comparable to classical latencies as a deployment claim.
    """
    results = json.loads((run_dir / "results.json").read_text())
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, res in results.items():
        agg = res.get("aggregate", {})
        train = agg.get("train_time_s", {}).get("mean")
        lat = agg.get("infer_latency_s", {}).get("mean")
        if train is None or lat is None:
            continue
        rows.append((name, float(train), float(lat) * 1e6))  # latency in microseconds
    if not rows:
        raise ValueError(f"no timing data in {run_dir}")
    rows.sort(key=lambda r: r[2])
    fastest = rows[0][2]

    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Model & Train (s) & Inference ($\\mu$s/sample) & Relative to fastest \\\\",
        "\\midrule",
    ]
    rows_csv = [
        ["model", "train_seconds", "inference_us_per_sample", "ratio_vs_fastest", "is_simulator"]
    ]
    for name, train, lat in rows:
        sim = name.startswith(QUANTUM_PREFIXES)
        mark = MARK_A if sim else ""
        lines.append(
            f"{display_model(name)}{mark} & {train:.2f} & {lat:.2f} & "
            f"{_count(lat / fastest)}$\\times$ \\\\"
        )
        rows_csv.append(
            [name, f"{train:.4f}", f"{lat:.4f}", f"{lat / fastest:.1f}", str(sim).lower()]
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "% a = simulated quantum model; latency is statevector-simulator time,",
        "% not a hardware measurement, and is not a deployment claim.",
    ]

    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", rows_csv)
    return tex


def blackbox_table(run_dir: Path, out_name: str) -> Path:
    """Query-based (HopSkipJump) robustness, identical budget for every family.

    Headline column is the median L-inf distortion in STANDARDIZED units that the
    attack needs to flip a prediction: larger = more robust, and unlike a fixed-eps
    F1 it does not reward a model for already being wrong. This is the control for
    the objection that surrogate-transfer attacks under-attack trees and kernel
    SVMs, so it is reported next to the transfer numbers, never instead of them.
    """
    per_seed = json.loads((run_dir / "results.json").read_text())["per_seed"]
    bb = per_seed[0]["blackbox"]
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    order = sorted(bb, key=lambda k: -bb[k]["median_linf_distortion"])
    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & Clean macro-F1 & F1 under attack & Median $\\ell_\\infty$ ($\\sigma$) "
        "& Queries/sample \\\\",
        "\\midrule",
    ]
    rows_csv = [["model", "clean_f1", "adv_f1", "median_linf_sigma", "queries_per_sample"]]
    for name in order:
        r = bb[name]
        lines.append(
            f"{display_model(name)} & {r['clean_f1']:.3f} & {r['adv_f1']:.3f} & "
            f"{r['median_linf_distortion']:.3f} & {_count(r['queries_per_sample'])} \\\\"
        )
        rows_csv.append(
            [
                name,
                f"{r['clean_f1']:.4f}",
                f"{r['adv_f1']:.4f}",
                f"{r['median_linf_distortion']:.4f}",
                f"{r['queries_per_sample']:.0f}",
            ]
        )
    lines += ["\\bottomrule", "\\end{tabular}"]

    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", rows_csv)
    return tex


def diagnostics_combined_table(run_dirs: dict[int, Path], out_name: str) -> Path:
    """Every encoding x qubit count in ONE table — the paper's diagnostics exhibit.

    The per-qubit-count tables remain for the model card; the paper needs the six
    cells side by side because the finding is a cross-cell pattern (large g never
    coincides with label alignment, and accuracy follows KTA).
    """
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Encoding & Qubits & Test macro-F1 & KTA$_q$ & KTA$_{\\mathrm{RBF}}$ "
        "& $g(K_C\\!\\to\\!K_Q)$ \\\\",
        "\\midrule",
    ]
    rows_csv = [
        ["encoding", "qubits", "macro_f1_mean", "kta_quantum", "kta_rbf", "geometric_difference"]
    ]
    for q, run_dir in sorted(run_dirs.items()):
        results = json.loads((run_dir / "results.json").read_text())
        for enc, res in results.items():
            d = res["diagnostics"]
            f1 = res["aggregate"]["macro_f1"]
            lines.append(
                f"{display_encoding(enc)} & {q} & {_fmt(f1['mean'], f1['std'])} & "
                f"{d['kta_quantum']:.3f} & "
                f"{d['kta_classical_rbf']:.3f} & {d['geometric_difference']:.2f} \\\\"
            )
            rows_csv.append(
                [
                    enc,
                    str(q),
                    f"{f1['mean']:.4f}",
                    f"{d['kta_quantum']:.4f}",
                    f"{d['kta_classical_rbf']:.4f}",
                    f"{d['geometric_difference']:.2f}",
                ]
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", rows_csv)
    return tex


def whitebox_cells(
    worst_radius_run_dir: Path, eps: float, per_family: dict, adv_run_dir: Path
) -> dict[str, dict]:
    """The white-box column of the robustness table, read from the worst-radius run.

    Per family: the retention read, the radius it was read at, the retention at the budget, and
    whether the attack ran one radius only. Fails when the run read another budget, or when the
    two families the shared protocol attacks white-box do not reproduce the adversarial run the
    table reads.
    """
    wr = json.loads((worst_radius_run_dir / "results.json").read_text())
    if abs(wr["budget"] - eps) > 1e-12:
        raise ValueError(
            f"{worst_radius_run_dir}: read inside {wr['budget']} sigma, the table is at {eps}"
        )
    models = wr["sources"]["msu"]["models"]
    wb = {
        fam: {
            "retention": models[fam]["worst"]["retention"],
            "radius": models[fam]["worst"]["radius"],
            "at_budget": models[fam]["at_budget"]["retention"],
            "single_radius": False,
        }
        for fam in ("qsvm", "rbf_svm", "mlp", "vqc")
    }
    # the shared protocol attacks these two white-box already, so the worst-radius run must
    # have read the run this table reads
    for fam in ("mlp", "vqc"):
        if abs(wb[fam]["at_budget"] - per_family[fam]["retention"]) > 1e-9:
            raise ValueError(
                f"{worst_radius_run_dir}: {fam}'s retention at the budget is "
                f"{wb[fam]['at_budget']!r}, {adv_run_dir} gives "
                f"{per_family[fam]['retention']!r}; the worst-radius run was built from another "
                "adversarial run"
            )
    return wb


def logreg_whitebox_cell(run_dir: Path, eps: float, clean_mean: float) -> dict:
    """Logistic regression's white-box cell, read from its own run (run_whitebox_logreg.py).

    Fails when the run read another budget or attacked a fit whose clean score is not the
    table's.
    """
    lw = json.loads((run_dir / "results.json").read_text())
    if lw.get("model") != "logreg":
        raise ValueError(f"{run_dir}: a run of {lw.get('model')!r}, not of logistic regression")
    if abs(lw["budget"] - eps) > 1e-12 or abs(lw["clean_f1"] - clean_mean) > 5e-5:
        raise ValueError(
            f"{run_dir}: budget {lw['budget']} and clean macro-F1 {lw['clean_f1']!r} do not "
            f"match this table's {eps} and {clean_mean!r}; the run attacked another fit"
        )
    return {
        "retention": lw["worst"]["retention"],
        "radius": lw["worst"]["radius"],
        "at_budget": lw["at_budget"]["retention"],
        "single_radius": False,
    }


# A score-based black-box run must have attacked the fit this table prints: its clean score,
# seed by seed, has to equal the robustness run's. Every model's does (difference 0.0 on the
# pinned runs) except XGBoost's, whose fit moves with the thread count: its runs used 10, 6
# and 3 threads and differ by up to 0.018 on one seed of the 500 evaluation rows. Another
# configuration of a model differs by more (the earlier VQC by 0.078 on one seed).
SCORE_ATTACK_CLEAN_TOL = 1e-6
SCORE_ATTACK_CLEAN_TOL_BY_MODEL = {"xgb": 0.02}
# attack letter -> the key its runner stores the attacked scores under, per seed and radius
SCORE_ATTACK_KEYS = {"S": "square", "E": "nes"}


def lowest_attacked(candidates: dict[str, tuple[float, float] | None]) -> tuple[float, str, float]:
    """The lowest attacked macro-F1 over the attacks run on a model, the attack that reached
    it and its radius. `candidates` maps an attack letter to (macro-F1, radius) or None: P
    the shared PGD, W the model's own white-box attack, S the Square search, E the
    evolution-strategies attack. Among the attacks within 0.0005 of the lowest score the
    model's own attack is named first, then the Square search, then evolution strategies
    (where the shared PGD is the model's own attack the two are one); the value and the radius
    returned are the named attack's own."""
    cells = {k: v for k, v in candidates.items() if v is not None}
    if not cells:
        raise ValueError("no attack was run on the model")
    low = min(v[0] for v in cells.values())
    letter = next(k for k in ("W", "S", "E", "P") if k in cells and cells[k][0] <= low + 5e-4)
    return cells[letter][0], letter, cells[letter][1]


def score_attack_cells(
    runs: list[dict], budget: float, shared_clean: dict[str, list[float]]
) -> dict[str, dict[str, tuple[float, float]]]:
    """{model: {attack letter: (macro-F1, radius)}} from score-based black-box runs
    (experiments/run_square_blackbox.py, experiments/run_score_based_blackbox.py), each read
    at the tested radius, at or below the budget, that leaves the lowest mean over its seeds.

    `runs` names, per run, the attack letter, the run folder and the models read from it;
    `shared_clean` is the robustness run's clean macro-F1 per model and seed. A model whose
    clean score in a run differs from the robustness run's on any seed is another fit, and
    the build fails."""
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for spec in runs:
        letter, run_dir = spec["attack"], Path(spec["run"])
        per_seed = json.loads((run_dir / "results.json").read_text())["per_seed"]
        key = SCORE_ATTACK_KEYS[letter]
        for fam in spec["models"]:
            for cell in per_seed:
                want = shared_clean[fam][cell["seed"]]
                tol = SCORE_ATTACK_CLEAN_TOL_BY_MODEL.get(fam, SCORE_ATTACK_CLEAN_TOL)
                if abs(cell["clean"][fam] - want) > tol:
                    raise ValueError(
                        f"{run_dir.name}: {fam}'s clean score at seed {cell['seed']} is "
                        f"{cell['clean'][fam]:.4f}, the robustness run's is {want:.4f}: "
                        "another fit of the model"
                    )
            radii = [r for r in per_seed[0][key] if float(r) <= budget + 1e-12]
            if not radii:
                raise ValueError(f"{run_dir.name}: no radius at or below {budget}")
            means = {r: float(np.mean([c[key][r][fam] for c in per_seed])) for r in radii}
            worst = min(means, key=means.get)
            if letter in out.setdefault(fam, {}):
                raise ValueError(f"{fam}: attack {letter} is read from two runs")
            out[fam][letter] = (means[worst], float(worst))
    return out


def robustness_summary_table(
    adv_run_dir: Path,
    random_run_dir: Path,
    blackbox_run_dir: Path,
    out_name: str,
    worst_radius_run_dir: Path,
    hsj_restart_run_dirs: list[Path] | None = None,
    weighted_mlp_run_dir: Path | None = None,
    logreg_whitebox_run_dir: Path | None = None,
    score_attack_runs: list[dict] | None = None,
) -> Path:
    """Retention under the shared protocol, the targeted white-box column, the
    random-perturbation control, and HSJ distortion in ONE table.

    `logreg_whitebox_run_dir` (experiments/run_whitebox_logreg.py) gives logistic regression its
    own attack's cell, read at its worst tested radius like the kernel machines'; the build
    fails unless that run attacked the fit this table's clean score comes from.

    `worst_radius_run_dir` (experiments/read_worst_radius.py; operator ruling (h), decision log
    2026-11-20T10:17Z) supplies the white-box column: each model's own attack read at the grid
    radius, at or below the shared protocol's budget, that leaves the lowest score. An attacker
    bounded by a radius may use any smaller perturbation, so the reading at the budget is not
    the worst case when the curve is not monotone (the two kernel machines here). The CSV
    records the radius read and the reading at the budget beside it.

    `weighted_mlp_run_dir` (E-M4b, decision log 2026-11-16T05:40Z) adds the class-weighted
    MLP as its own row: clean, the shared-protocol PGD replayed exactly, and its own
    white-box retention; its random-flip and HSJ cells were not measured and print "---".
    The unweighted MLP row stays, labelled as the transfer surrogate, because every
    transferred entry in the table is crafted on it.

    These columns must never be published separately: protocol retention alone
    crowns the fidelity-kernel QSVM as most robust, the HSJ column agrees for the
    wrong reason, and the random-flip and white-box columns expose both readings
    as attack-strength artifacts (P5a, decision log 2026-08-14). One table forces
    them to be read together (MODEL_CARD carries the same pairing).

    WB column semantics: retention under the strongest attack available for that
    family. mlp/vqc are white-box under the shared protocol already; qsvm and
    rbf_svm come from the P5a run (kernel machines, targeted gradient attack);
    tree families have no gradient-based white-box attack and show "---", which is itself the
    finding — their protocol retention is an upper bound, not a robustness claim.
    The class-weighted MLP's own attack ran at the budget only, so its cell is that reading,
    an upper bound on its worst case.

    `score_attack_runs` (`score_attack_cells`) adds the bottom line: each model's lowest
    macro-F1 over the attacks run on it at up to the budget, with the attack named (P: the
    shared PGD at the budget; W: its own white-box attack; S: the Square search; E: the
    evolution-strategies attack; the last three at their worst tested radius). Each attack
    is a mean over its own run's seeds, and a score-based run is read for a model only if
    it attacked the fit this table prints. A model no such run names takes its lowest over
    the other attacks.
    """
    st = robustness_stats(adv_run_dir)
    if not st:
        raise ValueError(f"no robustness stats in {adv_run_dir}")
    eps = st["epsilon"]
    ctrl = json.loads((random_run_dir / "results.json").read_text())["per_seed"][0][
        "random_control"
    ]
    bb = json.loads((blackbox_run_dir / "results.json").read_text())["per_seed"][0]["blackbox"]
    # HSJ over restarts (v2 correction batch): ART's initial point is unseeded, so a single
    # draw is not a measurement — when restart artifacts are supplied, the HSJ column
    # becomes mean ± std over restarts and a family missing from them FAILS the build.
    hsj_restarts: dict[str, dict] = {}
    for d in hsj_restart_run_dirs or []:
        hsj_restarts.update(json.loads((d / "results.json").read_text())["summary"])
    wb = whitebox_cells(worst_radius_run_dir, eps, st["per_family"], adv_run_dir)
    score_cells: dict[str, dict[str, tuple[float, float]]] = {}
    if score_attack_runs:
        adv_seeds = json.loads((adv_run_dir / "results.json").read_text())["per_seed"]
        shared_clean = {
            fam: [s["evasion"]["clean"][fam] for s in adv_seeds]
            for fam in adv_seeds[0]["evasion"]["clean"]
        }
        score_cells = score_attack_cells(score_attack_runs, eps, shared_clean)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # The stratified-random floor on THIS evaluation subset. Retention is a ratio and
    # rewards a model whose clean score is already at chance: every attacked score in
    # this table is below the floor, which "88.6% retention" hides (decision log
    # 2026-08-15T10:38Z). `stratified_cap` stratifies on y alone, so the subset is
    # reproduced exactly here without re-running the PCA.
    # Estimated with 2,000 draws, not the 20 that `trivial_baselines` uses for the
    # accuracy tables: a single draw has macro-F1 std ~0.021 here, so 20 draws leave a
    # standard error of ~0.005 — enough to flip the sign of a cell that sits AT the floor,
    # which the fidelity kernel's does. 2,000 draws puts the standard error near 0.0005.
    from sklearn.metrics import f1_score as _f1_score

    from qgridbench.data.preprocess import load_processed
    from qgridbench.models.classical.zoo import stratified_cap

    _X, _y, _sp = load_processed(st.get("variant", "binary"))
    y_tr, y_te = _y[_sp["train"]], _y[_sp["test"]]
    ev = stratified_cap(y_te, y_te, st.get("n_eval", 500), seed=0)
    y_ev = y_te[ev]
    _cls, _cnt = np.unique(y_tr, return_counts=True)
    _prior = _cnt / _cnt.sum()
    _rng = np.random.default_rng(0)
    floor = float(
        np.mean(
            [
                _f1_score(y_ev, _rng.choice(_cls, size=len(y_ev), p=_prior), average="macro")
                for _ in range(2000)
            ]
        )
    )

    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        f"Model & Clean F1 & PGD@{eps}$\\sigma$ & F1$-$floor & Retention "
        "& White-box retention & Lowest F1 & Random flip rate "
        "& HSJ median $\\ell_\\infty$ ($\\sigma$) \\\\",
        "\\midrule",
    ]
    rows_csv = [
        [
            "model",
            "clean_f1",
            "adv_f1",
            "adv_f1_minus_floor",
            "retention",
            "whitebox_retention",
            "random_flip_rate",
            "hsj_median_linf_sigma",
            "hsj_restart_std",
            "hsj_found_share",
            "hsj_n_restarts",
            "whitebox_radius_sigma",
            "whitebox_retention_at_budget",
            "whitebox_single_radius",
            "square_f1",
            "square_radius_sigma",
            "evolution_f1",
            "evolution_radius_sigma",
            "lowest_f1",
            "lowest_attack",
            "lowest_radius_sigma",
        ]
    ]
    rows = dict(st["per_family"])
    labels = {}
    if weighted_mlp_run_dir is not None:
        w = json.loads((weighted_mlp_run_dir / "results.json").read_text())["weighted_table3_row"]
        rows["mlp_weighted"] = {
            "clean_mean": w["clean_mean"],
            "adv_mean": w["shared_mean"],
            "retention": w["retention_shared"],
        }
        # its own attack was run at the budget only
        wb["mlp_weighted"] = {
            "retention": w["retention_whitebox"],
            "radius": eps,
            "at_budget": w["retention_whitebox"],
            "single_radius": True,
        }
        labels = {"mlp": "MLP (unweighted)", "mlp_weighted": "MLP (class-weighted)"}
    if logreg_whitebox_run_dir is not None:
        wb["logreg"] = logreg_whitebox_cell(
            logreg_whitebox_run_dir, eps, rows["logreg"]["clean_mean"]
        )
    for fam, v in sorted(rows.items(), key=lambda kv: -kv[1]["retention"]):
        measured = fam != "mlp_weighted"
        flip = ctrl.get(fam, {}).get("by_radius", {}).get(str(eps), {}).get("flip_rate")
        hsj = bb.get(fam, {}).get("median_linf_distortion")
        hsj_std = None
        hsj_share = None
        hsj_n = None
        if hsj_restarts and measured:
            if fam not in hsj_restarts:
                raise ValueError(f"HSJ restart summary missing family '{fam}'")
            hsj, hsj_std = hsj_restarts[fam]["mean"], hsj_restarts[fam]["std"]
            hsj_n = hsj_restarts[fam].get("n_restarts")  # recorded by the pooled run
            if "found_share" in hsj_restarts[fam]:
                # the attack returns a row unchanged when it cannot start on it: the distortion
                # is read over the rows it found, with their share (decision log 2026-11-20T02:03Z)
                hsj_share = hsj_restarts[fam]["found_share"]["mean"]
                hsj, hsj_std = hsj_restarts[fam]["found"]["mean"], hsj_restarts[fam]["found"]["std"]
        if hsj == 0 and hsj_share is None:
            raise ValueError(
                f"HSJ median for '{fam}' is exactly zero in a run that did not record the found "
                "share: rows the attack could not start on are not a zero distortion; re-run "
                "experiments/run_hopskipjump_restarts.py for this family"
            )
        cell = wb.get(fam)
        wbr = cell["retention"] if cell else None
        wb_radius = cell["radius"] if cell else None
        wb_budget = cell["at_budget"] if cell else None
        flip_tex = f"{flip * 100:.0f}\\%" if flip is not None else "---"
        if hsj is None:
            hsj_tex = "---"
        elif hsj_std is not None:
            hsj_tex = _fmt(hsj, hsj_std)
            if hsj_share is not None and hsj_share < 1:
                hsj_tex += f" ({hsj_share * 100:.0f}\\%)"
        else:
            hsj_tex = f"{hsj:.3f}"
        wb_tex = f"{wbr:.3f}" if wbr is not None else "---"
        if wbr is not None and abs(wb_radius - eps) > 1e-12:
            # read inside the ball: the radius is printed beside the retention
            wb_tex += f" ({wb_radius:g}$\\sigma$)"
        dfloor = v["adv_mean"] - floor
        sq_cell = score_cells.get(fam, {}).get("S")
        es_cell = score_cells.get(fam, {}).get("E")
        low_f1, low_attack, low_radius = lowest_attacked(
            {
                "P": (v["adv_mean"], eps),
                "W": (wbr * v["clean_mean"], wb_radius) if wbr is not None else None,
                "S": sq_cell,
                "E": es_cell,
            }
        )
        lines.append(
            f"{_row_tint(fam)}{labels[fam] if fam in labels else display_model(fam)} & "
            f"{v['clean_mean']:.3f} & "
            f"{v['adv_mean']:.3f} & "
            f"{_num(dfloor, '+.3f')} & {v['retention']:.3f} & {wb_tex} & "
            f"{low_f1:.3f} ({low_attack}) & {flip_tex} & "
            f"{hsj_tex} \\\\"
        )
        rows_csv.append(
            [
                fam,
                f"{v['clean_mean']:.4f}",
                f"{v['adv_mean']:.4f}",
                # six decimals: the robustness figure prints these at three, as this table does
                f"{dfloor:+.6f}",
                f"{v['retention']:.6f}",
                f"{wbr:.6f}" if wbr is not None else "",
                f"{flip:.4f}" if flip is not None else "",
                f"{hsj:.4f}" if hsj is not None else "",
                f"{hsj_std:.4f}" if hsj_std is not None else "",
                f"{hsj_share:.4f}" if hsj_share is not None else "",
                str(hsj_n) if hsj_n is not None else "",
                f"{wb_radius:g}" if wbr is not None else "",
                f"{wb_budget:.6f}" if wbr is not None else "",
                # 1: the attack ran this one radius only, so the cell is an upper bound on
                # the worst case and its radius was not chosen
                str(int(cell["single_radius"])) if cell else "",
                # six decimals: the robustness figure prints the lowest score at three, as this
                # table does from the unrounded value (four would round twice)
                f"{sq_cell[0]:.6f}" if sq_cell else "",
                f"{sq_cell[1]:g}" if sq_cell else "",
                f"{es_cell[0]:.6f}" if es_cell else "",
                f"{es_cell[1]:g}" if es_cell else "",
                f"{low_f1:.6f}",
                low_attack,
                f"{low_radius:g}",
            ]
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", rows_csv)
    return tex


def kernel_exhaustion_table(whitebox_run_dir: Path, out_name: str) -> Path:
    """P7b/P7c exhibit: the classical kernel exhausts under attack, neither quantum model does.

    Two failure modes, one table. `m/m0` is the mean BOUNDED-OUTPUT mass relative to clean
    -- kernel-row mass for the two kernel machines, |<PauliZ>| for the VQC, i.e. per family
    the quantity that would have to collapse for the model to lose output authority --
    and `F1-floor` the distance from the constant-predictor macro-F1 (0 = disabled,
    strongly negative = inverted). The LaTeX exhibit carries those two metrics for the
    three attackable families; the CSV additionally carries the predicted-probability
    spread behind them. Local sensitivity does not distinguish the families (P7a, killed);
    the budget sweep does.
    """
    d = json.loads((whitebox_run_dir / "results.json").read_text())["p7b_kernel_exhaustion"]
    per = d["per_model"]
    missing = [m for m in MODELS_EXHAUSTION if m not in per]
    if missing:
        raise KeyError(
            f"{whitebox_run_dir} predates the P7c sweep: missing {missing} in per_model. "
            f"Re-run experiments/run_whitebox_qsvm.py --only p7b."
        )
    radii = sorted(per["qsvm"], key=float)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "& \\multicolumn{2}{c}{Quantum kernel} & \\multicolumn{2}{c}{Variational (VQC)} "
        "& \\multicolumn{2}{c}{Classical RBF} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "$\\varepsilon$ ($\\sigma$) & $m/m_0$ & F1$-$const. & $m/m_0$ & F1$-$const. "
        "& $m/m_0$ & F1$-$const. \\\\",
        "\\midrule",
    ]
    rows_csv = [
        ["epsilon_sigma"]
        + [
            f"{p}_{stat}"
            for p in ("qsvm", "vqc", "rbf")
            for stat in ("output_mass_ratio", "proba_std", "f1_minus_floor")
        ]
    ]
    for e in radii:
        cells = [per[m][e] for m in MODELS_EXHAUSTION]
        lines.append(
            f"{float(e):g} & "
            + " & ".join(
                f"{c['kernel_mass_ratio']:.3f} & {_num(c['f1_minus_floor'], '+.3f')}" for c in cells
            )
            + " \\\\"
        )
        rows_csv.append(
            [f"{float(e):g}"]
            + [
                f"{c[k]:.4f}"
                for c in cells
                for k in ("kernel_mass_ratio", "proba_std", "f1_minus_floor")
            ]
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", rows_csv)
    return tex


def _write_csv(path: Path, rows) -> None:
    """RFC-4180 quoting: configuration labels carry commas ("QSVM ZZ, 8 qubits"), and a bare
    join silently shifts columns for any standard parser (BACKLOG v2 row, found by N1's
    positive control 2026-08-23T00:40Z)."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)


def _regime(mass: float, f1_minus_floor: float) -> str:
    """Classify a model's end-state under a full-budget attack.

    Thresholds are the ones pre-registered for P8a (decision log 2026-08-14T11:02Z), so
    the label is derived, never assigned by eye: DISABLED means the output collapsed and
    macro-F1 came back to the constant-predictor floor (the attacker holds an off-switch);
    INVERTED means the output survived and macro-F1 sits far below the floor (the attacker
    holds the steering wheel). Anything else is reported as partial rather than forced into
    one of the two.
    """
    if mass <= 0.05 and abs(f1_minus_floor) <= 0.05:
        return "disabled"
    if f1_minus_floor <= -0.2:
        return "inverted"
    return "partial"


def regime_table(
    lengthscale_run: Path, exhaustion_runs: list[tuple[str, Path]], out_name: str
) -> Path:
    """P8 exhibit: the attack end-state is set by LENGTH SCALE, not by classical vs quantum.

    One row per configuration at the largest attack radius. The RBF rows sweep gamma over
    three decades at fixed C and show the classical kernel crossing from disabled to
    inverted; the quantum rows show feature maps that never cross because a bounded,
    periodic encoding has no length scale to exceed. Clean macro-F1 is carried so a row
    that "survives" only because it never learned anything is visible rather than counted
    as evidence.
    """
    ls = json.loads((lengthscale_run / "results.json").read_text())["p8a_lengthscale_control"]
    floor = ls["constant_predictor_floor"]
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for mult, blk in ls["per_width"].items():
        curve = blk["curve"]
        radius = max(curve, key=float)
        last, clean = curve[radius], curve["0.0"]
        tag = "tuned" if float(mult) == 1.0 else f"$\\times${mult}"
        rows.append(
            (
                f"RBF $\\gamma={blk['gamma']:.4g}$ ({tag})",
                clean["macro_f1"],
                last["kernel_mass_ratio"],
                last["f1_minus_floor"],
                radius,
            )
        )
    for label, run in exhaustion_runs:
        per = json.loads((run / "results.json").read_text())["p7b_kernel_exhaustion"]["per_model"]
        name = "vqc" if label.lower().startswith("vqc") else "qsvm"
        curve = per[name]
        radius = max(curve, key=float)
        last, clean = curve[radius], curve["0.0"]
        rows.append(
            (label, clean["macro_f1"], last["kernel_mass_ratio"], last["f1_minus_floor"], radius)
        )

    radii = {r[4] for r in rows}
    if len(radii) != 1:
        raise ValueError(f"regime_table compares rows at different radii: {sorted(radii)}")

    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{@{}lrrrl@{}}",
        "\\toprule",
        "Configuration & \\begin{tabular}[b]{@{}r@{}}Clean\\\\ macro-F1\\end{tabular} & $m/m_0$ "
        "& F1$-$const. & End state \\\\",
        "\\midrule",
    ]
    csv_rows = [
        ["configuration", "clean_macro_f1", "output_mass_ratio", "f1_minus_floor", "regime"]
    ]
    for label, clean, mass, dfloor, _ in rows:
        reg = _regime(mass, dfloor)
        lines.append(
            f"{label} & {clean:.3f} & {mass:.3f} & {_num(dfloor, '+.3f')} & {reg.capitalize()} \\\\"
        )
        csv_rows.append([label, f"{clean:.4f}", f"{mass:.4f}", f"{dfloor:.4f}", reg])
    lines += ["\\bottomrule", "\\end{tabular}"]

    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", csv_rows)
    (TABLE_DIR / f"{out_name}_meta.json").write_text(
        json.dumps(
            {"radius_sigma": float(rows[0][4]), "constant_predictor_floor": floor}, indent=1
        ),
        encoding="utf-8",
    )
    return tex


def quantum_bandwidth_table(run_dir: Path, out_name: str) -> Path:
    """P9a exhibit: the quantum tuning knob swept the way the classical one was.

    The symmetric control to `regime_table`. Retained mass never reaches the disabled
    threshold at any bandwidth -- it is U-shaped, since both encoding extremes are
    degenerate in opposite directions -- yet the end state still moves from inverted to
    floor-level, so a defender has a quantum-side choice too. Clean macro-F1 is carried
    because the whole reading depends on those cells being non-degenerate.
    """
    d = json.loads((run_dir / "results.json").read_text())["p9a_quantum_bandwidth_regime"]
    per, floor = d["per_bandwidth"], d["constant_predictor_floor"]
    tuned = float(d["tuned_bandwidth"])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Bandwidth & \\begin{tabular}[b]{@{}r@{}}Clean\\\\ macro-F1\\end{tabular} & $m/m_0$ "
        "& Std $p$ & F1$-$const. & End state \\\\",
        "\\midrule",
    ]
    csv_rows = [
        [
            "bandwidth",
            "clean_macro_f1",
            "output_mass_ratio",
            "proba_std",
            "f1_minus_floor",
            "regime",
        ]
    ]
    for bw in sorted(per, key=float):
        blk = per[bw]
        curve = blk["curve"]
        last, clean = curve[max(curve, key=float)], curve["0.0"]
        reg = _regime(last["kernel_mass_ratio"], last["f1_minus_floor"])
        label = f"{float(bw):.3g}" + ("\\,(tuned)" if abs(float(bw) - tuned) < 1e-9 else "")
        lines.append(
            f"{label} & {clean['macro_f1']:.3f} & {last['kernel_mass_ratio']:.3f} & "
            f"{last['proba_std']:.3f} & {_num(last['f1_minus_floor'], '+.3f')} & "
            f"{reg.capitalize()} \\\\"
        )
        csv_rows.append(
            [
                f"{float(bw):g}",
                f"{clean['macro_f1']:.4f}",
                f"{last['kernel_mass_ratio']:.4f}",
                f"{last['proba_std']:.4f}",
                f"{last['f1_minus_floor']:.4f}",
                reg,
            ]
        )
    lines += ["\\bottomrule", "\\end{tabular}"]

    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", csv_rows)
    (TABLE_DIR / f"{out_name}_meta.json").write_text(
        json.dumps(
            {
                "constant_predictor_floor": floor,
                "tuned_bandwidth": tuned,
                "min_mass_ratio": min(
                    b["curve"][max(b["curve"], key=float)]["kernel_mass_ratio"]
                    for b in per.values()
                ),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return tex


def objective_regime_table(run_dir: Path, out_name: str) -> Path:
    """P11c/P11d exhibit: the end state belongs to the attacker's objective and the fit.

    Two panels from one artifact. Panel A holds everything fixed and varies only the attack
    OBJECTIVE, which is what `quantum_bandwidth_table` above could not separate -- every attack
    behind that table ascends classification loss, and classically loss-ascent implies distance
    while quantumly it does not. Panel B holds the attacked mass fixed and varies the SVM's
    regularization instead, which is what turns "the output survives" into a statement about the
    fitted model rather than the encoding.
    """
    d = json.loads((run_dir / "results.json").read_text())["p11c_objective_and_regularization"]
    a, b = d["part_a_objective"], d["part_b_regularization"]
    eps, tuned_bw, tuned_c = str(d["paper_budget_sigma"]), d["tuned_bandwidth"], d["tuned_C"]
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "\\multicolumn{5}{@{}l}{\\textit{\\textbf{(a) Objective only: same budget, steps, seed, "
        "and model}}} \\\\",
        "Bandwidth & Loss obj. & Mass obj. & Factor & Baseline \\\\",
        "\\midrule",
    ]
    csv_rows = [["panel", "key", "loss_objective", "mass_objective", "factor", "baseline"]]
    for bw in sorted(a, key=float):
        c = a[bw]["curve"][eps]
        lo, mo, base = c["loss_objective"], c["mass_objective"], a[bw]["baseline_permuted"]
        label = f"{float(bw):.3g}" + ("\\,(tuned)" if abs(float(bw) - tuned_bw) < 1e-9 else "")
        lines.append(
            f"{label} & {lo:.3f} & {mo:.3f} & {lo / max(mo, 1e-12):.1f}$\\times$ & {base:.3f} \\\\"
        )
        csv_rows.append(
            [
                "a",
                f"{float(bw):g}",
                f"{lo:.4f}",
                f"{mo:.4f}",
                f"{lo / max(mo, 1e-12):.3f}",
                f"{base:.4f}",
            ]
        )

    lines += [
        "\\midrule",
        "\\multicolumn{5}{@{}l}{\\textit{\\textbf{(b) Regularization only: attacked mass held "
        "fixed, tuned bandwidth}}} \\\\",
        "$C$ & Clean macro-F1 & $m/m_0$ & Spread & F1$-$const. \\\\",
        "\\midrule",
    ]
    for C in sorted(b, key=float):
        blk, m = b[C], b[C]["mean"]
        label = f"{float(C):.4g}" + ("\\,(tuned)" if abs(float(C) - tuned_c) < 1e-9 else "")
        # a cell whose CLEAN model is already degenerate or already constant cannot be read as
        # "the attacker disabled it" -- marked, never silently included
        flag = MARK_A if blk["degenerate"] or blk["clean_modal_frac"] > 0.99 else ""
        lines.append(
            f"{label}{flag} & {blk['clean_macro_f1']:.3f} & {m['mass_ratio']['mean']:.3f} & "
            f"{m['dec_spread_ratio']['mean']:.3f} & "
            f"{_num(m['f1_minus_floor']['mean'], '+.3f')}$\\pm$"
            f"{m['f1_minus_floor']['std']:.3f} \\\\"
        )
        csv_rows.append(
            [
                "b",
                f"{float(C):g}",
                f"{blk['clean_macro_f1']:.4f}",
                f"{m['mass_ratio']['mean']:.4f}",
                f"{m['dec_spread_ratio']['mean']:.4f}",
                f"{m['f1_minus_floor']['mean']:.4f}",
            ]
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "% a = clean model degenerate or already constant, so not read as disabled by the attack.",
    ]

    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", csv_rows)
    (TABLE_DIR / f"{out_name}_meta.json").write_text(
        json.dumps(
            {
                "constant_predictor_floor": d["constant_predictor_floor"],
                "paper_budget_sigma": d["paper_budget_sigma"],
                "n_steps": d["n_steps"],
                "n_eval": d["n_eval"],
                "tuned_bandwidth": tuned_bw,
                "tuned_C": tuned_c,
                "classical_positive_control": d["classical_positive_control"],
                "min_mass_objective": min(v["curve"][eps]["mass_objective"] for v in a.values()),
                "any_C_disabled": [k for k, v in b.items() if v["disabled"]],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return tex


def three_class_confusion_table(run_dir: Path, out_name: str, out_dir: Path = TABLE_DIR) -> Path:
    """Appendix exhibit (study protocol §8.8): where the three-class task is actually hard.

    Two panels, one per feature regime, from the two confusion artifacts. The two headline
    columns are the off-diagonal cells an operator cares about and a macro-F1 cannot show:
    how often a NATURAL disturbance is called an attack (the false-alarm direction) and how
    often a real attack is called natural (the miss direction). Per-class F1 follows so the
    reader can see which class carries the loss.

    Counts are summed over all 10 seeds before rates are formed, so a rate is the pooled
    estimate rather than a mean of per-seed ratios; per-class F1 keeps its across-seed std.
    `out_dir` is paper/tables; the R4-20 anchor writes a copy elsewhere to compare.
    """
    panels = [
        ("full", "confusion_triple_full.json", "(a) All 128 features"),
        ("pca", "confusion_triple_pca8.json", "(b) PCA-8, the matched-dimensionality regime"),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{@{}lrrrrr@{}}",
        "\\toprule",
    ]
    csv_rows = [
        [
            "regime",
            "model",
            "natural_called_attack",
            "attack_called_natural",
            "f1_noevent",
            "f1_natural",
            "f1_attack",
        ]
    ]
    meta = {}
    for i, (regime, fname, caption) in enumerate(panels):
        d = json.loads((run_dir / fname).read_text())
        meta[regime] = {
            "class_support": d["class_support"],
            "seeds": d["seeds"],
            "tuned_run": d["tuned_run"],
            "n_components": d["n_components"],
        }
        if i:
            lines.append("\\midrule")
        lines += [
            f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{\\textbf{{{caption}}}}}}} \\\\",
            "Model & Natural$\\to$attack & Attack$\\to$natural & F1 no-event & F1 natural "
            "& F1 attack \\\\",
            "\\midrule",
        ]
        for name, m in d["per_model"].items():
            pc = m["per_class"]
            f1 = [pc[str(c)]["f1"]["mean"] for c in (0, 1, 2)]
            # a model that has collapsed onto one class cannot be read as "cautious"; mark it
            flag = MARK_A if min(f1) < 0.10 else ""
            lines.append(
                f"{display_model(name)}{flag} & "
                f"{m['natural_called_attack']:.3f} & {m['attack_called_natural']:.3f} & "
                + " & ".join(f"{v:.3f}" for v in f1)
                + " \\\\"
            )
            csv_rows.append(
                [
                    regime,
                    name,
                    f"{m['natural_called_attack']:.4f}",
                    f"{m['attack_called_natural']:.4f}",
                ]
                + [f"{v:.4f}" for v in f1]
            )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "% a = model collapsed onto one class (an F1 below 0.10), not read as cautious.",
    ]

    tex = out_dir / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(out_dir / f"{out_name}.csv", csv_rows)
    (out_dir / f"{out_name}_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return tex


def multiclass_confusion_counts(run_dir: Path, out_name: str, out_dir: Path = TABLE_DIR) -> Path:
    """arXiv appendix (R4-20): the 37-class confusion counts behind the multiclass figures.

    One CSV in long form (regime, model, true scenario, predicted scenario, count), counts
    summed over the seeds, every cell including zeros, so any row-normalised rate in the
    figure can be recomputed; per-class F1 (mean and std over seeds) sits in a second CSV. A
    37 x 37 matrix does not fit a page as a table, which is why the exhibit is a figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = [["regime", "model", "true_scenario", "predicted_scenario", "count"]]
    f1_rows = [["regime", "model", "scenario", "support", "f1_mean", "f1_std"]]
    meta = {}
    for regime, fname in MULTICLASS_CONFUSION_FILES.items():
        d = json.loads((run_dir / fname).read_text())
        labels = d["labels"]
        meta[regime] = {
            k: d[k]
            for k in (
                "class_support",
                "seeds",
                "tuned_run",
                "n_components",
                "notes",
                "excluded_units",
            )
        }
        for name, m in d["per_model"].items():
            cm = m["confusion_counts"]
            for i, t in enumerate(labels):
                counts += [[regime, name, t, p, cm[i][j]] for j, p in enumerate(labels)]
                f1 = m["per_class"][str(t)]["f1"]
                f1_rows.append(
                    [
                        regime,
                        name,
                        t,
                        d["class_support"][str(t)],
                        f"{f1['mean']:.4f}",
                        f"{f1['std']:.4f}",
                    ]
                )
    out = out_dir / f"{out_name}.csv"
    _write_csv(out, counts)
    _write_csv(out_dir / f"{out_name}_per_class_f1.csv", f1_rows)
    (out_dir / f"{out_name}_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out


def relabel_control_table(run_dirs: list[Path], out_name: str) -> Path:
    """Positive-control table: engineered-label quantum wins per cell (appendix exhibit).

    One row per encoding x qubit x task cell of the relabel control (Huang et al.
    construction): geometric difference g, KTA of the quantum Gram on the grid labels
    vs the engineered labels, and the quantum win over the best freshly re-tuned
    classical opponent (mean +- std over the control's train/test resamples). The
    footer row carries the Pearson r(log g, win) across all cells — the statistic
    quoted in the paper — so it is generated, never hand-typed.
    """
    from scipy import stats as sps

    cells = []
    for d in run_dirs:
        cells.extend(json.loads((d / "relabel_control.json").read_text()))
    if not cells:
        raise ValueError(f"no relabel_control cells under {run_dirs}")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    r, p = sps.pearsonr(np.log([c["g"] for c in cells]), [c["quantum_win_mean"] for c in cells])
    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Cell & $g$ & KTA (grid $y$) & KTA (engineered $y$) & Quantum win \\\\",
        "\\midrule",
    ]
    rows_csv = [["cell", "g", "kta_grid_labels", "kta_engineered_labels", "win_mean", "win_std"]]
    for c in cells:
        lines.append(
            f"{display_cell(c['cell'])} & {c['g']:.2f} & {c['kta_q_grid_labels']:.4f} & "
            f"{c['kta_q_easy_labels']:.3f} & "
            f"{_fmt(c['quantum_win_mean'], c['quantum_win_std'])} \\\\"
        )
        rows_csv.append(
            [
                c["cell"],
                f"{c['g']:.2f}",
                f"{c['kta_q_grid_labels']:.4f}",
                f"{c['kta_q_easy_labels']:.4f}",
                f"{c['quantum_win_mean']:.4f}",
                f"{c['quantum_win_std']:.4f}",
            ]
        )
    lines += [
        "\\midrule",
        f"\\multicolumn{{5}}{{l}}{{Pearson $r(\\log g,\\ \\mathrm{{win}}) = {r:+.2f}$, "
        f"$p = {p:.3f}$ ({len(cells)} cells)}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    rows_csv.append(["pearson_r_log_g_win", f"{r:.4f}", "p_value", f"{p:.4f}", "", ""])
    (TABLE_DIR / f"{out_name}.csv").write_text(
        "\n".join(",".join(map(str, r_)) for r_ in rows_csv), encoding="utf-8"
    )
    return tex


def quantum_diagnostics_table(run_dir: Path, out_name: str) -> Path:
    """KTA / geometric-difference table for quantum kernels."""
    results = json.loads((run_dir / "results.json").read_text())
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "% auto-generated by qgridbench.eval.tables",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Encoding & Test macro-F1 & KTA$_q$ & KTA$_{\\mathrm{RBF}}$ & $g(K_C\\!\\to\\!K_Q)$ \\\\",
        "\\midrule",
    ]
    for enc, res in results.items():
        d = res["diagnostics"]
        f1 = res["aggregate"]["macro_f1"]
        lines.append(
            f"{display_encoding(enc)} & {_fmt(f1['mean'], f1['std'])} & {d['kta_quantum']:.3f} & "
            f"{d['kta_classical_rbf']:.3f} & {d['geometric_difference']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    return tex


def surrogate_control_table(run_dir: Path, out_name: str, arm: str | None = None) -> Path:
    """The transfer asymmetry under the published surrogate vs the corrected one.

    THE CONTROL BEHIND THE C1 CLAIM. Every transfer number in the study is crafted on the
    MLP, which is the one estimator whose class takes no `class_weight` argument and was
    therefore fitted with no imbalance handling (see `CLASS_BALANCE_ROUTE`). This table
    reports both fits side by side. The classical-damage column is the control: it is flat
    across the two fits, so the corrected surrogate is not merely a weaker attack, and the
    change in the VQC column cannot be read as one.

    Rows run in order of how far each surrogate clears the stratified-random floor, so the
    table reads as the dose-response it is (P11a, 2026-08-15T19:52Z): the third arm is a
    different function class rather than a refit, and the asymmetry falls monotonically with
    surrogate competence. `pos. rate` is an MLP-specific diagnostic, blank for that arm; the
    classical column excludes the surrogate itself, which is white-box against its own row.

    `Reverse damage` is the ratio's denominator: the damage VQC-crafted examples do to the
    surrogate itself. `QSVM retention` is the QSVM's transferred retention under each
    surrogate's attack, read only for a config-check arm, whose Table III stats carry the
    QSVM's clean score; it shows whether the attack-availability contrast survives a
    competent surrogate (red-team review 2026-11-19, M4).
    """
    res = json.loads((run_dir / "results.json").read_text())
    # `arm`: read the summary of one arm of a run_vqc_config_check run (R4-23: the surrogate
    # control repeated with the VQC at Table I's configuration, arms "depth4" / "depth6").
    s = res["arms"][arm]["surrogate_summary"] if arm else res["summary"]
    qsvm_clean = (
        res["arms"][arm]["table_iii_stats"]["per_family"]["qsvm"]["clean_mean"] if arm else None
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    arms = [
        ("mlp", "MLP, as tuned"),
        ("mlp_balanced", "MLP, balanced weights"),
        ("rbf_svm", "RBF-SVM (other class)"),
    ]
    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Surrogate & Clean macro-F1 & $-$floor & Positive rate & VQC damage & Classical damage "
        "& Reverse damage & Forward/reverse & QSVM retention \\\\",
        "\\midrule",
    ]
    rows_csv = [
        [
            "surrogate_fit",
            "clean_f1",
            "clean_minus_floor",
            "positive_rate",
            "vqc_damage",
            "classical_damage",
            "reverse_damage",
            "forward_over_reverse",
            "qsvm_transfer_retention",
        ]
    ]
    for key, label in arms:
        a = s[key]
        pos = a.get("surrogate_positive_rate")
        # the peer column is the honest control for an arm that is white-box against itself
        dmg = a["classical_peer_damage"] if key == "rbf_svm" else a["classical_damage"]
        q_ret = None if qsvm_clean is None else a["per_target_adv_f1"]["qsvm"] / qsvm_clean
        lines.append(
            f"{label} & {a['surrogate_clean']:.3f} & "
            f"{_num(a['surrogate_clean_minus_floor'], '+.3f')} & "
            f"{'---' if pos is None else f'{pos:.3f}'} & {a['vqc_damage']:.3f} & "
            f"{dmg:.3f} & {a['reverse_damage']:.3f} & {a['forward_over_reverse']:.2f} & "
            f"{'---' if q_ret is None else f'{q_ret:.3f}'} \\\\"
        )
        rows_csv.append(
            [
                key,
                f"{a['surrogate_clean']:.4f}",
                # six decimals: a figure prints these at three and must not round twice
                f"{a['surrogate_clean_minus_floor']:+.6f}",
                "" if pos is None else f"{pos:.4f}",
                f"{a['vqc_damage']:.4f}",
                f"{dmg:.4f}",
                f"{a['reverse_damage']:.4f}",
                f"{a['forward_over_reverse']:.6f}",
                "" if q_ret is None else f"{q_ret:.4f}",
            ]
        )
    lines += ["\\bottomrule", "\\end{tabular}"]

    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", rows_csv)
    if arm:
        # Every family's transferred retention under each surrogate's attack (red-team review
        # 2026-11-19, M4): does any family's reading depend on the surrogate's fit?
        clean = res["arms"][arm]["table_iii_stats"]["per_family"]
        fams = [f for f in clean if f in s["mlp"]["per_target_adv_f1"]]
        ret_csv = [["family", *(k for k, _ in arms)]]
        for f in fams:
            ret_csv.append(
                [f]
                + [f"{s[k]['per_target_adv_f1'][f] / clean[f]['clean_mean']:.4f}" for k, _ in arms]
            )
        _write_csv(TABLE_DIR / f"{out_name}_transfer_retention.csv", ret_csv)
    (TABLE_DIR / f"{out_name}_meta.json").write_text(
        json.dumps(
            {
                "floor": s["floor"],
                "eps_sigma": s["eps"],
                "eval_positive_rate": s["positive_rate_eval"],
                # a config-check arm records one unit per seed (unit_wall_s), not per_seed
                "n_seeds": len(res["arms"][arm]["unit_wall_s"]) if arm else len(res["per_seed"]),
                "source_run": run_dir.name if not arm else f"{run_dir.name}#{arm}",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return tex


def ray_crossings_table(quantum_run_dir: Path, classical_run_dir: Path, out_name: str) -> Path:
    """P17e exhibit: the interleaving mechanism, measured directly rather than inferred.

    The paper argues HopSkipJump overstates fidelity-kernel robustness because its line search
    assumes it can locate *a* boundary while the kernel's decision regions interleave. This
    samples the decision function densely along a fixed perturbation ray and counts sign
    changes, for the quantum kernel across bandwidth and the tuned RBF-SVM across a matched
    gamma span on the IDENTICAL rays.

    Read `>1 crossing` first -- that is the fraction of rays on which the line-search assumption
    fails. The honest comparison is at each family's own accuracy optimum (marked), where the
    margin is modest; the wide-length-scale divergence is larger but is not where the deployed
    model sits. Parity of the crossing count is NOT reported: it is algebraically identical to
    "the endpoints differ in sign", i.e. a tautology, established 2026-08-15T21:16Z.
    """
    q = json.loads((quantum_run_dir / "results.json").read_text())
    c = json.loads((classical_run_dir / "results.json").read_text())
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    qpeak = max(q["rows"], key=lambda r: r["clean_macro_f1"])["bandwidth"]
    cpeak = max(c["rows"], key=lambda r: r["clean_macro_f1"])["gamma_mult"]

    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Kernel & Length scale & Clean macro-F1 & Crossings/ray & Max & $>$1 crossing \\\\",
        "\\midrule",
    ]
    rows_csv = [
        ["kernel", "length_scale", "clean_f1", "crossings_mean", "crossings_max", "frac_gt1"]
    ]
    for r in q["rows"]:
        mark = MARK_A if r["bandwidth"] == qpeak else ""
        lines.append(
            rf"Fidelity (ZZ) & $\sigma_{{\mathrm{{bw}}}}={r['bandwidth']:.1f}${mark} & "
            f"{r['clean_macro_f1']:.3f} & {r['crossings_mean']:.2f} & {r['crossings_max']:d} & "
            f"{r['frac_rays_multi_crossing'] * 100:.0f}\\% \\\\"
        )
        rows_csv.append(
            [
                "qsvm_zz",
                f"bw={r['bandwidth']}",
                f"{r['clean_macro_f1']:.4f}",
                f"{r['crossings_mean']:.4f}",
                str(r["crossings_max"]),
                f"{r['frac_rays_multi_crossing']:.4f}",
            ]
        )
    lines.append(r"\midrule")
    for r in c["rows"]:
        mark = MARK_A if r["gamma_mult"] == cpeak else ""
        lines.append(
            f"RBF (classical) & $\\gamma\\times{r['gamma_mult']:g}${mark} & "
            f"{r['clean_macro_f1']:.3f} & {r['crossings_mean']:.2f} & {r['crossings_max']:d} & "
            f"{r['frac_rays_multi_crossing'] * 100:.0f}\\% \\\\"
        )
        rows_csv.append(
            [
                "rbf_svm",
                f"gamma_x{r['gamma_mult']}",
                f"{r['clean_macro_f1']:.4f}",
                f"{r['crossings_mean']:.4f}",
                str(r["crossings_max"]),
                f"{r['frac_rays_multi_crossing']:.4f}",
            ]
        )
    lines += [
        "\\bottomrule",
        r"\end{tabular}",
        "% a = that family's accuracy optimum, the comparison the paper quotes.",
    ]

    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", rows_csv)
    (TABLE_DIR / f"{out_name}_meta.json").write_text(
        json.dumps(
            {
                "quantum_run": quantum_run_dir.name,
                "classical_run": classical_run_dir.name,
                "quantum_peak_bandwidth": qpeak,
                "classical_peak_gamma_mult": cpeak,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return tex


def group_split_table(run_dir: Path, out_name: str, ladder_dir: Path | None = None) -> Path:
    """P12a exhibit: the same tuned models under the split protocols on this data.

    First column is this study's headline protocol (stratified split over pooled rows);
    last column holds whole source files out (mean +/- std over partitions). When a P25
    split-protocol-ladder run is supplied, a middle column adds the rule the benchmark
    was introduced with (Hink et al. 2014: rows split inside each source file), so the
    exhibit shows every rung measured. Every model here is tuned; the quantum arm runs
    at its published bandwidth/C, closing the probe's untuned-arm caveat (decision log
    2026-08-16T02:35Z, condition 2). The nearest-neighbor row is the mechanism: the
    fraction of test rows with a train neighbour closer than 0.1 in standardized
    full-feature space.
    """
    data = json.loads((run_dir / "results.json").read_text())
    grp = [r for r in data["rows"] if r["split"] == "group"]
    row = next(r for r in data["rows"] if r["split"] == "row (shipped)")
    wf = []
    if ladder_dir is not None:
        ladder = json.loads((ladder_dir / "results.json").read_text())
        wf = [r for r in ladder["rows"] if r["rule"] == "within-file (Hink)"]
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    def agg(rows_, key):
        vals = [r[key] for r in rows_]
        return float(np.mean(vals)), float(np.std(vals))

    def fmt_ms(m, s, signed=False, pct=False):
        # A spread too tight for three decimals is shown at two significant figures
        # (0.00015, never a misleading 0.000), matching how the prose quotes it.
        if pct:
            return f"{100 * m:.1f}\\,$\\pm$\\,{100 * s:.1f}\\%"
        dec = 3
        if s > 0 and round(s, 3) == 0:
            dec = math.ceil(-math.log10(s)) + 1
        if signed:
            return f"${m:+.3f}\\,\\pm\\,{s:.{dec}f}$"
        return f"{_num(m)}\\,$\\pm$\\,{s:.{dec}f}"

    def line(label, key, signed=False, pct=False):
        rv = row[key]
        first = f"{100 * rv:.1f}\\%" if pct else (f"${rv:+.3f}$" if signed else _num(rv))
        cells = [first]
        if wf:
            cells.append(fmt_ms(*agg(wf, key), signed=signed, pct=pct))
        cells.append(fmt_ms(*agg(grp, key), signed=signed, pct=pct))
        return f"{label} & " + " & ".join(cells) + " \\\\"

    ncols = 3 if wf else 2
    header = " & Row-level split"
    if wf:
        header += " & Within-file split"
    header += " & Whole files held out \\\\"
    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        rf"\begin{{tabular}}{{l{'r' * ncols}}}",
        r"\toprule",
        header,
        r"\midrule",
        line("Full 128 features, tuned LightGBM", "level_full_lgbm"),
        line("PCA-8 (2000-row cap), tuned XGBoost", "matched_classical"),
        line("PCA-8 (2000-row cap), tuned ZZ-map QSVM", "matched_quantum"),
        line(r"\quad Contrast (classical $-$ quantum)", "matched_contrast", signed=True),
        r"\midrule",
        line("Stratified-random floor", "floor_macro_f1"),
        line("Test rows with train neighbor $<0.1$", "nn_frac_within_0.1", pct=True),
        r"\bottomrule",
        r"\end{tabular}",
    ]

    keys = [
        "level_full_lgbm",
        "matched_classical",
        "matched_quantum",
        "matched_contrast",
        "floor_macro_f1",
        "nn_frac_within_0.1",
    ]
    head = ["quantity", "row_level"]
    if wf:
        head += ["within_file_mean", "within_file_std"]
    head += ["group_mean", "group_std"]
    rows_csv = [head]
    for k in keys:
        cells = [k, f"{row[k]:.4f}"]
        if wf:
            wm, ws = agg(wf, k)
            cells += [f"{wm:.4f}", f"{ws:.5f}"]
        gm, gs = agg(grp, k)
        cells += [f"{gm:.4f}", f"{gs:.4f}"]
        rows_csv.append(cells)

    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", rows_csv)
    meta = {"run": run_dir.name, "verdict": data["verdict"]}
    if ladder_dir is not None:
        meta["ladder_run"] = ladder_dir.name
    (TABLE_DIR / f"{out_name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return tex


def group_split_partitions_table(run_dir: Path, out_name: str) -> Path:
    """Per-partition detail behind the paired-protocol exhibit (appendix).

    The body quotes the aggregate and states that the matched contrast flips sign on one
    partition; this table is the exhibit for that sentence -- every held-out partition
    with its file ids, so the sign flip is visible rather than asserted.
    """
    data = json.loads((run_dir / "results.json").read_text())
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        "Partition & Held-out files & LightGBM (all features) & XGBoost & ZZ QSVM & Contrast "
        "& Floor \\\\",
        r"\midrule",
    ]
    rows_csv = [
        [
            "partition",
            "held_out_files",
            "level_full_lgbm",
            "matched_classical",
            "matched_quantum",
            "matched_contrast",
            "floor_macro_f1",
        ]
    ]
    for r in data["rows"]:
        name = "row (shipped)" if r["split"] == "row (shipped)" else f"p{r['partition']}"
        held = "--" if r["test_groups"] is None else ",".join(str(g) for g in r["test_groups"])
        shown = "Row-level" if r["split"] == "row (shipped)" else name
        lines.append(
            f"{shown} & {held.replace(',', ', ')} & {r['level_full_lgbm']:.4f} & "
            f"{r['matched_classical']:.4f} & "
            f"{r['matched_quantum']:.4f} & ${r['matched_contrast']:+.4f}$ & "
            f"{r['floor_macro_f1']:.4f} \\\\"
        )
        rows_csv.append(
            [
                name,
                held.replace(",", ";"),
                f"{r['level_full_lgbm']:.4f}",
                f"{r['matched_classical']:.4f}",
                f"{r['matched_quantum']:.4f}",
                f"{r['matched_contrast']:+.4f}",
                f"{r['floor_macro_f1']:.4f}",
            ]
        )
    lines += [r"\bottomrule", r"\end{tabular}"]

    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", rows_csv)
    (TABLE_DIR / f"{out_name}_meta.json").write_text(
        json.dumps({"run": run_dir.name}, indent=2), encoding="utf-8"
    )
    return tex


MLP_ROW_MARK = "(mlp row"  # how a merged run's sources.txt marks the run its MLP row is from
MLP_WEIGHTED_RUN = re.compile(r"(?<!un)weighted_mlp_baseline", re.IGNORECASE)


def _mlp_is_class_weighted(run_dir: Path) -> bool:
    """Whether a merged run's MLP row is the class-weighted fit: its `sources.txt` then has one
    line marked as the MLP row's source, and that line names a weighted-MLP run. With no such
    line the MLP row is the merged accuracy run's, the unweighted fit. A merged run without the
    file, or with two such lines, fails loud."""
    sources = Path(run_dir) / "sources.txt"
    if not sources.exists():
        raise FileNotFoundError(f"{sources}: needed to tell which MLP fit the run holds")
    rows = [ln for ln in sources.read_text(encoding="utf-8").splitlines() if MLP_ROW_MARK in ln]
    if len(rows) > 1:
        raise ValueError(f"{sources}: {len(rows)} lines claim the MLP row")
    return bool(rows) and MLP_WEIGHTED_RUN.search(rows[0]) is not None


def matched_contest_table(run_dirs: dict[int, Path], out_name: str) -> Path:
    """The matched contest at every dimensionality in one compact table.

    Test macro-F1 (the mean over seeds) per model and PCA dimensionality, read from the merged
    run of each dimensionality (the `results.json` that `accuracy_table` reads), with that
    table's marks: a, the best model at that dimensionality; b, significantly below it
    (Holm-corrected p < 0.05). Mark c: the MLP cell is the unweighted fit (each merged run's
    `sources.txt` says which fit it holds). A model with no run at a dimensionality prints a
    dash; a chance baseline is never the best model. The marks compare models with seeds, so a
    model without seeds (the deterministic QSVM) that leads a dimensionality fails loud.
    """
    dims = sorted(run_dirs)
    results = {d: json.loads((Path(run_dirs[d]) / "results.json").read_text()) for d in dims}
    mlp_weighted = {d: _mlp_is_class_weighted(run_dirs[d]) for d in dims}
    models = list(dict.fromkeys(m for d in dims for m in results[d]))
    if not models:
        raise ValueError("matched contest table: the merged runs hold no models")
    marks = {}
    for d in dims:
        contest = {m: r for m, r in results[d].items() if family(m) != "reference"}
        marks[d] = _holm_vs_best(contest)
        level = {m: r["aggregate"]["macro_f1"]["mean"] for m, r in contest.items()}
        leader = max(level, key=level.get)
        if marks[d][0] is not None and level[leader] > level[marks[d][0]]:
            raise ValueError(
                f"matched contest table: {leader} leads at {d} dimensions without seeds, so "
                f"mark a on {marks[d][0]} would mislead"
            )
    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{l" + "r" * len(dims) + "}",
        "\\toprule",
        f" & \\multicolumn{{{len(dims)}}}{{c}}{{Dimensions (qubits)}} \\\\",
        "Model & " + " & ".join(str(d) for d in dims) + " \\\\",
        "\\midrule",
    ]
    rows_csv = [["model", *[f"macro_f1_pca{d}" for d in dims]]]
    for m in models:
        shown, raw = [], []
        for d in dims:
            if m not in results[d]:
                shown.append("--")
                raw.append("")
                continue
            value = results[d][m]["aggregate"]["macro_f1"]["mean"]
            best, corrected = marks[d]
            letters = ["a"] if m == best else ["b"] if corrected.get(m, 1.0) < 0.05 else []
            if m == "mlp" and not mlp_weighted[d]:
                letters.append("c")
            mark = f"$^{{{','.join(letters)}}}$" if letters else ""
            shown.append(f"{_num(value)}{mark}")
            raw.append(f"{value:.4f}")
        lines.append(f"{_row_tint(m)}{display_model(m)} & " + " & ".join(shown) + " \\\\")
        rows_csv.append([m, *raw])
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "% a = best model at that dimensionality; b = significantly below it (Holm-corrected",
        "% p < 0.05); c = the unweighted MLP fit.",
    ]
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", rows_csv)
    return tex


SWEEP_GROUPS = (
    ("labelled_attack", "Rows the model labels attack"),
    ("labelled_natural", "Rows the model labels natural"),
)
SWEEP_NOISE_RULES = (
    ("", "rows_labelled_attack_noise_flips", "at least {n} of its draws"),
    ("_any_draw", "rows_labelled_attack_noise_flips_any_draw", "any draw"),
)
SWEEP_COUNT_WORDS = {1: "one", 2: "two", 3: "three"}


def _sweep_decimals(cells: list[dict]) -> int:
    """Decimals for one block of distortions: two, or three when a mean of the block is under
    0.1 sigma, so a block never mixes precisions."""
    return 3 if min(c["mean"] for c in cells) < 0.095 else 2


def _sweep_cell(mean: float, std: float, decimals: int) -> str:
    """A distortion in sigma with its spread over restarts at the block's decimals. A spread that
    is not zero but would print as zero is given as under the last printed digit."""
    spread = f"{std:.{decimals}f}"
    if std > 0 and float(spread) == 0:
        spread = f"$<$\\,{10**-decimals:.{decimals}f}"
    return f"{mean:.{decimals}f}\\,$\\pm$\\,{spread}"


def hsj_query_sweep_table(verdict_dir: Path, out_name: str) -> Path:
    """The HopSkipJump query-budget sweep (experiments/read_hsj_query_sweep.py): per budget, the
    registered median over both groups of found rows, the median within the rows a model labels
    attack and within those it labels natural, and how many of the attack-labelled rows that
    random noise flips the attack still places beyond the noise radius. Columns are the budgets
    in the verdict's order; each cell is a mean over that budget's restarts.
    """
    v = json.loads((verdict_dir / "results.json").read_text())
    after = v["after_the_result"]
    budgets = list(v["per_budget"])
    families = [f for f in v["per_budget"][budgets[0]] if f != "settings"]
    control = after["noise_control"]
    radius = control["radius_sigma"]

    lines = [
        "% auto-generated by qgridbench.eval.tables -- do not edit by hand",
        "\\begin{tabular}{l" + "r" * len(budgets) + "}",
        "\\toprule",
    ]
    queries = []
    for b in budgets:
        per_family = [v["per_budget"][b][f]["queries_per_sample"] for f in families]
        queries.append((min(q["min"] for q in per_family), max(q["max"] for q in per_family)))
    lines.append(
        "Queries per point & "
        + " & ".join(f"{_count(lo)}--{_count(hi)}" for lo, hi in queries)
        + " \\\\"
    )
    lines.append("\\midrule")
    rows_csv = [["block", "model", "rows", *budgets]]
    span = len(budgets) + 1

    def block(title: str) -> None:
        lines.append(f"\\multicolumn{{{span}}}{{l}}{{\\emph{{{title}}}}} \\\\")

    block("Median over both groups (registered)")
    pooled = {f: [v["per_budget"][b][f] for b in budgets] for f in families}
    decimals = _sweep_decimals([c for cells in pooled.values() for c in cells])
    for f, cells in pooled.items():
        lines.append(
            f"{display_model(f)} & "
            + " & ".join(_sweep_cell(c["mean"], c["std"], decimals) for c in cells)
            + " \\\\"
        )
        rows_csv.append(["both_groups", f, "", *[f"{c['mean']:.4f}" for c in cells]])
    for group, title in SWEEP_GROUPS:
        lines.append("\\midrule")
        block(title)
        grouped = {f: [after["by_direction"][b][f][group] for b in budgets] for f in families}
        decimals = _sweep_decimals([c for cells in grouped.values() for c in cells])
        for f, cells in grouped.items():
            rows = {c["rows"] for c in cells}
            if len(rows) != 1:
                raise ValueError(f"{f}: the budgets disagree on the rows of {group}: {rows}")
            n = rows.pop()
            lines.append(
                f"{display_model(f)} ({n} rows) & "
                + " & ".join(_sweep_cell(c["mean"], c["std"], decimals) for c in cells)
                + " \\\\"
            )
            rows_csv.append([group, f, str(n), *[f"{c['mean']:.4f}" for c in cells]])
    for suffix, key, rule in SWEEP_NOISE_RULES:
        rule = rule.format(n=SWEEP_COUNT_WORDS[control["flipped_in_at_least"]])
        lines.append("\\midrule")
        block(
            f"Attack-labelled rows that noise flips in {rule}: "
            f"how many the attack places beyond ${radius:g}\\sigma$"
        )
        for f in families:
            n = control[key][f]
            means = [
                after["noise_rows"][b][f]["mean"][f"noise_flips{suffix}_attack_beyond"]
                for b in budgets
            ]
            # a model with no such row has nothing to place
            shown = [f"{m:.1f}" if n else "--" for m in means]
            lines.append(f"{display_model(f)} ({n} rows) & " + " & ".join(shown) + " \\\\")
            rows_csv.append(
                [f"noise_flips{suffix}_attack_beyond", f, str(n), *[f"{m:.4f}" for m in means]]
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tex = TABLE_DIR / f"{out_name}.tex"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{out_name}.csv", rows_csv)
    return tex
