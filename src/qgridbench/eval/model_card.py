"""Rewrite MODEL_CARD.md's headline blocks from the generated table artifacts.

Current-state numbers are never hand-typed into doc prose: each copy is a pin that
drifts silently and costs a manual coherence sweep. The card instead carries
marker-fenced blocks that this module rewrites from `paper/tables/*.csv` — the
same artifacts the LaTeX tables are built from, so the card and the paper cannot
disagree. (The generated-block pattern from cog / terraform-docs, at file scale.)

Blocks are delimited by:
    <!-- BEGIN GENERATED: <key> -->
    ...
    <!-- END GENERATED: <key> -->
Everything outside the markers is authored prose and is left untouched. A missing
block is an error, not a silent skip: a card that quietly stops updating one table
is worse than one that fails to build.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from qgridbench.eval.stats import robustness_stats
from qgridbench.utils.run_tracking import REPO_ROOT, git_describe

CARD = REPO_ROOT / "MODEL_CARD.md"
TABLE_DIR = REPO_ROOT / "paper" / "tables"

BEGIN = "<!-- BEGIN GENERATED: {key} -->"
END = "<!-- END GENERATED: {key} -->"


def _read_csv(name: str) -> list[dict[str, str]]:
    path = TABLE_DIR / f"{name}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(row: dict[str, str], key: str, fmt: str = "{:.4f}") -> str:
    raw = row.get(key, "")
    try:
        return fmt.format(float(raw))
    except (TypeError, ValueError):
        return "—"


def _headline_block() -> str:
    """Best model per regime + every quantum model at matched dimensionality."""
    lines = [
        "| Model | regime | macro-F1 | AUPRC | ROC-AUC | Brier | ECE |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, regime in (
        ("accuracy_binary_full", "full (128)"),
        ("rq1_binary_pca8", "PCA-8"),
        ("rq1_binary_pca12", "PCA-12"),
        ("rq1_binary_pca16", "PCA-16"),
    ):
        rows = _read_csv(label)
        if not rows:
            continue
        real = [r for r in rows if not r["model"].startswith("baseline_")]
        best = max(real, key=lambda r: float(r["macro_f1"]))
        quantum = [r for r in rows if r["model"].startswith(("qsvm_", "vqc"))]
        # the chance floor belongs beside the models, not in a footnote: several
        # PCA-regime models sit within noise of it and one is below it
        floor = [r for r in rows if r["model"] == "baseline_stratified_random"]
        for row in [best, *quantum, *floor]:
            # bold marks the regime winner only — bolding every cell marks nothing
            f1 = _num(row, "macro_f1")
            if row is best:
                name = f"**{row['model']}** (best)"
            elif row["model"].startswith("baseline_"):
                name = f"_{row['model']}_"
            else:
                name = row["model"]
            lines.append(
                f"| {name} | {regime} | {f'**{f1}**' if row is best else f1} | "
                f"{_num(row, 'auprc')} | {_num(row, 'roc_auc')} | "
                f"{_num(row, 'brier')} | {_num(row, 'ece')} |"
            )
    return "\n".join(lines) if len(lines) > 2 else "_no accuracy artifacts yet_"


def _restart_cell(row: dict[str, str] | None) -> str:
    """HopSkipJump distortion over restarts, as the paper's robustness table prints it."""
    if not row or not row.get("hsj_median_linf_sigma"):
        return "—"
    cell = f"{float(row['hsj_median_linf_sigma']):.3f} ± {float(row['hsj_restart_std']):.3f}"
    share = row.get("hsj_found_share")
    if share and float(share) < 1:
        cell += f" on {float(share) * 100:.0f}% of points"
    if row.get("hsj_n_restarts"):
        cell += f" (n = {int(float(row['hsj_n_restarts']))})"
    return cell


def _blackbox_block() -> str:
    rows = _read_csv("blackbox_binary_q8")
    if not rows:
        return "_no black-box artifacts yet_"
    # the restart column is read from the paper's robustness table, so the card cannot quote
    # a restart figure the paper does not (decision log 2026-11-20T08:16Z); without that table
    # the column is left out, and a model missing from it fails the build
    robust = _read_csv("robustness_binary_q8")
    restarts = {r["model"]: r for r in robust}
    head = "| Model | clean F1 | F1 under attack | median $\\ell_\\infty$ (σ), one draw | "
    if robust:
        lines = [
            head + "queries/sample | **median $\\ell_\\infty$ (σ) over restarts (n restarts)** |",
            "|---|---|---|---|---|---|",
        ]
    else:
        lines = [head + "queries/sample |", "|---|---|---|---|---|"]
    not_started = False
    for r in rows:
        median = _num(r, "median_linf_sigma", "{:.3f}")
        # a median of exactly zero is the attack returning most inputs unchanged because it
        # found no starting point on them, not a distortion (decision log 2026-11-20T02:54Z)
        if r.get("median_linf_sigma") and float(r["median_linf_sigma"]) == 0:
            median, not_started = f"{median} †", True
        line = (
            f"| {r['model']} | {_num(r, 'clean_f1', '{:.3f}')} | "
            f"{_num(r, 'adv_f1', '{:.3f}')} | "
            f"{median} | "
            f"{_num(r, 'queries_per_sample', '{:.0f}')} |"
        )
        if robust:
            if r["model"] not in restarts:
                raise ValueError(
                    f"black-box model '{r['model']}' has no row in the robustness table, so its "
                    "restart cell cannot be filled"
                )
            line += f" {_restart_cell(restarts[r['model']])} |"
        lines.append(line)
    if not robust:
        lines += ["", "_No robustness table yet, so the restart column is not shown._"]
    if not_started:
        lines += [
            "",
            "† Not a distortion: on most of the 100 points the attack found no starting point "
            "and returned the input unchanged. See the note below the table.",
        ]
    return "\n".join(lines)


ROW_LABELS = {"mlp_weighted": "mlp (class-weighted)"}


def _robustness_block(adv_run: Path | None = None) -> str:
    """Retention under attack beside the white-box column and the random-flip control.

    These must be shown together. Protocol retention alone makes the fidelity kernel
    look like the most robust model in the study; the flip rate shows random noise
    breaks it more easily than the transferred attack does, and the white-box column
    confirms it directly (P5a). Reporting any of them without the others is the
    failure this table exists to prevent.

    The rows are the paper's robustness table (`robustness_binary_q8.csv`), read as written.
    This block used to recompute them from the newest `*_adversarial_binary_q8` run and so kept
    an earlier VQC than the paper's (decision log 2026-11-20T08:16Z). `adv_run` is the run that
    table was built from; it supplies the radius, the seed count and the Holm count.
    """
    rows = _read_csv("robustness_binary_q8")
    if not rows:
        return "_no robustness artifacts yet_"
    if adv_run is None:
        raise ValueError(
            "the robustness table exists but refresh() was not given the run it was built "
            "from (robustness_run); the block's radius, seed count and Holm count come from it"
        )
    st = robustness_stats(adv_run)
    if not st:
        raise ValueError(f"no robustness stats in {adv_run}")

    lines = [
        f"At ε = {st['epsilon']}σ, {st['n_seeds']} seeds. "
        "Retention = adversarial / clean macro-F1.",
        "",
        "| Model | clean | under attack | retention | white-box retention | "
        "random-perturbation flip rate |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        flip = f"{float(r['random_flip_rate']) * 100:.0f}%" if r.get("random_flip_rate") else "—"
        lines.append(
            f"| {ROW_LABELS.get(r['model'], r['model'])} | {_num(r, 'clean_f1', '{:.3f}')} | "
            f"{_num(r, 'adv_f1', '{:.3f}')} | {_num(r, 'retention', '{:.3f}')} | "
            f"{_whitebox_cell(r, st['epsilon'])} | {flip} |"
        )
    lines += [
        "",
        f"Quantum-vs-classical paired comparisons significant after Holm correction: "
        f"**{st['n_significant_holm']} of {st['n_comparisons']}** — in *both* directions, "
        "which is why no family-level quantum robustness claim is made.",
        "",
        "The **white-box retention** column is the model's own white-box attack, read at the "
        "grid radius inside the budget that leaves the lowest score: an attacker bounded by a "
        "radius may use any smaller perturbation. Where that radius is not the budget it is "
        "printed beside the figure. The class-weighted MLP's own attack ran at the budget only, "
        "so its figure is an upper bound on its worst case. A `—` "
        "there means the family received no white-box attack (the tree ensembles admit no "
        "gradient-based one, and no tree-specific attack was run), so its "
        "retention figure is an upper bound, not a robustness measurement. Logistic regression's "
        "figure is from its own run of ten attack starts. A `—` in the flip-rate column means it "
        "was not measured.",
    ]
    return "\n".join(lines)


def _whitebox_cell(row: dict, eps: float) -> str:
    """The model's own white-box retention, with its radius where that is inside the ball, and
    marked where the attack ran one radius only (an upper bound on the worst case)."""
    if not row.get("whitebox_retention"):
        return "—"
    if not row.get("whitebox_radius_sigma") or row.get("whitebox_single_radius") not in ("0", "1"):
        raise ValueError(
            f"{row.get('model')}: the robustness table gives a white-box retention without the "
            "radius it was read at; rebuild it with `python -m experiments.make_report`"
        )
    cell = f"{float(row['whitebox_retention']):.3f}"
    radius = float(row["whitebox_radius_sigma"])
    if row["whitebox_single_radius"] == "1":
        return f"{cell} at {radius:g}σ only"
    if abs(radius - eps) > 1e-12:
        cell += f" at {radius:g}σ"
    return cell


def _cost_block() -> str:
    """Train wall-clock and per-sample inference latency — the deployability axis.

    Grid detection is real-time, so accuracy alone does not decide anything. Quantum
    rows are simulator timings and are labelled; they are not a hardware claim.
    """
    rows = _read_csv("cost_binary_pca8")
    if not rows:
        return "_no timing artifacts yet_"
    lines = [
        "| Model | train (s) | inference (µs/sample) | vs fastest |",
        "|---|---|---|---|",
    ]
    for r in rows:
        sim = r.get("is_simulator") == "true"
        name = f"{r['model']} ‡" if sim else r["model"]
        lines.append(
            f"| {name} | {_num(r, 'train_seconds', '{:.2f}')} | "
            f"{_num(r, 'inference_us_per_sample', '{:.2f}')} | "
            f"{_num(r, 'ratio_vs_fastest', '{:.0f}')}× |"
        )
    lines += [
        "",
        "‡ simulated quantum model — statevector-simulator latency, not a device "
        "measurement and not a deployment claim.",
    ]
    return "\n".join(lines)


def _diagnostics_block() -> str:
    runs = REPO_ROOT / "results" / "runs"
    lines = [
        "| Encoding | qubits | KTA (quantum) | KTA (RBF) | geometric difference |",
        "|---|---|---|---|---|",
    ]
    for q in (8, 12, 16):
        hits = sorted(
            d for d in runs.glob(f"*quantum_qkernel_binary_q{q}*") if (d / "results.json").exists()
        )
        if not hits:
            continue
        for enc, res in json.loads((hits[-1] / "results.json").read_text()).items():
            d = res["diagnostics"]
            lines.append(
                f"| {enc} | {q} | {d['kta_quantum']:.4f} | "
                f"{d['kta_classical_rbf']:.4f} | {d['geometric_difference']:.2f} |"
            )
    return "\n".join(lines) if len(lines) > 2 else "_no quantum diagnostics yet_"


def _provenance_block() -> str:
    return (
        f"- Generated: **{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}**\n"
        f"- Code version: `{git_describe()}`\n"
        "- Regenerate with `python -m experiments.make_report` — every figure above is read\n"
        "  from `paper/tables/*.csv`, which is itself written only from `results/runs/`."
    )


BLOCKS = {
    "headline": _headline_block,
    "robustness": _robustness_block,
    "blackbox": _blackbox_block,
    "cost": _cost_block,
    "diagnostics": _diagnostics_block,
    "provenance": _provenance_block,
}


def refresh(card: Path = CARD, robustness_run: Path | None = None) -> list[str]:
    """Rewrite every generated block in the card. Returns the keys updated.

    `robustness_run` is the adversarial run the paper's robustness table was built from.
    """
    text = card.read_text(encoding="utf-8")
    updated = []
    builders = {**BLOCKS, "robustness": partial(_robustness_block, robustness_run)}
    for key, builder in builders.items():
        begin, end = BEGIN.format(key=key), END.format(key=key)
        if begin not in text or end not in text:
            raise ValueError(f"MODEL_CARD.md is missing the '{key}' generated block markers")
        head, rest = text.split(begin, 1)
        _stale, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{builder()}\n{end}{tail}"
        updated.append(key)
    card.write_text(text, encoding="utf-8")
    return updated
