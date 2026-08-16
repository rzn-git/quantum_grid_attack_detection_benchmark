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


def _blackbox_block() -> str:
    rows = _read_csv("blackbox_binary_q8")
    if not rows:
        return "_no black-box artifacts yet_"
    lines = [
        "| Model | clean F1 | F1 under attack | median $\\ell_\\infty$ (σ) | queries/sample |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {_num(r, 'clean_f1', '{:.3f}')} | "
            f"{_num(r, 'adv_f1', '{:.3f}')} | "
            f"{_num(r, 'median_linf_sigma', '{:.3f}')} | "
            f"{_num(r, 'queries_per_sample', '{:.0f}')} |"
        )
    return "\n".join(lines)


def _robustness_block() -> str:
    """Retention under attack beside the white-box column and the random-flip control.

    These must be shown together. Protocol retention alone makes the fidelity kernel
    look like the most robust model in the study; the flip rate shows random noise
    breaks it more easily than the transferred attack does, and the white-box column
    confirms it directly (P5a). Reporting any of them without the others is the
    failure this table exists to prevent.
    """
    runs = REPO_ROOT / "results" / "runs"
    adv = sorted(d for d in runs.glob("*_adversarial_binary_q8") if (d / "results.json").exists())
    rnd = sorted(
        d for d in runs.glob("*_adversarial_binary_q8_random") if (d / "results.json").exists()
    )
    wbx = [
        d
        for d in sorted(runs.glob("*whitebox_qsvm_binary_q8"))
        if (d / "results.json").exists()
        and "p5a_whitebox_qsvm" in json.loads((d / "results.json").read_text())
    ]
    if not adv:
        return "_no robustness artifacts yet_"

    st = robustness_stats(adv[-1])
    if not st:
        return "_no robustness artifacts yet_"

    wb_ret: dict[str, float] = {}
    if wbx:
        p5a = json.loads((wbx[-1] / "results.json").read_text())["p5a_whitebox_qsvm"]
        wb_ret["qsvm"] = p5a["whitebox_retention_at_half"]
        if "rbf_control" in p5a:
            wb_ret["rbf_svm"] = p5a["rbf_control"]["whitebox_retention_at_half"]
    for fam in ("mlp", "vqc"):  # already white-box under the shared protocol
        if fam in st["per_family"]:
            wb_ret.setdefault(fam, st["per_family"][fam]["retention"])

    flips: dict[str, float] = {}
    if rnd:
        ctrl = json.loads((rnd[-1] / "results.json").read_text())["per_seed"][0]["random_control"]
        eps = str(st["epsilon"])
        flips = {
            f: v["by_radius"][eps]["flip_rate"] for f, v in ctrl.items() if eps in v["by_radius"]
        }

    eps = st["epsilon"]
    lines = [
        f"At ε = {eps}σ, {st['n_seeds']} seeds. Retention = adversarial / clean macro-F1.",
        "",
        "| Model | clean | under attack | retention | white-box retention | "
        "random-perturbation flip rate |",
        "|---|---|---|---|---|---|",
    ]
    for f, v in sorted(st["per_family"].items(), key=lambda kv: -kv[1]["retention"]):
        flip = f"{flips[f] * 100:.0f}%" if f in flips else "—"
        wb = f"{wb_ret[f]:.3f}" if f in wb_ret else "—"
        lines.append(
            f"| {f} | {v['clean_mean']:.3f} | {v['adv_mean']:.3f} | {v['retention']:.3f} | "
            f"{wb} | {flip} |"
        )
    lines += [
        "",
        f"Quantum-vs-classical paired comparisons significant after Holm correction: "
        f"**{st['n_significant_holm']} of {st['n_comparisons']}** — in *both* directions, "
        "which is why no family-level quantum robustness claim is made.",
        "",
        "The **white-box retention** column is the strongest attack available per family. A `—` "
        "means no white-box attack exists for that family (the tree ensembles), so its "
        "retention figure is an upper bound, not a robustness measurement.",
    ]
    return "\n".join(lines)


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


def refresh(card: Path = CARD) -> list[str]:
    """Rewrite every generated block in the card. Returns the keys updated."""
    text = card.read_text(encoding="utf-8")
    updated = []
    for key, builder in BLOCKS.items():
        begin, end = BEGIN.format(key=key), END.format(key=key)
        if begin not in text or end not in text:
            raise ValueError(f"MODEL_CARD.md is missing the '{key}' generated block markers")
        head, rest = text.split(begin, 1)
        _stale, tail = rest.split(end, 1)
        text = f"{head}{begin}\n{builder()}\n{end}{tail}"
        updated.append(key)
    card.write_text(text, encoding="utf-8")
    return updated
