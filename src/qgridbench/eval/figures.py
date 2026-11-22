"""All paper figures are generated ONLY here, from results/runs JSON.

Design tokens (single source of truth) live in eval.design_tokens. Figures write
to figures/ as PDF (vector, paper-ready).
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter, ScalarFormatter
from pennylane.drawer import MPLDrawer

from qgridbench.eval.design_tokens import (
    COLORS,
    COLUMN_WIDTH_PT,
    FIGSIZE,
    FRAME_PT,
    LABEL_PT,
    MEANING,
    MODEL_MARKERS,
    OUTLINE_PT,
    apply_style,
    family,
    family_color,
    latex_color_defs,
    tint,
)
from qgridbench.eval.tables import (
    MULTICLASS_CONFUSION_FILES,
    display_cell,
    display_model,
)
from qgridbench.utils.run_tracking import REPO_ROOT

FIG_DIR = REPO_ROOT / "figures"

# Matplotlib stamps a wall-clock /CreationDate into every PDF, which makes an
# otherwise deterministic figure differ byte-for-byte on each run. Suppressing it
# is what lets a reviewer re-run make_report and get artifacts identical to the
# released ones — the same reason run artifacts carry a git SHA rather than a
# timestamp in their content. Tables were already byte-stable; this closes the gap.
PDF_METADATA = {"CreationDate": None}

# IEEE PES author kit, Part 4a: figure labels are 8-10 pt at the size they print. Every
# figure is drawn at its native placed width (design_tokens: one column or the text width),
# so the label size is the size it prints at; a figure scaled down in LaTeX shrinks its
# labels by the same factor. Circuits are drawn long and rescaled to one column instead.
CIRCUIT_LABEL_PT = LABEL_PT
CIRCUIT_PAD_IN = 0.02
TIGHT_PAD_IN = 0.02  # a cropped figure keeps this much margin
# PennyLane draws a gate box 0.75 layers wide and sizes its label for a drawing one inch per
# layer. Shrunk to one column the pitch is about 18 pt, so an 8-pt "RZ" (12 pt wide) overran its
# 13-pt box. A wider box (0.90 of the pitch) and thin outlines (design_tokens.FRAME_PT, the
# weight of every white-filled box) keep every label and the measurement glyph inside its box
# at the printed size.
CIRCUIT_BOX_LENGTH = 0.9
# The VQC drawing is saved at the ZZ drawing's scale times this factor, so the two circuits show
# nearly one gate size. Its three-letter "Rot" label needs the extra 6% to keep 1 pt of clearance
# inside its box.
CIRCUIT_VQC_RELATIVE_SCALE = 1.06
CIRCUIT_RC = {
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "patch.facecolor": "white",
    "patch.edgecolor": "black",
    "patch.force_edgecolor": True,
    "patch.linewidth": FRAME_PT,
    "lines.color": "black",
    "lines.linewidth": FRAME_PT,
    "text.color": "black",
    "path.sketch": None,
    "font.weight": "normal",
}


def write_design_colors(out: Path = REPO_ROOT / "paper" / "tables" / "design_colors.tex") -> Path:
    """The meaning colors as LaTeX `\\definecolor` lines for the TikZ diagrams and table rows, so
    the diagrams and these figures read one palette (design_tokens.MEANING)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex_color_defs(), encoding="utf-8", newline="\n")
    return out


def _model_style(model_key: str) -> dict:
    """Line style of one model: its family's hue and its own marker; quantum lines drawn heavier,
    because the paper's comparison is between the families."""
    quantum = family(model_key) == "quantum"
    return {
        "color": family_color(model_key),
        "marker": MODEL_MARKERS.get(model_key, "o"),
        "linewidth": 1.8 if quantum else 1.0,
        "markersize": 5 if quantum else 4,
        "zorder": 3 if quantum else 2,
    }


def _ramp(hex_color: str):
    """A one-hue sequential color map, white to `hex_color` (magnitude in one meaning)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("ramp", ["#FFFFFF", hex_color])


def _plain_log_ticks(axis, subs=(1.0, 3.0)) -> None:
    """Decimal labels on a log axis (0.1, 0.3, 1, 3): no 10^x superscripts, which print small."""
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_formatter(NullFormatter())


def robustness_curves(adv_run_dir: Path, out_name: str, attack: str = "pgd") -> Path:
    """macro-F1 vs epsilon per model family, mean over seeds."""
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads((adv_run_dir / "results.json").read_text())
    per_seed = data["per_seed"]
    families = list(per_seed[0]["evasion"]["curves"].keys())
    eps_grid = sorted(
        {float(e) for f in families for e in per_seed[0]["evasion"]["curves"][f][attack]}
    )

    fig, ax = plt.subplots(figsize=FIGSIZE["col_tall"])
    for fam in families:
        curve = np.array(
            [
                [
                    seed["evasion"]["curves"][fam][attack][_key(seed, fam, attack, e)]
                    for e in eps_grid
                ]
                for seed in per_seed
            ]
        )
        mean, std = curve.mean(0), curve.std(0)
        style = _model_style(fam)
        ax.plot(eps_grid, mean, label=display_model(fam), **style)
        ax.fill_between(eps_grid, mean - std, mean + std, color=style["color"], alpha=0.12, lw=0)
    # unit is fractions of the TRAIN per-feature sigma, not raw PCA coordinates
    # (attacks/evasion.py SCALE CONTRACT) — say so on the axis, not just in prose
    ax.set_xlabel(f"{attack.upper()} perturbation budget\n(training standard deviations)")
    ax.set_ylabel("Test macro-F1")
    # legend under the axes: eight 8-pt entries would cover the curves inside a column
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


def _key(seed, fam, attack, e):
    """Epsilon keys are JSON strings; match float e to the stored key."""
    keys = seed["evasion"]["curves"][fam][attack]
    for k in keys:
        if abs(float(k) - e) < 1e-12:
            return k
    raise KeyError(e)


def transfer_heatmap(adv_run_dir: Path, out_name: str) -> Path:
    """Source x target degradation matrix (mean over seeds), at max epsilon.

    Drawn at the full text width (place as a two-column `figure*`): eight target columns at
    one column width left the 8-pt cell values touching each other.
    """
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads((adv_run_dir / "results.json").read_text())
    mats = [s["evasion"]["transfer_matrix"] for s in data["per_seed"]]
    sources = list(mats[0].keys())
    targets = list(mats[0][sources[0]].keys())
    M = np.mean([[[m[s][t] for t in targets] for s in sources] for m in mats], axis=0)

    fig, ax = plt.subplots(figsize=FIGSIZE["text_heatmap"], layout="constrained")
    # attack hue, darker where the attack leaves less macro-F1 (more damage)
    im = ax.imshow(1 - M, cmap=_ramp(MEANING["attack"]), vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(
        range(len(targets)),
        [display_model(t) for t in targets],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(len(sources)), [display_model(t) for t in sources])
    ax.set_xlabel("Target model")
    ax.set_ylabel("Attack source\nmodel")
    for i in range(len(sources)):
        for j in range(len(targets)):
            ax.text(
                j,
                i,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=LABEL_PT,
                color="white" if M[i, j] < 0.4 else "black",
            )
    bar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        fraction=0.06,
        pad=0.04,
        label="Test macro-F1 under attack (darker: more damage)",
    )
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1.0], labels=["1.00", "0.75", "0.50", "0.25", "0.00"])
    out = FIG_DIR / f"{out_name}.pdf"
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


@contextmanager
def _circuit_style():
    """Draw circuits with the print-size outlines and box width above, then restore both."""
    previous = MPLDrawer._box_length
    MPLDrawer._box_length = CIRCUIT_BOX_LENGTH
    try:
        with plt.rc_context(CIRCUIT_RC):
            yield
    finally:
        MPLDrawer._box_length = previous


def _save_circuit(fig, out: Path, scale: float | None = None) -> float:
    """Scale a drawn circuit so its native width is one column, then save it tightly.

    PennyLane lays a circuit out at one inch per layer and sets gate labels in points, so a
    long circuit shrunk to a column loses its label size. Here the text stays at
    CIRCUIT_LABEL_PT and the drawing is resized instead: the gate boxes shrink around the
    text, which is why the circuits are drawn with few layers and without angle values.

    With `scale` given, the drawing is resized by that factor instead, so a shorter circuit
    takes the gate size of the one-column circuit that returned it and comes out narrower
    than the column. Returns the factor used.
    """
    if scale is None:
        target = COLUMN_WIDTH_PT - 2 * CIRCUIT_PAD_IN * 72
        tight_pt = fig.get_tightbbox(fig.canvas.get_renderer()).width * 72
        scale = target / tight_pt
    fig.set_size_inches(fig.get_figwidth() * scale, fig.get_figheight() * scale)
    fig.savefig(out, metadata=PDF_METADATA, bbox_inches="tight", pad_inches=CIRCUIT_PAD_IN)
    plt.close(fig)
    return scale


def circuit_diagrams(n_qubits: int = 3, zz_reps: int = 1) -> list[Path]:
    """Render the actual model circuits (paper methodology figure).

    Draws the ZZ feature map from qkernel's own circuit builder (SSOT: the drawing
    can never drift from what the experiments ran) and the VQC circuit
    (AngleEmbedding + StronglyEntanglingLayers, mirroring vqc.py's `circuit`),
    gate-expanded so the reader sees actual operations, not template boxes.

    The ZZ drawing is one 252-pt column wide natively and the VQC drawing is saved at the same
    scale times CIRCUIT_VQC_RELATIVE_SCALE, so the shorter VQC circuit is narrower than the column
    (place it at its natural size). Both carry 8-pt gate labels. To get there the drawings show
    three qubits (the experiments run 8--16), ONE repetition of the ZZ map (the deployed map
    repeats that block twice), and gate names without angle values: the angles are 2 x_i on
    the single-qubit RZ gates and 2 x_i x_j on the pair RZ gates, which the caption states.
    """
    import pennylane as qml

    from qgridbench.models.quantum.qkernel import _circuit_state_fn

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = np.round(np.linspace(0.4, 1.3, n_qubits), 2)
    draw = {"style": "rcParams", "decimals": None, "fontsize": CIRCUIT_LABEL_PT}
    outs: list[Path] = []
    apply_style()

    with _circuit_style():
        fig, _ = qml.draw_mpl(_circuit_state_fn("zz", n_qubits, n_reps=zz_reps), **draw)(x)
        out = FIG_DIR / "circuit_zz.pdf"
        scale = _save_circuit(fig, out)
    outs.append(out)

    dev = qml.device("default.qubit", wires=n_qubits)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=n_qubits)
    weights = np.round(np.linspace(0.1, 0.9, int(np.prod(shape))), 2).reshape(shape)

    @qml.qnode(dev)
    def vqc_circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    with _circuit_style():
        fig, _ = qml.draw_mpl(vqc_circuit, level="device", **draw)(x, weights)
        out = FIG_DIR / "circuit_vqc.pdf"
        _save_circuit(fig, out, scale=scale * CIRCUIT_VQC_RELATIVE_SCALE)
    outs.append(out)
    return outs


def bandwidth_diagnostics_figure(bw_json: Path, out_name: str) -> Path:
    """Bandwidth sweep per cell: spectrum shape (top row), validation macro-F1 (middle row) and
    the two screening quantities, g and KTA, on one log axis (bottom row). One axis per panel:
    a twin axis lets the two scales be lined up at will and suggests a relation they do not have.

    The exhibit behind the degeneracy finding: at small bandwidth the Gram is
    near-constant (median off-diagonal ~1, top eigenvalue share ~1), at large
    bandwidth near-identity (frac below 0.01 -> 1); F1 tracks KTA while g peaks
    exactly where the spectrum degenerates.
    """
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(bw_json.read_text())

    def _cell_key(cell: str) -> tuple[str, int]:
        # natural order: encoding first, then numeric qubit count (else q8 sorts after q16)
        enc, rest = cell.split("_q", 1)
        return enc, int(rest.split("_", 1)[0])

    cells = sorted({r["cell"] for r in rows}, key=_cell_key)

    fig, axes = plt.subplots(
        3, len(cells), figsize=(FIGSIZE["text_tall"][0], 5.6), sharex=True, sharey="row"
    )
    for j, cell in enumerate(cells):
        sub = sorted((r for r in rows if r["cell"] == cell), key=lambda r: r["bandwidth"])
        bw = [r["bandwidth"] for r in sub]

        ax = axes[0, j]
        ax.plot(
            bw,
            [r["median_offdiag"] for r in sub],
            marker="o",
            markersize=3,
            color=MEANING["quantum"],
            label="Median off-diagonal fidelity",
        )
        ax.plot(
            bw,
            [r["top_eig_share"] for r in sub],
            marker="s",
            markersize=3,
            color=MEANING["quantum"],
            linestyle="--",
            label="Top-eigenvalue share",
        )
        ax.plot(
            bw,
            [r["frac_below_0.01"] for r in sub],
            marker="^",
            markersize=3,
            color=MEANING["quantum"],
            linestyle=":",
            label="Fraction of pairs below 0.01",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_xscale("log")
        _plain_log_ticks(ax.xaxis, subs=(1.0,))  # panels are 120 pt wide: decades only
        ax.set_title(display_cell(cell))
        if j == 0:
            ax.set_ylabel("Spectral shape (fraction)")

        ax = axes[1, j]
        ax.plot(
            bw,
            [r["val_macro_f1"] for r in sub],
            marker="D",
            markersize=3,
            color=MEANING["quantum"],
        )
        ax.axvline(sub[0]["tuned_bandwidth"], color=MEANING["reference"], linestyle=":", lw=1)
        if j == 0:
            ax.set_ylabel("Validation macro-F1")

        ax = axes[2, j]
        ax.plot(
            bw,
            [r["g_vs_tuned_rbf"] for r in sub],
            marker="X",
            markersize=3.5,
            color=MEANING["classical"],
            linestyle="-.",
            label="Geometric difference",
        )
        ax.plot(
            bw,
            [r["kta_grid_labels"] for r in sub],
            marker="v",
            markersize=3,
            color=MEANING["quantum"],
            label="Kernel-target alignment",
        )
        ax.axvline(sub[0]["tuned_bandwidth"], color=MEANING["reference"], linestyle=":", lw=1)
        ax.set_yscale("log")
        _plain_log_ticks(ax.yaxis, subs=(1.0,))
        ax.set_xlabel("Encoding bandwidth\n(scale factor)")
        if j == 0:
            ax.set_ylabel("Geometric difference,\nalignment (log scale)")

    # legends sit outside the panels: seven 8-pt entries would cover the curves
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    # the bottom legend names the bottom row only; no mark repeats one of the top legend's
    fig.legend(*axes[2, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


def poisoning_sensor_figure(adv_run_dir: Path, out_name: str) -> Path:
    """Label-flip poisoning and benign sensor-noise degradation, one two-panel figure.

    Left: clean-test macro-F1 after retraining on flipped labels (single seed — the
    poisoning stage of run_adversarial retrains at seeds[0] only; the caption must
    say so). Right: macro-F1 under additive Gaussian sensor noise at 3 SNR levels,
    mean +- std over the 10 evasion seeds, with the clean point as reference.
    """
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads((adv_run_dir / "results.json").read_text())
    per_seed = data["per_seed"]
    poison = data["poisoning"]
    families = list(per_seed[0]["sensor_noise"].keys())

    fig, (ax_p, ax_n) = plt.subplots(1, 2, figsize=FIGSIZE["text"], sharey=True)

    rates = sorted(float(r) for r in poison)
    for fam in families:
        ax_p.plot(
            [r * 100 for r in rates],
            [poison[_num_key(poison, r)][fam] for r in rates],
            label=display_model(fam),
            **_model_style(fam),
        )
    ax_p.set_xlabel("Training labels flipped (%)")
    ax_p.set_ylabel("Macro-F1 on clean test data")

    snrs = sorted((int(s) for s in per_seed[0]["sensor_noise"][families[0]]), reverse=True)
    xs = list(range(len(snrs) + 1))  # clean + descending SNR = increasing severity
    for fam in families:
        style = _model_style(fam)
        clean = np.array([s["evasion"]["clean"][fam] for s in per_seed])
        curve = np.array([[s["sensor_noise"][fam][str(snr)] for snr in snrs] for s in per_seed])
        mean = np.concatenate([[clean.mean()], curve.mean(0)])
        std = np.concatenate([[clean.std()], curve.std(0)])
        ax_n.plot(xs, mean, **style)
        ax_n.fill_between(xs, mean - std, mean + std, color=style["color"], alpha=0.12, lw=0)
    ax_n.set_xticks(xs, ["Clean"] + [str(s) for s in snrs])
    ax_n.set_xlabel("Sensor-noise signal-to-noise ratio (dB)")

    fig.legend(*ax_p.get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


def _num_key(d: dict, x: float) -> str:
    """JSON object keys are strings; match float x to the stored key."""
    for k in d:
        if abs(float(k) - x) < 1e-12:
            return k
    raise KeyError(x)


def bandwidth_fragility_figure(frag_json: Path, out_name: str) -> Path:
    """Random-flip fragility co-peaks with accuracy along the bandwidth axis.

    Two stacked panels on one bandwidth axis: clean macro-F1 (top) and the random-perturbation
    flip rate at the largest radius (bottom), each on its own scale, with the accuracy-optimal
    bandwidth marked in both. Backs the no-safe-operating-point claim: the accuracy-optimal
    bandwidth is also the fragility optimum.
    """
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    points = json.loads(frag_json.read_text())
    bws = [p["bandwidth"] for p in points]
    f1 = [p["clean_macro_f1"] for p in points]
    eps = max(points[0]["flip_rate"], key=float)
    flips = np.array([p["flip_rate"][eps]["draws"] for p in points])
    best = bws[int(np.argmax(f1))]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=FIGSIZE["col"], sharex=True)
    ax.plot(bws, f1, marker="o", color=MEANING["quantum"])
    ax.set_xscale("log")
    _plain_log_ticks(ax.xaxis)
    ax.set_ylabel("QSVM clean\nmacro-F1")
    ax2.errorbar(
        bws, flips.mean(1), yerr=flips.std(1), marker="s", color=MEANING["control"], capsize=2
    )
    ax2.set_ylabel(f"Random-noise flip rate\n({float(eps)} standard deviations)")
    ax2.set_xlabel("Encoding bandwidth (scale factor)")
    for a in (ax, ax2):
        a.axvline(best, color=MEANING["reference"], linestyle=":", linewidth=1)

    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout()
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


def barren_plateau_figure(scan_json: Path, depth_json: Path, out_name: str) -> Path:
    """Trainability exhibit: gradient variance decays on the qubit axis, not depth.

    Left: single-parameter gradient variance at initialization vs qubit count
    (controlled scan, log10 y) with a 2^-n reference anchored at the smallest width.
    Right: gradient-norm variance during training vs circuit depth (depth sweep) —
    flat, on a linear axis so flatness is visible. The two panels measure different
    quantities (init-time single-parameter variance vs training-time norm variance);
    the caption must say so.
    """
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    scan = json.loads(scan_json.read_text())
    qubits = sorted(int(q) for q in scan["by_qubits"])
    var = [scan["by_qubits"][str(q)]["var_single_param"] for q in qubits]
    depth_sweep = json.loads(depth_json.read_text())
    depths = sorted(int(k.split("_")[1]) for k in depth_sweep)
    gvar = [depth_sweep[f"depth_{d}"]["grad_norm_var_mean"] for d in depths]

    fig, (ax_q, ax_d) = plt.subplots(1, 2, figsize=FIGSIZE["text"])
    # log10 on a linear axis with integer ticks: a log axis labels its decades 10^-7 ... in 5.6-pt
    # superscripts, and six decades are too many for decimal labels
    ref = [var[0] * 2.0 ** (qubits[0] - q) for q in qubits]
    ax_q.plot(qubits, np.log10(var), marker="o", color=MEANING["quantum"], label="Measured")
    ax_q.plot(
        qubits,
        np.log10(ref),
        linestyle="--",
        color=MEANING["reference"],
        label="Reference (halves per qubit)",
    )
    ax_q.set_xticks(qubits)
    ax_q.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_q.set_xlabel("Number of qubits (depth fixed)")
    ax_q.set_ylabel("Loss-gradient variance\nat initialization (log10)")
    ax_q.legend(loc="upper right", frameon=False)

    ax_d.plot(depths, gvar, marker="s", color=MEANING["quantum"])
    ax_d.set_xticks(depths)
    ax_d.set_ylim(0, max(gvar) * 1.3)
    ax_d.set_xlabel("Circuit depth (layers, 8 qubits)")
    ax_d.set_ylabel("Gradient-norm variance\nduring training")

    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout()
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


def data_efficiency_curve(
    de_run_dir: Path, out_name: str, filename: str = "data_efficiency.json"
) -> Path:
    """test macro-F1 vs training-set size for the best classical + quantum models.

    `filename`: the R4-24 regeneration writes the VQC at Table I's configuration to
    data_efficiency_vqc_table1.json in its own ablations run."""
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads((de_run_dir / filename).read_text())
    fig, ax = plt.subplots(figsize=(FIGSIZE["col"][0], 2.8))  # 2.8 in: fits its arXiv page
    for name, series in data.items():
        sizes = sorted(int(s) for s in series)
        mean = [series[str(s)]["mean"] for s in sizes]
        std = [series[str(s)]["std"] for s in sizes]
        ax.errorbar(
            sizes, mean, yerr=std, label=display_model(name), capsize=2, **_model_style(name)
        )
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Training-set size (samples)")
    ax.set_ylabel("Test macro-F1")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=1, frameon=False)
    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


MULTICLASS_TICK_EVERY = 4  # 37 scenario markers at 8 pt overrun a half-width axis; label every 4th


def multiclass_confusion_figure(run_dir: Path, regime: str, out_name: str) -> Path:
    """arXiv appendix (R4-20): 37 x 37 confusion matrices, one panel per model, one regime.

    Rows are the true scenario marker and are normalised to 1 (counts summed over seeds before
    the division, as in the three-class table), so a rare scenario's errors are as visible as
    a common one's. The counts themselves are in the companion CSV (tables.py).
    """
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = json.loads((run_dir / MULTICLASS_CONFUSION_FILES[regime]).read_text())
    labels, models = d["labels"], list(d["per_model"])
    ncols = 2
    nrows = math.ceil(len(models) / ncols)
    width = FIGSIZE["text"][0]
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, 0.5 * width * nrows), layout="constrained", squeeze=False
    )
    ticks = list(range(0, len(labels), MULTICLASS_TICK_EVERY))
    im = None
    for ax, name in zip(axes.flat, models, strict=False):
        cm = np.asarray(d["per_model"][name]["confusion_counts"], dtype=float)
        rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        # one hue per panel, the model's family (magnitude in one meaning, never a rainbow)
        im = ax.imshow(rates, cmap=_ramp(family_color(name)), vmin=0, vmax=1, aspect="equal")
        ax.grid(False)
        ax.set_title(display_model(name))
        ax.set_xticks(ticks, [str(labels[i]) for i in ticks])
        ax.set_yticks(ticks, [str(labels[i]) for i in ticks])
        ax.set_xlabel("Predicted scenario")
        ax.set_ylabel("True scenario")
    for ax in axes.flat[len(models) :]:
        ax.set_axis_off()
    fig.colorbar(
        im,
        ax=axes,
        orientation="horizontal",
        fraction=0.03,
        pad=0.02,
        label="Share of the true scenario's test rows",
    )
    out = FIG_DIR / f"{out_name}.pdf"
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


MULTICLASS_SUMMARY_PANELS = (("full", "lgbm"), ("pca", "lgbm"), ("pca", "qsvm"))
MULTICLASS_REGIME_NAMES = {"full": "all 128 features", "pca": "PCA-8, 2000-row cap"}


def multiclass_summary_figure(run_dir: Path, out_name: str) -> Path:
    """arXiv appendix (R4-20): the 37-scenario task in one exhibit.

    Top: test macro-F1 per model (mean over the run's seeds, per_seed.csv), all features (filled)
    against the matched PCA-8 regime (hollow), joined per model, with the uniform-guess level.
    Bottom: row-normalised confusion matrices for the best full-feature model, the same model at
    PCA-8, and the QSVM (MULTICLASS_SUMMARY_PANELS). The other panels are in
    `multiclass_confusion_figure` and the counts in the companion CSV.
    """
    import pandas as pd

    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(Path(run_dir) / "per_seed.csv").groupby(["regime", "model"])["macro_f1"]
    mean = scores.mean()
    conf = {
        regime: json.loads((Path(run_dir) / fname).read_text())
        for regime, fname in MULTICLASS_CONFUSION_FILES.items()
    }
    missing = [p for p in MULTICLASS_SUMMARY_PANELS if p[1] not in conf[p[0]]["per_model"]]
    if missing:
        raise KeyError(f"{run_dir}: no multiclass confusion for {missing}")
    n_classes = len(conf["full"]["labels"])
    models = sorted(
        {m for _, m in mean.index}, key=lambda m: -max(mean.get((r, m), 0.0) for r in conf)
    )
    fig = plt.figure(figsize=(FIGSIZE["text"][0], 4.6), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=(1.0, 1.75))
    ax = top.subplots()
    ys = np.arange(len(models))[::-1]
    for y, m in zip(ys, models, strict=True):
        hue = family_color(m)
        have = [r for r in ("full", "pca") if (r, m) in mean.index]
        if len(have) == 2:
            ax.plot([mean[("pca", m)], mean[("full", m)]], [y, y], color=hue, lw=1.0, zorder=1)
        for r in have:
            ax.plot(
                mean[(r, m)],
                y,
                "o",
                ms=6,
                mec=hue,
                mew=1.2,
                mfc=hue if r == "full" else "white",
                zorder=2,
            )
    ax.axvline(1 / n_classes, color=MEANING["reference"], lw=0.8, ls="--")
    ax.text(1 / n_classes + 0.01, ys[-1] - 0.62, f"uniform guess (1/{n_classes})", va="center")
    ax.set_yticks(ys, [display_model(m) for m in models])
    ax.set_ylim(-1.0, len(models) - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Test macro-F1, {n_classes} scenario classes")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[
            plt.Line2D([], [], ls="none", marker="o", ms=6, color=MEANING["reference"]),
            plt.Line2D(
                [], [], ls="none", marker="o", ms=6, mfc="white", mec=MEANING["reference"], mew=1.2
            ),
        ],
        labels=[MULTICLASS_REGIME_NAMES["full"], MULTICLASS_REGIME_NAMES["pca"]],
        loc="lower right",
        frameon=False,
    )
    axes = bottom.subplots(1, len(MULTICLASS_SUMMARY_PANELS))
    labels = conf["full"]["labels"]
    ticks = list(range(0, len(labels), MULTICLASS_TICK_EVERY))
    for a, (regime, m) in zip(axes, MULTICLASS_SUMMARY_PANELS, strict=True):
        cm = np.asarray(conf[regime]["per_model"][m]["confusion_counts"], dtype=float)
        rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = a.imshow(rates, cmap=_ramp(family_color(m)), vmin=0, vmax=1, aspect="equal")
        a.grid(False)
        a.set_title(f"{display_model(m)}, {MULTICLASS_REGIME_NAMES[regime]}")
        a.set_xticks(ticks, [str(labels[i]) for i in ticks])
        a.set_yticks(ticks, [str(labels[i]) for i in ticks])
        a.set_xlabel("Predicted scenario")
        bar = bottom.colorbar(im, ax=a, orientation="horizontal", fraction=0.05, pad=0.03)
        if a is axes[len(axes) // 2]:
            bar.set_label("Share of the true scenario's test rows")
    axes[0].set_ylabel("True scenario")
    out = FIG_DIR / f"{out_name}.pdf"
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


def split_leak_figure(
    group_run_dir: Path,
    decay_run_dir: Path,
    fdia_split_run_dir: Path,
    out_name: str,
    per_class_run_dir: Path,
) -> Path:
    """The split protocol as an evaluation choice, and its mechanism (draft v3, Fig. B).

    (a) Tuned full-feature LightGBM macro-F1 under the row-level split against holding whole
    groups out: MSU/ORNL whole source files (run_group_split_control), and the two synthetic
    FDIA grids' day folds (run_fdia_split_gap), which have no near-duplicate rows.
    (b) MSU/ORNL macro-F1 against the distance in released rows from a test row to its nearest
    training row (run_leak_decay_curve arm A, LightGBM, matched rows), with the runner's own
    fit f(g) = a + b exp(-(g - 1) / tau) and the whole-file level for reference; the score of the
    rows next to a training row is printed at its point.
    (c) Why, and which class pays, on the MSU/ORNL data under both splits: the share of test rows
    with a training row closer than 0.1 in standardized full-feature space
    (run_group_split_control), and the recall of each class (run_level_per_class). The per-class
    run must have reproduced this group-split run, or the build fails. Whole-file bars are means
    over the held-out partitions with their standard deviations, as in (a).
    """
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    group = json.loads((group_run_dir / "results.json").read_text())
    gv = group["verdict"]
    dec = json.loads((decay_run_dir / "results.json").read_text())
    fd = json.loads((fdia_split_run_dir / "results.json").read_text())["variants"]
    pc = json.loads((per_class_run_dir / "results.json").read_text())
    if pc["anchor"]["archived_run"] != group_run_dir.name or not pc["anchor"]["pass"]:
        raise ValueError(
            f"{per_class_run_dir}: anchored on {pc['anchor']['archived_run']!r} "
            f"(pass={pc['anchor']['pass']}), the figure reads {group_run_dir.name!r}"
        )
    labels = ["MSU/ORNL", "FDIA 14-bus", "FDIA 118-bus"]
    row = [gv["level_row"]]
    held = [gv["level_group_mean"]]
    held_sd = [gv["level_group_std"]]
    for v in ("fdia14_binary", "fdia118_binary"):
        r = next(x for x in fd[v]["rows"] if x["split"] == "row")
        s = fd[v]["summary"]["day_level_full_lgbm"]
        row.append(r["level_full_lgbm"])
        held.append(s["mean"])
        held_sd.append(s["std"])
    fig, (a1, a2, a3) = plt.subplots(
        1, 3, figsize=(FIGSIZE["text"][0], 1.95), gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]}
    )
    x = np.arange(len(labels))
    w = 0.38
    a1.bar(x - w / 2, row, w, color=MEANING["choice"], label="Row-level split")
    a1.bar(
        x + w / 2,
        held,
        w,
        yerr=held_sd,
        capsize=2,
        color=MEANING["control"],
        label="Whole files (MSU/ORNL) or days (FDIA) held out",
    )
    for i in range(len(labels)):
        a1.annotate(
            f"−{row[i] - held[i]:.3f}",  # a true minus (IEEE: a minus is not a hyphen)
            (x[i], max(row[i], held[i]) + 0.02),
            ha="center",
            fontsize=LABEL_PT,
        )
    a1.axhline(
        gv["floor_group_mean"],
        color=MEANING["reference"],
        lw=0.8,
        ls="--",
        label="Stratified-random floor, MSU/ORNL",
    )
    a1.set_xticks(x, [name.replace(" ", "\n").replace("/", "/\n") for name in labels])
    # bars start at zero (red-team review 2026-11-19, m14); headroom keeps the drop labels of the
    # two grids, whose bars reach 1.0, clear of the frame
    a1.set_ylim(0, 1.17)
    a1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a1.set_ylabel("Tuned LightGBM macro-F1")
    a1.set_title("(a) Score under each split", fontsize=LABEL_PT)
    curve = dec["curves"]["A_lgbm_matched"]
    fit = dec["fits"]["A_lgbm"]
    g = np.array([p["gap"] for p in curve], dtype=float)
    f1 = np.array([p["macro_f1"] for p in curve])
    gg = np.linspace(1, g.max(), 200)
    a2.plot(g, f1, "o", ms=3, color=MEANING["choice"])
    a2.plot(
        gg,
        fit["asymptote_fit"] + fit["amplitude_fit"] * np.exp(-(gg - 1) / fit["tau_released_rows"]),
        color=MEANING["choice"],
        lw=1,
    )
    a2.axhline(gv["level_group_mean"], color=MEANING["control"], lw=1, ls="--")
    nearest = int(np.argmin(g))
    if g[nearest] != 1:
        raise ValueError(f"{decay_run_dir}: the curve has no point at distance 1")
    a2.annotate(
        f"{f1[nearest]:.3f}",
        (g[nearest], f1[nearest]),
        xytext=(5, -2),
        textcoords="offset points",
        fontsize=LABEL_PT,
    )
    # the three marks are labeled in place: at this panel width a key would cover the curve
    a2.text(
        0.97,
        0.80,
        f"Exponential fit,\ndecay length {fit['tau_released_rows']:.1f} rows",
        transform=a2.transAxes,
        ha="right",
        va="top",
        fontsize=LABEL_PT,
    )
    a2.annotate(
        "Whole-file level",
        (g.max(), gv["level_group_mean"]),
        xytext=(0, -6),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=LABEL_PT,
        color=MEANING["control"],
    )
    a2.set_ylim(top=f1.max() + 0.02)
    a2.set_ylim(bottom=min(f1.min(), gv["level_group_mean"]) - 0.06)
    a2.set_xlabel("Distance to the nearest training row (rows)")
    a2.set_ylabel("Macro-F1 (MSU/ORNL)")
    a2.set_title("(b) The leak has a length", fontsize=LABEL_PT)

    near = "nn_frac_within_0.1"
    near_row = next(r[near] for r in group["rows"] if r["split"].startswith("row"))
    near_held = [r[near] for r in group["rows"] if r["split"] == "group"]
    arms = [u for name, u in pc["units"].items() if name != "row"]
    if len(arms) != len(near_held):
        raise ValueError(
            f"{per_class_run_dir}: {len(arms)} held-out arms, {group_run_dir.name} has "
            f"{len(near_held)} partitions"
        )
    classes = ("natural", "attack")
    row_c = [near_row] + [pc["units"]["row"]["per_class"][c]["recall"] for c in classes]
    held_all = [near_held] + [[u["per_class"][c]["recall"] for u in arms] for c in classes]
    held_c = [float(np.mean(v)) for v in held_all]
    held_c_sd = [float(np.std(v)) for v in held_all]
    xc = np.arange(len(row_c))
    a3.bar(xc - w / 2, row_c, w, color=MEANING["choice"])
    a3.bar(xc + w / 2, held_c, w, yerr=held_c_sd, capsize=2, color=MEANING["control"])
    for i in range(len(row_c)):
        fmt = "{:.1%}" if i == 0 else "{:.2f}"  # a share of rows in percent, a recall as printed
        a3.annotate(
            fmt.format(row_c[i]), (xc[i] - w / 2, row_c[i] + 0.02), ha="center", fontsize=LABEL_PT
        )
        # the held-out share is printed with its spread: no other ten-page exhibit carries it
        held_label = (
            f"{held_c[i] * 100:.1f} ± {held_c_sd[i] * 100:.1f}%"
            if i == 0
            else fmt.format(held_c[i])
        )
        a3.annotate(
            held_label,
            (xc[i] + w / 2, held_c[i] + held_c_sd[i] + 0.02),
            ha="center",
            fontsize=LABEL_PT,
        )
    a3.set_xticks(xc, ["Near-\ncopies", "Natural\nrecall", "Attack\nrecall"])
    a3.set_xlim(-0.62, len(row_c) - 0.38)  # room for the first value label
    a3.set_ylim(0, 1.17)
    a3.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a3.set_ylabel("Share of rows (MSU/ORNL)")
    a3.set_title("(c) Near-copies, and which class pays", fontsize=LABEL_PT)
    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout()
    # one key for (a) and (c), which share the two split colors; it hangs under the axes, which
    # tight_layout does not count: crop to the drawn artists so the page gets the lines back
    # (exhibit audit 2026-11-20, F3-5)
    handles, names = a1.get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=LABEL_PT,
    )
    fig.savefig(out, metadata=PDF_METADATA, bbox_inches="tight", pad_inches=TIGHT_PAD_IN)
    plt.close(fig)
    return out


FDIA_CLASSICAL_NONLINEAR = ("rbf_svm", "rf", "xgb", "lgbm", "mlp")


def fdia_budget_figure(square_runs: dict, h3_runs: dict, out_name: str) -> Path:
    """FDIA grids: the attack against noise by budget, and the bad-data test (draft v3, Fig. D).

    `square_runs` and `h3_runs` map a panel title to a run directory. Square panels: mean
    macro-F1 over seeds (per_seed.csv) under the Square black-box attack (solid) and random
    sign noise of the same budget (dashed), for the QSVM, the VQC, and the band spanned by the
    five non-linear classical families. Last panel: the chi-square bad-data test's flag rate on
    unconstrained (U) and state-consistent (C) MLP-crafted rows, per grid and budget
    (run_fdia_consistent_attack, ruling 5), against unperturbed rows.
    """
    import pandas as pd
    from matplotlib.patches import Patch

    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n = len(square_runs) + 1
    fig, axes = plt.subplots(1, n, figsize=(FIGSIZE["text"][0], 2.6))
    for ax, (title, run) in zip(axes, square_runs.items(), strict=False):
        p = pd.read_csv(Path(run) / "per_seed.csv")
        m = p.groupby(["family", "condition"])["value"].mean()
        eps = sorted({float(c.split("@")[1]) for c in m.index.get_level_values(1) if "@" in c})
        for kind, ls in (("square", "-"), ("random", "--")):
            band = np.array(
                [[m[(f, f"{kind}@{e:g}")] for f in FDIA_CLASSICAL_NONLINEAR] for e in eps]
            )
            ax.fill_between(
                eps,
                band.min(1),
                band.max(1),
                color=MEANING["classical"],
                alpha=0.22 if kind == "square" else 0.10,
                lw=0,
            )
            ax.plot(eps, band.mean(1), ls=ls, color=MEANING["classical"], lw=0.8)
            for fam in ("qsvm", "vqc"):
                ax.plot(
                    eps,
                    [m[(fam, f"{kind}@{e:g}")] for e in eps],
                    ls=ls,
                    marker=MODEL_MARKERS[fam],
                    ms=4,
                    color=MEANING["quantum"],
                )
        ax.set_xticks(eps)
        ax.set_xlabel("Budget (train standard deviations)")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title, fontsize=LABEL_PT)
    axes[0].set_ylabel("Macro-F1 under attack or noise")
    handles = [
        plt.Line2D(
            [], [], color=MEANING["quantum"], marker=MODEL_MARKERS["qsvm"], ms=4, label="QSVM"
        ),
        plt.Line2D(
            [], [], color=MEANING["quantum"], marker=MODEL_MARKERS["vqc"], ms=4, label="VQC"
        ),
        Patch(color=MEANING["classical"], alpha=0.3, label="Five non-linear classical models"),
        plt.Line2D([], [], color="black", ls="-", label="Square attack"),
        plt.Line2D([], [], color="black", ls="--", label="Random noise, same budget"),
    ]
    ax = axes[-1]
    ax.set_title("(c) Flagged by the bad-data test", fontsize=LABEL_PT)
    legend = fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=LABEL_PT
    )
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    # the bad-data test, as meter rows: the rates do not move with the budget, so a bar per
    # budget drew twelve bars to say two numbers per grid (and hid the near-zero ones). The panel
    # has no axis, so it also takes the band the other panels give to their tick labels.
    box = ax.get_position()
    ax.set_position((box.x0, 0.2, box.width, box.y1 - 0.2))
    width, _ = _points_axes(fig, ax)
    texts, pairs = [], []
    y = METER_PAD_PT
    for title, run in h3_runs.items():
        b = json.loads((Path(run) / "results.json").read_text())["summary"]["bdd_flag_rates"]
        eps = sorted({float(key.split("|")[1]) for key in b})
        texts.append(ax.text(0, y, title, va="top", fontweight="bold"))
        y += METER_LINE_PT * 1.1
        for space, label, shade in (
            ("U", "Ignores the physics", 1.0),
            ("C", "State-consistent", 0.45),
        ):
            rates = [b[f"{space}|{e:g}|mlp"]["flag_rate"] for e in eps]
            lo, hi = min(rates), max(rates)
            # one number when the budget does not move the rate, its range otherwise
            shown = f"{100 * hi:.3g}%" if hi - lo < 5e-4 else f"{100 * lo:.3g}–{100 * hi:.3g}%"
            pair = _meter_row(ax, y, width, label, shown, hi, MEANING["attack"], shade)
            texts += pair
            pairs.append(pair)
            y += METER_ROW_PT
        y += METER_PAD_PT
    _assert_text_fits(fig, ax, texts, pairs, clear_of=[legend])
    out = FIG_DIR / f"{out_name}.pdf"
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


# claim -> (row tag, the finding in plain words, its statistic with the bar for a yes). Tags 1-7
# are the checklist's numbers (experiments/instance_panels.csv); A-D are the other findings. An
# empty statistic: the row is read by a registered rule with no single number. Each printed
# statistic is the quantity its threshold tests (exhibit audit 2026-11-20, finding F7-f): row 2
# the factor by which the class-weighted fit moves the transfer ratio, row 6 the coefficient of
# variation of the QSVM's distortion over restarts. The marks run *, †, ‡, § down the rows. Rows
# 1 to 7 open with the name the checklist figure gives the choice (one name per row in both
# figures, blind read of 2026-11-20), and the rest of the name says what the row's cell
# measures (fix-check of 2026-11-21, F2: row 3 is one model's retention gap, not a ranking;
# row 7 is a random-noise flip ratio, not an attack).
REPLICATION_CLAIMS = {
    "split": ("1", "Split protocol inflates the score", "split gap ≥ 0.05"),
    "surrogate_specification": (
        "2",
        "Surrogate fit sets the transfer ratio",
        "fit moves it ≥ 2×",
    ),
    "attack_availability": (
        "3",
        "Attack availability moves QSVM retention",
        "gap ≥ 0.20*",
    ),
    "attack_objective": ("4", "Attack objective changes the failure", "gap†"),
    "reading_radius": (
        "5",
        "Reading radius moves white-box results",
        "gap ≥ 0.20",
    ),
    "starting_point": ("6", "Starting point moves black-box results", "spread ≥ 0.15‡"),
    "measurement_geometry": (
        "7",
        "Measurement geometry: noise flips the QSVM",
        "flip ratio ≥ 1.5",
    ),
    "accuracy": ("A", "No quantum accuracy advantage", "classical − quantum"),
    "kernel_whitebox_contrast": (
        "B",
        "White-box: QSVM below the RBF-SVM",
        "gap ≤ −0.10§",
    ),
    "per_class_reorder": (
        "C",
        "Retention order is not recall order",
        "Spearman ρ < 0.9",
    ),
    # from a gradient-based surrogate: on the 118-bus grid a tree surrogate's attack still
    # transfers to the other trees (exhibit audit 2026-11-20, finding F7-e)
    "physics_removes_tree_transfer": (
        "D",
        "Physics stops MLP-to-tree transfer",
        "",
    ),
}
REPLICATION_DATASETS = {
    "msu": "MSU/ORNL\n(main study)",
    "gas": "Gas\npipeline",
    "fdia14": "FDIA\n14-bus",
    "fdia118": "FDIA\n118-bus",
}
# reading -> (fill colour, printed word); the control hue: it held elsewhere. The study's own
# column has no fill: in grey it read as a column of "no" (fix-check and blind read of
# 2026-11-21); its head says "(main study)", since every column is this study, and its cells
# print the statistic alone.
REPLICATION_STYLE = {
    "origin": ("#FFFFFF", "found"),
    "yes": (tint(MEANING["control"], 0.45), "yes"),
    "partial": (tint(MEANING["control"], 0.16), "partly"),
    "no": (tint(MEANING["reference"], 0.22), "no"),
    "not_tested": ("#FFFFFF", "–"),
}
REPLICATION_HEAD_PT = 23.0  # two lines of column heads
REPLICATION_ROW_PT = 13.0
REPLICATION_GROUP_GAP_PT = 5.0  # between the checklist rows and the other findings
REPLICATION_STAT_X_PT = 208.0  # where the statistic column starts
REPLICATION_KEY_PT = 15.0  # the key under the rows: one line of swatches
REPLICATION_NOTE_PT = 11.0  # one line of notes
# the key says what each printed word means; the marks say where the gas data's threshold
# differs.
# The numbers repeat the runners' constants; tests/test_replication_matrix.py holds them equal.
REPLICATION_KEY = (
    ("yes", "it reproduces"),
    ("partial", "in part (row 1: a gap of 0.02 to 0.05)"),
    ("no", "it does not"),
    ("not_tested", "no registered check"),
)
REPLICATION_NOTE_LINES = (
    "On the gas data:   * 0.30.   † A rule on black-box rankings.   "
    "‡ Or a change of model order.   § −0.20 with p < 0.05.",
)
REPLICATION_NOTE = "   ".join(REPLICATION_NOTE_LINES)
REPLICATION_CELL_X_PT = 296.0  # where the four data columns start


def replication_cell_text(word: str, label: str, origin: bool = False) -> str:
    """One line per cell: the reading, then its statistic; the study's own column prints the
    statistic alone, or the word when the row has none. A two-line cell overflowed its row and
    printed into the next one (decision log 2026-11-19T14:21Z). A leading hyphen-minus becomes a
    true minus (IEEE: a minus is not a hyphen)."""
    if label.startswith("-"):
        label = "−" + label[1:]
    if origin:
        return label or word
    return f"{word} {label}" if label else word


def replication_matrix_figure(table_csv: Path, out_name: str) -> Path:
    """Which MSU/ORNL findings reproduce on the other data sources.

    Reads `paper/tables/replication_matrix.csv` (experiments.build_replication_matrix): one row
    per claim and data source with its reading and printed statistic. Rows follow
    REPLICATION_CLAIMS (the checklist's numbers, then the lettered findings), columns
    REPLICATION_DATASETS; an unknown or missing claim or dataset fails loud, and so does a text
    that leaves its column.
    """
    import csv

    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with Path(table_csv).open(encoding="utf-8") as f:
        cells = {(r["claim"], r["dataset"]): r for r in csv.DictReader(f)}
    unknown = {c for c, _ in cells} - set(REPLICATION_CLAIMS)
    unknown |= {d for _, d in cells} - set(REPLICATION_DATASETS)
    if unknown:
        raise ValueError(f"replication matrix: no display name for {sorted(unknown)}")
    claims, datasets = list(REPLICATION_CLAIMS), list(REPLICATION_DATASETS)
    lettered = sum(1 for c in claims if not REPLICATION_CLAIMS[c][0].isdigit())
    fig_w_pt = FIGSIZE["text"][0] * 72
    fig_h_pt = (
        REPLICATION_HEAD_PT
        + REPLICATION_ROW_PT * len(claims)
        + (REPLICATION_GROUP_GAP_PT if 0 < lettered < len(claims) else 0.0)
        + REPLICATION_KEY_PT
        + REPLICATION_NOTE_PT * len(REPLICATION_NOTE_LINES)
    )
    fig = plt.figure(figsize=(fig_w_pt / 72, fig_h_pt / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    _points_axes(fig, ax)
    cell_w = (fig_w_pt - REPLICATION_CELL_X_PT) / len(datasets)
    grey = COLORS["neutral"]
    head_y = REPLICATION_HEAD_PT - 4
    heads = [
        ax.text(0, head_y, "Finding on the MSU/ORNL data", va="bottom", color=grey),
        ax.text(REPLICATION_STAT_X_PT, head_y, "Statistic, threshold", va="bottom", color=grey),
    ]
    for j, d in enumerate(datasets):
        ax.text(
            REPLICATION_CELL_X_PT + (j + 0.5) * cell_w,
            head_y,
            REPLICATION_DATASETS[d],
            ha="center",
            va="bottom",
        )
    y, left_of_stat, left_of_cells, inside = REPLICATION_HEAD_PT, [heads[0]], [heads[1]], []
    for i, c in enumerate(claims):
        tag, finding, stat = REPLICATION_CLAIMS[c]
        if i and tag.isdigit() != REPLICATION_CLAIMS[claims[i - 1]][0].isdigit():
            y += REPLICATION_GROUP_GAP_PT
        mid = y + REPLICATION_ROW_PT / 2
        ax.text(0, mid, tag, va="center", fontweight="bold")
        left_of_stat.append(ax.text(11, mid, finding, va="center"))
        left_of_cells.append(ax.text(REPLICATION_STAT_X_PT, mid, stat, va="center", color=grey))
        for j, d in enumerate(datasets):
            if (c, d) not in cells:
                raise ValueError(f"replication matrix: no cell for {c}/{d}")
            r = cells[(c, d)]
            fill, word = REPLICATION_STYLE[r["reproduces"]]
            x = REPLICATION_CELL_X_PT + j * cell_w
            ax.add_patch(
                plt.Rectangle(
                    (x, y), cell_w, REPLICATION_ROW_PT, facecolor=fill, edgecolor="white", lw=1.5
                )
            )
            text = replication_cell_text(word, r["label"], origin=r["reproduces"] == "origin")
            inside.append((ax.text(x + cell_w / 2, mid, text, ha="center", va="center"), cell_w))
        y += REPLICATION_ROW_PT
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72
    # the key: each printed word in its fill, then what it means
    x, key_mid = 0.0, y + REPLICATION_KEY_PT / 2 + 1
    for reading, meaning in REPLICATION_KEY:
        fill, word = REPLICATION_STYLE[reading]
        ax.add_patch(
            plt.Rectangle((x, key_mid - 5), 30, 10, facecolor=fill, edgecolor=grey, lw=0.4)
        )
        ax.text(x + 15, key_mid, word, ha="center", va="center")
        label = ax.text(x + 34, key_mid, meaning, va="center", color=grey)
        x = label.get_window_extent(renderer).x1 / px + 14
    note_mid = y + REPLICATION_KEY_PT + REPLICATION_NOTE_PT / 2
    notes = [
        ax.text(0, note_mid + k * REPLICATION_NOTE_PT, line, va="center", color=grey)
        for k, line in enumerate(REPLICATION_NOTE_LINES)
    ]
    for t in (label, *notes):
        if t.get_window_extent(renderer).x1 > fig_w_pt * px:
            raise ValueError(f"replication matrix: text leaves the figure: {t.get_text()!r}")
    for group, limit in (
        (left_of_stat, REPLICATION_STAT_X_PT),
        (left_of_cells, REPLICATION_CELL_X_PT),
    ):
        for t in group:
            if t.get_window_extent(renderer).x1 > (limit - 3) * px:
                raise ValueError(f"replication matrix: text leaves its column: {t.get_text()!r}")
    for t, width in inside:
        if t.get_window_extent(renderer).width > (width - 3) * px:
            raise ValueError(f"replication matrix: text leaves its cell: {t.get_text()!r}")
    out = FIG_DIR / f"{out_name}.pdf"
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


# Meter rows (points): a panel is its title, its unit, and one row per reading (a label line with
# the value at its right, and a thin bar under it). Vertical bars with tick labels under them
# collided at 8 pt ("Transferred attack" ran into "White-box attack") and stood twice as tall.
METER_LINE_PT = 10.5  # one 8-pt text line
METER_BAR_PT = 4.5
METER_ROW_PT = METER_LINE_PT + METER_BAR_PT + 5.0
METER_PAD_PT = 3.0  # between the unit and the first reading


def _true_minus(label: str) -> str:
    """A leading hyphen-minus as a true minus (IEEE: a minus is not a hyphen)."""
    return "\u2212" + label[1:] if label.startswith("-") else label


def _points_axes(fig, ax) -> tuple[float, float]:
    """Turn `ax` into a blank panel measured in points from its top-left corner; returns its
    width and height in points."""
    box = ax.get_position()
    width = box.width * fig.get_figwidth() * 72
    height = box.height * fig.get_figheight() * 72
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return width, height


def _meter_row(
    ax,
    y: float,
    width: float,
    label: str,
    value_label: str,
    share: float,
    hue: str,
    shade: float = 1.0,
    from_right: bool = False,
) -> tuple:
    """One reading on a points panel: its label with the value at the right, and under them a bar
    filling `share` of a grey track. On a scale that runs below zero (`from_right`) the bar
    starts at the right edge, is hatched, and a tick marks zero there, so a negative reading does
    not look like a full positive one. Returns the two text artists."""
    name = ax.text(0, y, label, va="top")
    value = ax.text(width, y, value_label, va="top", ha="right")
    bar_y = y + METER_LINE_PT
    ax.add_patch(
        plt.Rectangle((0, bar_y), width, METER_BAR_PT, color=tint(MEANING["reference"], 0.14), lw=0)
    )
    length = width * share
    ax.add_patch(
        plt.Rectangle(
            (width - length if from_right else 0, bar_y),
            length,
            METER_BAR_PT,
            facecolor="white" if from_right else tint(hue, shade),
            edgecolor=hue,
            linewidth=OUTLINE_PT,
            hatch="//////" if from_right else None,
        )
    )
    if from_right:
        ax.plot(
            [width, width],
            [bar_y - 1.5, bar_y + METER_BAR_PT + 1.5],
            color=COLORS["neutral"],
            lw=0.8,
            clip_on=False,
        )
    return name, value


def _assert_text_fits(fig, ax, texts: list, pairs: list, clear_of: tuple | list = ()) -> None:
    """Fail loud when a label leaves its panel, a row's label reaches its value, or a label
    overlaps an artist in `clear_of` (a legend): a figure with overlapping text must not be
    written (decision log 2026-11-19T14:21Z and T21:50Z)."""
    renderer = fig.canvas.get_renderer()
    box = ax.get_window_extent(renderer)
    others = [o.get_window_extent(renderer) for o in clear_of]
    for t in texts:
        e = t.get_window_extent(renderer)
        outside = e.x0 < box.x0 - 0.5 or e.x1 > box.x1 + 0.5 or e.y0 < box.y0 - 0.5
        if outside or e.y1 > box.y1 + 0.5 or any(e.overlaps(o) for o in others):
            raise ValueError(f"figure text does not fit its panel: {t.get_text()!r}")
    for left, right in pairs:
        if left.get_window_extent(renderer).x1 > right.get_window_extent(renderer).x0 - 2:
            raise ValueError(
                f"figure label overlaps its value: {left.get_text()!r} / {right.get_text()!r}"
            )


# The checklist figure (points): one row per evaluation choice. A row is a header line (the
# choice in bold, what is read in grey) and two tiles: the uncontrolled choice (operator ruling
# 2026-11-21: only the split has cited studies behind "common") and its reading in the
# choice hue, the control and its reading in the control hue, each with a thin meter under it.
CHECK_KEY_PT = 13.0  # the colour key above the first row
CHECK_HEAD_PT = 10.5  # one 8-pt text line
CHECK_TILE_PT = 16.0  # a text line and the meter under it
CHECK_GAP_PT = 5.5  # between two rows
CHECK_ARROW_PT = 11.0  # between the two tiles
CHECK_PAD_PT = 3.0  # text inset inside a tile
CHECK_METER_PT = 2.4
CHECK_SIDES = {"a": ("choice", "the uncontrolled choice"), "b": ("control", "its control")}


def checklist_reading(parts: dict) -> tuple[str, tuple[float, float] | None, float | None]:
    """One tile's reading from its parts: the printed text, the meter's span on the reading's
    own scale (None without a scale) and a tick inside it (the mean of a mean-and-std reading).
    A reading is one value, a range (lo, hi) or a mean with its std; anything else fails loud."""
    kinds = frozenset(parts)
    first = next(iter(parts.values()))
    suffix = first["suffix"]
    scale = float(first["scale"]) if first["scale"] else None
    txt = {k: _true_minus(r["value_label"]) for k, r in parts.items()}
    val = {k: float(r["value"]) for k, r in parts.items()}
    if kinds == {"value"}:
        text, span, tick = f"{txt['value']}{suffix}", (0.0, val["value"]), None
    elif kinds == {"lo", "hi"}:
        text, span, tick = f"{txt['lo']} to {txt['hi']}{suffix}", (val["lo"], val["hi"]), None
    elif kinds == {"mean", "std"}:
        text = f"{txt['mean']} ± {txt['std']}{suffix}"
        span, tick = (val["mean"] - val["std"], val["mean"] + val["std"]), val["mean"]
    else:
        raise ValueError(f"checklist reading: parts {sorted(kinds)} are not a reading")
    if scale is None:
        return text, None, None
    # a mean-and-std span may pass the ends of the scale and is cut there; a printed value may not
    printed = [val[k] for k in val if k != "std"]
    if min(printed) < 0.0 or max(printed) > scale:
        raise ValueError(f"checklist reading: {text!r} leaves its scale of 0 to {scale:g}")
    clip = lambda x: min(max(x / scale, 0.0), 1.0)  # noqa: E731
    return text, (clip(span[0]), clip(span[1])), None if tick is None else clip(tick)


def checklist_figure(panels_csv: Path, out_name: str) -> Path:
    """The evaluation checklist with its readings, one row per choice (page 1 of both builds).

    Reads `paper/tables/instance_panels.csv` (experiments.build_replication_matrix): per choice,
    side a is the uncontrolled choice and side b its control, each a label and a reading read from a
    run artifact; a `mark` row draws a reference level on both meters. Every tile wears the hue
    of its side (design_tokens.MEANING: choice, control), so the two colours mean the same thing
    in every row. A label that reaches its value, or leaves its row, fails the build.
    """
    import csv

    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with Path(panels_csv).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    choices: dict = {}
    for r in rows:
        entry = choices.setdefault((int(r["order"]), r["panel"], r["unit"]), {"marks": []})
        if r["part"] == "mark":
            entry["marks"].append(r)
        elif r["side"] in CHECK_SIDES:
            entry.setdefault(r["side"], {})[r["part"]] = r
        else:
            raise ValueError(f"checklist figure: side {r['side']!r} is not a reading")
    if any(not {"a", "b"} <= set(v) for v in choices.values()):
        raise ValueError("every checklist row needs a reading 'a' and a reading 'b'")
    keys = sorted(choices)
    row_pt = CHECK_HEAD_PT + CHECK_TILE_PT + CHECK_GAP_PT
    fig_w_pt = COLUMN_WIDTH_PT
    fig_h_pt = CHECK_KEY_PT + row_pt * len(keys) - CHECK_GAP_PT
    fig = plt.figure(figsize=(fig_w_pt / 72, fig_h_pt / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    _points_axes(fig, ax)
    tile_w = (fig_w_pt - CHECK_ARROW_PT) / 2
    x0 = {"a": 0.0, "b": tile_w + CHECK_ARROW_PT}
    texts, pairs = [], []
    # the colour key: one swatch per side, above the tile column it names
    for side, (meaning, words) in CHECK_SIDES.items():
        hue = MEANING[meaning]
        ax.add_patch(
            plt.Rectangle(
                (x0[side] + 0.5, 1.5), 7, 7, facecolor=tint(hue), edgecolor=hue, lw=OUTLINE_PT
            )
        )
        texts.append(ax.text(x0[side] + 10.5, 0.5, words, va="top", color=COLORS["neutral"]))
    y = CHECK_KEY_PT
    for key in keys:
        order, title, unit = key
        head = ax.text(0, y, f"{order} {title}", va="top", fontweight="bold")
        what = ax.text(fig_w_pt, y, unit, va="top", ha="right", color=COLORS["neutral"])
        texts += [head, what]
        pairs.append((head, what))
        top = y + CHECK_HEAD_PT
        for side, (meaning, _) in CHECK_SIDES.items():
            hue = MEANING[meaning]
            parts = choices[key][side]
            text, span, tick = checklist_reading(parts)
            ax.add_patch(
                plt.Rectangle(
                    (x0[side] + 0.5, top),
                    tile_w - 1.0,
                    CHECK_TILE_PT,
                    facecolor=tint(hue),
                    edgecolor=hue,
                    lw=OUTLINE_PT,
                )
            )
            name = ax.text(
                x0[side] + CHECK_PAD_PT, top + 1.2, next(iter(parts.values()))["label"], va="top"
            )
            value = ax.text(
                x0[side] + tile_w - CHECK_PAD_PT,
                top + 1.2,
                text,
                va="top",
                ha="right",
                fontweight="bold",
            )
            texts += [name, value]
            pairs.append((name, value))
            if span is None:
                continue
            left, inner = x0[side] + CHECK_PAD_PT, tile_w - 2 * CHECK_PAD_PT
            bar_y = top + CHECK_TILE_PT - CHECK_METER_PT - 1.6
            ax.add_patch(plt.Rectangle((left, bar_y), inner, CHECK_METER_PT, color="white", lw=0))
            ax.add_patch(
                plt.Rectangle(
                    (left + inner * span[0], bar_y),
                    max(inner * (span[1] - span[0]), 0.6),
                    CHECK_METER_PT,
                    color=hue,
                    lw=0,
                )
            )
            ticks = [(tick, "black")] if tick is not None else []
            scale = float(next(iter(parts.values()))["scale"])
            for m in choices[key]["marks"]:
                if not 0.0 <= float(m["value"]) <= scale:
                    raise ValueError(
                        f"checklist figure: mark {m['label']!r} leaves its scale of 0 to {scale:g}"
                    )
            ticks += [(float(m["value"]) / scale, COLORS["neutral"]) for m in choices[key]["marks"]]
            for at, colour in ticks:
                ax.plot(
                    [left + inner * at] * 2,
                    [bar_y - 1.2, bar_y + CHECK_METER_PT + 1.2],
                    color=colour,
                    lw=0.8,
                )
        ax.annotate(
            "",
            xy=(x0["b"] - 0.5, top + CHECK_TILE_PT / 2),
            xytext=(tile_w + 0.5, top + CHECK_TILE_PT / 2),
            arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 0.6, "mutation_scale": 6},
        )
        y += row_pt
    _assert_text_fits(fig, ax, texts, pairs)
    out = FIG_DIR / f"{out_name}.pdf"
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


# the full-feature MLP is left out: it is the surrogate the attack is crafted on, and no finite
# threshold gives it a 1% validation false-positive rate (exhibit audit 2026-11-20, F8-a)
CONSOLE_MODELS = ("lgbm", "xgb", "logreg", "rf")
# a console verdict from the attacked operating point (the paper's words for the two failures):
# flooded = most normal rows raise an alarm; silent = almost no attack raises one
CONSOLE_FLOOD_FPR = 0.5
CONSOLE_SILENT_TPR = 0.05
CONSOLE_MIN_ARROW = 0.03  # a shorter move draws no arrow (its head would hide the dots)
CONSOLE_INNER_MS = 2.6  # the attacked dot of an unmoved reading, drawn inside the clean ring
# four rows: at 1.8 in the figure cost the ten-page build's page 8 a column break that left about
# eight lines of white space (decision log 2026-11-20, the fourth pass of the final sweep)
CONSOLE_HEIGHT_IN = 1.55


def console_verdict(adv_tpr: float, adv_fpr: float) -> str:
    """'flooded', 'silent' or '' for an attacked operating point."""
    if adv_fpr > CONSOLE_FLOOD_FPR:
        return "flooded"
    if adv_tpr < CONSOLE_SILENT_TPR:
        return "silent"
    return ""


def console_figure(run_dir: Path, out_name: str, budget: str = "0.01") -> Path:
    """What a transferred attack does to an operator's console (Discussion; ruling 16:18Z).

    Reads the operating-point probe (experiments/run_operating_point_full_features.py): the
    full-feature models at a validation-fitted threshold for a `budget` false-positive rate,
    clean and under transferred PGD at 0.5 standard deviations, mean over the run's seeds. Each
    model is a row and each panel a rate: the share of attack rows flagged (detection) and the
    share of natural rows flagged (false alarms). An arrow in the attack hue runs from the clean
    reading (hollow, classical hue) to the attacked one (filled), and the attacked reading is
    named by `console_verdict`.
    """
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = json.loads((Path(run_dir) / "results.json").read_text())
    s = res["summary"]
    missing = [m for m in CONSOLE_MODELS if m not in s]
    if missing:
        raise KeyError(f"{run_dir}: operating-point summary has no models {missing}")
    seeds = res["per_seed"]
    for per in seeds.values() if isinstance(seeds, dict) else seeds:
        for m in CONSOLE_MODELS:
            threshold = per["op"][m][budget]["clean"]["threshold"]
            if not np.isfinite(threshold):
                raise ValueError(
                    f"{run_dir}: {m} has no finite threshold at a {budget} false-positive "
                    "rate, so it has no operating point to draw"
                )
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE["col"][0], CONSOLE_HEIGHT_IN), sharey=True)
    ys = np.arange(len(CONSOLE_MODELS))[::-1]
    for y, m in zip(ys, CONSOLE_MODELS, strict=True):
        r = {k: s[m][f"{k}@{budget}_clean"] for k in ("tpr", "fpr")}
        a = {k: s[m][f"{k}@{budget}_adv"] for k in ("tpr", "fpr")}
        if r["tpr"] < CONSOLE_SILENT_TPR:
            raise ValueError(
                f"{run_dir}: {m} catches {r['tpr']:.3f} of attacks clean at the {budget} budget, "
                "so a silent reading under attack would not be the attack's doing"
            )
        verdict = console_verdict(a["tpr"], a["fpr"])
        for ax, k in zip(axes, ("tpr", "fpr"), strict=True):
            if abs(a[k] - r[k]) >= CONSOLE_MIN_ARROW:  # a shorter move: the dots say it
                ax.annotate(
                    "",
                    xy=(a[k], y),
                    xytext=(r[k], y),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": MEANING["attack"],
                        "lw": 1.0,
                        "shrinkA": 3.5,
                        "shrinkB": 3.5,
                    },
                )
            ax.plot(r[k], y, "o", ms=5.5, mfc="white", mec=MEANING["classical"], mew=1.2)
            # where the attack does not move the reading, the attacked dot sits inside the clean
            # ring, so neither hides the other (exhibit audit 2026-11-20, finding F8-b)
            same = abs(a[k] - r[k]) < CONSOLE_MIN_ARROW
            ax.plot(a[k], y, "o", ms=CONSOLE_INNER_MS if same else 5.5, color=MEANING["attack"])
        # the verdict, in words, right of the panels (the paper's names for the two failures)
        axes[1].text(1.12, y, verdict, ha="left", va="center", fontstyle="italic")
    axes[0].set_yticks(ys, [display_model(m) for m in CONSOLE_MODELS])
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_xlabel("Attacks\ncaught")
    axes[1].set_xlabel("Normal rows\nflagged")
    for ax in axes:
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_ylim(-0.6, len(CONSOLE_MODELS) - 0.4)
        ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([], [], ls="none", marker="o", ms=5.5, mfc="white", mec=MEANING["classical"]),
        plt.Line2D([], [], ls="none", marker="o", ms=5.5, color=MEANING["attack"]),
    ]
    fig.legend(
        handles,
        ["clean", "attacked"],
        loc="upper center",
        ncol=2,
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.8,
    )
    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout(rect=(0, 0, 0.92, 0.9), w_pad=2.2)
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


# The robustness table as a picture (Results; the TSG build prints it in place of the table).
# Panel geometry in points from the left edge: (a) after room for the longest model name, then
# the two retention columns (right-aligned at CONTROLS_RETENTION).
CONTROLS_HEIGHT_PT = 155.0
CONTROLS_TITLE_PT = 24.0  # two lines: the panel titles, and the heads of the retention columns
CONTROLS_FOOT_PT = 38.0  # tick labels and a two-line axis label
CONTROLS_KEY_PT = 13.0  # the marker key, under the axis labels
CONTROLS_F1 = (88.0, 100.0)
CONTROLS_RETENTION = (220.0, 252.0)
CONTROLS_HSJ = (264.0, 48.0)
CONTROLS_FLIP = (324.0, 48.0)
CONTROLS_CAUGHT = (396.0, 54.0)
CONTROLS_ALARM = (458.0, 54.0)
# without panel (d), panels (b) and (c) take its room, after a third column of (a): the score
# under the shared attack less the stratified-random floor (right-aligned at CONTROLS_FLOOR_GAP),
# so a high retention that is a score at chance reads as one (seventh blind read, item M1)
CONTROLS_FLOOR_GAP = 298.0
CONTROLS_HSJ_WIDE = (312.0, 82.0)
CONTROLS_FLIP_WIDE = (422.0, 82.0)
# the robustness table's names for the two MLP fits
CONTROLS_NAMES = {"mlp": "Unweighted MLP", "mlp_weighted": "Class-weighted MLP"}
CONTROLS_BAND_TINT = 0.14  # the row bands: the reference grey, light enough to sit under markers
# the shared PGD attack is itself white-box for these (configs/adversarial.yaml,
# white_box_models); every other row receives it by transfer from the unweighted MLP
CONTROLS_SHARED_IS_WHITE_BOX = ("mlp", "vqc")
CONTROLS_LABELLED = {"hsj": ("qsvm",), "flip": ("qsvm", "rbf_svm")}
# the per-class Square run's families; its VQC cell predates the depth-6 refit (decision log
# 2026-11-16T13:33Z) and the class-weighted MLP was not in it, so those two rows stay empty
CONTROLS_SQUARE_MODELS = ("qsvm", "logreg", "rbf_svm", "mlp", "rf", "lgbm", "xgb")
CONTROLS_MIN_ARROW = 0.08  # a move under two marker widths: the arrowhead would hide both marks


def _csv_rows(path: Path) -> list[dict]:
    import csv

    with Path(path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _stratified_floor(rob: list[dict], source: Path) -> float:
    """The stratified-random floor of the robustness table: every row's attacked macro-F1 less
    its distance to the floor. The rows must agree."""
    floors = {round(float(r["adv_f1"]) - float(r["adv_f1_minus_floor"]), 3) for r in rob}
    if len(floors) != 1:
        raise ValueError(f"{source}: rows disagree on the floor: {sorted(floors)}")
    return floors.pop()


def _attack_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    """An arrow in the attack hue from a clean reading to its attacked one."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": MEANING["attack"],
            "lw": 1.0,
            "shrinkA": 3,
            "shrinkB": 3,
        },
    )


def controls_figure(
    robustness_csv: Path, square_run: Path, out_name: str, with_catch: bool = True
) -> Path:
    """The robustness table as a picture, read with its controls: one row per model.

    Rows are the robustness table's (`robustness_summary_table`'s CSV) in its order: retention
    under the shared protocol, highest first.
    (a) Macro-F1 clean (hollow, the family's hue) and attacked (the attack hue): a circle under
    PGD transferred from the unweighted MLP, a diamond under the model's own white-box attack.
    The dashed line is the stratified-random floor, so a retention that is a score at chance
    shows as one. Beside the panel, the two retentions the text argues from, as the table prints
    them: under PGD, and under the model's own white-box attack (a dash where none was run).
    The three-panel form adds a third column, the lowest macro-F1 over the attacks run on
    the model and the attack that reached it (the robustness table's `lowest_f1`).
    (b) HopSkipJump's distortion, mean and standard deviation over restarts, in the family's
    hue, and (c) the share of predictions flipped by random noise of the table's radius, in the
    control hue: the attack against its control. Both use one marker shape, so no model's mark
    repeats a shape of the key.
    (d) Retention is not catch (operator ruling 2026-11-16 step 3(c)): the share of attack rows
    and of natural rows each detector flags, clean and under the Square search (a square), from
    the per-class Square run (experiments/run_evasion_square_per_class.py), mean over its seeds.
    `with_catch=False` draws panels (a) to (c) only, with (b) and (c) wider: the form the
    ten-page build prints.
    """
    from matplotlib.transforms import blended_transform_factory

    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_w, fig_h = FIGSIZE["text"][0] * 72, CONTROLS_HEIGHT_PT
    fig = plt.figure(figsize=(fig_w / 72, fig_h / 72))
    foot = CONTROLS_FOOT_PT + CONTROLS_KEY_PT
    plot_h = fig_h - CONTROLS_TITLE_PT - foot
    attack = MEANING["attack"]

    def head(x_pt: float, text: str, line: int = 0, ha: str = "left", **kwargs) -> None:
        y = 1 - (1 + line * METER_LINE_PT) / fig_h
        fig.text(x_pt / fig_w, y, text, va="top", ha=ha, **kwargs)

    def panel(geometry: tuple[float, float], **kwargs):
        x0, width = geometry
        return fig.add_axes((x0 / fig_w, foot / fig_h, width / fig_w, plot_h / fig_h), **kwargs)

    rob = _csv_rows(robustness_csv)[::-1]  # the table's first row on top
    if not any(r["model"] == "qsvm" for r in rob) or len(rob) < 2:
        raise KeyError(f"{robustness_csv}: needs a QSVM row and one more")
    floor = _stratified_floor(rob, robustness_csv)
    square = json.loads((Path(square_run) / "results.json").read_text())["summary"]["square"]
    missing = [m for m in CONTROLS_SQUARE_MODELS if m not in square]
    if missing:
        raise KeyError(f"{square_run}: per-class Square summary has no models {missing}")

    hsj_at = CONTROLS_HSJ if with_catch else CONTROLS_HSJ_WIDE
    flip_at = CONTROLS_FLIP if with_catch else CONTROLS_FLIP_WIDE
    ax_a = panel(CONTROLS_F1)
    ax_h = panel(hsj_at, sharey=ax_a)
    ax_f = panel(flip_at, sharey=ax_a)
    axes = [ax_a, ax_h, ax_f]

    # ---- (a) macro-F1: clean, under the shared attack, under the model's own attack -------
    head(0.0, "(a) Macro-F1, clean and attacked")
    head(CONTROLS_RETENTION[1], "Retention", ha="right")
    head(CONTROLS_RETENTION[0], "PGD", line=1, ha="right", color=COLORS["neutral"])
    head(CONTROLS_RETENTION[1], "own", line=1, ha="right", color=COLORS["neutral"])
    if not with_catch:
        head(CONTROLS_FLOOR_GAP, "Lowest F1", ha="right")
        head(CONTROLS_FLOOR_GAP, "(attack)", line=1, ha="right", color=COLORS["neutral"])
    ax_a.axvline(floor, color=MEANING["reference"], ls="--", lw=0.9)
    in_row = blended_transform_factory(fig.transFigure, ax_a.transData)
    # a light band behind every second row, across all panels: the eye follows a model from its
    # name through every reading (the panels share the rows but only the first names them)
    for y in range(0, len(rob), 2):
        fig.add_artist(
            plt.Rectangle(
                (0.0, y - 0.5),
                1.0,
                1.0,
                transform=in_row,
                facecolor=tint(MEANING["reference"], CONTROLS_BAND_TINT),
                edgecolor="none",
                zorder=-1,
            )
        )
    for y, r in enumerate(rob):
        m = r["model"]
        hue = family_color(m)
        clean, shared = float(r["clean_f1"]), float(r["adv_f1"])
        own = float(r["whitebox_retention"]) * clean if r["whitebox_retention"] else None
        shared_is_own = m in CONTROLS_SHARED_IS_WHITE_BOX
        if shared_is_own and (own is None or abs(own - shared) > 5e-4):
            raise ValueError(f"{robustness_csv}: {m}'s shared attack should be its white-box one")
        own_text = f"{float(r['whitebox_retention']):.3f}" if r["whitebox_retention"] else "–"
        for x_pt, text in zip(
            CONTROLS_RETENTION, (f"{float(r['retention']):.3f}", own_text), strict=True
        ):
            ax_a.text(x_pt / fig_w, y, text, transform=in_row, ha="right", va="center")
        if not with_catch:
            # the bottom line: the lowest macro-F1 over the attacks run on the model, and
            # which attack (P shared PGD, W own white-box, S Square search, E evolution
            # strategies; the last three at their worst tested radius), as the
            # robustness table prints it
            if not r.get("lowest_f1") or r.get("lowest_attack") not in ("P", "W", "S", "E"):
                raise ValueError(
                    f"{robustness_csv}: no lowest_f1 and lowest_attack for {m}; "
                    "regenerate the robustness table"
                )
            low = f"{float(r['lowest_f1']):.3f} {r['lowest_attack']}"
            ax_a.text(CONTROLS_FLOOR_GAP / fig_w, y, low, transform=in_row, ha="right", va="center")
        _attack_arrow(ax_a, (clean, y), (shared, y))
        ax_a.plot(clean, y, "o", ms=5, mfc="white", mec=hue, mew=1.1)
        if shared_is_own:
            ax_a.plot(shared, y, "D", ms=4.5, color=attack)
            continue
        if own is not None:  # the transferred reading, then the model's own attack
            ax_a.plot([shared, own], [y, y], color=attack, lw=0.8, ls=":")
            ax_a.plot(own, y, "D", ms=4.5, color=attack)
        ax_a.plot(shared, y, "o", ms=5, color=attack)
    names = [CONTROLS_NAMES.get(r["model"]) or display_model(r["model"]) for r in rob]
    ax_a.set_yticks(range(len(rob)), names)
    ax_a.set_xlim(-0.04, 0.72)
    ax_a.set_xticks([0, 0.2, 0.4, 0.6])
    ax_a.set_xlabel("Macro-F1")
    key = [
        plt.Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=COLORS["neutral"]),
        plt.Line2D([], [], ls="none", marker="o", ms=5, color=attack),
        plt.Line2D([], [], ls="none", marker="D", ms=4.5, color=attack),
        plt.Line2D([], [], ls="none", marker="s", ms=4.5, color=attack),
    ]
    words = ["clean", "shared PGD", "own white-box attack", "Square search"]
    shown = len(key) if with_catch else len(key) - 1  # the square belongs to panel (d)
    fig.legend(
        key[:shown],
        words[:shown],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=shown,
        frameon=False,
        borderpad=0,
        borderaxespad=0,
        handletextpad=0.1,
        columnspacing=1.2,
    )

    # ---- (b), (c) HopSkipJump against the random-noise control ----------------------------
    # side by side in the five-panel form, full titles would collide, so the panels are
    # named under their axes there; the three-panel form has room for both
    if with_catch:
        head(hsj_at[0], "(b)")
        head(flip_at[0], "(c)")
        ax_h.set_xlabel("HopSkipJump\n($\\sigma$)")
        ax_f.set_xlabel("Random\nflips (%)")
    else:
        head(hsj_at[0], "(b) Black-box attack")
        head(flip_at[0], "(c) Its noise control")
        ax_h.set_xlabel("HopSkipJump distortion\n($\\sigma$)")
        ax_f.set_xlabel("Predictions flipped by\nrandom noise (%)")
    measured = [r for r in rob if r["hsj_median_linf_sigma"] and r["random_flip_rate"]]
    control = MEANING["control"]
    for y, r in enumerate(rob):
        if r not in measured:
            continue
        m = r["model"]
        hue = family_color(m)
        dist, spread = float(r["hsj_median_linf_sigma"]), float(r["hsj_restart_std"])
        flip = 100 * float(r["random_flip_rate"])
        # hollow, with the share printed: the attack found a starting point on only that share
        # of the points, and the distortion is over those (decision log 2026-11-20T02:03Z)
        share = float(r["hsj_found_share"]) if r.get("hsj_found_share") else 1.0
        partial = share < 1
        ax_h.plot([0, dist], [y, y], color=hue, lw=1.2, solid_capstyle="butt")
        ax_h.errorbar(
            dist,
            y,
            xerr=spread,
            fmt="s" if partial else "o",
            color=hue,
            mfc="white" if partial else hue,
            ms=4,
            capsize=1.5,
            elinewidth=0.8,
        )
        if partial:
            note = f"on {share * 100:.0f}%" if with_catch else f"on {share * 100:.0f}% of points"
            ax_h.text(dist + spread + 0.7, y, note, ha="left", va="center")
        ax_f.plot([0, flip], [y, y], color=control, lw=1.2, solid_capstyle="butt")
        ax_f.plot(flip, y, "o", color=control, ms=4)
        if m in CONTROLS_LABELLED["hsj"]:
            ax_h.text(dist + spread + 0.3, y, f"{dist:.2f}", ha="left", va="center")
        if m in CONTROLS_LABELLED["flip"]:
            ax_f.text(flip + 5, y, f"{flip:.0f}%", ha="left", va="center")
    reach = max(float(r["hsj_median_linf_sigma"]) + float(r["hsj_restart_std"]) for r in measured)
    top_flip = 100 * max(float(r["random_flip_rate"]) for r in measured)
    # a mark at zero stays inside the frame, and the top row's value has room after its mark
    ax_h.set_xlim(-0.06 * reach, reach * 2.0)
    ax_h.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    ax_f.set_xlim(-0.06 * top_flip, top_flip * 1.9)
    ax_f.xaxis.set_major_locator(MaxNLocator(3, integer=True))

    # ---- (d) what each detector still flags under the Square search ----------------------
    if with_catch:
        ax_c = panel(CONTROLS_CAUGHT, sharey=ax_a)
        ax_n = panel(CONTROLS_ALARM, sharey=ax_a)
        axes += [ax_c, ax_n]
        head(CONTROLS_CAUGHT[0], "(d) Share of rows flagged")
        ax_c.set_xlabel("Attack rows\n(caught)")
        ax_n.set_xlabel("Normal rows\n(false alarms)")
    for y, r in enumerate(rob):
        m = r["model"]
        if not with_catch or m not in CONTROLS_SQUARE_MODELS:
            continue
        for ax, k in ((ax_c, "attack_recall"), (ax_n, "false_alarm")):
            clean, attacked = square[m][f"clean_{k}"], square[m][f"square_{k}"]
            if abs(attacked - clean) >= CONTROLS_MIN_ARROW:
                _attack_arrow(ax, (clean, y), (attacked, y))
            ax.plot(clean, y, "o", ms=5, mfc="white", mec=family_color(m), mew=1.1)
            ax.plot(attacked, y, "s", ms=4.5, color=attack)
    for ax in axes[3:]:
        ax.set_xlim(-0.09, 1.09)  # a mark at 0 or 1 stays inside the frame
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    for a in axes:
        a.set_ylim(-0.6, len(rob) - 0.4)
        a.grid(axis="y", visible=False)
        a.tick_params(axis="y", length=0, labelleft=a is ax_a)

    # the row bands sit behind the panels, so every panel is see-through, the last ones included
    for ax in fig.axes:
        ax.set_facecolor("none")
    out = FIG_DIR / f"{out_name}.pdf"
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


def width_sweep_figure(lengthscale_run: Path, out_name: str) -> Path:
    """One classical kernel walked through both end states by its width alone (Appendix).

    Reads the kernel-width sweep (run_whitebox_qsvm.py, `p8a_lengthscale_control`): one RBF-SVM
    at each multiple of its tuned gamma, clean (hollow) and under the full-budget white-box
    attack (filled), against the constant-predictor floor (dashed). The two words are
    `tables._regime`'s labels, from its fixed thresholds.
    """
    from qgridbench.eval.tables import _regime

    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ls = json.loads((Path(lengthscale_run) / "results.json").read_text())["p8a_lengthscale_control"]
    floor = ls["constant_predictor_floor"]
    sweep = []
    for mult, blk in ls["per_width"].items():
        curve = blk["curve"]
        radius = max(curve, key=float)
        last = curve[radius]
        sweep.append(
            (
                float(mult),
                curve["0.0"]["macro_f1"],
                last["macro_f1"],
                _regime(last["kernel_mass_ratio"], last["f1_minus_floor"]),
                radius,
            )
        )
    if len({s[4] for s in sweep}) != 1:
        raise ValueError(f"{lengthscale_run}: widths were attacked at different radii")
    sweep.sort()
    fig, ax = plt.subplots(figsize=(FIGSIZE["col"][0], 1.7))
    ax.axhline(floor, color=MEANING["reference"], ls="--", lw=0.9)
    for mult, clean, attacked, _, _ in sweep:
        _attack_arrow(ax, (mult, clean), (mult, attacked))
        ax.plot(mult, clean, "D", ms=4.5, mfc="white", mec=MEANING["classical"], mew=1.1)
        ax.plot(mult, attacked, "D", ms=4.5, color=MEANING["attack"])
    # each end state is named once: "inverted" in the widest gap between its widths (the arrows
    # are vertical, so a gap is free), "disabled" under the floor line its points sit on
    for word in ("inverted", "disabled"):
        xs = sorted(s[0] for s in sweep if s[3] == word)
        if not xs:
            continue
        if word == "inverted" and len(xs) > 1:
            lo, hi = max(zip(xs, xs[1:], strict=False), key=lambda p: p[1] / p[0])
            ax.text(math.sqrt(lo * hi), 0.24, word, ha="center", fontstyle="italic")
        else:
            ax.text(math.sqrt(xs[0] * xs[-1]), floor - 0.11, word, ha="center", fontstyle="italic")
    ax.set_xscale("log")
    ax.set_xlim(sweep[0][0] / 1.8, sweep[-1][0] * 1.8)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.06, 0.72)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("RBF $\\gamma$ over its tuned value (smaller: a wider kernel)")
    ax.grid(axis="x", visible=False)
    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout()
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


# Newer tabular models under both splits: (row label, run role, key of the row-level score, key
# of the score with whole files held out). The last row is the capacity rung the run itself
# matched to the FT-Transformer's row-level score (`verdict/matched/rung`).
TABULAR_ROWS = (
    ("LightGBM, tuned", "lgbm", "verdict/level_row", "verdict/level_group_mean"),
    ("TabM", "tabm", "verdict/row_mean", "verdict/group_mean"),
    ("FT-Transformer, defaults", "ftt", "verdict/row_mean", "verdict/group_mean"),
    ("LightGBM, 2000-row cap", "tabicl", "verdict/row_lgbm", "verdict/group_lgbm_mean"),
    ("TabICLv2, 2000-row cap", "tabicl", "verdict/row_tabicl", "verdict/group_tabicl_mean"),
)
TABULAR_MATCHED = ("LightGBM, regularized", "capacity")


def _at(node, key: str, source: Path):
    for part in key.split("/"):
        if part not in node:
            raise KeyError(f"{source}: no key {part!r} on the path {key!r}")
        node = node[part]
    return node


def tabular_split_figure(runs: dict[str, Path], out_name: str) -> Path:
    """Every tabular model lands in one band once whole files are held out (Results).

    `runs` maps a role to its run folder: the group-split control (`lgbm`), the TabICLv2 arm
    (`tabicl`), the FT-Transformer and TabM arms (`ftt`, `tabm`) and the capacity-matched control
    (`capacity`). Per model: the row-level score (hollow, the choice hue) and the score with whole
    files held out (filled, the control hue), joined by a rule, with the gap between them printed
    at the right. Rows run in order of the row-level score.
    """
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for role in {r[1] for r in TABULAR_ROWS} | {TABULAR_MATCHED[1]}:
        if role not in runs:
            raise KeyError(f"tabular_split_figure: no run for {role!r}")
        data[role] = json.loads((Path(runs[role]) / "results.json").read_text())
    rows = [
        (label, float(_at(data[role], a, runs[role])), float(_at(data[role], b, runs[role])))
        for label, role, a, b in TABULAR_ROWS
    ]
    capacity = data[TABULAR_MATCHED[1]]
    rung_name = _at(capacity, "verdict/matched/rung", runs[TABULAR_MATCHED[1]])
    rungs = [r for r in capacity["rungs"] if r["name"] == rung_name]
    if len(rungs) != 1:
        raise KeyError(f"{runs[TABULAR_MATCHED[1]]}: {len(rungs)} rungs named {rung_name!r}")
    rows.append((TABULAR_MATCHED[0], float(rungs[0]["row_mean"]), float(rungs[0]["group_mean"])))
    rows.sort(key=lambda r: r[1])  # the highest row-level score on top
    drawn = [score for row in rows for score in row[1:]]
    if min(drawn) <= TABULAR_XLIM[0] or max(drawn) > 1.0:
        raise ValueError(f"tabular_split_figure: a score lies outside the axis {TABULAR_XLIM}")

    fig, ax = plt.subplots(figsize=(FIGSIZE["col"][0], 1.75))
    for y, (_, row_level, held_out) in enumerate(rows):
        ax.plot([held_out, row_level], [y, y], color=MEANING["reference"], lw=1.0, zorder=1)
        ax.plot(row_level, y, "o", ms=5, mfc="white", mec=MEANING["choice"], mew=1.3, zorder=2)
        ax.plot(held_out, y, "o", ms=5, color=MEANING["control"], zorder=2)
        ax.text(1.065, y, f"{row_level - held_out:.3f}", ha="right", va="center")
    ax.text(1.065, len(rows) - 0.5, "gap", ha="right", va="bottom", color=COLORS["neutral"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) + 0.45)
    ax.set_xlim(*TABULAR_XLIM)  # room for the gap column after the highest score
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("Macro-F1")
    ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=MEANING["choice"]),
        plt.Line2D([], [], ls="none", marker="o", ms=5, color=MEANING["control"]),
    ]
    fig.legend(
        handles,
        ["row-level split", "whole files held out"],
        loc="upper center",
        ncol=2,
        frameon=False,
        handletextpad=0.2,
        columnspacing=1.0,
        borderaxespad=0,
    )
    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


# the two full-feature models the text names beside the matched contest
GAS_FULL_MODELS = ("rf", "lgbm")
# Fixed axes of the two dot charts. A reading outside one would be clipped without a word, so
# both figures refuse to draw it (red-team review 2026-11-20, RT-n8).
GAS_XLIM = (0.5, 0.9)
TABULAR_XLIM = (0.5, 1.07)


def gas_pressure_figure(before_run: Path, after_run: Path, out_name: str) -> Path:
    """What the gas pressure feature is worth once the pipeline can read it (Results).

    Reads two accuracy verdicts of experiments/run_gas_accuracy_verdict.py: the main-study pipeline
    (`before_run`) and the same pipeline with pressure passed through a fixed signed logarithm
    and every model re-tuned (`after_run`). Per matched model (PCA-8, 2000-row cap; ten-seed
    means): the score before (hollow) and after (filled) in the family's hue, rows in order of
    the score after. Under a rule, the two full-feature models the text names.
    """
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    before = json.loads((Path(before_run) / "results.json").read_text())
    after = json.loads((Path(after_run) / "results.json").read_text())
    matched = sorted(after["pca8"]["mean"], key=lambda m: after["pca8"]["mean"][m])
    missing = [m for m in matched if m not in before["pca8"]["mean"]]
    missing += [
        m for m in GAS_FULL_MODELS if m not in before["full_mean"] or m not in after["full_mean"]
    ]
    if missing:
        raise KeyError(f"gas_pressure_figure: no reading for {missing} in one of the two runs")
    full = sorted(GAS_FULL_MODELS, key=lambda m: after["full_mean"][m])
    rows = [(f"{display_model(m)}, all features", m, "full_mean") for m in full]
    rows += [(display_model(m), m, "pca8") for m in matched]

    def reading(run: dict, m: str, block: str) -> float:
        return float(run[block][m] if block == "full_mean" else run[block]["mean"][m])

    pairs = [(reading(before, m, block), reading(after, m, block)) for _, m, block in rows]
    drawn = [score for pair in pairs for score in pair]
    if min(drawn) <= GAS_XLIM[0] or max(drawn) >= GAS_XLIM[1]:
        raise ValueError(f"gas_pressure_figure: a score lies outside the axis {GAS_XLIM}")

    fig, ax = plt.subplots(figsize=(FIGSIZE["col"][0], 2.35))
    for y, ((_, m, _), (a, b)) in enumerate(zip(rows, pairs, strict=True)):
        hue = family_color(m)
        ax.plot([a, b], [y, y], color=MEANING["reference"], lw=1.0, zorder=1)
        ax.plot(a, y, "o", ms=5, mfc="white", mec=hue, mew=1.2, zorder=2)
        ax.plot(b, y, "o", ms=5, color=hue, zorder=2)
    ax.axhline(len(full) - 0.5, color=COLORS["neutral"], lw=0.6)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(*GAS_XLIM)
    ax.set_xlabel("Macro-F1 (ten-seed mean)")
    ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=COLORS["neutral"]),
        plt.Line2D([], [], ls="none", marker="o", ms=5, color=COLORS["neutral"]),
    ]
    fig.legend(
        handles,
        ["main-study pipeline", "pressure log-scaled, re-tuned"],
        loc="upper center",
        ncol=2,
        frameon=False,
        handletextpad=0.2,
        columnspacing=1.0,
        borderaxespad=0,
    )
    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


BRIDGE_DIMS = 16  # the dimensionality the cap-cost reading of Results (RQ1) is stated at
BRIDGE_COST_MODEL = "lgbm"  # the model the three steps are measured on
BRIDGE_XLIM = (0.47, 0.93)
BRIDGE_BAR = 0.46  # bar height in row units
BRIDGE_WIDE = 0.15  # a cost label needs a bar this wide to sit inside it, clear of the marker


def accuracy_bridge_figure(matched_csv: Path, cap_run: Path, out_name: str) -> Path:
    """Where the accuracy goes between the full-feature setting and the matched contest (RQ1).

    A bridge chart, top to bottom. Four rows are one model at one seed, from the
    `fixed_hyperparameters` block of experiments/run_kernel_cap_cost.py, and each step changes
    one thing: all features under the full-feature tuned set; all features under the matched
    regime's tuned set; PCA to BRIDGE_DIMS components; the 2000-row cap. Then the best classical
    and the best quantum model of the matched contest at that dimensionality (`matched_csv`, ten
    seeds). A grey bar is what a step costs and a marker is the score it leaves, in the family's
    hue; the scores are printed at the right. Row names are one line each: two-line names collide
    at this height.
    """
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    matched = {r["model"]: float(r["macro_f1"]) for r in _csv_rows(matched_csv)}
    probe = json.loads((Path(cap_run) / "results.json").read_text())
    try:
        own = probe["fixed_hyperparameters"]["full_features"][BRIDGE_COST_MODEL]
        regime = probe["fixed_hyperparameters"]["matched_regime"][BRIDGE_COST_MODEL]
        tuned, retuned = own["all_features"], regime["all_features"]
        reduced = regime[f"pca{BRIDGE_DIMS}"]["full_train"]
        capped = regime[f"pca{BRIDGE_DIMS}"]["capped"]
        legacy = probe[f"pca{BRIDGE_DIMS}_{BRIDGE_COST_MODEL}"]
    except KeyError as e:
        raise KeyError(
            f"accuracy_bridge_figure: {cap_run} has no fixed-hyperparameter path: {e}"
        ) from e
    # the path's regime readings repeat the probe's own cap-cost cell: one code path, read twice
    if abs(legacy["full_train"] - reduced) > 1e-9 or abs(legacy["capped"] - capped) > 1e-9:
        raise ValueError("accuracy_bridge_figure: the path and the cap-cost cell disagree")
    floors = [m for m in matched if m.endswith("stratified_random")]
    best = {
        fam: max((m for m in matched if family(m) == fam), key=lambda m: matched[m], default=None)
        for fam in ("classical", "quantum")
    }
    if len(floors) != 1 or None in best.values():
        raise KeyError("accuracy_bridge_figure: the matched table needs both families and a floor")
    if not tuned > retuned > reduced > capped:
        raise ValueError("accuracy_bridge_figure: the three steps should each cost accuracy")
    classical, quantum = matched[best["classical"]], matched[best["quantum"]]
    floor = matched[floors[0]]
    name = display_model(BRIDGE_COST_MODEL)
    rows = [  # (label, score, the score the step starts from, model key for the hue)
        (f"{name}, all features", tuned, None, BRIDGE_COST_MODEL),
        ("Matched-contest tuning", retuned, tuned, BRIDGE_COST_MODEL),
        (f"{BRIDGE_DIMS} components", reduced, retuned, BRIDGE_COST_MODEL),
        (f"{legacy['n_capped']}-row cap", capped, reduced, BRIDGE_COST_MODEL),
        (f"{display_model(best['classical'])}, matched", classical, None, best["classical"]),
        (f"{display_model(best['quantum'])}, matched", quantum, classical, best["quantum"]),
    ]
    drawn = [floor] + [r[1] for r in rows]
    if min(drawn) <= BRIDGE_XLIM[0] or max(drawn) >= BRIDGE_XLIM[1]:
        raise ValueError(f"accuracy_bridge_figure: a reading lies outside the axis {BRIDGE_XLIM}")

    fig, ax = plt.subplots(figsize=(FIGSIZE["col"][0], 1.62))
    fig.subplots_adjust(left=0.46, right=0.87, top=0.99, bottom=0.235)
    inside = []
    for y, (_, score, start, model) in enumerate(rows):
        if start is not None:
            ax.barh(
                y,
                start - score,
                left=score,
                height=BRIDGE_BAR,
                color=tint(MEANING["reference"]),
                edgecolor=MEANING["reference"],
                lw=OUTLINE_PT,
                zorder=1,
            )
            ax.plot(
                [start, start],
                [y - 1, y + BRIDGE_BAR / 2],
                color=MEANING["reference"],
                lw=0.6,
                ls=":",
                zorder=1,
            )
            cost = _true_minus(f"{score - start:+.3f}")
            # inside a wide bar, at its right end; beside a narrow one, on the side that stays
            # in the panel
            if start - score > BRIDGE_WIDE:
                x, ha = start - 0.008, "right"
            elif start + BRIDGE_WIDE <= BRIDGE_XLIM[1]:
                x, ha = start + 0.008, "left"
            else:
                x, ha = score - 0.02, "right"
            inside.append(ax.text(x, y, cost, ha=ha, va="center", fontsize=LABEL_PT))
        ax.plot(score, y, "o", ms=5.5, color=family_color(model), zorder=3)
        ax.text(
            1.03,
            y,
            f"{score:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=LABEL_PT,
        )
    ax.axhline(3.5, color=COLORS["neutral"], lw=0.6)
    ax.axvline(floor, color=MEANING["reference"], lw=0.8, ls="--", zorder=0)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlim(*BRIDGE_XLIM)
    ax.set_xlabel("Macro-F1")
    ax.grid(axis="y", visible=False)
    fig.canvas.draw()
    _assert_text_fits(fig, ax, inside, [])
    names = [t.get_window_extent() for t in ax.get_yticklabels()]
    axis_name = ax.xaxis.label.get_window_extent()
    outside = min(e.x0 for e in names) < 0 or axis_name.x1 > fig.bbox.x1 or axis_name.y0 < 0
    if outside or any(a.overlaps(b) for a, b in zip(names, names[1:], strict=False)):
        raise ValueError("accuracy_bridge_figure: a row name or the axis name does not fit")
    out = FIG_DIR / f"{out_name}.pdf"
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


RADIUS_MODELS = ("rbf_svm", "qsvm")
RADIUS_CLEAN_X = 0.004  # where the clean score is drawn, left of the smallest tested radius
RADIUS_NAME_OFFSET_PT = 2.5  # a curve's name above its right end


RADIUS_YLIM = (-0.04, 0.68)


def radius_curve_figure(
    worst_radius_run: Path, robustness_csv: Path, out_name: str, source: str = "msu"
) -> Path:
    """Macro-F1 of the two kernel machines under their own white-box attack, against the radius.

    Reads the pinned worst-radius run (experiments/read_worst_radius.py): per model the clean
    score, the mean attacked score at each tested radius with its spread over the attack starts,
    the worst radius, and the constant-predictor floor (dotted). The stratified-random floor
    (dashed, as in every other figure) is the robustness table's. The picture behind the
    reading-radius item of the checklist: the lowest score lies inside the budget, and the
    RBF-SVM's score rises back to the constant-predictor floor at the budget, where its output
    is constant. A score outside the drawn range fails loud.
    """
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    node = json.loads((Path(worst_radius_run) / "results.json").read_text())["sources"][source]
    missing = [m for m in RADIUS_MODELS if m not in node["models"]]
    if missing:
        raise KeyError(f"{worst_radius_run}: no white-box curve for {missing}")
    floor = node["constant_predictor_floor"]["floor"]
    chance = _stratified_floor(_csv_rows(robustness_csv), robustness_csv)
    attack = MEANING["attack"]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_PT / 72, 1.85))
    ax.axhline(floor, color=MEANING["reference"], ls=":", lw=0.9)
    ax.axhline(chance, color=MEANING["reference"], ls="--", lw=0.9)
    ticks = set()
    for m in RADIUS_MODELS:
        d = node["models"][m]
        radii = [float(r) for r in d["radii"]]
        keys = [_num_key(d["mean_curve"], r) for r in radii]
        mean = [d["mean_curve"][k] for k in keys]
        spread = [float(np.std(d["per_start"][k], ddof=1)) for k in keys]
        drawn = [d["clean"], *mean]
        if min(drawn) < RADIUS_YLIM[0] or max(drawn) > RADIUS_YLIM[1]:
            raise ValueError(f"{worst_radius_run}: {m} leaves the drawn range {RADIUS_YLIM}")
        hue = family_color(m)
        ax.plot([RADIUS_CLEAN_X, radii[0]], [d["clean"], mean[0]], color=hue, lw=0.8, ls=":")
        ax.plot(RADIUS_CLEAN_X, d["clean"], "o", ms=5, mfc="white", mec=hue, mew=1.1)
        ax.errorbar(
            radii,
            mean,
            yerr=spread,
            color=hue,
            marker="o",
            ms=3.5,
            lw=1.2,
            capsize=1.5,
            elinewidth=0.8,
        )
        ax.plot(d["worst"]["radius"], d["worst"]["mean"], "D", ms=5.5, color=attack, zorder=3)
        # each curve is named at its right end: above the RBF-SVM's, which ends at the floor,
        # and above the QSVM's, which ends near zero; the offset keeps the RBF-SVM's name between
        # the dotted and the dashed floor lines
        ax.annotate(
            display_model(m),
            (radii[-1], mean[-1]),
            xytext=(0, RADIUS_NAME_OFFSET_PT),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=hue,
        )
        ticks |= set(radii)
    ax.set_xscale("log")
    shown = sorted(ticks)
    ax.set_xticks([RADIUS_CLEAN_X, *shown], ["clean", *[f"{r:g}" for r in shown]])
    ax.minorticks_off()
    ax.set_xlim(RADIUS_CLEAN_X / 1.35, max(shown) * 1.25)
    ax.set_ylim(*RADIUS_YLIM)
    ax.set_xlabel("Radius of the white-box attack ($\\sigma$)")
    ax.set_ylabel("Macro-F1")
    ax.text(0.012, floor - 0.012, "constant-predictor floor", color=COLORS["neutral"], va="top")
    ax.text(0.012, chance + 0.012, "stratified-random floor", color=COLORS["neutral"], va="bottom")
    key = [
        plt.Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=COLORS["neutral"]),
        plt.Line2D([], [], ls="none", marker="D", ms=5.5, color=attack),
    ]
    ax.legend(
        key,
        ["clean", "worst radius"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.02),
        ncol=2,
        frameon=False,
        handletextpad=0.1,
        columnspacing=1.0,
        borderpad=0,
        borderaxespad=0.2,
    )
    fig.tight_layout(pad=0.3)
    out = FIG_DIR / f"{out_name}.pdf"
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out
