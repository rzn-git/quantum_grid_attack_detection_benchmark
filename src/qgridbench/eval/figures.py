"""All paper figures are generated ONLY here, from results/runs JSON.

Design tokens (single source of truth) live in eval.design_tokens. Figures write
to figures/ as PDF (vector, paper-ready).
"""

from __future__ import annotations

import json
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
    LABEL_PT,
    apply_style,
)
from qgridbench.eval.tables import display_cell, display_model
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
# PennyLane draws a gate box 0.75 layers wide and sizes its label for a drawing one inch per
# layer. Shrunk to one column the pitch is about 18 pt, so an 8-pt "RZ" (12 pt wide) overran its
# 13-pt box. A wider box (0.90 of the pitch) and 0.8-pt outlines keep every label and the
# measurement glyph inside its box at the printed size.
CIRCUIT_BOX_LENGTH = 0.9
CIRCUIT_RC = {
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "patch.facecolor": "white",
    "patch.edgecolor": "black",
    "patch.force_edgecolor": True,
    "patch.linewidth": 0.8,
    "lines.color": "black",
    "lines.linewidth": 0.8,
    "text.color": "black",
    "path.sketch": None,
    "font.weight": "normal",
}


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
    for i, fam in enumerate(families):
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
        color = COLORS["series"][i % len(COLORS["series"])]
        ax.plot(eps_grid, mean, marker="o", label=display_model(fam), color=color, linewidth=1.6)
        ax.fill_between(eps_grid, mean - std, mean + std, color=color, alpha=0.15)
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
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
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
                color="white" if M[i, j] < 0.6 else "black",
            )
    fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        fraction=0.06,
        pad=0.04,
        label="Test macro-F1 under attack",
    )
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


def _save_circuit(fig, out: Path) -> None:
    """Scale a drawn circuit so its native width is one column, then save it tightly.

    PennyLane lays a circuit out at one inch per layer and sets gate labels in points, so a
    long circuit shrunk to a column loses its label size. Here the text stays at
    CIRCUIT_LABEL_PT and the drawing is resized instead: the gate boxes shrink around the
    text, which is why the circuits are drawn with few layers and without angle values.
    """
    target = COLUMN_WIDTH_PT - 2 * CIRCUIT_PAD_IN * 72
    tight_pt = fig.get_tightbbox(fig.canvas.get_renderer()).width * 72
    k = target / tight_pt
    fig.set_size_inches(fig.get_figwidth() * k, fig.get_figheight() * k)
    fig.savefig(out, metadata=PDF_METADATA, bbox_inches="tight", pad_inches=CIRCUIT_PAD_IN)
    plt.close(fig)


def circuit_diagrams(n_qubits: int = 3, zz_reps: int = 1) -> list[Path]:
    """Render the actual model circuits (paper methodology figure).

    Draws the ZZ feature map from qkernel's own circuit builder (SSOT: the drawing
    can never drift from what the experiments ran) and the VQC circuit
    (AngleEmbedding + StronglyEntanglingLayers, mirroring vqc.py's `circuit`),
    gate-expanded so the reader sees actual operations, not template boxes.

    Both PDFs are one 252-pt column wide natively and carry 8-pt gate labels, so placing
    each at the column width keeps the labels at 8 pt. To get there the drawings show three
    qubits (the experiments run 8--16), ONE repetition of the ZZ map (the deployed map
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
        _save_circuit(fig, out)
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
        _save_circuit(fig, out)
    outs.append(out)
    return outs


def bandwidth_diagnostics_figure(bw_json: Path, out_name: str) -> Path:
    """Bandwidth sweep per cell: spectrum shape (top row) and F1/KTA/g (bottom row).

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

    fig, axes = plt.subplots(2, len(cells), figsize=FIGSIZE["text_tall"], sharex=True)
    for j, cell in enumerate(cells):
        sub = sorted((r for r in rows if r["cell"] == cell), key=lambda r: r["bandwidth"])
        bw = [r["bandwidth"] for r in sub]

        ax = axes[0, j]
        ax.plot(
            bw,
            [r["median_offdiag"] for r in sub],
            marker="o",
            markersize=3,
            color=COLORS["series"][0],
            label="Median off-diagonal fidelity",
        )
        ax.plot(
            bw,
            [r["top_eig_share"] for r in sub],
            marker="s",
            markersize=3,
            color=COLORS["series"][1],
            label="Top-eigenvalue share",
        )
        ax.plot(
            bw,
            [r["frac_below_0.01"] for r in sub],
            marker="^",
            markersize=3,
            color=COLORS["series"][2],
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
            marker="o",
            markersize=3,
            color=COLORS["series"][0],
            label="Validation macro-F1",
        )
        ax.axvline(sub[0]["tuned_bandwidth"], color="gray", linestyle=":", linewidth=1)
        ax2 = ax.twinx()
        ax2.plot(
            bw,
            [r["g_vs_tuned_rbf"] for r in sub],
            marker="s",
            markersize=3,
            color=COLORS["series"][1],
            label="Geometric difference",
        )
        ax2.plot(
            bw,
            [r["kta_grid_labels"] for r in sub],
            marker="^",
            markersize=3,
            color=COLORS["series"][2],
            label="Kernel-target alignment",
        )
        ax2.set_yscale("log")
        _plain_log_ticks(ax2.yaxis, subs=(1.0,))
        ax.set_xlabel("Encoding bandwidth\n(scale factor)")
        if j == 0:
            ax.set_ylabel("Validation macro-F1")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            bottom_handles = h1 + h2
            bottom_labels = l1 + l2
        if j == len(cells) - 1:
            ax2.set_ylabel("Geometric difference,\nalignment (log scale)")

    # legends sit outside the panels: seven 8-pt entries would cover the curves
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.legend(bottom_handles, bottom_labels, loc="lower center", ncol=3, frameon=False)
    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
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
    for i, fam in enumerate(families):
        color = COLORS["series"][i % len(COLORS["series"])]
        ax_p.plot(
            [r * 100 for r in rates],
            [poison[_num_key(poison, r)][fam] for r in rates],
            marker="o",
            label=display_model(fam),
            color=color,
            linewidth=1.4,
        )
    ax_p.set_xlabel("Training labels flipped (%)")
    ax_p.set_ylabel("Macro-F1 on clean test data")

    snrs = sorted((int(s) for s in per_seed[0]["sensor_noise"][families[0]]), reverse=True)
    xs = list(range(len(snrs) + 1))  # clean + descending SNR = increasing severity
    for i, fam in enumerate(families):
        color = COLORS["series"][i % len(COLORS["series"])]
        clean = np.array([s["evasion"]["clean"][fam] for s in per_seed])
        curve = np.array([[s["sensor_noise"][fam][str(snr)] for snr in snrs] for s in per_seed])
        mean = np.concatenate([[clean.mean()], curve.mean(0)])
        std = np.concatenate([[clean.std()], curve.std(0)])
        ax_n.plot(xs, mean, marker="o", color=color, linewidth=1.4)
        ax_n.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)
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

    Single panel, twin y: clean macro-F1 (left) and random-perturbation flip rate at
    the largest radius (right) vs encoding bandwidth. Backs the no-safe-operating-
    point claim: the accuracy-optimal bandwidth is also the fragility optimum.
    """
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    points = json.loads(frag_json.read_text())
    bws = [p["bandwidth"] for p in points]
    f1 = [p["clean_macro_f1"] for p in points]
    eps = max(points[0]["flip_rate"], key=float)
    flips = np.array([p["flip_rate"][eps]["draws"] for p in points])

    fig, ax = plt.subplots(figsize=FIGSIZE["col"])
    ax.plot(bws, f1, marker="o", color=COLORS["primary"], label="Clean macro-F1")
    ax.set_xscale("log")
    _plain_log_ticks(ax.xaxis)
    ax.set_xlabel("Encoding bandwidth (scale factor)")
    ax.set_ylabel("Clean macro-F1", color=COLORS["primary"])
    ax.tick_params(axis="y", labelcolor=COLORS["primary"])
    ax.axvline(bws[int(np.argmax(f1))], color=COLORS["neutral"], linestyle=":", linewidth=1)

    ax2 = ax.twinx()
    ax2.errorbar(
        bws,
        flips.mean(1),
        yerr=flips.std(1),
        marker="s",
        color=COLORS["accent"],
        capsize=2,
        label=f"Flip rate, random perturbation of {float(eps)} standard deviations",
    )
    ax2.set_ylabel(
        f"Random-perturbation flip rate\n(fraction, {float(eps)} standard deviations)",
        color=COLORS["accent"],
    )
    ax2.tick_params(axis="y", labelcolor=COLORS["accent"])
    ax2.grid(False)

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
    ax_q.plot(qubits, np.log10(var), marker="o", color=COLORS["primary"], label="Measured")
    ax_q.plot(
        qubits,
        np.log10(ref),
        linestyle="--",
        color=COLORS["neutral"],
        label="Reference (halves per qubit)",
    )
    ax_q.set_xticks(qubits)
    ax_q.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_q.set_xlabel("Number of qubits (depth fixed)")
    ax_q.set_ylabel("Loss-gradient variance\nat initialization (log10)")
    ax_q.legend(loc="upper right", frameon=False)

    ax_d.plot(depths, gvar, marker="s", color=COLORS["accent"])
    ax_d.set_xticks(depths)
    ax_d.set_ylim(0, max(gvar) * 1.3)
    ax_d.set_xlabel("Circuit depth (layers, 8 qubits)")
    ax_d.set_ylabel("Gradient-norm variance\nduring training")

    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout()
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


def data_efficiency_curve(de_run_dir: Path, out_name: str) -> Path:
    """test macro-F1 vs training-set size for the best classical + quantum models."""
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads((de_run_dir / "data_efficiency.json").read_text())
    fig, ax = plt.subplots(figsize=FIGSIZE["col"])
    for i, (name, series) in enumerate(data.items()):
        sizes = sorted(int(s) for s in series)
        mean = [series[str(s)]["mean"] for s in sizes]
        std = [series[str(s)]["std"] for s in sizes]
        color = COLORS["series"][i % len(COLORS["series"])]
        ax.errorbar(
            sizes, mean, yerr=std, marker="s", label=display_model(name), color=color, capsize=2
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
