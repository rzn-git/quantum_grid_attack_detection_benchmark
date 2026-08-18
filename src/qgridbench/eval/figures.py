"""All paper figures are generated ONLY here, from results/runs JSON.

Design tokens (single source of truth) live in eval.design_tokens. Figures write
to figures/ as PDF (vector, paper-ready).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qgridbench.eval.design_tokens import COLORS, FIGSIZE, apply_style
from qgridbench.utils.run_tracking import REPO_ROOT

FIG_DIR = REPO_ROOT / "figures"

# Matplotlib stamps a wall-clock /CreationDate into every PDF, which makes an
# otherwise deterministic figure differ byte-for-byte on each run. Suppressing it
# is what lets a reviewer re-run make_report and get artifacts identical to the
# released ones — the same reason run artifacts carry a git SHA rather than a
# timestamp in their content. Tables were already byte-stable; this closes the gap.
PDF_METADATA = {"CreationDate": None}


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

    fig, ax = plt.subplots(figsize=FIGSIZE["wide"])
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
        ax.plot(eps_grid, mean, marker="o", label=fam, color=color, linewidth=1.6)
        ax.fill_between(eps_grid, mean - std, mean + std, color=color, alpha=0.15)
    # unit is fractions of the TRAIN per-feature sigma, not raw PCA coordinates
    # (attacks/evasion.py SCALE CONTRACT) — say so on the axis, not just in prose
    ax.set_xlabel(rf"{attack.upper()} $\epsilon$  (L$_\infty$ budget, units of train $\sigma$)")
    ax.set_ylabel("test macro-F1")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout()
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
    """Source x target degradation matrix (mean over seeds), at max epsilon."""
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads((adv_run_dir / "results.json").read_text())
    mats = [s["evasion"]["transfer_matrix"] for s in data["per_seed"]]
    sources = list(mats[0].keys())
    targets = list(mats[0][sources[0]].keys())
    M = np.mean([[[m[s][t] for t in targets] for s in sources] for m in mats], axis=0)

    fig, ax = plt.subplots(figsize=FIGSIZE["square"])
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(targets)), targets, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(sources)), sources, fontsize=7)
    ax.set_xlabel("target model")
    ax.set_ylabel("attack source")
    for i in range(len(sources)):
        for j in range(len(targets)):
            ax.text(
                j,
                i,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if M[i, j] < 0.6 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout()
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out


def circuit_diagrams(n_qubits: int = 3) -> list[Path]:
    """Render the actual model circuits (paper methodology figure).

    Draws the ZZ feature map from qkernel's own circuit builder (SSOT — the drawing
    can never drift from what the experiments ran) and the VQC circuit
    (AngleEmbedding + StronglyEntanglingLayers, mirroring vqc.py's `circuit`),
    gate-expanded so the reader sees actual operations, not template boxes.
    Three qubits keeps the ZZ map's all-pairs entangling block legible at column
    width; the caption states the experiments run 8--16 qubits.
    """
    import pennylane as qml

    from qgridbench.models.quantum.qkernel import _circuit_state_fn

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = np.round(np.linspace(0.4, 1.3, n_qubits), 2)
    outs: list[Path] = []

    fig, _ = qml.draw_mpl(_circuit_state_fn("zz", n_qubits), style="black_white")(x)
    out = FIG_DIR / "circuit_zz.pdf"
    fig.savefig(out, metadata=PDF_METADATA, bbox_inches="tight")
    plt.close(fig)
    outs.append(out)

    dev = qml.device("default.qubit", wires=n_qubits)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=n_qubits)
    weights = np.round(np.linspace(0.1, 0.9, int(np.prod(shape))), 2).reshape(shape)

    @qml.qnode(dev)
    def vqc_circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    fig, _ = qml.draw_mpl(vqc_circuit, style="black_white", level="device")(x, weights)
    out = FIG_DIR / "circuit_vqc.pdf"
    fig.savefig(out, metadata=PDF_METADATA, bbox_inches="tight")
    plt.close(fig)
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

    fig, axes = plt.subplots(
        2, len(cells), figsize=(FIGSIZE["wide"][0] * 1.6, FIGSIZE["wide"][1] * 1.7), sharex=True
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
            color=COLORS["series"][0],
            label="median off-diag fidelity",
        )
        ax.plot(
            bw,
            [r["top_eig_share"] for r in sub],
            marker="s",
            markersize=3,
            color=COLORS["series"][1],
            label="top-eigenvalue share",
        )
        ax.plot(
            bw,
            [r["frac_below_0.01"] for r in sub],
            marker="^",
            markersize=3,
            color=COLORS["series"][2],
            label="frac. pairs < 0.01",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_xscale("log")
        ax.set_title(cell, fontsize=8)
        if j == 0:
            ax.set_ylabel("spectral shape")
            ax.legend(fontsize=5.5, frameon=False)

        ax = axes[1, j]
        ax.plot(
            bw,
            [r["val_macro_f1"] for r in sub],
            marker="o",
            markersize=3,
            color=COLORS["series"][0],
            label="val macro-F1",
        )
        ax.axvline(sub[0]["tuned_bandwidth"], color="gray", linestyle=":", linewidth=1)
        ax2 = ax.twinx()
        ax2.plot(
            bw,
            [r["g_vs_tuned_rbf"] for r in sub],
            marker="s",
            markersize=3,
            color=COLORS["series"][1],
            label="$g$",
        )
        ax2.plot(
            bw,
            [r["kta_grid_labels"] for r in sub],
            marker="^",
            markersize=3,
            color=COLORS["series"][2],
            label="KTA",
        )
        ax2.set_yscale("log")
        ax.set_xlabel("encoding bandwidth")
        if j == 0:
            ax.set_ylabel("val macro-F1")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=5.5, frameon=False)
        if j == len(cells) - 1:
            ax2.set_ylabel("$g$ / KTA (log)")

    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout()
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

    fig, (ax_p, ax_n) = plt.subplots(1, 2, figsize=FIGSIZE["double"], sharey=True)

    rates = sorted(float(r) for r in poison)
    for i, fam in enumerate(families):
        color = COLORS["series"][i % len(COLORS["series"])]
        ax_p.plot(
            [r * 100 for r in rates],
            [poison[_num_key(poison, r)][fam] for r in rates],
            marker="o",
            label=fam,
            color=color,
            linewidth=1.4,
        )
    ax_p.set_xlabel("training labels flipped (%)")
    ax_p.set_ylabel("clean-test macro-F1")
    ax_p.legend(fontsize=6.5, ncol=2, frameon=False)

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
    ax_n.set_xticks(xs, ["clean"] + [f"{s} dB" for s in snrs])
    ax_n.set_xlabel("sensor noise (SNR)")

    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout()
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
    ax.plot(bws, f1, marker="o", color=COLORS["primary"], label="clean macro-F1")
    ax.set_xscale("log")
    ax.set_xlabel("encoding bandwidth")
    ax.set_ylabel("clean macro-F1", color=COLORS["primary"])
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
        label=f"flip rate @ {float(eps)}$\\sigma$",
    )
    ax2.set_ylabel(f"random flip rate @ {float(eps)}$\\sigma$", color=COLORS["accent"])
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
    (controlled scan, log y) with a 2^-n reference anchored at the smallest width.
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

    fig, (ax_q, ax_d) = plt.subplots(1, 2, figsize=FIGSIZE["wide"])
    ax_q.semilogy(qubits, var, marker="o", color=COLORS["primary"], label="measured")
    ref = [var[0] * 2.0 ** (qubits[0] - q) for q in qubits]
    ax_q.semilogy(qubits, ref, linestyle="--", color=COLORS["neutral"], label=r"$2^{-n}$ reference")
    ax_q.set_xticks(qubits)
    ax_q.set_xlabel("qubits (depth fixed)")
    ax_q.set_ylabel("Var[$\\partial_\\theta \\mathcal{L}$] at init")
    ax_q.legend(fontsize=7, frameon=False)

    ax_d.plot(depths, gvar, marker="s", color=COLORS["accent"])
    ax_d.set_xticks(depths)
    ax_d.set_ylim(0, max(gvar) * 1.3)
    ax_d.set_xlabel("depth (8 qubits)")
    ax_d.set_ylabel("grad-norm variance (training)")

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
    fig, ax = plt.subplots(figsize=FIGSIZE["wide"])
    for i, (name, series) in enumerate(data.items()):
        sizes = sorted(int(s) for s in series)
        mean = [series[str(s)]["mean"] for s in sizes]
        std = [series[str(s)]["std"] for s in sizes]
        color = COLORS["series"][i % len(COLORS["series"])]
        ax.errorbar(sizes, mean, yerr=std, marker="s", label=name, color=color, capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("training-set size")
    ax.set_ylabel("test macro-F1")
    ax.legend(fontsize=7, frameon=False)
    out = FIG_DIR / f"{out_name}.pdf"
    fig.tight_layout()
    fig.savefig(out, metadata=PDF_METADATA)
    plt.close(fig)
    return out
