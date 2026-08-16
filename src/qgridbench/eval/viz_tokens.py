"""Design tokens: single source of truth for figure colors, sizes, typography.

Every figure reads from here — no inline hex codes or pixel values elsewhere
(CLAUDE.md design-token rule). Restrained palette: one primary, one accent,
a colorblind-safe categorical series, neutral grid.
"""

from __future__ import annotations

import matplotlib as mpl

COLORS = {
    "primary": "#1f4e79",  # deep grid-blue
    "accent": "#c55a11",  # burnt orange
    "neutral": "#5a5a5a",
    "grid": "#d9d9d9",
    # Okabe-Ito colorblind-safe categorical order
    "series": [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#999999",
    ],
}

FIGSIZE = {
    "col": (3.4, 2.4),  # single IEEE column
    "wide": (5.2, 3.0),
    "square": (3.6, 3.2),
    "double": (7.0, 3.0),  # full text width
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 8,
            "font.family": "serif",
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "axes.edgecolor": COLORS["neutral"],
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "legend.fontsize": 7,
            "lines.linewidth": 1.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
