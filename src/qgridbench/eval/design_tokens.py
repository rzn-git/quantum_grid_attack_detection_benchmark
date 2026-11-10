"""Design tokens: single source of truth for figure colors, sizes, typography.

Every figure reads from here — no inline hex codes or pixel values elsewhere
(the design-token rule). Restrained palette: one primary, one accent,
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

# IEEE Transactions on Smart Grid (PES author kit, Part 4a): figure labels print at 8-10 pt.
# Figures are drawn at the width they are placed at (one column or the full text width), so
# native width equals placed width and every label prints at LABEL_PT.
COLUMN_WIDTH_PT = 252.0
TEXT_WIDTH_PT = 516.0
LABEL_PT = 8.0

FIGSIZE = {
    "col": (COLUMN_WIDTH_PT / 72, 3.3),  # one column: place at \columnwidth
    "col_tall": (COLUMN_WIDTH_PT / 72, 3.9),
    "text_heatmap": (
        TEXT_WIDTH_PT / 72,
        3.2,
    ),  # full-width heat map: place as figure* at \textwidth
    "text": (TEXT_WIDTH_PT / 72, 3.0),  # full text width: place at \textwidth
    "text_tall": (TEXT_WIDTH_PT / 72, 4.8),
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            # embed TrueType (42) rather than Type 3 fonts: IEEE PDF checks reject Type 3
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": LABEL_PT,
            "font.family": "serif",
            "axes.titlesize": LABEL_PT,
            "axes.labelsize": LABEL_PT,
            "axes.edgecolor": COLORS["neutral"],
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "legend.fontsize": LABEL_PT,
            "lines.linewidth": 1.5,
            "xtick.labelsize": LABEL_PT,
            "ytick.labelsize": LABEL_PT,
        }
    )
