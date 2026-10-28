"""Shared figure style for every PNG the project produces (slides and dashboard).

Conventions:
  * one colour per hazard, the same in every figure (Okabe-Ito palette, which stays
    distinguishable for the common forms of colour-vision deficiency);
  * meaning is never carried by colour alone - lines also differ in style and
    markers, and everything is labelled;
  * units on every axis; fonts large enough to read when projected;
  * no rainbow colormaps: sequential maps go from light grey to the hazard colour.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HAZARD_COLOURS: dict[str, str] = {
    "noise": "#0072B2",     # blue
    "dust": "#E69F00",      # orange
    "heat": "#D55E00",      # vermillion
    "co2": "#009E73",       # bluish green
}
NEUTRAL = "#4D4D4D"
LIGHT = "#BDBDBD"
BASELINE_STYLES: dict[str, dict[str, Any]] = {
    "actual": {"color": "black", "linestyle": "-", "linewidth": 2.0},
    "persistence": {"color": NEUTRAL, "linestyle": "--", "linewidth": 1.6},
    "seasonal": {"color": NEUTRAL, "linestyle": ":", "linewidth": 1.8},
}
TARGET_HAZARD = {"noise_dba": "noise", "pm25_ugm3": "dust", "pm10_ugm3": "dust", "wbgt_c": "heat", "co2_ppm": "co2"}
DPI = 150


def apply() -> None:
    """Set matplotlib defaults for slide-legible figures."""
    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14, "xtick.labelsize": 12,
        "ytick.labelsize": 12, "legend.fontsize": 12, "figure.dpi": 100, "savefig.dpi": DPI,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
        "savefig.bbox": "tight",
    })


def sequential_cmap(hazard: str) -> LinearSegmentedColormap:
    """Light grey -> hazard colour (single-hue, ordered; never a rainbow)."""
    return LinearSegmentedColormap.from_list(f"seq_{hazard}", ["#F2F2F2", HAZARD_COLOURS[hazard]])


def colour_for_target(target: str) -> str:
    """Hazard colour for a target/series name."""
    return HAZARD_COLOURS[TARGET_HAZARD[target]]


def simulated_badge(ax: Any) -> None:
    """Mark a figure as SIMULATED DATA (CLAUDE.md: visible on everything this phase)."""
    ax.text(0.995, 1.01, "SIMULATED DATA", transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
            fontweight="bold", color="white", bbox={"boxstyle": "round,pad=0.25", "fc": "#B2182B", "ec": "none"})
