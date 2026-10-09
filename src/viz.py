"""Shared matplotlib style. Palette = validated categorical order (fixed slots, never cycled)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
DIV = ("#2a78d6", "#f0efec", "#e34948")  # blue <-> gray <-> red
DEM, REP = "#2a78d6", "#e34948"


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "legend.frameon": False, "lines.linewidth": 1.6, "font.family": "DejaVu Sans",
        "axes.prop_cycle": mpl.cycler(color=SLOTS), "figure.dpi": 110, "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def save(fig, name):
    from .config import FIG
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
