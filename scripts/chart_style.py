"""
Shared chart style for all Echo Breakers charts (one palette, one font, recessive axes).

Colours are a colour-blind-checked palette (blue/orange pass the CVD and contrast
checks on the light surface). To rebrand, change the values here only.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # titles, values
INK_2 = "#52514e"        # axis labels, tick labels
MUTED = "#898781"        # footnotes
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"         # series 1 (e.g. shown / feed)
ORANGE = "#eb6834"       # series 2 (e.g. clicked)
GREY = "#b5b3ab"         # "everything else" / reference
RED = "#e34948"

# Diverging: red (less than average) <- grey midpoint -> blue (more than average)
DIVERGING = LinearSegmentedColormap.from_list(
    "echo_div", ["#a8302f", RED, "#f0efec", BLUE, "#184f95"])


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 9.5,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
        "patch.edgecolor": SURFACE,   # thin surface gap between adjacent bars
        "patch.linewidth": 1.0,
        "lines.linewidth": 2,
    })


def value_grid(ax, axis="y"):
    """Hairline grid on the value axis only, drawn behind the marks."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis)
