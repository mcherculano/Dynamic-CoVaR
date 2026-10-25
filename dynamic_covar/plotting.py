"""Shared plotting helpers for the examples."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, RED, GREY, BLACK = "#1f4e9c", "#c0392b", "#7f7f7f", "#222222"
Z90 = 1.6448536269514722                              # 95th percentile of N(0, 1)
FIGURES = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def band(ax, mean, sd, color, label, x=None):
    """Mean with a 90% band; x defaults to the index of a pandas Series."""
    x = mean.index if x is None else x
    ax.fill_between(x, mean - Z90 * sd, mean + Z90 * sd, color=color, alpha=0.25, lw=0)
    ax.plot(x, mean, color=color, lw=1.2, label=label)


def save(fig, name, top=1.0):
    """Save to figures/; top < 1 leaves room for a legend placed above the panels."""
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
