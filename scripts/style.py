"""Shared matplotlib style for the toy-model figures."""

from pathlib import Path

import matplotlib.pyplot as plt

FIGDIR = Path(__file__).resolve().parent / "figs"

# Categorical palette, fixed order (blue, orange, aqua, yellow, magenta, green)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#1a1a19"
MUTED = "#6b6a63"
GRID = "#e4e3dc"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "legend.fontsize": 10.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    }
)


def save(fig, name):
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf")
    print(f"wrote {FIGDIR.name}/{name}.pdf")


def done(fig, name):
    """tight_layout, save and close"""
    fig.tight_layout()
    save(fig, name)
    plt.close(fig)


# white-edged markers
MARK = dict(mec="white", mew=1.2)


def zoom_td(
    t, h_ref, h_mod, lab_ref, lab_mod, figsize, zooms=((-1200, 60), (-150, 80))
):
    """Reference (solid) against model (dashed) waveform in the time domain: two panels, the second a zoom
    on the merger. Returns fig, (a1, a2)."""
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    for ax, (lo, hi) in zip(axes, zooms):
        k = (t > lo) & (t < hi)
        ax.plot(t[k], h_ref[k], color=INK, lw=2.0, label=lab_ref)
        ax.plot(t[k], h_mod[k], color=C[1], lw=1.2, ls=(0, (4, 2)), label=lab_mod)
        ax.set_xlim(lo, hi)
        ax.set_ylabel(r"$\Re\,h_{22}$")
    axes[1].set_xlabel(r"$(t-t_{\rm peak})/M$")
    return fig, axes
