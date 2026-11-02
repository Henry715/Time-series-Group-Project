"""Shared IEEE-conference figure style (single column 3.5 in, double column 7.16 in)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# IEEEtran conference: \columnwidth = 252 pt, \textwidth = 516 pt (1 in = 72.27 pt)
COL1 = 252 / 72.27   # 3.487 in
COL2 = 516 / 72.27   # 7.140 in

# Okabe-Ito colour-blind-safe palette; also separable in grayscale with markers/hatches
C = {
    "base": "#0072B2",   # blue
    "z":    "#E69F00",   # orange
    "tune": "#CC79A7",   # reddish purple
    "es":   "#009E73",   # green
    "gray": "#555555",
}
NAMES = {
    "base": "FCN (baseline)",
    "z":    "FCN + z-score",
    "tune": "FCN-Opt",
    "es":   "FCN-Opt + ES",
}
# Classes get their own neutral/vermillion pair so they never clash with the model colours above
CLS = {0: "#4D4D4D", 1: "#D55E00"}      # class -1: dark gray, class +1: vermillion
CLS_LS = {0: "-", 1: "--"}
CLS_MARK = {0: "o", 1: "^"}
HATCH = {"base": "", "z": "////", "tune": "xxxx", "es": "...."}
MARK = {"base": "o", "z": "s", "tune": "^", "es": "D"}


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        "legend.fancybox": False,
        "hatch.linewidth": 0.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,     # embed TrueType -> IEEE PDF eXpress friendly
        "ps.fonttype": 42,
    })


def panel_label(ax, s, x=-0.02, y=-0.30):
    """IEEE-style sub-caption '(a) ...' centred under the axes."""
    ax.text(0.5, y, s, transform=ax.transAxes, ha="center", va="top", fontsize=8)


def save(fig, name, outdir="figures", width=None):
    """Save so that the *cropped* PDF is exactly `width` wide (default: COL1 or COL2,
    picked from the figure size). Included with width=\\columnwidth / \\textwidth the
    figure is then not rescaled, so every label stays at exactly 8 pt."""
    import os
    os.makedirs(outdir, exist_ok=True)
    pad = mpl.rcParams["savefig.pad_inches"]
    w0, h0 = fig.get_size_inches()
    target = width or (COL2 if w0 > 5 else COL1)
    for _ in range(8):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        cur = bb.width + 2 * pad
        if abs(cur - target) < 0.002:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + (target - cur), h)
    fig.savefig(f"{outdir}/{name}.pdf")
    fig.savefig(f"{outdir}/{name}.png", dpi=300)
    plt.close(fig)
