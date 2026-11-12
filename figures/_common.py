"""Shared paths, palette and plotting style for the figure scripts.

DATA  : directory holding the intermediate tables produced by the scripts in analysis/
FIGOUT: where the figure PNGs are written

Author : Prof. Bruno Duarte Gomes
         Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB-UFPA
         Laboratório de Simulação e Biologia Computacional, CCAD-UFPA
"""
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import ROOT, DATA, FIGOUT  # noqa: E402

# table name -> path, so the figure scripts can be read next to the analysis scripts
P = {p.name: p for p in DATA.glob("*")}

NAV = "#1F4E79"   # records / primary series
RED = "#B3352B"   # outside STN, or processes that inflate q
GRY = "#9AA3AC"   # surrogates and reference lines
AMB = "#D99A3E"   # predictions, criteria, moment boundaries
DGR = "#55606B"   # secondary annotation text


def apply_figure_style():
    """Publication defaults: no top/right spines, consistent fonts and sizes."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
    })


def plet(ax, s, x=-0.10):
    """Panel letter, bold, outside the axes box."""
    ax.text(x, 1.03, s, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="left")
