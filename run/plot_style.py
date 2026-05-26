"""
plot_style.py — Shared matplotlib style for all PEMWE figures.

Import at the top of every figure script:

    from plot_style import apply_style, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, DARK

Then call apply_style() before creating any figure.
"""
from __future__ import annotations
import matplotlib as mpl


# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = "#2C73D2"
ORANGE = "#FF6B35"
GREEN  = "#44BBA4"
RED    = "#E63946"
PURPLE = "#7B2D8B"
GREY   = "#6C757D"
DARK   = "#1C1C2E"


def tex(s: str) -> str:
    """
    Escape a plain-text (non-math) string for safe use in a usetex label.

    Only escapes characters that LaTeX treats specially in text mode.
    Do NOT pass strings that already contain LaTeX commands or $...$ math.
    For labels with math, write the LaTeX directly as a raw string instead.
    """
    # Order matters: escape backslash first so later replacements aren't doubled.
    s = s.replace("\\", r"\textbackslash{}")
    for char, repl in [
        ("_",  r"\_"),
        ("^",  r"\^{}"),
        ("&",  r"\&"),
        ("#",  r"\#"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        # Common Unicode
        ("²",  r"$^2$"),
        ("³",  r"$^3$"),
        ("°",  r"$^\circ$"),
        ("µ",  r"$\mu$"),
        ("η",  r"$\eta$"),
        ("₂",  r"$_2$"),
        ("€",  r"\euro{}"),
        ("—",  r"---"),
        ("→",  r"$\to$"),
    ]:
        s = s.replace(char, repl)
    return s


def apply_style() -> None:
    """Apply the project-wide rcParams. Call once before creating figures."""
    mpl.rcParams.update({
        # ── Typography ────────────────────────────────────────────────────────
        "text.usetex"        : True,
        "font.family"        : "serif",
        "font.serif"         : ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{eurosym}",
        # ── Sizes ─────────────────────────────────────────────────────────────
        "font.size"          : 11,
        "axes.titlesize"     : 12,
        "axes.titleweight"   : "bold",
        "axes.labelsize"     : 11,
        "axes.labelweight"   : "normal",
        "xtick.labelsize"    : 10,
        "ytick.labelsize"    : 10,
        "legend.fontsize"    : 9,
        "lines.linewidth"    : 2.0,
        # ── Layout ────────────────────────────────────────────────────────────
        "figure.figsize"     : (7.0, 4.5),
        "savefig.dpi"        : 300,
        "figure.dpi"         : 150,
        # ── Axes ─────────────────────────────────────────────────────────────
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.grid"          : True,
        "grid.alpha"         : 0.30,
        "grid.linewidth"     : 0.8,
    })
