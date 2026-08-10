"""
comparison_lcoh_relative.py
============================
Single-panel bar chart: relative LCOH change vs. Load-following baseline [%].
All four controllers shown; bars coloured with standard project palette.

Usage
-----
    python3 run/comparison_lcoh_relative.py
Output
------
    results/comparison_lcoh_relative.pdf
    results/comparison_lcoh_relative.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "run"))
from plot_style import apply_style, BLUE, GREEN, ORANGE, PURPLE, GREY, DARK
apply_style()
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath}\usepackage{eurosym}\usepackage{amssymb}"
)

from pemwe.plant import load_plant

# ── Constants ─────────────────────────────────────────────────────────────────
plant       = load_plant(REPO_ROOT / "configs" / "plant_parameters.yaml")
econ        = plant["economics"]
V_EOL_MV    = float(plant["degradation"]["V_deg_EOL_V"]) * 1000
P_RATED_KW  = float(plant["stack"]["P_rating_W"]) / 1000
SYS_LIFE_YR = float(econ["system_lifetime_yr"])
WACC        = float(econ.get("wacc", 0.0))
ANNUITY     = (1 - (1 + WACC) ** -SYS_LIFE_YR) / WACC if WACC > 0 else SYS_LIFE_YR
SYS_CAPEX_EUR  = float(econ["system_capex_eur_per_kW"]) * P_RATED_KW
STACK_REPL_EUR = (float(econ["capex_usd_per_kW"])
                  * float(econ["eur_per_usd"])
                  * P_RATED_KW)

# ── Controller metadata ───────────────────────────────────────────────────────
CTRL_ORDER = ["LF", "PA", "DA", "LA"]
CTRL_META  = {
    "LF": {"label": "LF",  "color": BLUE,
           "csv": REPO_ROOT / "results/lifetime_5yr/load_following_5yr/load_following_5yr.csv"},
    "PA": {"label": "PA",  "color": GREEN,
           "csv": REPO_ROOT / "results/lifetime_5yr/price_aware_5yr/price_aware_5yr.csv"},
    "DA": {"label": "DA",  "color": ORANGE,
           "csv": REPO_ROOT / "results/lifetime_5yr/degradation_aware_5yr/degradation_aware_5yr.csv"},
    "LA": {"label": "LA",  "color": PURPLE,
           "csv": REPO_ROOT / "results/lifetime_5yr/lifetime_aware_5yr/lifetime_aware_5yr.csv"},
}

DT_H = 1.0

# ── LCOH helpers ──────────────────────────────────────────────────────────────

def _system_lcoh(h2_annual_kg):
    return SYS_CAPEX_EUR / (h2_annual_kg * ANNUITY) if h2_annual_kg > 0 else 0.0


def _replacement_lcoh(h2_annual_kg, life_cal_yr):
    if not (h2_annual_kg > 0 and np.isfinite(life_cal_yr) and life_cal_yr > 0):
        return 0.0
    times = np.arange(life_cal_yr, SYS_LIFE_YR + life_cal_yr, life_cal_yr)
    times = times[times <= SYS_LIFE_YR]
    if len(times) == 0:
        return 0.0
    npv = sum(STACK_REPL_EUR / (1 + WACC) ** t for t in times)
    return npv / (h2_annual_kg * ANNUITY)


def compute_lcoh(csv_path):
    df           = pd.read_csv(csv_path)
    total_h      = df["t_h"].max()
    sim_yr       = total_h / 8760.0
    total_H2     = (df["m_dot_H2_kg_h"] * DT_H).sum()
    total_elec   = (df["c_elec_eur_h"]   * DT_H).sum()
    total_shut   = df["c_shutdown_eur"].sum()
    h2_annual    = total_H2 / max(sim_yr, 1e-9)
    total_dV_mV  = df["dV_deg_V"].sum() * 1000      # robust to stack resets
    r_ann        = total_dV_mV / max(sim_yr, 1e-9)
    life_cal_yr  = V_EOL_MV / r_ann if r_ann > 0 else np.inf
    return ((total_elec + total_shut) / max(total_H2, 1e-12)
            + _replacement_lcoh(h2_annual, life_cal_yr)
            + _system_lcoh(h2_annual))


lcoh = {k: compute_lcoh(CTRL_META[k]["csv"]) for k in CTRL_ORDER}
ref  = lcoh["DA"]
# Normalise to each controller's own LCOH: "how much of their cost does DA save"
# DA itself is 0; for DA the denominator is ref (avoids div-by-zero, gives 0%)
pct  = {k: (lcoh[k] - ref) / lcoh[k] * 100 if k != "DA"
        else 0.0
        for k in CTRL_ORDER}

# stdout summary
print(f"\n{'Controller':<8}  {'LCOH [EUR/kg]':>14}  {'vs DA [%]':>10}")
print("-" * 38)
for k in CTRL_ORDER:
    print(f"  {k:<6}  {lcoh[k]:>14.3f}  {pct[k]:>+10.1f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2))

x      = np.arange(len(CTRL_ORDER))
bar_w  = 0.52
colors = [CTRL_META[k]["color"] for k in CTRL_ORDER]
values = [pct[k] for k in CTRL_ORDER]

bars = ax.bar(x, values, bar_w, color=colors, zorder=3,
              edgecolor=DARK, linewidth=0.5)

# Zero reference line
ax.axhline(0, color=DARK, lw=0.9, zorder=4)

# Annotate % value on each bar
for bar, v in zip(bars, values):
    yoff = 0.3 if v >= 0 else -0.5
    va   = "bottom" if v >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width() / 2, v + yoff,
            rf"${v:+.1f}\%$",
            ha="center", va=va, fontsize=9.5, color=DARK, zorder=5)

ax.set_xticks(x)
ax.set_xticklabels([CTRL_META[k]["label"] for k in CTRL_ORDER], fontsize=10.5)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.set_ylabel(r"$\Delta$LCOH vs.\ DA [\%]", fontsize=11, color=DARK)
ax.set_xlabel("Controller", fontsize=11, color=DARK)
ax.tick_params(colors=DARK)

# Extend ylim to give space for annotations
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin * 1.15, ymax * 1.20 if ymax > 0 else 2)

fig.tight_layout()

OUT = REPO_ROOT / "results"
for ext in (".pdf", ".png"):
    fig.savefig(OUT / f"comparison_lcoh_relative{ext}", bbox_inches="tight", dpi=200)
    print(f"Saved: comparison_lcoh_relative{ext}")
plt.close(fig)
