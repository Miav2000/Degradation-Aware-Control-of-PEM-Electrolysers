"""
sensitivity_temp_vs_j_decomposition.py
=======================================

Decomposes voltage-degradation savings into two components:

  (1)  j-effect   — change in degradation due to lower current density
                    (holding f_T = 1 for all controllers)
  (2)  T-effect   — additional reduction from lower stack temperature
                    (difference between j-only and actual degradation)

For each controller (commercial, cost_optimal, aware, aware_rul) the
per-step degradation rate is recomputed directly from the CSV columns
using the plant-parameter polynomial and the Arrhenius factor:

    dV_deg/dt = (α·j² + β·j + γ) · f_T(T)        [actual]
    dV_deg/dt = (α·j² + β·j + γ) · 1              [j-only]

Both are integrated cumulatively over the simulation horizon.

The "reference" controller is commercial. For each other controller:
  total_saving   = cumV_actual_ref[-1]  − cumV_actual[-1]
  j_saving       = cumV_jonly_ref[-1]   − cumV_jonly[-1]
  T_saving       = total_saving − j_saving

Positive saving = less degradation than commercial.

Run from project root:
    python run/sensitivity/sensitivity_temp_vs_j_decomposition.py

Produces:
    results/sensitivity_temp_vs_j_decomposition.png
and prints a summary table to stdout.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUT_FILE = RESULTS_DIR / "sensitivity_temp_vs_j_decomposition.png"

CONTROLLERS = {
    "load_following":  "load_following_wind_spot_dk1/load_following_wind_spot_dk1.csv",
    "price_aware":"price_aware_wind_spot_dk1/price_aware_wind_spot_dk1.csv",
    "degradation_aware":       "degradation_aware_wind_spot_dk1/degradation_aware_wind_spot_dk1.csv",
    "lifetime_aware":   "lifetime_aware_wind_spot_dk1/lifetime_aware_wind_spot_dk1.csv",
}

LABELS = {
    "load_following":  "Load-following",
    "price_aware":"Price-aware",
    "degradation_aware":       "Degradation-aware",
    "lifetime_aware":   "Aware+RUL",
}

COLORS = {
    "load_following":  "#555555",
    "price_aware":"#1f77b4",
    "degradation_aware":       "#2ca02c",
    "lifetime_aware":   "#d62728",
}

# ---------------------------------------------------------------------------
# Load plant parameters
# ---------------------------------------------------------------------------
from pemwe.plant import load_plant
plant = load_plant(ROOT / "configs" / "plant_parameters.yaml")

elec  = plant["electrochemistry"]
deg   = plant["degradation"]

alpha_V = deg["alpha_V_m4_per_A2_h"]   # [V m^4 A^-2 h^-1]
beta_V  = deg["beta_V_m2_per_A_h"]     # [V m^2 A^-1 h^-1]
gamma_V = deg["gamma_V_per_h"]         # [V h^-1]
Ea      = deg["Ea_eff_J_per_mol"]      # [J mol^-1]
T_ref   = deg["T_ref_K"]              # [K]
R_gas   = elec["R_J_per_molK"]        # [J mol^-1 K^-1]

print(f"Plant degradation params:")
print(f"  alpha_V = {alpha_V:.3e}  V·m^4/(A^2·h)")
print(f"  beta_V  = {beta_V:.3e}   V·m^2/(A·h)")
print(f"  gamma_V = {gamma_V:.3e}  V/h")
print(f"  Ea_eff  = {Ea/1e3:.1f}   kJ/mol")
print(f"  T_ref   = {T_ref-273.15:.1f}  °C\n")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def arrhenius(T_K: np.ndarray) -> np.ndarray:
    """Arrhenius temperature factor relative to T_ref."""
    return np.exp(-Ea / R_gas * (1.0 / T_K - 1.0 / T_ref))


def compute_degradation(df: pd.DataFrame):
    """
    Returns (t_h, cum_actual_mV, cum_jonly_mV, fT_series, rate_actual, rate_jonly)
    from a simulation CSV.
    """
    j   = df["j_A_per_m2"].to_numpy()
    T   = df["T_stack_actual_K"].to_numpy()
    t   = df["t_h"].to_numpy()

    # time step (constant 1 h throughout)
    dt  = np.diff(t, prepend=t[0] - 1.0)   # first dt = 1 h by construction

    poly   = alpha_V * j**2 + beta_V * j + gamma_V   # [V/h]
    fT     = arrhenius(T)

    rate_actual = poly * fT     # [V/h]
    rate_jonly  = poly          # [V/h]  (f_T ≡ 1)

    cum_actual_mV = np.cumsum(rate_actual * dt) * 1e3   # [mV]
    cum_jonly_mV  = np.cumsum(rate_jonly  * dt) * 1e3   # [mV]

    return t, cum_actual_mV, cum_jonly_mV, fT, rate_actual, rate_jonly


# ---------------------------------------------------------------------------
# Load and compute for each controller
# ---------------------------------------------------------------------------
data = {}
for key, rel_path in CONTROLLERS.items():
    csv_path = RESULTS_DIR / rel_path
    df = pd.read_csv(csv_path)
    t, cum_actual, cum_jonly, fT, rate_actual, rate_jonly = compute_degradation(df)
    data[key] = {
        "t":           t,
        "cum_actual":  cum_actual,
        "cum_jonly":   cum_jonly,
        "fT":          fT,
        "rate_actual": rate_actual,
        "rate_jonly":  rate_jonly,
        "T_K":         df["T_stack_actual_K"].to_numpy(),
        "j":           df["j_A_per_m2"].to_numpy(),
    }

# ---------------------------------------------------------------------------
# Decomposition table (relative to commercial)
# ---------------------------------------------------------------------------
ref = data["load_following"]
ref_final_actual = ref["cum_actual"][-1]
ref_final_jonly  = ref["cum_jonly"][-1]

print(f"{'Controller':<14}  {'V_deg_actual':>12}  {'V_deg_jonly':>12}  "
      f"{'Total saving':>13}  {'j saving':>10}  {'T saving':>10}  "
      f"{'j share':>8}  {'T share':>8}")
print("-" * 100)

decomp_data = {}
for key in CONTROLLERS:
    d = data[key]
    final_actual = d["cum_actual"][-1]
    final_jonly  = d["cum_jonly"][-1]
    total_save   = ref_final_actual - final_actual
    j_save       = ref_final_jonly  - final_jonly
    T_save       = total_save - j_save
    j_pct        = 100 * j_save / total_save if total_save != 0 else float("nan")
    T_pct        = 100 * T_save / total_save if total_save != 0 else float("nan")

    decomp_data[key] = dict(
        final_actual=final_actual,
        final_jonly=final_jonly,
        total_save=total_save,
        j_save=j_save,
        T_save=T_save,
        j_pct=j_pct,
        T_pct=T_pct,
    )

    print(f"{LABELS[key]:<14}  {final_actual:>12.2f}  {final_jonly:>12.2f}  "
          f"{total_save:>+13.2f}  {j_save:>+10.2f}  {T_save:>+10.2f}  "
          f"{j_pct:>7.1f}%  {T_pct:>7.1f}%")

print()
print("Units: mV.  Positive saving = less degradation than commercial baseline.")
print("j-only: f_T=1 (temperature factor removed); T-effect = total − j-only.\n")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
try:
    from run.make_plots import apply_style
    apply_style()
except Exception:
    plt.rcParams.update({"font.size": 9})

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle(
    r"Degradation decomposition: $j$-effect vs temperature effect",
    fontsize=11, y=0.99,
)

ax_cum   = axes[0, 0]   # Cumulative degradation (actual)
ax_jonly = axes[0, 1]   # Cumulative degradation (j-only)
ax_fT    = axes[1, 0]   # Arrhenius factor time series
ax_bar   = axes[1, 1]   # Decomposition bar chart

# ---- (a) Cumulative degradation — actual ----
for key in CONTROLLERS:
    d = data[key]
    ax_cum.plot(d["t"] / 24, d["cum_actual"],
                color=COLORS[key], lw=1.5, label=LABELS[key])

ax_cum.set_xlabel(r"Time [days]")
ax_cum.set_ylabel(r"$\Delta V_\mathrm{deg}$ [mV]")
ax_cum.set_title(r"(a) Cumulative degradation --- actual")
ax_cum.legend(fontsize=8)
ax_cum.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

# ---- (b) Cumulative degradation — j-only (f_T = 1) ----
for key in CONTROLLERS:
    d = data[key]
    ax_jonly.plot(d["t"] / 24, d["cum_jonly"],
                  color=COLORS[key], lw=1.5, label=LABELS[key],
                  linestyle="--")

ax_jonly.set_xlabel(r"Time [days]")
ax_jonly.set_ylabel(r"$\Delta V_\mathrm{deg}$ [mV]")
ax_jonly.set_title(r"(b) Cumulative degradation --- $j$-only ($f_T \equiv 1$)")
ax_jonly.legend(fontsize=8)
ax_jonly.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

# ---- (c) Arrhenius factor f_T over time ----
for key in CONTROLLERS:
    d = data[key]
    ax_fT.plot(d["t"] / 24, d["fT"],
               color=COLORS[key], lw=1.0, alpha=0.8, label=LABELS[key])

T_range = np.linspace(303.15, 363.15, 100)
ax_fT_twin = ax_fT.twinx()
ax_fT_twin.plot([], [])  # empty — just for second y-axis label spacing

ax_fT.axhline(1.0, color="k", lw=0.6, ls=":", label=r"$f_T=1$ at $T_\mathrm{ref}$")
ax_fT.set_xlabel(r"Time [days]")
ax_fT.set_ylabel(r"Arrhenius factor $f_T$")
ax_fT.set_title(r"(c) Arrhenius factor $f_T(T_\mathrm{stack})$")
ax_fT.legend(fontsize=8)
ax_fT_twin.set_visible(False)

# ---- (d) Decomposition bar chart ----
controllers_non_ref = [k for k in CONTROLLERS if k != "load_following"]
bar_labels = [LABELS[k] for k in controllers_non_ref]
j_saves  = [decomp_data[k]["j_save"]  for k in controllers_non_ref]
T_saves  = [decomp_data[k]["T_save"]  for k in controllers_non_ref]

x = np.arange(len(controllers_non_ref))
width = 0.35

bars_j = ax_bar.bar(x - width / 2, j_saves, width,
                    label=r"$j$-effect saving [mV]",
                    color=["#4c9be8", "#4cbe72", "#e84c4c"])
bars_T = ax_bar.bar(x + width / 2, T_saves, width,
                    label=r"$T$-effect saving [mV]",
                    color=["#a0c8f0", "#a0e0b4", "#f0a0a0"])

# Annotate percentages
for bar, key in zip(bars_j, controllers_non_ref):
    pct = decomp_data[key]["j_pct"]
    if np.isfinite(pct):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{pct:.0f}\\%", ha="center", va="bottom", fontsize=7)

for bar, key in zip(bars_T, controllers_non_ref):
    pct = decomp_data[key]["T_pct"]
    if np.isfinite(pct):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{pct:.0f}\\%", ha="center", va="bottom", fontsize=7)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(bar_labels)
ax_bar.set_ylabel(r"Degradation saving vs Commercial [mV]")
ax_bar.set_title(r"(d) Decomposition: $j$-effect vs $T$-effect")
ax_bar.axhline(0, color="k", lw=0.6)
ax_bar.legend(fontsize=8)

plt.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Figure saved: {OUT_FILE}")
plt.close(fig)
