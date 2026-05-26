"""
sensitivity_T_target_dynamics.py
==================================

Shows that the aware controller's temperature setpoint is genuinely dynamic —
varying with electricity price, available power, and accumulated degradation —
rather than simply hovering at a fixed low value.

If T_target correlates with price (higher T when cheap, lower T when expensive)
and with degradation state (lower T as V_deg accumulates), the controller is
doing something no fixed setpoint can replicate.

Panels
------
(a) Scatter: electricity price vs T_target — main correlation plot
(b) Scatter: P_avail vs T_target — is it just following load?
(c) T_target vs accumulated V_deg — does it tighten over time?
(d) Time series of a representative two-week window

Run from project root:
    python3 run/sensitivity/sensitivity_T_target_dynamics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CSV = REPO_ROOT / "results" / "degradation_aware_wind_spot_dk1" / "degradation_aware_wind_spot_dk1.csv"
OUT = REPO_ROOT / "results" / "sensitivity_T_target_dynamics.png"

# Representative window: pick a fortnight with good price variation
WINDOW_START_H = 2200
WINDOW_END_H   = 2368   # ~7 days

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV)
run = df["j_A_per_m2"] > 100          # active hours only
df_r = df[run].copy()

T_tgt_C  = df_r["T_target_K"]     - 273.15
T_act_C  = df_r["T_stack_actual_K"] - 273.15
price_ct = df_r["p_elec_eur_per_kWh"] * 100   # EUR/kWh → ct/kWh
P_avail_MW = df_r["P_avail_W"] / 1e6
V_deg_mV = df_r["V_deg_V"] * 1e3
t_h      = df_r["t_h"]

# Correlations
r_price, p_price = pearsonr(price_ct, T_tgt_C)
r_pavail, p_pavail = pearsonr(P_avail_MW, T_tgt_C)
r_vdeg, p_vdeg   = pearsonr(V_deg_mV, T_tgt_C)

print(f"Pearson r (price  vs T_target): {r_price:+.3f}  (p={p_price:.2e})")
print(f"Pearson r (P_avail vs T_target): {r_pavail:+.3f}  (p={p_pavail:.2e})")
print(f"Pearson r (V_deg  vs T_target): {r_vdeg:+.3f}  (p={p_vdeg:.2e})")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(REPO_ROOT / "run"))
    from make_plots import apply_style
    apply_style()
except Exception:
    plt.rcParams.update({"font.size": 9})

BLUE  = "#1f77b4"
RED   = "#d62728"
GREEN = "#2ca02c"
GREY  = "#888888"

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

ax_price  = fig.add_subplot(gs[0, 0])
ax_pavail = fig.add_subplot(gs[0, 1])
ax_vdeg   = fig.add_subplot(gs[0, 2])
ax_ts     = fig.add_subplot(gs[1, :])   # full-width time series

fig.suptitle(
    r"Aware controller: is $T_\mathrm{target}$ genuinely dynamic?",
    fontsize=11, y=1.01,
)

# Colour points by V_deg to show evolution over time
cmap = plt.cm.plasma
norm = plt.Normalize(V_deg_mV.min(), V_deg_mV.max())
sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# ---- (a) Price vs T_target ----
ax_price.scatter(price_ct, T_tgt_C,
                 c=V_deg_mV, cmap=cmap, norm=norm,
                 s=4, alpha=0.4, rasterized=True)

# Bin-mean line
bins = np.percentile(price_ct, np.linspace(5, 95, 15))
bin_idx = np.digitize(price_ct, bins)
bin_mean_p = [price_ct[bin_idx == i].mean() for i in range(1, len(bins)+1)
              if (bin_idx == i).sum() > 5]
bin_mean_T = [T_tgt_C[bin_idx == i].mean() for i in range(1, len(bins)+1)
              if (bin_idx == i).sum() > 5]
ax_price.plot(bin_mean_p, bin_mean_T, "k-", lw=1.8, zorder=5, label="Bin mean")

ax_price.set_xlabel(r"Electricity price [ct/kWh]")
ax_price.set_ylabel(r"$T_\mathrm{target}$ [$^\circ$C]")
ax_price.set_title(
    rf"(a) Price vs $T_\mathrm{{target}}$" + "\n" +
    rf"$r = {r_price:+.2f}$"
)
ax_price.legend(fontsize=7)
cb1 = plt.colorbar(sm, ax=ax_price, shrink=0.8)
cb1.set_label(r"$\Delta V_\mathrm{deg}$ [mV]", fontsize=7)

# ---- (b) P_avail vs T_target ----
ax_pavail.scatter(P_avail_MW, T_tgt_C,
                  c=V_deg_mV, cmap=cmap, norm=norm,
                  s=4, alpha=0.4, rasterized=True)

bins_p = np.percentile(P_avail_MW, np.linspace(5, 95, 15))
bidx_p = np.digitize(P_avail_MW, bins_p)
bmp = [P_avail_MW[bidx_p == i].mean() for i in range(1, len(bins_p)+1)
       if (bidx_p == i).sum() > 5]
bmT = [T_tgt_C[bidx_p == i].mean() for i in range(1, len(bins_p)+1)
       if (bidx_p == i).sum() > 5]
ax_pavail.plot(bmp, bmT, "k-", lw=1.8, zorder=5, label="Bin mean")

ax_pavail.set_xlabel(r"Available power [MW]")
ax_pavail.set_ylabel(r"$T_\mathrm{target}$ [$^\circ$C]")
ax_pavail.set_title(
    rf"(b) Power vs $T_\mathrm{{target}}$" + "\n" +
    rf"$r = {r_pavail:+.2f}$"
)
ax_pavail.legend(fontsize=7)
cb2 = plt.colorbar(sm, ax=ax_pavail, shrink=0.8)
cb2.set_label(r"$\Delta V_\mathrm{deg}$ [mV]", fontsize=7)

# ---- (c) V_deg vs T_target ----
ax_vdeg.scatter(V_deg_mV, T_tgt_C,
                c=price_ct, cmap="RdYlGn_r",
                norm=plt.Normalize(price_ct.quantile(0.02),
                                   price_ct.quantile(0.98)),
                s=4, alpha=0.4, rasterized=True)

bins_v = np.percentile(V_deg_mV, np.linspace(5, 95, 15))
bidx_v = np.digitize(V_deg_mV, bins_v)
bmv = [V_deg_mV[bidx_v == i].mean() for i in range(1, len(bins_v)+1)
       if (bidx_v == i).sum() > 5]
bmTv = [T_tgt_C[bidx_v == i].mean() for i in range(1, len(bins_v)+1)
        if (bidx_v == i).sum() > 5]
ax_vdeg.plot(bmv, bmTv, "k-", lw=1.8, zorder=5, label="Bin mean")

ax_vdeg.set_xlabel(r"Accumulated $\Delta V_\mathrm{deg}$ [mV]")
ax_vdeg.set_ylabel(r"$T_\mathrm{target}$ [$^\circ$C]")
ax_vdeg.set_title(
    rf"(c) Degradation state vs $T_\mathrm{{target}}$" + "\n" +
    rf"$r = {r_vdeg:+.2f}$"
)
ax_vdeg.legend(fontsize=7)
sm2 = plt.cm.ScalarMappable(
    cmap="RdYlGn_r",
    norm=plt.Normalize(price_ct.quantile(0.02), price_ct.quantile(0.98))
)
sm2.set_array([])
cb3 = plt.colorbar(sm2, ax=ax_vdeg, shrink=0.8)
cb3.set_label(r"Price [ct/kWh]", fontsize=7)

# ---- (d) Time series window ----
mask_w = (df["t_h"] >= WINDOW_START_H) & (df["t_h"] <= WINDOW_END_H)
df_w   = df[mask_w]
t_days = (df_w["t_h"] - WINDOW_START_H) / 24

run_w  = df_w["j_A_per_m2"] > 100
T_tgt_w  = df_w["T_target_K"]      - 273.15
T_act_w  = df_w["T_stack_actual_K"] - 273.15
price_w  = df_w["p_elec_eur_per_kWh"] * 100

ax_ts2 = ax_ts.twinx()

# Shade shutdown hours
for i in range(len(df_w) - 1):
    if not run_w.iloc[i]:
        ax_ts.axvspan(t_days.iloc[i], t_days.iloc[i+1],
                      color="#eeeeee", alpha=0.6, lw=0)

ax_ts.plot(t_days[run_w].to_numpy(), T_tgt_w[run_w].to_numpy(), color=BLUE, lw=1.2,
           label=r"$T_\mathrm{target}$ (optimizer)")
ax_ts.plot(t_days[run_w].to_numpy(), T_act_w[run_w].to_numpy(), color=BLUE, lw=0.7,
           ls="--", alpha=0.6, label=r"$T_\mathrm{actual}$")
ax_ts.axhline(60, color=GREY, lw=0.8, ls=":", label=r"$T_\mathrm{ref}$ = 60$^\circ$C")

ax_ts2.plot(t_days.to_numpy(), price_w.to_numpy(), color=RED, lw=1.0, alpha=0.7,
            label=r"Price [ct/kWh]")
ax_ts2.axhline(0, color=RED, lw=0.5, ls=":")

ax_ts.set_xlabel(r"Time [days from hour " + str(WINDOW_START_H) + r"]")
ax_ts.set_ylabel(r"Temperature [$^\circ$C]", color=BLUE)
ax_ts2.set_ylabel(r"Price [ct/kWh]", color=RED)
ax_ts.tick_params(axis="y", labelcolor=BLUE)
ax_ts2.tick_params(axis="y", labelcolor=RED)
ax_ts.set_title(r"(d) Representative window: $T_\mathrm{target}$ tracks price"
                r" (grey = shutdown)")

lines1, labs1 = ax_ts.get_legend_handles_labels()
lines2, labs2 = ax_ts2.get_legend_handles_labels()
ax_ts.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper right", ncol=4)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {OUT}")
plt.close(fig)
