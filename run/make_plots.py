"""
make_plots.py ---Report-quality figures from a PEMWE simulation output CSV.

Usage:
  python3 run/make_plots.py --input  results/aware_runs/aware_wind.csv
  python3 run/make_plots.py --input  results/aware_runs/aware_wind.csv  --outdir results/figures
  python3 run/make_plots.py --load-following results/load_following_wind_spot_dk1/load_following_wind_spot_dk1.csv \
                             --degradation-aware results/degradation_aware_wind_spot_dk1/degradation_aware_wind_spot_dk1.csv

Figures (PNG + combined PDF):
  fig1  dynamic_overview   ---P_avail / P_stack / j / T_stack / H2 rate / price
  fig2  polarization       ---Operating-point scatter + j histogram
  fig3  efficiency         ---System efficiency (LHV/HHV) + specific energy
  fig4  economics          ---Revenue, cost, cumulative profit, monthly bar
  fig5  degradation        ---V_deg accumulation, rate vs j, lifetime projection
  fig6  thermal            ---T_actual, deviation from target, heat flows, water flow
  fig7  comparison         ---Load-following vs degradation-aware (--load-following + --degradation-aware)
  fig7b rul_comparison     ---RUL diagnostic: V_deg trajectories + T_target comparison
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from matplotlib.colors import LinearSegmentedColormap
from plot_style import apply_style, tex, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, DARK
apply_style()

# Temperature colormap: palette green (cool) → cream → orange (warm)
T_CMAP  = LinearSegmentedColormap.from_list(
    "pemwe_T", [(0.0, GREEN), (0.30, "#FFD9A8"), (0.65, "#FF9A50"), (1.0, ORANGE)], N=256
)
# Load all constants from plant_parameters.yaml (single source of truth)
import yaml as _yaml
with open(Path(__file__).resolve().parent.parent / "configs" / "plant_parameters.yaml") as _f:
    _plant = _yaml.safe_load(_f)
N_CELLS   = int(_plant["stack"]["N_cells"])
A_CELL_M2 = float(_plant["stack"]["A_cell_m2"])
U_TN      = float(_plant["thermal"]["U_tn_V"])
HHV       = float(_plant["fluids"]["hydrogen"]["HHV_kWh_per_kg"])
LHV       = float(_plant["fluids"]["hydrogen"]["LHV_kWh_per_kg"])
V_DEG_EOL_MV = float(_plant["degradation"]["V_deg_EOL_V"]) * 1000  # [mV]
_P_RATED_KW        = float(_plant["stack"]["P_rating_W"]) / 1000.0
_SYS_CAPEX_EUR     = float(_plant["economics"]["system_capex_eur_per_kW"]) * _P_RATED_KW
_SYS_LIFETIME_YR   = float(_plant["economics"]["system_lifetime_yr"])
_WACC              = float(_plant["economics"].get("wacc", 0.0))
# Annuity factor A(r,N) = (1-(1+r)^-N)/r; falls back to flat N when wacc=0
_ANNUITY = (1 - (1 + _WACC) ** -_SYS_LIFETIME_YR) / _WACC if _WACC > 0 else _SYS_LIFETIME_YR
# Stack replacement cost [EUR] for NPV-based replacement LCOH (Eq. 20)
_STACK_REPL_EUR    = (float(_plant["economics"]["capex_usd_per_kW"]) *
                      float(_plant["economics"]["eur_per_usd"]) * _P_RATED_KW)


def _system_lcoh_per_kg(h2_annual_kg: float) -> float:
    """System (BOP) CAPEX contribution to LCOH [EUR/kg], amortised using WACC annuity factor."""
    if h2_annual_kg <= 0:
        return 0.0
    return _SYS_CAPEX_EUR / (h2_annual_kg * _ANNUITY)


def _replacement_lcoh_per_kg(h2_annual_kg: float, lifetime_yr: float) -> float:
    """NPV-discounted stack replacement cost per kg H2 (Eq. 20: CRF × NPV_rep / h2_annual)."""
    if h2_annual_kg <= 0 or lifetime_yr <= 0 or not np.isfinite(lifetime_yr):
        return 0.0
    times = np.arange(lifetime_yr, _SYS_LIFETIME_YR + lifetime_yr, lifetime_yr)
    times = times[times <= _SYS_LIFETIME_YR]
    if len(times) == 0:
        return 0.0
    npv = sum(_STACK_REPL_EUR / (1 + _WACC) ** t for t in times)
    return npv / (h2_annual_kg * _ANNUITY)  # CRF = 1/annuity

# ── Data helpers ─────────────────────────────────────────────────────────────
def col(df, name):
    """Always return a plain numpy array."""
    return np.asarray(df[name], dtype=float)

def kW(df, c):   return col(df,c)/1e3
def degC(df, c): return col(df,c)-273.15
def mV(df, c):   return col(df,c)*1e3
def uVh(df, c):  return col(df,c)*1e6

def mask(df, cond):
    """Boolean-mask a DataFrame, return new DataFrame."""
    return df[cond].reset_index(drop=True)

def annual(df):  return col(df,"t_h").max() > 100*24

def month_ticks(ax, max_h):
    if max_h > 8760 * 1.5:
        # Multi-year simulation: one tick per year
        n_years = int(round(max_h / 8760))
        tks = [i * 8760 for i in range(n_years + 1) if i * 8760 <= max_h]
        ax.set_xticks(tks)
        ax.set_xticklabels([str(i) for i in range(len(tks))])
        ax.set_xlabel("Time [yr]")
    else:
        lbl=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        tks=[i*730 for i in range(12)]
        ok =[(h,lbl[i]) for i,h in enumerate(tks) if h<=max_h]
        if len(ok)>2:
            ax.set_xticks([h for h,_ in ok])
            ax.set_xticklabels([l for _,l in ok])
        else:
            ax.set_xlabel("Time [h]")

def msum(df, c):
    """Monthly average per year (730 h buckets), scaled by dt_h.

    For multi-year runs the time axis wraps modulo 8760 so each year's data
    folds into the same 12 buckets; the result is then divided by the number
    of simulated years to give a per-year average.
    """
    t = col(df,"t_h")
    dt_h = float(t[1]-t[0]) if len(t)>1 else 1.0
    t_mod = t % 8760
    m = np.clip(t_mod // 730, 0, 11).astype(int)
    n_years = max(t.max() / 8760, 1.0)
    totals = np.array([np.nansum(col(df,c)[m==i]) * dt_h for i in range(12)])
    return totals / n_years

def savefig(fig, path):
    path = Path(path)
    fig.savefig(path, bbox_inches="tight")
    pdf = path.with_suffix(".pdf")
    try:
        fig.savefig(pdf, bbox_inches="tight")
    except Exception as e:
        print(f"  [PDF skipped for {path.name}: {e}]")
    print(f"  Saved: {path.name}")

# ── Fig 1 ---Dynamic overview ─────────────────────────────────────────────────
def fig_dynamic_overview(df, name, outdir):
    t=col(df,"t_h"); ann=annual(df)
    fig,axes=plt.subplots(5,1,figsize=(12,14),sharex=True)
    fig.suptitle(f"Dynamic Operation --- {tex(name)}",fontsize=13,fontweight="bold")

    ax=axes[0]
    ax.fill_between(t,kW(df,"P_avail_W"),alpha=0.2,color=BLUE)
    ax.plot(t,kW(df,"P_avail_W"), color=BLUE,  lw=0.8,label=r"$P_\mathrm{avail}$")
    ax.plot(t,kW(df,"P_total_W"), color=ORANGE,lw=1.0,label=r"$P_\mathrm{total}$")
    ax.plot(t,kW(df,"P_stack_W"), color=GREEN, lw=0.8,label=r"$P_\mathrm{stack}$",ls="--")
    ax.set_ylabel("Power [kW]"); ax.legend(loc="upper right",ncol=3)

    axes[1].plot(t,col(df,"j_A_per_m2")/10000,color=PURPLE,lw=0.8)
    axes[1].set_ylabel(r"$j$ [A/cm$^2$]")

    axes[2].plot(t,degC(df,"T_stack_actual_K"),color=RED,   lw=0.9,label="T actual")
    axes[2].axhline(60,color=GREY,lw=0.7,ls=":"); axes[2].set_ylabel(r"$T_\mathrm{stack}$ [$^\circ$C]"); axes[2].legend()

    axes[3].plot(t,col(df,"m_dot_H2_kg_h"),color=GREEN,lw=0.8)
    axes[3].set_ylabel(r"H$_2$ rate [kg/h]")

    axes[4].plot(t,col(df,"p_elec_eur_per_kWh")*1000,color=BLUE,lw=0.7)
    axes[4].axhline(0,color=GREY,lw=0.6,ls="--"); axes[4].set_ylabel(r"Spot price [\texteuro/MWh]")

    if ann: month_ticks(axes[4],t.max())
    else:   axes[4].set_xlabel("Time [h]")
    fig.tight_layout(); savefig(fig,outdir/f"{name}_fig1_dynamic_overview.png"); return fig

# ── Fig 1b ---Zoomed detail view (48 h window with most variable operation) ───
def fig_dynamic_zoom(df, name, outdir):
    """Pick a 48-hour window with high power variability and plot detail."""
    t = col(df, "t_h")
    dt_h = float(t[1] - t[0]) if len(t) > 1 else 1.0
    p_avail = col(df, "P_avail_W")

    # Find 48-h window with highest std dev in P_avail (most interesting control)
    win = int(48 / dt_h)
    if len(df) < win:
        return None
    stds = np.array([p_avail[i:i+win].std() for i in range(0, len(df) - win, max(win//4, 1))])
    best_idx = int(np.argmax(stds)) * max(win//4, 1)
    sl = slice(best_idx, best_idx + win)
    d = df.iloc[sl].reset_index(drop=True)
    t_d = col(d, "t_h")

    _LW  = 1.8   # main line width
    _LW2 = 2.2   # emphasis line width
    _FS  = 20    # axis label / legend font size

    fig, axes = plt.subplots(6, 1, figsize=(13, 15), sharex=True)

    # Panel 0 — Wind power (external input)
    ax = axes[0]
    ax.fill_between(t_d, kW(d, "P_avail_W"), alpha=0.2, color=BLUE)
    ax.plot(t_d, kW(d, "P_avail_W"), color=BLUE,   lw=_LW,  label=r"$P_\mathrm{avail}$")
    ax.plot(t_d, kW(d, "P_total_W"), color=ORANGE, lw=_LW2, label=r"$P_\mathrm{total}$")
    ax.plot(t_d, kW(d, "P_stack_W"), color=GREEN,  lw=_LW,  ls="--", label=r"$P_\mathrm{stack}$")
    ax.set_ylabel("Power [kW]", fontsize=_FS); ax.legend(loc="center right", ncol=1, fontsize=_FS)

    # Panel 1 — Spot price (external input)
    axes[1].plot(t_d, col(d, "p_elec_eur_per_kWh") * 1000, color=BLUE, lw=_LW)
    axes[1].axhline(0, color=GREY, lw=0.8, ls="--")
    axes[1].set_ylabel(r"$p_{elec}$ [\texteuro/MWh]", fontsize=_FS)

    # Panel 2 — Current density (controlled)
    axes[2].plot(t_d, col(d, "j_A_per_m2") / 10000, color=PURPLE, lw=_LW)
    axes[2].set_ylabel(r"$j$ [A/cm$^2$]", fontsize=_FS)

    # Panel 3 — Stack temperature (controlled)
    T_act = degC(d, "T_stack_actual_K")
    axes[3].plot(t_d, T_act, color=RED, lw=_LW2, label=r"$T$ (end of hour)")
    if "T_stack_max_K" in d.columns:
        T_mx = degC(d, "T_stack_max_K")
        axes[3].fill_between(t_d, T_act, T_mx, color=RED, alpha=0.20, label=r"$T$ range (within hour)")
    axes[3].axhline(60, color=GREY, lw=0.8, ls=":")
    axes[3].set_ylabel(r"$T$ [$^\circ$C]", fontsize=_FS)
    axes[3].legend(loc="center right", fontsize=_FS)

    # Panel 4 — Water management
    has_tw = "T_w_in_K" in d.columns and "T_mix_K" in d.columns
    if has_tw:
        axes[4].plot(t_d, degC(d, "T_w_in_K"), color=BLUE,  lw=_LW, label=r"$T_{w,\mathrm{in}}$")
        axes[4].plot(t_d, degC(d, "T_mix_K"),  color=GREEN, lw=_LW, ls="--", label=r"$T_\mathrm{mix}$")
        axes[4].axhline(60, color=GREY, lw=0.8, ls=":")
        axes[4].set_ylabel(r"$T_\mathrm{water}$ [$^\circ$C]", fontsize=_FS)
        axes[4].legend(ncol=2, fontsize=_FS)
    else:
        axes[4].plot(t_d, col(d, "m_dot_w_kg_s"), color=BLUE, lw=_LW)
        axes[4].set_ylabel(r"$\dot{m}_w$ [kg/s]", fontsize=_FS)

    # Panel 5 — H₂ production (output)
    axes[5].plot(t_d, col(d, "m_dot_H2_kg_h"), color=GREEN, lw=_LW)
    axes[5].set_ylabel(r"$\dot{H}_2$ [kg/h]", fontsize=_FS)
    axes[5].set_xlabel("Time [h]", fontsize=_FS)

    for ax in axes:
        ax.tick_params(labelsize=_FS - 1)

    fig.tight_layout(); savefig(fig, outdir / f"{name}_fig1b_detail_48h.png"); return fig


def fig_detail_10h(df, name, outdir):
    """10-hour detail ---zoomed view of startup and thermal dynamics."""
    t = col(df, "t_h")
    dt_h = float(t[1] - t[0]) if len(t) > 1 else 1.0
    p_avail = col(df, "P_avail_W")
    j = col(df, "j_A_per_m2")

    # Find a 10-h window. If sub-step CSV exists, use its time range.
    win = int(10 / dt_h)
    if len(df) < win:
        return None

    sub_csv = outdir / f"{name}_substeps.csv"
    best_idx = None

    if sub_csv.exists():
        # Use the sub-step CSV's time range to select the window
        sub_tmp = pd.read_csv(sub_csv)
        t_sub_start = sub_tmp["t_h"].min()
        # Find the hourly index closest to the sub-step start
        t_arr = col(df, "t_h")
        best_idx = int(np.argmin(np.abs(t_arr - t_sub_start)))
        best_idx = max(0, min(best_idx, len(df) - win))

    if best_idx is None:
        # Auto-detect: startup + high-T operation
        T_max_col = col(df, "T_stack_max_K") if "T_stack_max_K" in df.columns else None
        starts = [i for i in range(1, len(j)) if j[i-1] <= 100 and j[i] > 100]
        if not starts:
            return None

        if T_max_col is not None:
            for si in starts:
                window_end = min(si + win, len(df))
                if np.any(T_max_col[si:window_end] > 352.15):
                    best_idx = max(0, si - win // 6)
                    best_idx = min(best_idx, len(df) - win)
                    break

        if best_idx is None:
            mid_start = starts[len(starts) // 2]
            best_idx = max(0, mid_start - win // 4)
            best_idx = min(best_idx, len(df) - win)

    sl = slice(best_idx, best_idx + win)
    d = df.iloc[sl].reset_index(drop=True)
    t_d = col(d, "t_h")

    _LW  = 1.8
    _LW2 = 2.2
    _FS  = 20

    fig, axes = plt.subplots(6, 1, figsize=(13, 15), sharex=True)

    # Panel 0 — Wind power (external input)
    ax = axes[0]
    ax.fill_between(t_d, kW(d, "P_avail_W"), alpha=0.2, color=BLUE)
    ax.plot(t_d, kW(d, "P_avail_W"), color=BLUE,   lw=_LW,  label=r"$P_\mathrm{avail}$")
    ax.plot(t_d, kW(d, "P_total_W"), color=ORANGE, lw=_LW2, label=r"$P_\mathrm{total}$")
    ax.plot(t_d, kW(d, "P_stack_W"), color=GREEN,  lw=_LW,  ls="--", label=r"$P_\mathrm{stack}$")
    ax.set_ylabel("Power [kW]", fontsize=_FS); ax.legend(loc="center right", ncol=1, fontsize=_FS)

    # Panel 1 — Spot price (external input)
    axes[1].plot(t_d, col(d, "p_elec_eur_per_kWh") * 1000, color=BLUE, lw=_LW)
    axes[1].axhline(0, color=GREY, lw=0.8, ls="--")
    axes[1].set_ylabel(r"Spot price [\texteuro/MWh]", fontsize=_FS)

    # Panel 2 — Current density (controlled)
    axes[2].plot(t_d, col(d, "j_A_per_m2") / 10000, color=PURPLE, lw=_LW)
    axes[2].set_ylabel(r"$j$ [A/cm$^2$]", fontsize=_FS)

    # Panel 3 — Stack temperature (controlled)
    T_act = degC(d, "T_stack_actual_K")
    axes[3].plot(t_d, T_act, color=RED, lw=_LW2, label=r"$T$ (end of hour)")
    if "T_stack_max_K" in d.columns:
        T_mx = degC(d, "T_stack_max_K")
        axes[3].fill_between(t_d, T_act, T_mx, color=RED, alpha=0.20, label=r"$T$ range (within hour)")
    axes[3].axhline(60, color=GREY, lw=0.8, ls=":")
    axes[3].set_ylabel(r"$T_\mathrm{stack}$ [$^\circ$C]", fontsize=_FS)
    axes[3].legend(loc="center right", fontsize=_FS)

    # Panel 4 — Water management
    has_tw = "T_w_in_K" in d.columns and "T_mix_K" in d.columns
    if has_tw:
        axes[4].plot(t_d, degC(d, "T_w_in_K"), color=BLUE,  lw=_LW, label=r"$T_{w,\mathrm{in}}$")
        axes[4].plot(t_d, degC(d, "T_mix_K"),  color=GREEN, lw=_LW, ls="--", label=r"$T_\mathrm{mix}$")
        axes[4].axhline(60, color=GREY, lw=0.8, ls=":")
        axes[4].set_ylabel(r"$T_\mathrm{water}$ [$^\circ$C]", fontsize=_FS)
        axes[4].legend(ncol=2, fontsize=_FS)
    else:
        axes[4].plot(t_d, col(d, "m_dot_w_kg_s"), color=BLUE, lw=_LW)
        axes[4].set_ylabel(r"$\dot{m}_w$ [kg/s]", fontsize=_FS)

    # Panel 5 — H₂ production (output)
    axes[5].plot(t_d, col(d, "m_dot_H2_kg_h"), color=GREEN, lw=_LW)
    axes[5].set_ylabel(r"H$_2$ rate [kg/h]", fontsize=_FS)
    axes[5].set_xlabel("Time [h]", fontsize=_FS)

    for ax in axes:
        ax.tick_params(labelsize=_FS - 1)

    fig.tight_layout(); savefig(fig, outdir / f"{name}_fig1c_detail_10h.png"); return fig


# ── Fig 2 ---Polarization / operating-point scatter ───────────────────────────
def fig_polarization(df, name, outdir):
    a=mask(df, col(df,"j_A_per_m2")>100)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    fig.suptitle(f"Operating Points --- {tex(name)}",fontsize=13,fontweight="bold")

    sc=axes[0].scatter(col(a,"j_A_per_m2")/10000,kW(a,"P_stack_W"),
        c=degC(a,"T_stack_actual_K"),cmap=T_CMAP,s=4,alpha=0.85,rasterized=True)
    fig.colorbar(sc,ax=axes[0]).set_label(r"$T_\mathrm{stack}$ [$^\circ$C]")
    axes[0].set_xlabel(r"$j$ [A/cm$^2$]"); axes[0].set_ylabel(r"$P_\mathrm{stack}$ [kW]")
    axes[0].set_title("Power vs Current Density")

    j_all=col(a,"j_A_per_m2")/10000
    axes[1].hist(j_all,bins=np.linspace(0,j_all.max(),40),color=BLUE,edgecolor="white",lw=0.3)
    n_idle=int((col(df,"j_A_per_m2")<=100).sum())
    axes[1].annotate(rf"Idle: {100*n_idle/len(df):.1f}\% of hours",
        xy=(0.98,0.95),xycoords="axes fraction",ha="right",va="top",fontsize=9,color=GREY)
    axes[1].set_xlabel(r"$j$ [A/cm$^2$]"); axes[1].set_ylabel("Hours")
    axes[1].set_title("Current Density Distribution")

    fig.tight_layout(); savefig(fig,outdir/f"{name}_fig2_polarization.png"); return fig

# ── Fig 3 ---Efficiency ────────────────────────────────────────────────────────
def _eff(a):
    """Return (eta_system_HHV, eta_stack_HHV, V_cell, U_tn) arrays for active rows."""
    h2  = col(a, "m_dot_H2_kg_h")
    P_t = col(a, "P_total_W")
    P_s = col(a, "P_stack_W")
    j   = col(a, "j_A_per_m2")
    T   = col(a, "T_stack_actual_K")

    # System efficiency: total input vs H2 HHV
    eta_sys = (h2 * HHV * 1e3) / np.where(P_t > 0, P_t, np.nan)

    # Stack HHV efficiency: stack power only vs H2 HHV
    eta_stk = (h2 * HHV * 1e3) / np.where(P_s > 0, P_s, np.nan)

    # Cell voltage: V_cell = P_stack / (N_cells × I)
    I_stack = j * A_CELL_M2          # [A]
    V_cell  = P_s / np.where(N_CELLS * I_stack > 0, N_CELLS * I_stack, np.nan)

    # Voltage efficiency (HHV-based): U_tn / V_cell
    eta_V = U_TN / np.where(V_cell > 0, V_cell, np.nan)

    return eta_sys, eta_stk, eta_V, V_cell


def fig_efficiency(df, name, outdir):
    a   = mask(df, col(df,"P_total_W") > 100)
    ann = annual(df)
    t_a = col(a, "t_h")
    eta_sys, eta_stk, eta_V, V_cell = _eff(a)
    SEC = col(a,"P_total_W")/1e3 / np.where(col(a,"m_dot_H2_kg_h")>0, col(a,"m_dot_H2_kg_h"), np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Efficiency Breakdown --- {tex(name)}", fontsize=13, fontweight="bold")

    # ── Row 0: time series ────────────────────────────────────────────────────
    axes[0,0].plot(t_a, eta_sys*100, color=BLUE,   lw=0.7, label=r"$\eta_\mathrm{system}$ (HHV)")
    axes[0,0].plot(t_a, eta_stk*100, color=GREEN,  lw=0.7, label=r"$\eta_\mathrm{stack}$ (HHV)", ls="--")
    axes[0,0].plot(t_a, eta_V*100,   color=ORANGE, lw=0.7, label=r"$\eta_\mathrm{voltage}$",     ls=":")
    axes[0,0].set_ylabel(r"$\eta$ [\%]"); axes[0,0].set_title("Efficiency Over Time")
    axes[0,0].legend(fontsize=8)
    if ann: month_ticks(axes[0,0], col(df,"t_h").max())
    else:   axes[0,0].set_xlabel("Time [h]")

    axes[0,1].plot(t_a, np.where(V_cell > 0, V_cell, np.nan),
                   color=RED, lw=0.7, label=r"$V_\mathrm{cell}$")
    axes[0,1].axhline(U_TN, color=GREY, lw=0.8, ls="--", label=rf"$U_\mathrm{{tn}}$ = {U_TN} V")
    axes[0,1].set_ylabel("Voltage [V]"); axes[0,1].set_title(r"Cell Voltage vs $U_\mathrm{tn}$")
    axes[0,1].legend(fontsize=8)
    if ann: month_ticks(axes[0,1], col(df,"t_h").max())
    else:   axes[0,1].set_xlabel("Time [h]")

    # Aux overhead: fraction of total power consumed by auxiliaries
    aux_frac = col(a,"P_aux_W") / np.where(col(a,"P_total_W")>0, col(a,"P_total_W"), np.nan)
    axes[0,2].plot(t_a, aux_frac*100, color=PURPLE, lw=0.7)
    axes[0,2].set_ylabel(r"$P_\mathrm{aux} / P_\mathrm{total}$ [\%]"); axes[0,2].set_title("Auxiliary Power Overhead")
    if ann: month_ticks(axes[0,2], col(df,"t_h").max())
    else:   axes[0,2].set_xlabel("Time [h]")

    # ── Row 1: scatter vs operating point ────────────────────────────────────
    j_cm2 = col(a,"j_A_per_m2") / 10000
    T_C   = degC(a, "T_stack_actual_K")

    sc = axes[1,0].scatter(j_cm2, eta_sys*100,
                           c=T_C, cmap=T_CMAP, s=5, alpha=0.85, rasterized=True)
    fig.colorbar(sc, ax=axes[1,0]).set_label(r"$T_\mathrm{stack}$ [$^\circ$C]")
    axes[1,0].set_xlabel(r"$j$ [A/cm$^2$]"); axes[1,0].set_ylabel(r"$\eta_\mathrm{sys,HHV}$ [\%]")
    axes[1,0].set_title("System Efficiency vs j")

    sc2 = axes[1,1].scatter(j_cm2, np.where(V_cell>0,V_cell,np.nan),
                            c=T_C, cmap=T_CMAP, s=5, alpha=0.85, rasterized=True)
    fig.colorbar(sc2, ax=axes[1,1]).set_label(r"$T_\mathrm{stack}$ [$^\circ$C]")
    axes[1,1].set_xlabel(r"$j$ [A/cm$^2$]"); axes[1,1].set_ylabel(r"$V_\mathrm{cell}$ [V]")
    axes[1,1].set_title("Cell Voltage vs Current Density")

    sec = np.clip(SEC, 0, 100)
    axes[1,2].hist(sec[~np.isnan(sec)], bins=40, color=ORANGE, edgecolor="white", lw=0.3)
    mn = float(np.nanmean(sec))
    axes[1,2].axvline(mn, color=RED, lw=1.2, ls="--", label=f"Mean: {mn:.1f} kWh/kg")
    axes[1,2].set_xlabel(r"SEC [kWh/kg H$_2$]"); axes[1,2].set_ylabel("Hours")
    axes[1,2].set_title("Specific Energy Consumption"); axes[1,2].legend()

    fig.tight_layout()
    savefig(fig, outdir/f"{name}_fig3_efficiency.png")
    return fig

# ── Fig 4 ---Economics ─────────────────────────────────────────────────────────
def fig_economics(df, name, outdir):
    ann=annual(df); t=col(df,"t_h")
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    fig.suptitle(f"Economics --- {tex(name)}",fontsize=13,fontweight="bold")

    axes[0,0].plot(t,col(df,"r_H2_eur_h"),      color=GREEN, lw=0.7,label=r"H$_2$ revenue")
    axes[0,0].plot(t,col(df,"c_elec_eur_h"),     color=RED,   lw=0.7,label="Electricity cost")
    axes[0,0].plot(t,col(df,"true_profit_eur_h"),color=BLUE,  lw=0.9,label="Net profit")
    axes[0,0].axhline(0,color=GREY,lw=0.5,ls="--")
    axes[0,0].set_ylabel(r"[\texteuro/h]"); axes[0,0].set_title(r"Hourly Revenue \& Cost"); axes[0,0].legend(ncol=3)
    if ann: month_ticks(axes[0,0],t.max())
    else:   axes[0,0].set_xlabel("Time [h]")

    dt_h_ec = float(t[1]-t[0]) if len(t)>1 else 1.0
    axes[0,1].plot(t,np.nancumsum(col(df,"true_profit_eur_h")*dt_h_ec),color=BLUE,lw=1.2)
    axes[0,1].axhline(0,color=GREY,lw=0.5,ls="--")
    axes[0,1].set_ylabel(r"Cum. profit [\texteuro]"); axes[0,1].set_title("Cumulative Profit")
    if ann: month_ticks(axes[0,1],t.max())
    else:   axes[0,1].set_xlabel("Time [h]")

    if ann:
        m_h2=msum(df,"r_H2_eur_h"); m_el=msum(df,"c_elec_eur_h"); m_dg=msum(df,"c_deg_phys_eur_h")
        x=np.arange(12); lbl=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        axes[1,0].bar(x,m_h2,        label=r"H$_2$ revenue",    color=GREEN, alpha=0.85)
        axes[1,0].bar(x,-m_el,        label="Elec cost ($-$)", color=RED,   alpha=0.85)
        axes[1,0].bar(x,-m_dg,bottom=-m_el,label="Degr. cost ($-$)",color=ORANGE,alpha=0.85)
        axes[1,0].axhline(0,color=GREY,lw=0.7)
        axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(lbl)
        n_yr = t.max() / 8760
        mo_title = "Monthly Breakdown (annual avg)" if n_yr > 1.1 else "Monthly Breakdown"
        axes[1,0].set_ylabel(r"[\texteuro/month]"); axes[1,0].set_title(mo_title); axes[1,0].legend(fontsize=8)
    else:
        axes[1,0].text(0.5,0.5,"Monthly view:\nannual runs only",
            ha="center",va="center",transform=axes[1,0].transAxes,color=GREY)
        axes[1,0].set_title("Monthly Breakdown")

    dt_h = float(t[1]-t[0]) if len(t)>1 else 1.0
    cum=(col(df,"m_dot_H2_kg_h")*dt_h).cumsum()
    axes[1,1].plot(t,cum,color=GREEN,lw=1.2)
    axes[1,1].set_ylabel(r"Cum. H$_2$ [kg]"); axes[1,1].set_title(rf"Cumulative H$_2$  (total: {float(cum[-1]):.2f} kg)")
    if ann: month_ticks(axes[1,1],t.max())
    else:   axes[1,1].set_xlabel("Time [h]")

    fig.tight_layout(); savefig(fig,outdir/f"{name}_fig4_economics.png"); return fig

# ── Fig 5 ---Degradation ───────────────────────────────────────────────────────
def fig_degradation(df, name, outdir):
    ann=annual(df); t=col(df,"t_h")
    a=mask(df, col(df,"j_A_per_m2")>100)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    fig.suptitle(f"Stack Degradation --- {tex(name)}",fontsize=13,fontweight="bold")

    axes[0].plot(t,mV(df,"V_deg_V"),color=RED,lw=1.2)
    axes[0].axhline(V_DEG_EOL_MV,color=GREY,lw=0.8,ls="--",label=f"EOL = {V_DEG_EOL_MV:.0f} mV")
    axes[0].set_ylabel(r"$V_\mathrm{deg}$ [mV]"); axes[0].set_title("Cumulative Degradation"); axes[0].legend()
    if ann: month_ticks(axes[0],t.max())
    else:   axes[0].set_xlabel("Time [h]")

    sc=axes[1].scatter(col(a,"j_A_per_m2")/10000,uVh(a,"dV_deg_V"),
        c=degC(a,"T_stack_actual_K"),cmap=T_CMAP,s=5,alpha=0.85,rasterized=True)
    fig.colorbar(sc,ax=axes[1]).set_label(r"$T_\mathrm{stack}$ [$^\circ$C]")
    axes[1].set_xlabel(r"$j$ [A/cm$^2$]"); axes[1].set_ylabel(r"$\mathrm{d}V/\mathrm{d}t$ [$\mu$V/h]")
    axes[1].set_title("Degradation Rate vs Operating Point")

    total_h=t.max(); total_mV=col(df,"dV_deg_V").sum()*1000  # cumulative, robust to EOL resets
    n_active = (col(df,"j_A_per_m2") > 100).sum()
    dt_h_loc = t[1] - t[0] if len(t) > 1 else 1.0
    op_h = n_active * dt_h_loc
    rate_op     = total_mV * 1000 / max(op_h, 1)
    r_ann       = total_mV / max(total_h / 8760, 1e-9)
    life_op_kh  = V_DEG_EOL_MV * 1000 / rate_op / 1000 if rate_op > 0 else np.inf
    life_cal_yr = V_DEG_EOL_MV / r_ann if r_ann > 0 else np.inf
    t_pr = np.array([0, min(life_cal_yr, total_h/8760 + 5) * 8760])
    axes[2].plot(t/8760,mV(df,"V_deg_V"),color=RED,lw=1.5,label="Simulated")
    axes[2].plot(t_pr/8760,r_ann/8760*t_pr,color=RED,lw=1.0,ls="--",label="Linear extrapolation")
    axes[2].axhline(V_DEG_EOL_MV,color=GREY,lw=0.8,ls=":",label=f"EOL = {V_DEG_EOL_MV:.0f} mV")
    axes[2].set_xlabel("Years"); axes[2].set_ylabel(r"$V_\mathrm{deg}$ [mV]")
    axes[2].set_title(rf"Cal.\ lifetime: {life_cal_yr:.1f} yr  $|$  Op.\ lifetime: {life_op_kh:.0f} kh$_{{\rm op}}$  ({rate_op:.2f} $\mu$V/op.h)"); axes[2].legend(fontsize=8)

    fig.tight_layout(); savefig(fig,outdir/f"{name}_fig5_degradation.png"); return fig

# ── Fig 6 ---Thermal ───────────────────────────────────────────────────────────
def fig_thermal(df, name, outdir):
    ann=annual(df); t=col(df,"t_h")
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    fig.suptitle(f"Thermal Dynamics --- {tex(name)}",fontsize=13,fontweight="bold")

    axes[0,0].plot(t,degC(df,"T_stack_actual_K"),color=RED,   lw=0.8,label="T actual")
    axes[0,0].axhline(60,color=GREY,lw=0.6,ls=":")
    axes[0,0].set_ylabel(r"[$^\circ$C]"); axes[0,0].set_title("Stack Temperature"); axes[0,0].legend()
    if ann: month_ticks(axes[0,0],t.max())
    else:   axes[0,0].set_xlabel("Time [h]")

    # Temperature deviation from target (60 degC)
    err = degC(df,"T_stack_actual_K") - 60.0
    axes[0,1].plot(t,err,color=PURPLE,lw=0.7); axes[0,1].axhline(0,color=GREY,lw=0.6,ls="--")
    axes[0,1].set_ylabel(r"$T - 60\,^\circ$C"); axes[0,1].set_title("Temperature Deviation from Target")
    if ann: month_ticks(axes[0,1],t.max())
    else:   axes[0,1].set_xlabel("Time [h]")

    # Split Q_water into cooling (positive) and heating (negative = water heats stack)
    Q_water_raw = col(df,"Q_water_W")
    Q_water_pos = np.maximum(Q_water_raw, 0.0) / 1e3  # kW, water removes heat
    Q_heat_neg = np.maximum(-Q_water_raw, 0.0) / 1e3  # kW, water adds heat to stack
    axes[1,0].plot(t,kW(df,"Q_gen_W"), color=RED, lw=0.8,label=r"$\dot{Q}_\mathrm{gen}$")
    axes[1,0].plot(t,Q_water_pos,       color=BLUE,lw=0.8,label=r"$\dot{Q}_\mathrm{water}$")
    axes[1,0].plot(t,Q_heat_neg,       color=ORANGE,lw=0.8,label=r"$\dot{Q}_\mathrm{cool}$")
    axes[1,0].plot(t,kW(df,"Q_amb_W"), color=GREY,lw=0.7,label=r"$\dot{Q}_\mathrm{amb}$",ls="--")
    axes[1,0].set_ylabel("Heat flow [kW]"); axes[1,0].set_title("Thermal Power Balance"); axes[1,0].legend(ncol=4)
    if ann: month_ticks(axes[1,0],t.max())
    else:   axes[1,0].set_xlabel("Time [h]")

    # Water flow vs current density (scatter, active hours only)
    a = mask(df, col(df,"j_A_per_m2") > 100)
    j_cm2 = col(a,"j_A_per_m2") / 10000
    m_dot = col(a,"m_dot_w_kg_s")
    T_C = degC(a,"T_stack_actual_K")
    sc = axes[1,1].scatter(j_cm2, m_dot, c=T_C, cmap=T_CMAP, s=4, alpha=0.85, rasterized=True)
    fig.colorbar(sc, ax=axes[1,1]).set_label(r"$T_\mathrm{stack}$ [$^\circ$C]")
    # Mark thermoneutral current density (where V_cell = U_tn, Q_gen = 0)
    # Find j where Q_gen crosses zero from the data
    Q_gen_arr = col(a, "Q_gen_W")
    if len(Q_gen_arr) > 0 and np.any(Q_gen_arr > 0) and np.any(Q_gen_arr <= 0):
        # Sort by j to find crossing
        sort_idx = np.argsort(j_cm2)
        j_sorted = j_cm2[sort_idx]
        Q_sorted = Q_gen_arr[sort_idx]
        cross = np.where(np.diff(np.sign(Q_sorted)))[0]
        if len(cross) > 0:
            j_tn = j_sorted[cross[0]]
            axes[1,1].axvline(j_tn, color=GREY, ls="--", lw=0.8, label=f"Thermoneutral (j={j_tn:.2f})")
            axes[1,1].legend(fontsize=7)
    axes[1,1].set_xlabel(r"$j$ [A/cm$^2$]"); axes[1,1].set_ylabel("Water flow [kg/s]")
    axes[1,1].set_title("Water Flow vs Current Density")

    fig.tight_layout(); savefig(fig,outdir/f"{name}_fig6_thermal.png"); return fig

# ── Fig 7a ---Single-controller summary stats table ────────────────────────────
def fig_stats_table(df, name, outdir):
    """Summary statistics table for a single controller run."""
    a = mask(df, col(df,"j_A_per_m2") > 100)
    t = col(df,"t_h"); dt_h = float(t[1]-t[0]) if len(t)>1 else 1.0
    V_deg_mV    = mV(df,"V_deg_V")[-1]                        # current stack state (for display)
    total_dV_mV = col(df,"dV_deg_V").sum() * 1000             # cumulative, robust to EOL resets
    n_active = len(a)
    cap_f = n_active / max(len(df),1)
    op_h = n_active * dt_h
    total_h = t.max()
    rate_op     = total_dV_mV * 1000 / max(op_h, 1)                        # µV/op.h
    life_op_h   = V_DEG_EOL_MV * 1000 / rate_op if rate_op > 0 else np.inf # total op.h BOL→EOL
    life_op_kh  = life_op_h / 1000
    life_op_yr  = life_op_h / 8760
    r_ann       = total_dV_mV / max(total_h / 8760, 1e-9)                  # mV/yr (cumulative)
    life_cal_yr = V_DEG_EOL_MV / r_ann if r_ann > 0 else np.inf            # calendar yr BOL→EOL

    P_act = col(a,"P_total_W"); P_stack_act = col(a,"P_stack_W")
    H2_act = col(a,"m_dot_H2_kg_h")
    # Energy-weighted efficiency (total energy out / total energy in)
    E_H2 = np.nansum(H2_act * dt_h) * HHV * 1000   # total H2 energy [Wh]
    eta_sys = E_H2 / np.nansum(P_act * dt_h) if np.nansum(P_act * dt_h) > 0 else 0.0
    eta_stack = E_H2 / np.nansum(P_stack_act * dt_h) if np.nansum(P_stack_act * dt_h) > 0 else 0.0
    SEC = np.nansum(P_act/1000 * dt_h) / np.nansum(H2_act * dt_h) if np.nansum(H2_act * dt_h) > 0 else 0.0

    total_H2 = col(df,"m_dot_H2_kg_h").sum() * dt_h
    total_elec = col(df,"c_elec_eur_h").sum() * dt_h
    total_deg = col(df,"c_deg_phys_eur_h").sum() * dt_h
    total_shutdown = col(df,"c_shutdown_eur").sum()
    total_rev = col(df,"r_H2_eur_h").sum() * dt_h
    profit = col(df,"true_profit_eur_h").sum() * dt_h
    sim_yr       = total_h / 8760.0
    h2_annual_kg = total_H2 / max(sim_yr, 1e-9)
    LCOH         = (total_elec + total_shutdown) / max(total_H2, 1e-12) \
                   + _replacement_lcoh_per_kg(h2_annual_kg, life_cal_yr) \
                   + _system_lcoh_per_kg(h2_annual_kg)

    T_act = degC(a,"T_stack_actual_K")
    j_act = col(a,"j_A_per_m2")

    n_shutdowns = 0
    j_arr = col(df,"j_A_per_m2")
    for i in range(1, len(j_arr)):
        if j_arr[i-1] > 100 and j_arr[i] <= 100:
            n_shutdowns += 1

    rows = [
        ("", "Operating Profile", ""),
        ("Active hours", f"{n_active} / {len(df)}", f"({100*cap_f:.1f}%)"),
        ("Shutdown events", f"{n_shutdowns}", ""),
        ("", "", ""),
        ("", r"Current \& Temperature", ""),
        (r"$j$ avg (active)", rf"{j_act.mean()/10000:.2f} A/cm$^2$", rf"max {j_act.max()/10000:.2f}"),
        (r"$T$ avg (active)", rf"{T_act.mean():.1f} $^\circ$C", rf"max {T_act.max():.1f} $^\circ$C"),
        (r"Hours $T > 80\,^\circ$C", f"{(degC(df,'T_stack_actual_K') > 80).sum()}", ""),
        ("", "", ""),
        ("", r"Production \& Economics", ""),
        (r"H$_2$ produced", f"{total_H2:.0f} kg", ""),
        ("Revenue", f"{total_rev:.0f} EUR", ""),
        ("Electricity cost", f"{total_elec:.0f} EUR", ""),
        ("Degradation cost", f"{total_deg:.0f} EUR", ""),
        ("Shutdown cost", f"{total_shutdown:.0f} EUR", ""),
        ("Net profit", f"{profit:.0f} EUR", ""),
        ("LCOH", f"{LCOH:.2f} EUR/kg", ""),
        ("", "", ""),
        ("", "Efficiency (active)", ""),
        (r"$\eta$ sys (HHV)", rf"{eta_sys*100:.1f}\%", ""),
        (r"$\eta$ stack (HHV)", rf"{eta_stack*100:.1f}\%", ""),
        ("SEC", f"{SEC:.1f} kWh/kg", ""),
        ("", "", ""),
        ("", "Degradation", ""),
        (r"$V_\mathrm{deg}$ final", f"{V_deg_mV:.3f} mV", f"of {V_DEG_EOL_MV:.0f} mV EOL"),
        ("Stack replacements", f"{int(round(max(total_dV_mV - V_deg_mV, 0) / V_DEG_EOL_MV))}", "events"),
        (r"Degradation rate", rf"{rate_op:.3f} $\mu$V/op.h", ""),
        (r"Op.\ lifetime", rf"{life_op_kh:.1f} kh$_{{\rm op}}$", rf"{life_op_yr:.1f} yr$_{{\rm op}}$"),
        (r"Calendar lifetime", rf"{life_cal_yr:.1f} yr", r"(repeated annual duty cycle)"),
    ]

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.axis("off")
    ax.set_title(f"Summary Statistics --- {tex(name)}", fontsize=13, fontweight="bold", pad=20)

    y = 0.96
    dy = 0.032
    for label, value, note in rows:
        if label == "" and value != "":
            # Section header
            ax.text(0.05, y, value, fontsize=11, fontweight="bold",
                    transform=ax.transAxes, va="top")
        elif label != "":
            ax.text(0.08, y, label, fontsize=9, transform=ax.transAxes, va="top", color="#444444")
            ax.text(0.55, y, value, fontsize=9, fontweight="bold",
                    transform=ax.transAxes, va="top", ha="right")
            if note:
                ax.text(0.57, y, note, fontsize=8, transform=ax.transAxes, va="top", color=GREY)
        y -= dy

    fig.tight_layout()
    savefig(fig, outdir / f"{name}_fig0_stats.png")
    return fig

# ── Fig 7 ---Controller comparison: time-series ──────────────────────────────
def fig_comparison(db, dc, da, de=None, outdir=None):
    if outdir is None:
        raise ValueError("outdir must be provided")
    ann = annual(db)
    t_b = col(db,"t_h"); t_a = col(da,"t_h")
    dt_h_b = float(t_b[1]-t_b[0]) if len(t_b)>1 else 1.0
    dt_h_a = float(t_a[1]-t_a[0]) if len(t_a)>1 else 1.0
    mh = max(t_b.max(), t_a.max())
    if dc is not None:
        t_c = col(dc,"t_h")
        dt_h_c = float(t_c[1]-t_c[0]) if len(t_c)>1 else 1.0
        mh = max(mh, t_c.max())
    if de is not None:
        t_e = col(de,"t_h")
        dt_h_e = float(t_e[1]-t_e[0]) if len(t_e)>1 else 1.0
        mh = max(mh, t_e.max())

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    def cum_h2(d, dt):     return np.nancumsum(col(d,"m_dot_H2_kg_h") * dt)
    def cum_profit(d, dt): return np.nancumsum(col(d,"true_profit_eur_h") * dt)
    def cum_deg(d, dt):    return np.nancumsum(col(d,"c_deg_phys_eur_h") * dt)

    panels = [
        (lambda d, dt: col(d,"j_A_per_m2")/10000,
         r"$j$ [A/cm$^2$]",        "Current Density"),
        (lambda d, dt: mV(d,"V_deg_V"),
         r"$V_\mathrm{deg}$ [mV]",        "Cumulative Degradation"),
        (lambda d, dt: degC(d,"T_stack_actual_K"),
         r"$T$ [$^\circ$C]",      "Stack Temperature"),
        (cum_h2,
         r"H$_2$ [kg]",      r"Cumulative H$_2$ Production"),
        (cum_profit,
         r"Cum. profit [\texteuro]",   "Cumulative Profit"),
        (cum_deg,
         r"Cum. degr. cost [\texteuro]","Cumulative Degradation Cost"),
    ]
    for ax, (fn, yl, tt) in zip(axes.flatten(), panels):
        yb = fn(db, dt_h_b); ya = fn(da, dt_h_a)
        ax.plot(t_b, yb, color=BLUE,   lw=1.0, label="Load-following",  alpha=0.9)
        if dc is not None:
            yc = fn(dc, dt_h_c)
            ax.plot(t_c, yc, color=GREEN, lw=1.0, label="Price-aware", alpha=0.9)
        ax.plot(t_a, ya, color=ORANGE, lw=1.0, label="Degradation-aware", alpha=0.9)
        if de is not None:
            ye = fn(de, dt_h_e)
            ax.plot(t_e, ye, color=PURPLE, lw=1.0, label="Lifetime-aware", alpha=0.9)
        ax.set_ylabel(yl); ax.set_title(tt); ax.legend(loc="lower right")
        if ann: month_ticks(ax, mh)
        else:   ax.set_xlabel("Time [h]")

    fig.tight_layout()
    savefig(fig, outdir/"comparison_fig7_timeseries.png")
    return fig


# ── Fig 7b ---RUL Diagnostic Comparison ───────────────────────────────────────
def fig_rul_comparison(db, dc, da, de=None, outdir=None):
    """
    RUL diagnostic figure: shows how accumulated degradation state (V_deg)
    drives the RUL multiplier over time, and compares V_deg trajectories
    and T_target behaviour across controllers.

    Replaces the former EIS health indicator figure.
    """
    if outdir is None:
        raise ValueError("outdir must be provided")
    ann = annual(db)
    t_b = col(db,"t_h"); t_a = col(da,"t_h")
    mh = max(t_b.max(), t_a.max())
    if dc is not None:
        t_c = col(dc,"t_h")
        mh = max(mh, t_c.max())
    if de is not None:
        t_e = col(de,"t_h")
        mh = max(mh, t_e.max())

    V_EOL = V_DEG_EOL_MV / 1000.0  # V
    RUL_FLOOR = 0.05

    def rul_multiplier(df):
        vd = col(df, "V_deg_V")
        v_rem = np.maximum(V_EOL - vd, V_EOL * RUL_FLOOR)
        return V_EOL / v_rem

    def rul_fraction(df):
        return 1.0 - col(df, "V_deg_V") / V_EOL

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Top-left: V_deg trajectories
    axes[0,0].plot(t_b, mV(db,"V_deg_V"), color=BLUE,   lw=1.0, label="Load-following")
    if dc is not None:
        axes[0,0].plot(t_c, mV(dc,"V_deg_V"), color=GREEN, lw=1.0, label="Price-aware")
    axes[0,0].plot(t_a, mV(da,"V_deg_V"), color=ORANGE,  lw=1.0, label="Degradation-aware")
    if de is not None:
        axes[0,0].plot(t_e, mV(de,"V_deg_V"), color=PURPLE, lw=1.0, label="Lifetime-aware")
    axes[0,0].axhline(V_DEG_EOL_MV, color=GREY, lw=0.8, ls="--",
                      label=rf"EOL = {V_DEG_EOL_MV:.0f} mV")
    axes[0,0].set_ylabel(r"$\Delta V_\mathrm{deg}$ [mV]")
    axes[0,0].set_title("(a) Accumulated degradation (health state)")
    axes[0,0].legend(fontsize=8)
    if ann: month_ticks(axes[0,0], mh)
    else:   axes[0,0].set_xlabel("Time [h]")

    # Top-right: RUL fraction (remaining life)
    axes[0,1].plot(t_b, rul_fraction(db)*100, color=BLUE,   lw=1.0, label="Load-following")
    if dc is not None:
        axes[0,1].plot(t_c, rul_fraction(dc)*100, color=GREEN, lw=1.0, label="Price-aware")
    axes[0,1].plot(t_a, rul_fraction(da)*100, color=ORANGE,  lw=1.0, label="Degradation-aware")
    if de is not None:
        axes[0,1].plot(t_e, rul_fraction(de)*100, color=PURPLE, lw=1.0, label="Lifetime-aware")
    axes[0,1].axhline(100, color=GREY, lw=0.6, ls=":")
    axes[0,1].set_ylabel(r"Remaining useful life [\%]")
    axes[0,1].set_title("(b) RUL fraction (diagnostic signal)")
    axes[0,1].legend(fontsize=8)
    if ann: month_ticks(axes[0,1], mh)
    else:   axes[0,1].set_xlabel("Time [h]")

    # Bottom-left: RUL cost multiplier
    axes[1,0].plot(t_b, rul_multiplier(db), color=BLUE,   lw=1.0, label="Load-following")
    if dc is not None:
        axes[1,0].plot(t_c, rul_multiplier(dc), color=GREEN, lw=1.0, label="Price-aware")
    axes[1,0].plot(t_a, rul_multiplier(da), color=ORANGE,  lw=1.0, label="Degradation-aware")
    if de is not None:
        axes[1,0].plot(t_e, rul_multiplier(de), color=PURPLE, lw=1.0, label="Lifetime-aware")
    axes[1,0].axhline(1.0, color=GREY, lw=0.6, ls="--", label="BOL baseline")
    axes[1,0].set_ylabel(r"RUL cost multiplier $V_\mathrm{EOL}/(V_\mathrm{EOL}-V_\mathrm{deg})$")
    axes[1,0].set_title("(c) Degradation cost weight (feedback to optimizer)")
    axes[1,0].legend(fontsize=8)
    if ann: month_ticks(axes[1,0], mh)
    else:   axes[1,0].set_xlabel("Time [h]")

    # Bottom-right: T_target comparison (response to RUL feedback)
    if "T_target_K" in db.columns:
        axes[1,1].plot(t_b, degC(db,"T_target_K"), color=BLUE,   lw=0.6,
                       alpha=0.6, label="Load-following")
    if dc is not None and "T_target_K" in dc.columns:
        axes[1,1].plot(t_c, degC(dc,"T_target_K"), color=GREEN, lw=0.6,
                       alpha=0.6, label="Price-aware")
    if "T_target_K" in da.columns:
        axes[1,1].plot(t_a, degC(da,"T_target_K"), color=ORANGE,  lw=0.8,
                       label="Degradation-aware")
    if de is not None and "T_target_K" in de.columns:
        axes[1,1].plot(t_e, degC(de,"T_target_K"), color=PURPLE, lw=0.8,
                       label="Lifetime-aware")
    axes[1,1].axhline(60, color=GREY, lw=0.8, ls="--",
                      label=r"$T_\mathrm{ref}=60^\circ$C")
    axes[1,1].set_ylabel(r"$T_\mathrm{target}$ [$^\circ$C]")
    axes[1,1].set_title("(d) Temperature target (control response)")
    axes[1,1].legend(fontsize=8)
    if ann: month_ticks(axes[1,1], mh)
    else:   axes[1,1].set_xlabel("Time [h]")

    fig.tight_layout()
    savefig(fig, outdir/"comparison_fig7b_rul_diagnostic.png")
    return fig

# ── Fig 8 --- Load-following vs aware: KPI summary ────────────────────────────────
def _kpi(df):
    """Collect key performance indicators from a simulation DataFrame."""
    a   = mask(df, col(df,"j_A_per_m2") > 100)
    P   = col(a,"P_total_W"); h2 = col(a,"m_dot_H2_kg_h")
    eta = h2*HHV*1e3 / np.where(P > 0, P, np.nan)
    SEC = P/1e3 / np.where(h2 > 0, h2, np.nan)
    th  = col(df,"t_h").max()
    dt_h = col(df,"t_h")[1] - col(df,"t_h")[0] if len(df) > 1 else 1.0
    vmV         = mV(df,"V_deg_V")[-1]                       # current stack state (for display)
    total_dV_mV = col(df,"dV_deg_V").sum() * 1000            # cumulative, robust to EOL resets

    # Operating hours and capacity factor
    n_op = len(a)
    n_total = len(df)
    op_hours = n_op * dt_h
    cap_factor = n_op / max(n_total, 1)

    # Degradation rates (use cumulative dV, not final V_deg_V)
    rate_op  = total_dV_mV * 1000 / max(op_hours, 1)        # µV/op.h
    r_ann    = total_dV_mV / max(th / 8760, 1e-9)            # mV/yr

    # Projected lifetimes (BOL → EOL)
    life_op_h   = V_DEG_EOL_MV * 1000 / rate_op if rate_op > 0 else np.inf
    life_op_kh  = life_op_h / 1000
    life_op_yr  = life_op_h / 8760
    life_cal_yr = V_DEG_EOL_MV / r_ann if r_ann > 0 else np.inf

    h2_total     = float(np.nansum(col(df,"m_dot_H2_kg_h")) * dt_h)
    sim_yr       = th / 8760.0
    h2_annual_kg = h2_total / max(sim_yr, 1e-9)   # [kg/yr] — divides by sim length

    return dict(
        h2_kg          = h2_total,
        h2_annual_kg   = h2_annual_kg,
        profit_eur     = float(np.nansum(col(df,"true_profit_eur_h")) * dt_h),
        elec_eur       = float(np.nansum(col(df,"c_elec_eur_h")) * dt_h),
        h2_rev_eur     = float(np.nansum(col(df,"r_H2_eur_h")) * dt_h),
        deg_eur        = float(np.nansum(col(df,"c_deg_phys_eur_h")) * dt_h),
        shutdown_eur   = float(np.nansum(col(df,"c_shutdown_eur")) * dt_h),
        vdeg_mV        = vmV,
        rate_uVh       = rate_op,
        life_op_kh     = float(life_op_kh),
        life_op_yr     = float(life_op_yr),
        life_cal_yr    = float(life_cal_yr),
        op_hours       = float(op_hours),
        eta_pct        = float(np.nansum(h2*HHV*1e3*dt_h) / np.nansum(P*dt_h) * 100) if np.nansum(P*dt_h) > 0 else 0.0,
        SEC_kWh_kg     = float(np.nansum(P/1e3*dt_h) / np.nansum(h2*dt_h)) if np.nansum(h2*dt_h) > 0 else 0.0,
        T_mean_C       = float(degC(a,"T_stack_actual_K").mean()),
        op_pct         = 100 * cap_factor,
        n_replacements = int(round(max(total_dV_mV - vmV, 0) / V_DEG_EOL_MV)),
    )


def fig_kpi_summary(db, dc, da, de=None, outdir=None):
    if outdir is None:
        raise ValueError("outdir must be provided")
    kb = _kpi(db)
    ka = _kpi(da)
    kc = _kpi(dc) if dc is not None else None
    ke = _kpi(de) if de is not None else None

    # Per-kg H₂ metrics (normalised comparison)
    def _per_kg(val, h2):
        return val / h2 if h2 > 0 else 0.0

    def _lcoh(k):
        """Operational LCOH + NPV replacement + system CAPEX contribution [EUR/kg]."""
        op = _per_kg(k["elec_eur"] + k["shutdown_eur"], k["h2_kg"])
        return (op
                + _replacement_lcoh_per_kg(k["h2_annual_kg"], k["life_cal_yr"])
                + _system_lcoh_per_kg(k["h2_annual_kg"]))

    kb_lcoh      = _lcoh(kb)
    ka_lcoh      = _lcoh(ka)
    kb_profit_kg = _per_kg(kb["profit_eur"], kb["h2_kg"])
    ka_profit_kg = _per_kg(ka["profit_eur"], ka["h2_kg"])
    kb_elec_kg   = _per_kg(kb["elec_eur"], kb["h2_kg"])
    ka_elec_kg   = _per_kg(ka["elec_eur"], ka["h2_kg"])
    kb_deg_kg    = _per_kg(kb["deg_eur"], kb["h2_kg"])
    ka_deg_kg    = _per_kg(ka["deg_eur"], ka["h2_kg"])

    if kc is not None:
        kc_lcoh      = _lcoh(kc)
        kc_profit_kg = _per_kg(kc["profit_eur"], kc["h2_kg"])
        kc_elec_kg   = _per_kg(kc["elec_eur"], kc["h2_kg"])

    if ke is not None:
        ke_lcoh      = _lcoh(ke)
        ke_profit_kg = _per_kg(ke["profit_eur"], ke["h2_kg"])
        ke_elec_kg   = _per_kg(ke["elec_eur"], ke["h2_kg"])

    # ── layout: 5 rows × 3 cols ──────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 17))
    gs = fig.add_gridspec(5, 3, hspace=0.50, wspace=0.38)

    if ke is not None:
        labels  = ["Load-following", "Price-aware", "Degradation-aware", "Lifetime-aware"]
        bar_clr = [BLUE, GREEN, ORANGE, PURPLE]
        x       = np.array([0, 1, 2, 3])
    elif kc is not None:
        labels  = ["Load-following", "Price-aware", "Degradation-aware"]
        bar_clr = [BLUE, GREEN, ORANGE]
        x       = np.array([0, 1, 2])
    else:
        labels  = ["Load-following", "Degradation-aware"]
        bar_clr = [BLUE, ORANGE]
        x       = np.array([0, 1])

    def bar_ax(ax, vals, ylabel, title, fmt="{:.0f}", highlight="high",
               yextra=None):
        bars = ax.bar(x, vals, color=bar_clr, width=0.5,
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel); ax.set_title(title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                    fmt.format(v), ha="center", va="bottom")
        # shade the better bar
        better = int(np.argmax(vals)) if highlight == "high" else int(np.argmin(vals))
        bars[better].set_edgecolor("black"); bars[better].set_linewidth(1.5)
        if yextra is not None:
            ax.axhline(yextra, color=GREY, lw=0.8, ls=":")

    def v(*args):
        """Build a value list for bar_ax: [b, (c if present), a, (e if present)]."""
        if ke is not None:
            return list(args)   # caller passes (kb_val, kc_val, ka_val, ke_val)
        if kc is not None:
            return list(args[:3])   # caller passes (kb_val, kc_val, ka_val)
        return [args[0], args[2]]  # drop middle(s) when dc/de are None

    # Row 0 ─ absolute economics
    bar_ax(fig.add_subplot(gs[0,0]),
           v(kb["profit_eur"],
             kc["profit_eur"] if kc else 0,
             ka["profit_eur"],
             ke["profit_eur"] if ke else 0),
           r"\texteuro", "Net Profit", r"{:+.0f} \texteuro")
    bar_ax(fig.add_subplot(gs[0,1]),
           v(kb["elec_eur"],
             kc["elec_eur"] if kc else 0,
             ka["elec_eur"],
             ke["elec_eur"] if ke else 0),
           r"\texteuro", "Electricity Cost", r"{:.0f} \texteuro", highlight="low")
    bar_ax(fig.add_subplot(gs[0,2]),
           v(kb["h2_kg"],
             kc["h2_kg"] if kc else 0,
             ka["h2_kg"],
             ke["h2_kg"] if ke else 0),
           "kg", r"Total H$_2$ Produced", "{:.0f} kg")

    # Row 1 ─ per-kg H₂ metrics (normalised ---fair comparison)
    bar_ax(fig.add_subplot(gs[1,0]),
           v(kb_profit_kg, kc_profit_kg if kc else 0, ka_profit_kg,
             ke_profit_kg if ke else 0),
           r"\texteuro/kg", r"Profit per kg H$_2$", r"{:.2f} \texteuro/kg")
    bar_ax(fig.add_subplot(gs[1,1]),
           v(kb_elec_kg, kc_elec_kg if kc else 0, ka_elec_kg,
             ke_elec_kg if ke else 0),
           r"\texteuro/kg", r"Electricity Cost per kg H$_2$", r"{:.2f} \texteuro/kg", highlight="low")
    bar_ax(fig.add_subplot(gs[1,2]),
           v(kb_lcoh, kc_lcoh if kc else 0, ka_lcoh,
             ke_lcoh if ke else 0),
           r"\texteuro/kg", "Prod.-Gate LCOH (elec+deg)", r"{:.2f} \texteuro/kg", highlight="low")

    # Row 2 ─ degradation / lifetime
    bar_ax(fig.add_subplot(gs[2,0]),
           v(kb["vdeg_mV"],
             kc["vdeg_mV"] if kc else 0,
             ka["vdeg_mV"],
             ke["vdeg_mV"] if ke else 0),
           "mV", "Final Degradation", "{:.1f} mV",
           highlight="low", yextra=V_DEG_EOL_MV)
    bar_ax(fig.add_subplot(gs[2,1]),
           v(min(kb["life_cal_yr"], 99),
             min(kc["life_cal_yr"], 99) if kc else 0,
             min(ka["life_cal_yr"], 99),
             min(ke["life_cal_yr"], 99) if ke else 0),
           "years", r"Calendar Lifetime [yr]", "{:.1f} yr", highlight="high")
    bar_ax(fig.add_subplot(gs[2,2]),
           v(kb["deg_eur"],
             kc["deg_eur"] if kc else 0,
             ka["deg_eur"],
             ke["deg_eur"] if ke else 0),
           r"\texteuro", "Degradation Cost", r"{:.0f} \texteuro", highlight="low")

    # Row 3 ─ efficiency + stack replacements
    bar_ax(fig.add_subplot(gs[3,0]),
           v(kb["SEC_kWh_kg"],
             kc["SEC_kWh_kg"] if kc else 0,
             ka["SEC_kWh_kg"],
             ke["SEC_kWh_kg"] if ke else 0),
           "kWh/kg", "Mean SEC", "{:.1f}", highlight="low")
    bar_ax(fig.add_subplot(gs[3,1]),
           v(kb["eta_pct"],
             kc["eta_pct"] if kc else 0,
             ka["eta_pct"],
             ke["eta_pct"] if ke else 0),
           r"\%", r"Mean $\eta_\mathrm{HHV}$ (active)", r"{:.1f}\%", highlight="high")
    bar_ax(fig.add_subplot(gs[3,2]),
           v(kb["n_replacements"],
             kc["n_replacements"] if kc else 0,
             ka["n_replacements"],
             ke["n_replacements"] if ke else 0),
           "count", "Stack Replacements", "{:.0f}", highlight="low")

    # Summary text panel (spans all 3 columns of row 4)
    ax_txt = fig.add_subplot(gs[4, :])
    ax_txt.axis("off")
    diff_profit = ka["profit_eur"] - kb["profit_eur"]
    diff_eol    = min(ka["life_cal_yr"],99) - min(kb["life_cal_yr"],99)
    pct_profit  = 100*diff_profit/abs(kb["profit_eur"]) if kb["profit_eur"] else 0
    lines = [
        r"Key Differences (Degradation-aware $-$ Load-following)",
        r" ",
        rf"  dProfit:        {diff_profit:+.0f} \texteuro\ ({pct_profit:+.1f}\%)",
        rf"  dProfit/kg:     {ka_profit_kg-kb_profit_kg:+.2f} \texteuro/kg",
        rf"  dElec cost/kg:  {ka_elec_kg-kb_elec_kg:+.2f} \texteuro/kg",
        rf"  dLCOH:          {ka_lcoh-kb_lcoh:+.2f} \texteuro/kg",
        rf"  dDegradation:   {ka['vdeg_mV']-kb['vdeg_mV']:+.1f} mV",
        rf"  dLifetime:      {diff_eol:+.1f} yr",
    ]
    if kc is not None:
        diff_profit_c = kc["profit_eur"] - kb["profit_eur"]
        pct_profit_c  = 100*diff_profit_c/abs(kb["profit_eur"]) if kb["profit_eur"] else 0
        lines += [
            r" ",
            r"Key Differences (Price-aware $-$ Load-following)",
            r" ",
            rf"  dProfit:        {diff_profit_c:+.0f} \texteuro\ ({pct_profit_c:+.1f}\%)",
            rf"  dProfit/kg:     {kc_profit_kg-kb_profit_kg:+.2f} \texteuro/kg",
            rf"  dLCOH:          {kc_lcoh-kb_lcoh:+.2f} \texteuro/kg",
            rf"  dDegradation:   {kc['vdeg_mV']-kb['vdeg_mV']:+.1f} mV",
        ]
    if ke is not None:
        diff_profit_e = ke["profit_eur"] - ka["profit_eur"]
        pct_profit_e  = 100*diff_profit_e/abs(ka["profit_eur"]) if ka["profit_eur"] else 0
        diff_eol_e    = min(ke["life_cal_yr"],99) - min(ka["life_cal_yr"],99)
        lines += [
            r" ",
            r"Key Differences (Lifetime-aware $-$ Degradation-aware)",
            r" ",
            rf"  dProfit:        {diff_profit_e:+.0f} \texteuro\ ({pct_profit_e:+.1f}\%)",
            rf"  dProfit/kg:     {ke_profit_kg-ka_profit_kg:+.2f} \texteuro/kg",
            rf"  dLCOH:          {ke_lcoh-ka_lcoh:+.2f} \texteuro/kg",
            rf"  dDegradation:   {ke['vdeg_mV']-ka['vdeg_mV']:+.1f} mV",
            rf"  dLifetime:      {diff_eol_e:+.1f} yr",
        ]
    lines += [
        r" ",
        r"  Operating fraction:",
        rf"    Load-following:    {kb['op_pct']:.1f}\%",
    ]
    if kc is not None:
        lines.append(rf"    Price-aware:       {kc['op_pct']:.1f}\%")
    lines += [
        rf"    Degradation-aware: {ka['op_pct']:.1f}\%",
    ]
    if ke is not None:
        lines.append(rf"    Lifetime-aware:    {ke['op_pct']:.1f}\%")
    lines += [
        r" ",
        r"  Mean $T_{\mathrm{stack}}$:",
        rf"    Load-following:    {kb['T_mean_C']:.1f}$^\circ$C",
    ]
    if kc is not None:
        lines.append(rf"    Price-aware:       {kc['T_mean_C']:.1f}$^\circ$C")
    lines.append(rf"    Degradation-aware: {ka['T_mean_C']:.1f}$^\circ$C")
    if ke is not None:
        lines.append(rf"    Lifetime-aware:    {ke['T_mean_C']:.1f}$^\circ$C")

    ax_txt.text(0.05, 0.97, "\n".join(lines), transform=ax_txt.transAxes,
                va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f8", ec=GREY, lw=0.8))

    savefig(fig, outdir/"comparison_fig8_kpi.png")
    return fig


# ── Fig 9 ---Controller distributions ─────────────────────────────────────────
def fig_distributions(db, dc, da, de=None, outdir=None):
    if outdir is None:
        raise ValueError("outdir must be provided")
    _rc_keys = ["font.size","axes.titlesize","axes.labelsize",
                "xtick.labelsize","ytick.labelsize","legend.fontsize"]
    _saved_rc = {k: matplotlib.rcParams[k] for k in _rc_keys}
    matplotlib.rcParams.update({"font.size": 15, "axes.titlesize": 16,
                                 "axes.labelsize": 15, "xtick.labelsize": 14,
                                 "ytick.labelsize": 14, "legend.fontsize": 12})
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))

    # Active-hour masks
    ab = mask(db, col(db,"j_A_per_m2") > 100)
    aa = mask(da, col(da,"j_A_per_m2") > 100)
    n_b = len(db); n_a = len(da)
    idle_b = 100*(1 - len(ab)/n_b); idle_a = 100*(1 - len(aa)/n_a)
    if dc is not None:
        ac = mask(dc, col(dc,"j_A_per_m2") > 100)
        n_c = len(dc)
        idle_c = 100*(1 - len(ac)/n_c)
    if de is not None:
        ae = mask(de, col(de,"j_A_per_m2") > 100)
        n_e = len(de)
        idle_e = 100*(1 - len(ae)/n_e)

    # ── Row 0: ALL hours ─────────────────────────────────────────────────
    # j distribution (all hours ---shows idle fraction as j=0 peak)
    jb = col(db,"j_A_per_m2")/10000
    ja = col(da,"j_A_per_m2")/10000
    jmax = max(jb.max(), ja.max())
    if dc is not None:
        jc = col(dc,"j_A_per_m2")/10000
        jmax = max(jmax, jc.max())
    if de is not None:
        je = col(de,"j_A_per_m2")/10000
        jmax = max(jmax, je.max())
    bins = np.linspace(0, jmax, 50)
    axes[0,0].hist(jb, bins=bins, color=BLUE,   alpha=0.50, label="Load-following",  density=True)
    if dc is not None:
        axes[0,0].hist(jc, bins=bins, color=GREEN, alpha=0.50, label="Price-aware", density=True)
    axes[0,0].hist(ja, bins=bins, color=ORANGE, alpha=0.50, label="Degradation-aware", density=True)
    if de is not None:
        axes[0,0].hist(je, bins=bins, color=PURPLE, alpha=0.50, label="Lifetime-aware", density=True)
    axes[0,0].axvline(jb.mean(), color=BLUE, lw=1.2, ls="--", alpha=0.8)
    if dc is not None:
        axes[0,0].axvline(jc.mean(), color=GREEN, lw=1.2, ls="--", alpha=0.8)
    axes[0,0].axvline(ja.mean(), color=ORANGE, lw=1.5, ls="--", alpha=0.9)
    if de is not None:
        axes[0,0].axvline(je.mean(), color=PURPLE, lw=1.5, ls=":", alpha=1.0)
    axes[0,0].set_xlabel(r"$j$ [A/cm$^2$]"); axes[0,0].set_ylabel("Density")
    axes[0,0].set_title("Current Density --- All Hours")
    axes[0,0].legend(loc="upper right")

    # Temperature (all hours ---includes idle cooling toward ambient)
    Tb = degC(db,"T_stack_actual_K")
    Ta = degC(da,"T_stack_actual_K")
    tmin = min(Tb.min(), Ta.min())
    tmax = max(Tb.max(), Ta.max())
    if dc is not None:
        Tc = degC(dc,"T_stack_actual_K")
        tmin = min(tmin, Tc.min()); tmax = max(tmax, Tc.max())
    if de is not None:
        Te = degC(de,"T_stack_actual_K")
        tmin = min(tmin, Te.min()); tmax = max(tmax, Te.max())
    tbins = np.linspace(tmin-1, tmax+1, 40)
    axes[0,1].hist(Tb, bins=tbins, color=BLUE,   alpha=0.50, label="Load-following",  density=True)
    if dc is not None:
        axes[0,1].hist(Tc, bins=tbins, color=GREEN, alpha=0.50, label="Price-aware", density=True)
    axes[0,1].hist(Ta, bins=tbins, color=ORANGE, alpha=0.50, label="Degradation-aware", density=True)
    if de is not None:
        axes[0,1].hist(Te, bins=tbins, color=PURPLE, alpha=0.50, label="Lifetime-aware", density=True)
    axes[0,1].axvline(Tb.mean(), color=BLUE, lw=1.2, ls="--", alpha=0.8)
    if dc is not None:
        axes[0,1].axvline(Tc.mean(), color=GREEN, lw=1.2, ls="--", alpha=0.8)
    axes[0,1].axvline(Ta.mean(), color=ORANGE, lw=1.5, ls="--", alpha=0.9)
    if de is not None:
        axes[0,1].axvline(Te.mean(), color=PURPLE, lw=1.5, ls=":", alpha=1.0)
    axes[0,1].set_xlabel(r"$T$ [$^\circ$C]"); axes[0,1].set_ylabel("Density")
    axes[0,1].set_title("Stack Temperature --- All Hours")
    axes[0,1].legend(loc="upper right")

    # Degradation rate (all hours)
    rb_all = uVh(db,"dV_deg_V"); ra_all = uVh(da,"dV_deg_V")
    pos_arrs = [rb_all[rb_all>0], ra_all[ra_all>0]]
    if dc is not None:
        rc_all = uVh(dc,"dV_deg_V")
        pos_arrs.append(rc_all[rc_all>0])
    if de is not None:
        re_all = uVh(de,"dV_deg_V")
        pos_arrs.append(re_all[re_all>0])
    all_pos = np.concatenate([a for a in pos_arrs if len(a)>0])
    rmax = np.max(all_pos) if len(all_pos) > 0 else 1.0
    rbins = np.linspace(0, rmax, 40)
    axes[0,2].hist(rb_all, bins=rbins, color=BLUE,   alpha=0.50, label="Load-following",  density=True)
    if dc is not None:
        axes[0,2].hist(rc_all, bins=rbins, color=GREEN, alpha=0.50, label="Price-aware", density=True)
    axes[0,2].hist(ra_all, bins=rbins, color=ORANGE, alpha=0.50, label="Degradation-aware", density=True)
    if de is not None:
        axes[0,2].hist(re_all, bins=rbins, color=PURPLE, alpha=0.50, label="Lifetime-aware", density=True)
    axes[0,2].axvline(rb_all.mean(), color=BLUE, lw=1.2, ls="--", alpha=0.8)
    if dc is not None:
        axes[0,2].axvline(rc_all.mean(), color=GREEN, lw=1.2, ls="--", alpha=0.8)
    axes[0,2].axvline(ra_all.mean(), color=ORANGE, lw=1.5, ls="--", alpha=0.9)
    if de is not None:
        axes[0,2].axvline(re_all.mean(), color=PURPLE, lw=1.5, ls=":", alpha=1.0)
    axes[0,2].set_xlabel(r"$\dot{V}_\mathrm{deg}$ [$\mu$V/h]"); axes[0,2].set_ylabel("Density")
    axes[0,2].set_title("Degradation Rate --- All Hours")
    axes[0,2].legend(loc="upper right")

    # ── Row 1: ACTIVE hours only ─────────────────────────────────────────
    # j distribution (active only)
    jb_a = col(ab,"j_A_per_m2")/10000
    ja_a = col(aa,"j_A_per_m2")/10000
    bins_a = np.linspace(0, jmax, 50)
    axes[1,0].hist(jb_a, bins=bins_a, color=BLUE,   alpha=0.50, label="Load-following",  density=True)
    if dc is not None:
        jc_a = col(ac,"j_A_per_m2")/10000
        axes[1,0].hist(jc_a, bins=bins_a, color=GREEN, alpha=0.50, label="Price-aware", density=True)
    axes[1,0].hist(ja_a, bins=bins_a, color=ORANGE, alpha=0.50, label="Degradation-aware", density=True)
    if de is not None:
        je_a = col(ae,"j_A_per_m2")/10000
        axes[1,0].hist(je_a, bins=bins_a, color=PURPLE, alpha=0.50, label="Lifetime-aware", density=True)
    axes[1,0].axvline(jb_a.mean(), color=BLUE, lw=1.2, ls="--", alpha=0.8)
    if dc is not None:
        axes[1,0].axvline(jc_a.mean(), color=GREEN, lw=1.2, ls="--", alpha=0.8)
    axes[1,0].axvline(ja_a.mean(), color=ORANGE, lw=1.5, ls="--", alpha=0.9)
    if de is not None:
        axes[1,0].axvline(je_a.mean(), color=PURPLE, lw=1.5, ls=":", alpha=1.0)
    axes[1,0].set_xlabel(r"$j$ [A/cm$^2$]"); axes[1,0].set_ylabel("Density")
    axes[1,0].set_title("Current Density --- Active Hours")
    axes[1,0].legend(loc="upper right")

    # Temperature (active only ---should cluster around operating window)
    Tb_a = degC(ab,"T_stack_actual_K")
    Ta_a = degC(aa,"T_stack_actual_K")
    tmin_a = min(Tb_a.min(), Ta_a.min())
    tmax_a = max(Tb_a.max(), Ta_a.max())
    if dc is not None:
        Tc_a = degC(ac,"T_stack_actual_K")
        tmin_a = min(tmin_a, Tc_a.min()); tmax_a = max(tmax_a, Tc_a.max())
    if de is not None:
        Te_a = degC(ae,"T_stack_actual_K")
        tmin_a = min(tmin_a, Te_a.min()); tmax_a = max(tmax_a, Te_a.max())
    tbins_a = np.linspace(tmin_a-1, tmax_a+1, 40)
    axes[1,1].hist(Tb_a, bins=tbins_a, color=BLUE,   alpha=0.50, label="Load-following",  density=True)
    if dc is not None:
        axes[1,1].hist(Tc_a, bins=tbins_a, color=GREEN, alpha=0.50, label="Price-aware", density=True)
    axes[1,1].hist(Ta_a, bins=tbins_a, color=ORANGE, alpha=0.50, label="Degradation-aware", density=True)
    if de is not None:
        axes[1,1].hist(Te_a, bins=tbins_a, color=PURPLE, alpha=0.50, label="Lifetime-aware", density=True)
    axes[1,1].axvline(Tb_a.mean(), color=BLUE, lw=1.2, ls="--", alpha=0.8)
    if dc is not None:
        axes[1,1].axvline(Tc_a.mean(), color=GREEN, lw=1.2, ls="--", alpha=0.8)
    axes[1,1].axvline(Ta_a.mean(), color=ORANGE, lw=1.5, ls="--", alpha=0.9)
    if de is not None:
        axes[1,1].axvline(Te_a.mean(), color=PURPLE, lw=1.5, ls=":", alpha=1.0)
    axes[1,1].set_xlabel(r"$T$ [$^\circ$C]"); axes[1,1].set_ylabel("Density")
    axes[1,1].set_title("Stack Temperature --- Active Hours")
    axes[1,1].legend(loc="upper right")

    # Degradation rate (active only)
    rb = uVh(ab,"dV_deg_V"); ra = uVh(aa,"dV_deg_V")
    axes[1,2].hist(rb, bins=rbins, color=BLUE,   alpha=0.50, label="Load-following",  density=True)
    if dc is not None:
        rc = uVh(ac,"dV_deg_V")
        axes[1,2].hist(rc, bins=rbins, color=GREEN, alpha=0.50, label="Price-aware", density=True)
    axes[1,2].hist(ra, bins=rbins, color=ORANGE, alpha=0.50, label="Degradation-aware", density=True)
    if de is not None:
        re = uVh(ae,"dV_deg_V")
        axes[1,2].hist(re, bins=rbins, color=PURPLE, alpha=0.50, label="Lifetime-aware", density=True)
    axes[1,2].axvline(rb.mean(), color=BLUE, lw=1.2, ls="--", alpha=0.8)
    if dc is not None:
        axes[1,2].axvline(rc.mean(), color=GREEN, lw=1.2, ls="--", alpha=0.8)
    axes[1,2].axvline(ra.mean(), color=ORANGE, lw=1.5, ls="--", alpha=0.9)
    if de is not None:
        axes[1,2].axvline(re.mean(), color=PURPLE, lw=1.5, ls=":", alpha=1.0)
    axes[1,2].set_xlabel(r"$\dot{V}_\mathrm{deg}$ [$\mu$V/h]"); axes[1,2].set_ylabel("Density")
    axes[1,2].set_title("Degradation Rate --- Active Hours")
    axes[1,2].legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.18)
    savefig(fig, outdir/"comparison_fig9_distributions.png")
    matplotlib.rcParams.update(_saved_rc)
    return fig

# ── Fig 10 ---LCOH breakdown ──────────────────────────────────────────────────
def fig_lcoh_breakdown(db, dc, da, de=None, outdir=None):
    """Production-gate LCOH stacked-bar breakdown + monthly comparison."""
    if outdir is None:
        raise ValueError("outdir must be provided")
    kb = _kpi(db); ka = _kpi(da)
    kc = _kpi(dc) if dc is not None else None
    ke = _kpi(de) if de is not None else None

    def _per_kg(v, h2): return v / h2 if h2 > 0 else 0.0

    # Per-kg cost components (electricity + shutdown + replacement NPV + system CAPEX = full LCOH)
    def _costs(k):
        h2  = k["h2_kg"]
        e   = _per_kg(k["elec_eur"],     h2)
        s   = _per_kg(k["shutdown_eur"], h2)
        d   = _replacement_lcoh_per_kg(k["h2_annual_kg"], k["life_cal_yr"])
        sys = _system_lcoh_per_kg(k["h2_annual_kg"])
        return e, s, d, sys, e + s + d + sys

    elec_b, shut_b, deg_b, sys_b, lcoh_b = _costs(kb)
    elec_a, shut_a, deg_a, sys_a, lcoh_a = _costs(ka)
    elec_c, shut_c, deg_c, sys_c, lcoh_c = _costs(kc) if kc else (0, 0, 0, 0, 0)
    elec_e, shut_e, deg_e, sys_e, lcoh_e = _costs(ke) if ke else (0, 0, 0, 0, 0)

    p_H2 = float(_plant["economics"]["p_H2_eur_per_kg"])

    if ke is not None:
        labels    = ["Load-following", "Price-aware", "Degradation-aware", "Lifetime-aware"]
        bar_clr   = [BLUE, GREEN, ORANGE, PURPLE]
        x         = np.array([0, 1, 2, 3])
        elec_vals = [elec_b, elec_c, elec_a, elec_e]
        shut_vals = [shut_b, shut_c, shut_a, shut_e]
        deg_vals  = [deg_b,  deg_c,  deg_a,  deg_e]
        sys_vals  = [sys_b,  sys_c,  sys_a,  sys_e]
        lcoh_vals = [lcoh_b, lcoh_c, lcoh_a, lcoh_e]
        w_bar     = 0.35
    elif kc is not None:
        labels    = ["Load-following", "Price-aware", "Degradation-aware"]
        bar_clr   = [BLUE, GREEN, ORANGE]
        x         = np.array([0, 1, 2])
        elec_vals = [elec_b, elec_c, elec_a]
        shut_vals = [shut_b, shut_c, shut_a]
        deg_vals  = [deg_b,  deg_c,  deg_a]
        sys_vals  = [sys_b,  sys_c,  sys_a]
        lcoh_vals = [lcoh_b, lcoh_c, lcoh_a]
        w_bar     = 0.4
    else:
        labels    = ["Load-following", "Degradation-aware"]
        bar_clr   = [BLUE, ORANGE]
        x         = np.array([0, 1])
        elec_vals = [elec_b, elec_a]
        shut_vals = [shut_b, shut_a]
        deg_vals  = [deg_b,  deg_a]
        sys_vals  = [sys_b,  sys_a]
        lcoh_vals = [lcoh_b, lcoh_a]
        w_bar     = 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # ── Panel 1: stacked bar LCOH breakdown (elec + shutdown + degradation + system CAPEX) ──
    ax = axes[0]
    elec_arr = np.array(elec_vals)
    shut_arr = np.array(shut_vals)
    deg_arr  = np.array(deg_vals)
    sys_arr  = np.array(sys_vals)
    ax.bar(x, elec_arr, width=w_bar, color=RED,    alpha=0.85, label="Electricity")
    ax.bar(x, shut_arr, width=w_bar, bottom=elec_arr,
           color=BLUE,  alpha=0.85, label="Shutdown")
    ax.bar(x, deg_arr,  width=w_bar, bottom=elec_arr + shut_arr,
           color=ORANGE, alpha=0.85, label="Degradation (stack)")
    ax.bar(x, sys_arr,  width=w_bar, bottom=elec_arr + shut_arr + deg_arr,
           color=GREY,  alpha=0.85, label="System CAPEX")
    ax.axhline(p_H2, color=GREEN, lw=2, ls="--", label=rf"H$_2$ price ({p_H2:.0f} \texteuro/kg)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"LCOH [\texteuro/kg H$_2$]"); ax.set_title("LCOH Composition")
    ax.legend(fontsize=9)
    for i, lc in enumerate(lcoh_vals):
        ax.text(x[i], lc + 0.08, rf"{lc:.2f} \texteuro/kg",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for i, lc in enumerate(lcoh_vals):
        if lc < p_H2:
            ax.fill_between([x[i]-w_bar/2, x[i]+w_bar/2], lc, p_H2,
                            alpha=0.10, color=GREEN)

    # ── Panel 2: monthly LCOH (elec + shutdown + degradation) ───────────────
    ax = axes[1]
    ann_b = annual(db); ann_a = annual(da)
    ann_c = annual(dc) if dc is not None else False
    ann_e = annual(de) if de is not None else False

    def _monthly_lcoh(d):
        me = msum(d, "c_elec_eur_h")
        ms = msum(d, "c_shutdown_eur")
        md = msum(d, "c_deg_phys_eur_h")
        mh = msum(d, "m_dot_H2_kg_h")
        return np.where(mh > 0, (me + ms + md) / mh, np.nan)

    if ann_b and ann_a:
        lcoh_m_b = _monthly_lcoh(db)
        lcoh_m_a = _monthly_lcoh(da)
        xm  = np.arange(12)
        lbl = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
        if ke is not None and ann_e:
            w = 0.2
            lcoh_m_c = _monthly_lcoh(dc) if kc and ann_c else np.full(12, np.nan)
            lcoh_m_e = _monthly_lcoh(de)
            ax.bar(xm - 1.5*w, lcoh_m_b, w, color=BLUE,   alpha=0.85, label="Load-following")
            ax.bar(xm - 0.5*w, lcoh_m_c, w, color=GREEN,  alpha=0.85, label="Price-aware")
            ax.bar(xm + 0.5*w, lcoh_m_a, w, color=ORANGE, alpha=0.85, label="Degradation-aware")
            ax.bar(xm + 1.5*w, lcoh_m_e, w, color=PURPLE, alpha=0.85, label="Lifetime-aware")
        elif kc is not None and ann_c:
            w = 0.25
            lcoh_m_c = _monthly_lcoh(dc)
            ax.bar(xm - w, lcoh_m_b, w, color=BLUE,   alpha=0.85, label="Load-following")
            ax.bar(xm,     lcoh_m_c, w, color=GREEN,  alpha=0.85, label="Price-aware")
            ax.bar(xm + w, lcoh_m_a, w, color=ORANGE, alpha=0.85, label="Degradation-aware")
        else:
            w = 0.35
            ax.bar(xm - w/2, lcoh_m_b, w, color=BLUE,   alpha=0.85, label="Load-following")
            ax.bar(xm + w/2, lcoh_m_a, w, color=ORANGE, alpha=0.85, label="Degradation-aware")
        ax.axhline(p_H2, color=GREEN, lw=1.5, ls="--", alpha=0.7,
                   label=rf"H$_2$ price ({p_H2:.0f} \texteuro/kg)")
        ax.set_xticks(xm); ax.set_xticklabels(lbl, fontsize=8)
        ax.set_ylabel(r"\texteuro/kg H$_2$"); ax.set_title("Monthly LCOH")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Monthly view:\nannual runs only",
                ha="center", va="center", transform=ax.transAxes, color=GREY)
        ax.set_title("Monthly LCOH")

    # ── Panel 3: profit margin per kg (revenue − full LCOH) ─────────────────
    ax = axes[2]
    margins = [p_H2 - lc for lc in lcoh_vals]
    bars = ax.bar(x, margins, width=w_bar,
                  color=bar_clr, edgecolor="white", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"\texteuro/kg H$_2$"); ax.set_title(r"Profit Margin per kg H$_2$")
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    for b, v in zip(bars, margins):
        ypos = v + 0.03 if v >= 0 else v - 0.12
        ax.text(b.get_x() + b.get_width()/2, ypos,
                rf"{v:.2f} \texteuro/kg", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    better = int(np.argmax(margins))
    bars[better].set_edgecolor("black"); bars[better].set_linewidth(1.5)

    fig.tight_layout()
    savefig(fig, outdir/"comparison_fig10_lcoh.png")
    return fig


def fig_lcoh_composition(db, dc, da, de=None, outdir=None, figsize=(6.8, 4.2)):
    """Standalone LCOH composition stacked bar — no title, publication size."""
    if outdir is None:
        raise ValueError("outdir must be provided")
    kb = _kpi(db); ka = _kpi(da)
    kc = _kpi(dc) if dc is not None else None
    ke = _kpi(de) if de is not None else None

    def _per_kg(v, h2): return v / h2 if h2 > 0 else 0.0
    def _costs(k):
        h2  = k["h2_kg"]
        e   = _per_kg(k["elec_eur"],     h2)
        s   = _per_kg(k["shutdown_eur"], h2)
        d   = _replacement_lcoh_per_kg(k["h2_annual_kg"], k["life_cal_yr"])
        sys = _system_lcoh_per_kg(k["h2_annual_kg"])
        return e, s, d, sys, e + s + d + sys

    elec_b, shut_b, deg_b, sys_b, lcoh_b = _costs(kb)
    elec_a, shut_a, deg_a, sys_a, lcoh_a = _costs(ka)
    elec_c, shut_c, deg_c, sys_c, lcoh_c = _costs(kc) if kc else (0, 0, 0, 0, 0)
    elec_e, shut_e, deg_e, sys_e, lcoh_e = _costs(ke) if ke else (0, 0, 0, 0, 0)

    p_H2 = float(_plant["economics"]["p_H2_eur_per_kg"])

    if ke is not None:
        labels    = ["Load-following", "Price-aware", "Degradation-aware", "Lifetime-aware"]
        x         = np.array([0, 1, 2, 3])
        elec_vals = [elec_b, elec_c, elec_a, elec_e]
        shut_vals = [shut_b, shut_c, shut_a, shut_e]
        deg_vals  = [deg_b,  deg_c,  deg_a,  deg_e]
        sys_vals  = [sys_b,  sys_c,  sys_a,  sys_e]
        lcoh_vals = [lcoh_b, lcoh_c, lcoh_a, lcoh_e]
        w_bar     = 0.35
    elif kc is not None:
        labels    = ["Load-following", "Price-aware", "Degradation-aware"]
        x         = np.array([0, 1, 2])
        elec_vals = [elec_b, elec_c, elec_a]
        shut_vals = [shut_b, shut_c, shut_a]
        deg_vals  = [deg_b,  deg_c,  deg_a]
        sys_vals  = [sys_b,  sys_c,  sys_a]
        lcoh_vals = [lcoh_b, lcoh_c, lcoh_a]
        w_bar     = 0.4
    else:
        labels    = ["Load-following", "Degradation-aware"]
        x         = np.array([0, 1])
        elec_vals = [elec_b, elec_a]
        shut_vals = [shut_b, shut_a]
        deg_vals  = [deg_b,  deg_a]
        sys_vals  = [sys_b,  sys_a]
        lcoh_vals = [lcoh_b, lcoh_a]
        w_bar     = 0.5

    fig, ax = plt.subplots(figsize=figsize)
    elec_arr = np.array(elec_vals)
    shut_arr = np.array(shut_vals)
    deg_arr  = np.array(deg_vals)
    sys_arr  = np.array(sys_vals)
    ax.bar(x, elec_arr, width=w_bar, color=RED,    alpha=0.85, label="Electricity")
    ax.bar(x, shut_arr, width=w_bar, bottom=elec_arr,
           color=BLUE,  alpha=0.85, label="Shutdown")
    ax.bar(x, deg_arr,  width=w_bar, bottom=elec_arr + shut_arr,
           color=ORANGE, alpha=0.85, label="Degradation (stack)")
    ax.bar(x, sys_arr,  width=w_bar, bottom=elec_arr + shut_arr + deg_arr,
           color=GREY,  alpha=0.85, label="System CAPEX")
    ax.axhline(p_H2, color=GREEN, lw=2, ls="--",
               label=rf"H$_2$ price ({p_H2:.0f} \texteuro/kg)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"LCOH [\texteuro/kg H$_2$]")
    ax.legend(fontsize=9)
    for i, lc in enumerate(lcoh_vals):
        ax.text(x[i], lc + 0.06, rf"{lc:.2f} \texteuro/kg",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for i, lc in enumerate(lcoh_vals):
        if lc < p_H2:
            ax.fill_between([x[i]-w_bar/2, x[i]+w_bar/2], lc, p_H2,
                            alpha=0.10, color=GREEN)

    fig.tight_layout()
    png = outdir / "lcoh_composition.png"
    pdf = outdir / "lcoh_composition.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"  Saved: {png.name}")
    print(f"  Saved: {pdf.name}")
    return fig


# ── Summary stats ────────────────────────────────────────────────────────────
def print_summary(df, name):
    dt_h = float(col(df,"t_h")[1] - col(df,"t_h")[0]) if len(df) > 1 else 1.0
    a=mask(df, col(df,"j_A_per_m2")>100)
    P=col(a,"P_total_W"); h2=col(a,"m_dot_H2_kg_h")
    th=col(df,"t_h").max(); vmV=mV(df,"V_deg_V")[-1]

    # Operating hours and capacity factor
    n_op = len(a); n_total = len(df)
    op_hours = n_op * dt_h
    cap_factor = n_op / max(n_total, 1)

    rate_op     = vmV * 1000 / max(op_hours, 1)
    r_ann       = vmV / max(th / 8760, 1e-9)
    life_op_kh  = V_DEG_EOL_MV * 1000 / rate_op / 1000 if rate_op > 0 else float("inf")
    life_op_yr  = life_op_kh * 1000 / 8760
    life_cal_yr = V_DEG_EOL_MV / r_ann if r_ann > 0 else float("inf")

    print(f"\n{'='*52}\n  Summary: {name}\n{'='*52}")
    print(f"  Duration:              {th:.0f} h  ({th/8760*365:.0f} days)")
    print(f"  Total H₂ produced:     {col(df,'m_dot_H2_kg_h').sum()*dt_h:.2f} kg")
    print(f"  Operating hours:       {op_hours:.0f} / {n_total*dt_h:.0f}  ({100*cap_factor:.1f}%)")
    P_stk=col(a,"P_stack_W")
    eta_sys_annual = float(np.nansum(h2*HHV*1e3*dt_h) / np.nansum(P*dt_h) * 100) if np.nansum(P*dt_h) > 0 else 0.0
    eta_stk_annual = float(np.nansum(h2*HHV*1e3*dt_h) / np.nansum(P_stk*dt_h) * 100) if np.nansum(P_stk*dt_h) > 0 else 0.0
    SEC_annual = float(np.nansum(P/1e3*dt_h) / np.nansum(h2*dt_h)) if np.nansum(h2*dt_h) > 0 else 0.0
    print(f"  Mean η_system_HHV:     {eta_sys_annual:.1f}%")
    print(f"  Mean η_stack_HHV:      {eta_stk_annual:.1f}%")
    print(f"  Mean SEC (active):     {SEC_annual:.1f} kWh/kg H₂")
    print(f"  Mean j (active):       {col(a,'j_A_per_m2').mean()/10000:.2f} A/cm²")
    print(f"  Mean T_stack:          {degC(a,'T_stack_actual_K').mean():.1f}°C")
    print(f"  Electricity cost:      {np.nansum(col(df,'c_elec_eur_h'))*dt_h:.2f} €")
    print(f"  H₂ revenue:            {np.nansum(col(df,'r_H2_eur_h'))*dt_h:.2f} €")
    print(f"  Net profit:            {np.nansum(col(df,'true_profit_eur_h'))*dt_h:.2f} €")
    print(f"  Degradation cost:      {np.nansum(col(df,'c_deg_phys_eur_h'))*dt_h:.2f} €")
    print(f"  Final V_deg:           {vmV:.3f} mV")
    print(f"  Degradation rate:      {rate_op:.3f} µV/op.h")
    print(f"  Op. lifetime:          {life_op_kh:.1f} kh_op  /  {life_op_yr:.1f} yr_op")
    print(f"  Calendar lifetime:     {life_cal_yr:.1f} yr  (repeated annual duty cycle)")
    print(f"{'='*52}")

# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--input",        type=Path, default=None)
    p.add_argument("--load-following",      type=Path, default=None, dest="load_following")
    p.add_argument("--price-aware",         type=Path, default=None, dest="price_aware")
    p.add_argument("--degradation-aware",   type=Path, default=None, dest="degradation_aware")
    p.add_argument("--lifetime-aware",      type=Path, default=None, dest="lifetime_aware")
    p.add_argument("--name",             type=str,  default=None)
    p.add_argument("--outdir",           type=Path, default=Path("results/figures"))
    p.add_argument("--lcoh-composition", action="store_true",
                   help="Output standalone LCOH composition panel only (PDF + PNG)")
    return p.parse_args()

def main():
    args=parse_args(); args.outdir.mkdir(parents=True,exist_ok=True); figs=[]
    if args.input:
        df=pd.read_csv(args.input); name=args.name or args.input.stem
        print(f"\nPlotting: {args.input}  ({len(df)} steps)")
        print_summary(df,name)
        figs+=[fig_stats_table(df,name,args.outdir),
               fig_dynamic_overview(df,name,args.outdir),
               fig_dynamic_zoom(df,name,args.outdir),
               fig_detail_10h(df,name,args.outdir),
               fig_polarization(df,name,args.outdir),
               fig_efficiency(df,name,args.outdir),
               fig_economics(df,name,args.outdir),
               fig_degradation(df,name,args.outdir),
               fig_thermal(df,name,args.outdir)]
        figs = [f for f in figs if f is not None]
    if args.load_following and args.degradation_aware:
        db=pd.read_csv(args.load_following)
        dc=pd.read_csv(args.price_aware) if args.price_aware else None
        da=pd.read_csv(args.degradation_aware)
        de=pd.read_csv(args.lifetime_aware) if args.lifetime_aware else None
        print_summary(db,args.load_following.stem)
        if dc is not None: print_summary(dc,args.price_aware.stem)
        print_summary(da,args.degradation_aware.stem)
        if de is not None: print_summary(de,args.lifetime_aware.stem)
        if args.lcoh_composition:
            fig_lcoh_composition(db,dc,da,de,args.outdir)
            return
        figs += [fig_comparison(db,dc,da,de,args.outdir),
                 fig_rul_comparison(db,dc,da,de,args.outdir),
                 fig_kpi_summary(db,dc,da,de,args.outdir),
                 fig_distributions(db,dc,da,de,args.outdir),
                 fig_lcoh_breakdown(db,dc,da,de,args.outdir)]
    if not figs:
        print("Specify --input  or  --load-following + --degradation-aware"); sys.exit(1)
    pdf_name=(args.name or (args.input.stem if args.input else "comparison"))+"_all_figures.pdf"
    try:
        with PdfPages(args.outdir/pdf_name) as pdf:
            for fig in figs:
                pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
        print(f"\n  PDF: {args.outdir/pdf_name}\nDone.")
    except Exception as e:
        print(f"\n  [PDF skipped ---{e}]\nDone.")

if __name__=="__main__":
    main()
