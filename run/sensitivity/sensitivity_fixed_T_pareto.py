"""
sensitivity_fixed_T_pareto.py
==============================

Tests whether the aware controller's dynamic temperature management genuinely
outperforms simpler fixed-setpoint policies on the profit-vs-degradation Pareto
frontier.

For each fixed temperature T ∈ {50, 52.5, 55, 57.5, 60, 62.5, 65, 67.5, 70}°C, a
degradation-aware j-optimizer runs with T pinned (T_fixed_K in ctrl_cfg).
This represents the literature state of the art: include degradation in the j
objective, but do not co-optimise temperature.

The four existing controllers (commercial, cost_optimal, aware, aware_rul) are
loaded from pre-computed CSVs and added to the same plot. A 5 year simulation is chosen for comparison,
to also see the effect of accumulated degrdation, which Lifetime-aware should mitigate better than Degradation-aware.

If the aware controller lies on the Pareto frontier — strictly better profit
AND strictly less degradation than every fixed-T policy — then dynamic
temperature variation adds value that no single setpoint can replicate.
If it merely clusters near a fixed temperature point, a simpler fixed-T policy would
suffice.

Usage
-----
Run all fixed-T simulations:
    python3 run/sensitivity/sensitivity_fixed_T_pareto.py --run

Plot only (after simulations are done):
    python3 run/sensitivity/sensitivity_fixed_T_pareto.py --plots-only

Run a single temperature (useful for parallelism):
    python3 run/sensitivity/sensitivity_fixed_T_pareto.py --run --only-T 60

Output
------
    results/sensitivity_fixed_T/fixedT_XXC/fixedT_XXC.csv   (one per T)
    results/sensitivity_fixed_T_pareto.png
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pemwe.plant import load_plant, load_config
from pemwe.simulation import run_simulation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIXED_T_DEGC = [55, 57.5, 60, 62.5, 65, 67.5, 70]   # °C

N_YEARS      = 5
HOURS_PER_YR = 8760
N_STEPS      = N_YEARS * HOURS_PER_YR
DT_S         = 3600.0

PLANT_CFG  = REPO_ROOT / "configs" / "plant_parameters.yaml"
POWER_CFG  = REPO_ROOT / "configs" / "power_profiles" / "wind.yaml"
PRICE_CFG  = REPO_ROOT / "configs" / "price_profiles" / "spot_dk1.yaml"

# System CAPEX term (same as make_plots.py)
_plant_econ    = load_plant(PLANT_CFG)["economics"]
_P_RATED_KW    = load_plant(PLANT_CFG)["stack"]["P_rating_W"] / 1000.0
_SYS_CAPEX_EUR  = float(_plant_econ["system_capex_eur_per_kW"]) * _P_RATED_KW
_SYS_LIFE_YR    = float(_plant_econ["system_lifetime_yr"])
_WACC           = float(_plant_econ.get("wacc", 0.0))
_ANNUITY        = (1 - (1 + _WACC) ** -_SYS_LIFE_YR) / _WACC if _WACC > 0 else _SYS_LIFE_YR
_STACK_REPL_EUR = float(_plant_econ["capex_usd_per_kW"]) * float(_plant_econ["eur_per_usd"]) * _P_RATED_KW

OUT_BASE = REPO_ROOT / "results" / "sensitivity_fixed_T_5yr"

# Existing controller CSVs (pre-computed 5-year runs)
EXISTING = {
    "Load-following":   REPO_ROOT / "results" / "lifetime_5yr" / "load_following_5yr"   / "load_following_5yr.csv",
    "Price-aware": REPO_ROOT / "results" / "lifetime_5yr" / "price_aware_5yr" / "price_aware_5yr.csv",
    "Degradation-aware":        REPO_ROOT / "results" / "lifetime_5yr" / "degradation_aware_5yr"        / "degradation_aware_5yr.csv",
    "Lifetime-aware":    REPO_ROOT / "results" / "lifetime_5yr" / "lifetime_aware_5yr"    / "lifetime_aware_5yr.csv",
}

EXISTING_COLORS = {
    "Load-following":    "#2C73D2",   # BLUE
    "Price-aware":       "#44BBA4",   # GREEN
    "Degradation-aware": "#FF6B35",   # ORANGE
    "Lifetime-aware":    "#7B2D8B",   # PURPLE
}
EXISTING_MARKERS = {
    "Load-following":   "s",
    "Price-aware": "D",
    "Degradation-aware":        "*",
    "Lifetime-aware":    "P",
}

# ---------------------------------------------------------------------------
# Helper: build ctrl_cfg for a fixed-T deg-aware run
# ---------------------------------------------------------------------------

def fixed_T_ctrl_cfg(T_K: float) -> dict:
    return {
        "type": "supervisory_optimizer",
        "T_fixed_K": T_K,
        "objective": {
            "include_degradation": True,
            "ignore_arrhenius_in_deg": True,   # j-only degradation in objective (no Arrhenius)
            "use_eis_feedback": False,
            "eis_feedback_gain": 0.0,
        },
        "solver": {
            "name": "cobyqa",
            "max_iter": 500,
            "tol": 1.0e-4,
        },
    }


# ---------------------------------------------------------------------------
# Helper: extract KPIs from a result CSV
# ---------------------------------------------------------------------------

def kpis(csv_path: Path) -> dict:
    df       = pd.read_csv(csv_path)
    dt_h     = 1.0
    n_steps  = len(df)
    running  = df["j_A_per_m2"] > 100

    profit        = df["true_profit_eur_h"].sum() * dt_h
    annual_profit = profit / N_YEARS
    vdeg_mV       = df["V_deg_V"].iloc[-1] * 1e3
    deg_rate      = df["dV_deg_V"].sum() / n_steps   # cumulative rate, robust to EOL resets
    lifetime_yr   = (0.1 / deg_rate / 8760.0) if deg_rate > 0 else float("inf")  # first stack only
    total_H2      = (df["m_dot_H2_kg_h"] * dt_h).sum()
    H2_per_yr     = total_H2 / N_YEARS
    total_cost    = ((df["c_elec_eur_h"]) * dt_h + df["c_shutdown_eur"]).sum()
    h2_annual     = total_H2 / N_YEARS
    # NPV-discounted replacement cost (Eq. 20)
    if lifetime_yr > 0 and np.isfinite(lifetime_yr) and h2_annual > 0:
        times = np.arange(lifetime_yr, _SYS_LIFE_YR + lifetime_yr, lifetime_yr)
        times = times[times <= _SYS_LIFE_YR]
        npv_rep = sum(_STACK_REPL_EUR / (1 + _WACC) ** t for t in times) if len(times) > 0 else 0.0
        lcoh_rep = npv_rep / (h2_annual * _ANNUITY)
    else:
        lcoh_rep = 0.0
    lcoh_sys      = _SYS_CAPEX_EUR / (h2_annual * _ANNUITY) if h2_annual > 0 else 0.0
    lcoh          = total_cost / total_H2 + lcoh_rep + lcoh_sys if total_H2 > 0 else float("nan")
    T_mean_C   = df.loc[running, "T_stack_actual_K"].mean() - 273.15
    j_mean     = df.loc[running, "j_A_per_m2"].mean()

    return dict(profit=profit, annual_profit=annual_profit,
                lcoh=lcoh, vdeg_mV=vdeg_mV, lifetime_yr=lifetime_yr,
                H2_per_yr=H2_per_yr, T_mean_C=T_mean_C, j_mean=j_mean)


# ---------------------------------------------------------------------------
# Run a fixed-T simulation
# ---------------------------------------------------------------------------

def _tile_profile(values: list, n: int) -> list:
    """Tile a 1-year profile to n steps, wrapping as needed."""
    arr   = np.array(values)
    valid = arr[~np.isnan(arr)]
    reps  = (n // len(valid)) + 1
    return np.tile(valid, reps)[:n].tolist()


def run_fixed_T(T_degC: float, plant: dict, power_cfg: dict, price_cfg: dict) -> Path:
    T_K   = T_degC + 273.15
    # Use 1 decimal place in name only when needed (avoids "fixedT_52C" vs "fixedT_52.5C")
    name  = f"fixedT_{T_degC:g}C"
    outdir = OUT_BASE / name
    csv_path = outdir / f"{name}.csv"

    if csv_path.exists():
        print(f"  [skip] {name} — CSV already exists")
        return csv_path

    print(f"  [run]  {name}  (T_target = {T_degC:g} °C, {N_YEARS} years) ...")

    # Build tiled 5-year profiles
    from run.single_simulation import load_power_profile, load_price_profile
    p_1yr   = load_power_profile(power_cfg, HOURS_PER_YR, DT_S, REPO_ROOT)
    pr_1yr  = load_price_profile(price_cfg, HOURS_PER_YR, REPO_ROOT)
    p_avail = _tile_profile(p_1yr,  N_STEPS)
    price   = _tile_profile(pr_1yr, N_STEPS)

    ctrl_cfg = fixed_T_ctrl_cfg(T_K)

    df = run_simulation(
        price_series=price,
        p_avail_series=p_avail,
        plant=plant,
        ctrl_cfg=ctrl_cfg,
        dt_s=DT_S,
        detail_window_h=(2386, 2396),
    )

    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    k = kpis(csv_path)
    print(f"    profit={k['profit']:+.0f} EUR  V_deg={k['vdeg_mV']:.2f} mV  "
          f"T_mean={k['T_mean_C']:.1f} °C  j_mean={k['j_mean']:.0f} A/m2")
    return csv_path


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------

def is_pareto_efficient(profits: np.ndarray, vdegs: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-efficient points (max profit, min vdeg)."""
    n = len(profits)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if profits[j] >= profits[i] and vdegs[j] <= vdegs[i]:
                if profits[j] > profits[i] or vdegs[j] < vdegs[i]:
                    dominated[i] = True
                    break
    return ~dominated


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_COL_REF = "#6C757D"
_COL_LEG = "#9E9E9E"   # neutral grey used for all legend marker fills

# Named-controller marker shapes (colour and size set at plot time)
_CTRL_MARKER = {
    "Load-following":   dict(marker="s", zorder=5, label="Load-following"),
    "Price-aware": dict(marker="D", zorder=5, label="Price-aware"),
    "Degradation-aware":        dict(marker="*", zorder=6, label=r"Degradation-aware"),
    "Lifetime-aware":    dict(marker="P", zorder=5, label="Lifetime-aware"),
}

# Three-level j size encoding — thresholds sit in the natural gaps of the data
# Low  : j < 0.36  →  fixed-T high-T cases only   (0.33–0.34 A/cm²)
# Med  : 0.36–0.60 →  aware, fixed-T low-T, cost-optimal (0.38–0.46 A/cm²)
# High : j > 0.60  →  commercial only              (0.71 A/cm²)
_J_LOW   = 0.36
_J_HIGH  = 0.60
_SZ_SMALL  = 55
_SZ_MEDIUM = 145
_SZ_LARGE  = 340

def _j_size(j_cm2: float, star: bool = False) -> float:
    """Map j [A/cm²] to one of three discrete marker areas."""
    if j_cm2 < _J_LOW:
        s = _SZ_SMALL
    elif j_cm2 > _J_HIGH:
        s = _SZ_LARGE
    else:
        s = _SZ_MEDIUM
    return s * 2.2 if star else s

# Shared colormap: T_set for fixed-T circles, T_run for named controllers
_T_NORM_MIN = 55.0
_T_NORM_MAX = 70.0


def _make_figure(fixed_T_results: list, existing_results: dict,
                 x_metric: str = "lcoh") -> None:
    """
    x_metric : "lcoh"   → LCOH [€/kg H₂]                    (field-standard)
               "h2"     → Mean annual H₂ production [t/yr]  (physical)
               "profit" → Mean annual profit [k€/yr]         (economic)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.colors import LinearSegmentedColormap

    sys.path.insert(0, str(REPO_ROOT / "run"))
    from plot_style import apply_style, GREEN, ORANGE, DARK
    apply_style()
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage{eurosym}\usepackage{amssymb}"
    )

    cmap = LinearSegmentedColormap.from_list(
        "pemwe_T", [(0.0, GREEN), (0.30, "#FFD9A8"), (0.65, "#FF9A50"), (1.0, ORANGE)], N=256
    )
    norm = plt.Normalize(_T_NORM_MIN, _T_NORM_MAX)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    # ── Build x values ────────────────────────────────────────────────────────
    def _xval(r):
        if x_metric == "lcoh":   return r["lcoh"]
        if x_metric == "h2":     return r["H2_per_yr"] / 1e3
        return r["annual_profit"] / 1e3

    if x_metric == "lcoh":
        xlabel   = r"LCOH [\euro/kg$_{\mathrm{H_2}}$]"
        x_invert = True    # lower LCOH is better → invert so "better" is right
    elif x_metric == "h2":
        xlabel   = r"Mean annual H$_2$ production [t/yr]"
        x_invert = False
    else:
        xlabel   = r"Mean annual profit [k\euro/yr]"
        x_invert = False

    # ── Fixed-T circles: colour = T_set, size = j_run ────────────────────────
    Ts  = np.array([r["T_set_C"]     for r in fixed_T_results])
    js  = np.array([r["j_mean"]       for r in fixed_T_results]) / 1e4   # A/cm²
    lts = np.array([r["lifetime_yr"] for r in fixed_T_results])
    xs  = np.array([_xval(r)         for r in fixed_T_results])

    sizes = np.array([_j_size(j) for j in js])

    # Scatter — no legend entries (all proxies are explicit below)
    sc = ax.scatter(xs, lts, c=Ts, cmap=cmap, norm=norm,
                    s=sizes, zorder=3, marker="o",
                    edgecolors=DARK, linewidths=0.55, alpha=0.92,
                    label="_nolegend_")

    # Guide-line through fixed-T points in T_set order
    ax.plot(xs, lts, ls="--", color=_COL_REF, lw=0.9, alpha=0.45, zorder=2)

    # ── Named controllers: colour = T_run, size = j level ────────────────────
    for name, k in existing_results.items():
        st = _CTRL_MARKER.get(name)
        if st is None:
            continue
        face = cmap(norm(k["T_mean_C"]))
        star = st["marker"] == "*"
        ax.scatter(_xval(k), k["lifetime_yr"],
                   color=face,
                   s=_j_size(k["j_mean"] / 1e4, star=star),
                   marker=st["marker"], zorder=st["zorder"],
                   edgecolors=DARK, linewidths=0.7,
                   label="_nolegend_")

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cb = fig.colorbar(sc, ax=ax, shrink=0.88, pad=0.02)
    cb.set_label(r"$T$ [$^\circ$C]", fontsize=10.5)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel(xlabel,                               fontsize=11)
    ax.set_ylabel(r"Projected stack lifetime [yr]",     fontsize=11)
    if x_invert:
        ax.invert_xaxis()   # lower LCOH → right = better

    # ── Legend — grey fill everywhere; colour read from temperature bar ─────────
    proxies = []

    # Controller / fixed-T shape entries (all grey, uniform size)
    for name, _ in existing_results.items():
        st = _CTRL_MARKER.get(name)
        if st is None:
            continue
        ms = 11 if st["marker"] == "*" else 8
        proxies.append(
            mlines.Line2D([], [], marker=st["marker"], linestyle="none",
                          markerfacecolor=_COL_LEG, markeredgecolor=DARK,
                          markeredgewidth=0.6, markersize=ms,
                          label=st["label"])
        )
    proxies.append(
        mlines.Line2D([], [], marker="o", linestyle="none",
                      markerfacecolor=_COL_LEG, markeredgecolor=DARK,
                      markeredgewidth=0.5, markersize=8,
                      label=r"Fixed-$T$")
    )
    # Size encoding (low/med/high j) is described in the figure caption.

    ax.legend(handles=proxies, fontsize=8.5, loc="lower right",
              framealpha=0.95, edgecolor=_COL_REF, borderpad=0.8,
              handletextpad=0.6)

    fig.tight_layout()

    for ext in (".pdf", ".png"):
        out = REPO_ROOT / "results" / f"sensitivity_fixed_T_pareto_5yr{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"  Saved: {out.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run",        action="store_true", help="Run fixed-T simulations")
    p.add_argument("--plots-only", action="store_true", help="Skip simulations, plot only")
    p.add_argument("--only-T",     type=float, default=None,
                   help="Run only this temperature (°C)")
    return p.parse_args()


def main():
    args = parse_args()

    plant     = load_plant(PLANT_CFG)
    power_cfg = load_config(POWER_CFG)
    price_cfg = load_config(PRICE_CFG)

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # --- Run simulations ---
    if args.run and not args.plots_only:
        temps = [args.only_T] if args.only_T is not None else FIXED_T_DEGC
        print(f"\nRunning {len(temps)} fixed-T simulations...")
        for T in temps:
            run_fixed_T(T, plant, power_cfg, price_cfg)

    # --- Collect KPIs ---
    fixed_T_results = []
    for T in FIXED_T_DEGC:
        name = f"fixedT_{T:g}C"
        csv_path = OUT_BASE / name / f"{name}.csv"
        if csv_path.exists():
            k = kpis(csv_path)
            k["T_set_C"] = T
            k["label"]   = f"{T:g}$^\\circ$C"
            fixed_T_results.append(k)
        else:
            print(f"  [missing] {csv_path} — run with --run first")

    existing_results = {}
    for label, csv_path in EXISTING.items():
        if csv_path.exists():
            existing_results[label] = kpis(csv_path)
        else:
            print(f"  [missing] {csv_path}")

    if not fixed_T_results and not existing_results:
        print("No results found. Run with --run first.")
        return

    # --- Print table ---
    all_rows = []
    for r in fixed_T_results:
        all_rows.append(("Fixed T", r["label"], r["T_mean_C"], r["j_mean"],
                          r["profit"], r["vdeg_mV"], r["lcoh"], r["lifetime_yr"]))
    for label, k in existing_results.items():
        all_rows.append(("Controller", label, k["T_mean_C"], k["j_mean"],
                          k["profit"], k["vdeg_mV"], k["lcoh"], k["lifetime_yr"]))

    print(f"\n{'Type':<12}  {'Name':<20}  {'T_mean':>7}  {'j_mean':>7}  "
          f"{'Profit':>10}  {'V_deg':>8}  {'LCOH':>7}  {'Life':>6}")
    print("-" * 92)
    for row in sorted(all_rows, key=lambda x: x[5]):  # sort by degradation
        life = f"{row[7]:.1f}" if row[7] < 99 else ">99"
        print(f"{row[0]:<12}  {row[1]:<20}  {row[2]:>6.1f}C  {row[3]:>7.0f}  "
              f"{row[4]:>+10.0f}  {row[5]:>7.2f} mV  {row[6]:>6.2f}  {life:>5} yr")

    # --- Plot ---
    _make_figure(fixed_T_results, existing_results, x_metric="lcoh")


if __name__ == "__main__":
    main()
