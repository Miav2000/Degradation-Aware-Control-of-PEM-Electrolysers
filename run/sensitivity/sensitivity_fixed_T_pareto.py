"""
sensitivity_fixed_T_pareto.py
==============================

Quantifies the value of dynamic temperature optimisation: does jointly
optimising (j, T) outperform the simpler approach of optimising j alone at
a fixed temperature setpoint?

A degradation-aware j-optimiser is run at fixed T ∈ {56, 58, 60, 62, 64, 66, 68,
70}°C. If the full aware controllers (which co-optimise temperature) lie beyond
the Pareto frontier of this fixed-T sweep in LCOH vs. lifetime space, dynamic
temperature management adds irreplaceable value. If they cluster near a fixed-T
point, a simpler setpoint policy would suffice.

The analysis is repeated for Ea_eff = 26, 55, and 70 kJ/mol to test whether
the conclusion holds regardless of how temperature-sensitive degradation is.

Usage
-----
Run new Ea=26 and Ea=70 simulations (Ea=55 cached, will be skipped):
    python3 run/sensitivity/sensitivity_fixed_T_pareto.py --run

Plot only (after simulations are done):
    python3 run/sensitivity/sensitivity_fixed_T_pareto.py --plots-only

Run a specific Ea value and/or a single temperature:
    python3 run/sensitivity/sensitivity_fixed_T_pareto.py --run --ea 26
    python3 run/sensitivity/sensitivity_fixed_T_pareto.py --run --ea 70 --only-T 60

Output
------
    results/sensitivity_fixed_T_5yr/fixedT_XXC/fixedT_XXC.csv        (Ea=55, existing)
    results/sensitivity_fixed_T_5yr_Ea26/fixedT_XXC/fixedT_XXC.csv   (Ea=26, new)
    results/sensitivity_fixed_T_5yr_Ea70/fixedT_XXC/fixedT_XXC.csv   (Ea=70, new)
    results/sensitivity_fixed_T_pareto_5yr.pdf/.png         (Ea=55 figure)
    results/sensitivity_fixed_T_pareto_5yr_Ea26.pdf/.png    (Ea=26 figure)
    results/sensitivity_fixed_T_pareto_5yr_Ea70.pdf/.png    (Ea=70 figure)
    results/sensitivity_fixed_T_Ea_comparison_table.tex     (LaTeX comparison table)
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

FIXED_T_DEGC = [56, 58, 60, 62, 64, 66, 68, 70]   # °C

# Activation energy cases [kJ/mol]. 55 = base (existing CSVs); 26 and 70 are new.
EA_CASES = [26, 55, 70]

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

# Ea=55 base output directory (existing results live here)
OUT_BASE = REPO_ROOT / "results" / "sensitivity_fixed_T_5yr"

# Existing controller CSVs (pre-computed 5-year runs)
EXISTING = {
    "Load-following":    REPO_ROOT / "results" / "lifetime_5yr" / "load_following_5yr"    / "load_following_5yr.csv",
    "Price-aware":       REPO_ROOT / "results" / "lifetime_5yr" / "price_aware_5yr"       / "price_aware_5yr.csv",
    "Degradation-aware": REPO_ROOT / "results" / "lifetime_5yr" / "degradation_aware_5yr" / "degradation_aware_5yr.csv",
    "Lifetime-aware":    REPO_ROOT / "results" / "lifetime_5yr" / "lifetime_aware_5yr"    / "lifetime_aware_5yr.csv",
}

EXISTING_COLORS = {
    "Load-following":    "#2C73D2",
    "Price-aware":       "#44BBA4",
    "Degradation-aware": "#FF6B35",
    "Lifetime-aware":    "#7B2D8B",
}
EXISTING_MARKERS = {
    "Load-following":   "s",
    "Price-aware":      "D",
    "Degradation-aware": "*",
    "Lifetime-aware":   "P",
}


# ---------------------------------------------------------------------------
# Ea helpers
# ---------------------------------------------------------------------------

def _out_base(ea_kJ: float) -> Path:
    """Output directory for a given Ea value."""
    if ea_kJ == 55:
        return OUT_BASE
    return REPO_ROOT / "results" / f"sensitivity_fixed_T_5yr_Ea{ea_kJ:g}"


def _apply_ea(base_plant: dict, ea_kJ: float) -> dict:
    """Return a deep copy of plant with Ea_eff_J_per_mol overridden."""
    p = copy.deepcopy(base_plant)
    p["degradation"]["Ea_eff_J_per_mol"] = ea_kJ * 1e3
    return p


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
    deg_rate      = df["dV_deg_V"].sum() / n_steps
    lifetime_yr   = (0.1 / deg_rate / 8760.0) if deg_rate > 0 else float("inf")
    total_H2      = (df["m_dot_H2_kg_h"] * dt_h).sum()
    H2_per_yr     = total_H2 / N_YEARS
    total_cost    = ((df["c_elec_eur_h"]) * dt_h + df["c_shutdown_eur"]).sum()
    h2_annual     = total_H2 / N_YEARS
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
    arr   = np.array(values)
    valid = arr[~np.isnan(arr)]
    reps  = (n // len(valid)) + 1
    return np.tile(valid, reps)[:n].tolist()


def run_fixed_T(T_degC: float, plant: dict, power_cfg: dict, price_cfg: dict,
                out_base: Path) -> Path:
    T_K      = T_degC + 273.15
    name     = f"fixedT_{T_degC:g}C"
    outdir   = out_base / name
    csv_path = outdir / f"{name}.csv"

    if csv_path.exists():
        print(f"  [skip] {name} — CSV already exists")
        return csv_path

    print(f"  [run]  {name}  (T_target = {T_degC:g} °C, {N_YEARS} years) ...")

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
# Run a DA joint (j+T) simulation for a non-base Ea value
# ---------------------------------------------------------------------------

def run_joint_da(plant: dict, power_cfg: dict, price_cfg: dict,
                 out_base: Path) -> Path:
    """Run DA with joint j+T optimisation at a modified Ea and save results."""
    name     = "joint_da"
    outdir   = out_base / name
    csv_path = outdir / f"{name}.csv"

    if csv_path.exists():
        print(f"  [skip] {name} — CSV already exists")
        return csv_path

    print(f"  [run]  {name}  (joint j+T, {N_YEARS} years) ...")

    from run.single_simulation import load_power_profile, load_price_profile
    p_1yr   = load_power_profile(power_cfg, HOURS_PER_YR, DT_S, REPO_ROOT)
    pr_1yr  = load_price_profile(price_cfg, HOURS_PER_YR, REPO_ROOT)
    p_avail = _tile_profile(p_1yr,  N_STEPS)
    price   = _tile_profile(pr_1yr, N_STEPS)

    ctrl_cfg = load_config(REPO_ROOT / "configs" / "controllers" / "degradation_aware.yaml")

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
# Figure (per Ea)
# ---------------------------------------------------------------------------

_COL_REF = "#6C757D"
_COL_LEG = "#9E9E9E"

_CTRL_MARKER = {
    "Load-following":    dict(marker="s", zorder=5, label="Load-following"),
    "Price-aware":       dict(marker="D", zorder=5, label="Price-aware"),
    "Degradation-aware": dict(marker="*", zorder=6, label=r"Degradation-aware"),
    "Lifetime-aware":    dict(marker="P", zorder=5, label="Lifetime-aware"),
}

_J_LOW     = 0.36
_J_HIGH    = 0.60
_SZ_SMALL  = 55
_SZ_MEDIUM = 145
_SZ_LARGE  = 340

def _j_size(j_cm2: float, star: bool = False) -> float:
    if j_cm2 < _J_LOW:
        s = _SZ_SMALL
    elif j_cm2 > _J_HIGH:
        s = _SZ_LARGE
    else:
        s = _SZ_MEDIUM
    return s * 2.2 if star else s

_T_NORM_MIN = 55.0
_T_NORM_MAX = 70.0


def _make_figure(fixed_T_results: list, existing_results: dict,
                 ea_kJ: float, x_metric: str = "lcoh") -> None:
    """
    Pareto figure for one Ea value.
    ea_kJ  : activation energy [kJ/mol] — used for filename and subtitle.
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

    def _xval(r):
        if x_metric == "lcoh":   return r["lcoh"]
        if x_metric == "h2":     return r["H2_per_yr"] / 1e3
        return r["annual_profit"] / 1e3

    if x_metric == "lcoh":
        xlabel   = r"LCOH [\euro/kg$_{\mathrm{H_2}}$]"
        x_invert = True
    elif x_metric == "h2":
        xlabel   = r"Mean annual H$_2$ production [t/yr]"
        x_invert = False
    else:
        xlabel   = r"Mean annual profit [k\euro/yr]"
        x_invert = False

    Ts  = np.array([r["T_set_C"]     for r in fixed_T_results])
    js  = np.array([r["j_mean"]       for r in fixed_T_results]) / 1e4
    lts = np.array([r["lifetime_yr"] for r in fixed_T_results])
    xs  = np.array([_xval(r)         for r in fixed_T_results])

    sizes = np.array([_j_size(j) for j in js])

    sc = ax.scatter(xs, lts, c=Ts, cmap=cmap, norm=norm,
                    s=sizes, zorder=3, marker="o",
                    edgecolors=DARK, linewidths=0.55, alpha=0.92,
                    label="_nolegend_")
    ax.plot(xs, lts, ls="--", color=_COL_REF, lw=0.9, alpha=0.45, zorder=2)

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

    cb = fig.colorbar(sc, ax=ax, shrink=0.88, pad=0.02)
    cb.set_label(r"$T$ [$^\circ$C]", fontsize=13)

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(r"Projected stack lifetime [yr]", fontsize=15)
    if x_invert:
        ax.invert_xaxis()

    # Subtitle with Ea value
    ea_label = rf"$E_{{\mathrm{{a,eff}}}} = {ea_kJ:g}$ kJ/mol"
    ax.set_title(ea_label, fontsize=13, pad=4)

    proxies = []
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
    ax.legend(handles=proxies, fontsize=11, loc="lower right",
              framealpha=0.95, edgecolor=_COL_REF, borderpad=0.8,
              handletextpad=0.6)

    fig.tight_layout()

    suffix = "" if ea_kJ == 55 else f"_Ea{ea_kJ:g}"
    for ext in (".pdf", ".png"):
        out = REPO_ROOT / "results" / f"sensitivity_fixed_T_pareto_5yr{suffix}{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"  Saved: {out.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ea comparison table (LaTeX + stdout)
# ---------------------------------------------------------------------------

def _make_ea_comparison_table(all_ea: dict[float, tuple[list, dict]]) -> None:
    """
    Print a comparison table and save a LaTeX booktabs table.

    all_ea : {ea_kJ: (fixed_T_results, existing_results)}
    Rows   : each fixed-T setpoint, then each named controller.
    Columns: Ea=26 | Ea=55 | Ea=70  each showing LCOH [EUR/kg] and lifetime [yr].
    """
    ea_vals = sorted(all_ea.keys())

    # Collect all row keys in display order
    ctrl_order = ["Load-following", "Price-aware", "Degradation-aware", "Lifetime-aware"]

    # For each (ea, row) fetch lcoh and lifetime_yr
    def _get(ea, T=None, ctrl=None):
        fixed_rows, existing = all_ea[ea]
        if T is not None:
            for r in fixed_rows:
                if r["T_set_C"] == T:
                    return r["lcoh"], r["lifetime_yr"]
            return float("nan"), float("nan")
        if ctrl is not None:
            k = existing.get(ctrl)
            if k is None:
                return float("nan"), float("nan")
            return k["lcoh"], k["lifetime_yr"]

    def _fmt_life(v):
        if not np.isfinite(v):
            return r"$>$99"
        return f"{v:.1f}"

    # ----- stdout table -----
    col_w = 17
    header1 = f"{'':28s}" + "".join(f"{'Ea='+str(int(e))+' kJ/mol':^{col_w}s}" for e in ea_vals)
    header2 = f"{'':28s}" + "".join(f"{'LCOH':>7s}  {'Life':>6s}   " for _ in ea_vals)
    sep     = "-" * (28 + col_w * len(ea_vals))
    print(f"\n{'Ea sensitivity comparison':^{len(sep)}}")
    print(sep)
    print(header1)
    print(header2)
    print(sep)

    def _print_row(label, T=None, ctrl=None):
        cells = ""
        for ea in ea_vals:
            lcoh, lt = _get(ea, T=T, ctrl=ctrl)
            lcoh_s = f"{lcoh:.2f}" if np.isfinite(lcoh) else "  n/a"
            lt_s   = _fmt_life(lt)
            cells += f"  {lcoh_s:>6s}  {lt_s:>5s}  "
        print(f"  {label:<26s}{cells}")

    print("  Fixed-T setpoints")
    for T in FIXED_T_DEGC:
        _print_row(f"  T = {T:g} °C", T=T)
    print(sep)
    print("  Named controllers")
    for ctrl in ctrl_order:
        _print_row(f"  {ctrl}", ctrl=ctrl)
    print(sep)

    # ----- LaTeX table -----
    n_ea = len(ea_vals)
    col_spec = "l" + "".join("rr" for _ in ea_vals)
    ea_headers = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{$E_{{\mathrm{{a,eff}}}}={int(e)}\,\mathrm{{kJ/mol}}$}}"
        for e in ea_vals
    )
    sub_header = " & ".join(
        [r""] + [r"LCOH [\euro/kg] & Life [yr]"] * n_ea
    )
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + 2*i}--{3 + 2*i}}}"
        for i in range(n_ea)
    )

    def _latex_row(label, T=None, ctrl=None):
        cells = []
        for ea in ea_vals:
            lcoh, lt = _get(ea, T=T, ctrl=ctrl)
            lcoh_s = f"{lcoh:.2f}" if np.isfinite(lcoh) else r"\text{--}"
            lt_s   = _fmt_life(lt) if np.isfinite(lt) else r"$>$99"
            cells.append(f"{lcoh_s} & {lt_s}")
        return f"  {label} & " + " & ".join(cells) + r" \\"

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        (r"  \caption{Fixed-$T$ Pareto sensitivity to activation energy $E_{\mathrm{a,eff}}$."
         r" Each cell reports LCOH [\euro/kg$_{\mathrm{H_2}}$] and projected stack lifetime [yr]"
         r" for the given $E_{\mathrm{a,eff}}$ value. Fixed-$T$ rows use the degradation-aware"
         r" $j$-optimiser with temperature pinned; named controllers use the pre-computed"
         r" 5-year runs (Ea = 55\,kJ/mol for the named controllers).}"),
        r"  \label{tab:ea_sensitivity}",
        r"  \small",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    & {ea_headers} \\\\",
        f"    {cmidrules}",
        f"    {sub_header} \\\\",
        r"    \midrule",
        r"    \multicolumn{" + str(1 + 2*n_ea) + r"}{l}{\textit{Fixed-$T$ setpoints}} \\",
    ]
    for T in FIXED_T_DEGC:
        lines.append(_latex_row(f"\\quad $T={T:g}\\,^\\circ$C", T=T))
    lines += [
        r"    \midrule",
        r"    \multicolumn{" + str(1 + 2*n_ea) + r"}{l}{\textit{Named controllers}} \\",
    ]
    for ctrl in ctrl_order:
        lines.append(_latex_row(f"\\quad {ctrl}", ctrl=ctrl))
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex_path = REPO_ROOT / "results" / "sensitivity_fixed_T_Ea_comparison_table.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"\n  Saved LaTeX table: {tex_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run",        action="store_true", help="Run fixed-T simulations")
    p.add_argument("--plots-only", action="store_true", help="Skip simulations, plot only")
    p.add_argument("--only-T",     type=float, default=None,
                   help="Run only this temperature (°C)")
    p.add_argument("--ea",         type=float, nargs="+", default=None,
                   help="Ea values [kJ/mol] to run (default: all three: 26 55 70)")
    return p.parse_args()


def main():
    args = parse_args()

    ea_to_run = args.ea if args.ea is not None else EA_CASES

    base_plant = load_plant(PLANT_CFG)
    power_cfg  = load_config(POWER_CFG)
    price_cfg  = load_config(PRICE_CFG)

    # --- Run simulations for each Ea ---
    if args.run and not args.plots_only:
        for ea in ea_to_run:
            ob = _out_base(ea)
            ob.mkdir(parents=True, exist_ok=True)
            plant = _apply_ea(base_plant, ea)
            temps = [args.only_T] if args.only_T is not None else FIXED_T_DEGC
            print(f"\nEa = {ea:g} kJ/mol  →  {ob.name}")
            # For non-base Ea values, also run a DA joint (j+T) simulation so the
            # Pareto comparison is fair (all runs use the same Ea).
            if ea != 55:
                run_joint_da(plant, power_cfg, price_cfg, ob)
            print(f"Running {len(temps)} fixed-T simulations...")
            for T in temps:
                run_fixed_T(T, plant, power_cfg, price_cfg, ob)

    # --- Collect KPIs for all Ea values ---
    all_ea: dict[float, tuple[list, dict]] = {}

    for ea in EA_CASES:
        ob = _out_base(ea)
        fixed_T_results = []
        for T in FIXED_T_DEGC:
            name     = f"fixedT_{T:g}C"
            csv_path = ob / name / f"{name}.csv"
            if csv_path.exists():
                k = kpis(csv_path)
                k["T_set_C"] = T
                k["label"]   = f"{T:g}$^\\circ$C"
                fixed_T_results.append(k)
            elif ea in ea_to_run:
                print(f"  [missing] {csv_path} — run with --run first")

        # For Ea=55: load all four controllers (full comparison figure).
        # For Ea≠55: only load the Ea-specific DA joint run — LF/PA/LA are
        # calibrated to Ea=55 and do not belong on a different-Ea Pareto plot.
        existing_results = {}
        if ea == 55:
            for label, csv_path in EXISTING.items():
                if csv_path.exists():
                    existing_results[label] = kpis(csv_path)
                elif ea == EA_CASES[0]:
                    print(f"  [missing] {csv_path}")
        else:
            ea_specific = _out_base(ea) / "joint_da" / "joint_da.csv"
            if ea_specific.exists():
                existing_results["Degradation-aware"] = kpis(ea_specific)
            else:
                print(f"  [missing] joint_da for Ea={ea:g} — run with --run first")

        if fixed_T_results or existing_results:
            all_ea[ea] = (fixed_T_results, existing_results)

    if not all_ea:
        print("No results found. Run with --run first.")
        return

    # --- Per-Ea stdout tables and figures ---
    for ea, (fixed_T_results, existing_results) in all_ea.items():
        print(f"\n=== Ea = {ea:g} kJ/mol ===")
        all_rows = []
        for r in fixed_T_results:
            all_rows.append(("Fixed T", r["label"], r["T_mean_C"], r["j_mean"],
                              r["profit"], r["vdeg_mV"], r["lcoh"], r["lifetime_yr"]))
        for label, k in existing_results.items():
            all_rows.append(("Controller", label, k["T_mean_C"], k["j_mean"],
                              k["profit"], k["vdeg_mV"], k["lcoh"], k["lifetime_yr"]))

        print(f"{'Type':<12}  {'Name':<20}  {'T_mean':>7}  {'j_mean':>7}  "
              f"{'Profit':>10}  {'V_deg':>8}  {'LCOH':>7}  {'Life':>6}")
        print("-" * 92)
        for row in sorted(all_rows, key=lambda x: x[5]):
            life = f"{row[7]:.1f}" if row[7] < 99 else ">99"
            print(f"{row[0]:<12}  {row[1]:<20}  {row[2]:>6.1f}C  {row[3]:>7.0f}  "
                  f"{row[4]:>+10.0f}  {row[5]:>7.2f} mV  {row[6]:>6.2f}  {life:>5} yr")

        if fixed_T_results:
            _make_figure(fixed_T_results, existing_results, ea_kJ=ea, x_metric="lcoh")

    # --- Cross-Ea comparison table ---
    if len(all_ea) >= 2:
        _make_ea_comparison_table(all_ea)


if __name__ == "__main__":
    main()
