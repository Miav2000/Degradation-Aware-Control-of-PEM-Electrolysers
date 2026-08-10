"""
sensitivity_tornado.py — OAT sensitivity analysis with parallel execution.

Varies four parameters one-at-a-time (low / base / high), runs DA for each case, and produces two tornado plots:
  sa_tornado_lcoh.pdf/.png      — LCOH [EUR/kg]
  sa_tornado_lifetime.pdf/.png  — Projected stack lifetime [yr]

Usage:
    python3 run/sensitivity/sensitivity_tornado.py
    python3 run/sensitivity/sensitivity_tornado.py --workers 9
    nohup python3 run/sensitivity/sensitivity_tornado.py --workers 9 \
    > results/sensitivity_tornado/run.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "run"))

# ── Paths ─────────────────────────────────────────────────────────────────────
PLANT_CFG      = REPO_ROOT / "configs" / "plant_parameters.yaml"
CTRL_AWARE_CFG = REPO_ROOT / "configs" / "controllers" / "degradation_aware.yaml"
CTRL_COMM_CFG  = REPO_ROOT / "configs" / "controllers" / "load_following.yaml"
POWER_CFG      = REPO_ROOT / "configs" / "power_profiles" / "wind.yaml"
PRICE_CFG      = REPO_ROOT / "configs" / "price_profiles" / "spot_dk1.yaml"
OUTDIR         = REPO_ROOT / "results" / "sensitivity_tornado"
OUTDIR.mkdir(parents=True, exist_ok=True)

LEVELS = ["low", "base", "high"]

PARAMS_REDUCED_KEYS = {"capex", "ea", "p_h2", "deg_rate"}

PARAMS: list[dict] = [
    {
        "key":    "capex",
        "label":  r"$c_{stack}$ [\$/kW]",
        "values": [337.5, 450.0, 562.5],       # ±25 %
    },
    {
        "key":    "ea",
        "label":  r"$E_{a,\mathrm{eff}}$ [kJ/mol]",
        "values": [41.25, 55.0, 68.75],         # ±25 %
    },
    {
        "key":    "p_h2",
        "label":  r"$p_{\mathrm{H_2}}$ [€/kg]",
        "values": [4.5, 6.0, 7.5],              # ±25 %, base = 6.0
    },
    {
        "key":    "deg_rate",
        "label":  r"$\dot{V}_{\mathrm{deg,ref}}$ [$\mu$V/h]",
        "values": [7.2, 9.6, 12.0],             # ±25 %
    },
    {
        "key":    "c_shutdown",
        "label":  r"$c_{\mathrm{shutdown}}$ [€/event]",
        "values": [37.5, 50.0, 62.5],           # ±25 %
    },
    {
        "key":    "system_capex",
        "label":  r"$c_{\mathrm{sys}}$ [€/kW]",
        "values": [525.0, 700.0, 875.0],        # ±25 %
    },
    {
        "key":    "spot_price",
        "label":  r"$s_{\mathrm{elec}}$ [-]",
        "values": [0.75, 1.0, 1.25],            # ±25 % on price series
    },
    {
        "key":    "wind",
        "label":  r"$s_{\mathrm{wind}}$ [-]",
        "values": [0.75, 1.0, 1.25],            # ±25 % on available power series
    },
    {
        "key":    "EOL",
        "label":  r"$V_{\mathrm{EOL}}$ [mV]",
        "values": [75, 100, 125],            # ±25 %
    },
    {
        "key":    "T_set_K",
        "label":  r"$T_{\mathrm{set}}$ [$^\circ$C]",
        "values": [45, 60, 75],              # ±25 % of 60 °C baseline
    },
]


# ── Timestamped logging (safe for nohup) ─────────────────────────────────────
def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Parameter application (plain function — picklable for multiprocessing) ───
def _scale_deg_coeffs(plant: dict, target_uV_per_h: float) -> None:
    from pemwe.plant import derive_degradation_coeffs
    plant["degradation"]["DDR_ref_uV_per_h"] = target_uV_per_h
    derive_degradation_coeffs(plant)


def _apply_param(plant: dict, key: str, val: float) -> None:
    if key == "capex":
        plant["economics"]["capex_usd_per_kW"] = val
    elif key == "ea":
        plant["degradation"]["Ea_eff_J_per_mol"] = val * 1e3
    elif key == "p_h2":
        plant["economics"]["p_H2_eur_per_kg"] = val
    elif key == "deg_rate":
        _scale_deg_coeffs(plant, val)
    elif key == "c_shutdown":
        plant["economics"]["c_shutdown_eur"] = val
    elif key == "system_capex":
        pass  # post-hoc only — no effect on simulation physics
    elif key in ("spot_price", "wind"):
        pass  # applied as series scale in _run_case
    elif key == "EOL":
        plant["degradation"]["V_deg_EOL_V"] = val / 1000.0   # mV → V
    elif key == "T_set_K":
        plant["water_feed"]["T_set_K"] = val + 273.15         # °C → K
    else:
        raise ValueError(f"Unknown sensitivity parameter key: {key!r}")


# ── Worker function (runs in a separate process) ──────────────────────────────
def _run_case(case: dict) -> dict:
    """
    Run one sensitivity case and return a metrics dict.
    Loads all configs fresh (process-safe) and caches result to CSV.
    """
    # These imports happen inside the worker process
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(case["repo_root"])))
    _sys.path.insert(0, str(_Path(case["repo_root"]) / "run"))

    import pandas as _pd
    import numpy as _np
    from pemwe.plant import load_plant, load_config
    from pemwe.simulation import run_simulation
    from single_simulation import load_power_profile, load_price_profile

    tag       = case["tag"]
    cache_csv = _Path(case["cache_csv"])

    # If capex is the varied parameter, stack replacement cost must use that varied value.
    stack_repl_eur = (float(case["param_val"]) * case["eur_per_usd"] * case["P_rated_kW"]
                      if case["param_key"] == "capex" else case["stack_repl_eur"])

    # c_shutdown and system_capex only affect accounting — recompute LCOH analytically.
    if case["param_key"] in ("c_shutdown", "system_capex"):
        import time as _time
        base_csv = _Path(case["baseline_csv"])
        # Wait for the capex base simulation to finish (runs in parallel)
        for _ in range(720):   # up to 2 hours
            if base_csv.exists():
                break
            _time.sleep(10)
        else:
            raise TimeoutError(f"Baseline CSV never appeared: {base_csv}")
        df = _pd.read_csv(base_csv)
        dt_h        = case["dt_h"]
        n_steps     = case["n_steps"]
        total_H2_kg = (df["m_dot_H2_kg_h"] * dt_h).sum()
        sim_yr      = n_steps * dt_h / 8760.0
        h2_annual   = total_H2_kg / max(sim_yr, 1e-9)
        n_shutdowns = int((df["c_shutdown_eur"] > 0).sum())

        V_EOL       = case["V_EOL"]
        deg_rate    = df["dV_deg_V"].sum() / (n_steps * dt_h)
        lifetime_yr = (V_EOL / deg_rate / 8760.0) if deg_rate > 0 else _np.inf

        if case["param_key"] == "c_shutdown":
            c_sd       = float(case["param_val"])
            op_cost    = (df["c_elec_eur_h"] * dt_h).sum() + c_sd * n_shutdowns
            sys_capex_eur = case["sys_capex_eur"]
        else:  # system_capex
            op_cost    = (df["c_elec_eur_h"] * dt_h + df["c_shutdown_eur"]).sum()
            sys_capex_eur = float(case["param_val"]) * case["P_rated_kW"]

        repl_times  = _np.arange(lifetime_yr, case["sys_lifetime_yr"] + lifetime_yr, lifetime_yr)
        repl_times  = repl_times[repl_times <= case["sys_lifetime_yr"]] if _np.isfinite(lifetime_yr) else _np.array([])
        npv_rep     = sum(stack_repl_eur / (1 + case["wacc"]) ** t for t in repl_times)
        lcoh_rep    = npv_rep / (h2_annual * case["annuity"]) if h2_annual > 0 else 0.0
        lcoh_sys    = sys_capex_eur / (h2_annual * case["annuity"]) if h2_annual > 0 else 0.0
        lcoh        = op_cost / total_H2_kg + lcoh_rep + lcoh_sys if total_H2_kg > 0 else _np.nan
        return {
            "tag": tag, "status": "analytical",
            "param_key": case["param_key"], "level": case["level"],
            "param_val": case["param_val"], "ctrl_name": case["ctrl_name"],
            "lcoh": lcoh, "lifetime_yr": lifetime_yr,
        }

    if cache_csv.exists():
        df = _pd.read_csv(cache_csv)
        status = "cache"
    else:
        plant_cfg = _Path(case["plant_cfg"])
        ctrl_cfg  = load_config(_Path(case["ctrl_cfg"]))
        plant     = load_plant(plant_cfg)
        _apply_param(plant, case["param_key"], case["param_val"])

        power_cfg = load_config(_Path(case["power_cfg"]))
        price_cfg = load_config(_Path(case["price_cfg"]))
        sim_cfg   = power_cfg["simulation"]
        dt_s      = float(sim_cfg["dt_s"])
        n_steps   = int(round(float(sim_cfg["t_end_s"]) / dt_s))
        repo_root = _Path(case["repo_root"])

        p_avail = load_power_profile(power_cfg, n_steps, dt_s, repo_root)
        prices  = load_price_profile(price_cfg, n_steps, repo_root)

        if case["param_key"] == "spot_price":
            prices  = _np.array(prices)  * float(case["param_val"])
        elif case["param_key"] == "wind":
            p_avail = _np.array(p_avail) * float(case["param_val"])

        df = run_simulation(
            price_series=prices,
            p_avail_series=p_avail,
            plant=plant,
            ctrl_cfg=ctrl_cfg,
            dt_s=dt_s,
        )
        df.to_csv(cache_csv, index=False)
        status = "done"

    # Metrics from simulation output columns
    dt_h        = case["dt_h"]
    n_steps     = case["n_steps"]
    total_H2_kg = (df["m_dot_H2_kg_h"] * dt_h).sum()
    op_cost     = (df["c_elec_eur_h"] * dt_h + df["c_shutdown_eur"]).sum()
    sim_yr      = n_steps * dt_h / 8760.0
    h2_annual   = total_H2_kg / max(sim_yr, 1e-9)
    V_EOL       = case["V_EOL"]
    deg_rate    = df["dV_deg_V"].sum() / (n_steps * dt_h)
    lifetime_yr = (V_EOL / deg_rate / 8760.0) if deg_rate > 0 else _np.inf
    repl_times  = _np.arange(lifetime_yr, case["sys_lifetime_yr"] + lifetime_yr, lifetime_yr)
    repl_times  = repl_times[repl_times <= case["sys_lifetime_yr"]] if _np.isfinite(lifetime_yr) else _np.array([])
    npv_rep     = sum(stack_repl_eur / (1 + case["wacc"]) ** t for t in repl_times)
    lcoh_rep    = npv_rep / (h2_annual * case["annuity"]) if h2_annual > 0 else 0.0
    lcoh_sys    = case["sys_capex_eur"] / (h2_annual * case["annuity"]) if h2_annual > 0 else 0.0
    lcoh        = op_cost / total_H2_kg + lcoh_rep + lcoh_sys if total_H2_kg > 0 else _np.nan

    return {
        "tag":         tag,
        "status":      status,
        "param_key":   case["param_key"],
        "level":       case["level"],
        "param_val":   case["param_val"],
        "ctrl_name":   case["ctrl_name"],
        "lcoh":        lcoh,
        "lifetime_yr": lifetime_yr,
    }


# ── Build case list ───────────────────────────────────────────────────────────
def _build_cases(base_plant: dict, dt_h: float, n_steps: int) -> list[dict]:
    P_rated_kW      = float(base_plant["stack"]["P_rating_W"]) / 1000.0
    sys_capex_eur   = float(base_plant["economics"]["system_capex_eur_per_kW"]) * P_rated_kW
    sys_lifetime_yr = float(base_plant["economics"]["system_lifetime_yr"])
    wacc            = float(base_plant["economics"].get("wacc", 0.0))
    annuity         = (1 - (1 + wacc) ** -sys_lifetime_yr) / wacc if wacc > 0 else sys_lifetime_yr
    stack_repl_eur  = float(base_plant["economics"]["capex_usd_per_kW"]) * float(base_plant["economics"]["eur_per_usd"]) * P_rated_kW

    cases = []
    for param in PARAMS:
        for level, val in zip(LEVELS, param["values"]):
            for ctrl_name, ctrl_path in [("degradation_aware",      str(CTRL_AWARE_CFG)),
                                          ("load_following", str(CTRL_COMM_CFG))]:
                tag = f"sa_{param['key']}_{level}_{ctrl_name}"
                entry = {
                    "tag":            tag,
                    "param_key":      param["key"],
                    "level":          level,
                    "param_val":      val,
                    "ctrl_name":      ctrl_name,
                    "ctrl_cfg":       ctrl_path,
                    "plant_cfg":      str(PLANT_CFG),
                    "power_cfg":      str(POWER_CFG),
                    "price_cfg":      str(PRICE_CFG),
                    "repo_root":      str(REPO_ROOT),
                    "cache_csv":      str(OUTDIR / f"{tag}.csv"),
                    "dt_h":           dt_h,
                    "n_steps":        n_steps,
                    "V_EOL":          float(base_plant["degradation"]["V_deg_EOL_V"]),
                    "sys_capex_eur":   sys_capex_eur,
                    "P_rated_kW":      P_rated_kW,
                    "sys_lifetime_yr": sys_lifetime_yr,
                    "annuity":         annuity,
                    "stack_repl_eur":  stack_repl_eur,
                    "wacc":            wacc,
                    "eur_per_usd":     float(base_plant["economics"]["eur_per_usd"]),
                }
                if param["key"] in ("c_shutdown", "system_capex"):
                    # Point to the capex-base CSV (same simulation, only accounting differs)
                    entry["baseline_csv"] = str(OUTDIR / f"sa_capex_base_{ctrl_name}.csv")
                if param["key"] == "EOL":
                    entry["V_EOL"] = val / 1000.0   # mV → V; also applied to plant in _apply_param
                cases.append(entry)

    return cases


# ── Tornado plot ──────────────────────────────────────────────────────────────
# Colour palette inspired by the viridis-based scatter plots in the project.
_COL_AWARE = "#21918C"   # viridis teal
_COL_COMM  = "#F77F00"   # warm amber-orange
_COL_REF   = "#6C757D"   # neutral grey for reference line
_COL_STRIPE = "#F4F6F8"  # very light stripe for alternating rows


def _draw_tornado_ax(ax, results: dict, metric: str, xlabel: str,
                     order: list, show_ylabels: bool = True,
                     show_legend: bool = True, params: list = None,
                     tick_step: float = 0.25) -> None:
    """Draw one tornado panel onto *ax*. order: sorted indices into params (default: PARAMS)."""
    from plot_style import DARK
    if params is None:
        params = PARAMS

    _FS   = 14
    n     = len(params)
    bar_h = 0.42

    all_vals = []
    for p in params:
        for lv in LEVELS:
            v = results[p["key"]][lv]["degradation_aware"][metric]
            if np.isfinite(v):
                all_vals.append(v)
    x_min, x_max = min(all_vals), max(all_vals)
    x_span       = x_max - x_min
    CLUMP_THRESH = 0.06 * x_span

    for rank, idx in enumerate(order):
        param = params[idx]
        key   = param["key"]
        vals  = param["values"]
        y     = n - 1 - rank

        if rank % 2 == 1:
            ax.axhspan(y - 0.5, y + 0.5, color=_COL_STRIPE, zorder=0)

        lo_a   = results[key]["low"]["degradation_aware"][metric]
        hi_a   = results[key]["high"]["degradation_aware"][metric]
        base_a = results[key]["base"]["degradation_aware"][metric]

        bar_left  = min(lo_a, hi_a)
        bar_right = max(lo_a, hi_a)
        bar_width = bar_right - bar_left

        ax.barh(y, bar_width, left=bar_left, height=bar_h,
                color=_COL_AWARE, alpha=0.85, zorder=3,
                linewidth=0.6, edgecolor="white")
        ax.plot(base_a, y, marker="|", color="white", ms=13, mew=2.5, zorder=6)

        if bar_width >= CLUMP_THRESH:
            ax.text(lo_a, y + bar_h * 0.58, f"{vals[0]:g}",
                    va="bottom", ha="center", fontsize=11, color=DARK, zorder=7)
            ax.text(hi_a, y + bar_h * 0.58, f"{vals[2]:g}",
                    va="bottom", ha="center", fontsize=11, color=DARK, zorder=7)
        else:
            # Narrow bar — spread outward from bar edges
            # bar_left/right use min/max, so labels must follow lo_a vs hi_a direction
            left_label  = vals[0] if lo_a <= hi_a else vals[2]
            right_label = vals[2] if lo_a <= hi_a else vals[0]
            ax.text(bar_left,  y + bar_h * 0.58, f"{left_label:g}",
                    va="bottom", ha="right", fontsize=11, color=DARK, zorder=7)
            ax.text(bar_right, y + bar_h * 0.58, f"{right_label:g}",
                    va="bottom", ha="left", fontsize=11, color=DARK, zorder=7)

    ax.set_yticks(range(n))
    if show_ylabels:
        ax.set_yticklabels([params[order[n - 1 - i]]["label"] for i in range(n)],
                           fontsize=_FS)
    else:
        ax.tick_params(labelleft=False)

    ax.axvline(results[params[0]["key"]]["base"]["degradation_aware"][metric],
               color=_COL_REF, lw=1.2, ls="--", zorder=2, alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=_FS)
    ax.tick_params(axis="x", labelsize=_FS)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(x_min - 0.06 * x_span, x_max + 0.06 * x_span)
    ax.grid(axis="x", alpha=0.20, zorder=1, color=_COL_REF)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))


def make_tornado(results: dict, metric: str, xlabel: str, filename: str,
                 tick_step: float = 0.25) -> None:
    from plot_style import apply_style
    apply_style()
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage{eurosym}\usepackage{amssymb}"
    )
    n = len(PARAMS)
    widths = [abs(results[p["key"]]["high"]["degradation_aware"][metric]
                  - results[p["key"]]["low"]["degradation_aware"][metric])
              for p in PARAMS]
    order = list(np.argsort(widths)[::-1])
    fig, ax = plt.subplots(figsize=(6.5, 0.65 * n + 1.2))
    _draw_tornado_ax(ax, results, metric, xlabel, order,
                     show_ylabels=True, show_legend=True, tick_step=tick_step)
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        out = OUTDIR / (filename + ext)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        _log(f"Saved: {out.name}")
    plt.close(fig)


def make_tornado_reduced(results: dict, metric: str, xlabel: str, filename: str,
                         tick_step: float = 0.25) -> None:
    """Compact tornado with only the 4 most publication-relevant parameters."""
    from plot_style import apply_style
    apply_style()
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage{eurosym}\usepackage{amssymb}"
    )
    params = [p for p in PARAMS if p["key"] in PARAMS_REDUCED_KEYS]
    n = len(params)
    widths = [abs(results[p["key"]]["high"]["degradation_aware"][metric]
                  - results[p["key"]]["low"]["degradation_aware"][metric])
              for p in params]
    order = list(np.argsort(widths)[::-1])
    fig, ax = plt.subplots(figsize=(6.5, 0.65 * n + 1.2))
    _draw_tornado_ax(ax, results, metric, xlabel, order,
                     show_ylabels=True, show_legend=True, params=params,
                     tick_step=tick_step)
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        out = OUTDIR / (filename + ext)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        _log(f"Saved: {out.name}")
    plt.close(fig)



# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: all cores)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Re-draw figures from cached CSVs without re-running simulations")
    args = parser.parse_args()

    n_workers = args.workers or os.cpu_count()

    from pemwe.plant import load_plant, load_config
    from single_simulation import load_power_profile, load_price_profile

    base_plant = load_plant(PLANT_CFG)
    power_cfg  = load_config(POWER_CFG)
    sim_cfg    = power_cfg["simulation"]
    dt_s       = float(sim_cfg["dt_s"])
    n_steps    = int(round(float(sim_cfg["t_end_s"]) / dt_s))
    dt_h       = dt_s / 3600.0

    cases = _build_cases(base_plant, dt_h, n_steps)
    n_total  = len(cases)
    n_cached = sum(1 for c in cases if Path(c["cache_csv"]).exists())
    n_run    = n_total - n_cached

    _log(f"Sensitivity tornado: {n_total} cases "
         f"({n_cached} cached, {n_run} to run) — {n_workers} workers")
    _log(f"Output dir: {OUTDIR}")

    t0 = time.time()

    raw_results: list[dict] = []

    if args.plots_only:
        for i, c in enumerate(cases, 1):
            try:
                res = _run_case(c)
                raw_results.append(res)
                _log(f"[{i:2d}/{n_total}] {res['tag']:40s} "
                     f"({res['status']})  "
                     f"LCOH={res['lcoh']:.3f} €/kg  "
                     f"life={res['lifetime_yr']:.2f} yr")
            except Exception as exc:
                _log(f"[{i:2d}/{n_total}] {c['tag']} FAILED: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_case, c): c["tag"] for c in cases}
            for i, fut in enumerate(as_completed(futures), 1):
                tag = futures[fut]
                try:
                    res = fut.result()
                    raw_results.append(res)
                    _log(f"[{i:2d}/{n_total}] {res['tag']:40s} "
                         f"({res['status']})  "
                         f"LCOH={res['lcoh']:.3f} €/kg  "
                         f"life={res['lifetime_yr']:.2f} yr")
                except Exception as exc:
                    _log(f"[{i:2d}/{n_total}] {tag} FAILED: {exc}")

    n_failed = n_total - len(raw_results)
    elapsed = time.time() - t0
    _log(f"All cases done in {elapsed/60:.1f} min  ({n_failed} failed)")

    # Restructure into results[param_key][level][ctrl_name]
    results: dict = {p["key"]: {lv: {} for lv in LEVELS} for p in PARAMS}
    for r in raw_results:
        results[r["param_key"]][r["level"]][r["ctrl_name"]] = {
            "lcoh":        r["lcoh"],
            "lifetime_yr": r["lifetime_yr"],
        }

    # Tornado plots
    make_tornado(results, "lcoh",
                 r"LCOH [€ kg$_{\mathrm{H_2}}^{-1}$]",
                 "sa_tornado_lcoh")
    make_tornado(results, "lifetime_yr",
                 "Projected stack lifetime [yr]",
                 "sa_tornado_lifetime", tick_step=2.5)
    make_tornado_reduced(results, "lcoh",
                         r"LCOH [€ kg$_{\mathrm{H_2}}^{-1}$]",
                         "sa_tornado_lcoh_reduced")
    make_tornado_reduced(results, "lifetime_yr",
                         "Projected stack lifetime [yr]",
                         "sa_tornado_lifetime_reduced", tick_step=2.5)

    # Summary CSV
    rows = []
    for param in PARAMS:
        for level, val in zip(LEVELS, param["values"]):
            for ctrl in ["degradation_aware", "load_following"]:
                m = results[param["key"]][level][ctrl]
                rows.append({"parameter": param["key"], "level": level,
                             "param_value": val, "controller": ctrl,
                             "lcoh_eur_kg": m["lcoh"],
                             "lifetime_yr": m["lifetime_yr"]})
    pd.DataFrame(rows).to_csv(OUTDIR / "sa_summary.csv", index=False)

    # Console table
    print("\n" + "=" * 68, flush=True)
    for metric, mname, fmt in [("lcoh",        "LCOH [€/kg]",   ".3f"),
                                 ("lifetime_yr", "Lifetime [yr]", ".2f")]:
        print(f"\n  {mname}", flush=True)
        print(f"  {'Parameter':<10} {'Level':<6} {'Val':>6}  "
              f"{'Degradation-aware':>18}  {'Load-following':>15}", flush=True)
        print(f"  {'-'*46}", flush=True)
        for param in PARAMS:
            key = param["key"]
            for level, val in zip(LEVELS, param["values"]):
                aw = results[key][level]["degradation_aware"][metric]
                cm = results[key][level]["load_following"][metric]
                mark = " ←" if level == "base" else ""
                print(f"  {key:<10} {level:<6} {val:>6.1f}  "
                      f"{aw:>8{fmt}}  {cm:>11{fmt}}{mark}", flush=True)
            print(flush=True)

    _log(f"Done. All outputs → {OUTDIR}")


if __name__ == "__main__":
    main()
