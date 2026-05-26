"""
sensitivity_substeps.py — Sub-step resolution sensitivity for thermal oscillations.

Runs the commercial controller with three sub-step configurations over the full
year, captures sub-step detail in a chosen high-load steady-state window, and
produces a publication-quality comparison figure.

Configurations
--------------
  base:  n_sim=40,  n_preview=20   (current default — 90 s sub-steps)
  test1: n_sim=60,  n_preview=30   (60 s sub-steps)
  test2: n_sim=120, n_preview=60   (30 s sub-steps)

Output
------
  results/sensitivity_substeps/
    base_substeps.csv      (sub-step detail, ~window_h hours × 40  rows)
    test1_substeps.csv     (                              × 60  rows)
    test2_substeps.csv     (                              × 120 rows)
    fig_substep_comparison.png / .pdf

Usage
-----
  python3 run/sensitivity/sensitivity_substeps.py            # run + plot
  python3 run/sensitivity/sensitivity_substeps.py --plots-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pemwe.plant import load_plant, load_config
from pemwe.simulation import run_simulation
from run.single_simulation import load_power_profile, load_price_profile

sys.path.insert(0, str(REPO_ROOT / "run"))
from plot_style import apply_style, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, DARK

OUTDIR = REPO_ROOT / "results" / "sensitivity_substeps"

# System CAPEX constants (single source of truth from plant config)
_plant_cfg = load_plant(REPO_ROOT / "configs" / "plant_parameters.yaml")
_P_RATED_KW      = float(_plant_cfg["stack"]["P_rating_W"]) / 1000.0
_SYS_CAPEX_EUR   = float(_plant_cfg["economics"]["system_capex_eur_per_kW"]) * _P_RATED_KW
_SYS_LIFETIME_YR = float(_plant_cfg["economics"]["system_lifetime_yr"])
_WACC            = float(_plant_cfg["economics"].get("wacc", 0.0))
_ANNUITY         = (1 - (1 + _WACC) ** -_SYS_LIFETIME_YR) / _WACC if _WACC > 0 else _SYS_LIFETIME_YR
_STACK_REPL_EUR  = float(_plant_cfg["economics"]["capex_usd_per_kW"]) * float(_plant_cfg["economics"]["eur_per_usd"]) * _P_RATED_KW

# Detail window — sustained high-load steady-state region
DETAIL_START_H = 7823.0
DETAIL_END_H   = 7833.0

CONFIGS = [
    dict(label="20 sub-steps (180 s)", n_sim=20,  n_prev=10,  color=RED,       ls="-", lw=1.5, alpha=1.0),
    dict(label="40 sub-steps (90 s)",  n_sim=40,  n_prev=20,  color=BLUE,      ls="-", lw=1.5, alpha=1.0),
    dict(label="60 sub-steps (60 s)",  n_sim=60,  n_prev=30,  color=GREY,      ls="-", lw=1.5, alpha=1.0),
    dict(label="80 sub-steps (45 s)",  n_sim=80,  n_prev=40,  color=ORANGE,    ls="-", lw=1.5, alpha=1.0),
    dict(label="120 sub-steps (30 s)", n_sim=120, n_prev=60,  color="#A0522D", ls="-", lw=1.5, alpha=1.0),
    dict(label="160 sub-steps (22 s)", n_sim=160, n_prev=80,  color=GREEN,     ls="-", lw=1.5, alpha=1.0),
    dict(label="320 sub-steps (11 s)", n_sim=320, n_prev=160, color=PURPLE,    ls="-", lw=1.5, alpha=1.0),
]


def run_one(cfg: dict, plant_path: Path, ctrl_path: Path,
            power_cfg_path: Path, price_cfg_path: Path) -> tuple:
    """Run full-year simulation for one substep config.
    Returns (hourly_df, substep_df, t_wall_s).
    """
    import time
    plant     = load_plant(plant_path)
    power_cfg = load_config(power_cfg_path)
    price_cfg = load_config(price_cfg_path)
    ctrl_cfg  = load_config(ctrl_path)

    plant["fast_control"]["n_substeps"]         = cfg["n_sim"]
    plant["fast_control"]["n_preview_substeps"] = cfg["n_sim"] // 2

    sim_cfg = power_cfg["simulation"]
    dt_s    = float(sim_cfg["dt_s"])
    n_steps = int(round(float(sim_cfg["t_end_s"]) / dt_s))

    p_avail = load_power_profile(power_cfg, n_steps, dt_s, REPO_ROOT)
    price   = load_price_profile(price_cfg, n_steps, REPO_ROOT)

    print(f"  Running n_sim={cfg['n_sim']} …", flush=True)
    t0 = time.perf_counter()
    df = run_simulation(
        price_series=price,
        p_avail_series=p_avail,
        plant=plant,
        ctrl_cfg=ctrl_cfg,
        dt_s=dt_s,
        detail_window_h=(DETAIL_START_H, DETAIL_END_H),
    )
    t_wall_s = time.perf_counter() - t0

    sub_df = df.attrs.get("substep_df", pd.DataFrame())
    return df, sub_df, t_wall_s


def _worker(args):
    """Top-level worker for parallel execution (must be picklable)."""
    cfg, plant_path, ctrl_path, power_cfg_path, price_cfg_path, hourly_csv, sub_csv = args
    df, sub_df, t_wall_s = run_one(cfg, plant_path, ctrl_path, power_cfg_path, price_cfg_path)
    df["t_wall_s"] = t_wall_s
    hourly_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(hourly_csv, index=False)
    if not sub_df.empty:
        sub_df.to_csv(sub_csv, index=False)
    print(f"  Saved {hourly_csv.name}  ({len(df)} rows, {t_wall_s:.1f} s)", flush=True)
    return cfg["n_sim"], t_wall_s


def kpis(df: pd.DataFrame) -> dict:
    """Extract annual KPIs from hourly simulation DataFrame."""
    dt_h   = 1.0
    active = df["j_A_per_m2"] > 100
    act    = df.loc[active]

    total_H2    = (df["m_dot_H2_kg_h"] * dt_h).sum()
    total_cost  = (df["c_elec_eur_h"] + df["c_shutdown_eur"]).sum()
    sim_yr      = df["t_h"].max() / 8760.0
    h2_annual   = total_H2 / max(sim_yr, 1e-9)

    V_deg_EOL    = 0.10
    V_deg_cum    = df["dV_deg_V"].sum()
    tau_stack_yr = V_deg_EOL / max(V_deg_cum / max(sim_yr, 1e-9), 1e-12)

    repl_times  = np.arange(tau_stack_yr, _SYS_LIFETIME_YR + tau_stack_yr, tau_stack_yr)
    repl_times  = repl_times[repl_times <= _SYS_LIFETIME_YR] if np.isfinite(tau_stack_yr) else np.array([])
    npv_rep     = sum(_STACK_REPL_EUR / (1 + _WACC) ** t for t in repl_times)
    lcoh_rep    = npv_rep / (h2_annual * _ANNUITY) if h2_annual > 0 else 0.0
    lcoh_sys    = _SYS_CAPEX_EUR / (h2_annual * _ANNUITY) if h2_annual > 0 else 0.0
    lcoh        = total_cost / max(total_H2, 1e-12) + lcoh_rep + lcoh_sys

    energy_kWh = (act["P_total_W"] * dt_h / 1000.0).sum()
    H2_active  = (act["m_dot_H2_kg_h"] * dt_h).sum()
    sec_kWh_kg = energy_kWh / max(H2_active, 1e-12)

    t_wall = df["t_wall_s"].iloc[0] if "t_wall_s" in df.columns else float("nan")
    return dict(H2_kg=total_H2, lcoh_eur_kg=lcoh,
                tau_stack_yr=tau_stack_yr, sec_kWh_kg=sec_kWh_kg,
                t_wall_s=t_wall)


def load_or_run(cfg: dict, hourly_csv: Path, sub_csv: Path, args) -> tuple:
    if args.plots_only or (args.skip_existing and hourly_csv.exists()):
        if not hourly_csv.exists():
            raise FileNotFoundError(f"--plots-only but missing: {hourly_csv}")
        print(f"  Loading {hourly_csv.name}")
        df     = pd.read_csv(hourly_csv)
        sub_df = pd.read_csv(sub_csv) if sub_csv.exists() else pd.DataFrame()
        if "t_wall_s" in df.columns:
            cfg["t_wall_s"] = df["t_wall_s"].iloc[0]
        return df, sub_df
    df, sub_df, t_wall_s = run_one(
        cfg,
        REPO_ROOT / "configs" / "plant_parameters.yaml",
        REPO_ROOT / "configs" / "controllers" / "load_following.yaml",
        REPO_ROOT / "configs" / "power_profiles" / "wind.yaml",
        REPO_ROOT / "configs" / "price_profiles" / "spot_dk1.yaml",
    )
    cfg["t_wall_s"] = t_wall_s
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df["t_wall_s"] = t_wall_s
    df.to_csv(hourly_csv, index=False)
    if not sub_df.empty:
        sub_df.to_csv(sub_csv, index=False)
    print(f"    Saved {hourly_csv.name}  ({len(df)} rows, {t_wall_s:.1f} s)")
    return df, sub_df


def make_figure(dfs: list[pd.DataFrame], cfgs: list[dict]) -> None:
    apply_style()

    fig, axes = plt.subplots(len(cfgs), 1, figsize=(9, 2.5 * len(cfgs)), sharex=True)
    fig.suptitle(
        rf"Thermal sub-step resolution sensitivity --- commercial controller"
        "\n"
        rf"High-load window: h{DETAIL_START_H:.0f}--{DETAIL_END_H:.0f}  "
        r"($T_{\rm target} = 60\,^\circ$C)",
        fontsize=11,
    )

    dt_h_sup = 1.0  # supervisory dt = 1 h
    T_target_C = 60.0

    for ax, df, cfg in zip(axes, dfs, cfgs):
        n_sim = cfg["n_sim"]
        dt_sub_h = dt_h_sup / n_sim

        # Reconstruct absolute time for each sub-step row
        # sub-step record has 't_h' assigned in simulation.py
        t = df["t_h"].values
        T = df["T_stack_K"].values - 273.15

        ax.axhline(T_target_C, color=GREY, lw=0.8, ls="--", label=r"$T_{\rm target} = 60\,^\circ$C")
        ax.plot(t, T, color=cfg["color"], lw=cfg["lw"], label=cfg["label"])
        ax.fill_between(t, T_target_C, T, where=(T > T_target_C),
                        color=cfg["color"], alpha=0.12)

        ax.set_ylabel(r"$T_{\rm stack}$ [$^\circ$C]", fontsize=10)

        # Error stats
        err = T - T_target_C
        pos = err[err > 0]
        ax.text(0.99, 0.97,
                rf"mean err = {err.mean():.1f}$^\circ$C   "
                rf"max = {err.max():.1f}$^\circ$C   "
                rf"p95 = {np.percentile(err, 95):.1f}$^\circ$C",
                transform=ax.transAxes, va="top", ha="right",
                fontsize=8.5, color=DARK)

        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylim(56, 74)

        # Minor x-grid every hour
        ax.set_xticks(np.arange(DETAIL_START_H, DETAIL_END_H + 1, 1))
        ax.set_xticklabels([f"h{int(h)}" for h in np.arange(DETAIL_START_H, DETAIL_END_H + 1, 1)],
                           fontsize=8)

    axes[-1].set_xlabel("Simulation time [h]", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    png = OUTDIR / "fig_substep_comparison.png"
    pdf = OUTDIR / "fig_substep_comparison.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"  Saved {png.name}")
    print(f"  Saved {pdf.name}")
    plt.close(fig)


def make_overlay_figure(dfs: list[pd.DataFrame], cfgs: list[dict]) -> None:
    """Single-panel overlay — all three traces on one axes."""
    apply_style()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_title(
        rf"Sub-step resolution: thermal oscillations --- commercial controller  "
        rf"(h{DETAIL_START_H:.0f}--{DETAIL_END_H:.0f},  "
        r"$T_{\rm target} = 60\,^\circ$C)",
        fontsize=10,
    )

    T_target_C = 60.0
    ax.axhline(T_target_C, color=GREY, lw=1.0, ls="--", zorder=1)

    for df, cfg in zip(dfs, cfgs):
        t = df["t_h"].values
        T = df["T_stack_K"].values - 273.15
        ax.plot(t, T, color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"],
                alpha=cfg["alpha"], label=cfg["label"])

    ax.set_xlabel("Simulation time [h]", fontsize=10)
    ax.set_ylabel(r"$T_{\rm stack}$ [$^\circ$C]", fontsize=10)
    ax.set_ylim(56, 74)
    ax.set_xticks(np.arange(DETAIL_START_H, DETAIL_END_H + 1, 1))
    ax.set_xticklabels([f"h{int(h)}" for h in np.arange(DETAIL_START_H, DETAIL_END_H + 1, 1)],
                       fontsize=8)
    ax.legend(fontsize=9)

    fig.tight_layout()
    png = OUTDIR / "fig_substep_overlay.png"
    pdf = OUTDIR / "fig_substep_overlay.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"  Saved {png.name}")
    print(f"  Saved {pdf.name}")
    plt.close(fig)


def make_kpi_figure(kpi_list: list[dict], cfgs: list[dict]) -> None:
    """Table figure showing KPI deviation relative to the finest sub-step resolution."""
    apply_style()

    METRICS = [
        ("H2_kg",        "H2 produced",        "kg",      "{:.1f}"),
        ("lcoh_eur_kg",  "LCOH",               "€/kg",    "{:.4f}"),
        ("tau_stack_yr", "Proj. lifetime",     "yr",      "{:.2f}"),
        ("sec_kWh_kg",   "SEC",                "kWh/kg",  "{:.4f}"),
        ("t_wall_s",     "Wall-clock time",    "s",       "{:.1f}"),
    ]

    # Finest resolution is the last entry — used as reference
    ref = kpi_list[-1]

    # Build columns: KPI | Unit | val_0 | Δ% | val_1 | Δ% | val_ref (ref)
    # Each value and deviation in separate columns so no newlines needed.
    col_labels = ["KPI", "Unit"]
    for i, c in enumerate(cfgs):
        n = c["n_sim"]
        col_labels.append(f"n={n}")
        if i < len(cfgs) - 1:
            col_labels.append(f"Δ% vs n={cfgs[-1]['n_sim']}")

    rows = []
    for key, name, unit, fmt in METRICS:
        ref_val = ref[key]
        row = [name, unit]
        for i, k in enumerate(kpi_list):
            v = k[key]
            row.append(fmt.format(v))
            if i < len(kpi_list) - 1:
                pct = (v - ref_val) / abs(ref_val) * 100 if ref_val != 0 else 0.0
                row.append(f"{pct:+.4f} %")
        rows.append(row)

    n_rows = len(METRICS)
    n_cols = len(col_labels)

    # Disable usetex for the table — multiline/unicode cells are incompatible with it
    with plt.rc_context({"text.usetex": False}):
        fig, ax = plt.subplots(figsize=(13, 1.4 + 0.55 * n_rows))
        ax.axis("off")
        fig.suptitle(
            "Full-year KPI sensitivity to thermal sub-step resolution\n"
            "(load-following controller, 1-year wind + DK1"
            u" — deviation relative to finest step)",
            fontsize=11, y=0.98,
        )

        tbl = ax.table(
            cellText=rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1.0, 1.9)

        # Header row
        for col in range(n_cols):
            cell = tbl[0, col]
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")

        # KPI label and unit columns
        for row in range(1, n_rows + 1):
            tbl[row, 0].set_text_props(ha="left", fontweight="bold")
            tbl[row, 0].set_facecolor("#ecf0f1")
            tbl[row, 1].set_facecolor("#ecf0f1")

        # Reference (finest) column — neutral highlight
        ref_col = n_cols - 1
        for row in range(1, n_rows + 1):
            tbl[row, ref_col].set_facecolor("#dfe6e9")

        # Colour-code Δ% columns by magnitude: near zero → light green, large → red
        # Saturates at 0.05 % deviation.
        delta_cols = [c for c in range(2, n_cols) if col_labels[c].startswith("Δ")]
        for row in range(1, n_rows + 1):
            for col in delta_cols:
                cell = tbl[row, col]
                text = cell.get_text().get_text()
                try:
                    pct = float(text.replace("%", "").strip())
                    intensity = min(abs(pct) / 0.05, 1.0)
                    r = 0.85 + 0.15 * intensity
                    g = 0.95 - 0.45 * intensity
                    b = 0.85 - 0.45 * intensity
                    cell.set_facecolor((r, g, b))
                except ValueError:
                    pass

        fig.tight_layout(rect=[0, 0, 1, 0.92])

        png = OUTDIR / "fig_substep_kpi.png"
        pdf = OUTDIR / "fig_substep_kpi.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)

    # Console summary
    print(f"\n{'Config':<25}  {'H2 [kg]':>10}  {'LCOH [€/kg]':>12}  "
          f"{'Lifetime [yr]':>14}  {'SEC [kWh/kg]':>13}  {'Wall [s]':>9}")
    print("-" * 95)
    for cfg, k in zip(cfgs, kpi_list):
        tw = cfg.get("t_wall_s", float("nan"))
        print(f"{cfg['label']:<25}  {k['H2_kg']:>10.1f}  {k['lcoh_eur_kg']:>12.4f}  "
              f"{k['tau_stack_yr']:>14.2f}  {k['sec_kWh_kg']:>13.4f}  {tw:>9.1f}")
    print(f"\n  Reference: {cfgs[-1]['label']} (finest step)")
    print(f"  Saved {png.name}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plots-only", action="store_true",
                   help="Skip simulations; load existing CSVs and replot")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip simulation if hourly CSV already exists")
    p.add_argument("--parallel", action="store_true",
                   help="Run missing simulations in parallel using all available cores")
    return p.parse_args()


def main():
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    names = [f"n{cfg['n_sim']:03d}" for cfg in CONFIGS]
    plant_path     = REPO_ROOT / "configs" / "plant_parameters.yaml"
    ctrl_path      = REPO_ROOT / "configs" / "controllers" / "load_following.yaml"
    power_cfg_path = REPO_ROOT / "configs" / "power_profiles" / "wind.yaml"
    price_cfg_path = REPO_ROOT / "configs" / "price_profiles" / "spot_dk1.yaml"

    # --- parallel branch: launch all missing runs simultaneously ---
    if args.parallel and not args.plots_only:
        worker_args = []
        for name, cfg in zip(names, CONFIGS):
            hourly_csv = OUTDIR / f"{name}_hourly.csv"
            sub_csv    = OUTDIR / f"{name}_substeps.csv"
            if args.skip_existing and hourly_csv.exists():
                print(f"  Skipping {name} (CSV exists)")
            else:
                worker_args.append((cfg, plant_path, ctrl_path,
                                    power_cfg_path, price_cfg_path,
                                    hourly_csv, sub_csv))

        if worker_args:
            print(f"Launching {len(worker_args)} simulation(s) in parallel …")
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(_worker, wa): wa[0]["n_sim"]
                           for wa in worker_args}
                for fut in as_completed(futures):
                    n_sim, t_wall = fut.result()
                    print(f"  n={n_sim} finished in {t_wall:.1f} s", flush=True)

    # --- sequential load / run ---
    dfs, sub_dfs, kpi_list = [], [], []
    for name, cfg in zip(names, CONFIGS):
        hourly_csv = OUTDIR / f"{name}_hourly.csv"
        sub_csv    = OUTDIR / f"{name}_substeps.csv"
        if not args.parallel:
            print(f"--- {cfg['label']} ---")
        df, sub_df = load_or_run(cfg, hourly_csv, sub_csv,
                                 type("A", (), {"plots_only": args.plots_only or args.parallel,
                                                "skip_existing": True})())
        dfs.append(df)
        sub_dfs.append(sub_df)
        kpi_list.append(kpis(df))

    print("\nGenerating figures …")
    make_kpi_figure(kpi_list, CONFIGS)
    non_empty = [s for s in sub_dfs if not s.empty]
    if len(non_empty) == len(CONFIGS):
        make_figure(non_empty, CONFIGS)
        make_overlay_figure(non_empty, CONFIGS)
    print("Done.")


if __name__ == "__main__":
    main()
