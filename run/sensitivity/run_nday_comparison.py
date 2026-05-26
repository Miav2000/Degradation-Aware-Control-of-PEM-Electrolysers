"""
run_nday_comparison.py — Short-horizon comparison run for controller validation.

Runs all four controllers over the first N days of the wind/price data and
generates temperature dynamics plots. Useful for validating regulatory-layer
behaviour (PI tracking, startup sequence, flow response) before committing
to a full 1-year or 5-year run.

Output:
  results/nday_comparison/<N>day/
    commercial_<N>day.csv
    cost_optimal_<N>day.csv
    aware_<N>day.csv
    aware_rul_<N>day.csv
    dynamics_temperature.png
    dynamics_current_flow.png

Usage:
  # Run one controller (for parallel launch):
  python3 run/sensitivity/run_nday_comparison.py --days 5 --controller commercial
  python3 run/sensitivity/run_nday_comparison.py --days 5 --controller cost_optimal
  python3 run/sensitivity/run_nday_comparison.py --days 5 --controller aware
  python3 run/sensitivity/run_nday_comparison.py --days 5 --controller aware_rul

  # Once all CSVs exist, generate plots:
  python3 run/sensitivity/run_nday_comparison.py --days 5 --plots-only

  # Run all four in parallel (bash):
  for c in commercial cost_optimal aware aware_rul; do
    nohup python3 run/sensitivity/run_nday_comparison.py --days 5 --controller $c \
      > /tmp/nday_$c.log 2>&1 &
  done
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pemwe.plant import load_plant, load_config
from pemwe.simulation import run_simulation


PLANT_CFG   = REPO_ROOT / "configs" / "plant_parameters.yaml"
WIND_CSV    = REPO_ROOT / "configs" / "power_profiles" / "energinet_wind_price_data" / "wind_power.csv"
PRICE_CSV   = REPO_ROOT / "configs" / "power_profiles" / "energinet_wind_price_data" / "spot_price.csv"

CONTROLLERS = {
    "load_following":   REPO_ROOT / "configs" / "controllers" / "load_following.yaml",
    "price_aware": REPO_ROOT / "configs" / "controllers" / "price_aware.yaml",
    "degradation_aware":        REPO_ROOT / "configs" / "controllers" / "degradation_aware.yaml",
    "lifetime_aware":    REPO_ROOT / "configs" / "controllers" / "lifetime_aware.yaml",
}

COLORS = {"load_following": "C0", "price_aware": "C1", "degradation_aware": "C2", "lifetime_aware": "C3"}
LABELS = {"load_following": "Load-following", "price_aware": "Price-aware",
          "degradation_aware": "Degradation-aware", "lifetime_aware": "Aware+RUL"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Short N-day comparison run.")
    p.add_argument("--days", type=int, default=5, help="Number of days to simulate (default 5)")
    p.add_argument("--controller", choices=list(CONTROLLERS),
                   help="Run a single controller (for parallel launch).")
    p.add_argument("--plots-only", action="store_true",
                   help="Skip simulation; regenerate plots from existing CSVs.")
    return p.parse_args()


def load_profiles(n_hours: int):
    wind  = pd.read_csv(WIND_CSV)
    price = pd.read_csv(PRICE_CSV)
    p_avail = wind["P_W"].values[:n_hours].tolist()
    prices  = price["price_eur_per_kWh"].values[:n_hours].tolist()
    return p_avail, prices


def run_one(name: str, n_hours: int, out_dir: Path):
    # Remove stale outputs for this controller before running
    csv_path_pre = out_dir / f"{name}_{n_hours}h.csv"
    if csv_path_pre.exists():
        csv_path_pre.unlink()
    for png in out_dir.glob("dynamics_*.png"):
        png.unlink()

    plant = load_plant(PLANT_CFG)
    p_avail, prices = load_profiles(n_hours)
    ctrl_cfg = load_config(CONTROLLERS[name])
    print(f"  Running {name} ({n_hours}h)...")
    df = run_simulation(
        price_series=prices,
        p_avail_series=p_avail,
        plant=plant,
        ctrl_cfg=ctrl_cfg,
        dt_s=3600.0,
    )
    csv_path = out_dir / f"{name}_{n_hours}h.csv"
    df.to_csv(csv_path, index=False)
    on = df[df["j_A_per_m2"] > 0]
    print(f"    {name}: T_mean(running)={on['T_stack_avg_K'].mean()-273.15:.1f}°C  "
          f"T_max={df['T_stack_max_K'].max()-273.15:.1f}°C  saved {csv_path.name}")


def load_results(n_hours: int, out_dir: Path) -> dict:
    results = {}
    for name in CONTROLLERS:
        csv_path = out_dir / f"{name}_{n_hours}h.csv"
        if csv_path.exists():
            results[name] = pd.read_csv(csv_path)
        else:
            print(f"  [warning] missing {csv_path.name}")
    return results


def make_plots(results: dict, p_avail: list, out_dir: Path, n_hours: int):
    t = results["load_following"]["t_h"].to_numpy()

    # ------------------------------------------------------------------ #
    # Figure 1: Temperature dynamics                                      #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Temperature dynamics — {n_hours//24}-day run (PI regulatory controller)")

    # Available power
    ax = axes[0]
    ax.fill_between(t, np.array(p_avail) / 1e6, alpha=0.3, color="grey", label="P_avail")
    ax.set_ylabel("P_avail [MW]")
    ax.legend(loc="upper right")

    # Stack temperature: actual (solid) vs target (dashed)
    ax = axes[1]
    for name, df in results.items():
        ax.plot(t, df["T_stack_avg_K"].to_numpy() - 273.15,
                color=COLORS[name], label=f"{LABELS[name]} actual", lw=1.4)
        if "T_target_K" in df.columns:
            ax.plot(t, df["T_target_K"].to_numpy() - 273.15,
                    color=COLORS[name], lw=0.8, ls="--", alpha=0.6,
                    label=f"{LABELS[name]} target")
    ax.axhline(50, color="k", lw=0.8, ls=":", label="50°C")
    ax.set_ylabel("T_stack [°C]")
    ax.legend(fontsize=6, ncol=3, loc="lower right")

    # Offset: T_actual - T_target (shows PI tracking quality)
    ax = axes[2]
    for name, df in results.items():
        if "T_target_K" in df.columns:
            offset = (df["T_stack_avg_K"] - df["T_target_K"]).to_numpy()
            ax.plot(t, offset, color=COLORS[name], label=LABELS[name], lw=1.0)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_ylabel("T_actual − T_target [K]")
    ax.set_xlabel("Time [h]")
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    out_path = out_dir / "dynamics_temperature.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path.name}")

    # ------------------------------------------------------------------ #
    # Figure 2: Current density and water flow                           #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"Current density and flow — {n_hours//24}-day run")

    ax = axes[0]
    for name, df in results.items():
        ax.plot(t, df["j_A_per_m2"].to_numpy(), color=COLORS[name], label=LABELS[name], lw=1.0)
    ax.set_ylabel("j [A/m²]")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for name, df in results.items():
        ax.plot(t, df["m_dot_w_kg_s"].to_numpy(), color=COLORS[name], label=LABELS[name], lw=1.0)
    ax.set_ylabel("m_dot_w [kg/s]")
    ax.set_xlabel("Time [h]")
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    out_path = out_dir / "dynamics_current_flow.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path.name}")


def print_summary(results: dict):
    print("\n--- Tracking summary (running hours only) ---")
    for name, df in results.items():
        on = df[df["j_A_per_m2"] > 0]
        if "T_target_K" in df.columns:
            offset = (on["T_stack_avg_K"] - on["T_target_K"])
            print(f"  {name:15s}  T_mean={on['T_stack_avg_K'].mean()-273.15:.1f}°C  "
                  f"offset mean={offset.mean():.2f}K  max={offset.max():.2f}K")
        else:
            print(f"  {name:15s}  T_mean={on['T_stack_avg_K'].mean()-273.15:.1f}°C")


def main():
    args = parse_args()
    n_hours = args.days * 24

    out_dir = REPO_ROOT / "results" / "nday_comparison" / f"{args.days}day"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        results = load_results(n_hours, out_dir)
        p_avail, _ = load_profiles(n_hours)
        make_plots(results, p_avail, out_dir, n_hours)
        print_summary(results)
        return

    if args.controller:
        run_one(args.controller, n_hours, out_dir)
        return

    # No --controller specified: run all sequentially
    print(f"=== {args.days}-day comparison ({n_hours} hours) ===")
    for name in CONTROLLERS:
        run_one(name, n_hours, out_dir)
    results = load_results(n_hours, out_dir)
    p_avail, _ = load_profiles(n_hours)
    make_plots(results, p_avail, out_dir, n_hours)
    print_summary(results)
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
