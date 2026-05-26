"""
lifetime_5yr.py — 5-year lifetime comparison.

Runs each of the four controllers over a 5-year simulated period (the 1-year
wind/price data is wrapped/tiled to fill 5 × 8760 hours). The point is to let
multi-year degradation effects accumulate so the lifetime-aware controller
can show its long-term difference in operation compared to degradation-aware.

The single-year runs use linear extrapolation from the first year, which
underestimates the degradation. Thus, more runs more accurately capture long-term operation.

Output structure:
  results/lifetime_5yr/
    load_following_5yr/  -- CSV + figures
    price_aware_5yr/
    degradation_aware_5yr/
    lifetime_aware_5yr/
    comparison_5yr/  -- four-way comparison plots

Usage:
  python3 run/sensitivity/lifetime_5yr.py --controller commercial
  python3 run/sensitivity/lifetime_5yr.py --controller cost_optimal
  python3 run/sensitivity/lifetime_5yr.py --controller aware
  python3 run/sensitivity/lifetime_5yr.py --controller aware_rul
  python3 run/sensitivity/lifetime_5yr.py --plots-only

To run all four in parallel (4 cores):
  cd PEMWE_system_model
  for c in commercial cost_optimal aware aware_rul; do
    nohup python3 run/sensitivity/lifetime_5yr.py --controller $c \
      > /tmp/lt5yr_$c.log 2>&1 &
  done

Then once all four CSVs exist:
  python3 run/sensitivity/lifetime_5yr.py --plots-only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pemwe.plant import load_plant, load_config
from pemwe.simulation import run_simulation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_YEARS = 5    # Number of years to simulate. Can be changed if you want a shorter/longer test run.
HOURS_PER_YEAR = 8760
N_STEPS = N_YEARS * HOURS_PER_YEAR
DT_S = 3600.0

PLANT_CFG = REPO_ROOT / "configs" / "plant_parameters.yaml"
POWER_CFG = REPO_ROOT / "configs" / "power_profiles" / "wind.yaml"
PRICE_CFG = REPO_ROOT / "configs" / "price_profiles" / "spot_dk1.yaml"

CONTROLLERS = {
    "load_following":   REPO_ROOT / "configs" / "controllers" / "load_following.yaml",
    "price_aware": REPO_ROOT / "configs" / "controllers" / "price_aware.yaml",
    "degradation_aware":        REPO_ROOT / "configs" / "controllers" / "degradation_aware.yaml",
    "lifetime_aware":    REPO_ROOT / "configs" / "controllers" / "lifetime_aware.yaml",
}

OUT_BASE = REPO_ROOT / "results" / "lifetime_5yr"


# ---------------------------------------------------------------------------
# Data loading with wrapping for multi-year runs
# ---------------------------------------------------------------------------

def _load_csv_column(path: Path, col: str) -> np.ndarray:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not in {path}: {list(df.columns)}")
    series = df[col].dropna().values
    return series


def load_5yr_power_profile() -> list[float]:
    cfg = load_config(POWER_CFG)
    p_cfg = cfg["P_avail_profile"]
    if p_cfg["type"] != "file":
        raise ValueError("Only file-type power profiles supported for 5yr run.")
    csv_path = REPO_ROOT / p_cfg["path"]
    valid = _load_csv_column(csv_path, p_cfg.get("power_col", "P_W"))
    n_valid = len(valid)
    n_reps = (N_STEPS // n_valid) + 1
    tiled = np.tile(valid, n_reps)[:N_STEPS]
    print(f"  Power: tiled {n_valid} hours x {n_reps} -> {len(tiled)} hours ({N_YEARS} years)")
    return tiled.tolist()


def load_5yr_price_profile() -> list[float]:
    cfg = load_config(PRICE_CFG)
    pr_cfg = cfg["price_profile"]
    if pr_cfg["type"] != "file":
        raise ValueError("Only file-type price profiles supported for 5yr run.")
    csv_path = REPO_ROOT / pr_cfg["path"]
    valid = _load_csv_column(csv_path, pr_cfg.get("price_col", "price_eur_per_kWh"))
    n_valid = len(valid)
    n_reps = (N_STEPS // n_valid) + 1
    tiled = np.tile(valid, n_reps)[:N_STEPS]
    print(f"  Price: tiled {n_valid} hours x {n_reps} -> {len(tiled)} hours ({N_YEARS} years)")
    return tiled.tolist()


# ---------------------------------------------------------------------------
# Single controller run
# ---------------------------------------------------------------------------

def run_one_controller(name: str) -> Path:
    """Run one 5-year simulation and return the CSV path."""
    if name not in CONTROLLERS:
        raise KeyError(f"Unknown controller: {name}. Choose from {list(CONTROLLERS)}")

    out_dir = OUT_BASE / f"{name}_5yr"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {name}_5yr ===")
    plant = load_plant(PLANT_CFG)
    ctrl_cfg = load_config(CONTROLLERS[name])

    print("Loading 5-year wrapped profiles...")
    p_avail = load_5yr_power_profile()
    price   = load_5yr_price_profile()

    print(f"Running {N_STEPS} hours ({N_YEARS} years)...")
    df = run_simulation(
        price_series=price,
        p_avail_series=p_avail,
        plant=plant,
        ctrl_cfg=ctrl_cfg,
        dt_s=DT_S,
    )

    csv_path = out_dir / f"{name}_5yr.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Per-controller plots
    plot_script = REPO_ROOT / "run" / "make_plots.py"
    print(f"Generating per-controller figures...")
    ret = subprocess.run(
        [sys.executable, str(plot_script),
         "--input", str(csv_path),
         "--name", f"{name}_5yr",
         "--outdir", str(out_dir)],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"  [plot warning] {ret.stderr.strip()[:300]}")
    else:
        print(ret.stdout.strip().splitlines()[-2:])

    return csv_path


# ---------------------------------------------------------------------------
# Comparison plots from existing CSVs
# ---------------------------------------------------------------------------

def make_comparison_plots():
    """Generate four-way comparison plots from the four 5-year CSVs."""
    cmp_dir = OUT_BASE / "comparison_5yr"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    csvs = {}
    for name in CONTROLLERS:
        csv = OUT_BASE / f"{name}_5yr" / f"{name}_5yr.csv"
        if not csv.exists():
            print(f"  [warning] missing CSV: {csv} -- skipping plot for {name}")
        else:
            csvs[name] = csv

    if "load_following" not in csvs or "degradation_aware" not in csvs:
        print("Need at least commercial and aware CSVs for comparison plots.")
        return

    cmd = [
        sys.executable, str(REPO_ROOT / "run" / "make_plots.py"),
        "--load-following",    str(csvs["load_following"]),
        "--degradation-aware", str(csvs["degradation_aware"]),
        "--name",       "comparison_5yr",
        "--outdir",     str(cmp_dir),
    ]
    if "price_aware" in csvs:
        cmd += ["--price-aware", str(csvs["price_aware"])]
    if "lifetime_aware" in csvs:
        cmd += ["--lifetime-aware", str(csvs["lifetime_aware"])]

    print("Generating comparison plots...")
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode == 0:
        print(ret.stdout.strip())
    else:
        print(f"  [error] {ret.stderr.strip()[:500]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-year lifetime sensitivity comparison.")
    p.add_argument("--controller", choices=list(CONTROLLERS),
                   help="Run a single controller. Omit + use --plots-only to make figures.")
    p.add_argument("--plots-only", action="store_true",
                   help="Skip simulation; just regenerate comparison plots from existing CSVs.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.plots_only:
        make_comparison_plots()
        return

    if args.controller is None:
        print("Specify --controller <name> or --plots-only", file=sys.stderr)
        sys.exit(1)

    run_one_controller(args.controller)


if __name__ == "__main__":
    main()
