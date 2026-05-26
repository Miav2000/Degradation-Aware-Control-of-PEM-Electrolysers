"""
single_simulation.py — Run a single PEMWE simulation case from YAML configs.

The run name and output folder are auto-derived from the config file names:
  controller=aware + power=wind + price=spot_dk1  →  results/degradation_aware_wind_spot_dk1/

Usage (via Makefile):
  python3 run/single_simulation.py \
      --plant configs/plant_parameters.yaml \
      --controller configs/controllers/degradation_aware.yaml \
      --power-profile configs/power_profiles/wind.yaml \
      --price-profile configs/price_profiles/spot_dk1.yaml

Override auto-naming:
  python3 run/single_simulation.py ... --name my_run --outdir results/my_run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pemwe.plant import load_plant, load_config
from pemwe.simulation import run_simulation


# ---------------------------------------------------------------------------
# Profile loaders
# ---------------------------------------------------------------------------

def _load_file_profile(cfg: dict, key_col: str, repo_root: Path) -> list[float]:
    """Load a time-series column from a CSV file referenced in a profile config.

    If the column contains NaN values, the valid portion is repeated (wrapped)
    to fill the gaps — e.g. incomplete price data wraps from the start.
    """
    path = repo_root / cfg["path"]
    df = pd.read_csv(path)
    if key_col not in df.columns:
        raise KeyError(f"Column '{key_col}' not found in {path}. Available: {list(df.columns)}")
    series = df[key_col]
    if series.isna().any():
        valid = series.dropna().values
        n_total = len(series)
        filled = np.tile(valid, (n_total // len(valid)) + 1)[:n_total]
        print(f"  [note] {key_col} in {path.name}: {n_total - len(valid)} NaN values "
              f"filled by wrapping first {len(valid)} valid entries")
        return filled.tolist()
    return series.tolist()


def load_power_profile(power_cfg: dict, n_steps: int, dt_s: float, repo_root: Path) -> list[float]:
    """Build P_avail series from a power profile config."""
    p_cfg = power_cfg["P_avail_profile"]
    p_type = p_cfg["type"]

    if p_type == "file":
        p_avail = _load_file_profile(p_cfg, p_cfg.get("power_col", "P_W"), repo_root)
        if len(p_avail) < n_steps:
            raise ValueError(
                f"Power CSV has {len(p_avail)} rows but simulation needs {n_steps} steps."
            )
        return p_avail[:n_steps]

    elif p_type == "sine":
        t = np.arange(n_steps) * dt_s
        P_mean = float(p_cfg["P_mean_W"])
        P_amp  = float(p_cfg["P_amp_W"])
        period = float(p_cfg["period_s"])
        phase  = float(p_cfg.get("phase_rad", 0.0))
        P_min  = float(p_cfg.get("P_min_W", 0.0))
        P_max  = float(p_cfg.get("P_max_W", 1e12))
        return np.clip(P_mean + P_amp * np.sin(2 * np.pi * t / period + phase), P_min, P_max).tolist()

    elif p_type == "constant":
        return [float(p_cfg["P_avail_W"])] * n_steps

    else:
        raise ValueError(f"Unknown P_avail_profile type: '{p_type}'")


def load_price_profile(price_cfg: dict, n_steps: int, repo_root: Path) -> list[float]:
    """Build price series from a price profile config."""
    pr_cfg = price_cfg["price_profile"]
    pr_type = pr_cfg["type"]

    if pr_type == "file":
        price = _load_file_profile(pr_cfg, pr_cfg.get("price_col", "price_eur_per_kWh"), repo_root)
        if len(price) < n_steps:
            raise ValueError(
                f"Price CSV has {len(price)} rows but simulation needs {n_steps} steps."
            )
        return price[:n_steps]

    elif pr_type == "constant":
        return [float(pr_cfg["price_eur_per_kWh"])] * n_steps

    else:
        raise ValueError(f"Unknown price_profile type: '{pr_type}'")


def derive_run_name(controller: Path, power_profile: Path, price_profile: Path) -> str:
    """Auto-derive a run name from config file stems.

    Example: aware + wind + spot_dk1 → 'degradation_aware_wind_spot_dk1'
    """
    ctrl_stem  = controller.stem if controller else "nocontrol"
    power_stem = power_profile.stem
    price_stem = price_profile.stem
    return f"{ctrl_stem}_{power_stem}_{price_stem}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single PEMWE simulation case.")
    parser.add_argument("--plant",         required=True,  type=Path, help="plant_parameters.yaml")
    parser.add_argument("--controller",    required=False, type=Path, default=None,
                        help="Controller YAML (optional; omit for validation runs)")
    parser.add_argument("--power-profile", required=True,  type=Path, help="Power profile YAML")
    parser.add_argument("--price-profile", required=True,  type=Path, help="Price profile YAML")
    parser.add_argument("--name",          required=False, type=str,  default=None,
                        help="Run name (auto-derived from configs if omitted)")
    parser.add_argument("--outdir",        required=False, type=Path, default=None,
                        help="Output directory (auto: results/<name>/ if omitted)")
    parser.add_argument("--detail-window", required=False, type=str, default=None,
                        help="Sub-step detail window as 'start,end' in hours (e.g. '2386,2396')")
    parser.add_argument("--n-substeps", type=int, default=None, dest="n_substeps",
                        help="Override n_substeps from plant config (for substep sensitivity tests)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Auto-derive name and outdir ---
    if args.name is None:
        args.name = derive_run_name(args.controller, args.power_profile, args.price_profile)
    if args.outdir is None:
        args.outdir = REPO_ROOT / "results" / args.name

    # --- Load configs ---
    plant     = load_plant(args.plant)
    if args.n_substeps is not None:
        plant["fast_control"]["n_substeps"] = int(args.n_substeps)
        print(f"  n_substeps:    {args.n_substeps} (overridden)")
    power_cfg = load_config(args.power_profile)
    price_cfg = load_config(args.price_profile)
    ctrl_cfg  = load_config(args.controller) if args.controller else None

    # --- Simulation parameters (from power profile) ---
    sim_cfg    = power_cfg["simulation"]
    dt_s       = float(sim_cfg["dt_s"])
    t_end_s    = float(sim_cfg["t_end_s"])
    save_every = int(sim_cfg.get("save_every", 1))
    n_steps    = int(round(t_end_s / dt_s))

    print(f"=== single_simulation: {args.name} ===")
    print(f"  Plant:         {args.plant}")
    print(f"  Controller:    {args.controller}")
    print(f"  Power profile: {args.power_profile}")
    print(f"  Price profile: {args.price_profile}")
    print(f"  Output:        {args.outdir}")
    print(f"  Duration:      {n_steps} steps × {dt_s} s = {t_end_s/3600:.1f} h")

    # --- Build input profiles ---
    p_avail_series = load_power_profile(power_cfg, n_steps, dt_s, REPO_ROOT)
    price_series   = load_price_profile(price_cfg, n_steps, REPO_ROOT)

    # --- Parse detail window ---
    if args.detail_window:
        parts = args.detail_window.split(",")
        detail_window_h = (float(parts[0]), float(parts[1]))
    else:
        detail_window_h = (2386, 2396)  # default: startup event for DK1 wind profile
    print(f"  Detail window: hours {detail_window_h[0]:.0f} to {detail_window_h[1]:.0f}")
    # --- Run simulation ---
    print(f"  Running simulation...")
    df = run_simulation(
        price_series=price_series,
        p_avail_series=p_avail_series,
        plant=plant,
        ctrl_cfg=ctrl_cfg,
        dt_s=dt_s,
        detail_window_h=detail_window_h,
    )

    # Apply save_every subsampling
    if save_every > 1:
        df = df.iloc[::save_every].reset_index(drop=True)

    # --- Save sub-step detail CSV if available ---
    if hasattr(df, 'attrs') and "substep_df" in df.attrs:
        sub_df = df.attrs["substep_df"]
        args.outdir.mkdir(parents=True, exist_ok=True)
        sub_csv = args.outdir / f"{args.name}_substeps.csv"
        sub_df.to_csv(sub_csv, index=False)
        print(f"  Sub-step detail: {sub_csv}  ({len(sub_df)} rows)")

    # --- Save results (CSV + figures in same folder) ---
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = args.outdir / f"{args.name}.csv"
    df.to_csv(out_csv, index=False)

    dt_h = dt_s / 3600.0
    # --- Summary ---
    t_wall = df.attrs.get("t_wall_s", float("nan"))
    t_step_mean = df["t_step_s"].mean()
    t_step_max  = df["t_step_s"].max()

    print(f"\nResults saved: {out_csv}  ({len(df)} rows)")
    print(f"  Total H2 produced:   {df['m_dot_H2_kg_h'].sum() * dt_h:.2f} kg")
    print(f"  Total true profit:   {df['true_profit_eur_h'].sum() * dt_h:.2f} €")
    print(f"  Final degradation:   {df['V_deg_V'].iloc[-1]*1000:.3f} mV")
    print(f"  Mean stack temp:     {df['T_stack_actual_K'].mean() - 273.15:.1f} °C")
    print(f"  Wall-clock time:     {t_wall:.1f} s  ({t_wall/60:.1f} min)")
    print(f"  Time per step:       mean {t_step_mean*1000:.1f} ms   max {t_step_max*1000:.1f} ms")

    # --- Auto-generate figures (into same outdir) ---
    plot_script = REPO_ROOT / "run" / "make_plots.py"
    print(f"\n  Generating figures → {args.outdir} ...")
    import subprocess
    ret = subprocess.run(
        [sys.executable, str(plot_script),
         "--input",  str(out_csv),
         "--name",   args.name,
         "--outdir", str(args.outdir)],
        capture_output=True, text=True,
    )
    if ret.returncode == 0:
        print(ret.stdout.strip())
    else:
        print(f"  [make_plots warning] {ret.stderr.strip()[:300]}")


if __name__ == "__main__":
    main()
