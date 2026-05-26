"""
preprocess_energinet.py — Convert raw Energinet EnergyReport CSV to model-ready profiles.

Reads:
  configs/power_profiles/energinet_wind_price_data/EnergyReport_EUR.csv
    Columns (semicolon-delimited, European decimal commas):
      HourUTC | HourDK | OffshoreWindGe100MW_MWhDK1 | DK1_EUR/MWh

Writes (to same folder):
  wind_power.csv   — columns: t_s, P_W
  spot_price.csv   — columns: t_s, price_eur_per_kWh

Scaling logic
-------------
Wind values are MWh produced in each hour = MW of average power that hour.
The dataset covers the entire DK1 offshore wind fleet (O(GW)).
We scale it down so the peak matches P_rated+10% (P_stack_rated + P_aux).

  P_plant_rated = P_stack_rated + P_aux(j_rated)
  S_norm        = P_plant_rated / (peak_wind_MW * 1e6)   [dimensionless]
  P_avail_W     = wind_MW * 1e6 * S_norm                  [W]

Price conversion
----------------
  EUR/kWh = EUR/MWh  /  1000

Usage
-----
  python configs/power_profiles/energinet_wind_price_data/preprocess_energinet.py
  python configs/power_profiles/energinet_wind_price_data/preprocess_energinet.py [--rated-power-w 100000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE      = Path(__file__).resolve().parent          # configs/power_profiles/energinet_wind_price_data/
REPO_ROOT = HERE.parent.parent.parent                # repo root
RAW_CSV   = HERE / "EnergyReport_EUR.csv"
OUT_DIR   = HERE
PLANT_YAML = REPO_ROOT / "configs" / "plant_parameters.yaml"


def load_plant_params(plant_path: Path) -> dict:
    """
    Read P_rating, P_min_frac, and aux_margin_frac from plant_parameters.yaml.

    Returns dict with P_plant_rated and P_min_frac.
    """
    with open(plant_path) as f:
        p = yaml.safe_load(f)
    stack = p["stack"]
    P_rating    = float(stack["P_rating_W"])
    P_min_frac  = float(stack["P_min_frac"])
    aux_margin  = float(p["auxiliaries"]["aux_margin_frac"])
    P_plant_rated = P_rating * (1.0 + aux_margin)

    print(f"  P_stack_rated:  {P_rating/1e3:.1f} kW")
    print(f"  Aux margin:     {aux_margin*100:.0f}%")
    print(f"  P_plant_rated:  {P_plant_rated/1e3:.1f} kW  (wind peak)")
    print(f"  P_min_frac:     {P_min_frac*100:.0f}%  →  P_min = {P_rating*P_min_frac/1e3:.1f} kW")

    return {
        "P_plant_rated": P_plant_rated,
        "P_min_frac": P_min_frac,
        "P_rating": P_rating,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Energinet EnergyReport CSV.")
    parser.add_argument(
        "--plant", type=Path, default=PLANT_YAML,
        help="Path to plant_parameters.yaml.",
    )
    parser.add_argument(
        "--raw-csv", type=Path, default=RAW_CSV,
        help="Path to raw EnergyReport_EUR.csv.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help="Output directory for processed CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- All parameters from plant_parameters.yaml ---
    params = load_plant_params(args.plant)
    P_rated_W = params["P_plant_rated"]
    P_min_W   = params["P_min_frac"] * params["P_rating"]

    # --- Read raw CSV ---
    print(f"\nReading: {args.raw_csv}")
    df = pd.read_csv(
        args.raw_csv,
        sep=";",
        decimal=",",          # European format: 1.285,866 → 1285.866
        parse_dates=["HourDK"],
    )
    print(f"  Rows: {len(df)},  columns: {list(df.columns)}")

    # Rename for clarity
    df = df.rename(columns={
        "OffshoreWindGe100MW_MWhDK1": "wind_MWh",
        "DK1_EUR/MWh":               "price_eur_mwh",
    })

    # Sort by time (should already be sorted, but to be safe)
    df = df.sort_values("HourDK").reset_index(drop=True)

    n_hours = len(df)
    dt_s    = 3600.0

    # --- Wind scaling ---
    peak_wind_MW = df["wind_MWh"].max()   # MWh/h ≡ MW
    S_norm       = P_rated_W / (peak_wind_MW * 1e6)
    df["P_avail_W"] = df["wind_MWh"] * 1e6 * S_norm

    # No P_min floor applied — the wind profile represents real available power.
    # The simulation loop handles shutdown when P_avail < P_min.
    n_below_pmin = (df["P_avail_W"] < P_min_W).sum()

    print(f"\nWind scaling:")
    print(f"  Peak raw wind:      {peak_wind_MW:.1f} MW")
    print(f"  Scale factor:       {S_norm:.6f}")
    print(f"  P_rated:            {P_rated_W/1e3:.1f} kW")
    print(f"  P_min reference:    {P_min_W/1e3:.2f} kW ({params['P_min_frac']*100:.0f}% of P_rating)")
    print(f"  Hours below P_min:  {n_below_pmin} / {n_hours}  ({100*n_below_pmin/n_hours:.1f}%)")
    print(f"  P_avail mean:       {df['P_avail_W'].mean()/1e3:.2f} kW")

    # --- Price conversion: EUR/MWh → EUR/kWh ---
    df["price_eur_per_kWh"] = df["price_eur_mwh"] / 1000.0

    print(f"\nSpot price (EUR/kWh):")
    print(f"  Min:  {df['price_eur_per_kWh'].min():.5f}")
    print(f"  Mean: {df['price_eur_per_kWh'].mean():.5f}")
    print(f"  Max:  {df['price_eur_per_kWh'].max():.5f}")
    n_neg = (df["price_eur_per_kWh"] < 0).sum()
    if n_neg:
        print(f"  Note: {n_neg} hours have negative prices (kept - optimizer will exploit them).")

    # --- Build t_s column ---
    df["t_s"] = [k * dt_s for k in range(n_hours)]

    # --- Write outputs ---
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wind_out  = args.out_dir / "wind_power.csv"
    price_out = args.out_dir / "spot_price.csv"

    df[["t_s", "P_avail_W"]].rename(columns={"P_avail_W": "P_W"}).to_csv(
        wind_out, index=False
    )
    df[["t_s", "price_eur_per_kWh"]].to_csv(price_out, index=False)

    print(f"\nOutputs written:")
    print(f"  {wind_out}")
    print(f"  {price_out}")
    print(f"\nSimulation duration: {n_hours} hours  ({n_hours/8760*365:.0f} days)")
    print("Done.")


if __name__ == "__main__":
    main()
