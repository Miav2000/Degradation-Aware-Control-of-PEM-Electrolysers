"""
make_input_profiles_figure.py — Clean two-panel figure of the 48-hour
wind-power and spot-price mission profile used as simulation inputs.

Annotations:
  • P_rating × 1.10 — system power ceiling (stack + 10 % aux margin)
  • P_min = 10 % × P_rating = 45 kW — shutdown threshold

Usage:
    python3 run/make_input_profiles_figure.py
    python3 run/make_input_profiles_figure.py --hours 72 --outdir results/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

mpl.use("Agg")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_style, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, DARK
apply_style()
mpl.rcParams["figure.figsize"] = (7.0, 6.0)   # taller for two panels

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
WIND_CSV   = ROOT / "configs/power_profiles/energinet_wind_price_data/wind_power.csv"
PRICE_CSV  = ROOT / "configs/power_profiles/energinet_wind_price_data/spot_price.csv"
PLANT_YAML = ROOT / "configs/plant_parameters.yaml"


def main(n_hours: int, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load plant parameters ─────────────────────────────────────────────────
    with open(PLANT_YAML) as f:
        plant = yaml.safe_load(f)

    P_rating_kW = plant["stack"]["P_rating_W"] / 1e3        # 450 kW
    aux_margin  = plant["auxiliaries"]["aux_margin_frac"]    # 0.10
    P_min_frac  = plant["stack"]["P_min_frac"]               # 0.10
    P_system_kW = P_rating_kW * (1.0 + aux_margin)          # 495 kW (system ceiling)
    P_min_kW    = P_min_frac * P_rating_kW                   # 45 kW  (shutdown threshold)

    # ── Load data ─────────────────────────────────────────────────────────────
    wind  = pd.read_csv(WIND_CSV)
    price = pd.read_csv(PRICE_CSV)

    t_h   = wind["t_s"].values[:n_hours] / 3600.0
    P_kW  = wind["P_W"].values[:n_hours] / 1e3
    c_mwh = price["price_eur_per_kWh"].values[:n_hours] * 1e3   # EUR/MWh

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.10},
    )

    # ── Panel 1: Wind power ───────────────────────────────────────────────────
    ax1.fill_between(t_h, P_kW, alpha=0.25, color=BLUE)
    ax1.plot(t_h, P_kW, color=BLUE, lw=2.0,
             label=r"$P_{\mathrm{wind}}$")

    ax1.axhline(P_system_kW, color=DARK, lw=1.4, ls="--", alpha=0.70,
                label=rf"$P_{{\mathrm{{max}}}} = {P_system_kW:.0f}\ \mathrm{{kW}}$")
    ax1.axhline(P_rating_kW, color=GREY, lw=1.2, ls=":",  alpha=0.80,
                label=rf"$P_{{\mathrm{{rating}}}} = {P_rating_kW:.0f}\ \mathrm{{kW}}$")
    ax1.axhline(P_min_kW,    color=RED,  lw=1.4, ls="--", alpha=0.75,
                label=rf"$P_{{\mathrm{{min}}}} = {P_min_kW:.0f}\ \mathrm{{kW}}$")

    ax1.fill_between(t_h, 0, P_min_kW, alpha=0.08, color=RED)
    ax1.text(n_hours * 0.01, P_min_kW / 2, "shutdown zone",
             va="center", fontsize=8, color=RED, alpha=0.85)

    ax1.set_ylabel(r"Available wind power (kW)")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="lower right", framealpha=0.92, borderpad=0.6)

    # ── Panel 2: Spot price ───────────────────────────────────────────────────
    ax2.fill_between(t_h, c_mwh, alpha=0.25, color=ORANGE)
    ax2.plot(t_h, c_mwh, color=ORANGE, lw=2.0,
             label=r"$p_{\mathrm{elec}}$")

    ax2.set_ylabel(r"Spot price (EUR/MWh)")
    ax2.set_xlabel(r"Time (h)")
    ax2.set_xlim(0, n_hours)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="lower right", framealpha=0.92, borderpad=0.6)

    ax2.xaxis.set_major_locator(mticker.MultipleLocator(6))
    ax2.xaxis.set_minor_locator(mticker.MultipleLocator(1))

    # ── Save ──────────────────────────────────────────────────────────────────
    for ext in ("png", "pdf"):
        out = outdir / f"input_profiles_{n_hours}h.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out}")

    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours",  type=int, default=48,
                   help="Number of hours to show (default: 48)")
    p.add_argument("--outdir", default="results/figures",
                   help="Output directory (default: results/figures)")
    args = p.parse_args()
    main(args.hours, ROOT / args.outdir)
