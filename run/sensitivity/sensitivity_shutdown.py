"""
sensitivity_shutdown.py — Sensitivity of optimising controllers to shutdown cost
in the objective function.

Currently c_shutdown is charged to LCOH post-hoc but does NOT influence the
optimizer's decisions.  This script enables the 'include_shutdown' flag so the
optimizer may voluntarily choose j=0 when operating is net-costly and the
one-time shutdown penalty is smaller than the interval operating loss.

Controllers tested: price_aware, degradation_aware, lifetime_aware
c_shutdown levels : 50, 100, 150 EUR/event
Baseline          : existing 1-year results (shutdown NOT in objective)

Output:
  results/sensitivity_shutdown/
    {ctrl}_sd{cost}EUR.csv
    fig_shutdown_sensitivity.png
    table_shutdown_sensitivity.tex

Usage:
  python3 run/sensitivity/sensitivity_shutdown.py            # run + plot
  python3 run/sensitivity/sensitivity_shutdown.py --parallel # parallel runs
  python3 run/sensitivity/sensitivity_shutdown.py --plots-only
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "run"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pemwe.plant import load_plant, load_config
from pemwe.simulation import run_simulation
from plot_style import apply_style, BLUE, ORANGE, GREEN, PURPLE, GREY

apply_style()

# ── Constants ────────────────────────────────────────────────────────────────

OUT_DIR   = REPO_ROOT / "results" / "sensitivity_shutdown"
PLANT_CFG = REPO_ROOT / "configs" / "plant_parameters.yaml"

CTRL_CFGS = {
    "price_aware":       REPO_ROOT / "configs" / "controllers" / "price_aware.yaml",
    "degradation_aware": REPO_ROOT / "configs" / "controllers" / "degradation_aware.yaml",
    "lifetime_aware":    REPO_ROOT / "configs" / "controllers" / "lifetime_aware.yaml",
}

BASELINE_CSVS = {
    "price_aware":       REPO_ROOT / "results" / "price_aware_wind_spot_dk1"       / "price_aware_wind_spot_dk1.csv",
    "degradation_aware": REPO_ROOT / "results" / "degradation_aware_wind_spot_dk1" / "degradation_aware_wind_spot_dk1.csv",
    "lifetime_aware":    REPO_ROOT / "results" / "lifetime_aware_wind_spot_dk1"    / "lifetime_aware_wind_spot_dk1.csv",
}

SHUTDOWN_COSTS = [50, 100, 150]   # EUR/event

CTRL_LABEL = {
    "price_aware":       "Price-aware",
    "degradation_aware": "Deg.-aware",
    "lifetime_aware":    "Lifetime-aware",
}
CTRL_COLOR = {
    "price_aware":       GREEN,
    "degradation_aware": ORANGE,
    "lifetime_aware":    PURPLE,
}

V_DEG_EOL = 0.10   # [V] end-of-life threshold
DT_H      = 1.0    # supervisory interval [h]


# ── Profile loading ──────────────────────────────────────────────────────────

def _load_series(path: Path, col: str, n: int = 8760) -> list[float]:
    """Load a column, wrapping shorter/NaN-padded series to exactly n steps."""
    raw = pd.read_csv(path)[col]
    valid = raw.dropna().values
    reps  = (n // len(valid)) + 1
    return np.tile(valid, reps)[:n].tolist()


def _load_profiles() -> tuple[list[float], list[float]]:
    data_dir  = REPO_ROOT / "configs" / "power_profiles" / "energinet_wind_price_data"
    p_avail = _load_series(data_dir / "wind_power.csv",  "P_W")
    price   = _load_series(data_dir / "spot_price.csv",  "price_eur_per_kWh")
    return p_avail, price


# ── KPI extractor ────────────────────────────────────────────────────────────

def kpis(df: pd.DataFrame) -> dict:
    active = df["j_A_per_m2"] > 100
    act    = df.loc[active]

    total_H2   = (df["m_dot_H2_kg_h"] * DT_H).sum()
    total_cost = (df["c_elec_eur_h"] + df["c_deg_phys_eur_h"] + df["c_shutdown_eur"]).sum()
    lcoh       = total_cost / max(total_H2, 1e-12)

    V_deg_final = df["V_deg_V"].iloc[-1]
    lifetime_yr = V_DEG_EOL / max(V_deg_final, 1e-12)

    energy_kWh  = (act["P_total_W"] * DT_H / 1000.0).sum()
    H2_active   = (act["m_dot_H2_kg_h"] * DT_H).sum()
    sec_kWh_kg  = energy_kWh / max(H2_active, 1e-12)

    n_shutdowns = int((df["c_shutdown_eur"] > 0).sum())

    return dict(H2_kg=total_H2, lcoh_eur_kg=lcoh, lifetime_yr=lifetime_yr,
                sec_kWh_kg=sec_kWh_kg, n_shutdowns=n_shutdowns)


# ── Single run ───────────────────────────────────────────────────────────────

def run_one(ctrl_name: str, c_shutdown: int) -> tuple[str, int, pd.DataFrame]:
    p_avail, price = _load_profiles()
    plant    = load_plant(PLANT_CFG)
    ctrl_cfg = load_config(CTRL_CFGS[ctrl_name])

    # Override shutdown cost and enable voluntary shutdown in objective
    plant["economics"]["c_shutdown_eur"] = float(c_shutdown)
    ctrl_cfg["objective"]["include_shutdown"] = True

    df = run_simulation(
        price_series=price,
        p_avail_series=p_avail,
        plant=plant,
        ctrl_cfg=ctrl_cfg,
        dt_s=3600.0,
    )
    return ctrl_name, c_shutdown, df


def _worker(args):
    ctrl_name, c_shutdown = args
    return run_one(ctrl_name, c_shutdown)


def csv_path(ctrl_name: str, c_shutdown: int) -> Path:
    return OUT_DIR / f"{ctrl_name}_sd{c_shutdown}EUR.csv"


def load_or_run(ctrl_name: str, c_shutdown: int, skip_existing: bool = False) -> pd.DataFrame:
    p = csv_path(ctrl_name, c_shutdown)
    if skip_existing and p.exists():
        return pd.read_csv(p)
    t0 = time.perf_counter()
    _, _, df = run_one(ctrl_name, c_shutdown)
    t_wall = time.perf_counter() - t0
    df["t_wall_s"] = t_wall
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"  [{ctrl_name} sd={c_shutdown}] done in {t_wall:.0f}s → {p.name}")
    return df


# ── Figure ───────────────────────────────────────────────────────────────────

METRICS = [
    ("n_shutdowns",  "Shutdown events",   "",           "{:.0f}"),
    ("H2_kg",        r"H$_2$ produced",   "kg",         "{:.0f}"),
    ("lcoh_eur_kg",  "LCOH",              r"\euro/kg",  "{:.3f}"),
    ("lifetime_yr",  "Proj. lifetime",    "yr",         "{:.1f}"),
]


def make_figure(results: dict, outdir: Path) -> None:
    """
    results: {ctrl_name: {"baseline": kpi_dict, 50: kpi_dict, 100: ..., 150: ...}}
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    x_labels = ["Baseline"] + [f"{c}\,\euro" for c in SHUTDOWN_COSTS]
    x = np.arange(len(x_labels))
    n_ctrl = len(CTRL_CFGS)
    width  = 0.22
    offsets = np.linspace(-(n_ctrl - 1) / 2, (n_ctrl - 1) / 2, n_ctrl) * width

    for ax, (key, label, unit, fmt) in zip(axes, METRICS):
        for i, (ctrl, color) in enumerate(CTRL_COLOR.items()):
            vals = [results[ctrl]["baseline"][key]]
            for c in SHUTDOWN_COSTS:
                vals.append(results[ctrl][c][key])
            ax.bar(x + offsets[i], vals, width=width, color=color,
                   label=CTRL_LABEL[ctrl], alpha=0.85, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ylabel = f"{label}" + (f" [{unit}]" if unit else "")
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)

    axes[0].legend(loc="upper right")
    fig.suptitle(r"Shutdown cost in objective --- sensitivity by controller and $c_\mathrm{sd}$",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = outdir / "fig_shutdown_sensitivity.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"  Saved: {out.name}")
    plt.close(fig)


# ── LaTeX table ──────────────────────────────────────────────────────────────

def make_table(results: dict, outdir: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Shutdown cost sensitivity: KPIs across controllers and $c_{\rm sd}$ levels.}",
        r"\label{tab:shutdown_sensitivity}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Controller & $c_{\rm sd}$ [\euro] & Shutdowns & H$_2$ [kg] & LCOH [\euro/kg] & Lifetime [yr] & SEC [kWh/kg] \\",
        r"\midrule",
    ]
    for ctrl, label in CTRL_LABEL.items():
        for j, case in enumerate(["baseline"] + SHUTDOWN_COSTS):
            k = results[ctrl][case]
            case_str = "No vol.\ stop" if case == "baseline" else str(case)
            ctrl_str = label if j == 0 else ""
            lines.append(
                rf"{ctrl_str} & {case_str} & {k['n_shutdowns']:.0f} & {k['H2_kg']:.0f} "
                rf"& {k['lcoh_eur_kg']:.3f} & {k['lifetime_yr']:.1f} & {k['sec_kWh_kg']:.2f} \\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]

    out = outdir / "table_shutdown_sensitivity.tex"
    out.write_text("\n".join(lines))
    print(f"  Saved: {out.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Shutdown cost in objective sensitivity")
    parser.add_argument("--parallel",    action="store_true", help="Run cases in parallel")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already computed CSVs")
    parser.add_argument("--plots-only",  action="store_true", help="Skip simulation, load CSVs")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = [(ctrl, c) for ctrl in CTRL_CFGS for c in SHUTDOWN_COSTS]

    # ── Run simulations ───────────────────────────────────────────────────────
    if not args.plots_only:
        if args.parallel:
            print(f"Running {len(cases)} cases in parallel...")
            with ProcessPoolExecutor() as ex:
                futs = {ex.submit(_worker, (ctrl, c)): (ctrl, c) for ctrl, c in cases}
                for fut in as_completed(futs):
                    ctrl, c = futs[fut]
                    try:
                        ctrl_name, c_shutdown, df = fut.result()
                        p = csv_path(ctrl_name, c_shutdown)
                        df.to_csv(p, index=False)
                        print(f"  [{ctrl_name} sd={c_shutdown}] saved → {p.name}")
                    except Exception as e:
                        print(f"  [{ctrl} sd={c}] FAILED: {e}")
        else:
            skip = args.skip_existing
            for ctrl, c in cases:
                load_or_run(ctrl, c, skip_existing=skip)

    # ── Load results and compute KPIs ─────────────────────────────────────────
    print("\nComputing KPIs...")
    results: dict = {}
    for ctrl in CTRL_CFGS:
        results[ctrl] = {}

        # Baseline: existing 1yr run (no voluntary shutdown)
        b_csv = BASELINE_CSVS[ctrl]
        if not b_csv.exists():
            print(f"  [warning] baseline CSV missing for {ctrl}: {b_csv}")
            results[ctrl]["baseline"] = {k: float("nan") for k in
                                         ["n_shutdowns","H2_kg","lcoh_eur_kg","lifetime_yr","sec_kWh_kg"]}
        else:
            results[ctrl]["baseline"] = kpis(pd.read_csv(b_csv))

        # Sensitivity cases
        for c in SHUTDOWN_COSTS:
            p = csv_path(ctrl, c)
            if not p.exists():
                print(f"  [warning] CSV missing: {p.name}")
                results[ctrl][c] = {k: float("nan") for k in results[ctrl]["baseline"]}
            else:
                results[ctrl][c] = kpis(pd.read_csv(p))

        # Print summary
        b = results[ctrl]["baseline"]
        print(f"\n  {CTRL_LABEL[ctrl]}:")
        print(f"    {'Case':<18} {'Stops':>6} {'H2 [kg]':>9} {'LCOH':>8} {'Life [yr]':>10} {'SEC':>8}")
        for case in ["baseline"] + SHUTDOWN_COSTS:
            k = results[ctrl][case]
            tag = "Baseline" if case == "baseline" else f"sd={case} EUR"
            print(f"    {tag:<18} {k['n_shutdowns']:>6.0f} {k['H2_kg']:>9.0f} "
                  f"{k['lcoh_eur_kg']:>8.4f} {k['lifetime_yr']:>10.2f} {k['sec_kWh_kg']:>8.2f}")

    # ── Figures and table ─────────────────────────────────────────────────────
    make_figure(results, OUT_DIR)
    make_table(results, OUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
