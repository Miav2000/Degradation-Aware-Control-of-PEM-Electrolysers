"""
monte_carlo.py  --  Monte Carlo uncertainty analysis for PEMWE techno-economic model.

Runs N Monte Carlo realizations of the 5-year simulation for a specified controller,
varying five uncertain parameters simultaneously. Results are written to CSV immediately
as each simulation completes, with automatic resume on restart.

Uncertain parameters (all Uniform):
    elec_multiplier   Uniform(0.80, 1.20)   -- scales hourly spot-price series
    capex_usd_per_kW  Uniform(100, 450)     -- stack replacement cost [USD/kW]
    wacc              Uniform(0.06, 0.10)   -- post-hoc discount rate only
    p_h2_eur_per_kg   Uniform(4, 8)         -- hydrogen selling price [EUR/kg]
    ddr_ref_uV_per_h  Uniform(5, 15)        -- reference degradation rate [uV/h]

WACC is post-hoc: it does not enter the simulation dispatch objective, only the
NPV-based LCOH calculation performed after each simulation completes.

The Monte Carlo sample file (mc_samples.csv) is generated once and reused by all
controllers, ensuring identical uncertainty realisations for a fair comparison.

Modes of operation:
  1. Run simulations for a single controller:
        python3 run/sensitivity/monte_carlo.py --controller DA --samples 1000 --workers 60

  2. Regenerate figures from completed results.csv (no simulations):
        python3 run/sensitivity/monte_carlo.py --controller DA --plots-only

  3. Cross-controller comparison (after running at least 2 controllers):
        python3 run/sensitivity/monte_carlo.py --compare
        python3 run/sensitivity/monte_carlo.py --compare --controllers DA LF

Nohup example (recommended for DA/LA long runs):
    mkdir -p logs
    nohup python3 run/sensitivity/monte_carlo.py \\
        --controller DA --samples 1000 --workers 60 --seed 42 \\
        > logs/monte_carlo_DA.log 2>&1 &

Estimated wall-clock runtimes at 60 workers, 1000 samples:
    LF  (load-following):     ~2-4 h   (fast: no optimiser)
    PA  (price-aware):        ~8-15 h
    DA  (degradation-aware):  ~60-80 h
    LA  (lifetime-aware):     ~60-80 h
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "run"))

# ---- Paths -------------------------------------------------------------------
PLANT_CFG   = REPO_ROOT / "configs" / "plant_parameters.yaml"
POWER_CFG   = REPO_ROOT / "configs" / "power_profiles" / "wind.yaml"
PRICE_CFG   = REPO_ROOT / "configs" / "price_profiles" / "spot_dk1.yaml"
MC_OUTDIR   = REPO_ROOT / "results" / "monte_carlo"
SAMPLES_CSV = MC_OUTDIR / "samples" / "mc_samples.csv"

CONTROLLERS = {
    "LF": REPO_ROOT / "configs" / "controllers" / "load_following.yaml",
    "PA": REPO_ROOT / "configs" / "controllers" / "price_aware.yaml",
    "DA": REPO_ROOT / "configs" / "controllers" / "degradation_aware.yaml",
    "LA": REPO_ROOT / "configs" / "controllers" / "lifetime_aware.yaml",
}

CTRL_LABELS = {
    "DA": "Degradation-aware",
    "LF": "Load-following",
    "PA": "Price-aware",
    "LA": "Lifetime-aware",
}

# ---- 5-year simulation constants ---------------------------------------------
N_YEARS        = 5
HOURS_PER_YEAR = 8760
N_STEPS        = N_YEARS * HOURS_PER_YEAR
DT_S           = 3600.0

# ---- Uncertain parameter ranges ---------------------------------------------
PARAM_RANGES = {
    "elec_multiplier":  (0.80, 1.20),
    "capex_usd_per_kW": (100.0, 1231.03),
    "wacc":             (0.06, 0.10),
    "p_h2_eur_per_kg":  (4.0, 8.0),
    "ddr_ref_uV_per_h": (5.0, 15.0),
}

PARAM_LABELS = {
    "elec_multiplier":  r"$p_{\mathrm{elec}}$ scale",
    "capex_usd_per_kW": r"CAPEX [USD/kW]",
    "wacc":             r"WACC",
    "p_h2_eur_per_kg":  r"$p_{\mathrm{H_2}}$ [\euro/kg]",
    "ddr_ref_uV_per_h": r"DDR [\textmu V/h]",
}

INPUT_COLS = list(PARAM_RANGES.keys())

# j threshold [A/m2] -- above this the stack is considered active
ACTIVE_J_THRESHOLD = 100.0

PROGRESS_INTERVAL = 25

RESULT_COLS = [
    "sample_id",
    "elec_multiplier", "capex_usd_per_kW", "wacc",
    "p_h2_eur_per_kg", "ddr_ref_uV_per_h",
    "avg_j_A_per_m2", "avg_T_stack_C", "SEC_kWh_per_kg",
    "h2_annual_kg", "lcoh_eur_per_kg", "lifetime_yr",
    "n_replacements", "h2_lifetime_kg",
    "status", "error",
]


# ---- Logging -----------------------------------------------------------------
def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---- Sample generation / loading --------------------------------------------
def load_or_generate_samples(n: int, seed: int) -> pd.DataFrame:
    """
    Return the MC sample DataFrame.
    Generates and saves mc_samples.csv if it does not already exist;
    otherwise loads and returns the existing file.
    """
    if SAMPLES_CSV.exists():
        df = pd.read_csv(SAMPLES_CSV)
        _log(f"Loaded existing samples: {SAMPLES_CSV}  ({len(df)} rows)")
        if len(df) < n:
            _log(f"WARNING: existing sample file has {len(df)} rows but --samples={n}. "
                 f"Using {len(df)} samples.")
        return df

    _log(f"Generating {n} Monte Carlo samples (seed={seed}) ...")
    SAMPLES_CSV.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data: dict = {"sample_id": np.arange(n)}
    for name, (lo, hi) in PARAM_RANGES.items():
        data[name] = rng.uniform(lo, hi, n)
    df = pd.DataFrame(data)
    df.to_csv(SAMPLES_CSV, index=False)
    _log(f"Saved: {SAMPLES_CSV}")
    return df


# ---- Worker (executes in child process) -------------------------------------
def _worker(case: dict) -> dict:
    """
    Execute one Monte Carlo realization in a child process.
    All imports are local so the function is safely picklable.
    """
    import os as _os
    import sys as _sys
    from pathlib import Path as _Path

    # Prevent numpy/OpenBLAS from spawning one thread per core inside each worker.
    # Without this, 60 workers × 64 threads = 3840 threads on 64 cores (severe thrashing).
    _os.environ["OMP_NUM_THREADS"]      = "1"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"]      = "1"
    _os.environ["NUMEXPR_NUM_THREADS"]  = "1"

    import numpy as _np

    _sys.path.insert(0, case["repo_root"])
    _sys.path.insert(0, str(_Path(case["repo_root"]) / "run"))

    from pemwe.plant import load_plant, load_config, derive_degradation_coeffs
    from pemwe.simulation import run_simulation
    from single_simulation import load_power_profile, load_price_profile

    sid = int(case["sample_id"])

    try:
        # ---- Load plant and apply sample parameters --------------------------
        plant = load_plant(_Path(case["plant_cfg"]))

        plant["economics"]["capex_usd_per_kW"] = float(case["capex_usd_per_kW"])
        plant["economics"]["p_H2_eur_per_kg"]  = float(case["p_h2_eur_per_kg"])
        # WACC is post-hoc -- not applied to plant dict
        plant["degradation"]["DDR_ref_uV_per_h"] = float(case["ddr_ref_uV_per_h"])
        derive_degradation_coeffs(plant)

        # ---- Load 5-year tiled profiles --------------------------------------
        # Use the standard profile loaders (same as tornado / single_simulation)
        # for 1-year data, then tile to 5 years (same approach as lifetime_5yr.py).
        repo = _Path(case["repo_root"])

        power_cfg  = load_config(_Path(case["power_cfg"]))
        price_cfg  = load_config(_Path(case["price_cfg"]))
        sim_cfg    = power_cfg["simulation"]
        dt_s_cfg   = float(sim_cfg["dt_s"])
        n_steps_yr = int(round(float(sim_cfg["t_end_s"]) / dt_s_cfg))

        p1yr  = _np.array(load_power_profile(power_cfg, n_steps_yr, dt_s_cfg, repo))
        pr1yr = _np.array(load_price_profile(price_cfg, n_steps_yr, repo))

        n_steps = case["n_steps"]
        p_avail = _np.tile(p1yr,  (n_steps // n_steps_yr) + 1)[:n_steps]
        prices  = _np.tile(pr1yr, (n_steps // n_steps_yr) + 1)[:n_steps]

        prices = prices * float(case["elec_multiplier"])

        # ---- Run 5-year simulation -------------------------------------------
        ctrl_cfg = load_config(_Path(case["ctrl_cfg"]))
        df = run_simulation(
            price_series=prices.tolist(),
            p_avail_series=p_avail.tolist(),
            plant=plant,
            ctrl_cfg=ctrl_cfg,
            dt_s=float(case["dt_s"]),
        )

        # ---- Derive metrics --------------------------------------------------
        dt_h    = float(case["dt_s"]) / 3600.0
        n_steps = len(df)

        j_arr  = _np.asarray(df["j_A_per_m2"],       dtype=float)
        T_arr  = _np.asarray(df["T_stack_actual_K"],  dtype=float)
        P_arr  = _np.asarray(df["P_total_W"],         dtype=float)
        H2_arr = _np.asarray(df["m_dot_H2_kg_h"],     dtype=float)
        ce_arr = _np.asarray(df["c_elec_eur_h"],      dtype=float)
        cs_arr = _np.asarray(df["c_shutdown_eur"],    dtype=float)
        dv_arr = _np.asarray(df["dV_deg_V"],          dtype=float)

        active = j_arr > ACTIVE_J_THRESHOLD

        avg_j = float(j_arr[active].mean())     if active.any() else 0.0
        avg_T = float((T_arr[active] - 273.15).mean()) if active.any() else 0.0

        h2_act_kg    = float((H2_arr[active] * dt_h).sum())
        elec_act_kWh = float((P_arr[active] / 1000.0 * dt_h).sum())
        SEC          = elec_act_kWh / h2_act_kg if h2_act_kg > 0 else _np.nan

        total_H2_kg = float((H2_arr * dt_h).sum())
        total_elec  = float((ce_arr  * dt_h).sum())
        total_sd    = float(cs_arr.sum())

        sim_yr       = n_steps * dt_h / 8760.0
        h2_annual_kg = total_H2_kg / max(sim_yr, 1e-9)

        V_EOL       = float(case["V_EOL"])
        deg_rate_Vh = dv_arr.sum() / (n_steps * dt_h)
        lifetime_yr = (V_EOL / deg_rate_Vh / 8760.0) if deg_rate_Vh > 0 else _np.inf

        # ---- LCOH (post-hoc WACC) -------------------------------------------
        wacc      = float(case["wacc"])
        sys_lt    = float(case["sys_lifetime_yr"])
        annuity   = (1.0 - (1.0 + wacc) ** -sys_lt) / wacc if wacc > 0 else sys_lt

        eur_per_usd = float(case["eur_per_usd"])
        P_rated_kW  = float(case["P_rated_kW"])
        stack_repl  = float(case["capex_usd_per_kW"]) * eur_per_usd * P_rated_kW
        sys_capex   = float(case["sys_capex_eur"])

        if _np.isfinite(lifetime_yr) and lifetime_yr > 0:
            repl_t  = _np.arange(lifetime_yr, sys_lt + lifetime_yr, lifetime_yr)
            repl_t  = repl_t[repl_t <= sys_lt]
            npv_rep = float(sum(stack_repl / (1.0 + wacc) ** t for t in repl_t))
            n_rep   = int(len(repl_t))
        else:
            npv_rep = 0.0
            n_rep   = 0

        lcoh_rep = npv_rep / (h2_annual_kg * annuity) if h2_annual_kg > 0 else 0.0
        lcoh_sys = sys_capex / (h2_annual_kg * annuity) if h2_annual_kg > 0 else 0.0
        lcoh = (
            (total_elec + total_sd) / total_H2_kg + lcoh_rep + lcoh_sys
            if total_H2_kg > 0 else _np.nan
        )

        h2_lifetime_kg = h2_annual_kg * sys_lt

        return {
            "sample_id":         sid,
            "elec_multiplier":   case["elec_multiplier"],
            "capex_usd_per_kW":  case["capex_usd_per_kW"],
            "wacc":              case["wacc"],
            "p_h2_eur_per_kg":   case["p_h2_eur_per_kg"],
            "ddr_ref_uV_per_h":  case["ddr_ref_uV_per_h"],
            "avg_j_A_per_m2":    avg_j,
            "avg_T_stack_C":     avg_T,
            "SEC_kWh_per_kg":    SEC,
            "h2_annual_kg":      h2_annual_kg,
            "lcoh_eur_per_kg":   lcoh,
            "lifetime_yr":       lifetime_yr,
            "n_replacements":    n_rep,
            "h2_lifetime_kg":    h2_lifetime_kg,
            "status":            "success",
            "error":             "",
        }

    except Exception as exc:
        return {
            "sample_id":         sid,
            "elec_multiplier":   case.get("elec_multiplier",  float("nan")),
            "capex_usd_per_kW":  case.get("capex_usd_per_kW", float("nan")),
            "wacc":              case.get("wacc",              float("nan")),
            "p_h2_eur_per_kg":   case.get("p_h2_eur_per_kg",  float("nan")),
            "ddr_ref_uV_per_h":  case.get("ddr_ref_uV_per_h", float("nan")),
            "avg_j_A_per_m2":    float("nan"),
            "avg_T_stack_C":     float("nan"),
            "SEC_kWh_per_kg":    float("nan"),
            "h2_annual_kg":      float("nan"),
            "lcoh_eur_per_kg":   float("nan"),
            "lifetime_yr":       float("nan"),
            "n_replacements":    float("nan"),
            "h2_lifetime_kg":    float("nan"),
            "status":            "failed",
            "error":             f"{type(exc).__name__}: {exc}",
        }


# ---- Case builder -----------------------------------------------------------
def _build_cases(samples: pd.DataFrame, ctrl_key: str,
                 plant_consts: dict) -> list[dict]:
    cases = []
    for _, row in samples.iterrows():
        cases.append({
            "repo_root":        str(REPO_ROOT),
            "plant_cfg":        str(PLANT_CFG),
            "power_cfg":        str(POWER_CFG),
            "price_cfg":        str(PRICE_CFG),
            "ctrl_cfg":         str(CONTROLLERS[ctrl_key]),
            "n_steps":          N_STEPS,
            "dt_s":             DT_S,
            "sample_id":        int(row["sample_id"]),
            "elec_multiplier":  float(row["elec_multiplier"]),
            "capex_usd_per_kW": float(row["capex_usd_per_kW"]),
            "wacc":             float(row["wacc"]),
            "p_h2_eur_per_kg":  float(row["p_h2_eur_per_kg"]),
            "ddr_ref_uV_per_h": float(row["ddr_ref_uV_per_h"]),
            **plant_consts,
        })
    return cases


# ---- Figure generation (per-controller) -------------------------------------
def make_figures(results_csv: Path, ctrl_key: str, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from plot_style import apply_style, BLUE, GREEN, ORANGE, GREY

    apply_style()
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage{eurosym}"
    )

    df = pd.read_csv(results_csv)
    df = df[df["status"] == "success"].copy()
    n  = len(df)
    _log(f"  Plotting {n} successful realizations ...")

    def _hist_cdf(ax_h, ax_c, data, color, xlabel, fmt=".2f"):
        data = data.dropna().sort_values()
        ax_h.hist(data, bins=40, color=color, edgecolor="white",
                  linewidth=0.4, alpha=0.85)
        ax_h.set_xlabel(xlabel)
        ax_h.set_ylabel("Count")
        mean_val = data.mean()
        ax_h.axvline(mean_val, color=GREY, lw=1.5, ls="--",
                     label=rf"Mean = {mean_val:{fmt}}")
        ax_h.axvline(data.quantile(0.05), color=GREY, lw=1.0, ls=":",
                     label=r"5th--95th pct")
        ax_h.axvline(data.quantile(0.95), color=GREY, lw=1.0, ls=":")
        ax_h.legend(fontsize=9)
        cdf_y = np.arange(1, len(data) + 1) / len(data)
        ax_c.plot(data.values, cdf_y, color=color, lw=1.8)
        ax_c.axvline(mean_val, color=GREY, lw=1.2, ls="--")
        ax_c.set_xlabel(xlabel)
        ax_c.set_ylabel("CDF")
        ax_c.set_ylim(0, 1)
        ax_c.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    metrics = [
        ("lcoh_eur_per_kg", BLUE,   r"LCOH [\euro/kg$_{\mathrm{H_2}}$]",
         "lcoh_histogram", ".2f"),
        ("lifetime_yr",     GREEN,  r"Projected lifetime [yr]",
         "lifetime_histogram", ".1f"),
        ("h2_annual_kg",    ORANGE, r"Annual H$_2$ production [kg/yr]",
         "hydrogen_histogram", ".0f"),
    ]

    for col, color, xlabel, fname, fmt in metrics:
        if col not in df.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        _hist_cdf(axes[0], axes[1], df[col], color, xlabel, fmt)
        ctrl_label = CTRL_LABELS.get(ctrl_key, ctrl_key)
        fig.suptitle(
            rf"\textbf{{{ctrl_label}}} -- {xlabel.split('[')[0].strip()} "
            rf"({n}\,realizations)",
            fontsize=11,
        )
        fig.tight_layout()
        for ext in (".pdf", ".png"):
            fig.savefig(out_dir / (fname + ext), bbox_inches="tight", dpi=150)
        plt.close(fig)
        _log(f"  Saved: {fname}.pdf/.png")

    # Spearman correlations
    try:
        from scipy.stats import spearmanr
    except ImportError:
        _log("  scipy not available -- skipping Spearman correlations")
        return

    output_cols = ["lcoh_eur_per_kg", "lifetime_yr"]
    corr_rows = []
    for inp in INPUT_COLS:
        row = {"parameter": inp}
        for out in output_cols:
            valid = df[[inp, out]].dropna()
            if len(valid) >= 10:
                rho, pval = spearmanr(valid[inp], valid[out])
                row[f"rho_{out}"]  = round(rho,  4)
                row[f"pval_{out}"] = round(pval, 4)
            else:
                row[f"rho_{out}"]  = float("nan")
                row[f"pval_{out}"] = float("nan")
        corr_rows.append(row)
    pd.DataFrame(corr_rows).to_csv(out_dir / "spearman_correlations.csv", index=False)
    _log(f"  Saved: spearman_correlations.csv")

    # Summary statistics
    stat_cols = ["lcoh_eur_per_kg", "lifetime_yr", "h2_annual_kg", "h2_lifetime_kg"]
    rows = []
    for c in stat_cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        rows.append({
            "metric": c, "mean": s.mean(), "median": s.median(),
            "std": s.std(), "min": s.min(), "max": s.max(),
            "p05": s.quantile(0.05), "p95": s.quantile(0.95), "n": len(s),
        })
    pd.DataFrame(rows).to_csv(out_dir / "summary_statistics.csv", index=False)
    _log(f"  Saved: summary_statistics.csv")

    # ---- Paper histograms: individual + combined side-by-side ---------------
    def _single_hist(ax, data, xlabel, fmt_mean, fmt_pct):
        data   = data.dropna().values
        mean   = float(np.mean(data))
        median = float(np.median(data))
        p05    = float(np.percentile(data, 5))
        p95    = float(np.percentile(data, 95))
        ax.hist(data, bins=30, color=BLUE, alpha=0.85,
                edgecolor="white", linewidth=0.4)
        ax.axvline(mean,   color="black",  lw=1.6, linestyle="--",
                   label=rf"Mean = {mean:{fmt_mean}}")
        ax.axvline(median, color=GREEN,    lw=1.6, linestyle="-.",
                   label=rf"Median = {median:{fmt_mean}}")
        ax.axvline(p05,    color=ORANGE,   lw=1.3, linestyle=":",
                   label=rf"5th--95th: {p05:{fmt_pct}}--{p95:{fmt_pct}}")
        ax.axvline(p95,    color=ORANGE,   lw=1.3, linestyle=":")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    paper_metrics = [
        ("lcoh_eur_per_kg", r"LCOH [\euro/kg$_{\mathrm{H_2}}$]",
         "hist_lcoh", ".2f", ".2f"),
        ("lifetime_yr", r"Projected stack lifetime [yr]",
         "hist_lifetime", ".1f", ".1f"),
    ]

    datas_for_combined = []

    for col, xlabel, fname, fmt_mean, fmt_pct in paper_metrics:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        datas_for_combined.append((data, xlabel, fmt_mean, fmt_pct))

        fig_s, ax_s = plt.subplots(figsize=(5.5, 4.0))
        _single_hist(ax_s, data, xlabel, fmt_mean, fmt_pct)
        fig_s.tight_layout()
        for ext in (".pdf", ".png"):
            fig_s.savefig(out_dir / (fname + ext), bbox_inches="tight", dpi=300)
        plt.close(fig_s)
        _log(f"  Saved: {fname}.pdf/.png")

    if len(datas_for_combined) == 2:
        import matplotlib.patches as mpatches
        fig_c, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))
        for ax, (data, xlabel, fmt_mean, fmt_pct) in zip(
                (ax1, ax2), datas_for_combined):
            _single_hist(ax, data, xlabel, fmt_mean, fmt_pct)

        fig_c.tight_layout()
        for ext in (".pdf", ".png"):
            fig_c.savefig(out_dir / ("hist_combined" + ext),
                          bbox_inches="tight", dpi=300)
        plt.close(fig_c)
        _log("  Saved: hist_combined.pdf/.png")


# ---- Cross-controller comparison --------------------------------------------
def run_comparison(ctrl_keys: list[str]) -> None:
    """
    Merge completed results.csv files across controllers on sample_id
    and produce:
      results/monte_carlo/comparison/controller_comparison.csv
      results/monte_carlo/comparison/winner_probabilities.csv
      results/monte_carlo/comparison/lcoh_violin.pdf/.png
      results/monte_carlo/comparison/lifetime_violin.pdf/.png
      results/monte_carlo/comparison/scatter_lcoh_<A>_vs_<B>.pdf/.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO_ROOT / "run"))
    from plot_style import apply_style, BLUE, GREEN, ORANGE, PURPLE, GREY

    apply_style()
    plt.rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage{eurosym}"
    )

    colors = [BLUE, GREEN, ORANGE, PURPLE]
    out_dir = MC_OUTDIR / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"=== Cross-controller comparison: {ctrl_keys} ===")
    _log(f"Output: {out_dir}")

    # Load results
    dfs: dict[str, pd.DataFrame] = {}
    for key in ctrl_keys:
        csv_path = MC_OUTDIR / key / "results.csv"
        if not csv_path.exists():
            _log(f"  WARNING: {csv_path} not found -- skipping {key}")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["status"] == "success"].copy()
        if df.empty:
            _log(f"  WARNING: no successful samples for {key} -- skipping")
            continue
        dfs[key] = df
        _log(f"  Loaded {key}: {len(df)} successful samples")

    available = [k for k in ctrl_keys if k in dfs]
    if len(available) < 2:
        _log("ERROR: need at least 2 controllers with completed results.")
        return

    # Merge on sample_id
    merged = dfs[available[0]].copy()
    merged.columns = [
        f"{c}_{available[0]}" if c not in INPUT_COLS + ["sample_id"] else c
        for c in merged.columns
    ]
    for key in available[1:]:
        right = dfs[key].copy()
        right.columns = [
            f"{c}_{key}" if c not in INPUT_COLS + ["sample_id"] else c
            for c in right.columns
        ]
        non_dup = [c for c in right.columns
                   if c == "sample_id" or c.endswith(f"_{key}")]
        merged = merged.merge(right[non_dup], on="sample_id", how="inner")
    _log(f"  Common samples: {len(merged)}")
    n = len(merged)

    # Winner columns
    lcoh_cols = [f"lcoh_eur_per_kg_{k}" for k in available]
    lt_cols   = [f"lifetime_yr_{k}"      for k in available]
    lcoh_arr  = merged[lcoh_cols].values
    lt_arr    = merged[lt_cols].values
    merged["best_lcoh_ctrl"]     = [available[i] for i in np.argmin(lcoh_arr, axis=1)]
    merged["best_lifetime_ctrl"] = [available[i] for i in np.argmax(lt_arr,   axis=1)]

    # LCOH difference columns relative to first controller
    ref = available[0]
    for k in available[1:]:
        merged[f"delta_lcoh_{ref}_vs_{k}"] = (
            merged[f"lcoh_eur_per_kg_{ref}"] - merged[f"lcoh_eur_per_kg_{k}"]
        )

    merged.to_csv(out_dir / "controller_comparison.csv", index=False)
    _log(f"  Saved: controller_comparison.csv")

    # Winner probabilities
    prob_rows = []
    for k in available:
        lc = f"lcoh_eur_per_kg_{k}"
        lt = f"lifetime_yr_{k}"
        prob_rows.append({
            "controller":           k,
            "label":                CTRL_LABELS.get(k, k),
            "P_lowest_LCOH":        round((merged["best_lcoh_ctrl"] == k).sum() / n, 4),
            "P_longest_lifetime":   round((merged["best_lifetime_ctrl"] == k).sum() / n, 4),
            "mean_LCOH_eur_per_kg":   round(merged[lc].mean(), 4),
            "median_LCOH_eur_per_kg": round(merged[lc].median(), 4),
            "p05_LCOH":               round(merged[lc].quantile(0.05), 4),
            "p95_LCOH":               round(merged[lc].quantile(0.95), 4),
            "mean_lifetime_yr":       round(merged[lt].mean(), 4),
            "median_lifetime_yr":     round(merged[lt].median(), 4),
            "p05_lifetime_yr":        round(merged[lt].quantile(0.05), 4),
            "p95_lifetime_yr":        round(merged[lt].quantile(0.95), 4),
        })
    prob_df = pd.DataFrame(prob_rows)
    prob_df.to_csv(out_dir / "winner_probabilities.csv", index=False)
    _log(f"  Saved: winner_probabilities.csv")
    _log("\nWinner probabilities:")
    _log(prob_df[["controller", "label", "P_lowest_LCOH",
                  "P_longest_lifetime", "mean_LCOH_eur_per_kg",
                  "mean_lifetime_yr"]].to_string(index=False))

    # ---- Violin: LCOH -------------------------------------------------------
    lcoh_data = [merged[f"lcoh_eur_per_kg_{k}"].dropna().values for k in available]
    labels    = [CTRL_LABELS.get(k, k) for k in available]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    parts = ax.violinplot(lcoh_data, positions=range(len(available)),
                          showmedians=True, showextrema=False, widths=0.6)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.75)
    parts["cmedians"].set_color(GREY)
    parts["cmedians"].set_linewidth(2.0)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"LCOH [\euro/kg$_{\mathrm{H_2}}$]")
    ax.set_xlabel("Controller")
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(out_dir / f"lcoh_violin{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    _log("  Saved: lcoh_violin.pdf/.png")

    # ---- Violin: lifetime ---------------------------------------------------
    lt_data = [merged[f"lifetime_yr_{k}"].dropna().values for k in available]
    fig, ax  = plt.subplots(figsize=(7.0, 4.5))
    parts = ax.violinplot(lt_data, positions=range(len(available)),
                          showmedians=True, showextrema=False, widths=0.6)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.75)
    parts["cmedians"].set_color(GREY)
    parts["cmedians"].set_linewidth(2.0)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"Projected stack lifetime [yr]")
    ax.set_xlabel("Controller")
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(out_dir / f"lifetime_violin{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    _log("  Saved: lifetime_violin.pdf/.png")

    # ---- Scatter: LCOH difference vs. uncertain parameters ------------------
    try:
        from scipy.stats import spearmanr
        has_scipy = True
    except ImportError:
        has_scipy = False

    for k in available[1:]:
        dcol  = f"delta_lcoh_{ref}_vs_{k}"
        delta = merged[dcol].dropna()
        n_p   = len(INPUT_COLS)
        fig, axes = plt.subplots(1, n_p, figsize=(3.0 * n_p, 4.0), sharey=True)
        for ax, param in zip(axes, INPUT_COLS):
            x = merged.loc[delta.index, param]
            ax.scatter(x, delta, s=5, alpha=0.3, color=BLUE)
            ax.axhline(0, color=GREY, lw=1.0, ls="--")
            ax.set_xlabel(PARAM_LABELS.get(param, param), fontsize=9)
            if has_scipy:
                rho, _ = spearmanr(x, delta)
                ax.set_title(rf"$\rho={rho:+.2f}$", fontsize=9)
        axes[0].set_ylabel(
            rf"$\Delta\mathrm{{LCOH}}$ "
            rf"({CTRL_LABELS.get(ref, ref)} $-$ {CTRL_LABELS.get(k, k)}) "
            r"[\euro/kg]",
            fontsize=9,
        )
        fig.suptitle(
            rf"{CTRL_LABELS.get(ref, ref)} vs.\ {CTRL_LABELS.get(k, k)}",
            fontsize=11,
        )
        fig.tight_layout()
        fname = f"scatter_lcoh_{ref}_vs_{k}"
        for ext in (".pdf", ".png"):
            fig.savefig(out_dir / (fname + ext), bbox_inches="tight", dpi=150)
        plt.close(fig)
        _log(f"  Saved: {fname}.pdf/.png")

    # ---- Convergence plots: running mean of LCOH and lifetime ------------------
    # Uses each controller's full results independently (not inner-joined),
    # so we see the full curve for LF and partial curves for still-running ones.
    conv_metrics = [
        ("lcoh_eur_per_kg", r"Running mean LCOH [\euro/kg$_{\mathrm{H_2}}$]",
         "convergence_lcoh"),
        ("lifetime_yr",     r"Running mean lifetime [yr]",
         "convergence_lifetime"),
    ]
    colors_map = {k: c for k, c in zip(available, colors)}

    for metric_col, ylabel, fname in conv_metrics:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for key in available:
            df_ctrl = dfs[key].copy()
            df_ctrl = df_ctrl.sort_values("sample_id").reset_index(drop=True)
            series  = df_ctrl[metric_col].dropna()
            if series.empty:
                continue
            running_mean = series.expanding().mean()
            n_samples    = np.arange(1, len(running_mean) + 1)
            ax.plot(n_samples, running_mean.values,
                    color=colors_map[key], lw=1.8,
                    label=CTRL_LABELS.get(key, key))
        ax.set_xlabel("Number of samples")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        fig.tight_layout()
        for ext in (".pdf", ".png"):
            fig.savefig(out_dir / (fname + ext), bbox_inches="tight", dpi=150)
        plt.close(fig)
        _log(f"  Saved: {fname}.pdf/.png")

    _log(f"All comparison outputs -> {out_dir}")


# ---- Main execution ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo uncertainty analysis and cross-controller comparison."
    )
    # Simulation mode arguments
    sim_group = parser.add_argument_group("Simulation mode (default)")
    sim_group.add_argument("--controller", choices=list(CONTROLLERS),
                           help="Controller to run: DA, LF, PA, or LA")
    sim_group.add_argument("--samples",  type=int,  default=1000)
    sim_group.add_argument("--workers",  type=int,  default=60)
    sim_group.add_argument("--seed",     type=int,  default=42)
    sim_group.add_argument("--plots-only", action="store_true",
                           help="Skip simulations; regenerate figures from existing results.csv")

    # Comparison mode arguments
    cmp_group = parser.add_argument_group("Comparison mode (--compare)")
    cmp_group.add_argument("--compare", action="store_true",
                           help="Cross-controller comparison from completed results")
    cmp_group.add_argument("--controllers", nargs="+",
                           choices=list(CONTROLLERS), default=list(CONTROLLERS),
                           help="Controllers to include in comparison (default: all four)")
    args = parser.parse_args()

    # ---- Comparison mode ----------------------------------------------------
    if args.compare:
        run_comparison(args.controllers)
        return

    # ---- Simulation mode ----------------------------------------------------
    if not args.controller:
        parser.error("--controller is required unless --compare is specified.")

    ctrl_key = args.controller
    out_dir  = MC_OUTDIR / ctrl_key
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"

    _log(f"=== Monte Carlo | controller={ctrl_key} | samples={args.samples} "
         f"| workers={args.workers} | seed={args.seed} ===")
    _log(f"Output directory: {out_dir}")

    if args.plots_only:
        if not results_csv.exists():
            _log(f"ERROR: {results_csv} not found. Run simulations first.")
            sys.exit(1)
        _log("Plots-only mode -- regenerating figures ...")
        make_figures(results_csv, ctrl_key, out_dir)
        return

    # Load plant constants (passed to workers as JSON-safe scalars)
    from pemwe.plant import load_plant
    base_plant = load_plant(PLANT_CFG)
    P_rated_kW = float(base_plant["stack"]["P_rating_W"]) / 1000.0
    plant_consts = {
        "V_EOL":           float(base_plant["degradation"]["V_deg_EOL_V"]),
        "sys_lifetime_yr": float(base_plant["economics"]["system_lifetime_yr"]),
        "eur_per_usd":     float(base_plant["economics"]["eur_per_usd"]),
        "P_rated_kW":      P_rated_kW,
        "sys_capex_eur":   float(base_plant["economics"]["system_capex_eur_per_kW"])
                           * P_rated_kW,
    }

    samples_df = load_or_generate_samples(args.samples, args.seed)
    samples_df = samples_df.head(args.samples)

    # Resume: find already-completed sample IDs
    done_ids: set[int] = set()
    if results_csv.exists():
        try:
            done_df  = pd.read_csv(results_csv)
            done_ids = set(done_df["sample_id"].astype(int).tolist())
            _log(f"Resuming: {len(done_ids)} samples already completed, "
                 f"{len(samples_df) - len(done_ids)} remaining.")
        except Exception as e:
            _log(f"WARNING: could not read existing results.csv ({e}). Starting fresh.")

    all_cases  = _build_cases(samples_df, ctrl_key, plant_consts)
    todo_cases = [c for c in all_cases if c["sample_id"] not in done_ids]
    n_total    = len(samples_df)
    n_done     = len(done_ids)
    n_todo     = len(todo_cases)

    if n_todo == 0:
        _log("All samples already completed. Regenerating figures ...")
        make_figures(results_csv, ctrl_key, out_dir)
        return

    _log(f"Submitting {n_todo} simulations to {args.workers} workers ...")

    write_header = not results_csv.exists() or n_done == 0
    results_fh   = open(results_csv, "a", newline="")
    writer       = csv.DictWriter(results_fh, fieldnames=RESULT_COLS,
                                  extrasaction="ignore")
    if write_header:
        writer.writeheader()
        results_fh.flush()

    t0         = time.time()
    n_this_run = 0
    n_failed   = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, c): c["sample_id"] for c in todo_cases}

        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                _log(f"  [FATAL] sample {sid} raised unhandled exception: {exc}")
                res = {col: float("nan") for col in RESULT_COLS}
                res["sample_id"] = sid
                res["status"]    = "failed"
                res["error"]     = str(exc)

            writer.writerow({k: res.get(k, "") for k in RESULT_COLS})
            results_fh.flush()

            n_this_run += 1
            n_done     += 1
            if res.get("status") != "success":
                n_failed += 1

            if n_this_run % PROGRESS_INTERVAL == 0 or n_this_run == n_todo:
                elapsed = time.time() - t0
                rate    = n_this_run / elapsed
                eta_s   = (n_todo - n_this_run) / rate if rate > 0 else float("inf")
                eta_str = str(timedelta(seconds=int(eta_s))) if eta_s < 1e6 else "inf"
                _log(
                    f"  Progress: {n_done}/{n_total} total "
                    f"({n_this_run}/{n_todo} this run) | "
                    f"elapsed {timedelta(seconds=int(elapsed))} | "
                    f"ETA {eta_str} | failed {n_failed}"
                )
                if res.get("status") == "success":
                    _log(
                        f"    sample {sid}: "
                        f"LCOH={res['lcoh_eur_per_kg']:.3f} EUR/kg  "
                        f"lifetime={res['lifetime_yr']:.2f} yr  "
                        f"H2={res['h2_annual_kg']:.0f} kg/yr"
                    )

    results_fh.close()
    elapsed_total = time.time() - t0
    _log(f"Done. {n_this_run} simulations in {timedelta(seconds=int(elapsed_total))} "
         f"({n_failed} failed).")
    _log(f"Results: {results_csv}")

    _log("Generating figures ...")
    make_figures(results_csv, ctrl_key, out_dir)
    _log(f"All outputs -> {out_dir}")


if __name__ == "__main__":
    main()
