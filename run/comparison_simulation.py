"""
comparison_simulation.py — Run all four controllers and generate comparison plots.

Each controller gets its own results folder (CSV + individual figures).
Comparison figures go into a separate comparison folder.

Output structure (auto-derived from config names):
  results/
    load_following_wind_spot_dk1/     <- CSV + fig1-fig6
    price_aware_wind_spot_dk1/   <- CSV + fig1-fig6
    degradation_aware_wind_spot_dk1/          <- CSV + fig1-fig6
    lifetime_aware_wind_spot_dk1/      <- CSV + fig1-fig6
    comparison_wind_spot_dk1/     <- fig7-fig10

Usage (via Makefile):
  python3 run/comparison_simulation.py \
      --plant configs/plant_parameters.yaml \
      --power-profile configs/power_profiles/wind.yaml \
      --price-profile configs/price_profiles/spot_dk1.yaml \
      --load-following-controller configs/controllers/load_following.yaml \
      --price-aware-controller configs/controllers/price_aware.yaml \
      --degradation-aware-controller configs/controllers/degradation_aware.yaml \
      --lifetime-aware-controller configs/controllers/lifetime_aware.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def derive_run_name(controller: Path, power_profile: Path, price_profile: Path) -> str:
    """Auto-derive a run name from config file stems."""
    return f"{controller.stem}_{power_profile.stem}_{price_profile.stem}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run three-controller comparison.")
    p.add_argument("--plant", required=True, type=Path)
    p.add_argument("--power-profile", required=True, type=Path)
    p.add_argument("--price-profile", required=True, type=Path)
    p.add_argument("--load-following-controller",    required=True, type=Path)
    p.add_argument("--price-aware-controller",       required=True, type=Path)
    p.add_argument("--degradation-aware-controller", required=True, type=Path)
    p.add_argument("--lifetime-aware-controller",    required=True, type=Path)
    return p.parse_args()


def run_case(plant, controller, power_profile, price_profile, name, outdir, detail_window=None):
    """Run a single simulation case via single_simulation.py, return the CSV path."""
    csv_path = outdir / f"{name}.csv"
    if csv_path.exists():
        print(f"  [skip] {csv_path} already exists")
        return csv_path

    cmd = [
        sys.executable, str(REPO_ROOT / "run" / "single_simulation.py"),
        "--plant", str(plant),
        "--controller", str(controller),
        "--power-profile", str(power_profile),
        "--price-profile", str(price_profile),
        "--name", name,
        "--outdir", str(outdir),
    ]
    if detail_window:
        cmd += ["--detail-window", detail_window]
    print(f"  Running: {name} ...")
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"  [ERROR] {name} failed:\n{ret.stderr[:500]}")
        sys.exit(1)
    print(ret.stdout.strip())
    return csv_path


def main():
    args = parse_args()

    # --- Auto-derive names ---
    lf_name = derive_run_name(
        args.load_following_controller, args.power_profile, args.price_profile)
    pa_name = derive_run_name(
        args.price_aware_controller, args.power_profile, args.price_profile)
    da_name = derive_run_name(
        args.degradation_aware_controller, args.power_profile, args.price_profile)
    la_name = derive_run_name(
        args.lifetime_aware_controller, args.power_profile, args.price_profile)
    comparison_name = f"comparison_{args.power_profile.stem}_{args.price_profile.stem}"

    results_dir = REPO_ROOT / "results"
    lf_dir  = results_dir / lf_name
    pa_dir  = results_dir / pa_name
    da_dir  = results_dir / da_name
    la_dir  = results_dir / la_name
    comparison_dir = results_dir / comparison_name

    for d in [lf_dir, pa_dir, da_dir, la_dir, comparison_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"=== Comparison: {comparison_name} ===")
    print(f"  Load-following      -> {lf_dir}")
    print(f"  Price-aware         -> {pa_dir}")
    print(f"  Degradation-aware   -> {da_dir}")
    print(f"  Lifetime-aware      -> {la_dir}")
    print(f"  Comparison          -> {comparison_dir}")

    # --- 1. Run (or skip) all four simulations ---
    # Run load-following first (fast), then find a 10h detail window for sub-step logging
    lf_csv = run_case(
        args.plant, args.load_following_controller,
        args.power_profile, args.price_profile,
        lf_name, lf_dir,
    )

    # Find detail window from load-following results (startup + high T)
    detail_window = None
    if lf_csv.exists():
        import pandas as pd, numpy as np
        df_c = pd.read_csv(lf_csv)
        j_arr = df_c["j_A_per_m2"].values
        starts = [i for i in range(1, len(j_arr)) if j_arr[i-1] <= 100 and j_arr[i] > 100]
        T_max_col = df_c["T_stack_max_K"].values if "T_stack_max_K" in df_c.columns else None
        for si in starts:
            end = min(si + 10, len(j_arr))
            if T_max_col is not None and np.any(T_max_col[si:end] > 352.15):
                t_start = max(1, si - 2)
                detail_window = f"{t_start},{t_start + 10}"
                print(f"  Detail window (from load-following): hours {t_start}-{t_start+10}")
                break
        if detail_window is None and starts:
            si = starts[len(starts)//2]
            t_start = max(1, si - 2)
            detail_window = f"{t_start},{t_start + 10}"
            print(f"  Detail window (fallback): hours {t_start}-{t_start+10}")

    # Re-run load-following with detail window (if not already done with sub-steps)
    substep_csv = lf_dir / f"{lf_name}_substeps.csv"
    if detail_window and not substep_csv.exists():
        lf_csv.unlink(missing_ok=True)
        lf_csv = run_case(
            args.plant, args.load_following_controller,
            args.power_profile, args.price_profile,
            lf_name, lf_dir, detail_window=detail_window,
        )

    # Run price-aware, degradation-aware, and lifetime-aware with same detail window
    pa_csv = run_case(
        args.plant, args.price_aware_controller,
        args.power_profile, args.price_profile,
        pa_name, pa_dir, detail_window=detail_window,
    )

    da_csv = run_case(
        args.plant, args.degradation_aware_controller,
        args.power_profile, args.price_profile,
        da_name, da_dir, detail_window=detail_window,
    )

    la_csv = run_case(
        args.plant, args.lifetime_aware_controller,
        args.power_profile, args.price_profile,
        la_name, la_dir, detail_window=detail_window,
    )

    # --- 2. Generate per-controller plots ---
    for csv_path, name, outdir in [
        (lf_csv, lf_name, lf_dir),
        (pa_csv, pa_name, pa_dir),
        (da_csv, da_name, da_dir),
        (la_csv, la_name, la_dir),
    ]:
        print(f"\n  Generating figures for {name} -> {outdir} ...")
        cmd = [
            sys.executable, str(REPO_ROOT / "run" / "make_plots.py"),
            "--input", str(csv_path),
            "--name", name,
            "--outdir", str(outdir),
        ]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        if ret.returncode == 0:
            print(ret.stdout.strip())
        else:
            print(f"  [plot warning] {ret.stderr.strip()[:500]}")

    # --- 3. Generate comparison plots ---
    print(f"\n  Generating comparison figures -> {comparison_dir} ...")
    cmd = [
        sys.executable, str(REPO_ROOT / "run" / "make_plots.py"),
        "--load-following",    str(lf_csv),
        "--price-aware",       str(pa_csv),
        "--degradation-aware", str(da_csv),
        "--lifetime-aware",    str(la_csv),
        "--name", comparison_name,
        "--outdir", str(comparison_dir),
    ]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode == 0:
        print(ret.stdout.strip())
    else:
        print(f"  [plot warning] {ret.stderr.strip()[:500]}")

    print(f"\nDone: {comparison_name}")


if __name__ == "__main__":
    main()
