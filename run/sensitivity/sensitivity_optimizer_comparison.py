"""
sensitivity_optimizer_comparison.py
=====================================

Benchmarks three optimization methods against a dense-grid reference
on 10 physically defined representative operating scenarios.

The test scenarios are defined entirely from physical/economic parameters —
no prior simulation results are used.  This makes the comparison controller-
neutral and reproducible from scratch.

Methods
-------
  SLSQP-1   SLSQP, single warm-start (previous operating point)
  SLSQP-3   SLSQP, 3-start  (same 3-start logic as production aware controller)
  COBYQA    Derivative-free constrained optimizer (standalone cobyqa package v1.1+)

Reference
---------
  Dense grid search (81×41 by default, after convergence check).
  Note: even an 81×41 grid may not find the exact global optimum — it is a
  dense-grid reference, not a mathematical ground truth.  It is used to
  detect gross failures (solver stuck far from the grid optimum), not to
  measure exact optimality gaps.

Grid convergence study
-----------------------
Three grids (31×21, 51×31, 81×41) are run on one scenario.
Convergence is declared when BOTH:
  - relative objective change < 0.1 %
  - |Δj| < 50 A/m²  AND  |ΔT| < 0.5 K
The converged grid size is then used as the reference for all scenarios.

All three solver methods evaluate their final solution with the same
eval_point() function (one preview call, cost computed inline) so that
objective values are directly comparable.

Metrics recorded per method × scenario
----------------------------------------
  J           Objective at solution [EUR/h]
  j*, T*      Decision variables [A/m², K]
  gap         ΔJ = J_solver − J_grid  (positive = solver worse than grid)
  rel_gap     ΔJ / |J_grid|
  delta_j     j* − j_grid  [A/m²]
  delta_T     T* − T_grid  [K]
  runtime_s   Wall-clock time [s]
  success     Solver success flag
  feasible    Power constraint satisfied at solution

Output
------
  results/sensitivity_optimizer/grid_convergence.csv
  results/sensitivity_optimizer/comparison_table.csv
  results/sensitivity_optimizer/optimizer_comparison.png
  results/sensitivity_optimizer/grid_convergence.png

Usage
-----
  python3 run/sensitivity/sensitivity_optimizer_comparison.py --run
  python3 run/sensitivity/sensitivity_optimizer_comparison.py --plots-only
  python3 run/sensitivity/sensitivity_optimizer_comparison.py --run --conv-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize as scipy_minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pemwe.plant import load_plant, load_config
from pemwe.control.objective import preview_interval_jT

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

PLANT_CFG = REPO_ROOT / "configs" / "plant_parameters.yaml"
OUT_DIR   = REPO_ROOT / "results" / "sensitivity_optimizer"
DT_S      = 3600.0

# Factor-of-2 refinement: 2× in each direction per level (4× points).
# Physical motivation: j range ~14000 A/m², T range ~25 K.
#   Coarse:   30×15  → Δj≈500 A/m², ΔT≈2 K
#   Medium:   60×30  → Δj≈250 A/m², ΔT≈1 K
#   Fine:    120×60  → Δj≈125 A/m², ΔT≈0.5 K
#   X-Fine:  240×120 → Δj≈60 A/m²,  ΔT≈0.2 K
#   XX-Fine: 480×240 → Δj≈30 A/m²,  ΔT≈0.1 K
GRID_SIZES     = [(30, 15), (60, 30), (120, 60), (240, 120), (480, 240)]
GRID_CONV_RTOL = 0.001   # relative objective change threshold
GRID_CONV_DJ   = 50.0    # A/m²  — also require |Δj| < this
GRID_CONV_DT   = 0.5     # K     — also require |ΔT| < this

# ---------------------------------------------------------------------------
# Physically defined representative scenarios
# (no prior simulation results used — controller-neutral)
# ---------------------------------------------------------------------------
# Each entry: (description, P_avail_kW, p_elec_EUR_per_kWh, T_stack_C, V_deg_mV)
# T_stack: current measured stack temperature (thermal state)
# V_deg:   accumulated degradation (health state)
# j_prev and m_prev are set to nominal steady-state values inside build_state()

SCENARIOS = [
    # load × price sweep
    ("Low load,  low price",      80,   0.02,  57,  1.0),
    ("Low load,  high price",     80,   0.35,  57,  1.0),
    ("Med load,  low price",     220,   0.02,  60,  3.0),
    ("Med load,  mid price",     220,   0.08,  60,  3.0),
    ("Med load,  high price",    220,   0.35,  60,  3.0),
    ("High load, low price",     450,   0.02,  63,  5.0),
    ("High load, mid price",     450,   0.08,  63,  5.0),
    # edge cases
    ("Near power constraint",    490,   0.05,  64,  5.0),
    ("Peak wind (current limit active)", 495,  0.05,  64,  5.0),
    ("Negative electricity price",300, -0.02,  61,  3.0),
    ("High degradation (BOL+14 mV)", 250, 0.10, 60, 14.0),
    # high-temperature cases (aware controller routinely reaches 70-77°C)
    ("High T, high load, low price",  450,  0.02,  70,  5.0),
    ("High T, high load, mid price",  450,  0.08,  73,  7.0),
    ("Near T_max, med load",          300,  0.05,  75,  8.0),
]
# Scenarios used for grid convergence study (indices into SCENARIOS list)
# 3 = Med load / mid price  (baseline)
# 7 = Near power constraint (widest j range — hardest for the grid)
# 10 = High T, high load (exercises high-temperature landscape)
# 0 = Low load / low price  (confirm coarse grids are already fine)
CONV_SCENARIO_IDXS = [3, 7, 8, 10, 0]

# COBYQA: try standalone package (works with scipy < 1.14)
try:
    from cobyqa import minimize as cobyqa_minimize
    COBYQA_AVAILABLE = True
except ImportError:
    COBYQA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Plant / config helpers
# ---------------------------------------------------------------------------

def stack_capex_eur(plant: dict) -> float:
    econ = plant["economics"]
    return (float(econ["capex_usd_per_kW"]) * float(econ["eur_per_usd"])
            * float(plant["stack"]["P_rating_W"]) / 1000.0)


def build_bounds(
    plant: dict,
    ctrl_cfg: dict,
    P_avail_W: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """Return (j_lo, j_hi, T_lo, T_hi) honouring plant limits and P_avail."""
    stack = plant["stack"]
    j_lo = float(stack["j_min_A_per_m2"])
    j_hi = float(stack["j_max_A_per_m2"])
    T_lo = float(stack["T_min_K"])
    T_hi = float(stack["T_max_K"])
    if "T_max_K" in ctrl_cfg:
        T_hi = min(T_hi, float(ctrl_cfg["T_max_K"]))
    if P_avail_W is not None and P_avail_W > 0:
        N_cells = int(stack["N_cells"])
        A_cell  = float(stack["A_cell_m2"])
        j_pm = P_avail_W / (N_cells * A_cell * 1.8)
        j_hi = min(j_hi, max(j_pm, j_lo))
    return j_lo, j_hi, T_lo, T_hi


def aware_ctrl_cfg(plant: dict) -> dict:
    """Aware controller config (degradation on, no RUL, no fixed T)."""
    T_max = min(float(plant["stack"]["T_max_K"]), 348.15)
    return {
        "type": "supervisory_optimizer",
        "T_max_K": T_max,
        "objective": {"include_degradation": True, "use_rul_feedback": False},
        "solver": {"name": "scipy_slsqp", "max_iter": 100, "tol": 1e-6},
    }


# ---------------------------------------------------------------------------
# Single (j, T) evaluation — one preview call, cost computed inline
# ---------------------------------------------------------------------------

def eval_point(
    j: float, T: float,
    state: dict, plant: dict, ctrl_cfg: dict,
) -> Tuple[float, bool, float]:
    """
    Evaluate objective at (j, T).
    Returns (cost_eur_h, power_feasible, P_total_max_W).
    Uses exactly one preview call so grid search doesn't double-evaluate.
    """
    try:
        prev = preview_interval_jT(
            j_A_per_m2=j,
            T_target_K=T,
            V_deg_V=state["V_deg_V"],
            P_avail_W=state["P_avail_W"],
            plant=plant,
            T_stack_K=state["T_stack_K"],
            dt_s=state["dt_s"],
            m_prev_kg_s=state["m_prev_kg_s"],
        )
    except Exception:
        return float("inf"), False, float("inf")

    feasible = prev.P_total_max_W <= state["P_avail_W"] * 1.005

    econ  = plant["economics"]
    capex = stack_capex_eur(plant)
    p_H2  = float(econ["p_H2_eur_per_kg"])
    V_EOL = float(plant["degradation"]["V_deg_EOL_V"])
    dt_h  = state["dt_s"] / 3600.0

    c_elec = state["p_elec_eur_per_kWh"] * prev.P_total_avg_W / 1000.0
    c_h2   = p_H2 * prev.m_dot_H2_avg_kg_h
    inc_deg = bool(ctrl_cfg["objective"].get("include_degradation", True))
    c_deg  = capex * prev.dV_deg_V / V_EOL / dt_h if inc_deg else 0.0

    cost = c_elec - c_h2 + c_deg
    return float(cost), feasible, float(prev.P_total_max_W)


# ---------------------------------------------------------------------------
# Build optimizer state from a physical scenario (no prior CSV needed)
# ---------------------------------------------------------------------------

T_PREV_TARGET_C = 65.0   # fixed warm-start for all scenarios — mid-range of [T_min, T_max]
                         # decoupled from T_stack so the benchmark does not favour
                         # solvers that are already initialised near the optimum


def build_state(
    P_avail_kW: float,
    p_elec: float,
    T_stack_C: float,
    V_deg_mV: float,
    plant: dict,
) -> dict:
    """
    Construct an optimizer input state from physical parameters.

    j_prev is set to a nominal value consistent with the available power
    (mid-range of the j search space at that P_avail).
    T_prev_target is fixed at T_PREV_TARGET_C for all scenarios (not tied to
    T_stack) so the warm-start point is the same regardless of thermal state.
    m_prev is set to stoichiometric water flow at j_prev.
    """
    stack = plant["stack"]
    j_min  = float(stack["j_min_A_per_m2"])
    j_max  = float(stack["j_max_A_per_m2"])
    N_cells = int(stack["N_cells"])
    A_cell  = float(stack["A_cell_m2"])

    P_avail_W = P_avail_kW * 1e3
    # Estimate mid-range j for this power level (V_est = 1.9 V to be conservative)
    j_pm  = P_avail_W / (N_cells * A_cell * 1.9)
    j_prev = float(np.clip(0.5 * (j_min + min(j_max, j_pm)), j_min, j_max))

    T_stack_K = T_stack_C + 273.15

    # Stoichiometric water flow at j_prev
    ec   = plant["electrochemistry"]
    wf   = plant["water_feed"]
    mm   = plant["fluids"]["molar_masses"]
    F    = float(ec["F_C_per_mol"])
    n_e  = float(ec["n_e"])
    M_H2O = float(mm["H2O_kg_per_mol"])
    lam   = float(wf["nu_w"])
    n_H2  = N_cells * (j_prev * A_cell) / (n_e * F)
    m_prev = lam * n_H2 * M_H2O

    return {
        "j_prev_A_per_m2":    j_prev,
        "T_prev_target_K":    T_PREV_TARGET_C + 273.15,
        "V_deg_V":            V_deg_mV * 1e-3,
        "p_elec_eur_per_kWh": p_elec,
        "P_avail_W":          P_avail_W,
        "T_stack_K":          T_stack_K,
        "m_prev_kg_s":        m_prev,
        "dt_s":               DT_S,
    }


def build_all_states(plant: dict) -> Tuple[List[dict], List[dict]]:
    """
    Build states for all SCENARIOS.
    Returns (scenario_dicts, state_dicts).
    """
    scenarios = []
    states    = []
    for (desc, P_kW, p, T_C, V_mV) in SCENARIOS:
        scenarios.append({"desc": desc, "P_kW": P_kW, "p_elec": p,
                          "T_C": T_C, "V_mV": V_mV})
        states.append(build_state(P_kW, p, T_C, V_mV, plant))
    return scenarios, states


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid(
    n_j: int, n_T: int,
    state: dict, plant: dict, ctrl_cfg: dict,
) -> dict:
    """Evaluate objective on a (n_j × n_T) grid. Return best feasible point."""
    j_lo, j_hi, T_lo, T_hi = build_bounds(plant, ctrl_cfg, state["P_avail_W"])
    js = np.linspace(j_lo, j_hi, n_j)
    Ts = np.linspace(T_lo, T_hi, n_T)

    t0 = time.perf_counter()
    best_J, j_best, T_best = float("inf"), j_lo, T_lo
    n_feas = 0

    for j in js:
        for T_val in Ts:
            cost, feas, _ = eval_point(j, T_val, state, plant, ctrl_cfg)
            if feas and np.isfinite(cost):
                n_feas += 1
                if cost < best_J:
                    best_J, j_best, T_best = cost, j, T_val

    rt = time.perf_counter() - t0
    return {
        "J": best_J, "j": j_best, "T": T_best,
        "runtime_s": rt, "n_j": n_j, "n_T": n_T, "n_feas": n_feas,
        "success": n_feas > 0, "feasible": n_feas > 0,
    }


def grid_convergence_study(
    state: dict, plant: dict, ctrl_cfg: dict,
) -> Tuple[pd.DataFrame, Tuple[int, int]]:
    """
    Run grids of increasing density on a single state.

    Convergence is evaluated vs the finest grid (480×240), not vs the previous
    coarser grid, so the criterion is absolute accuracy rather than incremental change.

    Convergence requires ALL three conditions vs the finest grid:
      - relative objective change  < GRID_CONV_RTOL  (0.1 %)
      - |Δj|                       < GRID_CONV_DJ    (50 A/m²)
      - |ΔT|                       < GRID_CONV_DT    (0.5 K)

    Secondary criterion: if ΔJ < GRID_CONV_RTOL/100 (i.e. < 0.001%), the objective
    is effectively flat and ΔT/Δj exceedances are due to degeneracy, not missing
    the optimum.  Convergence is also declared in that case.

    The coarsest grid that satisfies the criteria vs the finest grid is adopted
    as the reference.
    """
    FLAT_RTOL = GRID_CONV_RTOL / 100   # 0.001 % — "objective is flat" override
    rows: list = []

    # --- pass 1: run all grids ---
    for n_j, n_T in GRID_SIZES:
        print(f"    Grid {n_j}×{n_T} ({n_j*n_T} pts) ...", end=" ", flush=True)
        r = run_grid(n_j, n_T, state, plant, ctrl_cfg)
        r["rel_change"] = float("nan")
        r["delta_j"]    = float("nan")
        r["delta_T"]    = float("nan")
        print(f"J={r['J']:.6f}  j={r['j']:.0f}  T={r['T']-273.15:.1f}°C"
              f"  rt={r['runtime_s']:.1f}s")
        rows.append(r)

    # --- pass 2: compute deviations vs finest grid ---
    finest = rows[-1]
    ref_grid = GRID_SIZES[-1]   # default: finest needed

    for idx, r in enumerate(rows[:-1]):   # skip finest itself
        r["rel_change"] = abs(r["J"] - finest["J"]) / (abs(finest["J"]) + 1e-10)
        r["delta_j"]    = abs(r["j"] - finest["j"])
        r["delta_T"]    = abs(r["T"] - finest["T"])

    # --- pass 3: find coarsest grid that converges vs finest ---
    for idx, r in enumerate(rows[:-1]):
        primary  = (r["rel_change"] < GRID_CONV_RTOL
                    and r["delta_j"] < GRID_CONV_DJ
                    and r["delta_T"] < GRID_CONV_DT)
        flat_obj = r["rel_change"] < FLAT_RTOL
        if primary or flat_obj:
            ref_grid = GRID_SIZES[idx]
            break   # coarsest that converges

    # --- print summary table ---
    print(f"    {'Grid':>10}  {'ΔJ%':>7}  {'Δj':>6}  {'ΔT':>6}  {'converged':>10}")
    for idx, r in enumerate(rows[:-1]):
        primary  = (r["rel_change"] < GRID_CONV_RTOL
                    and r["delta_j"] < GRID_CONV_DJ
                    and r["delta_T"] < GRID_CONV_DT)
        flat_obj = r["rel_change"] < FLAT_RTOL
        mark = "✓" if (primary or flat_obj) else "✗"
        n_j, n_T = GRID_SIZES[idx]
        print(f"    {n_j}×{n_T:>3}      "
              f"  {r['rel_change']*100:>6.3f}%"
              f"  {r['delta_j']:>6.0f}"
              f"  {r['delta_T']:>5.2f}K"
              f"  {mark}")

    return pd.DataFrame(rows), ref_grid


def multi_scenario_convergence(
    scenarios: List[dict],
    state_list: List[dict],
    plant: dict,
    ctrl_cfg: dict,
) -> Tuple[pd.DataFrame, Tuple[int, int]]:
    """
    Run grid_convergence_study on CONV_SCENARIO_IDXS scenarios.
    The final reference grid is the worst-case (finest needed) across all scenarios.
    """
    grid_rank = {sz: i for i, sz in enumerate(GRID_SIZES)}  # (31,21)->0, (51,31)->1, (81,41)->2
    all_rows  = []
    worst_ref = GRID_SIZES[0]   # start optimistic; upgrade if any scenario needs finer

    for idx in CONV_SCENARIO_IDXS:
        sc    = scenarios[idx]
        state = state_list[idx]
        print(f"\n  Scenario [{sc['desc']}]"
              f"  P={sc['P_kW']:.0f} kW  p={sc['p_elec']:+.3f} EUR/kWh")
        df_sc, ref = grid_convergence_study(state, plant, ctrl_cfg)
        df_sc["scenario"] = sc["desc"]
        all_rows.append(df_sc)
        print(f"  → converged at reference: {ref[0]}×{ref[1]}")

        # Keep the finest ref grid needed across all scenarios
        if grid_rank[ref] > grid_rank[worst_ref]:
            worst_ref = ref

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"\n  Final reference grid (worst case across scenarios): "
          f"{worst_ref[0]}×{worst_ref[1]}")
    return combined, worst_ref


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def _normalised_interface(state, plant, ctrl_cfg):
    """
    Build normalised obj(x) and power_slack(x) functions plus denorm().
    x ∈ [0,1]²  maps to [j_lo,j_hi] × [T_lo,T_hi].
    """
    j_lo, j_hi, T_lo, T_hi = build_bounds(plant, ctrl_cfg, state["P_avail_W"])
    j_range = max(j_hi - j_lo, 1.0)
    T_range = max(T_hi - T_lo, 0.1)

    def denorm(x):
        j = float(np.clip(j_lo + x[0] * j_range, j_lo, j_hi))
        T = float(np.clip(T_lo + x[1] * T_range, T_lo, T_hi))
        return j, T

    def obj(x):
        j, T = denorm(x)
        cost, _, _ = eval_point(j, T, state, plant, ctrl_cfg)
        return float(cost) if np.isfinite(cost) else 1e6

    def power_slack(x):
        """≥ 0 when feasible."""
        j, T = denorm(x)
        _, _, P_max = eval_point(j, T, state, plant, ctrl_cfg)
        return float(state["P_avail_W"] - P_max)

    return obj, power_slack, denorm, (j_lo, j_hi, T_lo, T_hi), (j_range, T_range)


def run_slsqp_1start(state: dict, plant: dict, ctrl_cfg: dict) -> dict:
    """SLSQP with a single warm-start from the previous hour."""
    obj, power_slack, denorm, (j_lo, j_hi, T_lo, T_hi), (j_range, T_range) = \
        _normalised_interface(state, plant, ctrl_cfg)

    x0_j = float(np.clip((state["j_prev_A_per_m2"] - j_lo) / j_range, 0.0, 1.0))
    x0_T = float(np.clip((state["T_prev_target_K"] - T_lo) / T_range, 0.0, 1.0))
    x0 = np.array([x0_j, x0_T])

    t0 = time.perf_counter()
    success = False
    try:
        res = scipy_minimize(
            obj, x0, method="SLSQP",
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            constraints=[{"type": "ineq", "fun": power_slack}],
            options={"ftol": 1e-6, "maxiter": 100, "eps": 1e-3},
        )
        j_opt, T_opt = denorm(res.x)
        cost = float(res.fun) if np.isfinite(res.fun) else obj(res.x)
        success = bool(res.success) or np.isfinite(res.fun)
    except Exception:
        j_opt  = float(np.clip(state["j_prev_A_per_m2"], j_lo, j_hi))
        T_opt  = float(np.clip(state["T_prev_target_K"],  T_lo, T_hi))
        cost   = obj(np.array([x0_j, x0_T]))

    rt = time.perf_counter() - t0

    # Re-evaluate for consistent feasibility check
    cost2, feas, _ = eval_point(j_opt, T_opt, state, plant, ctrl_cfg)
    return {"J": cost2, "j": j_opt, "T": T_opt,
            "runtime_s": rt, "success": success, "feasible": feas}


def run_slsqp_3start(state: dict, plant: dict, ctrl_cfg: dict) -> dict:
    """SLSQP with 3 starting points (production implementation)."""
    from pemwe.control.optimizer import optimize_jT

    t0 = time.perf_counter()
    j_opt, T_opt, _ = optimize_jT(
        j_prev_A_per_m2=state["j_prev_A_per_m2"],
        T_prev_target_K=state["T_prev_target_K"],
        V_deg_V=state["V_deg_V"],
        p_elec_eur_per_kWh=state["p_elec_eur_per_kWh"],
        P_avail_W=state["P_avail_W"],
        dt_s=state["dt_s"],
        plant=plant,
        ctrl_cfg=ctrl_cfg,
        T_stack_K=state["T_stack_K"],
        m_prev_kg_s=state["m_prev_kg_s"],
    )
    rt = time.perf_counter() - t0

    # Evaluate on same function as other methods for apples-to-apples comparison
    cost, feas, _ = eval_point(j_opt, T_opt, state, plant, ctrl_cfg)
    return {"J": cost, "j": j_opt, "T": T_opt,
            "runtime_s": rt, "success": True, "feasible": feas}


def run_cobyqa(state: dict, plant: dict, ctrl_cfg: dict) -> Optional[dict]:
    """COBYQA derivative-free constrained optimizer."""
    if not COBYQA_AVAILABLE:
        return None

    obj, power_slack, denorm, _, _ = \
        _normalised_interface(state, plant, ctrl_cfg)

    x0 = np.array([0.5, 0.5])   # centre of normalised space
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    constraints = [{"type": "ineq", "fun": power_slack}]

    t0 = time.perf_counter()
    try:
        res = cobyqa_minimize(
            obj, x0,
            bounds=bounds,
            constraints=constraints,
            options={"maxfev": 500, "feasibility_tol": 1e-4},
        )
        j_opt, T_opt = denorm(res.x)
        success = bool(res.success)
    except Exception as e:
        print(f"  COBYQA error: {e}")
        return None

    rt = time.perf_counter() - t0
    cost, feas, _ = eval_point(j_opt, T_opt, state, plant, ctrl_cfg)
    return {"J": cost, "j": j_opt, "T": T_opt,
            "runtime_s": rt, "success": success, "feasible": feas}


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------

def run_comparison(
    hours: List[dict],
    state_list: List[dict],
    plant: dict,
    ctrl_cfg: dict,
    ref_grid: Tuple[int, int],
) -> pd.DataFrame:
    n_j, n_T = ref_grid
    records = []

    for sc, state in zip(hours, state_list):
        desc = sc["desc"]
        print(f"\n  [{desc}]"
              f"  P={state['P_avail_W']/1000:.0f} kW"
              f"  p={state['p_elec_eur_per_kWh']:.3f} EUR/kWh")

        # ---- Grid reference ----
        print(f"    Grid {n_j}×{n_T} ...", end=" ", flush=True)
        grd = run_grid(n_j, n_T, state, plant, ctrl_cfg)
        J_ref = grd["J"]
        print(f"J={J_ref:.5f}  j={grd['j']:.0f}  T={grd['T']-273.15:.1f}°C"
              f"  rt={grd['runtime_s']:.1f}s")

        base = {"desc": desc, "J_grid": J_ref,
                "j_grid": grd["j"], "T_grid": grd["T"]}

        def record(method, r):
            gap = r["J"] - J_ref
            rel = gap / (abs(J_ref) + 1e-10)
            dj  = r["j"] - grd["j"]
            dT  = r["T"] - grd["T"]
            print(f"    {method:<10} J={r['J']:.5f}  gap={gap:+.4f}"
                  f"  rel={rel*100:+.3f}%  Δj={dj:+.0f}  ΔT={dT:+.1f}K"
                  f"  rt={r['runtime_s']*1000:.1f}ms"
                  f"  ok={r['success']}  feas={r['feasible']}")
            records.append({
                **base, "method": method,
                "J": r["J"], "j": r["j"], "T": r["T"],
                "gap": gap, "rel_gap": rel,
                "delta_j": dj, "delta_T": dT,
                "runtime_s": r["runtime_s"],
                "success": r["success"], "feasible": r["feasible"],
            })

        record("SLSQP-1", run_slsqp_1start(state, plant, ctrl_cfg))
        record("SLSQP-3", run_slsqp_3start(state, plant, ctrl_cfg))

        if COBYQA_AVAILABLE:
            rc = run_cobyqa(state, plant, ctrl_cfg)
            if rc is not None:
                record("COBYQA", rc)
        else:
            print(f"    COBYQA     skipped (not installed)")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, conv_df: Optional[pd.DataFrame], out_dir: Path,
               ref_grid: Optional[Tuple[int, int]] = None):
    sys.path.insert(0, str(REPO_ROOT / "run"))
    from plot_style import apply_style, BLUE, ORANGE, GREEN, RED, GREY, DARK
    apply_style()

    palette = {"SLSQP-1": BLUE, "SLSQP-3": ORANGE, "COBYQA": GREEN}
    has_comp = "method" in df.columns and not df.empty
    has_conv = conv_df is not None and not conv_df.empty
    methods  = df["method"].unique().tolist() if has_comp else []

    # ---- Comparison figure (gap + runtime) ----
    if has_comp:
        ref_label = (f"{ref_grid[0]}\\times{ref_grid[1]}" if ref_grid else "dense-grid")
        fig, (ax_gap, ax_rt) = plt.subplots(1, 2, figsize=(9, 4.5))

        bp_kw = dict(
            patch_artist=True,
            medianprops ={"color": DARK, "lw": 2.0},
            whiskerprops={"color": DARK, "lw": 1.2},
            capprops    ={"color": DARK, "lw": 1.2},
            boxprops    ={"edgecolor": DARK, "lw": 1.2},
            flierprops  ={"marker": "o", "ms": 4, "alpha": 0.6,
                          "markeredgecolor": DARK, "markeredgewidth": 0.5},
        )

        # (a) Relative objective gap
        data_gap = [df.loc[df["method"] == m, "rel_gap"].values * 100 for m in methods]
        bp = ax_gap.boxplot(data_gap, labels=methods, **bp_kw)
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(palette.get(m, GREY))
            patch.set_alpha(0.75)
        ax_gap.axhline(0, ls="--", lw=1.0, color=DARK, alpha=0.35,
                       label=rf"${ref_label}$ grid reference")
        ax_gap.set_ylabel(r"Relative gap to grid reference [\%]")
        ax_gap.legend()

        # (b) Runtime
        data_rt = [df.loc[df["method"] == m, "runtime_s"].values * 1000 for m in methods]
        bp2 = ax_rt.boxplot(data_rt, labels=methods, **bp_kw)
        for patch, m in zip(bp2["boxes"], methods):
            patch.set_facecolor(palette.get(m, GREY))
            patch.set_alpha(0.75)
        ax_rt.set_ylabel(r"Wall-clock time [ms]")
        ax_rt.set_yscale("log")

        fig.tight_layout()
        out_png = out_dir / "optimizer_comparison.png"
        out_pdf = out_dir / "optimizer_comparison.pdf"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure: {out_png}")
        print(f"  Figure: {out_pdf}")

    # ---- Grid convergence figure ----
    if has_conv:
        if "scenario" in conv_df.columns:
            groups = [(name, grp.reset_index(drop=True))
                      for name, grp in conv_df.groupby("scenario", sort=False)]
        else:
            groups = [("", conv_df.reset_index(drop=True))]

        sc_colors = [BLUE, RED, GREEN, ORANGE]
        n_sc      = len(groups)
        THRESH    = GRID_CONV_RTOL * 100

        fig2, axes = plt.subplots(1, n_sc, figsize=(4.2 * n_sc, 4.5), sharey=False)
        if n_sc == 1:
            axes = [axes]

        for sc_idx, (ax_sc, (sc_name, grp)) in enumerate(zip(axes, groups)):
            color       = sc_colors[sc_idx % len(sc_colors)]
            grid_labels = [rf"${int(r.n_j)}\times{int(r.n_T)}$"
                           for r in grp.itertuples()]
            J_vals = grp["J"].values
            xs     = range(len(grid_labels))

            ax_sc.plot(xs, J_vals, "o-", color=color, ms=7, lw=1.8)
            ax_sc.set_xticks(xs)
            ax_sc.set_xticklabels(grid_labels)
            ax_sc.set_xlabel(r"Grid size ($n_j \times n_T$)")
            ax_sc.set_ylabel(r"Best objective [\euro/h]")
            ax_sc.set_title(sc_name if sc_name else "Convergence")

            for i in range(1, len(J_vals)):
                rel    = abs(J_vals[i] - J_vals[i-1]) / (abs(J_vals[i-1]) + 1e-10) * 100
                passed = rel < THRESH
                ax_sc.annotate(
                    rf"{rel:.3f}\%",
                    xy=(i, J_vals[i]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9,
                    color=GREEN if passed else RED,
                )

        fig2.suptitle(r"Grid convergence study --- worst-case scenarios")
        fig2.tight_layout()
        out_conv_png = out_dir / "grid_convergence.png"
        out_conv_pdf = out_dir / "grid_convergence.pdf"
        fig2.savefig(out_conv_png, dpi=300, bbox_inches="tight")
        fig2.savefig(out_conv_pdf, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Figure: {out_conv_png}")

    # ---- Summary table ----
    if not has_comp:
        return
    print("\n=== Summary ===")
    summary = (
        df.groupby("method")
        .agg(
            avg_gap_pct    = ("rel_gap",   lambda x: x.mean()   * 100),
            worst_gap_pct  = ("rel_gap",   lambda x: x.max()    * 100),
            median_rt_ms   = ("runtime_s", lambda x: x.median() * 1000),
            avg_rt_ms      = ("runtime_s", lambda x: x.mean()   * 1000),
            worst_rt_ms    = ("runtime_s", lambda x: x.max()    * 1000),
            n_success      = ("success",   "sum"),
            n_feasible     = ("feasible",  "sum"),
            n_total        = ("success",   "count"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False, float_format="{:.3f}".format))
    print("\nNote: median_rt_ms matches the boxplot centre; avg_rt_ms is pulled up by outliers.")


# ---------------------------------------------------------------------------
# Validate: re-run optimizer on hours from the aware simulation and compare
# ---------------------------------------------------------------------------

AWARE_CSV = REPO_ROOT / "results" / "degradation_aware_wind_spot_dk1" / "degradation_aware_wind_spot_dk1.csv"

# Number of hours to check
N_VALIDATE_HOURS = 5


def _state_from_csv_row(df: pd.DataFrame, h: int) -> dict:
    """Extract exact optimizer input state from CSV row h (previous row for warm start)."""
    row  = df.iloc[h]
    prev = df.iloc[max(0, h - 1)]

    def g(r, *keys, default=0.0):
        for k in keys:
            if k in r.index and pd.notna(r[k]):
                return float(r[k])
        return default

    return {
        "j_prev_A_per_m2":    g(prev, "j_A_per_m2",        default=2000.0),
        "T_prev_target_K":    g(prev, "T_target_K",
                                "T_stack_actual_K",         default=333.15),
        "V_deg_V":            g(row,  "V_deg_V",            default=0.0),
        "p_elec_eur_per_kWh": g(row,  "p_elec_eur_per_kWh", default=0.05),
        "P_avail_W":          g(row,  "P_avail_W",          default=1e5),
        "T_stack_K":          g(row,  "T_stack_actual_K",   default=333.15),
        "m_prev_kg_s":        g(prev, "m_dot_w_kg_s",       default=0.0),
        "dt_s":               DT_S,
    }


def run_validate(plant: dict, ctrl_cfg: dict) -> None:
    """
    Cross-check: re-run optimize_jT on hours extracted from the aware simulation
    CSV and compare the returned (j*, T*) and objective against what the simulation
    recorded.

    This confirms that the benchmark code path (optimize_jT → preview_interval_jT
    → fast layer) is identical to what ran during the actual simulation.

    Also checks whether eval_point(j*, T*) == CostTerms.total_eur_h, i.e. whether
    the inline cost calculation in eval_point agrees with compute_cost_jT.
    """
    from pemwe.control.optimizer import optimize_jT

    if not AWARE_CSV.exists():
        print(f"  [skip] {AWARE_CSV} not found — run 1yr aware simulation first.")
        return

    df = pd.read_csv(AWARE_CSV)
    op = df[df["j_A_per_m2"] > 100]

    # Select a spread of representative hours from the simulation
    idxs = [
        op["P_avail_W"].idxmin(),
        (op["P_avail_W"] - op["P_avail_W"].quantile(0.25)).abs().idxmin(),
        (op["P_avail_W"] - op["P_avail_W"].median()).abs().idxmin(),
        (op["P_avail_W"] - op["P_avail_W"].quantile(0.75)).abs().idxmin(),
        op["P_avail_W"].idxmax(),
    ]
    idxs = list(dict.fromkeys(idxs))[:N_VALIDATE_HOURS]  # dedup, cap at N

    # --- Part 1: optimizer target vs simulation target (apples-to-apples) ---
    print("\nPart 1 — optimizer target vs simulation supervisory target")
    print(f"  (j_opt vs j_target_A_per_m2,  T_opt vs T_target_K)")
    print(f"\n{'h':>5}  {'desc':<22}  {'j_tgt':>7}  {'j_opt':>7}  {'Δj':>6}"
          f"  {'T_tgt':>6}  {'T_opt':>6}  {'ΔT':>5}"
          f"  {'J_ct':>9}  {'J_ep':>9}  {'ΔJ_rel':>8}")
    print("-" * 108)

    max_dj = max_dT = max_dJ_rel = 0.0

    for h in idxs:
        row   = df.iloc[h]
        state = _state_from_csv_row(df, h)

        # Supervisory targets recorded in the simulation (correct comparands)
        j_tgt = float(row.get("j_target_A_per_m2", row["j_A_per_m2"]))
        T_tgt = float(row["T_target_K"])

        desc = f"P={state['P_avail_W']/1000:.0f}kW p={state['p_elec_eur_per_kWh']:.3f}"

        j_opt, T_opt, ct = optimize_jT(
            j_prev_A_per_m2    = state["j_prev_A_per_m2"],
            T_prev_target_K    = state["T_prev_target_K"],
            V_deg_V            = state["V_deg_V"],
            p_elec_eur_per_kWh = state["p_elec_eur_per_kWh"],
            P_avail_W          = state["P_avail_W"],
            dt_s               = state["dt_s"],
            plant              = plant,
            ctrl_cfg           = ctrl_cfg,
            T_stack_K          = state["T_stack_K"],
            m_prev_kg_s        = state["m_prev_kg_s"],
        )

        J_ct = float(ct.total_eur_h)
        J_ep, _, _ = eval_point(j_opt, T_opt, state, plant, ctrl_cfg)

        dj     = j_opt - j_tgt
        dT     = T_opt - T_tgt
        dJ_rel = abs(J_ct - J_ep) / (abs(J_ct) + 1e-10)

        max_dj     = max(max_dj,     abs(dj))
        max_dT     = max(max_dT,     abs(dT))
        max_dJ_rel = max(max_dJ_rel, dJ_rel)

        print(f"{h:5d}  {desc:<22}  {j_tgt:7.0f}  {j_opt:7.0f}  {dj:+6.0f}"
              f"  {T_tgt-273.15:5.1f}C  {T_opt-273.15:5.1f}C  {dT:+5.1f}K"
              f"  {J_ct:9.5f}  {J_ep:9.5f}  {dJ_rel*100:7.4f}%")

    print("-" * 108)
    print(f"  Max |Δj| (target vs re-opt) = {max_dj:.0f} A/m²")
    print(f"  Max |ΔT| (target vs re-opt) = {max_dT:.2f} K")
    print(f"  Max |ΔJ_rel| (CostTerms vs eval_point) = {max_dJ_rel*100:.4f}%")

    # --- Part 2: supervisory target vs realized (informational only) ---
    print("\nPart 2 — supervisory target vs realized plant state  (informational)")
    print(f"  (j_target vs j_A_per_m2,  T_target vs T_stack_actual_K)")
    print(f"\n{'h':>5}  {'j_tgt':>7}  {'j_real':>7}  {'Δj_real':>8}"
          f"  {'T_tgt':>6}  {'T_real':>6}  {'ΔT_real':>8}")
    print("-" * 60)
    for h in idxs:
        row   = df.iloc[h]
        j_tgt  = float(row.get("j_target_A_per_m2", row["j_A_per_m2"]))
        T_tgt  = float(row["T_target_K"])
        j_real = float(row["j_A_per_m2"])
        T_real = float(row["T_stack_actual_K"])
        print(f"{h:5d}  {j_tgt:7.0f}  {j_real:7.0f}  {j_real-j_tgt:+8.0f}"
              f"  {T_tgt-273.15:5.1f}C  {T_real-273.15:5.1f}C  {T_real-T_tgt:+8.2f}K")
    print()
    print("  ΔT_real is the steady-state thermal error from the P controller.")
    print("  Δj_real is the fast-layer j clipping for power-budget enforcement.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_flat_valley_check(plant: dict, ctrl_cfg: dict) -> None:
    """
    For hours 7732 and 8108 (large ΔT in --validate), hold j fixed at the
    optimized value and sweep T from 50 → 75 °C.  Plots J(T) to show whether
    the objective is genuinely flat in T at those conditions.
    """
    from pemwe.control.optimizer import optimize_jT

    if not AWARE_CSV.exists():
        print(f"  [skip] {AWARE_CSV} not found.")
        return

    df = pd.read_csv(AWARE_CSV)

    HOURS_TO_CHECK = [7732, 8108]
    T_SWEEP_C = np.linspace(50, 75, 51)

    try:
        sys.path.insert(0, str(REPO_ROOT / "run"))
        from make_plots import apply_style
        apply_style()
    except Exception:
        plt.rcParams.update({"font.size": 9})

    fig, axes = plt.subplots(1, len(HOURS_TO_CHECK), figsize=(10, 4))
    fig.suptitle(r"Flat-valley check: $J$ vs $T_\mathrm{target}$ at fixed $j^*$",
                 fontsize=11)

    for ax, h in zip(axes, HOURS_TO_CHECK):
        state = _state_from_csv_row(df, h)
        row   = df.iloc[h]

        # --- State dump: confirm inputs are identical for both hours ---
        j_lo, j_hi, T_lo, T_hi = build_bounds(plant, ctrl_cfg, state["P_avail_W"])
        n_sub = int(plant["fast_control"]["n_preview_substeps"])
        print(f"\n  Hour {h} — full optimizer input state")
        print(f"    j_lo            = {j_lo:.1f} A/m²")
        print(f"    j_hi            = {j_hi:.1f} A/m²  (clipped to P_avail)")
        print(f"    T_lo            = {T_lo-273.15:.1f} °C")
        print(f"    T_hi            = {T_hi-273.15:.1f} °C  (ctrl T_max_K applied)")
        print(f"    P_avail_W       = {state['P_avail_W']:.1f} W")
        print(f"    V_deg_V         = {state['V_deg_V']*1e3:.4f} mV")
        print(f"    T_stack_K       = {state['T_stack_K']-273.15:.2f} °C")
        print(f"    m_prev_kg_s     = {state['m_prev_kg_s']:.5f} kg/s")
        print(f"    j_prev_A_per_m2 = {state['j_prev_A_per_m2']:.1f} A/m²")
        print(f"    T_prev_target_K = {state['T_prev_target_K']-273.15:.2f} °C")
        print(f"    n_preview_subs  = {n_sub}")
        print(f"    T_target (sim)  = {float(row['T_target_K'])-273.15:.2f} °C")

        # Get optimized j for this hour
        j_opt, T_opt, ct = optimize_jT(
            j_prev_A_per_m2    = state["j_prev_A_per_m2"],
            T_prev_target_K    = state["T_prev_target_K"],
            V_deg_V            = state["V_deg_V"],
            p_elec_eur_per_kWh = state["p_elec_eur_per_kWh"],
            P_avail_W          = state["P_avail_W"],
            dt_s               = state["dt_s"],
            plant              = plant,
            ctrl_cfg           = ctrl_cfg,
            T_stack_K          = state["T_stack_K"],
            m_prev_kg_s        = state["m_prev_kg_s"],
        )

        T_tgt_sim = float(row["T_target_K"])

        # Sweep T with j fixed at j_opt
        J_vals, feasible_mask = [], []
        for T_C in T_SWEEP_C:
            T_K = T_C + 273.15
            cost, feas, _ = eval_point(j_opt, T_K, state, plant, ctrl_cfg)
            J_vals.append(cost if feas and np.isfinite(cost) else np.nan)
            feasible_mask.append(feas)

        J_arr = np.array(J_vals)
        J_ref = float(ct.total_eur_h)   # optimizer's reported optimum

        # Plot
        ax.plot(T_SWEEP_C, J_arr, lw=1.5, color="#1f77b4", label=r"$J(T)$ at $j^*$")
        ax.axvline(T_opt - 273.15, ls="--", lw=1.0, color="#d62728",
                   label=f"$T^*_\\mathrm{{opt}}$ = {T_opt-273.15:.1f}°C")
        ax.axvline(T_tgt_sim - 273.15, ls=":", lw=1.0, color="#2ca02c",
                   label=f"$T_\\mathrm{{sim}}$ = {T_tgt_sim-273.15:.1f}°C")
        ax.axhline(J_ref, ls="-.", lw=0.8, color="k", alpha=0.4,
                   label=f"$J^*$ = {J_ref:.4f}")

        # Annotate the J range to quantify flatness
        valid = J_arr[~np.isnan(J_arr)]
        if len(valid) > 1:
            J_range = valid.max() - valid.min()
            J_rel   = J_range / (abs(J_ref) + 1e-10)
            ax.text(0.03, 0.97,
                    f"$\\Delta J$ over sweep = {J_range:.4f} EUR/h\n"
                    f"({J_rel*100:.3f}% of $|J^*|$)",
                    transform=ax.transAxes, va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_xlabel(r"$T_\mathrm{target}$ [°C]")
        ax.set_ylabel(r"$J$ [EUR/h]")
        ax.set_title(f"Hour {h}  —  "
                     f"P={state['P_avail_W']/1000:.0f} kW, "
                     f"p={state['p_elec_eur_per_kWh']:.3f} EUR/kWh, "
                     f"$j^*$={j_opt:.0f} A/m²")
        ax.legend(fontsize=7)

        print(f"  h={h}  j*={j_opt:.0f}  T_opt={T_opt-273.15:.1f}°C"
              f"  T_sim={T_tgt_sim-273.15:.1f}°C"
              f"  ΔJ_sweep={valid.max()-valid.min():.5f} EUR/h"
              f"  ({(valid.max()-valid.min())/(abs(J_ref)+1e-10)*100:.4f}%)")

    plt.tight_layout()
    out_png = OUT_DIR / "flat_valley_check.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {out_png}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run",         action="store_true", help="Run full benchmark")
    p.add_argument("--plots-only",  action="store_true", help="Plot from saved CSV")
    p.add_argument("--conv-only",   action="store_true", help="Only run grid convergence study")
    p.add_argument("--validate",    action="store_true",
                   help="Cross-check optimizer vs aware simulation CSV")
    p.add_argument("--flat-valley", action="store_true",
                   help="T sweep at fixed j* for hours with large ΔT in --validate")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plant    = load_plant(PLANT_CFG)
    ctrl_cfg = aware_ctrl_cfg(plant)

    print(f"scipy {scipy.__version__}  |  COBYQA available: {COBYQA_AVAILABLE}")

    if args.validate:
        print("\n=== Validation: re-running optimizer on aware simulation hours ===")
        run_validate(plant, ctrl_cfg)
        return

    if args.flat_valley:
        print("\n=== Flat-valley check: J vs T at fixed j* (hours 7732, 8108) ===")
        run_flat_valley_check(plant, ctrl_cfg)
        return

    # Build controller-neutral states from physical scenarios (no prior CSV)
    scenarios, state_list = build_all_states(plant)

    print(f"\n{len(scenarios)} representative scenarios (physically defined, no prior CSV):")
    for sc, _ in zip(scenarios, state_list):
        print(f"  {sc['desc']:<35}  P={sc['P_kW']:4.0f} kW"
              f"  p={sc['p_elec']:+.3f} EUR/kWh"
              f"  T={sc['T_C']:.0f}°C  V_deg={sc['V_mV']:.1f} mV")

    # ---- Grid convergence ----
    conv_csv = OUT_DIR / "grid_convergence.csv"
    conv_df  = None
    ref_grid = GRID_SIZES[-1]   # default: finest

    if args.conv_only or (args.run and not conv_csv.exists()):
        descs = [scenarios[i]["desc"] for i in CONV_SCENARIO_IDXS]
        print(f"\n--- Grid convergence study  ({', '.join(descs)}) ---")
        conv_df, ref_grid = multi_scenario_convergence(
            scenarios, state_list, plant, ctrl_cfg)
        conv_df.to_csv(conv_csv, index=False)
        print(f"  Reference grid chosen: {ref_grid[0]}×{ref_grid[1]}")
    elif conv_csv.exists():
        conv_df = pd.read_csv(conv_csv)
        # Recover worst-case ref grid: per scenario, find the coarsest grid that
        # converges vs the finest (new methodology — deviations vs finest grid).
        grid_rank = {(sz[0], sz[1]): i for i, sz in enumerate(GRID_SIZES)}
        worst_rank = 0
        FLAT_RTOL = GRID_CONV_RTOL / 100
        for sc_name, grp in (conv_df.groupby("scenario") if "scenario" in conv_df.columns
                              else [("single", conv_df)]):
            grp = grp.reset_index(drop=True)
            sc_ref = GRID_SIZES[-1]
            # Rows are ordered coarse→fine; skip the finest row (NaN deviations)
            for i in range(len(grp) - 1):
                row = grp.iloc[i]
                rel = row.get("rel_change", float("nan"))
                dj  = row.get("delta_j",    float("nan"))
                dT  = row.get("delta_T",    float("nan"))
                primary  = (rel < GRID_CONV_RTOL and dj < GRID_CONV_DJ and dT < GRID_CONV_DT)
                flat_obj = (rel < FLAT_RTOL)
                if primary or flat_obj:
                    sc_ref = (int(row["n_j"]), int(row["n_T"]))
                    break
            rank = grid_rank.get(sc_ref, len(GRID_SIZES) - 1)
            if rank > worst_rank:
                worst_rank = rank
                ref_grid = sc_ref

    if args.conv_only:
        if conv_df is not None:
            make_plots(pd.DataFrame(), conv_df, OUT_DIR, ref_grid)
        return

    # ---- Full comparison ----
    comp_csv = OUT_DIR / "comparison_table.csv"

    if args.run and not args.plots_only:
        print(f"\n--- Optimizer comparison  (dense-grid ref {ref_grid[0]}×{ref_grid[1]}) ---")
        df_comp = run_comparison(scenarios, state_list, plant, ctrl_cfg, ref_grid)
        df_comp.to_csv(comp_csv, index=False)
        print(f"\nTable saved: {comp_csv}")
    elif comp_csv.exists():
        df_comp = pd.read_csv(comp_csv)
    else:
        print("No comparison CSV. Run with --run first.")
        return

    if df_comp.empty:
        print("Empty comparison table.")
        return

    print("\n--- Generating plots ---")
    make_plots(df_comp, conv_df, OUT_DIR, ref_grid)


if __name__ == "__main__":
    main()
