from __future__ import annotations

"""
OPTIMIZER
=========

Finds the cost-minimising (j*, T_target*) for a single PEMWE stack.

Decision variables
------------------
  j       [A/m^2]   target current density
  T       [K]       target stack temperature

Important
---------
The candidate decision (j, T_target) is evaluated using the same shared
fast layer and plant blocks as the main simulation. This makes the
supervisory optimisation consistent with the realised plant behavior.

Constraints
-----------
  1. Power budget: predicted realised P_total <= P_avail_W
  2. Temperature bounds: T_min <= T_target <= T_max (via bounds)

Notes
-----
- The fast layer is shared by both controllers.
- The only intended difference between controllers is whether degradation
  is included in the supervisory objective.
- Shutdown is handled by the simulation loop (P_avail < P_min), not
  by the optimiser. The optimiser always chooses j >= j_min > 0.
"""

from typing import Any, Dict, Tuple

import numpy as np
from scipy.optimize import minimize

try:
    from cobyqa import minimize as cobyqa_minimize
    _COBYQA_AVAILABLE = True
except ImportError:
    _COBYQA_AVAILABLE = False

from .objective import (
    CostTerms,
    compute_cost_jT,
)


def optimize_jT(
    *,
    j_prev_A_per_m2: float,
    T_prev_target_K: float,
    V_deg_V: float,
    p_elec_eur_per_kWh: float,
    P_avail_W: float,
    dt_s: float,
    plant: Dict[str, Any],
    ctrl_cfg: Dict[str, Any],
    T_stack_K: float,
    m_prev_kg_s: float = 0.0,
) -> Tuple[float, float, CostTerms]:
    """
    Minimise supervisory cost [EUR/h] over (j, T_target).

    Parameters
    ----------
    j_prev_A_per_m2 : float
        Previous supervisory current-density target [A/m^2]
    T_prev_target_K : float
        Previous supervisory temperature target [K]
    V_deg_V : float
        Accumulated degradation voltage [V]
    p_elec_eur_per_kWh : float
        Spot electricity price [EUR/kWh]
    P_avail_W : float
        Available power [W]
    dt_s : float
        Supervisory control interval [s]
    plant, ctrl_cfg : dict
        Plant / controller configuration
    T_stack_K : float
        Current measured stack temperature [K]
    Returns
    -------
    (j_opt, T_target_opt, CostTerms at optimum)
    """
    # --- Bounds from plant parameters (optimiser bounds) ---
    stack_limits = plant["stack"]
    j_min = float(stack_limits["j_min_A_per_m2"])
    j_max = float(stack_limits["j_max_A_per_m2"])
    T_min = float(stack_limits["T_min_K"])  # Not a hard limit, just optimisation
    T_max = float(stack_limits["T_max_K"])  # Not a hard limit, just optimisation

    # Controller-specific T bound override (conservative margin for preview error)
    if "T_max_K" in ctrl_cfg:
        T_max = min(T_max, float(ctrl_cfg["T_max_K"]))

    # Fixed-T mode: pin temperature to a single value (T lever disabled)
    if "T_fixed_K" in ctrl_cfg:
        T_fixed = float(ctrl_cfg["T_fixed_K"])
        T_min = T_fixed
        T_max = T_fixed

    # Rough tightening of upper j bound using a simple stack-power estimate.
    # This is only to help the numerical search. The true power feasibility
    # is enforced by the preview-based constraint below.
    stack_cfg = plant["stack"]
    N_cells = int(stack_cfg["N_cells"])
    A_cell = float(stack_cfg["A_cell_m2"])
    V_est = 1.8
    j_power_max = P_avail_W / (N_cells * A_cell * V_est)

    j_lo = j_min
    j_hi = min(j_max, j_power_max)
    if j_hi < j_lo:
        j_hi = j_lo

    # Keep target-temperature bounds simple and physical.
    T_lo = T_min
    T_hi = T_max

    j_range = max(j_hi - j_lo, 1.0)
    T_range = max(T_hi - T_lo, 0.1)

    def denorm(x: np.ndarray) -> Tuple[float, float]:
        j = float(np.clip(j_lo + x[0] * j_range, j_lo, j_hi))
        T = float(np.clip(T_lo + x[1] * T_range, T_lo, T_hi))
        return j, T

    # Single-entry cache: obj() and power_slack() share one preview per point.
    _cache: list = [None, None]  # [x_cached, cost_cached]

    def _eval(x: np.ndarray) -> CostTerms:
        if _cache[0] is None or not np.array_equal(x, _cache[0]):
            j, T_tgt = denorm(x)
            _cache[1] = compute_cost_jT(
                j,
                T_tgt,
                V_deg_V=V_deg_V,
                p_elec_eur_per_kWh=p_elec_eur_per_kWh,
                plant=plant,
                ctrl_cfg=ctrl_cfg,
                T_stack_K=T_stack_K,
                dt_s=dt_s,
                P_avail_W=P_avail_W,
                m_prev_kg_s=m_prev_kg_s,
            )
            _cache[0] = x.copy()
        return _cache[1]

    def obj(x: np.ndarray) -> float:
        return float(_eval(x).total_eur_h)

    def power_slack(x: np.ndarray) -> float:
        """
        Feasibility constraint: predicted peak total power must not exceed P_avail.
        """
        return float(P_avail_W - _eval(x).P_total_max_W)

    constraints_list = [
        {"type": "ineq", "fun": power_slack},
    ]

    solver_cfg = ctrl_cfg["solver"]
    solver_name = str(solver_cfg.get("name", "scipy_slsqp")).lower()
    ftol = float(solver_cfg["tol"])
    maxiter = int(solver_cfg["max_iter"])

    x0_prev_j = float(np.clip((j_prev_A_per_m2 - j_lo) / j_range, 0.0, 1.0))
    x0_prev_T = float(np.clip((T_prev_target_K - T_lo) / T_range, 0.0, 1.0))

    best_cost = float("inf")
    j_opt = float(np.clip(j_prev_A_per_m2, j_lo, j_hi))
    T_opt = float(np.clip(T_prev_target_K, T_lo, T_hi))

    if solver_name == "cobyqa":
        if not _COBYQA_AVAILABLE:
            raise RuntimeError("cobyqa package not installed — run: pip install cobyqa")
        x0 = np.array([x0_prev_j, x0_prev_T])
        try:
            result = cobyqa_minimize(
                obj, x0,
                bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
                constraints=constraints_list,
                options={"maxfev": maxiter, "feasibility_tol": 1e-4},
            )
            if np.isfinite(result.fun):
                j_cand, T_cand = denorm(result.x)
                if power_slack(result.x) >= -0.005 * P_avail_W:
                    j_opt, T_opt = j_cand, T_cand
        except (ValueError, ArithmeticError):
            pass

    else:
        # SLSQP with 3 starts:
        #  - warm start from previous hour
        #  - high-j (explore high-production region)
        #  - low-j  (explore low-degradation region)
        starts = [
            np.array([x0_prev_j, x0_prev_T]),
            np.array([0.90, 0.33]),
            np.array([0.10, 0.50]),
        ]

        for x0 in starts:
            try:
                result = minimize(
                    obj, x0,
                    method="SLSQP",
                    bounds=[(0.0, 1.0), (0.0, 1.0)],
                    constraints=constraints_list,
                    options={"ftol": ftol, "maxiter": maxiter, "eps": 1e-3},
                )

                # Accept even if SLSQP reports failure — it often flags
                # "Positive directional derivative for linesearch" at
                # bound-active solutions but the point is still usable.
                if not np.isfinite(result.fun):
                    continue

                j_cand, T_cand = denorm(result.x)

                if power_slack(result.x) < -0.005 * P_avail_W:
                    continue

                if float(result.fun) < best_cost:
                    best_cost = float(result.fun)
                    j_opt, T_opt = j_cand, T_cand

            except (ValueError, ArithmeticError):
                continue

    # Final evaluation at optimum
    cost_opt = compute_cost_jT(
        j_opt,
        T_opt,
        V_deg_V=V_deg_V,
        p_elec_eur_per_kWh=p_elec_eur_per_kWh,
        plant=plant,
        ctrl_cfg=ctrl_cfg,
        T_stack_K=T_stack_K,
        dt_s=dt_s,
        P_avail_W=P_avail_W,
        m_prev_kg_s=m_prev_kg_s,
    )

    return j_opt, T_opt, cost_opt