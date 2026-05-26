from __future__ import annotations

"""
FAST CONTROL LAYER (shared by both controllers)
===============================================

Shared low-level thermal / flow control layer.

Behavior:
- Supervisory layer provides (j_target, T_target)
- Fast layer computes water flow command m_dot based on:
    1) stoichiometric-flow baseline
    2) three-zone temperature feedback (P control)
    3) actuator dynamics and rate limits
- j is normally kept at j_target

Important:
- If j_target <= 0, the system is treated as fully off:
  no electrolysis, no pump flow, no heater, no cooler.
- Startup is not a separate warm-circulation mode.  When power becomes
  available again and the supervisory controller requests j_target > 0,
  the fast layer enters normal active operation immediately.

Three-zone temperature-feedback logic:
  Zone 1 (T > T_target)         : strong cooling — error relative to target
  Zone 2 (T_w_in < T < T_target): mild cooling   — error relative to inlet water
  Zone 3 (T < T_w_in)           : heating        — warm inlet water heats the stack
"""

from typing import Any, Dict, Tuple

from .objective import total_power_instant_W
from ..blocks.recirculation import min_water_feed_kg_s



def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


def _first_order_rate_limited_step(
    *,
    u_prev: float,
    u_cmd: float,
    tau_s: float,
    du_max_per_s: float,
    dt_s: float,
) -> float:
    """First-order actuator response + symmetric rate limit."""
    alpha = float(dt_s) / (float(tau_s) + float(dt_s))
    u_lag = float(u_prev) + alpha * (float(u_cmd) - float(u_prev))
    du = u_lag - float(u_prev)
    du_max = float(du_max_per_s) * float(dt_s)
    du = _clip(du, -du_max, du_max)
    return float(u_prev) + du


def fast_plc_step(
    *,
    j_target: float,
    T_target_K: float,
    T_stack_K: float,
    P_avail_W: float,
    V_deg_V: float,
    plant: Dict[str, Any],
    m_prev_kg_s: float,
    dt_s: float,
) -> Tuple[float, float]:
    """
    Shared fast low-level P control layer.

    Parameters
    ----------
    j_target    : Supervisory current-density target [A/m^2]; 0 when P_avail < P_min
    T_target_K  : Supervisory temperature target [K]
    T_stack_K   : Current stack temperature [K]
    P_avail_W   : Available electrical power [W]
    V_deg_V     : Accumulated degradation voltage [V]
    plant       : Plant parameters
    m_prev_kg_s : Previous realized flow [kg/s]
    dt_s        : Sub-step duration [s]

    Returns
    -------
    (j_sub, m_sub)
    """
    fc = plant["fast_control"]
    wf = plant["water_feed"]

    T_w_in           = float(wf["T_set_K"])
    tau_flow_s       = float(wf["tau_flow_s"])
    dm_dot_max_kg_s2 = float(wf["dm_dot_max_kg_s2"])
    m_max_phys       = float(wf["m_dot_max_kg_s"])

    Kp_cool = float(fc["Kp_cool_kg_s_per_K"])
    Kp_heat = float(fc["Kp_heat_kg_s_per_K"])

    j_sub = float(j_target)

    # ------------------------------------------------------------------
    # 1)  Fully off case
    # ------------------------------------------------------------------
    if j_target <= 0.0:
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # 2)  Active operation — three-zone P temperature control
    # ------------------------------------------------------------------
    m_min_feed = min_water_feed_kg_s(j_target, plant)
    T_stack  = float(T_stack_K)
    T_target = float(T_target_K)

    if T_stack > T_target:
        # Zone 1: above target — proportional cooling relative to target
        m_cmd = m_min_feed + Kp_cool * (T_stack - T_target)
    elif T_w_in <= T_stack <= T_target:
        # Zone 2: between inlet water and target — minimum flow only.
        # No extra cooling needed: electrolysis heat keeps T_stack near T_target
        # naturally. Using m_min ensures continuity at the Zone 1/2 boundary
        # (both sides give m_min as T_stack → T_target).
        m_cmd = m_min_feed
    else:
        # Zone 3: below inlet water (cold start) — more flow = more heating
        m_cmd = m_min_feed + Kp_heat * (T_w_in - T_stack)

    m_cmd = _clip(m_cmd, m_min_feed, m_max_phys)

    # ------------------------------------------------------------------
    # 3)  Actuator dynamics (first-order lag + rate limit)
    # ------------------------------------------------------------------
    m_sub = _first_order_rate_limited_step(
        u_prev=float(m_prev_kg_s),
        u_cmd=m_cmd,
        tau_s=tau_flow_s,
        du_max_per_s=dm_dot_max_kg_s2,
        dt_s=dt_s,
    )
    m_sub = _clip(m_sub, m_min_feed, m_max_phys)

    # ------------------------------------------------------------------
    # 4)  Power-budget enforcement (derate j if P_total > P_avail)
    # ------------------------------------------------------------------
    for _ in range(10):
        P_tot = total_power_instant_W(
            j_sub, m_sub,
            V_deg_V=V_deg_V,
            plant=plant,
            T_stack_K=T_stack_K,
        )
        if P_tot <= float(P_avail_W) * 1.001:
            break
        if j_sub > 0.0:
            j_sub *= float(P_avail_W) / P_tot
            j_sub = max(j_sub, 0.0)
            m_min_feed = min_water_feed_kg_s(j_sub, plant)
            m_sub = _clip(m_sub, m_min_feed, m_max_phys)
        else:
            return 0.0, 0.0

    return float(j_sub), float(m_sub)
