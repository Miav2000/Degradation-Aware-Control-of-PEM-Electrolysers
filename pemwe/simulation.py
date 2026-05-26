from __future__ import annotations

"""
CLOSED-LOOP DYNAMIC SIMULATION
==============================

Runs the PEMWE supervisory controller over time series of electricity
prices and available power, with full dynamic state evolution.

State variables
---------------
  T_stack_K  : stack temperature [K]
  deg_state  : cumulative voltage degradation state [V]

Architecture
------------
  - Supervisory layer chooses (j_target, T_target) each control interval
  - Shared fast layer adjusts water flow within the interval when ON
  - Plant dynamics evolve through electrochemistry, species, water
    management, and thermal blocks

Important:
  - The fast layer is shared by both controllers so the comparison remains fair.
  - The only intended difference between controllers is the supervisory logic.
  - If j = 0, the plant is treated as
    fully shut down for that interval:
      * no electrolysis
      * no water circulation
      * no heater / cooler / pump
      * passive thermal drift only
      * small off-state degradation may still accumulate via gamma_V
"""

import time
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .control import make_policy
from .degradation.voltage_model import DegradationState
from .propagator import propagate_interval


def run_simulation(
    *,
    price_series: Sequence[float],
    p_avail_series: Sequence[float],
    plant: Dict[str, Any],
    ctrl_cfg: Dict[str, Any],
    dt_s: float,
    n_thermal_sub: Optional[int] = None,
    T_stack_init_K: Optional[float] = None,
    V_deg_init_V: float = 0.0,
    detail_window_h: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Run a closed-loop simulation over a price / available-power time series.

    Parameters
    ----------
    price_series : Sequence[float]
        Electricity price at each step [EUR/kWh]
    p_avail_series : Sequence[float]
        Available electrical power at each step [W]
    plant : dict
        Plant parameter dictionary
    ctrl_cfg : dict
        Controller configuration dictionary
    dt_s : float
        Supervisory control interval [s]
    n_thermal_sub : int | None
        Number of fast sub-steps per interval. Defaults to plant config.
    T_stack_init_K : float | None
        Initial stack temperature [K]. Defaults to ambient.
    V_deg_init_V : float, default 0.0
        Initial cumulative degradation voltage [V]

    Returns
    -------
    pd.DataFrame
        One row per supervisory interval
    """
    if len(price_series) != len(p_avail_series):
        raise ValueError("price_series and p_avail_series must have the same length")

    # --- Defaults ---
    if n_thermal_sub is None:
        n_thermal_sub = int(plant["fast_control"]["n_substeps"])

    if T_stack_init_K is None:
        T_stack_init_K = float(plant["thermal"]["T_amb_K"])

    dt_h = dt_s / 3600.0

    # --- Initial state ---
    T_stack: float = float(T_stack_init_K)
    deg_state: DegradationState = DegradationState(V_deg_V=V_deg_init_V)

    # Fast-layer actuator state: previous flow command / realized flow
    m_flow_prev_kg_s: float = 0.0

    # --- Controller ---
    policy = make_policy(ctrl_cfg, plant)

    # --- Logging / counters ---
    records: List[dict] = []
    substep_records: List[dict] = []
    was_on_prev = False
    n_shutdowns = 0
    n_replacements = 0

    # Shared shutdown threshold — includes auxiliary margin so both controllers
    # are only called when enough power exists for j_min + auxiliaries.
    P_min_frac = float(plant["stack"]["P_min_frac"])
    aux_margin_frac = float(plant["auxiliaries"]["aux_margin_frac"])
    from .control.objective import nominal_stack_power_W, _stack_capex_eur
    P_min = P_min_frac * nominal_stack_power_W(plant) * (1.0 + aux_margin_frac)

    t_wall_start = time.perf_counter()

    for k, (p_elec, P_avail) in enumerate(zip(price_series, p_avail_series)):
        t_step_start = time.perf_counter()
        t_h = (k + 1) * dt_h

        # Save on-state at start of step (before shutdown logic updates it)
        was_on_prev_at_step_start = was_on_prev

        # --- 1) Supervisory decision / shutdown logic ---
        if float(P_avail) < P_min:
            # Forced shutdown
            j_opt = 0.0
            T_target_K = T_stack

            # Sync supervisory memory with the actual forced-off state
            if hasattr(policy, "_j_prev"):
                policy._j_prev = 0.0
            if hasattr(policy, "_T_prev"):
                policy._T_prev = float(T_stack)
            if hasattr(policy, "_m_prev"):
                policy._m_prev = 0.0
        else:
            # Pass current EIS health fractions for asymmetric feedback (if enabled)
            j_opt, T_target_K, _cost = policy.step(
                deg_state=deg_state,
                P_avail_W=float(P_avail),
                p_elec_eur_per_kWh=float(p_elec),
                dt_s=dt_s,
                T_stack_K=T_stack,
            )

        # Shutdown event counting (single tracker for both logging and summary)
        is_on = (j_opt > 0.0)
        if was_on_prev and not is_on:
            n_shutdowns += 1
        was_on_prev = is_on

        # If the supervisory layer chose OFF, the actuator state should decay to zero.
        # Since OFF means full shutdown in this model, set previous flow to zero.
        if not is_on:
            m_flow_prev_kg_s = 0.0

        # --- 2) Propagate plant over one supervisory interval ---
        log_subs = (detail_window_h is not None
                    and detail_window_h[0] <= t_h < detail_window_h[1])

        iv = propagate_interval(
            j_target=j_opt,
            T_target_K=T_target_K,
            is_on=is_on,
            T_stack_K=T_stack,
            P_avail_W=float(P_avail),
            V_deg_V=deg_state.V_deg_V,
            plant=plant,
            m_prev_kg_s=m_flow_prev_kg_s,
            dt_s=dt_s,
            n_sub=n_thermal_sub,
            log_substeps=log_subs,
        )

        # Update actuator state
        m_flow_prev_kg_s = iv.m_fin_kg_s
        T_sub = iv.T_fin_K

        # Extract averaged values for logging
        j_actual = iv.j_avg_A_per_m2
        m_actual = iv.m_avg_kg_s
        P_stack_avg = iv.P_stack_avg_W
        P_heater_avg = iv.P_heater_avg_W
        P_cooler_avg = iv.P_cooler_avg_W
        Q_heater_avg = iv.Q_heater_avg_W
        H2_avg = iv.m_dot_H2_avg_kg_h
        Q_gen_avg = iv.Q_gen_avg_W
        Q_water_avg = iv.Q_water_avg_W
        Q_amb_avg = iv.Q_amb_avg_W
        T_avg = iv.T_avg_K
        T_mix_avg = iv.T_mix_avg_K
        T_w_in_avg = iv.T_w_in_avg_K
        T_liq_return_avg = iv.T_liq_return_avg_K
        T_mix_fin = iv.T_mix_fin_K
        T_mix_min = iv.T_mix_min_K
        T_w_in_fin = iv.T_w_in_fin_K
        T_stack_max = iv.T_max_K
        dV_deg_acc = iv.dV_deg_V
        P_aux_actual = iv.P_aux_avg_W
        P_total_actual = iv.P_total_avg_W

        # Collect sub-step records if in detail window
        if log_subs and iv.substep_records:
            for sr in iv.substep_records:
                t_start = t_h - dt_h  # start of this interval
                sr["t_h"] = t_start + (sr["t_sub"] + 1) * (dt_s / n_thermal_sub / 3600.0)
                substep_records.append(sr)

        # --- 3) Degradation update using actual realized substep operation ---
        V_EOL_check = float(plant["degradation"]["V_deg_EOL_V"])
        new_V_deg = deg_state.V_deg_V + dV_deg_acc
        if new_V_deg >= V_EOL_check:
            new_V_deg = 0.0
            n_replacements += 1
        deg_state = DegradationState(V_deg_V=new_V_deg)

        # --- 4) Actual economics from realized operation ---
        p_H2 = float(plant["economics"]["p_H2_eur_per_kg"])

        c_elec_actual = float(p_elec) * P_total_actual / 1000.0  # EUR/h
        # H2 production already includes eta_F (applied in electrochem_step)
        r_H2_actual = p_H2 * H2_avg                                # EUR/h

        capex_eur = _stack_capex_eur(plant)
        c_deg_phys = capex_eur * dV_deg_acc / V_EOL_check / dt_h

        c_shutdown_eur_per_event = float(plant["economics"]["c_shutdown_eur"])
        # Shutdown cost is applied if the stack was ON at the start of the step
        # and OFF after the current supervisory decision.
        shutdown_this_step = 1.0 if (was_on_prev_at_step_start and not is_on) else 0.0
        c_shutdown_step = c_shutdown_eur_per_event * shutdown_this_step

        # --- 6) Log ---
        records.append({
            "step": k,
            "t_h": t_h,
            "p_elec_eur_per_kWh": p_elec,
            "P_avail_W": P_avail,
            "j_A_per_m2": j_actual,
            "j_target_A_per_m2": j_opt,
            "T_target_K": T_target_K,
            "m_dot_w_kg_s": m_actual,
            "T_stack_actual_K": T_sub,
            "T_stack_avg_K": T_avg,
            "T_stack_max_K": T_stack_max,
            "T_mix_K": T_mix_avg,
            "T_mix_fin_K": T_mix_fin,
            "T_mix_min_K": T_mix_min,
            "T_w_in_fin_K": T_w_in_fin,
            "T_liq_return_avg_K": T_liq_return_avg,
            "V_deg_V": deg_state.V_deg_V,
            "dV_deg_V": dV_deg_acc,
            "P_stack_W": P_stack_avg,
            "P_aux_W": P_aux_actual,
            "P_heater_W": P_heater_avg,
            "P_cooler_W": P_cooler_avg,
            "P_total_W": P_total_actual,
            "m_dot_H2_kg_h": H2_avg,
            "c_elec_eur_h": c_elec_actual,
            "r_H2_eur_h": r_H2_actual,
            "c_deg_phys_eur_h": c_deg_phys,
            "c_shutdown_eur": c_shutdown_step,
            "true_profit_eur_h": r_H2_actual - c_elec_actual - c_deg_phys - c_shutdown_step,
            "t_step_s": time.perf_counter() - t_step_start,
            "Q_gen_W": Q_gen_avg,
            "Q_water_W": Q_water_avg,
            "Q_amb_W": Q_amb_avg,
            "Q_heater_W": Q_heater_avg,
            "eta_sys_HHV": (H2_avg * float(plant["fluids"]["hydrogen"]["HHV_kWh_per_kg"]) * 1000.0
                            / P_total_actual) if P_total_actual > 0 else 0.0,
        })

        # --- 7) Advance state ---
        T_stack = T_sub

    df = pd.DataFrame(records)
    df.attrs["t_wall_s"] = time.perf_counter() - t_wall_start
    df.attrs["n_shutdowns"] = n_shutdowns
    df.attrs["n_replacements"] = n_replacements
    df.attrs["c_shutdown_total_eur"] = (
        n_shutdowns * float(plant["economics"]["c_shutdown_eur"])
    )
    if substep_records:
        df.attrs["substep_df"] = pd.DataFrame(substep_records)
    return df