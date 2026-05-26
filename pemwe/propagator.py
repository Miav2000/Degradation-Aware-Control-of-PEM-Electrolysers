from __future__ import annotations

"""
INTERVAL PROPAGATOR
===================

Shared one-interval plant simulation used by both:
  - the optimiser (to evaluate candidate operating points)
  - the closed-loop simulation (to advance the plant state)

This is the single source of truth for sub-step physics.
Both the optimiser preview and the simulation call this function,
ensuring consistency between planning and execution.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .blocks.stack.electrochemistry import electrochem_step
from .blocks.stack.heat import heat_step
from .blocks.stack.species import species_step
from .blocks.recirculation import water_management_step
from .blocks.thermal import thermal_step
from .blocks.dehumidifier import dehumidifier_step
from .blocks.auxiliaries import auxiliaries_step
from .control.fast_layer import fast_plc_step
from .degradation.voltage_model import vdeg_rate_V_per_h


@dataclass(frozen=True)
class IntervalResult:
    """Result of propagating the plant over one supervisory interval."""
    # Temperature
    T_fin_K: float
    T_avg_K: float
    T_max_K: float

    # Actuator averages
    j_avg_A_per_m2: float
    m_avg_kg_s: float
    m_fin_kg_s: float

    # Power averages
    P_stack_avg_W: float
    P_aux_avg_W: float
    P_total_avg_W: float
    P_total_max_W: float
    P_heater_avg_W: float
    P_cooler_avg_W: float

    # Production
    m_dot_H2_avg_kg_h: float

    # Degradation
    dV_deg_V: float

    # Heat flows
    Q_gen_avg_W: float
    Q_water_avg_W: float
    Q_amb_avg_W: float
    Q_heater_avg_W: float

    # Water temperatures (averages)
    T_mix_avg_K: float
    T_w_in_avg_K: float
    T_liq_return_avg_K: float

    # Water temperatures (end of interval — consistent with T_fin_K)
    T_mix_fin_K: float
    T_mix_min_K: float
    T_w_in_fin_K: float

    # Sub-step records (only populated if requested)
    substep_records: Optional[List[dict]] = None


def propagate_interval(
    *,
    j_target: float,
    T_target_K: float,
    is_on: bool,
    T_stack_K: float,
    P_avail_W: float,
    V_deg_V: float,
    plant: Dict[str, Any],
    m_prev_kg_s: float,
    dt_s: float,
    n_sub: int,
    log_substeps: bool = False,
) -> IntervalResult:
    """
    Propagate the plant over one supervisory interval.

    Parameters
    ----------
    j_target : float
        Supervisory current-density target [A/m^2]
    T_target_K : float
        Supervisory temperature target [K]
    is_on : bool
        Whether the stack should be running this interval
    T_stack_K : float
        Stack temperature at start of interval [K]
    P_avail_W : float
        Available electrical power [W]
    V_deg_V : float
        Accumulated degradation voltage [V]
    plant : dict
        Plant parameters
    m_prev_kg_s : float
        Previous water flow [kg/s]
    dt_s : float
        Total interval duration [s]
    n_sub : int
        Number of sub-steps
    log_substeps : bool
        If True, populate substep_records in the result

    Returns
    -------
    IntervalResult
    """
    stack_cfg = plant["stack"]
    p_an = float(stack_cfg["p_an_Pa"])
    p_cath = float(stack_cfg["p_cath_Pa"])

    dt_sub = dt_s / n_sub
    dt_sub_h = dt_sub / 3600.0
    dt_h = dt_s / 3600.0

    T_sub = float(T_stack_K)
    m_prev = float(m_prev_kg_s)

    if not is_on:
        m_prev = 0.0

    # Accumulators
    T_time_int = 0.0
    j_time_int = 0.0
    m_time_int = 0.0
    P_stack_int = 0.0
    P_aux_int = 0.0
    P_total_int = 0.0
    P_heater_int = 0.0
    P_cooler_int = 0.0
    Q_gen_int = 0.0
    Q_water_int = 0.0
    Q_amb_int = 0.0
    Q_heater_int = 0.0
    H2_int = 0.0
    dV_deg_sum = 0.0

    T_mix_int = 0.0
    T_w_in_int = 0.0
    T_liq_return_int = 0.0
    T_mix_last = float(T_stack_K)       # end-of-interval values
    T_mix_min = float(T_stack_K)
    T_w_in_last = float(T_stack_K)
    T_max = T_sub

    P_total_max = 0.0
    m_final = 0.0

    sub_records: Optional[List[dict]] = [] if log_substeps else None

    for i_sub in range(n_sub):
        # ==============================================================
        # SHUTDOWN MODE
        # ==============================================================
        if not is_on:
            j_sub = 0.0
            m_sub = 0.0

            th_sub = thermal_step(
                T_stack_K=T_sub,
                Q_gen_W=0.0,
                m_dot_w_in_kg_s=0.0,
                T_w_in_K=T_sub,
                dt_s=dt_sub,
                plant=plant,
            )

            T_mid = 0.5 * (T_sub + th_sub.T_stack_K)

            T_time_int += T_mid * dt_sub_h
            dV_deg_sum += vdeg_rate_V_per_h(0.0, T_mid, plant) * dt_sub_h

            Q_water_int += th_sub.Q_water_W * dt_sub_h
            Q_amb_int += th_sub.Q_amb_W * dt_sub_h

            T_mix_int += T_sub * dt_sub_h
            T_w_in_int += T_sub * dt_sub_h
            T_liq_return_int += T_sub * dt_sub_h

            if log_substeps:
                sub_records.append({
                    "t_sub": i_sub,
                    "j_A_per_m2": 0.0,
                    "m_dot_w_kg_s": 0.0,
                    "T_stack_K": th_sub.T_stack_K,
                    "T_w_in_K": T_sub,
                    "T_mix_K": T_sub,
                    "P_stack_W": 0.0,
                    "P_aux_W": 0.0,
                    "P_heater_W": 0.0,
                    "P_cooler_W": 0.0,
                    "P_total_W": 0.0,
                    "Q_gen_W": 0.0,
                    "Q_water_W": th_sub.Q_water_W,
                    "Q_amb_W": th_sub.Q_amb_W,
                    "m_dot_H2_kg_h": 0.0,
                })

            T_sub = th_sub.T_stack_K
            continue

        # ==============================================================
        # NORMAL ON MODE
        # ==============================================================
        j_sub, m_sub = fast_plc_step(
            j_target=float(j_target),
            T_target_K=float(T_target_K),
            T_stack_K=T_sub,
            P_avail_W=float(P_avail_W),
            V_deg_V=V_deg_V,
            plant=plant,
            m_prev_kg_s=m_prev,
            dt_s=dt_sub,
        )
        m_prev = m_sub

        ec_sub = electrochem_step(
            j_A_per_m2=j_sub,
            T_stack_K=T_sub,
            p_an_Pa=p_an,
            p_cath_Pa=p_cath,
            plant=plant,
            V_deg_V=V_deg_V,
        )

        sp_sub = species_step(
            I_stack_A=ec_sub.I_stack_A,
            T_stack_K=T_sub,
            p_an_Pa=p_an,
            p_cath_Pa=p_cath,
            m_dot_w_in_kg_s=m_sub,
            plant=plant,
        )

        wm_sub = water_management_step(
            m_dot_total_cmd_kg_s=m_sub,
            m_dot_liq_return_kg_s=sp_sub.m_dot_liq_out_kg_s,
            T_liq_return_K=T_sub,
            plant=plant,
        )

        heat_sub = heat_step(
            I_stack_A=ec_sub.I_stack_A,
            V_cell_V=ec_sub.V_cell_V,
            plant=plant,
        )

        th_sub = thermal_step(
            T_stack_K=T_sub,
            Q_gen_W=heat_sub,
            m_dot_w_in_kg_s=wm_sub.m_dot_stack_in_kg_s,
            T_w_in_K=wm_sub.T_stack_in_K,
            dt_s=dt_sub,
            plant=plant,
        )

        dryer_sub = dehumidifier_step(
            m_dot_H2_dry_kg_s=sp_sub.m_dot_H2_dry_kg_s,
            m_dot_H2O_vap_in_kg_s=sp_sub.m_dot_H2O_vap_cath_kg_s,
            T_stack_K=T_sub,
            plant=plant,
        )

        aux_sub = auxiliaries_step(
            m_dot_water_total_kg_s=wm_sub.m_dot_stack_in_kg_s,
            T_water_K=wm_sub.T_stack_in_K,
            P_dryer_W=dryer_sub.P_dryer_W,
            P_heater_W=wm_sub.P_heater_W,
            P_cooler_W=wm_sub.P_cooler_W,
            plant=plant,
        )

        P_aux_sub = aux_sub.P_aux_W
        P_total_sub = ec_sub.P_stack_W + P_aux_sub
        H2_sub_kg_h = sp_sub.m_dot_H2_dry_kg_s * 3600.0

        T_mid = 0.5 * (T_sub + th_sub.T_stack_K)

        # Time-weighted accumulation
        T_time_int += T_mid * dt_sub_h
        j_time_int += j_sub * dt_sub_h
        m_time_int += m_sub * dt_sub_h
        P_stack_int += ec_sub.P_stack_W * dt_sub_h
        P_aux_int += P_aux_sub * dt_sub_h
        P_total_int += P_total_sub * dt_sub_h
        P_heater_int += wm_sub.P_heater_W * dt_sub_h
        P_cooler_int += wm_sub.P_cooler_W * dt_sub_h
        Q_gen_int += heat_sub * dt_sub_h
        Q_water_int += th_sub.Q_water_W * dt_sub_h
        Q_amb_int += th_sub.Q_amb_W * dt_sub_h
        Q_heater_int += wm_sub.Q_heater_W * dt_sub_h
        H2_int += H2_sub_kg_h * dt_sub_h

        T_mix_int += wm_sub.T_mix_K * dt_sub_h
        T_mix_last = wm_sub.T_mix_K
        T_mix_min = min(T_mix_min, wm_sub.T_mix_K)
        T_w_in_last = wm_sub.T_stack_in_K
        T_w_in_int += wm_sub.T_stack_in_K * dt_sub_h
        T_liq_return_int += T_sub * dt_sub_h

        dV_deg_sum += vdeg_rate_V_per_h(j_sub, T_mid, plant) * dt_sub_h

        P_total_max = max(P_total_max, P_total_sub)
        T_max = max(T_max, T_sub, th_sub.T_stack_K)
        m_final = m_sub

        if log_substeps:
            sub_records.append({
                "t_sub": i_sub,
                "j_A_per_m2": j_sub,
                "m_dot_w_kg_s": m_sub,
                "T_stack_K": th_sub.T_stack_K,
                "T_w_in_K": wm_sub.T_stack_in_K,
                "T_mix_K": wm_sub.T_mix_K,
                "P_stack_W": ec_sub.P_stack_W,
                "P_aux_W": P_aux_sub,
                "P_heater_W": wm_sub.P_heater_W,
                "P_cooler_W": wm_sub.P_cooler_W,
                "P_total_W": P_total_sub,
                "Q_gen_W": heat_sub,
                "Q_water_W": th_sub.Q_water_W,
                "Q_amb_W": th_sub.Q_amb_W,
                "m_dot_H2_kg_h": H2_sub_kg_h,
            })

        T_sub = th_sub.T_stack_K

    # Compute time-averaged values
    return IntervalResult(
        T_fin_K=T_sub,
        T_avg_K=T_time_int / dt_h,
        T_max_K=T_max,
        j_avg_A_per_m2=j_time_int / dt_h,
        m_avg_kg_s=m_time_int / dt_h,
        m_fin_kg_s=m_final,
        P_stack_avg_W=P_stack_int / dt_h,
        P_aux_avg_W=P_aux_int / dt_h,
        P_total_avg_W=P_total_int / dt_h,
        P_total_max_W=P_total_max,
        P_heater_avg_W=P_heater_int / dt_h,
        P_cooler_avg_W=P_cooler_int / dt_h,
        m_dot_H2_avg_kg_h=H2_int / dt_h,
        dV_deg_V=dV_deg_sum,
        Q_gen_avg_W=Q_gen_int / dt_h,
        Q_water_avg_W=Q_water_int / dt_h,
        Q_amb_avg_W=Q_amb_int / dt_h,
        Q_heater_avg_W=Q_heater_int / dt_h,
        T_mix_avg_K=T_mix_int / dt_h,
        T_w_in_avg_K=T_w_in_int / dt_h,
        T_liq_return_avg_K=T_liq_return_int / dt_h,
        T_mix_fin_K=T_mix_last,
        T_mix_min_K=T_mix_min,
        T_w_in_fin_K=T_w_in_last,
        substep_records=sub_records,
    )
