from __future__ import annotations

"""
CONTROL OBJECTIVE
=================

Economic objective for the PEMWE supervisory controller.

For each candidate supervisory decision (j_target, T_target), the objective
predicts the realised response of the plant over one supervisory interval
using the same shared fast layer and plant blocks as the main simulation.

Objective:
    J_k = c_elec,k - r_H2,k + c_deg,k

where:
    c_elec     = p_elec [EUR/kWh] * P_total [kW]
    r_H2       = p_H2   [EUR/kg]  * m_dot_H2 [kg/h]
    c_deg      = C_stack [EUR]    * dV_deg / V_EOL

Shutdown cost is handled by the simulation loop only (not visible to optimiser).

Important:
  - The fast layer is shared by both controllers.
  - The only intended difference between controllers is whether degradation
    is included in the supervisory objective.
  - If j = 0, the preview treats the plant
    as fully shut down for that interval:
      * no electrolysis
      * no water circulation
      * no heater / cooler / pump / dryer
      * passive thermal drift only
      * small off-state degradation may still accumulate via gamma_V
"""

from dataclasses import dataclass
from typing import Any, Dict

from ..blocks.stack.electrochemistry import electrochem_step
from ..blocks.stack.species import species_step
from ..blocks.recirculation import water_management_step
from ..blocks.dehumidifier import dehumidifier_step
from ..blocks.auxiliaries import auxiliaries_step
from ..degradation.voltage_model import vdeg_rate_V_per_h


@dataclass(frozen=True)
class CostTerms:
    """
    Breakdown of economic terms [EUR/h] at the predicted realised operating point.
    """
    c_electricity_eur_h: float
    c_H2_revenue_eur_h: float
    c_degradation_eur_h: float
    c_degradation_phys_eur_h: float
    total_eur_h: float
    total_true_eur_h: float

    P_stack_W: float
    P_aux_W: float
    P_total_W: float
    P_total_max_W: float
    m_dot_H2_kg_h: float
    dV_deg_V: float
    T_stack_K: float
    m_dot_w_kg_s: float
    m_dot_w_fin_kg_s: float
    P_heat_W: float
    P_cooler_W: float


@dataclass(frozen=True)
class PreviewOutputs:
    """
    Predicted realised quantities over one supervisory interval.
    """
    T_avg_K: float
    T_fin_K: float
    j_avg_A_per_m2: float
    m_avg_kg_s: float

    P_stack_avg_W: float
    P_aux_avg_W: float
    P_total_avg_W: float
    P_total_max_W: float

    P_heat_avg_W: float
    P_cooler_avg_W: float
    m_dot_H2_avg_kg_h: float
    m_fin_kg_s: float

    dV_deg_V: float


def nominal_stack_power_W(plant: Dict[str, Any]) -> float:
    """Rated stack power [W] from plant parameters."""
    return float(plant["stack"]["P_rating_W"])


def _stack_capex_eur(plant: Dict[str, Any]) -> float:
    """
    Stack replacement CAPEX [EUR] from specific stack cost and rated power.
    """
    econ = plant["economics"]
    usd_per_kW = float(econ["capex_usd_per_kW"])
    eur_per_usd = float(econ["eur_per_usd"])
    P_rating_kW = float(plant["stack"]["P_rating_W"]) / 1000.0
    return usd_per_kW * P_rating_kW * eur_per_usd


def total_power_instant_W(
    j_A_per_m2: float,
    m_dot_w_kg_s: float,
    *,
    V_deg_V: float,
    plant: Dict[str, Any],
    T_stack_K: float,
) -> float:
    """
    Instantaneous total electrical power [W] at the current stack temperature.

    This is used by the fast layer for power-budget enforcement.
    """
    if float(j_A_per_m2) <= 0.0:
        return 0.0

    stack_cfg = plant["stack"]
    p_an = float(stack_cfg["p_an_Pa"])
    p_cath = float(stack_cfg["p_cath_Pa"])

    ec = electrochem_step(
        j_A_per_m2=float(j_A_per_m2),
        T_stack_K=float(T_stack_K),
        p_an_Pa=p_an,
        p_cath_Pa=p_cath,
        plant=plant,
        V_deg_V=V_deg_V,
    )

    sp = species_step(
        I_stack_A=ec.I_stack_A,
        T_stack_K=float(T_stack_K),
        p_an_Pa=p_an,
        p_cath_Pa=p_cath,
        m_dot_w_in_kg_s=float(m_dot_w_kg_s),
        plant=plant,
    )

    wm = water_management_step(
        m_dot_total_cmd_kg_s=float(m_dot_w_kg_s),
        m_dot_liq_return_kg_s=sp.m_dot_liq_out_kg_s,
        T_liq_return_K=float(T_stack_K),
        plant=plant,
    )

    dryer = dehumidifier_step(
        m_dot_H2_dry_kg_s=sp.m_dot_H2_dry_kg_s,
        m_dot_H2O_vap_in_kg_s=sp.m_dot_H2O_vap_cath_kg_s,
        T_stack_K=float(T_stack_K),
        plant=plant,
    )

    aux = auxiliaries_step(
        m_dot_water_total_kg_s=wm.m_dot_stack_in_kg_s,
        T_water_K=wm.T_stack_in_K,
        P_dryer_W=dryer.P_dryer_W,
        P_heater_W=wm.P_heater_W,
        P_cooler_W=wm.P_cooler_W,
        plant=plant,
    )

    return ec.P_stack_W + aux.P_aux_W


def preview_interval_jT(
    j_A_per_m2: float,
    T_target_K: float,
    *,
    V_deg_V: float,
    P_avail_W: float,
    plant: Dict[str, Any],
    T_stack_K: float,
    dt_s: float,
    m_prev_kg_s: float = 0.0,
) -> "PreviewOutputs":
    """
    Predict realised plant response over one supervisory interval.
    """
    from ..propagator import propagate_interval

    fc = plant["fast_control"]
    n_sub = int(fc["n_preview_substeps"])

    result = propagate_interval(
        j_target=float(j_A_per_m2),
        T_target_K=float(T_target_K),
        is_on=(float(j_A_per_m2) > 0.0),
        T_stack_K=float(T_stack_K),
        P_avail_W=float(P_avail_W),
        V_deg_V=V_deg_V,
        plant=plant,
        m_prev_kg_s=float(m_prev_kg_s),
        dt_s=float(dt_s),
        n_sub=n_sub,
    )

    return PreviewOutputs(
        T_avg_K=result.T_avg_K,
        T_fin_K=result.T_fin_K,
        j_avg_A_per_m2=result.j_avg_A_per_m2,
        m_avg_kg_s=result.m_avg_kg_s,
        P_stack_avg_W=result.P_stack_avg_W,
        P_aux_avg_W=result.P_aux_avg_W,
        P_total_avg_W=result.P_total_avg_W,
        P_total_max_W=result.P_total_max_W,
        P_heat_avg_W=result.P_heater_avg_W,
        P_cooler_avg_W=result.P_cooler_avg_W,
        m_dot_H2_avg_kg_h=result.m_dot_H2_avg_kg_h,
        m_fin_kg_s=result.m_fin_kg_s,
        dV_deg_V=result.dV_deg_V,
    )


def compute_cost_jT(
    j_A_per_m2: float,
    T_target_K: float,
    *,
    V_deg_V: float,
    p_elec_eur_per_kWh: float,
    plant: Dict[str, Any],
    ctrl_cfg: Dict[str, Any],
    T_stack_K: float,
    dt_s: float,
    P_avail_W: float = 1.0e30,
    m_prev_kg_s: float = 0.0,
) -> CostTerms:
    """
    Evaluate economic cost at candidate (j_target, T_target).

    Predicts the realised plant response over the interval using the shared
    fast layer and the actual dynamic plant blocks.

    RUL feedback (optional)
    -----------------------
    If ctrl_cfg["objective"]["use_rul_feedback"] is true, the degradation
    cost is weighted by the inverse of remaining useful life:
        c_deg = c_deg_phys * V_EOL / max(V_EOL - V_deg, V_EOL * rul_floor)
    At BOL (V_deg=0) this equals c_deg_phys. As V_deg approaches V_EOL the
    cost rises, making the optimizer increasingly conservative. This implements
    the diagnostic feedback path using the accumulated health state V_deg as the diagnostic signal.
    """
    obj_cfg = ctrl_cfg["objective"]
    include_deg = bool(obj_cfg["include_degradation"])
    use_rul_fb  = bool(obj_cfg.get("use_rul_feedback", False))
    rul_floor   = float(obj_cfg.get("rul_floor", 0.05))   # floor at 5% remaining
    
    # If True, Arrhenius factor f_T(T)=1 in the degradation cost term —
    # optimiser penalises j-dependent wear only, not temperature-driven wear.
    ignore_arrhenius = bool(obj_cfg.get("ignore_arrhenius_in_deg", False))

    econ = plant["economics"]
    capex_EUR = _stack_capex_eur(plant)
    p_H2 = float(econ["p_H2_eur_per_kg"])
    V_EOL = float(plant["degradation"]["V_deg_EOL_V"])
    dt_h = float(dt_s) / 3600.0

    preview = preview_interval_jT(
        j_A_per_m2=float(j_A_per_m2),
        T_target_K=float(T_target_K),
        V_deg_V=V_deg_V,
        P_avail_W=float(P_avail_W),
        plant=plant,
        T_stack_K=float(T_stack_K),
        dt_s=float(dt_s),
        m_prev_kg_s=float(m_prev_kg_s),
    )

    P_total_kW = preview.P_total_avg_W / 1000.0

    c_elec = float(p_elec_eur_per_kWh) * P_total_kW
    # H2 production already includes eta_F (applied in electrochem_step)
    c_H2_rev = p_H2 * preview.m_dot_H2_avg_kg_h

    if ignore_arrhenius:
        # Recompute degradation at the preview's average j but with Arrhenius = 1
        # (T = T_ref, f_T = 1). This removes temperature from the optimiser's
        # degradation signal entirely, matching the j-only degradation formulations
        # common in the literature.
        dV_deg_for_obj = vdeg_rate_V_per_h(
            preview.j_avg_A_per_m2,
            float(plant["degradation"]["T_ref_K"]),
            plant,
        ) * dt_h
    else:
        dV_deg_for_obj = preview.dV_deg_V

    c_deg_phys = capex_EUR * dV_deg_for_obj / V_EOL / dt_h

    if include_deg:
        if use_rul_fb:
            # RUL-weighted cost: penalty rises as remaining voltage budget shrinks.
            # V_remaining = how much degradation budget is left before EOL.
            # Multiplier = V_EOL / V_remaining = 1 at BOL, → 1/floor near EOL.
            V_remaining = max(V_EOL - float(V_deg_V), V_EOL * rul_floor)
            rul_multiplier = V_EOL / V_remaining
            c_deg = c_deg_phys * rul_multiplier
        else:
            c_deg = c_deg_phys
    else:
        c_deg = 0.0

    total = c_elec - c_H2_rev + c_deg
    total_true = c_elec - c_H2_rev + c_deg_phys

    return CostTerms(
        c_electricity_eur_h=c_elec,
        c_H2_revenue_eur_h=c_H2_rev,
        c_degradation_eur_h=c_deg,
        c_degradation_phys_eur_h=c_deg_phys,
        total_eur_h=total,
        total_true_eur_h=total_true,
        P_stack_W=preview.P_stack_avg_W,
        P_aux_W=preview.P_aux_avg_W,
        P_total_W=preview.P_total_avg_W,
        P_total_max_W=preview.P_total_max_W,
        m_dot_H2_kg_h=preview.m_dot_H2_avg_kg_h,
        dV_deg_V=preview.dV_deg_V,
        T_stack_K=preview.T_avg_K,
        m_dot_w_kg_s=preview.m_avg_kg_s,
        m_dot_w_fin_kg_s=preview.m_fin_kg_s,
        P_heat_W=preview.P_heat_avg_W,
        P_cooler_W=preview.P_cooler_avg_W,
    )