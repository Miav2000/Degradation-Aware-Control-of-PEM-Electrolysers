from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


"""
WATER MANAGEMENT BLOCK (mix + cooler/heater)

Purpose:
  Models process water supply + recirculation conditioning for a PEMWE stack:
    1) Infinite tank provides cold make-up water (25°C).
    2) Returned anode-side liquid water is recirculated.
    3) Fresh + recirc are mixed to equilibrium temperature.
    4) If T_mix > T_set: cooler brings it down to T_set.
       If T_mix < T_set: heater brings it up to T_set.

Inputs:
  - m_dot_total_cmd_kg_s   [kg/s] desired total water flow to stack (control input)
  - m_dot_liq_return_kg_s  [kg/s] liquid water returning from anode side
  - T_liq_return_K         [K]    temperature of returned liquid (from stack/thermal)
  - plant                  [-]    parameters dictionary

Outputs:
  - m_dot_fresh_kg_s       [kg/s] cold make-up flow from tank
  - m_dot_recirc_kg_s      [kg/s] recirculated flow actually used
  - m_dot_stack_in_kg_s    [kg/s] total flow to stack
  - T_mix_K                [K]    equilibrium mix temperature (before cooler/heater)
  - T_stack_in_K           [K]    inlet temperature to stack (after cooler/heater)
  - Q_heater_W             [W]    thermal duty added by heater
  - P_heater_W             [W]    electrical power for heater
  - Q_cooler_W             [W]    thermal duty rejected by cooler
  - P_cooler_W             [W]    electrical power for cooler
"""


def min_water_feed_kg_s(j_A_per_m2: float, plant: Dict[str, Any]) -> float:
    """
    Minimum stoichiometric water feed for the full stack [kg/s].

        m_dot_min = lambda_min * N_cells * (j * A_cell) / (n_e * F) * M_H2O

    This is the lower bound on the commanded flow passed to water_management_step.
    lambda_min ensures sufficient water supply without starvation.
    """
    ec  = plant["electrochemistry"]
    wf  = plant["water_feed"]
    mm  = plant["fluids"]["molar_masses"]
    F        = float(ec["F_C_per_mol"])
    n_e      = float(ec["n_e"])
    N_cells  = int(plant["stack"]["N_cells"])
    A_cell   = float(plant["stack"]["A_cell_m2"])
    M_H2O    = float(mm["H2O_kg_per_mol"])
    lambda_min = float(wf["lambda_min"])
    n_dot_H2O = N_cells * (float(j_A_per_m2) * A_cell) / (n_e * F)
    return lambda_min * n_dot_H2O * M_H2O


@dataclass(frozen=True)
class WaterOut:
    m_dot_fresh_kg_s: float
    m_dot_recirc_kg_s: float
    m_dot_stack_in_kg_s: float
    T_mix_K: float              # equilibrium mix temperature (before cooler/heater)
    T_stack_in_K: float         # actual inlet temperature (after cooler/heater)
    Q_heater_W: float
    P_heater_W: float
    Q_cooler_W: float
    P_cooler_W: float


def water_management_step(
    *,
    m_dot_total_cmd_kg_s: float,
    m_dot_liq_return_kg_s: float,
    T_liq_return_K: float,
    plant: Dict[str, Any],
) -> WaterOut:
    # --- Parameters ---
    wf = plant["water_feed"]
    heater = plant["auxiliaries"]["heater"]

    cp_w = float(plant["fluids"]["water"]["cp_J_per_kgK"])

    T_tank = float(wf["T_tank_K"])
    T_set = float(wf["T_set_K"])

    # heater
    eta_h = float(heater["eta"])

    # cooler
    cooler_cfg = plant["auxiliaries"]["cooler"]
    COP_cooler = float(cooler_cfg["COP"])

    # --- 1) Requested total flow ---
    m_cmd = float(m_dot_total_cmd_kg_s)

    # --- 2) Recirc availability ---
    m_recirc = min(float(m_dot_liq_return_kg_s), m_cmd)
    m_fresh = m_cmd - m_recirc

    # --- 3) Mixing temperature ---
    m_tot = m_fresh + m_recirc
    T_mix = (m_fresh * T_tank + m_recirc * float(T_liq_return_K)) / m_tot

    # --- 4) Cooler + heater to reach setpoint ---
    if T_mix > T_set:
        Q_cooler = m_tot * cp_w * (T_mix - T_set)
        P_cooler = Q_cooler / COP_cooler
        Q_heater = 0.0
        P_heater = 0.0
    elif T_mix < T_set:
        Q_cooler = 0.0
        P_cooler = 0.0
        Q_heater = m_tot * cp_w * (T_set - T_mix)
        P_heater = Q_heater / eta_h
    else:
        Q_cooler = 0.0
        P_cooler = 0.0
        Q_heater = 0.0
        P_heater = 0.0

    return WaterOut(
        m_dot_fresh_kg_s=m_fresh,
        m_dot_recirc_kg_s=m_recirc,
        m_dot_stack_in_kg_s=m_tot,
        T_mix_K=T_mix,
        T_stack_in_K=T_set,
        Q_heater_W=Q_heater,
        P_heater_W=P_heater,
        Q_cooler_W=Q_cooler,
        P_cooler_W=P_cooler,
    )
