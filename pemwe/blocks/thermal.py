from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict


"""
THERMAL DYNAMICS BLOCK

Purpose:
  Updates stack temperature using lumped energy balance:
    C_th dT/dt = Q_gen - Q_water - Q_amb

Inputs:
  - T_stack_K        [K]     current stack temperature
  - Q_gen_W          [W]     heat generation from stack (heat.py)
  - m_dot_w_in_kg_s  [kg/s]  water through stack for cooling
  - T_w_in_K         [K]     inlet water temperature
  - dt_s             [s]     timestep
  - plant            [-]     thermal + fluids.water + stack temperature bounds

Outputs:
  - T_stack_K [K]  updated temperature
  - Q_water_W  [W]  cooling extracted by water
  - Q_amb_W   [W]  ambient loss
"""


@dataclass(frozen=True)
class ThermalOut:
    T_stack_K: float
    Q_water_W: float
    Q_amb_W: float


def thermal_step(
    *,
    T_stack_K: float,
    Q_gen_W: float,
    m_dot_w_in_kg_s: float,
    T_w_in_K: float,
    dt_s: float,
    plant: Dict[str, Any],
) -> ThermalOut:
    from .properties import stack_UA_W_per_K
    th   = plant["thermal"]
    C_th = float(th["C_th_J_per_K"])
    UA = stack_UA_W_per_K(T_stack_K, plant)
    T_amb = float(th["T_amb_K"])

    cp_w = float(plant["fluids"]["water"]["cp_J_per_kgK"])

    # Water-side conductance: assumes perfect heat exchange (T_out = T_stack),
    K_water = float(m_dot_w_in_kg_s) * cp_w                 # [W/K]

    # Analytical (exponential) integration: T(t) = T_ss + (T0 - T_ss)*exp(-dt/tau)
    # Unconditionally stable for any dt, even when dt >> tau_thermal.
    K         = K_water + UA                                 # [W/K]
    T_ref_eff = (K_water * float(T_w_in_K) + UA * T_amb) / K
    T_ss_unc  = T_ref_eff + float(Q_gen_W) / K
    tau       = C_th / K
    alpha     = math.exp(-float(dt_s) / tau)
    T_next    = T_ss_unc + (float(T_stack_K) - T_ss_unc) * alpha

    # Heat flows evaluated at the mean temperature over the step
    T_mean = 0.5 * (float(T_stack_K) + T_next)
    Q_water = K_water * (T_mean - float(T_w_in_K))
    Q_amb  = UA * (T_mean - T_amb)

    return ThermalOut(T_stack_K=T_next, Q_water_W=Q_water, Q_amb_W=Q_amb)
