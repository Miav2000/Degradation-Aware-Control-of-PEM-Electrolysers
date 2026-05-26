from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .properties import water_density_kg_m3


"""
AUXILIARIES BLOCK

Purpose:
  Computes all auxiliary electrical loads: pump + dehumidifier + heater + cooler.

Inputs:
  - m_dot_water_total_kg_s  [kg/s] total water flow being pumped
  - T_water_K               [K]    water temperature (for density calculation)
  - P_dryer_W               [W]    dryer power (from dehumidifier block)
  - P_heater_W              [W]    heater power (from recirculation block)
  - P_cooler_W              [W]    cooler power (from recirculation block)
  - plant                   [-]    auxiliaries.pump + fluids.water

Outputs:
  - P_pump_W    [W]
  - P_dryer_W   [W]
  - P_heater_W  [W]
  - P_cooler_W  [W]
  - P_aux_W     [W]  total aux power (pump + dehumidifier + heater + cooler)
"""


@dataclass(frozen=True)
class AuxOut:
    P_pump_W: float
    P_dryer_W: float
    P_heater_W: float
    P_cooler_W: float
    P_aux_W: float


def pump_power_W(*, m_dot_kg_s: float, dp_Pa: float, eta: float, rho_kg_m3: float) -> float:
    V_dot = float(m_dot_kg_s) / float(rho_kg_m3)
    return float(dp_Pa) * V_dot / float(eta)


def auxiliaries_step(
    *,
    m_dot_water_total_kg_s: float,
    T_water_K: float,
    P_dryer_W: float,
    P_heater_W: float,
    P_cooler_W: float,
    plant: Dict[str, Any],
) -> AuxOut:
    pump = plant["auxiliaries"]["pump"]
    rho = water_density_kg_m3(T_water_K, plant)

    eta = float(pump["eta"])
    dp = float(pump["dp_Pa"])

    P_pump = pump_power_W(m_dot_kg_s=m_dot_water_total_kg_s, dp_Pa=dp, eta=eta, rho_kg_m3=rho)

    return AuxOut(
        P_pump_W=P_pump,
        P_dryer_W=float(P_dryer_W),
        P_heater_W=float(P_heater_W),
        P_cooler_W=float(P_cooler_W),
        P_aux_W=P_pump + float(P_dryer_W) + float(P_heater_W) + float(P_cooler_W),
    )
