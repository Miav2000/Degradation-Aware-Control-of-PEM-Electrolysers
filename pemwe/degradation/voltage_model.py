from __future__ import annotations

"""
VOLTAGE-POLYNOMIAL DEGRADATION MODEL
======================================

Primary degradation driver: cumulative cell voltage penalty V_deg [V].

Rate law (Nezhadkhatami et al. 2026, shape from Tejera Stage 0->3, 168 h):
  dV_deg/dt = (alpha_V * j^2 + beta_V * j + gamma_V) * f_T(T)

where f_T(T) = exp(-Ea_eff/R * (1/T - 1/T_ref)) is the Arrhenius correction
relative to T_ref = 60 deg C.  At T = T_ref, f_T = 1.

The constant term gamma_V represents calendar/chemical degradation that
accumulates even at zero current (idle or shutdown).  Set gamma_V = 0 to
suppress it if only electrochemical degradation is desired.

Polynomial origin — two-step approach
--------------------------------------
  Shape source (Tejera 2024 / Nezhadkhatami 2026):
    Nezhadkhatami extracted the J-V degradation signal by subtracting the
    Tejera Stage 0 (BOL) from Stage 3 (after 3 × 168 h = 504 h accelerated
    testing at j = 0.092 A/cm^2) polarization curve:
      ΔV = 37.4·J^2 + 0.779·J + 0.0017  [V],  J in [A/cm^2]

  Magnitude calibration (Thummalacherla & Bhattacharya 2026):
    Tejera's experiment comprised two periods of constant 2 V operation and
    one period at 0.092 A/cm^2, each 168 h, yielding degradation rates well
    above those observed in dynamic commercial operation.  Coefficients are
    uniformly rescaled so that the rate at j = 2.0 A/cm^2, T = T_ref matches
    the representative dynamic operating rate of 9.6 µV/h reported by
    Thummalacherla & Bhattacharya (2026).  This preserves the observed
    current-density dependence while anchoring the magnitude to a value that
    reflects realistic dynamic operation, partially accounting for degradation
    mechanisms associated with load cycling that are not explicitly modelled.

    EOL criterion: V_deg_EOL = 100 mV (Nezhadkhatami et al. 2026, Table 1).
"""

import math
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class DegradationState:
    """
    Primary degradation state: cumulative voltage penalty [V].

    V_deg_V = 0 at BOL; reaches V_deg_EOL at end of life.
    Pass this value directly to electrochem_step() as V_deg_V.
    """
    V_deg_V: float


def initial_state(plant: Dict[str, Any]) -> DegradationState:
    """Return beginning-of-life state: V_deg = 0."""
    return DegradationState(V_deg_V=0.0)


def arrhenius_factor(T_K: float, plant: Dict[str, Any]) -> float:
    """
    Arrhenius temperature correction factor relative to T_ref.

    f_T(T) = exp(-Ea_eff / R_gas * (1/T - 1/T_ref))

    f_T = 1.0 at T = T_ref (60 deg C); increases above T_ref.

    Parameters
    ----------
    T_K   : float   Stack temperature [K]
    plant : dict

    Returns
    -------
    float  Dimensionless Arrhenius factor [-]
    """
    deg   = plant["degradation"]
    R_gas = float(plant["electrochemistry"]["R_J_per_molK"])
    Ea    = float(deg["Ea_eff_J_per_mol"])
    T_ref = float(deg["T_ref_K"])
    return math.exp(-Ea / R_gas * (1.0 / T_K - 1.0 / T_ref))


def vdeg_rate_V_per_h(
    j_A_per_m2: float,
    T_K: float,
    plant: Dict[str, Any],
) -> float:
    """
    Instantaneous voltage degradation rate [V/h].

    dV_deg/dt = (alpha_V * j^2 + beta_V * j + gamma_V) * f_T(T)

    The polynomial is clamped to j ≥ 0 so shutdown (j = 0) only accumulates
    the small constant term gamma_V (baseline chemical decay).

    Parameters
    ----------
    j_A_per_m2 : float   Current density [A/m^2]
    T_K        : float   Stack temperature [K]
    plant      : dict

    Returns
    -------
    float  Voltage degradation rate [V/h]
    """
    deg     = plant["degradation"]
    alpha_V = float(deg["alpha_V_m4_per_A2_h"])
    beta_V  = float(deg["beta_V_m2_per_A_h"])
    gamma_V = float(deg["gamma_V_per_h"])

    j = max(j_A_per_m2, 0.0)
    poly = alpha_V * j ** 2 + beta_V * j + gamma_V
    return poly * arrhenius_factor(T_K, plant)


def step_degradation(
    state: DegradationState,
    *,
    j_A_per_m2: float,
    T_K: float,
    dt_h: float,
    plant: Dict[str, Any],
) -> DegradationState:
    """
    Advance V_deg by one Euler forward step.

    Parameters
    ----------
    state      : DegradationState  Current state
    j_A_per_m2 : float             Applied current density [A/m^2]
    T_K        : float             Stack temperature [K]
    dt_h       : float             Time step [h]
    plant      : dict

    Returns
    -------
    DegradationState  Updated V_deg after dt_h hours.
    """
    dV = vdeg_rate_V_per_h(j_A_per_m2, T_K, plant) * dt_h
    return DegradationState(V_deg_V=state.V_deg_V + dV)
