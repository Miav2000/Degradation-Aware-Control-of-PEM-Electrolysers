from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

from .thermo import PartialPressures, partial_pressures_saturated


"""
ELECTROCHEMISTRY BLOCK

Purpose:
  Computes reversible voltage (temperature + Nernst pressure correction),
  activation + ohmic losses, and electrical power for the PEMWE stack.


Inputs:
  - j_A_per_m2     [A/m^2]  applied current density (control input)
  - T_stack_K      [K]      stack temperature
  - p_an_Pa        [Pa]     anode total pressure
  - p_cath_Pa      [Pa]     cathode total pressure
  - plant          [-]      plant parameter dictionary (from plant_parameters.yaml)

Outputs:
  - E_rev_V        [V]      reversible / OCV (Urev(T,1bar) + Nernst)
  - eta_act_V      [V]      activation overpotential
  - eta_ohm_V      [V]      ohmic overpotential
  - V_cell_fresh_V [V]      BOL cell voltage  (E_rev + eta_act + eta_ohm)
  - V_cell_V       [V]      actual cell voltage (V_cell_fresh + Delta_V_deg)
  - V_stack_V      [V]      stack voltage
  - I_stack_A      [A]      stack current
  - P_stack_W      [W]      stack electrical power
  Species production (H2, O2, H2O) is handled by species.py using I_stack.
  - pp             [-]      PartialPressures object
"""


@dataclass(frozen=True)
class ElectrochemOutputs:
    E_rev_V: float
    eta_act_V: float
    eta_ohm_V: float
    V_deg_V: float          # Degradation voltage penalty [V] (0 at BOL)

    V_cell_fresh_V: float   # BOL cell voltage: E_rev + eta_act + eta_ohm  [V]
    V_cell_V: float         # Actual cell voltage: V_cell_fresh + V_deg     [V]
    V_stack_V: float
    I_stack_A: float
    P_stack_W: float

    eta_voltage: float      # Voltage efficiency: E_rev / V_cell [-]
    eta_stack_HHV: float    # Stack HHV efficiency: U_tn_HHV / V_cell [-]

    pp: PartialPressures


def urev_T_1bar_V(T_K: float, plant: Dict[str, Any]) -> float:
    """
    Reversible voltage at 1 bar using correlation from plant_parameters.yaml:

      Urev(T) = a0 + a1*T + a2*T*ln(T) + a3*T^2   [V]
    """
    ec = plant["electrochemistry"]
    a0 = float(ec["Urev_a0_V"])
    a1 = float(ec["Urev_a1_V_per_K"])
    a2 = float(ec["Urev_a2_V_per_K"])
    a3 = float(ec["Urev_a3_V_per_K2"])
    return a0 + a1 * T_K + a2 * T_K * math.log(T_K) + a3 * (T_K**2)


def electrochem_step(
    *,
    j_A_per_m2: float,
    T_stack_K: float,
    p_an_Pa: float,
    p_cath_Pa: float,
    plant: Dict[str, Any],
    V_deg_V: float = 0.0,
) -> ElectrochemOutputs:
    """
    V_deg_V : float, optional
        Cumulative voltage degradation penalty [V] from the degradation model.
        Defaults to 0.0 (BOL, no ageing).  Pass DegradationState.V_deg_V here.
        V_cell = E_rev + eta_act(BOL j0) + eta_ohm(BOL R_area) + V_deg_V.
    """
    stack = plant["stack"]
    ec = plant["electrochemistry"]
    wf = plant["water_feed"]
    th = plant["thermal"]

    # Geometry
    N_cells = float(stack["N_cells"])
    A_cell = float(stack["A_cell_m2"])

    # Constants
    F = float(ec["F_C_per_mol"])
    R = float(ec["R_J_per_molK"])
    n_e = float(ec["n_e"])
    p_ref = float(stack["p_ref_Pa"])


    # Kinetics
    alpha_an = float(ec["alpha_an"])
    alpha_cat = float(ec["alpha_cat"])

    # Membrane properties for ohmic calculation
    mem_thickness_cm = float(th["mem_thickness_cm"])   # [cm]
    lam = float(wf["lambda"])                           # [-] membrane hydration number (Springer)

    # 1) Current
    # In a series stack: I_stack = j * A_cell (same current through all N_cells).
    I_stack = j_A_per_m2 * A_cell                    # [A]  cell current (= stack current)

    # 2) Partial pressures 
    pp = partial_pressures_saturated(
        T_stack_K=T_stack_K,
        p_an_Pa=p_an_Pa,
        p_cath_Pa=p_cath_Pa,
        plant=plant,
    )

    # 3) Reversible voltage (T + Nernst pressure correction)
    E_rev_1bar = urev_T_1bar_V(T_stack_K, plant)
    E_nernst = (R * T_stack_K) / (n_e * F) * math.log(
        (pp.p_H2_dry_Pa / p_ref) * math.sqrt(pp.p_O2_dry_Pa / p_ref)
    )
    E_rev = E_rev_1bar + E_nernst

    # 4) Exchange current density laws [A/m^2], T in [K]
    # Anode: Ir / IrO2-like OER
    j0_an = 1.055 * T_stack_K - 312.25

    # Cathode: Pt-like HER
    j0_cat = -5.46 * T_stack_K + 3327.0

    # 5) Activation loss (separate Tafel terms)
    if j_A_per_m2 == 0.0:
        eta_act = 0.0
    else:
        eta_act_an = (R * T_stack_K) / (alpha_an * F) * math.log(j_A_per_m2 / j0_an)
        eta_act_cat = (R * T_stack_K) / (alpha_cat * F) * math.log(j_A_per_m2 / j0_cat)
        eta_act = eta_act_an + eta_act_cat

    # 6) Ohmic loss
    # Membrane conductivity correlation assuming hydrated Nafion membrane.
    # sigma_mem is in [S/cm], membrane thickness in [cm] and
    # current density in [A/cm^2].
    sigma_mem_S_per_cm = (0.005139 * lam - 0.00326) * math.exp(
        1268.0 * (1.0 / 303.0 - 1.0 / T_stack_K)
    )

    j_A_per_cm2 = j_A_per_m2 / 1.0e4  # [A/cm^2]

    if j_A_per_m2 == 0.0:
        eta_ohm = 0.0
    else:
        eta_ohm = mem_thickness_cm * j_A_per_cm2 / sigma_mem_S_per_cm


    # 6) Voltages + power
    V_cell_fresh = E_rev + eta_act + eta_ohm          # BOL performance (no ageing)
    V_cell = V_cell_fresh + V_deg_V                    # Actual: BOL + degradation
    V_stack = N_cells * V_cell
    P_stack = V_stack * I_stack

    # 7) Efficiencies
    # Voltage efficiency: fraction of reversible voltage actually used
    eta_voltage = E_rev / V_cell
    # Stack HHV efficiency: thermoneutral voltage / cell voltage
    # eta > 1 possible below thermoneutral (endothermic operation)
    U_tn = float(plant["thermal"]["U_tn_V"])
    eta_stack_HHV = U_tn / V_cell

    return ElectrochemOutputs(
        E_rev_V=E_rev,
        eta_act_V=eta_act,
        eta_ohm_V=eta_ohm,
        V_deg_V=V_deg_V,
        V_cell_fresh_V=V_cell_fresh,
        V_cell_V=V_cell,
        V_stack_V=V_stack,
        I_stack_A=I_stack,
        P_stack_W=P_stack,
        eta_voltage=eta_voltage,
        eta_stack_HHV=eta_stack_HHV,
        pp=pp,
    )