from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .thermo import partial_pressures_saturated


"""
SPECIES BLOCK

Purpose:
  Mass-balance based species flows from Faraday rates + saturated cathode outlet assumption.
  Computes:
    - dry H2 and O2 production
    - stoichiometric water consumption
    - cathode water vapour content assuming saturation at T_stack
    - remaining liquid water out (for recirculation)

Inputs:
  - I_stack_A          [A]     stack current (= cell current in series stack)
  - T_stack_K          [K]     stack temperature
  - p_an_Pa            [Pa]    anode total pressure
  - p_cath_Pa          [Pa]    cathode total pressure
  - m_dot_w_in_kg_s    [kg/s]  water feed into the stack
  - plant              [-]     parameters (molar masses, eta_F, electrochemistry constants)

Outputs:
  - m_dot_H2_dry_kg_s          [kg/s]
  - m_dot_O2_kg_s              [kg/s]
  - m_dot_H2O_consumed_kg_s    [kg/s]
  - m_dot_H2O_vap_cath_kg_s    [kg/s]  water vapour leaving with H2 (cathode)
  - m_dot_H2O_vap_an_kg_s      [kg/s]  water vapour leaving with O2 (anode)
  - m_dot_liq_out_kg_s         [kg/s]  total liquid returned to mixing tank
  - m_dot_H2_mixed_kg_s        [kg/s]  (H2 + water vapour, pre-dryer)
  - y_H2O_cath                 [-]     vapour mole fraction at cathode outlet
  - y_H2O_an                   [-]     vapour mole fraction at anode outlet
  - p_sat_Pa                   [Pa]    saturation pressure at T_stack
"""


@dataclass(frozen=True)
class SpeciesOutputs:
    m_dot_H2_dry_kg_s: float
    m_dot_O2_kg_s: float

    m_dot_H2O_consumed_kg_s: float
    m_dot_H2O_vap_cath_kg_s: float   # vapour leaving with H2 → dryer
    m_dot_H2O_vap_an_kg_s: float     # vapour leaving with O2 → anode separator
    m_dot_liq_out_kg_s: float        # anode-side liquid back to recirculation tank

    m_dot_H2_mixed_kg_s: float

    y_H2O_cath: float
    y_H2O_an: float
    p_sat_Pa: float


def species_step(
    *,
    I_stack_A: float,
    T_stack_K: float,
    p_an_Pa: float,
    p_cath_Pa: float,
    m_dot_w_in_kg_s: float,
    plant: Dict[str, Any],
) -> SpeciesOutputs:
    ec = plant["electrochemistry"]
    mm = plant["fluids"]["molar_masses"]
    F = float(ec["F_C_per_mol"])
    n_e = float(ec["n_e"])
    N_cells = int(plant["stack"]["N_cells"])
    eta_F = float(plant["stack"]["eta_F"])
    M_H2 = float(mm["H2_kg_per_mol"])
    M_O2 = float(mm["O2_kg_per_mol"])
    M_H2O = float(mm["H2O_kg_per_mol"])

    # All species rates computed directly from Faraday's law and current.
    # Useful H2 collected on the cathode is reduced by eta_F.
    n_dot_Faraday = N_cells * float(I_stack_A) / (n_e * F)  # [mol/s] ideal rate
    n_dot_H2      = eta_F * n_dot_Faraday                    # [mol/s] actual H2
    n_dot_O2      = 0.5 * n_dot_Faraday                      # [mol/s] O2 (full current)
    n_dot_H2O_cons = n_dot_Faraday                            # [mol/s] water consumed (full current)
    m_dot_H2O_cons = n_dot_H2O_cons * M_H2O

    # Saturation-based partial pressures (single source of truth)
    pp = partial_pressures_saturated(
        T_stack_K=T_stack_K,
        p_an_Pa=p_an_Pa,
        p_cath_Pa=p_cath_Pa,
        plant=plant,
    )
    # --- Cathode outlet: H2 + water vapour (saturated) ---
    # y = n_vap / (n_H2 + n_vap)  →  n_vap = y/(1−y) * n_H2
    y_cath = pp.y_H2O_cath
    n_dot_H2O_vap_cath = (y_cath / (1.0 - y_cath)) * n_dot_H2
    m_dot_H2O_vap_cath = n_dot_H2O_vap_cath * M_H2O

    # --- Anode outlet: O2 + water vapour (saturated) ---
    y_an = pp.y_H2O_an
    n_dot_H2O_vap_an = (y_an / (1.0 - y_an)) * n_dot_O2
    m_dot_H2O_vap_an = n_dot_H2O_vap_an * M_H2O

    # --- Product masses ---
    m_dot_H2 = n_dot_H2 * M_H2
    m_dot_O2 = n_dot_O2 * M_O2

    m_dot_H2_mixed = m_dot_H2 + m_dot_H2O_vap_cath

    # Liquid return: water in minus consumed minus all vapour losses
    m_dot_liq_out = max(
        float(m_dot_w_in_kg_s) - m_dot_H2O_cons - m_dot_H2O_vap_cath - m_dot_H2O_vap_an,
        0.0,
    )

    return SpeciesOutputs(
        m_dot_H2_dry_kg_s=m_dot_H2,
        m_dot_O2_kg_s=m_dot_O2,
        m_dot_H2O_consumed_kg_s=m_dot_H2O_cons,
        m_dot_H2O_vap_cath_kg_s=m_dot_H2O_vap_cath,
        m_dot_H2O_vap_an_kg_s=m_dot_H2O_vap_an,
        m_dot_liq_out_kg_s=m_dot_liq_out,
        m_dot_H2_mixed_kg_s=m_dot_H2_mixed,
        y_H2O_cath=y_cath,
        y_H2O_an=y_an,
        p_sat_Pa=pp.p_sat_Pa,
    )