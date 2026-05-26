from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class DryerOut:
    m_dot_H2_dry_kg_s: float
    m_dot_H2O_removed_kg_s: float
    m_dot_H2O_remaining_kg_s: float
    P_dryer_W: float


def dehumidifier_step(
    *,
    m_dot_H2_dry_kg_s: float,
    m_dot_H2O_vap_in_kg_s: float,
    T_stack_K: float,
    plant: Dict[str, Any],
) -> DryerOut:
    """
    The dehumidifier cools the cathode gas (H2 + H2O vapor) from T_stack to T_out, for water condensatation.
    The remaining vapor is saturated at T_out.

    Power includes:
      - Sensible cooling of the H2 gas: Q_sens = m_H2 * cp_H2 * (T_stack - T_out)
      - Latent heat of condensed water:  Q_lat  = m_condensed * h_fg
      - Total: P_dryer = (Q_sens + Q_lat) / COP
    """
    dryer = plant["auxiliaries"]["dryer"]
    p_cath = float(plant["stack"]["p_cath_Pa"])

    T_out = float(dryer["T_out_K"])
    COP = float(dryer["COP"])
    h_fg = float(dryer["h_fg_J_per_kg"])
    cp_H2 = float(dryer["cp_gas_J_per_kgK"])

    m_H2 = float(m_dot_H2_dry_kg_s)
    m_H2O_in = float(m_dot_H2O_vap_in_kg_s)

    # Saturation vapor pressure at dryer outlet temperature
    from .stack.thermo import p_sat_water_antoine_Pa
    p_sat_out = p_sat_water_antoine_Pa(T_out, plant)

    # H2O mole fraction at outlet (saturated at T_out, p_cath)
    y_H2O_out = p_sat_out / p_cath

    # Remaining H2O vapor at outlet (mass flow)
    # From mole fraction: m_H2O_out / M_H2O = y_H2O * (m_H2/M_H2 + m_H2O_out/M_H2O)
    # Solve: m_H2O_out = y_H2O * m_H2 * M_H2O / (M_H2 * (1 - y_H2O))
    M_H2 = float(plant["fluids"]["molar_masses"]["H2_kg_per_mol"])
    M_H2O = float(plant["fluids"]["molar_masses"]["H2O_kg_per_mol"])

    m_H2O_remaining = y_H2O_out * m_H2 * M_H2O / (M_H2 * (1.0 - y_H2O_out))
    m_H2O_remaining = min(m_H2O_remaining, m_H2O_in)
    m_condensed = m_H2O_in - m_H2O_remaining

    # Power: Sensible cooling of the incoming wet cathode gas, 
    # approximated using a single representative gas heat capacity.
    dT = max(float(T_stack_K) - T_out, 0.0)  # no cooling needed if stack is cold
    Q_sensible = (m_H2 + m_H2O_in) * cp_H2 * dT
    Q_latent = m_condensed * h_fg
    P_dryer = (Q_sensible + Q_latent) / COP

    return DryerOut(
        m_dot_H2_dry_kg_s=m_H2,
        m_dot_H2O_removed_kg_s=m_condensed,
        m_dot_H2O_remaining_kg_s=m_H2O_remaining,
        P_dryer_W=P_dryer,
    )
