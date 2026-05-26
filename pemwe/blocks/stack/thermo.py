from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class PartialPressures:
    """
    Partial pressures used consistently by electrochemistry (Nernst) and species (wet outlet).

    Both sides share T_stack, so p_H2O = p_sat on both sides.
    p_H2_dry and p_O2_dry are the remaining dry-gas partial pressures.

    Note: This is a *simple* assumption:
      - outlet gases are saturated (RH = 100%) at T_stack
      - p_dry = p_total - p_sat
    """
    p_sat_Pa: float

    p_H2_dry_Pa: float
    p_O2_dry_Pa: float

    y_H2O_cath: float
    y_H2O_an: float


def p_sat_water_antoine_Pa(T_K: float, plant: Dict[str, Any]) -> float:
    """
    Water saturation pressure via Antoine equation.

    Source: plant["fluids"]["water_vapor"]["antoine"]

    Uses:
      log10(P_bar) = A - B/(T_K + C)   [Stull, 1947; NIST Chemistry WebBook, SRD 69]
      valid 255.9-373 K
      p_Pa = P_bar * bar_to_Pa
    """
    ant = plant["fluids"]["water_vapor"]["antoine"]
    A = float(ant["A"])
    B = float(ant["B"])              # [K]
    C = float(ant["C"])              # [K]
    bar_to_Pa = float(ant["bar_to_Pa"])

    p_bar = 10.0 ** (A - B / (T_K + C))
    return p_bar * bar_to_Pa


def partial_pressures_saturated(
    *,
    T_stack_K: float,
    p_an_Pa: float,
    p_cath_Pa: float,
    plant: Dict[str, Any],
) -> PartialPressures:
    """
    Compute saturation-limited partial pressures for the anode and cathode.
    """
    p_sat = p_sat_water_antoine_Pa(T_stack_K, plant)

    if p_sat >= p_an_Pa:
        raise ValueError(f"p_sat ({p_sat:.0f} Pa) >= p_an ({p_an_Pa:.0f} Pa): saturated-vapor assumption invalid at anode, T={T_stack_K-273.15:.1f}°C")
    if p_sat >= p_cath_Pa:
        raise ValueError(f"p_sat ({p_sat:.0f} Pa) >= p_cath ({p_cath_Pa:.0f} Pa): saturated-vapor assumption invalid at cathode, T={T_stack_K-273.15:.1f}°C")

    p_H2_dry = p_cath_Pa - p_sat
    p_O2_dry = p_an_Pa   - p_sat

    y_H2O_cath = p_sat / p_cath_Pa
    y_H2O_an   = p_sat / p_an_Pa

    return PartialPressures(
        p_sat_Pa=p_sat,
        p_H2_dry_Pa=p_H2_dry,
        p_O2_dry_Pa=p_O2_dry,
        y_H2O_cath=y_H2O_cath,
        y_H2O_an=y_H2O_an,
    )