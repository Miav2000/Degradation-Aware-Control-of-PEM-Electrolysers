from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def derive_thermal_capacity(plant: Dict[str, Any]) -> None:
    """Compute C_th_J_per_K from stack geometry and Tiktak reference, write into plant dict.

    C_th = C_th_ref * (N_cells / 100) * (A_cell_m2 * 1e4 / 1000)  [J/K]
    """
    th        = plant["thermal"]
    stk       = plant["stack"]
    C_th_ref  = float(th["C_th_ref_kJ_per_K"]) * 1e3   # → J/K
    N_cells   = float(stk["N_cells"])
    A_cell_m2 = float(stk["A_cell_m2"])
    th["C_th_J_per_K"] = C_th_ref * (N_cells / 100.0) * (A_cell_m2 * 1e4 / 1000.0)


def derive_degradation_coeffs(plant: Dict[str, Any]) -> None:
    """Compute alpha_V, beta_V, gamma_V from source params and write into plant dict.

    Called once at load time (and by sensitivity scripts after changing DDR_ref).
    shape_ref = a*j_ref^2 + b*j_ref + c  (J in A/cm^2)
    k         = DDR_ref [V/h] / shape_ref
    alpha_V   = k * a / 1e8   [V·m^4/(A^2 h)]
    beta_V    = k * b / 1e4   [V·m^2/(A h)]
    gamma_V   = k * c         [V/h]
    """
    deg = plant["degradation"]
    a       = float(deg["poly_a_V_per_Acm2_sq"])
    b       = float(deg["poly_b_V_per_Acm2"])
    c       = float(deg["poly_c_V"])
    DDR_ref = float(deg["DDR_ref_uV_per_h"]) * 1e-6   # → V/h
    j_ref   = float(deg["j_ref_A_per_cm2"])

    shape_ref = a * j_ref ** 2 + b * j_ref + c
    k = DDR_ref / shape_ref

    deg["alpha_V_m4_per_A2_h"] = k * a / 1e8
    deg["beta_V_m2_per_A_h"]   = k * b / 1e4
    deg["gamma_V_per_h"]       = k * c


def load_plant(path: Union[str, Path]) -> Dict[str, Any]:
    """Load plant parameters from a YAML file."""
    with open(path) as f:
        plant = yaml.safe_load(f)
    derive_thermal_capacity(plant)
    derive_degradation_coeffs(plant)
    return plant


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load any YAML config file (controller, scenario, etc.)."""
    with open(path) as f:
        return yaml.safe_load(f)
