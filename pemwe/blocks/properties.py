"""
Thermophysical property functions.
"""
import math
from typing import Any, Dict


def water_density_kg_m3(T_K: float, plant: Dict[str, Any]) -> float:
    """Saturated liquid water density [kg/m^3] from DIPPR105 correlation.

    rho = A / B^(1 + (1 - T/C)^D)

    Parameters from Dortmund Data Bank (DDB), valid 273-648 K.
    """
    d = plant["fluids"]["water"]["rho_DIPPR105"]
    A = float(d["A"])
    B = float(d["B"])
    C = float(d["C"])
    D = float(d["D"])
    T = float(T_K)
    return A / B ** (1.0 + (1.0 - T / C) ** D)


def stack_UA_W_per_K(T_stack_K: float, plant: Dict[str, Any]) -> float:
    """Compute stack-to-ambient heat loss coefficient UA [W/K].

    UA = (h_conv + h_rad) * A_surface

    h_rad = 4 * emissivity * sigma * T_stack^3 (linearised radiation)
    A_surface estimated from stack geometry:
      cell face = sqrt(A_cell) x sqrt(A_cell)
      stack length = N_cells * cell_thickness
      A_surface = 2*A_cell + 4*sqrt(A_cell)*L_stack
    """
    th = plant["thermal"]
    stack = plant["stack"]

    h_conv = float(th["h_conv_W_per_m2K"])
    eps = float(th["emissivity"])
    sigma = float(th["sigma_W_per_m2K4"])
    d_cell = float(th["cell_thickness_m"])

    N_cells = int(stack["N_cells"])
    A_cell = float(stack["A_cell_m2"])

    # Stack outer surface area estimate
    side = math.sqrt(A_cell)              # cell is roughly square
    L_stack = N_cells * d_cell            # stack length
    A_surface = 2.0 * A_cell + 4.0 * side * L_stack  # 2 end faces + 4 sides

    # Heat transfer coefficients
    T = float(T_stack_K)
    h_rad = 4.0 * eps * sigma * T ** 3
    h_total = h_conv + h_rad

    return h_total * A_surface
