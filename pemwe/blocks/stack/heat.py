from typing import Any, Dict


"""
HEAT GENERATION BLOCK

Purpose:
  Computes stack heat generation
    Q_gen = N_cells * I_stack * (V_cell - U_tn)

Inputs:
  - I_stack_A  [A]   stack current
  - V_cell_V   [V]   cell voltage
  - N_cells    [-]   number of cells in the stack 
  - plant      [-]   plant parameters (thermal.U_tn_V)

Outputs:
  - Q_gen_W    [W]   heat generation (positive = heat to remove)
"""


def heat_step(
    *,
    I_stack_A: float,
    V_cell_V: float,
    plant: Dict[str, Any],
) -> float:
    """Return Q_gen [W] — heat generation (positive = heat to remove)."""
    U_tn = float(plant["thermal"]["U_tn_V"])
    N_cells = int(plant["stack"]["N_cells"])
    return N_cells * float(I_stack_A) * (float(V_cell_V) - U_tn)
