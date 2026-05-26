from __future__ import annotations

"""
LOAD-FOLLOWING COMMERCIAL CONTROLLER
===================================

Commercial-style supervisory controller:
  - fixed temperature target
  - simple load-following current target from available power
  - no degradation term in the decision logic
  - no preview-based feasibility search

Behavior
--------
  1. The simulation loop handles forced shutdown when P_avail < P_min.
  2. This policy assumes it is called only during normal operation.
  3. For the fixed T_target, it maps available power to a current target
     using a simple reference-voltage load-following rule.
  4. The shared fast layer and actual plant dynamics enforce the true
     realized operation and ensure P_total does not exceed P_avail.

This is intended to represent a practical commercial baseline:
simple supervisory logic, with low-level plant/PLC behavior handled
elsewhere.
"""

from typing import Any, Dict, Tuple

from ..degradation import DegradationState
from .objective import CostTerms, compute_cost_jT


class LoadFollowerPolicy:
    """
    Rule-based commercial PEMWE controller.

    Returns
    -------
    (j_target, T_target, cost_terms)
    """

    def __init__(self, ctrl_cfg: Dict[str, Any], plant: Dict[str, Any]) -> None:
        self._cfg = ctrl_cfg
        self._plant = plant
        self._j_prev: float = 0.0

    @property
    def j_prev(self) -> float:
        return self._j_prev

    def _j_from_available_power(
        self,
        *,
        P_avail_W: float,
        j_max: float,
    ) -> float:
        """
        Simple commercial load-following rule.

        Estimate supervisory current target from available power using a
        fixed reference cell voltage:

            j_cmd = P_avail / (N_cells * A_cell * V_ref)

        Auxiliary power margin is handled at the plant level (P_min includes
        aux margin, wind scaling includes aux margin). The fast layer's power
        budget enforcement ensures P_total does not exceed P_avail.
        """
        stack = self._plant["stack"]
        lf_cfg = self._cfg["load_following"]

        N_cells = float(stack["N_cells"])
        A_cell = float(stack["A_cell_m2"])
        V_ref_cell = float(lf_cfg["V_ref_cell_V"])

        j_cmd = float(P_avail_W) / (N_cells * A_cell * V_ref_cell)

        return min(float(j_cmd), float(j_max))

    def step(
        self,
        *,
        deg_state: DegradationState,
        P_avail_W: float,
        p_elec_eur_per_kWh: float,
        dt_s: float,
        T_stack_K: float,
    ) -> Tuple[float, float, CostTerms]:
        """
        Compute (j_target, T_target) for this control interval.

        Returns
        -------
        (j_target, T_target_K, CostTerms)
        """
        plant = self._plant
        cfg = self._cfg
        setpoints = cfg["setpoints"]
        stack_limits = plant["stack"]

        T_target_K = float(setpoints["T_target_K"])
        j_min = float(stack_limits["j_min_A_per_m2"])
        j_max = float(stack_limits["j_max_A_per_m2"])
        V_deg_V = deg_state.V_deg_V

        T_now = float(T_stack_K)

        # 1) Simple load-following current target from available power
        j_set = self._j_from_available_power(
            P_avail_W=float(P_avail_W),
            j_max=j_max,
        )

        # 2) Do not operate below j_min (gas crossover safety).
        #    Shutdown only happens when P_avail < P_min (handled by simulation loop).
        if 0.0 < j_set < j_min:
            j_set = j_min

        # 4) Cost terms for logging only
        # Commercial controller does not include degradation in the supervisory decision.
        dummy_ctrl = {
            "objective": {"include_degradation": False},
        }

        cost = compute_cost_jT(
            j_set,
            T_target_K,
            V_deg_V=V_deg_V,
            p_elec_eur_per_kWh=float(p_elec_eur_per_kWh),
            plant=plant,
            ctrl_cfg=dummy_ctrl,
            T_stack_K=T_now,
            dt_s=float(dt_s),
            P_avail_W=float(P_avail_W),
            m_prev_kg_s=0.0,
        )

        self._j_prev = float(j_set)

        return float(j_set), float(T_target_K), cost

    def reset(self, j_init: float = 0.0) -> None:
        self._j_prev = float(j_init)