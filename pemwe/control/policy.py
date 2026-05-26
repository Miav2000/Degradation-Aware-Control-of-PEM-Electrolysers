from __future__ import annotations

"""
CONTROL POLICY
==============

High-level supervisory policy that wraps the (j, T_target) optimiser and
exposes a simple step() interface for the simulation loop.

Decision variables (set by optimiser)
-------------------------------------
  j_target   [A/m^2]   target current density
  T_target   [K]       target stack temperature

The shared fast layer then adjusts water flow to track T_target.

Important
---------
This policy is only the supervisory layer. The shared low-level fast layer,
plant dynamics, and safety logic are handled elsewhere so both controllers
can be compared fairly.
"""

from typing import Any, Dict, Tuple

from ..degradation import DegradationState
from .objective import CostTerms
from .optimizer import optimize_jT


class ControlPolicy:
    """
    Supervisory control policy.

    Returns
    -------
    (j_target, T_target, cost_terms)
    """

    def __init__(self, ctrl_cfg: Dict[str, Any], plant: Dict[str, Any]) -> None:
        self._cfg = ctrl_cfg
        self._plant = plant

        self._j_prev: float = 0.0
        self._T_prev: float = float(plant["thermal"]["T_amb_K"])
        self._m_prev: float = 0.0

    @property
    def j_prev(self) -> float:
        return self._j_prev

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
        Compute the optimal supervisory decision for this control interval.

        Parameters
        ----------
        deg_state : DegradationState
            Current degradation state
        P_avail_W : float
            Available electrical power [W]
        p_elec_eur_per_kWh : float
            Spot electricity price [EUR/kWh]
        dt_s : float
            Supervisory interval [s]
        T_stack_K : float
            Current measured stack temperature [K]

        Returns
        -------
        (j_target, T_target_K, CostTerms)
        """
        T_now = float(T_stack_K)

        j_opt, T_opt, cost = optimize_jT(
            j_prev_A_per_m2=self._j_prev,
            T_prev_target_K=self._T_prev,
            V_deg_V=deg_state.V_deg_V,
            p_elec_eur_per_kWh=float(p_elec_eur_per_kWh),
            P_avail_W=float(P_avail_W),
            dt_s=float(dt_s),
            plant=self._plant,
            ctrl_cfg=self._cfg,
            T_stack_K=T_now,
            m_prev_kg_s=self._m_prev,
        )

        # Optional voluntary shutdown: if operating this interval costs more than
        # the one-time shutdown penalty, choose j=0 (only when transitioning from on).
        if bool(self._cfg["objective"].get("include_shutdown", False)) and self._j_prev > 0.0:
            c_sd   = float(self._plant["economics"]["c_shutdown_eur"])
            dt_h   = float(dt_s) / 3600.0
            # Convert one-time EUR cost to an equivalent EUR/h rate for this interval
            if cost.total_eur_h > c_sd / dt_h:
                j_opt = 0.0
                T_opt = T_now

        self._j_prev = float(j_opt)
        self._T_prev = float(T_opt)
        self._m_prev = cost.m_dot_w_fin_kg_s

        return j_opt, T_opt, cost

    def reset(self, j_init: float = 0.0) -> None:
        """
        Reset internal supervisory memory.
        """
        self._j_prev = float(j_init)
        self._T_prev = float(self._plant["thermal"]["T_amb_K"])
        self._m_prev = 0.0