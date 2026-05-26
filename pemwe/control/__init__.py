from .objective import CostTerms, compute_cost_jT
from .optimizer import optimize_jT
from .policy import ControlPolicy
from .load_follower import LoadFollowerPolicy


def make_policy(ctrl_cfg, plant):
    """
    Factory: return the appropriate control policy.

    ctrl_cfg["type"]:
        "supervisory_optimizer"  -> degradation-aware or baseline optimizer
        "load_follower"          -> commercial-style rule-based controller
        "commercial_rule_based"  -> alias for load follower
    """
    ctrl_type = ctrl_cfg["type"]

    if ctrl_type in ("load_follower", "commercial_rule_based"):
        return LoadFollowerPolicy(ctrl_cfg, plant)

    return ControlPolicy(ctrl_cfg, plant)


__all__ = [
    "CostTerms",
    "compute_cost_jT",
    "optimize_jT",
    "ControlPolicy",
    "LoadFollowerPolicy",
    "make_policy",
]