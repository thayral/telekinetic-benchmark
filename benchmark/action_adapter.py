from __future__ import annotations

from telekinetics.simulator.core.action import TelekineticAction

from telekinetics.benchmark.symbolic_actions import ActionInstance, action_delta_xy


def symbolic_to_telekinetic_action(action: ActionInstance) -> TelekineticAction:
    """Adapt a benchmark action into the simulator's low-level command."""
    if action.spec.action_type != "translate":
        raise ValueError(f"Unsupported action type: {action.spec.action_type}")
    return TelekineticAction(
        object_index=int(action.target_index),
        dxy=action_delta_xy(action.spec),
        frame=action.spec.frame,
        steps=int(action.rollout_steps),
        settle_steps=int(action.settle_steps),
    )
