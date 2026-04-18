from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpec:
    """Benchmark-level symbolic action template.

    This is the semantic action vocabulary stored in the dataset. It remains
    independent from the low-level telekinetic backend.
    """

    action_id: str
    action_type: str
    frame: str
    direction: str
    magnitude: float
    label: str


@dataclass(frozen=True)
class ActionInstance:
    """Concrete symbolic action grounded to one scene object."""

    target_index: int
    target_name: str
    spec: ActionSpec
    rollout_steps: int = 80
    settle_steps: int = 10

    @property
    def text_label(self) -> str:
        return f"{self.spec.label} {self.target_name}"

    def to_dict(self) -> dict:
        return {
            "target_index": int(self.target_index),
            "target_name": str(self.target_name),
            "action_type": self.spec.action_type,
            "frame": self.spec.frame,
            "direction": self.spec.direction,
            "magnitude": float(self.spec.magnitude),
            "action_id": self.spec.action_id,
            "label": self.spec.label,
            "text_label": self.text_label,
            "rollout_steps": int(self.rollout_steps),
            "settle_steps": int(self.settle_steps),
        }


CAMERA_DIRECTION_TO_DXY: dict[str, tuple[float, float]] = {
    "right": (1.0, 0.0),
    "left": (-1.0, 0.0),
    "forward": (0.0, 1.0),
    "back": (0.0, -1.0),
}


def action_delta_xy(spec: ActionSpec) -> tuple[float, float]:
    if spec.action_type != "translate":
        raise ValueError(f"Unsupported action type: {spec.action_type}")
    try:
        unit_dxy = CAMERA_DIRECTION_TO_DXY[spec.direction]
    except KeyError as exc:
        raise ValueError(f"Unsupported direction: {spec.direction}") from exc
    return (spec.magnitude * unit_dxy[0], spec.magnitude * unit_dxy[1])


def instantiate_action(
    spec: ActionSpec,
    target_index: int,
    target_name: str,
    rollout_steps: int = 80,
    settle_steps: int = 10,
) -> ActionInstance:
    return ActionInstance(
        target_index=int(target_index),
        target_name=str(target_name),
        spec=spec,
        rollout_steps=int(rollout_steps),
        settle_steps=int(settle_steps),
    )
