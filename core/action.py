from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


class BaseActionInterface:
    def apply(self, env, action) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class TelekineticAction:
    """Low-level control command executed by an action interface."""

    object_index: int
    dxy: tuple[float, float]
    frame: str
    steps: int = 4
    settle_steps: int = 0


@dataclass(frozen=True)
class ActionSpec:
    """Symbolic benchmark action template.

    This is the action vocabulary that the dataset will store. Execution-specific
    details remain in the action interface.
    """

    action_id: str
    action_type: str
    frame: str
    direction: str
    magnitude: float
    label: str


@dataclass(frozen=True)
class ActionInstance:
    """A concrete symbolic action applied to a specific object in a scene."""

    target_index: int
    target_name: str
    spec: ActionSpec
    rollout_steps: int = 80
    settle_steps: int = 10

    @property
    def label(self) -> str:
        return f"{self.spec.label} {self.target_name}"


CAMERA_DIRECTION_TO_DXY: dict[str, tuple[float, float]] = {
    "right": (1.0, 0.0),
    "left": (-1.0, 0.0),
    "forward": (0.0, 1.0),
    "back": (0.0, -1.0),
}


def build_translation_action_library(
    magnitudes: Iterable[float] = (0.06,),
    frame: str = "camera",
) -> list[ActionSpec]:
    """Return a tiny fixed symbolic library for scene-conditioned translation.

    The returned actions are benchmark-level actions. They are intentionally
    simple and do not commit to a particular control backend.
    """

    if frame not in {"camera", "world"}:
        raise ValueError(f"Unsupported frame: {frame}")

    specs: list[ActionSpec] = []
    for magnitude in magnitudes:
        mag = float(magnitude)
        mag_tag = str(mag).replace(".", "p")
        for direction in ("forward", "back", "left", "right"):
            specs.append(
                ActionSpec(
                    action_id=f"translate_{frame}_{direction}_{mag_tag}",
                    action_type="translate",
                    frame=frame,
                    direction=direction,
                    magnitude=mag,
                    label=f"move {direction}",
                )
            )
    return specs


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


def sample_action_instance(
    rng,
    object_states: list[dict],
    action_library: list[ActionSpec],
    rollout_steps: int = 80,
    settle_steps: int = 10,
) -> ActionInstance:
    if not object_states:
        raise ValueError("Cannot sample an action without objects")
    if not action_library:
        raise ValueError("Action library is empty")

    obj = object_states[int(rng.integers(len(object_states)))]
    spec = action_library[int(rng.integers(len(action_library)))]
    return instantiate_action(
        spec=spec,
        target_index=obj["index"],
        target_name=obj["name"],
        rollout_steps=rollout_steps,
        settle_steps=settle_steps,
    )
