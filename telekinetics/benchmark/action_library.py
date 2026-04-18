from __future__ import annotations

from typing import Iterable

from telekinetics.benchmark.symbolic_actions import ActionSpec, instantiate_action, ActionInstance


def build_translation_action_library(
    magnitudes: Iterable[float] = (0.06,),
    frame: str = "camera",
) -> list[ActionSpec]:
    """Return a small fixed symbolic action library for benchmark generation."""

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


def sample_action_instance(
    rng,
    object_states: list[dict],
    action_library: list[ActionSpec],
    rollout_steps: int = 80,
    settle_steps: int = 10,
    target_name_key: str = "label",
) -> ActionInstance:
    if not object_states:
        raise ValueError("Cannot sample an action without objects")
    if not action_library:
        raise ValueError("Action library is empty")

    obj = object_states[int(rng.integers(len(object_states)))]
    spec = action_library[int(rng.integers(len(action_library)))]
    target_name = obj.get(target_name_key) or obj.get("name") or f"object_{obj['index']}"
    return instantiate_action(
        spec=spec,
        target_index=obj["index"],
        target_name=target_name,
        rollout_steps=rollout_steps,
        settle_steps=settle_steps,
    )
