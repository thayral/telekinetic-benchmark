from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

import mujoco
import numpy as np

from telekinetics.benchmark.hashing import compute_state_hash


@dataclass(frozen=True)
class StateRecord:
    state_hash: str
    seed: int | None
    scene_name: str
    object_states: list[dict[str, Any]]
    checkpoint: dict[str, list]
    render_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))



def _as_serializable_array(x) -> list:
    return np.asarray(x, dtype=float).round(8).tolist()

def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=float).round(8).tolist()
    if isinstance(x, np.floating):
        return round(float(x), 8)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]
    if isinstance(x, float):
        return round(x, 8)
    return x

def capture_checkpoint(env) -> dict[str, list]:
    return {
        "qpos": _as_serializable_array(env.data.qpos),
        "qvel": _as_serializable_array(env.data.qvel),
        "mocap_pos": _as_serializable_array(env.data.mocap_pos),
        "mocap_quat": _as_serializable_array(env.data.mocap_quat),
    }



def capture_state(env, render_path: str | None = None, seed: int | None = None) -> StateRecord:
    object_states = to_jsonable(env.get_object_states())
    checkpoint = to_jsonable(capture_checkpoint(env))
    scene_name = type(env.scene).__name__
    state_hash = compute_state_hash(
        scene_name=scene_name,
        object_states=object_states,
        checkpoint=checkpoint,
    )
    return StateRecord(
        state_hash=state_hash,
        seed=seed,
        scene_name=scene_name,
        object_states=object_states,
        checkpoint=checkpoint,
        render_path=render_path,
    )



def restore_state(env, state: StateRecord | dict[str, Any]) -> None:
    checkpoint = state.checkpoint if isinstance(state, StateRecord) else state["checkpoint"]
    env.data.qpos[:] = np.asarray(checkpoint["qpos"], dtype=float)
    env.data.qvel[:] = np.asarray(checkpoint["qvel"], dtype=float)
    env.data.mocap_pos[:] = np.asarray(checkpoint["mocap_pos"], dtype=float)
    env.data.mocap_quat[:] = np.asarray(checkpoint["mocap_quat"], dtype=float)
    mujoco.mj_forward(env.model, env.data)
