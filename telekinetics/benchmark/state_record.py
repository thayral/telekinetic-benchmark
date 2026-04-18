from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import mujoco
import numpy as np

from telekinetics.benchmark.hashing import canonicalize_for_hashing, compute_state_hash
from telekinetics.simulator.scenes.tabletop_obstacles import restore_checkpoint_into_env



@dataclass(frozen=True)
class StateRecord:
    state_hash: str
    seed: int | None
    scene_name: str
    scene_spec: dict[str, Any]
    checkpoint: dict[str, Any]
    object_metadata: list[dict[str, Any]]
    render_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateRecord":
        return cls(
            state_hash=str(data["state_hash"]),
            seed=data.get("seed"),
            scene_name=str(data["scene_name"]),
            scene_spec=dict(data["scene_spec"]),
            checkpoint=dict(data["checkpoint"]),
            object_metadata=list(data.get("object_metadata", [])),
            render_path=data.get("render_path"),
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "StateRecord":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def _as_serializable_array(x: Any) -> list[Any]:
    return np.asarray(x, dtype=float).round(8).tolist()


def to_jsonable(x: Any) -> Any:
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


def capture_checkpoint(env) -> dict[str, list[Any]]:
    return {
        "qpos": _as_serializable_array(env.data.qpos),
        "qvel": _as_serializable_array(env.data.qvel),
        "mocap_pos": _as_serializable_array(env.data.mocap_pos),
        "mocap_quat": _as_serializable_array(env.data.mocap_quat),
    }


def capture_object_metadata(env) -> list[dict[str, Any]]:
    """Store symbolic object metadata without duplicating dynamic simulator state."""
    out: list[dict[str, Any]] = []
    for idx, name in enumerate(getattr(env.meta, "object_names", [])):
        record: dict[str, Any] = {"index": idx, "name": name}
        record.update(getattr(env.meta, "object_attributes", {}).get(name, {}))
        out.append(to_jsonable(record))
    return out


def _capture_scene_spec(env) -> dict[str, Any]:
    if not hasattr(env.scene, "to_scene_spec"):
        raise RuntimeError("Scene does not expose to_scene_spec(); cannot capture benchmark state.")

    scene_spec_obj = env.scene.to_scene_spec()
    if hasattr(scene_spec_obj, "to_dict"):
        return canonicalize_for_hashing(scene_spec_obj.to_dict())
    if isinstance(scene_spec_obj, dict):
        return canonicalize_for_hashing(scene_spec_obj)
    raise TypeError("Scene spec must be a dict or expose to_dict().")


def capture_state(env, render_path: str | None = None, seed: int | None = None) -> StateRecord:
    scene_spec = _capture_scene_spec(env)
    checkpoint = canonicalize_for_hashing(capture_checkpoint(env))
    object_metadata = capture_object_metadata(env)
    scene_name = type(env.scene).__name__
    state_hash = compute_state_hash(scene_spec=scene_spec, checkpoint=checkpoint)
    return StateRecord(
        state_hash=state_hash,
        seed=seed,
        scene_name=scene_name,
        scene_spec=scene_spec,
        checkpoint=checkpoint,
        object_metadata=object_metadata,
        render_path=render_path,
    )

def restore_state(env, state: StateRecord | dict[str, Any]) -> None:
    checkpoint = state.checkpoint if isinstance(state, StateRecord) else state["checkpoint"]
    restore_checkpoint_into_env(env, checkpoint)