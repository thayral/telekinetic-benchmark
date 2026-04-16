from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonicalize(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    if isinstance(value, tuple):
        return [_canonicalize(v) for v in value]
    if isinstance(value, float):
        return round(value, 8)
    return value



def compute_state_hash(scene_name: str, object_states: list[dict], checkpoint: dict) -> str:
    payload = {
        "scene_name": scene_name,
        "object_states": _canonicalize(object_states),
        "checkpoint": _canonicalize(checkpoint),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
