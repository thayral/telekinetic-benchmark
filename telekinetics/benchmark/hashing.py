from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _canonicalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).round(8).tolist()
    if isinstance(value, np.floating):
        return round(float(value), 8)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {k: _canonicalize(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    if isinstance(value, tuple):
        return [_canonicalize(v) for v in value]
    if isinstance(value, float):
        return round(value, 8)
    return value


def canonicalize_for_hashing(value: Any) -> Any:
    """Return a JSON-stable representation suitable for hashing/storage comparisons."""
    return _canonicalize(value)


def compute_state_hash(scene_spec: dict[str, Any], checkpoint: dict[str, Any]) -> str:
    """Hash benchmark state identity as structural world + runtime simulator state."""
    payload = {
        "scene_spec": _canonicalize(scene_spec),
        "checkpoint": _canonicalize(checkpoint),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
