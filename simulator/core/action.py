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

