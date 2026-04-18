from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import copy


@dataclass(frozen=True)
class ObstacleSpec:
    """Prototype obstacle definition used by tabletop scenes."""

    spec_id: str
    shape: str = "box"
    size: tuple[float, ...] = (0.08, 0.03, 0.03)
    rgba: tuple[float, float, float, float] = (0.45, 0.45, 0.48, 1.0)
    display_name: str | None = None

    def label(self) -> str:
        return self.display_name or self.spec_id


@dataclass(frozen=True)
class SceneObstacle:
    """Concrete obstacle instance placed in one scene."""

    name: str
    spec_id: str
    shape: str
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float]
    pos: tuple[float, float, float]
    display_name: str | None = None

    def label(self) -> str:
        return self.display_name or self.spec_id

DEFAULT_OBSTACLE_LIBRARY: tuple[ObstacleSpec, ...] = (
    ObstacleSpec(
        spec_id="bar_small",
        shape="box",
        size=(0.08, 0.03, 0.03),
        rgba=(0.45, 0.45, 0.48, 1.0),
        display_name="grey bar",
    ),
    ObstacleSpec(
        spec_id="bar_long",
        shape="box",
        size=(0.12, 0.025, 0.03),
        rgba=(0.45, 0.45, 0.48, 1.0),
        display_name="grey long bar",
    ),
    ObstacleSpec(
        spec_id="block_square",
        shape="box",
        size=(0.05, 0.05, 0.04),
        rgba=(0.45, 0.45, 0.48, 1.0),
        display_name="grey block",
    ),
)


def default_obstacle_library() -> list[ObstacleSpec]:
    return list(copy.deepcopy(DEFAULT_OBSTACLE_LIBRARY))



def sample_obstacle_specs(
    rng,
    library: Iterable[ObstacleSpec],
    min_count: int,
    max_count: int,
) -> list[ObstacleSpec]:
    lib = list(library)
    if not lib:
        raise ValueError("Obstacle library must not be empty.")
    if min_count < 0 or max_count < min_count:
        raise ValueError("Invalid obstacle count range.")

    n_obstacles = int(rng.integers(min_count, max_count + 1))
    if n_obstacles == 0:
        return []

    if n_obstacles <= len(lib):
        indices = rng.choice(len(lib), size=n_obstacles, replace=False)
        return [copy.deepcopy(lib[int(i)]) for i in indices]

    out = []
    while len(out) < n_obstacles:
        take = min(len(lib), n_obstacles - len(out))
        indices = rng.choice(len(lib), size=take, replace=False)
        out.extend(copy.deepcopy(lib[int(i)]) for i in indices)
    return out
