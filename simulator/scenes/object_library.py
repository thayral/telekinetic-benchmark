from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import copy


@dataclass(frozen=True)
class PrimitiveObjectSpec:
    """Simple object prototype used by tabletop scenes.

    The spec is intentionally small and simulator-oriented: enough to build MJCF,
    expose stable labels to the dataset later, and keep scene construction easy to
    extend.
    """

    spec_id: str
    shape: str
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float]
    display_name: str | None = None
    mass: float = 1.0

    def label(self) -> str:
        return self.display_name or self.spec_id


DEFAULT_OBJECT_LIBRARY: tuple[PrimitiveObjectSpec, ...] = (
    PrimitiveObjectSpec(
        spec_id="red_box_small",
        display_name="red box",
        shape="box",
        size=(0.03, 0.03, 0.03),
        rgba=(1.0, 0.1, 0.1, 1.0),
    ),
    PrimitiveObjectSpec(
        spec_id="blue_sphere_small",
        display_name="blue sphere",
        shape="sphere",
        size=(0.035,),
        rgba=(0.1, 0.6, 1.0, 1.0),
    ),
    PrimitiveObjectSpec(
        spec_id="green_cylinder_tall",
        display_name="green cylinder",
        shape="cylinder",
        size=(0.03, 0.05),
        rgba=(0.2, 0.9, 0.2, 1.0),
    ),
    PrimitiveObjectSpec(
        spec_id="yellow_box_wide",
        display_name="yellow box",
        shape="box",
        size=(0.045, 0.025, 0.025),
        rgba=(1.0, 0.85, 0.1, 1.0),
    ),
    PrimitiveObjectSpec(
        spec_id="purple_sphere_small",
        display_name="purple sphere",
        shape="sphere",
        size=(0.03,),
        rgba=(0.7, 0.2, 0.9, 1.0),
    ),
)


def default_object_library() -> list[PrimitiveObjectSpec]:
    return list(copy.deepcopy(DEFAULT_OBJECT_LIBRARY))



def sample_object_specs(rng, library: Iterable[PrimitiveObjectSpec], n_objects: int) -> list[PrimitiveObjectSpec]:
    """Sample a fixed-size scene object set from the library.

    Sampling is without replacement by default. If the requested number exceeds the
    library size, the library is cycled with replacement so the API stays simple.
    """

    lib = list(library)
    if not lib:
        raise ValueError("Object library must not be empty.")
    if n_objects <= 0:
        raise ValueError("n_objects must be positive.")

    if n_objects <= len(lib):
        indices = rng.choice(len(lib), size=n_objects, replace=False)
        return [copy.deepcopy(lib[int(i)]) for i in indices]

    out = []
    while len(out) < n_objects:
        take = min(len(lib), n_objects - len(out))
        indices = rng.choice(len(lib), size=take, replace=False)
        out.extend(copy.deepcopy(lib[int(i)]) for i in indices)
    return out
