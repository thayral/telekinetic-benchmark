from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


@dataclass(frozen=True)
class BoundingSphere:
    """Conservative 3D collision primitive used during scene sampling.

    The center is assumed to be the simulator/body position of the object or
    obstacle. The radius is derived from an explicit shape -> bounding-box-size
    mapping, then converted to a conservative sphere with:

        radius = 0.5 * ||size_xyz||_2

    where size_xyz is the full axis-aligned bounding box dimension.
    """

    name: str
    center: tuple[float, float, float]
    radius: float


class HasPrimitiveSpec(Protocol):
    shape: str
    size: tuple[float, ...]


def bounding_box_size_xyz(spec: HasPrimitiveSpec) -> tuple[float, float, float]:
    """Return full bounding-box dimensions for a primitive spec.

    The current specs use MuJoCo-style geom sizes:
    - box:      (half_x, half_y, half_z)
    - sphere:   (radius,)
    - cylinder: (radius, half_height)

    This function is intentionally explicit instead of shape-agnostic so new
    shapes fail clearly until their collision mapping is defined.
    """

    shape = spec.shape.lower()
    size = tuple(float(v) for v in spec.size)

    if shape == "box":
        if len(size) != 3:
            raise ValueError(f"Box spec requires 3 size values, got {size!r}.")
        hx, hy, hz = size
        return (2.0 * hx, 2.0 * hy, 2.0 * hz)

    if shape == "sphere":
        if len(size) != 1:
            raise ValueError(f"Sphere spec requires 1 size value, got {size!r}.")
        r = size[0]
        return (2.0 * r, 2.0 * r, 2.0 * r)

    if shape == "cylinder":
        if len(size) != 2:
            raise ValueError(f"Cylinder spec requires 2 size values, got {size!r}.")
        r, half_height = size
        return (2.0 * r, 2.0 * r, 2.0 * half_height)

    raise ValueError(f"No conservative-sphere mapping defined for shape {spec.shape!r}.")


def conservative_sphere_radius(spec: HasPrimitiveSpec) -> float:
    size_xyz = np.asarray(bounding_box_size_xyz(spec), dtype=float)
    return float(0.5 * np.linalg.norm(size_xyz, ord=np.inf))


def make_bounding_sphere(
    *,
    name: str,
    spec: HasPrimitiveSpec,
    center: tuple[float, float, float],
) -> BoundingSphere:
    return BoundingSphere(
        name=name,
        center=tuple(float(v) for v in center),
        radius=conservative_sphere_radius(spec),
    )


def spheres_overlap(a: BoundingSphere, b: BoundingSphere, margin: float = 0.0) -> bool:
    ca = np.asarray(a.center, dtype=float)
    cb = np.asarray(b.center, dtype=float)
    min_dist = float(a.radius + b.radius + margin)
    return float(np.linalg.norm(ca - cb)) < min_dist


def first_sphere_collision(
    candidate: BoundingSphere,
    existing: Iterable[BoundingSphere],
    *,
    margin: float = 0.0,
) -> tuple[str, str] | None:
    for other in existing:
        if spheres_overlap(candidate, other, margin=margin):
            return (candidate.name, other.name)
    return None


def all_sphere_collisions(
    spheres: Iterable[BoundingSphere],
    *,
    margin: float = 0.0,
) -> list[tuple[str, str]]:
    items = list(spheres)
    collisions: list[tuple[str, str]] = []
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            if spheres_overlap(a, b, margin=margin):
                collisions.append((a.name, b.name))
    return collisions
