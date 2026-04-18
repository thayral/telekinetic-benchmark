from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import mujoco
import numpy as np

from telekinetics.simulator.core.scene import BaseScene, SceneMetadata
from telekinetics.simulator.scenes.object_library import (
    PrimitiveObjectSpec,
    default_object_library,
    sample_object_specs,
)
from telekinetics.simulator.scenes.obstacle_library import (
    ObstacleSpec,
    SceneObstacle,
    default_obstacle_library,
    sample_obstacle_specs,
)


@dataclass(frozen=True)
class TabletopObstacleSceneConfig:
    """Sampling policy for tabletop obstacle scenes.

    This is generation-time policy chosen by the caller. It is not the realized
    scene itself and should not be mutated by the scene.
    """

    n_objects: int = 4
    seed: int = 0
    plane_z: float = 0.53
    gravity_on: bool = False
    table_half_size: float = 0.55
    object_spawn_half_range: float = 0.20
    min_object_separation: float = 0.12
    object_library: list[PrimitiveObjectSpec] = field(default_factory=default_object_library)

    min_obstacles: int = 1
    max_obstacles: int = 2
    obstacle_spawn_half_range: float = 0.24
    min_obstacle_separation: float = 0.12
    obstacle_library: list[ObstacleSpec] = field(default_factory=default_obstacle_library)
    obstacle_rgba: tuple[float, float, float, float] = (0.45, 0.45, 0.48, 1.0)

    world_preset: str = "tabletop_obstacles_v1"
    scene_spec_version: int = 1


@dataclass(frozen=True)
class ObstaclePlacement:
    """Structural obstacle realization that belongs in the scene spec."""

    name: str
    spec: ObstacleSpec
    pos: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class TabletopObstacleSceneSpec:
    """Realized structural world description.

    This stores only structural information needed to rebuild the world. Dynamic
    object state remains in the MuJoCo checkpoint.
    """

    scene_type: str
    version: int
    scene_seed: int
    world_preset: str
    gravity_on: bool
    plane_z: float
    table_half_size: float
    object_specs: tuple[PrimitiveObjectSpec, ...]
    obstacles: tuple[ObstaclePlacement, ...]

    def to_dict(self) -> dict:
        return {
            "scene_type": self.scene_type,
            "version": self.version,
            "scene_seed": self.scene_seed,
            "world": {
                "preset": self.world_preset,
                "gravity_on": self.gravity_on,
                "plane_z": self.plane_z,
                "table_half_size": self.table_half_size,
            },
            "object_specs": [asdict(spec) for spec in self.object_specs],
            "obstacles": [
                {
                    "name": obs.name,
                    "spec": asdict(obs.spec),
                    "rgba": list(obs.rgba),
                    "pos": list(obs.pos),
                }
                for obs in self.obstacles
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TabletopObstacleSceneSpec":
        world = data.get("world", {})
        object_specs = tuple(
            PrimitiveObjectSpec(
                spec_id=item["spec_id"],
                shape=item["shape"],
                size=tuple(item["size"]),
                rgba=tuple(item["rgba"]),
                display_name=item.get("display_name"),
                mass=float(item.get("mass", 1.0)),
            )
            for item in data.get("object_specs", [])
        )
        obstacles = []
        for item in data.get("obstacles", []):
            spec_data = item["spec"]
            spec = ObstacleSpec(
                spec_id=spec_data["spec_id"],
                shape=spec_data.get("shape", "box"),
                size=tuple(spec_data.get("size", (0.08, 0.03, 0.03))),
                rgba=tuple(spec_data.get("rgba", (0.45, 0.45, 0.48, 1.0))),
                display_name=spec_data.get("display_name"),
            )
            obstacles.append(
                ObstaclePlacement(
                    name=item["name"],
                    spec=spec,
                    pos=tuple(item["pos"]),
                    rgba=tuple(item.get("rgba", spec.rgba)),
                )
            )
        return cls(
            scene_type=data.get("scene_type", "tabletop_obstacle"),
            version=int(data.get("version", 1)),
            scene_seed=int(data.get("scene_seed", 0)),
            world_preset=world.get("preset", "tabletop_obstacles_v1"),
            gravity_on=bool(world.get("gravity_on", False)),
            plane_z=float(world.get("plane_z", 0.53)),
            table_half_size=float(world.get("table_half_size", 0.55)),
            object_specs=object_specs,
            obstacles=tuple(obstacles),
        )


class TabletopObstacleSceneFactory:
    """Samples realized scene specs from a config."""

    def __init__(self, cfg: TabletopObstacleSceneConfig):
        self.cfg = cfg

    def create(self, seed: int | None = None) -> "TabletopObstacleScene":
        spec = self.sample_scene_spec(seed=seed)
        return TabletopObstacleScene(spec=spec, config=self.cfg)

    def sample_scene_spec(self, seed: int | None = None) -> TabletopObstacleSceneSpec:
        seed = self.cfg.seed if seed is None else int(seed)
        rng = np.random.default_rng(seed)

        object_specs = tuple(
            sample_object_specs(
                rng=rng,
                library=self.cfg.object_library,
                n_objects=self.cfg.n_objects,
            )
        )
        obstacle_specs = sample_obstacle_specs(
            rng=rng,
            library=self.cfg.obstacle_library,
            min_count=self.cfg.min_obstacles,
            max_count=self.cfg.max_obstacles,
        )
        obstacles = tuple(self._instantiate_obstacles(rng, obstacle_specs))

        return TabletopObstacleSceneSpec(
            scene_type="tabletop_obstacle",
            version=self.cfg.scene_spec_version,
            scene_seed=seed,
            world_preset=self.cfg.world_preset,
            gravity_on=self.cfg.gravity_on,
            plane_z=self.cfg.plane_z,
            table_half_size=self.cfg.table_half_size,
            object_specs=object_specs,
            obstacles=obstacles,
        )

    def reset_positions(self, scene: "TabletopObstacleScene", env, rng) -> None:
        """Resample dynamic object pose only inside an already-built world."""
        spawn = scene.config.object_spawn_half_range #if scene.config is not None else 0.30
        min_sep = scene.config.min_object_separation #if scene.config is not None else 0.12
        plane_z = scene.spec.plane_z
        placed_xy: list[tuple[float, float]] = []

        for spec, name in zip(scene.object_specs, scene._meta.object_names):
            jx = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_x")
            jy = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_y")
            jyaw = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_yaw")

            radius = self._object_planar_radius(spec)
            x, y = self._sample_spawn_xy(
                rng=rng,
                spawn=spawn,
                min_sep=min_sep,
                placed_xy=placed_xy,
                object_radius=radius,
                obstacles=scene.scene_obstacles,
            )
            placed_xy.append((x, y))

            env.data.qpos[env.model.jnt_qposadr[jx]] = x
            env.data.qpos[env.model.jnt_qposadr[jy]] = y
            env.data.qpos[env.model.jnt_qposadr[jyaw]] = float(rng.uniform(-0.2, 0.2))

        mujoco.mj_forward(env.model, env.data)

        for i, name in enumerate(scene._meta.object_names):
            body_id = env.object_body_ids[name]
            pos = env.data.xpos[body_id].copy()
            env.data.mocap_pos[i, 0] = pos[0]
            env.data.mocap_pos[i, 1] = pos[1]
            env.data.mocap_pos[i, 2] = plane_z

    def _instantiate_obstacles(self, rng, obstacle_specs: list[ObstacleSpec]) -> list[ObstaclePlacement]:
        placed_xy: list[tuple[float, float]] = []
        out: list[ObstaclePlacement] = []
        for i, spec in enumerate(obstacle_specs):
            x, y = self._sample_obstacle_xy(rng, placed_xy, spec)
            placed_xy.append((x, y))
            z = float(spec.size[-1]) if len(spec.size) >= 3 else 0.03
            out.append(
                ObstaclePlacement(
                    name=f"obstacle_{i}",
                    spec=spec,
                    pos=(x, y, z),
                    rgba=tuple(self.cfg.obstacle_rgba),
                )
            )
        return out

    def _sample_obstacle_xy(self, rng, placed_xy: list[tuple[float, float]], spec: ObstacleSpec) -> tuple[float, float]:
        hx = float(spec.size[0])
        hy = float(spec.size[1]) if len(spec.size) > 1 else hx
        limit = self.cfg.obstacle_spawn_half_range
        margin = 0.04
        lo_x = -(limit - hx - margin)
        hi_x = (limit - hx - margin)
        lo_y = -(limit - hy - margin)
        hi_y = (limit - hy - margin)
        if lo_x > hi_x or lo_y > hi_y:
            return 0.0, 0.0

        for _ in range(200):
            x = float(rng.uniform(lo_x, hi_x))
            y = float(rng.uniform(lo_y, hi_y))
            if all((x - px) ** 2 + (y - py) ** 2 >= self.cfg.min_obstacle_separation ** 2 for px, py in placed_xy):
                return x, y
        return float(rng.uniform(lo_x, hi_x)), float(rng.uniform(lo_y, hi_y))

    def _sample_spawn_xy(
        self,
        rng,
        spawn: float,
        min_sep: float,
        placed_xy: list[tuple[float, float]],
        object_radius: float,
        obstacles: list[SceneObstacle],
    ) -> tuple[float, float]:
        for _ in range(200):
            x = float(rng.uniform(-spawn, spawn))
            y = float(rng.uniform(-spawn, spawn))
            if not all((x - px) ** 2 + (y - py) ** 2 >= min_sep ** 2 for px, py in placed_xy):
                continue
            if self._blocked_by_obstacle(x, y, object_radius, obstacles):
                continue
            return x, y
        return float(rng.uniform(-spawn, spawn)), float(rng.uniform(-spawn, spawn))

    def _blocked_by_obstacle(self, x: float, y: float, radius: float, obstacles: list[SceneObstacle]) -> bool:
        for obs in obstacles:
            ox, oy, _ = obs.pos
            hx = float(obs.size[0])
            hy = float(obs.size[1]) if len(obs.size) > 1 else hx
            if abs(x - ox) <= (hx + radius + 0.01) and abs(y - oy) <= (hy + radius + 0.01):
                return True
        return False

    def _object_planar_radius(self, spec: PrimitiveObjectSpec) -> float:
        if spec.shape == "box":
            return float((spec.size[0] ** 2 + spec.size[1] ** 2) ** 0.5)
        if spec.shape in {"sphere", "cylinder"}:
            return float(spec.size[0])
        return 0.05


class TabletopObstacleScene(BaseScene):
    """Realized tabletop scene built from a structural scene spec."""

    def __init__(self, spec: TabletopObstacleSceneSpec, config: TabletopObstacleSceneConfig | None = None):
        self.spec = spec
        self.config = config
        self.object_specs = list(spec.object_specs)
        self.scene_obstacles = [
            SceneObstacle(
                name=obs.name,
                spec_id=obs.spec.spec_id,
                shape=obs.spec.shape,
                size=tuple(obs.spec.size),
                rgba=tuple(obs.rgba),
                pos=tuple(obs.pos),
                display_name=obs.spec.display_name,
            )
            for obs in spec.obstacles
        ]
        self._meta = SceneMetadata(
            object_names=[f"obj_{i}" for i in range(len(self.object_specs))],
            obstacle_names=[obs.name for obs in self.scene_obstacles],
            mocap_names=[f"mocap_{i}" for i in range(len(self.object_specs))],
            object_attributes={
                f"obj_{i}": {
                    "spec_id": spec.spec_id,
                    "label": spec.label(),
                    "shape": spec.shape,
                    "size": list(spec.size),
                    "rgba": list(spec.rgba),
                    "mass": spec.mass,
                }
                for i, spec in enumerate(self.object_specs)
            },
            obstacle_attributes={
                obs.name: {
                    "spec_id": obs.spec_id,
                    "label": obs.label(),
                    "shape": obs.shape,
                    "size": list(obs.size),
                    "rgba": list(obs.rgba),
                    "pos": list(obs.pos),
                }
                for obs in self.scene_obstacles
            },
            goal_geom_name="goal_region",
            plane_z=spec.plane_z,
        )

    def metadata(self):
        return self._meta

    def to_scene_spec(self) -> TabletopObstacleSceneSpec:
        return self.spec

    @classmethod
    def from_scene_spec(
        cls,
        scene_spec: TabletopObstacleSceneSpec | dict,
        config: TabletopObstacleSceneConfig | None = None,
    ) -> "TabletopObstacleScene":
        spec = scene_spec if isinstance(scene_spec, TabletopObstacleSceneSpec) else TabletopObstacleSceneSpec.from_dict(scene_spec)
        return cls(spec=spec, config=config)

    def save_scene_spec(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.spec.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_scene_spec(
        cls,
        path: str | Path,
        config: TabletopObstacleSceneConfig | None = None,
    ) -> "TabletopObstacleScene":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_scene_spec(data, config=config)

    def build_mjcf(self) -> str:
        if self.spec.world_preset != "tabletop_obstacles_v1":
            raise ValueError(f"Unsupported world preset: {self.spec.world_preset}")

        gravity = "0 0 -9.81" if self.spec.gravity_on else "0 0 0"
        h = self.spec.table_half_size

        object_bodies = []
        mocap_bodies = []
        welds = []
        for i, spec in enumerate(self.object_specs):
            obj_xml, mocap_xml, weld_xml = self._make_object_bundle(i, spec)
            object_bodies.append(obj_xml)
            mocap_bodies.append(mocap_xml)
            welds.append(weld_xml)

        obstacle_geoms = [self._make_obstacle_geom_xml(obs) for obs in self.scene_obstacles]

        return f"""
<mujoco model="tabletop_telekinesis_obstacles">
  <option timestep="0.002" gravity="{gravity}" integrator="RK4"/>
  <compiler angle="degree" inertiafromgeom="true"/>

  <default>
    <geom friction="0.9 0.05 0.001" solref="0.01 1" solimp="0.9 0.95 0.001" condim="4"/>
    <joint damping="1.0"/>
  </default>

  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>

  <worldbody>
    <light pos="0 0 2.0" dir="0 0 -1"/>
    <camera name="cam_oblique" pos="-0.009 -0.950 1.597" xyaxes="1.000 -0.000 0.000 0.000 0.754 0.657" fovy="25"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.95 0.95 0.95 1"/>

    <body name="table" pos="0 0 0.45">
      <geom name="table_geom" type="box" size="{h} {h} 0.05" rgba="0.75 0.75 0.78 1"/>

      <geom name="wall_north" type="box" pos="0 {h} 0.08" size="{h} 0.01 0.08" rgba="0.35 0.35 0.35 1"/>
      <geom name="wall_south" type="box" pos="0 -{h} 0.08" size="{h} 0.01 0.08" rgba="0.35 0.35 0.35 1"/>
      <geom name="wall_east"  type="box" pos="{h} 0 0.08" size="0.01 {h} 0.08" rgba="0.35 0.35 0.35 1"/>
      <geom name="wall_west"  type="box" pos="-{h} 0 0.08" size="0.01 {h} 0.08" rgba="0.35 0.35 0.35 1"/>

      {''.join(obstacle_geoms)}

      <geom name="goal_region" type="box" pos="0.25 0.25 0.002" size="0.06 0.06 0.002"
            rgba="0.15 0.85 0.20 0.35" contype="0" conaffinity="0"/>
    </body>

    {''.join(object_bodies)}
    {''.join(mocap_bodies)}
  </worldbody>

  <equality>
    {''.join(welds)}
  </equality>
</mujoco>
"""

    def reset_layout(self, env, rng) -> None:
        """Only resample dynamic object positions. Structural changes require a new scene."""
        if self.config is None:
            raise RuntimeError("Scene.reset_layout requires a config-backed scene.")
        TabletopObstacleSceneFactory(self.config).reset_positions(self, env, rng)

    def _make_object_bundle(self, index: int, spec: PrimitiveObjectSpec):
        name = f"obj_{index}"
        mocap_name = f"mocap_{index}"

        shape = spec.shape
        rgba = " ".join(map(str, spec.rgba))
        size = " ".join(map(str, spec.size))

        x0 = 0.0
        y0 = 0.0
        z0 = self.spec.plane_z
        table_half_size = self.config.table_half_size if self.config is not None else self.spec.table_half_size
        slide_limit = table_half_size - 0.07

        geom_xml = f'<geom name="{name}_geom" type="{shape}" size="{size}" rgba="{rgba}" mass="{spec.mass}"/>'

        object_xml = f"""
    <body name="{name}" pos="{x0} {y0} {z0}">
      <joint name="{name}_x" type="slide" axis="1 0 0" range="-{slide_limit} {slide_limit}"/>
      <joint name="{name}_y" type="slide" axis="0 1 0" range="-{slide_limit} {slide_limit}"/>
      <joint name="{name}_yaw" type="hinge" axis="0 0 1" range="-180 180"/>
      {geom_xml}
    </body>
"""
        mocap_xml = f"""
    <body name="{mocap_name}" mocap="true" pos="{x0} {y0} {z0}">
      <geom type="sphere" size="0.015" rgba="0 0 0 0.22" contype="0" conaffinity="0"/>
    </body>
"""
        weld_xml = f'<weld body1="{mocap_name}" body2="{name}" solref="0.04 1"/>\n'
        return object_xml, mocap_xml, weld_xml

    def _make_obstacle_geom_xml(self, obstacle: SceneObstacle) -> str:
        pos = " ".join(map(str, obstacle.pos))
        size = " ".join(map(str, obstacle.size))
        rgba = " ".join(map(str, obstacle.rgba))
        return (
            f'<geom name="{obstacle.name}" '
            f'type="{obstacle.shape}" '
            f'pos="{pos}" '
            f'size="{size}" '
            f'rgba="{rgba}"/>'
        )
    

    
def build_scene_from_scene_spec(
    scene_spec: TabletopObstacleSceneSpec | dict,
    config: TabletopObstacleSceneConfig | None = None,
) -> TabletopObstacleScene:
    """Canonical structural reconstruction path from serialized scene spec."""
    return TabletopObstacleScene.from_scene_spec(scene_spec, config=config)


def build_env_from_scene(
    scene: TabletopObstacleScene,
    *,
    env_cls=None,
    action_interface=None,
    observation_provider=None,
):
    """Build a runtime env from any realized scene.

    Freshly sampled scenes and dataset-loaded scenes should both go through this
    same helper so env wiring stays consistent.
    """
    if env_cls is None:
        from telekinetics.simulator.core.env import TelekinesisEnv as env_cls
    if action_interface is None:
        from telekinetics.simulator.control.telekinesis import TelekinesisActionInterface
        action_interface = TelekinesisActionInterface()
    if observation_provider is None:
        from telekinetics.simulator.observations.oracle import OracleSceneObservation
        observation_provider = OracleSceneObservation()

    return env_cls(
        scene=scene,
        action_interface=action_interface,
        observation_provider=observation_provider,
    )


def restore_checkpoint_into_env(env, checkpoint: dict) -> None:
    """Apply a saved MuJoCo checkpoint into an already-built env."""
    env.data.qpos[:] = np.asarray(checkpoint["qpos"], dtype=float)
    env.data.qvel[:] = np.asarray(checkpoint["qvel"], dtype=float)
    env.data.mocap_pos[:] = np.asarray(checkpoint["mocap_pos"], dtype=float)
    env.data.mocap_quat[:] = np.asarray(checkpoint["mocap_quat"], dtype=float)
    mujoco.mj_forward(env.model, env.data)


def build_env_from_scene_spec(
    scene_spec: TabletopObstacleSceneSpec | dict,
    *,
    config: TabletopObstacleSceneConfig | None = None,
    env_cls=None,
    action_interface=None,
    observation_provider=None,
):
    """Canonical path for dataset scene reload without runtime restore."""
    scene = build_scene_from_scene_spec(scene_spec, config=config)
    return build_env_from_scene(
        scene,
        env_cls=env_cls,
        action_interface=action_interface,
        observation_provider=observation_provider,
    )


def build_env_from_state_record(
    state_record,
    *,
    config: TabletopObstacleSceneConfig | None = None,
    env_cls=None,
    action_interface=None,
    observation_provider=None,
):
    """Canonical full restore path: rebuild structural world, then restore checkpoint."""
    scene_spec = state_record.scene_spec if hasattr(state_record, "scene_spec") else state_record["scene_spec"]
    checkpoint = state_record.checkpoint if hasattr(state_record, "checkpoint") else state_record["checkpoint"]
    env = build_env_from_scene_spec(
        scene_spec,
        config=config,
        env_cls=env_cls,
        action_interface=action_interface,
        observation_provider=observation_provider,
    )
    restore_checkpoint_into_env(env, checkpoint)
    return env

