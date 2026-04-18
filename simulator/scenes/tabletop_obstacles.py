from __future__ import annotations

from dataclasses import dataclass, field, replace
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


@dataclass
class TabletopObstacleSceneConfig:
    n_objects: int = 4
    seed: int = 0
    plane_z: float = 0.53
    gravity_on: bool = False
    table_half_size: float = 0.55
    object_spawn_half_range: float = 0.30
    min_object_separation: float = 0.12
    object_library: list[PrimitiveObjectSpec] = field(default_factory=default_object_library)
    object_specs: list[PrimitiveObjectSpec] | None = None

    min_obstacles: int = 0
    max_obstacles: int = 2
    obstacle_spawn_half_range: float = 0.24
    min_obstacle_separation: float = 0.12
    obstacle_library: list[ObstacleSpec] = field(default_factory=default_obstacle_library)
    obstacle_specs: list[ObstacleSpec] | None = None
    obstacle_rgba: tuple[float, float, float, float] = (0.45, 0.45, 0.48, 1.0)


class TabletopObstacleSceneFactory:
    """Owns scene sampling and partial re-sampling policy."""

    def __init__(self, cfg: TabletopObstacleSceneConfig):
        self.cfg = cfg

    def create(self, seed: int | None = None) -> "TabletopObstacleScene":
        seed = self.cfg.seed if seed is None else int(seed)
        rng = np.random.default_rng(seed)

        object_specs = list(self.cfg.object_specs) if self.cfg.object_specs is not None else sample_object_specs(
            rng=rng,
            library=self.cfg.object_library,
            n_objects=self.cfg.n_objects,
        )
        obstacle_specs = list(self.cfg.obstacle_specs) if self.cfg.obstacle_specs is not None else sample_obstacle_specs(
            rng=rng,
            library=self.cfg.obstacle_library,
            min_count=self.cfg.min_obstacles,
            max_count=self.cfg.max_obstacles,
        )
        scene_obstacles = self._instantiate_obstacles(rng, obstacle_specs)
        scene_cfg = replace(
            self.cfg,
            seed=seed,
            object_specs=list(object_specs),
            obstacle_specs=list(obstacle_specs),
        )
        return TabletopObstacleScene(
            cfg=scene_cfg,
            object_specs=object_specs,
            scene_obstacles=scene_obstacles,
        )

    def resample_object_specs(self, scene: "TabletopObstacleScene", seed: int | None = None) -> "TabletopObstacleScene":
        seed = scene.cfg.seed if seed is None else int(seed)
        rng = np.random.default_rng(seed)
        object_specs = sample_object_specs(
            rng=rng,
            library=scene.cfg.object_library,
            n_objects=scene.cfg.n_objects,
        )
        scene_cfg = replace(
            scene.cfg,
            seed=seed,
            object_specs=list(object_specs),
            obstacle_specs=list(scene.obstacle_specs),
        )
        return TabletopObstacleScene(
            cfg=scene_cfg,
            object_specs=object_specs,
            scene_obstacles=list(scene.scene_obstacles),
            factory=self,
        )

    def resample_environment_specs(self, scene: "TabletopObstacleScene", seed: int | None = None) -> "TabletopObstacleScene":
        seed = scene.cfg.seed if seed is None else int(seed)
        rng = np.random.default_rng(seed)
        obstacle_specs = sample_obstacle_specs(
            rng=rng,
            library=scene.cfg.obstacle_library,
            min_count=scene.cfg.min_obstacles,
            max_count=scene.cfg.max_obstacles,
        )
        scene_obstacles = self._instantiate_obstacles(rng, obstacle_specs)
        scene_cfg = replace(
            scene.cfg,
            seed=seed,
            object_specs=list(scene.object_specs),
            obstacle_specs=list(obstacle_specs),
        )
        return TabletopObstacleScene(
            cfg=scene_cfg,
            object_specs=list(scene.object_specs),
            scene_obstacles=scene_obstacles,
            factory=self,
        )

    def resample_full(self, scene: "TabletopObstacleScene" | None = None, seed: int | None = None) -> "TabletopObstacleScene":
        seed = (self.cfg.seed if scene is None else scene.cfg.seed) if seed is None else int(seed)
        base_cfg = self.cfg if scene is None else replace(
            scene.cfg,
            object_specs=None,
            obstacle_specs=None,
        )
        return TabletopObstacleSceneFactory(base_cfg).create(seed=seed)

    def reset_positions(self, scene: "TabletopObstacleScene", env, rng) -> None:
        spawn = scene.cfg.object_spawn_half_range
        placed_xy: list[tuple[float, float]] = []
        obstacle_records = scene.scene_obstacles

        for spec, name in zip(scene.object_specs, scene._meta.object_names):
            jx = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_x")
            jy = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_y")
            jyaw = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_yaw")

            radius = self._object_planar_radius(spec)
            x, y = self._sample_spawn_xy(rng, spawn, placed_xy, radius, obstacle_records)
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
            env.data.mocap_pos[i, 2] = scene.cfg.plane_z

    def _instantiate_obstacles(self, rng, obstacle_specs: list[ObstacleSpec]) -> list[SceneObstacle]:
        placed_xy: list[tuple[float, float]] = []
        out: list[SceneObstacle] = []
        for i, spec in enumerate(obstacle_specs):
            x, y = self._sample_obstacle_xy(rng, placed_xy, spec)
            placed_xy.append((x, y))
            z = float(spec.size[-1]) if len(spec.size) >= 3 else 0.03
            out.append(
                SceneObstacle(
                    name=f"obstacle_{i}",
                    spec_id=spec.spec_id,
                    shape=spec.shape,
                    size=tuple(spec.size),
                    rgba=tuple(self.cfg.obstacle_rgba),
                    pos=(x, y, z),
                    display_name=spec.display_name,
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
        placed_xy: list[tuple[float, float]],
        object_radius: float,
        obstacles: list[SceneObstacle],
    ) -> tuple[float, float]:
        for _ in range(200):
            x = float(rng.uniform(-spawn, spawn))
            y = float(rng.uniform(-spawn, spawn))
            if not all((x - px) ** 2 + (y - py) ** 2 >= self.cfg.min_object_separation ** 2 for px, py in placed_xy):
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
    """Planar tabletop scene with coherent object/mocap bundles."""

    def __init__(
        self,
        cfg: TabletopObstacleSceneConfig,
        *,
        object_specs: list[PrimitiveObjectSpec] | None = None,
        scene_obstacles: list[SceneObstacle] | None = None,
    ):
        self.cfg = cfg
        self.object_specs = list(object_specs) if object_specs is not None else list(cfg.object_specs or [])
        self.obstacle_specs = list(cfg.obstacle_specs) if cfg.obstacle_specs is not None else []
        self.scene_obstacles = list(scene_obstacles) if scene_obstacles is not None else []
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
                    "label": obs.spec_id,
                    "shape": obs.shape,
                    "size": list(obs.size),
                    "rgba": list(obs.rgba),
                    "pos": list(obs.pos),
                }
                for obs in self.scene_obstacles
            },
            goal_geom_name="goal_region",
            plane_z=cfg.plane_z,
        )

    def metadata(self):
        return self._meta

    def build_mjcf(self) -> str:
        gravity = "0 0 -9.81" if self.cfg.gravity_on else "0 0 0"
        h = self.cfg.table_half_size

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

    def _make_object_bundle(self, index: int, spec: PrimitiveObjectSpec):
        name = f"obj_{index}"
        mocap_name = f"mocap_{index}"

        shape = spec.shape
        rgba = " ".join(map(str, spec.rgba))
        size = " ".join(map(str, spec.size))

        x0 = 0.0
        y0 = 0.0
        z0 = self.cfg.plane_z
        slide_limit = self.cfg.table_half_size - 0.07

        geom_xml = (
            f'<geom name="{name}_geom" type="{shape}" size="{size}" rgba="{rgba}" mass="{spec.mass}"/>'
        )

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
