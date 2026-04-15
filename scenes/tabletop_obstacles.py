from dataclasses import dataclass
import mujoco

from telekinetics.core.scene import BaseScene, SceneMetadata


@dataclass
class TabletopObstacleSceneConfig:
    n_objects: int = 4
    seed: int = 0
    plane_z: float = 0.53
    gravity_on: bool = False
    table_half_size: float = 0.55
    object_spawn_half_range: float = 0.30


class TabletopObstacleScene(BaseScene):
    """Planar tabletop scene with coherent object/mocap bundles."""

    def __init__(self, cfg: TabletopObstacleSceneConfig):
        self.cfg = cfg
        self._meta = SceneMetadata(
            object_names=[f"obj_{i}" for i in range(cfg.n_objects)],
            obstacle_names=["obstacle_center"],
            mocap_names=[f"mocap_{i}" for i in range(cfg.n_objects)],
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
        for i in range(self.cfg.n_objects):
            obj_xml, mocap_xml, weld_xml = self._make_object_bundle(i)
            object_bodies.append(obj_xml)
            mocap_bodies.append(mocap_xml)
            welds.append(weld_xml)

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
    <camera name="cam_oblique" pos="0.9 -1.2 0.85" quat="0.86 0.40 0.23 0.16" fovy="55"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.95 0.95 0.95 1"/>

    <body name="table" pos="0 0 0.45">
      <geom name="table_geom" type="box" size="{h} {h} 0.05" rgba="0.75 0.75 0.78 1"/>

      <geom name="wall_north" type="box" pos="0 {h} 0.08" size="{h} 0.01 0.08" rgba="0.35 0.35 0.35 1"/>
      <geom name="wall_south" type="box" pos="0 -{h} 0.08" size="{h} 0.01 0.08" rgba="0.35 0.35 0.35 1"/>
      <geom name="wall_east"  type="box" pos="{h} 0 0.08" size="0.01 {h} 0.08" rgba="0.35 0.35 0.35 1"/>
      <geom name="wall_west"  type="box" pos="-{h} 0 0.08" size="0.01 {h} 0.08" rgba="0.35 0.35 0.35 1"/>

      <geom name="obstacle_center" type="box" pos="0 0 0.03" size="0.08 0.03 0.03" rgba="0.45 0.20 0.20 1"/>

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

    def _make_object_bundle(self, index: int):
        name = f"obj_{index}"
        mocap_name = f"mocap_{index}"

        shapes = ["box", "sphere", "cylinder"]
        palette = [
            (1.0, 0.1, 0.1, 1.0),
            (0.1, 0.6, 1.0, 1.0),
            (0.2, 0.9, 0.2, 1.0),
            (1.0, 0.85, 0.1, 1.0),
            (0.7, 0.2, 0.9, 1.0),
        ]

        shape = shapes[index % len(shapes)]
        rgba = " ".join(map(str, palette[index % len(palette)]))

        # Critical: shared reference pose for object body and mocap body.
        x0 = 0.0
        y0 = 0.0
        z0 = self.cfg.plane_z
        slide_limit = self.cfg.table_half_size - 0.07

        if shape == "box":
            geom_xml = f'<geom name="{name}_geom" type="box" size="0.03 0.03 0.03" rgba="{rgba}"/>'
        elif shape == "sphere":
            geom_xml = f'<geom name="{name}_geom" type="sphere" size="0.035" rgba="{rgba}"/>'
        else:
            geom_xml = f'<geom name="{name}_geom" type="cylinder" size="0.03 0.05" rgba="{rgba}"/>'

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

    def reset_layout(self, env, rng) -> None:
        spawn = self.cfg.object_spawn_half_range

        for name in self._meta.object_names:
            jx = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_x")
            jy = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_y")
            jyaw = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_yaw")

            env.data.qpos[env.model.jnt_qposadr[jx]] = float(rng.uniform(-spawn, spawn))
            env.data.qpos[env.model.jnt_qposadr[jy]] = float(rng.uniform(-spawn, spawn))
            env.data.qpos[env.model.jnt_qposadr[jyaw]] = float(rng.uniform(-0.2, 0.2))

        mujoco.mj_forward(env.model, env.data)

        # Keep mocaps exactly aligned with resulting object poses.
        for i, name in enumerate(self._meta.object_names):
            body_id = env.object_body_ids[name]
            pos = env.data.xpos[body_id].copy()
            env.data.mocap_pos[i, 0] = pos[0]
            env.data.mocap_pos[i, 1] = pos[1]
            env.data.mocap_pos[i, 2] = self.cfg.plane_z
