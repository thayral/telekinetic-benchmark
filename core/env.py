from dataclasses import dataclass
import mujoco
import numpy as np

@dataclass
class EnvConfig:
    seed: int = 0
    settle_steps: int = 40

class TelekinesisEnv:
    def __init__(self, scene, action_interface, observation_provider, task=None, cfg=None):
        self.scene = scene
        self.action_interface = action_interface
        self.observation_provider = observation_provider
        self.task = task
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        xml = self.scene.build_mjcf()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = None # mujoco.Renderer(self.model, height=480, width=640) # lazy init because bug at deletion if unused ? ...
        self.meta = self.scene.metadata()

        self.object_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.meta.object_names
        }
        self.object_geom_ids = {
            f"{name}_geom": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
            for name in self.meta.object_names
        }
        self.obstacle_geom_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.meta.obstacle_names
        }
        self.goal_geom_id = None
        if self.meta.goal_geom_name is not None:
            self.goal_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.meta.goal_geom_name)
        self.current_selection = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)




        self.scene.reset_layout(self, self.rng)
        for _ in range(self.cfg.settle_steps):
            mujoco.mj_step(self.model, self.data)
        if self.task is not None:
            self.task.reset(self, self.rng)
        return self.get_observation()

    def step(self, action):
        self.action_interface.apply(self, action)
        obs = self.get_observation()
        reward = self.task.reward(self) if self.task else 0.0
        done = self.task.done(self) if self.task else False
        info = self.task.info(self) if self.task else {}
        return obs, reward, done, info

    def get_observation(self):
        obs = self.observation_provider.get_observation(self)
        if self.task is not None:
            obs["task"] = self.task.info(self)
        return obs

    def render_rgb(self, camera_name="cam_oblique"):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.renderer.update_scene(self.data, camera=cam_id)
        return self.renderer.render()

    def object_position(self, object_index):
        name = self.meta.object_names[object_index]
        return self.data.xpos[self.object_body_ids[name]].copy()

    def get_object_states(self):
        out = []
        for idx, name in enumerate(self.meta.object_names):
            record = {"index": idx, "name": name, "pos": self.data.xpos[self.object_body_ids[name]].copy()}
            record.update(self.meta.object_attributes.get(name, {}))
            out.append(record)
        return out

    def get_obstacle_states(self):
        out = []
        for name, gid in self.obstacle_geom_ids.items():
            out.append({
                "name": name,
                "pos": self.model.geom_pos[gid].copy(),
                "size": self.model.geom_size[gid].copy(),
            })
        return out

    def set_goal_region(self, center_xy, half_extents_xy):
        if self.goal_geom_id is None:
            return

        # goal_region lives in the table body's local frame, so keep it as a thin
        # tabletop patch 
        self.model.geom_pos[self.goal_geom_id, 0] = float(center_xy[0])
        self.model.geom_pos[self.goal_geom_id, 1] = float(center_xy[1])
        self.model.geom_pos[self.goal_geom_id, 2] = 0.002

        self.model.geom_size[self.goal_geom_id, 0] = float(half_extents_xy[0])
        self.model.geom_size[self.goal_geom_id, 1] = float(half_extents_xy[1])

        # Keep thickness fixed; this is a visual marker, not a volumetric target.
        self.model.geom_size[self.goal_geom_id, 2] = 0.002
    
        self.model.geom_rgba[self.goal_geom_id] = [0.0, 1.0, 1.0, 0.85]



    # used for observation dump only ?
    def highlight_selected_object(self, object_index):
        self.current_selection = object_index

    def close(self):
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None