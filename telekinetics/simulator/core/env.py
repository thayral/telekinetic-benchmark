from dataclasses import dataclass
import inspect
import mujoco
import numpy as np

@dataclass
class EnvConfig:
    seed: int = 0
    settle_steps: int = 40

class TelekinesisEnv:
    def __init__(self, scene, action_interface, observation_provider, task=None, cfg=None, step_end_callbacks=None):
        self.scene = scene
        self.action_interface = action_interface
        self.observation_provider = observation_provider
        self.task = task
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.step_end_callbacks = list(step_end_callbacks or [])
        self.step_count = 0

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
        self.step_count = 0




        self.scene.reset_layout(self, self.rng)
        for _ in range(self.cfg.settle_steps):
            self.env_mj_step(phase="reset_settle", dispatch_callbacks=False)
        # Treat reset settling as initialization, not interaction rollout time.
        # This keeps debug rollout frame names starting from step-000001 after
        # each position reset / freshly constructed scene.
        self.step_count = 0
        if self.task is not None:
            self.task.reset(self, self.rng)
        return self.get_observation()

    def step(self, action, on_step_end=None):
        step_callbacks = []
        if on_step_end is not None:
            if isinstance(on_step_end, (list, tuple)):
                step_callbacks.extend(on_step_end)
            else:
                step_callbacks.append(on_step_end)

        self.action_interface.apply(self, action, step_end_callbacks=step_callbacks)
        obs = self.get_observation()
        reward = self.task.reward(self) if self.task else 0.0
        done = self.task.done(self) if self.task else False
        info = self.task.info(self) if self.task else {}
        return obs, reward, done, info

    def env_mj_step(self, *, phase=None, step_end_callbacks=None, dispatch_callbacks=True):
        """Advance MuJoCo by one step and dispatch end-of-step callbacks.

        Runtime code should call this helper instead of calling ``mujoco.mj_step``
        directly.  Persistent callbacks live on ``self.step_end_callbacks`` and
        per-call callbacks can be passed through ``TelekinesisEnv.step``.
        """
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        if dispatch_callbacks:
            callbacks = [*self.step_end_callbacks]
            if step_end_callbacks:
                callbacks.extend(step_end_callbacks)
            for callback in callbacks:
                self._dispatch_step_end_callback(callback, phase=phase)

    def _dispatch_step_end_callback(self, callback, *, phase=None):
        """Call callbacks with a tolerant signature.

        Preferred signature is ``callback(env, step_count, phase=None)``.
        Two-argument callbacks of the form ``callback(env, step_count)`` are also
        supported to match lightweight ad-hoc callbacks.
        """
        try:
            signature = inspect.signature(callback)
            params = signature.parameters
            accepts_phase = (
                any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                or "phase" in params
            )
            accepts_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
            if accepts_phase:
                callback(self, self.step_count, phase=phase)
            elif accepts_varargs or len(params) >= 3:
                callback(self, self.step_count, phase)
            else:
                callback(self, self.step_count)
        except (TypeError, ValueError):
            # Some callable objects do not expose inspectable signatures. Fall
            # back to the preferred signature, then the compact one.
            try:
                callback(self, self.step_count, phase=phase)
            except TypeError:
                callback(self, self.step_count)

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
            record = {
                "name": name,
                "pos": self.model.geom_pos[gid].copy(),
                "size": self.model.geom_size[gid].copy(),
            }
            record.update(self.meta.obstacle_attributes.get(name, {}))
            out.append(record)
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
