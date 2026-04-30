import numpy as np
import mujoco
from telekinetics.simulator.core.action import BaseActionInterface, TelekineticAction

class TelekinesisActionInterface(BaseActionInterface):
    def __init__(self, plane_z=0.53, 
                 x_bounds=(-0.48, 0.48), 
                 y_bounds=(-0.48, 0.48),
                max_mocap_offset=0.04,
                default_camera_name="cam_oblique",
        ):
        self.plane_z = plane_z
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.max_mocap_offset = max_mocap_offset
        self.default_camera_name = default_camera_name

    def _sync_inactive_mocaps(self, env, active_index):
        for i in range(len(env.meta.object_names)):
            if i == active_index:
                continue
            body_pos = env.object_position(i)
            env.data.mocap_pos[i, 0] = body_pos[0]
            env.data.mocap_pos[i, 1] = body_pos[1]
            env.data.mocap_pos[i, 2] = self.plane_z



    def _clip_mocap_xy_to_table(self, xy):
        xy = np.asarray(xy, dtype=float).copy()
        xy[0] = np.clip(xy[0], *self.x_bounds)
        xy[1] = np.clip(xy[1], *self.y_bounds)
        return xy

    def _sync_active_mocap_to_object(self, env, active_index):
        if active_index is None or active_index < 0:
            return
        body_pos = env.object_position(active_index)
        env.data.mocap_pos[active_index, 0] = body_pos[0]
        env.data.mocap_pos[active_index, 1] = body_pos[1]
        env.data.mocap_pos[active_index, 2] = self.plane_z

    def _set_drag_mocap_target(self, env, active_index, target_xy):
        """Place the active mocap a fixed lead distance ahead of the object.

        The action's dxy defines a distant *world target* for the object.  At
        every simulation step we recompute the mocap position from the current
        object position, so rollout_steps now controls how long the object is
        dragged instead of how long it waits at one fixed mocap location.

        ``max_mocap_offset`` is used as the lead distance/force proxy.
        """
        obj_pos = env.object_position(active_index)
        remaining_xy = np.asarray(target_xy, dtype=float) - obj_pos[:2]
        remaining_dist = np.linalg.norm(remaining_xy)

        if remaining_dist < 1e-8:
            desired_xy = obj_pos[:2]
        else:
            lead_dist = min(float(self.max_mocap_offset), float(remaining_dist))
            desired_xy = obj_pos[:2] + remaining_xy / remaining_dist * lead_dist

        desired_xy = self._clip_mocap_xy_to_table(desired_xy)
        env.data.mocap_pos[active_index, 0] = desired_xy[0]
        env.data.mocap_pos[active_index, 1] = desired_xy[1]
        env.data.mocap_pos[active_index, 2] = self.plane_z

    def apply(self, env, action, step_end_callbacks=None):
        low_level = action # self._to_low_level_action(action)
        idx = None if low_level is None else low_level.object_index
        steps = 4 if low_level is None else int(low_level.steps)
        settle_steps = 0 if low_level is None else int(getattr(low_level, "settle_steps", 0))
        target_xy = None

        if idx is not None and idx >= 0:
            if low_level.frame == "world":
                delta_world_xy = low_level.dxy
            elif low_level.frame == "camera":
                delta_world_xy = self._action_dxy_world(env, low_level)
            else:
                raise ValueError(f"Unknown action frame: {low_level.frame}")

            idx = int(low_level.object_index)
            env.current_selection = idx
            self._sync_inactive_mocaps(env, idx)

            dxy = np.asarray(delta_world_xy, dtype=float)
            start_xy = env.object_position(idx)[:2].copy()
            target_xy = self._clip_mocap_xy_to_table(start_xy + dxy)

        for _ in range(steps):
            self._sync_inactive_mocaps(env, idx)
            if idx is not None and idx >= 0 and target_xy is not None:
                self._set_drag_mocap_target(env, idx, target_xy)
            env.env_mj_step(phase="move", step_end_callbacks=step_end_callbacks)

        # Release the active mocap before settling.  This prevents settle_steps
        # from silently extending the commanded drag distance.
        self._sync_active_mocap_to_object(env, idx)

        for _ in range(settle_steps):
            self._sync_inactive_mocaps(env, idx)
            env.env_mj_step(phase="settle", step_end_callbacks=step_end_callbacks)


    def _normalize_xy(self, v, eps=1e-8):
        n = np.linalg.norm(v)
        if n < eps:
            return None
        return v / n

    def _camera_planar_basis(self, env, camera_name=None):
        camera_name = camera_name or self.default_camera_name

        cam_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        if cam_id < 0:
            return np.array([1.0, 0.0]), np.array([0.0, 1.0])

        xmat = env.data.cam_xmat[cam_id].reshape(3, 3)

        # MuJoCo camera axes can be a little unintuitive across setups, so verify
        # with one render/debug pass; this is the intended structure:
        right_world = xmat[:, 0]
        up_world = xmat[:, 1]
        forward_world = -xmat[:, 2]

        right_xy = self._normalize_xy(right_world[:2])
        forward_xy = self._normalize_xy(forward_world[:2])

        if right_xy is None and forward_xy is None:
            return np.array([1.0, 0.0]), np.array([0.0, 1.0])

        if right_xy is None:
            forward_xy = forward_xy / np.linalg.norm(forward_xy)
            right_xy = np.array([forward_xy[1], -forward_xy[0]])

        if forward_xy is None:
            right_xy = right_xy / np.linalg.norm(right_xy)
            forward_xy = np.array([-right_xy[1], right_xy[0]])

        # optional re-orthogonalization for numerical stability
        forward_xy = forward_xy - np.dot(forward_xy, right_xy) * right_xy
        forward_xy = self._normalize_xy(forward_xy)
        if forward_xy is None:
            forward_xy = np.array([-right_xy[1], right_xy[0]])

        return right_xy, forward_xy

    def _action_dxy_world(self, env, action):
        frame = getattr(action, "frame", "world")
        dxy = np.asarray(action.dxy, dtype=float)

        if frame == "world":
            return dxy

        if frame == "camera":
            right_xy, forward_xy = self._camera_planar_basis(env)
            return dxy[0] * right_xy + dxy[1] * forward_xy

        raise ValueError(f"Unknown action frame: {frame}")