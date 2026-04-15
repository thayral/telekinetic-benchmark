import numpy as np
import mujoco
from telekinetics.core.action import BaseActionInterface

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



    def apply(self, env, action):
        idx = None if action is None else action.object_index
        steps = 4 if action is None else int(action.steps)

        if idx is not None and idx >= 0:

            if action.frame == "world":
                delta_world_xy = action.dxy
            elif action.frame == "camera":
                delta_world_xy = self._action_dxy_world(env, action)
            else:
                raise ValueError(...)


            # update selected mocap target
            idx = int(action.object_index)
            env.current_selection = idx
            self._sync_inactive_mocaps(env, idx)

            dxy = np.asarray(delta_world_xy, dtype=float)
            new_pos = env.data.mocap_pos[idx].copy()
            new_pos[0] += dxy[0]
            new_pos[1] += dxy[1]
            new_pos[2] = self.plane_z

            # table space mocap pos clip
            new_pos[0] = np.clip(new_pos[0], *self.x_bounds)
            new_pos[1] = np.clip(new_pos[1], *self.y_bounds)
            
            # mocap - obj distance clip
            obj_pos = env.object_position(idx)
            offset_xy = new_pos[:2] - obj_pos[:2]
            dist = np.linalg.norm(offset_xy)
            if dist > self.max_mocap_offset and dist > 1e-8:
                # offset_xy = offset_xy / dist * self.max_mocap_offset
                new_pos[0] = obj_pos[0] + offset_xy[0] / dist * self.max_mocap_offset
                new_pos[1] = obj_pos[1] + offset_xy[1] / dist * self.max_mocap_offset


            env.data.mocap_pos[idx] = new_pos

        for _ in range(steps):

            self._sync_inactive_mocaps(env, idx)

            mujoco.mj_step(env.model, env.data)


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