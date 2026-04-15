import numpy as np
from telekinetics.core.task import BaseTask

class PlaceObjectInRegionTask(BaseTask):
    def __init__(self, sampler, success_hold_steps=5):
        self.sampler = sampler
        self.success_hold_steps = success_hold_steps
        self.target_object = 0
        self.goal_center_xy = (0.2, 0.2)
        self.goal_half_extents_xy = (0.06, 0.06)
        self._success_counter = 0

    def reset(self, env, rng):
        params = self.sampler.sample(env, rng)
        self.target_object = params.target_object
        self.goal_center_xy = params.goal_center_xy
        self.goal_half_extents_xy = params.goal_half_extents_xy
        self._success_counter = 0
        env.set_goal_region(self.goal_center_xy, self.goal_half_extents_xy)
        env.highlight_selected_object(self.target_object)

    def _in_goal(self, env):
        pos = env.object_position(self.target_object)
        dx = abs(pos[0] - self.goal_center_xy[0])
        dy = abs(pos[1] - self.goal_center_xy[1])
        return dx <= self.goal_half_extents_xy[0] and dy <= self.goal_half_extents_xy[1]

    def reward(self, env):
        pos = env.object_position(self.target_object)
        goal = np.asarray(self.goal_center_xy)
        return -float(np.linalg.norm(pos[:2] - goal))

    def success(self, env):
        if self._in_goal(env):
            self._success_counter += 1
        else:
            self._success_counter = 0
        return self._success_counter >= self.success_hold_steps

    def info(self, env):
        return {
            "task_name": "place_object_in_region",
            "target_object": self.target_object,
            "goal_center_xy": self.goal_center_xy,
            "goal_half_extents_xy": self.goal_half_extents_xy,
            "in_goal": self._in_goal(env),
            "success_counter": self._success_counter,
        }
