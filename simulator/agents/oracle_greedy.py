import numpy as np
from telekinetics.simulator.agents.base_agent import BaseAgent
from telekinetics.simulator.core.action import TelekineticAction

class OracleGreedyAgent(BaseAgent):
    def __init__(self, step_size: float = 0.01, steps_per_action: int = 4, goal_tolerance: float = 0.01):
        self.step_size = step_size
        self.steps_per_action = steps_per_action
        self.goal_tolerance = goal_tolerance

    def _get_target_object_xy(self, obs: dict, target_index: int) -> np.ndarray:
        for obj in obs["objects"]:
            if int(obj["index"]) == int(target_index):
                return np.asarray(obj["pos"][:2], dtype=float)
        raise KeyError(f"Target object index {target_index} not found.")

    def act(self, obs: dict):
        task = obs["task"]
        target_idx = int(task["target_object"])
        goal_xy = np.asarray(task["goal_center_xy"], dtype=float)
        obj_xy = self._get_target_object_xy(obs, target_idx)

        delta = goal_xy - obj_xy
        dist = float(np.linalg.norm(delta))
        if dist < self.goal_tolerance:
            return TelekineticAction(object_index=target_idx, dxy=(0.0, 0.0), steps=self.steps_per_action, frame="world")

        direction = delta / max(dist, 1e-8)
        dxy = direction * min(self.step_size, dist)
        return TelekineticAction(
            object_index=target_idx,
            dxy=(float(dxy[0]), float(dxy[1])),
            steps=self.steps_per_action,
            frame="world"
        )
