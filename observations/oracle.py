from telekinetics.core.observation import BaseObservationProvider

class OracleSceneObservation(BaseObservationProvider):
    def __init__(self, include_obstacles=True):
        self.include_obstacles = include_obstacles

    def get_observation(self, env):
        obs = {
            "objects": env.get_object_states(),
            "selected_object": env.current_selection,
        }
        if self.include_obstacles:
            obs["obstacles"] = env.get_obstacle_states()
        return obs
