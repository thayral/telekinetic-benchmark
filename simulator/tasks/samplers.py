from dataclasses import dataclass

@dataclass
class PlaceInRegionParams:
    target_object: int
    goal_center_xy: tuple[float, float]
    goal_half_extents_xy: tuple[float, float]

class PlaceInRegionSampler:
    def __init__(self, x_range=(-0.35, 0.35), y_range=(-0.35, 0.35), goal_half_extents=(0.06, 0.06)):
        self.x_range = x_range
        self.y_range = y_range
        self.goal_half_extents = goal_half_extents

    def sample(self, env, rng):
        target_object = int(rng.integers(0, len(env.meta.object_names)))
        gx = float(rng.uniform(*self.x_range))
        gy = float(rng.uniform(*self.y_range))
        return PlaceInRegionParams(
            target_object=target_object,
            goal_center_xy=(gx, gy),
            goal_half_extents_xy=self.goal_half_extents,
        )
