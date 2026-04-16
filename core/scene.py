from dataclasses import dataclass, field

@dataclass
class SceneMetadata:
    object_names: list[str] = field(default_factory=list)
    obstacle_names: list[str] = field(default_factory=list)
    mocap_names: list[str] = field(default_factory=list)
    object_attributes: dict[str, dict] = field(default_factory=dict) # the name label
    goal_geom_name: str | None = None
    plane_z: float = 0.53

class BaseScene:
    """Minimal scene interface."""

    def build_mjcf(self) -> str:
        raise NotImplementedError

    def metadata(self) -> SceneMetadata:
        raise NotImplementedError

    def reset_layout(self, env, rng) -> None:
        return None
