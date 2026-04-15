from dataclasses import dataclass

class BaseActionInterface:
    def apply(self, env, action) -> None:
        raise NotImplementedError

@dataclass
class TelekineticAction:
    object_index: int
    dxy: tuple[float, float]
    frame: str
    steps: int = 4
