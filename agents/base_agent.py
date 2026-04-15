class BaseAgent:
    def reset(self) -> None:
        return None

    def act(self, obs: dict):
        raise NotImplementedError
