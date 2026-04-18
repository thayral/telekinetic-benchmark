class BaseTask:
    def reset(self, env, rng) -> None:
        raise NotImplementedError

    def reward(self, env) -> float:
        return 0.0

    def success(self, env) -> bool:
        return False

    def done(self, env) -> bool:
        return self.success(env)

    def info(self, env) -> dict:
        return {}
