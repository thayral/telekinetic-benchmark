class BaseObservationProvider:
    def get_observation(self, env) -> dict:
        raise NotImplementedError
