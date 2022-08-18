import dm_env
import numpy as np


class DistillationCoefObserver:
    def __init__(self):
        self._alphas = None

    def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep, actor_extras, **kwargs) -> None:
        self._alphas = []

    def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
                action: np.ndarray, actor_extras, **kwargs) -> None:
        self._alphas.append(float(actor_extras['alpha']))


    def get_metrics(self, **kwargs):
        return {'Mean Distillation Alpha': np.mean(self._alphas)}