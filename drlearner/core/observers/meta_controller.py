import dm_env
import numpy as np

class MetaControllerObserver:
    def __init__(self):
        self._mixture_indices = None
        self._is_eval = None

    def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep, actor_extras, **kwargs) -> None:
        self._mixture_indices = int(actor_extras['mixture_idx'])
        self._is_eval = int(actor_extras['is_eval'])


    def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
                action: np.ndarray, actor_extras, **kwargs) -> None:
        pass

    def get_metrics(self, **kwargs):
        return {
           'mixture_idx': self._mixture_indices,
            'is_eval': self._is_eval
        }