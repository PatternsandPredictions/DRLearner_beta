import dm_env
import numpy as np


class ActionProbObserver:
    def __init__(self, num_actions):
        self._num_actions = num_actions
        self._action_counter = None

    def observe_first(self, *args, **kwargs) -> None:
        # todo: defaultdict
        self._action_counter = {i: 0 for i in range(self._num_actions)}

    def observe(self, *args, **kwargs) -> None:
        env, timestamp, action, actor_extras = args
        self._action_counter[int(action)] += 1

    def get_metrics(self, **kwargs):
        total_actions = sum(self._action_counter.values())
        return {f'Action: {i}': self._action_counter[i] / total_actions for i in range(self._num_actions)}
