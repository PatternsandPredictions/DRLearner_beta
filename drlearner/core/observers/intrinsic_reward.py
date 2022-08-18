import dm_env
import numpy as np


class IntrinsicRewardObserver:
    def __init__(self):
        self._intrinsic_rewards = None

    def observe_first(self, *args, **kwargs) -> None:
        env, timestep, actor_extras = args

        self._intrinsic_rewards = []
        self._intrinsic_rewards.append(float(actor_extras['intrinsic_reward']))

    def observe(self, *args, **kwargs) -> None:
        env, timestep, action, actor_extras = args
        self._intrinsic_rewards.append(float(actor_extras['intrinsic_reward']))

    def get_metrics(self, **kwargs):
        return {
            "intrinsic_rewards_sum": np.sum(self._intrinsic_rewards),
            "intrinsic_rewards_mean": np.mean(self._intrinsic_rewards)
        }
