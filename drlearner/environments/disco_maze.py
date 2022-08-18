from typing import Sequence, Union, Optional

import dm_env
import gym_discomaze
import numpy as np
from acme import wrappers
from acme.wrappers import base
from dm_env import specs


class DiscoMazeWrapper(base.EnvironmentWrapper):
    def __init__(self, environment: dm_env.Environment, *, to_float: bool = False,
                 max_episode_len: Optional[int] = None):
        """
        The wrapper performs the following actions:
        1. Converts observations to float (if applied)
        2. Truncates episodes to maximum number of steps (if applied).
        3. Remove action that allows no movement.
        """
        super(DiscoMazeWrapper, self).__init__(environment)
        self._to_float = to_float

        if not max_episode_len:
            max_episode_len = np.inf
        self._episode_len = 0
        self._max_episode_len = max_episode_len

        self._observation_spec = self._init_observation_spec()
        self._action_spec = self._init_action_spec()

    def _init_observation_spec(self):
        observation_spec = self.environment.observation_spec()
        if self._to_float:
            observation_shape = observation_spec.shape
            dtype = 'float64'
            observation_spec = observation_spec.replace(
                dtype=dtype,
                minimum=(observation_spec.minimum.astype(dtype) / 255.),
                maximum=(observation_spec.maximum.astype(dtype) / 255.)
            )
        return observation_spec

    def _init_action_spec(self):
        action_spec = self.environment.action_spec()

        action_spec = action_spec.replace(num_values=action_spec.num_values - 1)
        return action_spec

    def step(self, action) -> dm_env.TimeStep:
        action = action + 1
        timestep = self.environment.step(action)

        if self._to_float:
            observation = timestep.observation.astype(float) / 255.
            timestep = timestep._replace(observation=observation)

        self._episode_len += 1
        if self._episode_len == self._max_episode_len:
            timestep = timestep._replace(step_type=dm_env.StepType.LAST)

        return timestep

    def reset(self) -> dm_env.TimeStep:
        timestep = self.environment.reset()

        if self._to_float:
            observation = timestep.observation.astype(float) / 255.
            timestep = timestep._replace(observation=observation)

        self._episode_len = 0
        return timestep

    def observation_spec(self) -> Union[specs.Array, Sequence[specs.Array]]:
        return self._observation_spec

    def action_spec(self) -> Union[specs.Array, Sequence[specs.Array]]:
        return self._action_spec


def make_discomaze_environment(seed: int) -> dm_env.Environment:
    """Create 21x21 disco maze environment with 5 random colors and no target"""
    env = gym_discomaze.RandomDiscoMaze(n_row=10, n_col=10, n_colors=5, n_targets=0, generator=seed)
    env = wrappers.GymWrapper(env)
    env = DiscoMazeWrapper(env, to_float=True, max_episode_len=5000)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.ObservationActionRewardWrapper(env)
    return env


if __name__ == '__main__':
    env = make_discomaze_environment(0)
    print(env.action_spec().replace(num_values=4))
