# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for different experiment flavours."""

import functools

import dm_env
import gym
from acme import wrappers


def make_environment(level: str = 'PongNoFrameskip-v4',
                     oar_wrapper: bool = False) -> dm_env.Environment:
    """Loads the Atari environment."""
    env = gym.make(level, full_action_space=True)

    # Always use episodes of 108k steps as this is standard, matching the paper.
    max_episode_len = 108_000
    wrapper_list = [
        wrappers.GymAtariAdapter,
        functools.partial(
            wrappers.AtariWrapper,
            action_repeats=4,
            pooled_frames=4,
            zero_discount_on_life_loss=False,
            expose_lives_observation=False,
            num_stacked_frames=1,
            max_episode_len=max_episode_len,
            to_float=True,
            grayscaling=True
        ),
    ]
    if oar_wrapper:
        # E.g. IMPALA and R2D2 use this particular variant.
        wrapper_list.append(wrappers.ObservationActionRewardWrapper)
    wrapper_list.append(wrappers.SinglePrecisionWrapper)

    return wrappers.wrap_all(env, wrapper_list)
