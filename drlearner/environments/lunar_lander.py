import dm_env
import gym
from acme import wrappers


def make_ll_environment(seed: int) -> dm_env.Environment:
    env_name = "LunarLander-v2"

    env = gym.make(env_name)
    env = wrappers.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.ObservationActionRewardWrapper(env)

    return env
