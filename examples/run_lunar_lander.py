"""Example running Local Layout DRLearner on Lunar Lander environment."""

import os

import acme
from absl import app
from absl import flags

from drlearner.drlearner import networks_zoo, DRLearner
from drlearner.configs.config_lunar_lander import LunarLanderDRLearnerConfig
from drlearner.core.environment_loop import EnvironmentLoop
from drlearner.environments.lunar_lander import make_ll_environment
from drlearner.core.observers import IntrinsicRewardObserver, DistillationCoefObserver
from drlearner.utils.utils import make_tf_logger

flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes to train for.')
flags.DEFINE_string('exp_path', 'experiments/default', 'Run name.')

flags.DEFINE_integer('seed', 42, 'Random seed.')

FLAGS = flags.FLAGS


def main(_):
    config = LunarLanderDRLearnerConfig
    print(config)
    if not os.path.exists(FLAGS.exp_path):
        os.makedirs(FLAGS.exp_path)
    with open(os.path.join(FLAGS.exp_path, 'config.txt'), 'w') as f:
        f.write(str(config))

    env = make_ll_environment(FLAGS.seed)
    env_spec = acme.make_environment_spec(env)

    networks = networks_zoo.make_lunar_lander_nets(config, env_spec)

    agent = DRLearner(
        env_spec,
        networks=networks,
        config=config,
        seed=FLAGS.seed,
        workdir=FLAGS.exp_path
    )

    observers = [IntrinsicRewardObserver(), DistillationCoefObserver()]

    logger = make_tf_logger(FLAGS.exp_path)

    loop = EnvironmentLoop(env, agent, logger=logger, observers=observers)
    loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
    app.run(main)
