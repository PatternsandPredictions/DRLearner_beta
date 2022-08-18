"""Example running Local Layout DRLearner agent, on Atari-like environments."""

import os

import acme
from absl import app
from absl import flags

from drlearner.drlearner import networks_zoo, DRLearner
from drlearner.configs.config_atari import AtariDRLearnerConfig
from drlearner.core.environment_loop import EnvironmentLoop
from drlearner.environments.atari import make_environment
from drlearner.utils.utils import make_tf_logger

flags.DEFINE_string('level', 'PongNoFrameskip-v4', 'Which game to play.')
flags.DEFINE_integer('num_episodes', 100000, 'Number of episodes to train for.')
flags.DEFINE_string('exp_path', 'experiments/atari', 'Run name.')

flags.DEFINE_integer('seed', 42, 'Random seed.')

FLAGS = flags.FLAGS


def main(_):
    config = AtariDRLearnerConfig
    print(config)
    if not os.path.exists(FLAGS.exp_path):
        os.makedirs(FLAGS.exp_path)
    with open(os.path.join(FLAGS.exp_path, 'config.txt'), 'w') as f:
        f.write(str(config))

    env = make_environment(FLAGS.level, oar_wrapper=True)
    env_spec = acme.make_environment_spec(env)

    networks = networks_zoo.make_atari_nets(config, env_spec)

    agent = DRLearner(
        env_spec,
        networks=networks,
        config=config,
        seed=FLAGS.seed,
        workdir=FLAGS.exp_path
    )

    logger = make_tf_logger(FLAGS.exp_path)

    loop = EnvironmentLoop(env, agent, logger=logger)
    loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
    app.run(main)
