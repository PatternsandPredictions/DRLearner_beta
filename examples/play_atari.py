"""Example running a trained DRLearner agent on Atari and recording its play."""

import acme

from absl import flags
from absl import app

from drlearner.drlearner import DRLearner, networks_zoo
from drlearner.core.environment_loop import EnvironmentLoop
from drlearner.environments.atari import make_environment
from drlearner.configs.config_atari import AtariDRLearnerConfig
from drlearner.utils.utils import make_tf_logger
from drlearner.core.observers import StorageVideoObserver

flags.DEFINE_string('level', 'BoxingNoFrameskip-v4', 'Which game to play.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to train for.')
flags.DEFINE_string('exp_path', 'experiments/atari_distributed', 'Run name.')
flags.DEFINE_string('checkpoint_path', 'experiments/atari_distributed', 'Path to dir where checkpoints/ are')

FLAGS = flags.FLAGS


def load_and_evaluate(_):
    config = AtariDRLearnerConfig
    config.batch_size = 1
    config.num_mixtures = 32
    config.beta_max = 0.  # if num_mixtures == 1 beta == beta_max
    config.n_arms = 32
    config.logs_dir = FLAGS.exp_path
    config.video_log_period = 1
    config.env_library = 'gym'
    config.actor_epsilon = 0.01
    config.epsilon = 0.01
    config.mc_epsilon = 0.01

    env = make_environment(FLAGS.level, oar_wrapper=True)
    env_spec = acme.make_environment_spec(env)

    agent = DRLearner(
        env_spec,
        networks=networks_zoo.make_atari_nets(config, env_spec),
        config=config,
        seed=FLAGS.seed,
        workdir=FLAGS.checkpoint_path
    )

    observers = [StorageVideoObserver(config)]
    logger = make_tf_logger(FLAGS.exp_path, label='evaluator')

    loop = EnvironmentLoop(env, agent, logger=logger, observers=observers, should_update=False)
    loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
    app.run(load_and_evaluate)
