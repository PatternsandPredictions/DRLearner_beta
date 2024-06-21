"""Example running Local Layout DRLearner on DiscoMaze environment."""

import os

import acme
from absl import app
from absl import flags

from drlearner.drlearner import networks_zoo, DRLearner
from drlearner.configs.config_discomaze import DiscomazeDRLearnerConfig
from drlearner.core.environment_loop import EnvironmentLoop
from drlearner.core.observers import UniqueStatesDiscoMazeObserver, IntrinsicRewardObserver, ActionProbObserver
from drlearner.environments.disco_maze import make_discomaze_environment
from drlearner.utils.utils import make_tf_logger

flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to train for.')
flags.DEFINE_string('exp_path', 'experiments/default',
                    'Experiment data storage.')
flags.DEFINE_string('exp_name', 'my first run', 'Run name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

FLAGS = flags.FLAGS


def main(_):
    config = DiscomazeDRLearnerConfig
    print(config)
    if not os.path.exists(FLAGS.exp_path):
        os.makedirs(FLAGS.exp_path)
    with open(os.path.join(FLAGS.exp_path, 'config.txt'), 'w') as f:
        f.write(str(config))

    env = make_discomaze_environment(FLAGS.seed)
    env_spec = acme.make_environment_spec(env)

    networks = networks_zoo.make_discomaze_nets(config, env_spec)

    agent = DRLearner(
        env_spec,
        networks=networks,
        config=config,
        seed=FLAGS.seed)

    logger = make_tf_logger(FLAGS.exp_path)

    observers = [
        UniqueStatesDiscoMazeObserver(),
        IntrinsicRewardObserver(),
        ActionProbObserver(num_actions=env_spec.actions.num_values),
    ]
    loop = EnvironmentLoop(env, agent, logger=logger, observers=observers)
    loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
    app.run(main)
