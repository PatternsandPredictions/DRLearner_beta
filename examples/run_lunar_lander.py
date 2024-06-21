"""
Example running Local Layout DRLearner on Lunar Lander environment.

This module contains the main function to run the Lunar Lander environment using a Deep Reinforcement Learning (DRL) agent.

Imports:
    os: Provides a way of using operating system dependent functionality.
    flags: Command line flag module.
    LunarLanderDRLearnerConfig: Configuration for the DRL agent.
    make_ll_environment: Function to create a Lunar Lander environment.
    acme: DeepMind's library of reinforcement learning components.
    networks_zoo: Contains the network architectures for the DRL agent.
    DRLearner: The DRL agent.
    IntrinsicRewardObserver, DistillationCoefObserver: Observers for the DRL agent.
    make_wandb_logger: Function to create a Weights & Biases logger.
    EnvironmentLoop: Acme's main loop for running environments.

Functions:
    main(_):
        The main function to run the Lunar Lander environment.

        It sets up the environment, the DRL agent, the observers, and the logger, and then runs the environment loop for a specified number of episodes.
"""
import os

import acme
from absl import app
from absl import flags

from drlearner.drlearner import networks_zoo, DRLearner
from drlearner.configs.config_lunar_lander import LunarLanderDRLearnerConfig
from drlearner.core.environment_loop import EnvironmentLoop
from drlearner.environments.lunar_lander import make_ll_environment
from drlearner.core.observers import IntrinsicRewardObserver, DistillationCoefObserver, StorageVideoObserver
from drlearner.utils.utils import make_wandb_logger


# Command line flags
flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to train for.')
flags.DEFINE_string('exp_path', 'experiments/default',
                    'Experiment data storage.')
flags.DEFINE_string('exp_name', 'my first run', 'Run name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')

FLAGS = flags.FLAGS


def main(_):
    # Configuration for the DRL agent hyperparameters
    config = LunarLanderDRLearnerConfig

    print(config)
    if not os.path.exists(FLAGS.exp_path):
        os.makedirs(FLAGS.exp_path)
    with open(os.path.join(FLAGS.exp_path, 'config.txt'), 'w') as f:
        f.write(str(config))

    # Create a Weights & Biases loggers for the environment and the actor
    logger_env = make_wandb_logger(
        FLAGS.exp_path, label='enviroment', hyperparams=config, exp_name=FLAGS.exp_name)
    logger_actor = make_wandb_logger(
        FLAGS.exp_path, label='actor', hyperparams=config, exp_name=FLAGS.exp_name)

    # Create the Lunar Lander environment
    env = make_ll_environment(FLAGS.seed)
    # Create the environment specification
    env_spec = acme.make_environment_spec(env)

    # Create the networks for the DRL agent learning algorithm
    networks = networks_zoo.make_lunar_lander_nets(config, env_spec)

    # Create the DRL agent
    agent = DRLearner(
        spec=env_spec,
        networks=networks,
        config=config,
        seed=FLAGS.seed,
        workdir=FLAGS.exp_path,
        logger=logger_actor
    )
    # Create the observers for the DRL agent
    observers = [IntrinsicRewardObserver(), DistillationCoefObserver(),
                 StorageVideoObserver(config)]

    # Create the environment loop
    loop = EnvironmentLoop(
        environment=env,
        actor=agent,
        logger=logger_env,
        observers=observers
    )
    # Run the environment loop for a specified number of episodes
    loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
    app.run(main)
