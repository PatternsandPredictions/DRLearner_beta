"""
Example running Local Layout DRLearner agent, on Atari-like environments.

This module contains the main function to run the Atari environment using a Deep Reinforcement Learning (DRL) agent.

Imports:
    os: Provides a way of using operating system dependent functionality.
    flags: Command line flag module.
    AtariDRLearnerConfig: Configuration for the DRL agent.
    make_environment: Function to create an Atari environment.
    acme: DeepMind's library of reinforcement learning components.
    networks_zoo: Contains the network architectures for the DRL agent.
    DRLearner: The DRL agent.
    make_wandb_logger: Function to create a Weights & Biases logger.
    EnvironmentLoop: Acme's main loop for running environments.

Functions:
    main(_):
        The main function to run the Atari environment.

        It sets up the environment, the DRL agent, and the logger, and then runs the environment loop for a specified number of episodes.
"""
import os

import acme
from absl import app
from absl import flags

from drlearner.drlearner import networks_zoo, DRLearner
from drlearner.configs.config_atari import AtariDRLearnerConfig
from drlearner.core.environment_loop import EnvironmentLoop
from drlearner.environments.atari import make_environment
from drlearner.core.observers import IntrinsicRewardObserver, DistillationCoefObserver
from drlearner.utils.utils import make_wandb_logger

# Command line flags
flags.DEFINE_string('level', 'PongNoFrameskip-v4', 'Which game to play.')
flags.DEFINE_integer('num_episodes', 7, 'Number of episodes to train for.')
flags.DEFINE_string('exp_path', 'experiments/default',
                    'Experiment data storage.')
flags.DEFINE_string('exp_name', 'my first run', 'Run name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

flags.DEFINE_bool('force_sync_run', False, 'Skip deadlock warning.')

FLAGS = flags.FLAGS


def main(_):
    # Configuration for the DRL agent hyperparameters
    config = AtariDRLearnerConfig
    # To avoid the deadlock when running reverb in the synchronous set-up,
    # this setting ensures rate limiter won't be called.
    # @see https://github.com/google-deepmind/acme/issues/207 for additional information.
    if config.samples_per_insert != 0:
        if not FLAGS.force_sync_run:
            while True:
                user_answer = input("\nThe simulation may deadlock if run in the synchronous set-up with samples_per_rate != 0. "
                                    "Do you want to continue? (yes/no): ")

                if user_answer.lower() in ["yes", "y"]:
                    print("Proceeding...")
                    break
                elif user_answer.lower() in ["no", "n"]:
                    print("Exiting...")
                    return
                else:
                    print("Invalid input. Please enter yes/no.")

    print(config)
    if not os.path.exists(FLAGS.exp_path):
        os.makedirs(FLAGS.exp_path)
    with open(os.path.join(FLAGS.exp_path, 'config.txt'), 'w') as f:
        f.write(str(config))

    # Create the Atari environment
    env = make_environment(FLAGS.level, oar_wrapper=True)
    # Create the environment specification
    env_spec = acme.make_environment_spec(env)

    # Create the networks for the DRL agent learning algorithm
    networks = networks_zoo.make_atari_nets(config, env_spec)

    # Create a Weights & Biases loggers for the environment and the actor
    logger_env = make_wandb_logger(
        FLAGS.exp_path, label='enviroment', hyperparams=config, exp_name=FLAGS.exp_name)
    logger_actor = make_wandb_logger(
        FLAGS.exp_path, label='actor', hyperparams=config, exp_name=FLAGS.exp_name)

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
    # TODO: Add StorageVideoObserver
    observers = [IntrinsicRewardObserver(), DistillationCoefObserver()]

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
