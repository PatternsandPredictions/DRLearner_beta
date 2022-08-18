"""Example running distributed DRLearner agent, on Lunar Lander."""

import functools
import logging
import os

import acme
from acme.jax import utils
import launchpad as lp
from absl import app
from absl import flags
from acme import specs

from drlearner.drlearner import DistributedDRLearnerFromConfig, networks_zoo
from drlearner.core.observers import IntrinsicRewardObserver, MetaControllerObserver, DistillationCoefObserver
from drlearner.configs.config_lunar_lander import LunarLanderDRLearnerConfig
from drlearner.configs.resources import get_toy_env_vertex_resources, get_local_resources
from drlearner.environments.lunar_lander import make_ll_environment
from drlearner.drlearner.networks import make_policy_networks
from drlearner.utils.utils import evaluator_factory_logger_choice, make_tf_logger

flags.DEFINE_string('level', 'LunarLander-v2', 'Which game to play.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes to train for.')
flags.DEFINE_string('exp_path', 'experiments/default', 'Run name.')
flags.DEFINE_integer('num_actors_per_mixture', 1, 'Number of parallel actors per mixture.')

flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_bool('run_on_vertex', False, 'Whether to run training in multiple processes or on Vertex AI.')
flags.DEFINE_bool('colocate_learner_and_reverb', False, 'Flag indicating whether to colocate learner and reverb.')

FLAGS = flags.FLAGS

def make_program():
    config = LunarLanderDRLearnerConfig

    config_dir = os.path.join('experiments/', FLAGS.exp_path.strip('/').split('/')[-1])
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    with open(os.path.join(config_dir, 'config.txt'), 'w') as f:
        f.write(str(config))

    print(config)

    env = make_ll_environment(0)
    env_spec = acme.make_environment_spec(env)

    def net_factory(env_spec: specs.EnvironmentSpec):
        return networks_zoo.make_lunar_lander_nets(config, env_spec)

    def env_factory(seed: int):
        return make_ll_environment(seed)

    observers = [
        IntrinsicRewardObserver(),
        MetaControllerObserver(),
        DistillationCoefObserver(),
    ]

    evaluator_logger_fn = functools.partial(make_tf_logger, FLAGS.exp_path,
                                            'evaluator', save_data=True,
                                            time_delta=1, asynchronous=True,
                                            serialize_fn=utils.fetch_devicearray,
                                            print_fn=logging.info,
                                            steps_key='evaluator_steps')

    learner_logger_function = functools.partial(make_tf_logger, FLAGS.exp_path,
                                                'learner', save_data=False,
                                                time_delta=1, asynchronous=True,
                                                serialize_fn=utils.fetch_devicearray,
                                                print_fn=logging.info,
                                                steps_key='learner_steps')

    program = DistributedDRLearnerFromConfig(
        seed=FLAGS.seed,
        environment_factory=env_factory,
        network_factory=net_factory,
        config=config,
        workdir=FLAGS.exp_path,
        num_actors_per_mixture=FLAGS.num_actors_per_mixture,
        environment_spec=env_spec,
        actor_observers=observers,
        learner_logger_fn=learner_logger_function,
        evaluator_observers=observers,
        evaluator_factories=[
            evaluator_factory_logger_choice(
                environment_factory=make_ll_environment,
                network_factory=net_factory,
                policy_factory=lambda networks: make_policy_networks(networks, config, evaluation=True),
                logger_fn=evaluator_logger_fn,
                observers=observers
            )
        ],
        multithreading_colocate_learner_and_reverb=FLAGS.colocate_learner_and_reverb

    ).build(name=FLAGS.exp_path.strip('/').split('/')[-1])

    return program


def main(_):
    program = make_program()

    if FLAGS.run_on_vertex:
        resources = get_toy_env_vertex_resources()
        lp.launch(
            program,
            launch_type=lp.LaunchType.VERTEX_AI,
            xm_resources=resources)
    else:
        resources = get_local_resources()
        lp.launch(
            program,
            lp.LaunchType.LOCAL_MULTI_PROCESSING,
            local_resources=resources
        )


if __name__ == '__main__':
    app.run(main)
