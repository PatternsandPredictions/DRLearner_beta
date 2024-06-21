"""Defines distributed DRLearner agent, using JAX."""

import functools
from typing import Callable, Optional, Sequence

import dm_env
from acme import specs
from acme.jax import utils
from acme.utils import loggers

from ..core import distributed_layout
from .config import DRLearnerConfig
from .builder import DRLearnerBuilder
from .networks import DRLearnerNetworks, make_policy_networks

NetworkFactory = Callable[[specs.EnvironmentSpec], DRLearnerNetworks]
EnvironmentFactory = Callable[[int], dm_env.Environment]


class DistributedDRLearnerFromConfig(distributed_layout.DistributedLayout):
    """Distributed DRLearner agents from config."""

    def __init__(
            self,
            environment_factory: EnvironmentFactory,
            environment_spec: specs.EnvironmentSpec,
            network_factory: NetworkFactory,
            config: DRLearnerConfig,
            seed: int,
            num_actors_per_mixture: int,
            workdir: str = '~/acme',
            device_prefetch: bool = False,
            log_to_bigtable: bool = True,
            log_every: float = 10.0,
            # TODO: Refactor: `max_episodes` and `max_steps`` sould be defined on the experiment level,
            #       not on the agent level, similarly to other experiment related abstractions
            max_episodes: Optional[int] = None,
            max_steps: Optional[int] = None,
            evaluator_factories: Optional[Sequence[
                distributed_layout.EvaluatorFactory]] = None,
            actor_observers=(),
            evaluator_observers=(),
            learner_logger_fn: Optional[Callable[[], loggers.Logger]] = None,
            multithreading_colocate_learner_and_reverb: bool = False
    ):
        learner_logger_fn = learner_logger_fn or functools.partial(loggers.make_default_logger,
                                      'learner', log_to_bigtable,
                                      time_delta=log_every, asynchronous=True,
                                      serialize_fn=utils.fetch_devicearray,
                                      steps_key='learner_steps')
        drlearner_builder = DRLearnerBuilder(
            networks=network_factory(environment_spec),
            config=config,
            num_actors_per_mixture=num_actors_per_mixture,
            logger=learner_logger_fn)
        policy_network_factory = (
            lambda networks: make_policy_networks(networks, config))
        if evaluator_factories is None:
            evaluator_policy_network_factory = (
                lambda networks: make_policy_networks(networks, config, evaluation=True))
            evaluator_factories = [
                distributed_layout.default_evaluator_factory(
                    environment_factory=environment_factory,
                    network_factory=network_factory,
                    policy_factory=evaluator_policy_network_factory,
                    log_to_bigtable=log_to_bigtable,
                    observers=evaluator_observers
                )
            ]
        super().__init__(
            seed=seed,
            environment_factory=environment_factory,
            network_factory=network_factory,
            builder=drlearner_builder,
            policy_network=policy_network_factory,
            evaluator_factories=evaluator_factories,
            num_actors=num_actors_per_mixture * config.num_mixtures,
            environment_spec=environment_spec,
            device_prefetch=device_prefetch,
            log_to_bigtable=log_to_bigtable,
            max_episodes=max_episodes,
            max_steps=max_steps,
            actor_logger_fn=distributed_layout.get_default_logger_fn(
                log_to_bigtable, log_every),
            prefetch_size=config.prefetch_size,
            checkpointing_config=distributed_layout.CheckpointingConfig(
                directory=workdir, add_uid=(workdir == '~/acme')),
            observers=actor_observers,
            multithreading_colocate_learner_and_reverb=multithreading_colocate_learner_and_reverb
        )
