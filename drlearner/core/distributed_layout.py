import dataclasses
import logging
from typing import Any, Callable, Optional, Sequence

from acme import core
from acme import specs
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import savers
from acme.jax import types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
from acme.utils import observers as observers_lib
from acme.utils import signals
import dm_env
import jax
import launchpad as lp
import reverb
import time 

from .environment_loop import EnvironmentLoop

ActorId = int
AgentNetwork = Any
PolicyNetwork = Any
NetworkFactory = Callable[[specs.EnvironmentSpec], AgentNetwork]
PolicyFactory = Callable[[AgentNetwork], PolicyNetwork]
Seed = int
EnvironmentFactory = Callable[[Seed], dm_env.Environment]
MakeActorFn = Callable[[types.PRNGKey, PolicyNetwork, core.VariableSource],
                       core.Actor]
EvaluatorFactory = Callable[[
                                types.PRNGKey,
                                core.VariableSource,
                                counting.Counter,
                                MakeActorFn,
                            ], core.Worker]


def get_default_logger_fn(
        log_to_bigtable: bool = False,
        log_every: float = 10) -> Callable[[ActorId], loggers.Logger]:
    """Creates an actor logger."""

    def create_logger(actor_id: ActorId):
        return loggers.make_default_logger(
            'actor',
            save_data=(log_to_bigtable and actor_id == 0),
            time_delta=log_every,
            steps_key='actor_steps')

    return create_logger


def default_evaluator_factory(
        environment_factory: EnvironmentFactory,
        network_factory: NetworkFactory,
        policy_factory: PolicyFactory,
        observers: Sequence[observers_lib.EnvLoopObserver] = (),
        log_to_bigtable: bool = False) -> EvaluatorFactory:
    """Returns a default evaluator process."""

    def evaluator(
            random_key: networks_lib.PRNGKey,
            variable_source: core.VariableSource,
            counter: counting.Counter,
            make_actor: MakeActorFn,
    ):
        """The evaluation process."""

        # Create environment and evaluator networks
        environment_key, actor_key = jax.random.split(random_key)
        # Environments normally require uint32 as a seed.
        environment = environment_factory(utils.sample_uint32(environment_key))
        networks = network_factory(specs.make_environment_spec(environment))

        actor = make_actor(actor_key, policy_factory(networks), variable_source)

        # Create logger and counter.
        counter = counting.Counter(counter, 'evaluator')
        logger = loggers.make_default_logger('evaluator', log_to_bigtable,
                                             steps_key='actor_steps')

        # Create the run loop and return it.
        return EnvironmentLoop(environment, actor, counter,
                               logger, observers=observers)

    return evaluator


class StepsEpisodesLimiter:
  """Process that terminates an experiment when `max_steps` or `max_episodes` is reached."""

  def __init__(self,
               counter: counting.Counter,
               max_steps: int,
               max_episodes: int,
               steps_key: str = 'actor_steps',
               episodes_key: str = 'actor_episodes'
               ):
    self._counter = counter
    self._max_steps = max_steps
    self._max_episodes = max_episodes
    self._steps_key = steps_key
    self._episodes_key = episodes_key

  def run(self):
    """Run steps/episodes limiter to terminate an experiment when max_steps is reached.
    """

    logging.info('StepsEpisodesLimiter: Starting with max_steps = %d (%s)',
                 self._max_steps, self._steps_key)
    with signals.runtime_terminator():
      while True:
        # Update the counts.
        counts = self._counter.get_counts()
        num_steps = counts.get(self._steps_key, 0)
        num_episodes = counts.get(self._episodes_key, 0)

        logging.info('StepsEpisodesLimiter: Reached %d recorded steps in the %d episode', num_steps, num_episodes)

        if num_steps > self._max_steps:
          logging.info('StepsEpisodesLimiter: Max steps of %d was reached, terminating',
                       self._max_steps)
          # Avoid importing Launchpad until it is actually used.
          import launchpad as lp  # pylint: disable=g-import-not-at-top
          lp.stop()
        
        if num_episodes > self._max_episodes:
          logging.info('StepsEpisodesLimiter: Max episodes of %d was reached, terminating',
                       self._max_episodes)
          # Avoid importing Launchpad until it is actually used.
          import launchpad as lp  # pylint: disable=g-import-not-at-top
          lp.stop()
        
        # Don't spam the counter.
        for _ in range(10):
          # Do not sleep for a long period of time to avoid LaunchPad program
          # termination hangs (time.sleep is not interruptible).
          time.sleep(1)


@dataclasses.dataclass
class CheckpointingConfig:
    """Configuration options for learner checkpointer."""
    # The maximum number of checkpoints to keep.
    max_to_keep: int = 1
    # Which directory to put the checkpoint in.
    directory: str = '~/acme'
    # If True adds a UID to the checkpoint path, see
    # `paths.get_unique_id()` for how this UID is generated.
    add_uid: bool = True


class DistributedLayout:
    """Program definition for a distributed agent based on a builder."""

    def __init__(
            self,
            seed: int,
            environment_factory: EnvironmentFactory,
            network_factory: NetworkFactory,
            builder: builders.GenericActorLearnerBuilder,
            policy_network: PolicyFactory,
            num_actors: int,
            environment_spec: Optional[specs.EnvironmentSpec] = None,
            actor_logger_fn: Optional[Callable[[ActorId], loggers.Logger]] = None,
            evaluator_factories: Sequence[EvaluatorFactory] = (),
            device_prefetch: bool = True,
            prefetch_size: int = 1,
            log_to_bigtable: bool = False,
            # TODO: Refactor: `max_episodes` and `max_steps`` sould be defined on the experiment level,
            #       not on the agent level, similarly to other experiment related abstractions
            max_episodes: Optional[int] = None,
            max_steps: Optional[int] = None,
            observers: Sequence[observers_lib.EnvLoopObserver] = (),
            multithreading_colocate_learner_and_reverb: bool = False,
            checkpointing_config: Optional[CheckpointingConfig] = None):

        if prefetch_size < 0:
            raise ValueError(f'Prefetch size={prefetch_size} should be non negative')

        actor_logger_fn = actor_logger_fn or get_default_logger_fn(log_to_bigtable)

        self._seed = seed
        self._builder = builder
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._policy_network = policy_network
        self._environment_spec = environment_spec
        self._num_actors = num_actors
        self._device_prefetch = device_prefetch
        self._log_to_bigtable = log_to_bigtable
        self._prefetch_size = prefetch_size
        self._max_episodes = max_episodes
        self._max_steps = max_steps
        self._actor_logger_fn = actor_logger_fn
        self._evaluator_factories = evaluator_factories
        self._observers = observers
        self._multithreading_colocate_learner_and_reverb = (
            multithreading_colocate_learner_and_reverb)
        self._checkpointing_config = checkpointing_config

    def replay(self):
        """The replay storage."""
        dummy_seed = 1
        environment_spec = (
                self._environment_spec or
                specs.make_environment_spec(self._environment_factory(dummy_seed)))
        return self._builder.make_replay_tables(environment_spec)

    def counter(self):
        kwargs = {}
        if self._checkpointing_config:
            kwargs = vars(self._checkpointing_config)
        return savers.CheckpointingRunner(
            counting.Counter(),
            key='counter',
            subdirectory='counter',
            time_delta_minutes=5,
            **kwargs)

    def learner(
            self,
            random_key: networks_lib.PRNGKey,
            replay: reverb.Client,
            counter: counting.Counter,
    ):
        """The Learning part of the agent."""

        iterator = self._builder.make_dataset_iterator(replay)

        dummy_seed = 1
        environment_spec = (
                self._environment_spec or
                specs.make_environment_spec(self._environment_factory(dummy_seed)))

        # Creates the networks to optimize (online) and target networks.
        networks = self._network_factory(environment_spec)

        if self._prefetch_size > 1:
            # When working with single GPU we should prefetch to device for
            # efficiency. If running on TPU this isn't necessary as the computation
            # and input placement can be done automatically. For multi-gpu currently
            # the best solution is to pre-fetch to host although this may change in
            # the future.
            device = jax.devices()[0] if self._device_prefetch else None
            iterator = utils.prefetch(
                iterator, buffer_size=self._prefetch_size, device=device)
        else:
            logging.info('Not prefetching the iterator.')

        counter = counting.Counter(counter, 'learner')

        learner = self._builder.make_learner(random_key, networks, iterator, replay,
                                             counter)
        kwargs = {}
        if self._checkpointing_config:
            kwargs = vars(self._checkpointing_config)
        # Return the learning agent.
        return savers.CheckpointingRunner(
            learner,
            key='learner',
            subdirectory='learner',
            time_delta_minutes=5,
            **kwargs)

    def actor(self, random_key: networks_lib.PRNGKey, replay: reverb.Client,
              variable_source: core.VariableSource, counter: counting.Counter,
              actor_id: ActorId) -> EnvironmentLoop:
        """The actor process."""
        adder = self._builder.make_adder(replay)

        environment_key, actor_key = jax.random.split(random_key)
        # Create environment and policy core.

        # Environments normally require uint32 as a seed.
        environment = self._environment_factory(
            utils.sample_uint32(environment_key))

        networks = self._network_factory(specs.make_environment_spec(environment))
        policy_network = self._policy_network(networks)
        actor = self._builder.make_actor(actor_key, policy_network, adder,
                                         variable_source, actor_id, False)

        # Create logger and counter.
        counter = counting.Counter(counter, 'actor')
        # Only actor #0 will write to bigtable in order not to spam it too much.
        logger = self._actor_logger_fn(actor_id)
        # Create the loop to connect environment and agent.
        return EnvironmentLoop(environment, actor, counter,
                               logger, observers=self._observers)

    def coordinator(self, counter: counting.Counter, max_steps: int, max_episodes: int):
        return StepsEpisodesLimiter(counter, max_steps, max_episodes)

    def build(self, name='agent', program: Optional[lp.Program] = None):
        """Build the distributed agent topology."""
        if not program:
            program = lp.Program(name=name)

        key = jax.random.PRNGKey(self._seed)

        replay_node = lp.ReverbNode(self.replay)
        with program.group('replay'):
            if self._multithreading_colocate_learner_and_reverb:
                replay = replay_node.create_handle()
            else:
                replay = program.add_node(replay_node)

        with program.group('counter'):
            counter = program.add_node(lp.CourierNode(self.counter))
            if self._max_steps and self._max_episodes:
                program.add_node(
                    lp.CourierNode(
                        self.coordinator, counter,
                        self._max_steps, self._max_episodes))

        learner_key, key = jax.random.split(key)
        learner_node = lp.CourierNode(self.learner, learner_key, replay, counter)
        with program.group('learner'):
            if self._multithreading_colocate_learner_and_reverb:
                learner = learner_node.create_handle()
                program.add_node(
                    lp.MultiThreadingColocation([learner_node, replay_node]))
            else:
                learner = program.add_node(learner_node)

        def make_evaluator_actor(random_key: networks_lib.PRNGKey,
                                 policy_network: PolicyNetwork,
                                 variable_source: core.VariableSource,
                                 actor_id: int = 0) -> core.Actor:
            # assign id 0 for evaluator actor => beta = beta_min
            return self._builder.make_actor(
                random_key, policy_network, variable_source=variable_source, actor_id=actor_id, is_evaluator=True)

        with program.group('evaluator'):
            for evaluator in self._evaluator_factories:
                evaluator_key, key = jax.random.split(key)
                program.add_node(
                    lp.CourierNode(evaluator, evaluator_key, learner, counter,
                                   make_evaluator_actor))

        with program.group('actor'):
            for actor_id in range(self._num_actors):
                actor_key, key = jax.random.split(key)
                program.add_node(
                    lp.CourierNode(self.actor, actor_key, replay, learner, counter,
                                   actor_id))

        return program
