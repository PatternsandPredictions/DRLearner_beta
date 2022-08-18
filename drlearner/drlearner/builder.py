"""DRLearner Builder."""
from typing import Callable, Iterator, List, Optional
from copy import deepcopy

import acme
import jax
import jax.numpy as jnp
import optax
import reverb
import tensorflow as tf
from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import builders
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers

from .config import DRLearnerConfig
from .actor import DRLearnerActor
from .actor_core import get_actor_core
from .learning import DRLearnerLearner
from .networks import DRLearnerNetworks

# run CPU-only tensorflow for data loading
tf.config.set_visible_devices([], "GPU")


class DRLearnerBuilder(builders.ActorLearnerBuilder):
    """DRLearner Builder.

    """

    def __init__(self,
                 networks: DRLearnerNetworks,
                 config: DRLearnerConfig,
                 num_actors_per_mixture: int,
                 logger_fn: Callable[[], loggers.Logger] = lambda: None, ):
        """Creates DRLearner learner, a behavior policy and an eval actor.

        Args:
          networks: DRLearner networks, used to build core state spec.
          config: a config with DRLearner hps
          logger_fn: a logger factory for the learner
        """
        self._networks = networks
        self._config = config
        self._num_actors_per_mixture = num_actors_per_mixture
        self._logger_fn = logger_fn

        # Sequence length for dataset iterator.
        self._sequence_length = (
                self._config.burn_in_length + self._config.trace_length + 1)

        # Construct the core state spec.
        dummy_key = jax.random.PRNGKey(0)
        intrinsic_initial_state_params = networks.uvfa_net.initial_state.init(dummy_key, 1)
        intrinsic_initial_state = networks.uvfa_net.initial_state.apply(intrinsic_initial_state_params,
                                                                        dummy_key, 1)
        extrinsic_initial_state_params = networks.uvfa_net.initial_state.init(dummy_key, 1)
        extrinsic_initial_state = networks.uvfa_net.initial_state.apply(extrinsic_initial_state_params,
                                                                        dummy_key, 1)
        intrinsic_core_state_spec = utils.squeeze_batch_dim(intrinsic_initial_state)
        extrinsic_core_state_spec = utils.squeeze_batch_dim(extrinsic_initial_state)
        self._extra_spec = {
            'intrinsic_core_state': intrinsic_core_state_spec,
            'extrinsic_core_state': extrinsic_core_state_spec
        }

    def make_learner(
            self,
            random_key: networks_lib.PRNGKey,
            networks: DRLearnerNetworks,
            dataset: Iterator[reverb.ReplaySample],
            replay_client: Optional[reverb.Client] = None,
            counter: Optional[counting.Counter] = None,
    ) -> core.Learner:
        # The learner updates the parameters (and initializes them).
        return DRLearnerLearner(
            uvfa_unroll=networks.uvfa_net.unroll,
            uvfa_initial_state=networks.uvfa_net.initial_state,
            idm_action_pred=networks.embedding_net.predict_action,
            distillation_embed=networks.distillation_net.embed_sequence,
            batch_size=self._config.batch_size,
            random_key=random_key,
            burn_in_length=self._config.burn_in_length,
            beta_min=self._config.beta_min,
            beta_max=self._config.beta_max,
            gamma_min=self._config.gamma_min,
            gamma_max=self._config.gamma_max,
            num_mixtures=self._config.num_mixtures,
            target_epsilon=self._config.target_epsilon,
            importance_sampling_exponent=(
                self._config.importance_sampling_exponent),
            max_priority_weight=self._config.max_priority_weight,
            target_update_period=self._config.target_update_period,
            iterator=dataset,
            uvfa_optimizer=optax.adam(self._config.uvfa_learning_rate),
            idm_optimizer=optax.adamw(self._config.idm_learning_rate,
                                      weight_decay=self._config.idm_weight_decay),
            distillation_optimizer=optax.adamw(self._config.distillation_learning_rate,
                                               weight_decay=self._config.distillation_weight_decay),
            idm_clip_steps=self._config.idm_clip_steps,
            distillation_clip_steps=self._config.distillation_clip_steps,
            retrace_lambda=self._config.retrace_lambda,
            tx_pair=self._config.tx_pair,
            clip_rewards=self._config.clip_rewards,
            max_abs_reward=self._config.max_absolute_reward,
            replay_client=replay_client,
            counter=counter,
            logger=self._logger_fn())

    def make_replay_tables(
            self,
            environment_spec: specs.EnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""
        if self._config.samples_per_insert:
            samples_per_insert_tolerance = (
                    self._config.samples_per_insert_tolerance_rate *
                    self._config.samples_per_insert)
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=error_buffer)
        else:
            limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)

        # add intrinsic rewards and mixture_idx (intrinsic reward beta) to extra_specs
        self._extra_spec['intrinsic_reward'] = specs.Array(
            shape=environment_spec.rewards.shape,
            dtype=jnp.float32,
            name='intrinsic_reward'
        )
        self._extra_spec['mixture_idx'] = specs.Array(
            shape=environment_spec.rewards.shape,
            dtype=jnp.int32,
            name='mixture_idx'
        )
        # add probability of action under behavior policy
        self._extra_spec['behavior_action_prob'] = specs.Array(
            shape=environment_spec.rewards.shape,
            dtype=jnp.float32,
            name='behavior_action_prob'
        )

        # add the mode of evaluator
        self._extra_spec['is_eval'] = specs.Array(
            shape=environment_spec.rewards.shape,
            dtype=jnp.int32,
            name='is_eval'
        )

        self._extra_spec['alpha'] = specs.Array(
            shape=environment_spec.rewards.shape,
            dtype=jnp.float32,
            name='alpha'
        )


        return [
            reverb.Table(
                name=self._config.replay_table_name,
                sampler=reverb.selectors.Prioritized(
                    self._config.priority_exponent),
                remover=reverb.selectors.Fifo(),
                max_size=self._config.max_replay_size,
                rate_limiter=limiter,
                signature=adders_reverb.SequenceAdder.signature(
                    environment_spec, self._extra_spec))
        ]

    def make_dataset_iterator(
            self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=replay_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size,
            num_parallel_calls=self._config.num_parallel_calls)
        return dataset.as_numpy_iterator()

    def make_adder(self,
                   replay_client: reverb.Client) -> Optional[adders.Adder]:
        """Create an adder which records data generated by the actor/environment."""
        return adders_reverb.SequenceAdder(
            client=replay_client,
            period=self._config.sequence_period,
            sequence_length=self._sequence_length,
            delta_encoded=True)

    def make_actor(
            self,
            random_key: networks_lib.PRNGKey,
            policy_networks,
            adder: Optional[adders.Adder] = None,
            variable_source: Optional[core.VariableSource] = None,
            actor_id: int = 0,
            is_evaluator: bool = False,
    ) -> acme.Actor:

        # Create variable client.
        variable_client = variable_utils.VariableClient(
            variable_source,
            key='actor_variables',
            update_period=self._config.variable_update_period)
        variable_client.update_and_wait()

        intrinsic_initial_state_key1, intrinsic_initial_state_key2, \
            extrinsic_initial_state_key1, extrinsic_initial_state_key2, random_key = jax.random.split(random_key, 5)
        intrinsic_actor_initial_state_params = self._networks.uvfa_net.initial_state.init(
            intrinsic_initial_state_key1, 1)
        intrinsic_actor_initial_state = self._networks.uvfa_net.initial_state.apply(
            intrinsic_actor_initial_state_params, intrinsic_initial_state_key2, 1)
        extrinsic_actor_initial_state_params = self._networks.uvfa_net.initial_state.init(
            extrinsic_initial_state_key1, 1)
        extrinsic_actor_initial_state = self._networks.uvfa_net.initial_state.apply(
            extrinsic_actor_initial_state_params, extrinsic_initial_state_key2, 1)

        config = deepcopy(self._config)
        if is_evaluator:
            config.window = self._config.evaluation_window
            config.epsilon = self._config.evaluation_epsilon
            config.mc_epsilon = self._config.evaluation_mc_epsilon
        else:
            config.window = self._config.actor_window
            config.epsilon = self._config.actor_epsilon
            config.mc_epsilon = self._config.actor_mc_epsilon


        actor_core = get_actor_core(policy_networks,
                                    intrinsic_actor_initial_state,
                                    extrinsic_actor_initial_state,
                                    actor_id,
                                    self._num_actors_per_mixture,
                                    config,
                                    jit=True)

        mixture_idx = actor_id // self._num_actors_per_mixture


        return DRLearnerActor(
            actor_core, mixture_idx, random_key, variable_client, adder, backend='cpu', jit=True)
