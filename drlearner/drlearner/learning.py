"""DRLearner learner implementation."""
import dataclasses
import functools
import time
from typing import Dict, Iterator, List, Optional, Tuple, Sequence

import acme
import jax
import jax.numpy as jnp
import optax
import reverb
import rlax
import tree
from acme.adders import reverb as adders
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import async_utils
from acme.utils import counting
from acme.utils import loggers

from .networks import UVFANetworkInput
from .drlearner_types import DRLearnerNetworksParams, DRLearnerNetworksOptStates, TrainingState
from .utils import epsilon_greedy_prob, get_beta, get_gamma

_PMAP_AXIS_NAME = 'data'


class DRLearnerLearner(acme.Learner):
    """DRLearnerlearner."""

    def __init__(self,
                 uvfa_unroll: networks_lib.FeedForwardNetwork,
                 uvfa_initial_state: networks_lib.FeedForwardNetwork,
                 idm_action_pred: networks_lib.FeedForwardNetwork,
                 distillation_embed: networks_lib.FeedForwardNetwork,
                 batch_size: int,
                 beta_min: float,
                 beta_max: float,
                 gamma_min: float,
                 gamma_max: float,
                 num_mixtures: int,
                 random_key: networks_lib.PRNGKey,
                 burn_in_length: int,
                 target_epsilon: float,
                 importance_sampling_exponent: float,
                 max_priority_weight: float,
                 target_update_period: int,
                 iterator: Iterator[reverb.ReplaySample],
                 uvfa_optimizer: optax.GradientTransformation,
                 idm_optimizer: optax.GradientTransformation,
                 distillation_optimizer: optax.GradientTransformation,
                 idm_clip_steps: int,
                 distillation_clip_steps: int,
                 retrace_lambda: float,
                 tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR,
                 clip_rewards: bool = False,
                 max_abs_reward: float = 1.,
                 use_core_state: bool = True,
                 prefetch_size: int = 2,
                 replay_client: Optional[reverb.Client] = None,
                 counter: Optional[counting.Counter] = None,
                 logger: Optional[loggers.Logger] = None):
        """Initializes the learner."""

        batched_epsilon_greedy_prob = jax.vmap(
            jax.vmap(epsilon_greedy_prob, in_axes=(0, None), out_axes=0),
            in_axes=(0, None),
            out_axes=0
        )

        def uvfa_loss(
                uvfa_params: networks_lib.Params,
                uvfa_target_params: networks_lib.Params,
                key_grad: networks_lib.PRNGKey,
                sample: reverb.ReplaySample,
                rewards_t: jnp.ndarray,
                core_state_extraction_name: str = 'extrinsic_core_state'
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Computes mean transformed N-step loss for a batch of sequences.
            """

            # Convert sample data to sequence-major format [T, B, ...].
            data = utils.batch_to_sequence(sample.data)

            # Get core state & warm it up on observations for a burn-in period.
            if use_core_state:
                # Replay core state.
                online_state = jax.tree_map(lambda x: x[0], data.extras[core_state_extraction_name])
            else:
                online_state = uvfa_initial_state
            target_state = online_state

            # Maybe burn the core state in.
            if burn_in_length:
                key_grad, key1, key2 = jax.random.split(key_grad, 3)

                burn_in_input = UVFANetworkInput(
                    oar=jax.tree_map(lambda x: x[1:burn_in_length + 1], data.observation),  # x_t, r_tm1, a_tm1
                    intrinsic_reward=jax.tree_map(lambda x: x[:burn_in_length], data.extras['intrinsic_reward']),
                    mixture_idx=jax.tree_map(lambda x: x[:burn_in_length], data.extras['mixture_idx'])
                )

                _, online_state = uvfa_unroll.apply(uvfa_params, key1, burn_in_input, online_state)
                _, target_state = uvfa_unroll.apply(uvfa_target_params, key2, burn_in_input,
                                                    target_state)

                # Only get data to learn on from after the end of the burn in period.
                data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

            # Unroll on sequences to get online and target Q-Values.
            key1, key2 = jax.random.split(key_grad)

            # extract data from nested structure
            observations_t = jax.tree_map(lambda x: x[1:], data.observation)  # x_t, r_tm1, a_tm1
            done_t = jax.tree_map(lambda x: x[1:], data.discount)  # 1 if not end of episode, 0 - end of episode
            mixture_idx_t = jax.tree_map(lambda x: x[1:], data.extras['mixture_idx'])
            actions_t = jax.tree_map(lambda x: x[1:], data.action)
            mu_t = jax.tree_map(lambda x: x[1:], data.extras['behavior_action_prob'])

            discount_t = get_gamma(mixture_idx_t, gamma_min, gamma_max, num_mixtures)

            intrinsic_rewards_tm1 = jax.tree_map(lambda x: x[:-1], data.extras['intrinsic_reward'])
            mixture_idx_tm1 = jax.tree_map(lambda x: x[:-1], data.extras['mixture_idx'])

            # make q - network input
            network_input_t = UVFANetworkInput(
                oar=observations_t,
                intrinsic_reward=intrinsic_rewards_tm1,
                mixture_idx=mixture_idx_tm1
            )

            online_q_t, _ = uvfa_unroll.apply(uvfa_params, key1, network_input_t, online_state)
            target_q_t, _ = uvfa_unroll.apply(uvfa_target_params, key2, network_input_t, target_state)

            # Get probability of actions under online policy
            pi_t = batched_epsilon_greedy_prob(online_q_t, target_epsilon)

            # Preprocess discounts.
            discounts_t = (done_t * discount_t).astype(online_q_t.dtype)

            # Get Retrace error and loss
            batch_retrace_error_fn = jax.vmap(
                functools.partial(
                    rlax.transformed_retrace,
                    lambda_=retrace_lambda,
                    tx_pair=tx_pair),
                in_axes=1,
                out_axes=1
            )

            batch_retrace_error = batch_retrace_error_fn(
                online_q_t[:-1],
                target_q_t[1:],
                actions_t[:-1],
                actions_t[1:],
                rewards_t[1:],
                discounts_t[1:],
                pi_t[1:],
                mu_t[1:],
            )
            batch_loss = 0.5 * jnp.square(batch_retrace_error).sum(axis=0)

            # Importance weighting.
            probs = sample.info.probability
            importance_weights = (1. / (probs + 1e-6)).astype(online_q_t.dtype)
            importance_weights **= importance_sampling_exponent
            importance_weights /= jnp.max(importance_weights)
            mean_loss = jnp.mean(importance_weights * batch_loss)

            return mean_loss, batch_retrace_error

        def idm_loss(
                idm_params: networks_lib.Params,
                key_grad: networks_lib.PRNGKey,
                sample: reverb.ReplaySample
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            data = sample.data

            observation_t = jax.tree_map(
                lambda x: x[:, 1:idm_clip_steps + 1],
                data.observation
            )  # [B, T1, ...]
            observation_tm1 = jax.tree_map(
                lambda x: x[:, :idm_clip_steps],
                data.observation
            )  # [B, T1, ...]
            action_tm1 = jax.tree_map(
                lambda x: x[:, :idm_clip_steps],
                data.action
            )

            batch_size, seq_length = action_tm1.shape
            action_tm1_pred_logits = idm_action_pred.apply(idm_params, key_grad,  # [T1, B, ...]
                                                           observation_tm1,
                                                           observation_t)
            action_tm1_pred_log_prob = jax.nn.log_softmax(action_tm1_pred_logits, axis=-1)
            action_tm1_one_hot_labels = jax.nn.one_hot(action_tm1, num_classes=action_tm1_pred_log_prob.shape[-1])

            loss = -jnp.sum(action_tm1_one_hot_labels * action_tm1_pred_log_prob) / (seq_length * batch_size)
            accuracy = jnp.mean(action_tm1 == jnp.argmax(action_tm1_pred_log_prob, axis=-1))
            return loss, accuracy

        def distillation_loss(
                distillation_params: networks_lib.Params,
                distillation_random_params: networks_lib.Params,
                key_grad: networks_lib.PRNGKey,
                sample: reverb.ReplaySample
        ) -> jnp.ndarray:
            data = sample.data
            key1, key2 = jax.random.split(key_grad)

            observation_t = jax.tree_map(
                lambda x: x[:, :distillation_clip_steps],
                data.observation
            )  # [B, ...]

            learnt_embeddings = distillation_embed.apply(
                distillation_params, key1, observation_t
            )
            random_embeddings = distillation_embed.apply(
                distillation_random_params, key2, observation_t
            )

            loss = jnp.mean(jnp.sum((random_embeddings - learnt_embeddings) ** 2, axis=-1))
            return loss

        def sgd_step(
                state: TrainingState,
                samples: reverb.ReplaySample
        ) -> Tuple[TrainingState, jnp.ndarray, Dict[str, jnp.ndarray]]:
            """Performs an update step, averaging over pmap replicas."""

            params = state.params
            opt_state = state.opt_state
            data = utils.batch_to_sequence(samples.data)

            # extract needed & preprocess data from structure
            mixture_idx_t = jax.tree_map(lambda x: x[1:], data.extras['mixture_idx'])
            beta_t = get_beta(mixture_idx_t, beta_min, beta_max, num_mixtures)
            intrinsic_rewards = jax.tree_map(lambda x: x[1:], data.extras['intrinsic_reward'])
            extrinsic_rewards = jax.tree_map(lambda x: x[1:], data.reward)
            if clip_rewards:
                extrinsic_rewards = jnp.clip(extrinsic_rewards, -max_abs_reward, max_abs_reward)

            # Compute loss, priorities and gradients.
            uvfa_grad_fn = jax.value_and_grad(uvfa_loss, has_aux=True)
            key, intrinsic_uvfa_key, extrinsic_uvfa_key, idm_key, distillation_key = jax.random.split(state.random_key,
                                                                                                      5)

            (intrinsic_uvfa_loss_value, intrinsic_batch_retrace_error), intrinsic_uvfa_gradients = uvfa_grad_fn(
                params.intrinsic_uvfa_params,
                params.intrinsic_uvfa_target_params,
                intrinsic_uvfa_key,
                samples,
                intrinsic_rewards,
                core_state_extraction_name='intrinsic_core_state')

            (extrinsic_uvfa_loss_value, extrinsic_batch_retrace_error), extrinsic_uvfa_gradients = uvfa_grad_fn(
                params.extrinsic_uvfa_params,
                params.extrinsic_uvfa_target_params,
                extrinsic_uvfa_key,
                samples,
                extrinsic_rewards)

            # Calculate priorities as a mixture of max and mean sequence errors from intrinsic & extrinsic errors.
            batch_retrace_error = extrinsic_batch_retrace_error + beta_t[1:] * intrinsic_batch_retrace_error
            abs_retrace_error = jnp.abs(batch_retrace_error).astype(batch_retrace_error.dtype)
            max_priority = max_priority_weight * jnp.max(abs_retrace_error, axis=0)
            mean_priority = (1 - max_priority_weight) * jnp.mean(abs_retrace_error, axis=0)
            priorities = (max_priority + mean_priority)

            # Average gradients over pmap replicas before optimizer update.
            intrinsic_uvfa_gradients = jax.lax.pmean(intrinsic_uvfa_gradients, _PMAP_AXIS_NAME)
            extrinsic_uvfa_gradients = jax.lax.pmean(extrinsic_uvfa_gradients, _PMAP_AXIS_NAME)

            # Apply optimizer updates.
            intrinsic_uvfa_updates, intrinsic_uvfa_new_opt_state = uvfa_optimizer.update(
                intrinsic_uvfa_gradients,
                opt_state.intrinsic_uvfa_opt_state)
            extrinsic_uvfa_updates, extrinsic_uvfa_new_opt_state = uvfa_optimizer.update(
                extrinsic_uvfa_gradients,
                opt_state.extrinsic_uvfa_opt_state)
            intrinsic_uvfa_new_params = optax.apply_updates(params.intrinsic_uvfa_params, intrinsic_uvfa_updates)
            extrinsic_uvfa_new_params = optax.apply_updates(params.extrinsic_uvfa_params, extrinsic_uvfa_updates)

            # Periodically update target networks.
            steps = state.steps + 1
            intrinsic_uvfa_new_target_params = rlax.periodic_update(
                intrinsic_uvfa_new_params, params.intrinsic_uvfa_target_params, steps, self._target_update_period)
            extrinsic_uvfa_new_target_params = rlax.periodic_update(
                extrinsic_uvfa_new_params, params.extrinsic_uvfa_target_params, steps, self._target_update_period)

            # update embedding network
            idm_grad_fn = jax.value_and_grad(idm_loss, has_aux=True)
            (idm_loss_value, idm_accuracy), idm_gradients = idm_grad_fn(params.idm_params,
                                                                        idm_key,
                                                                        samples)

            # Average gradients over pmap replicas before optimizer update.
            idm_gradients = jax.lax.pmean(idm_gradients, _PMAP_AXIS_NAME)

            # Apply optimizer updates.
            idm_updates, idm_new_opt_state = idm_optimizer.update(idm_gradients,
                                                                  opt_state.idm_opt_state,
                                                                  params.idm_params)
            idm_new_params = optax.apply_updates(params.idm_params, idm_updates)

            # update distillation network
            distillation_grad_fn = jax.value_and_grad(distillation_loss)
            distillation_loss_value, distillation_gradients = distillation_grad_fn(params.distillation_params,
                                                                                   params.distillation_random_params,
                                                                                   distillation_key,
                                                                                   samples)

            # Average gradients over pmap replicas before optimizer update.
            distillation_gradients = jax.lax.pmean(distillation_gradients, _PMAP_AXIS_NAME)

            # Apply optimizer updates.
            distillation_updates, distillation_new_opt_state = \
                distillation_optimizer.update(distillation_gradients,
                                              opt_state.distillation_opt_state,
                                              params.distillation_params)
            distillation_new_params = optax.apply_updates(params.distillation_params,
                                                          distillation_updates)

            new_params = dataclasses.replace(
                params,
                intrinsic_uvfa_params=intrinsic_uvfa_new_params,
                extrinsic_uvfa_params=extrinsic_uvfa_new_params,
                intrinsic_uvfa_target_params=intrinsic_uvfa_new_target_params,
                extrinsic_uvfa_target_params=extrinsic_uvfa_new_target_params,
                idm_params=idm_new_params,
                distillation_params=distillation_new_params,
            )

            new_opt_state = dataclasses.replace(
                opt_state,
                intrinsic_uvfa_opt_state=intrinsic_uvfa_new_opt_state,
                extrinsic_uvfa_opt_state=extrinsic_uvfa_new_opt_state,
                idm_opt_state=idm_new_opt_state,
                distillation_opt_state=distillation_new_opt_state
            )

            new_state = TrainingState(
                params=new_params,
                opt_state=new_opt_state,
                steps=steps,
                random_key=key)
            return new_state, priorities, {
                'intrinsic UVFA loss': intrinsic_uvfa_loss_value,
                'extrinsic UVFA loss': extrinsic_uvfa_loss_value,
                'IDM loss': idm_loss_value,
                'IDM accuracy': idm_accuracy,
                'RND loss': distillation_loss_value
            }

        def update_priorities(
                keys_and_priorities: Tuple[jnp.ndarray, jnp.ndarray]):
            keys, priorities = keys_and_priorities
            keys, priorities = tree.map_structure(
                # Fetch array and combine device and batch dimensions.
                lambda x: utils.fetch_devicearray(x).reshape((-1,) + x.shape[2:]),
                (keys, priorities))
            replay_client.mutate_priorities(  # pytype: disable=attribute-error
                table=adders.DEFAULT_PRIORITY_TABLE,
                updates=dict(zip(keys, priorities)))

        # Internalise components, hyperparameters, logger, counter, and methods.
        self._replay_client = replay_client
        self._target_update_period = target_update_period
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            'learner',
            asynchronous=True,
            serialize_fn=utils.fetch_devicearray,
            time_delta=1.)

        self._sgd_step = jax.pmap(sgd_step, axis_name=_PMAP_AXIS_NAME)
        self._async_priority_updater = async_utils.AsyncExecutor(update_priorities)

        # Initialise Recurrent network state
        random_key, intrinsic_key_initial_1, extrinsic_key_initial_1, \
            intrinsic_key_initial_2, extrinsic_key_initial_2 = jax.random.split(random_key, 5)
        intrinsic_uvfa_initial_state_params = uvfa_initial_state.init(intrinsic_key_initial_1, batch_size)
        extrinsic_uvfa_initial_state_params = uvfa_initial_state.init(extrinsic_key_initial_1, batch_size)
        intrinsic_uvfa_initial_state = uvfa_initial_state.apply(intrinsic_uvfa_initial_state_params,
                                                                intrinsic_key_initial_2, batch_size)
        extrinsic_uvfa_initial_state = uvfa_initial_state.apply(extrinsic_uvfa_initial_state_params,
                                                                extrinsic_key_initial_2, batch_size)

        # Initialise and internalise training state (parameters/optimiser state).
        random_key, intrinsic_key_init, extrinsic_key_init = jax.random.split(random_key, 3)
        intrinsic_uvfa_initial_params = uvfa_unroll.init(intrinsic_key_init, intrinsic_uvfa_initial_state)
        intrinsic_uvfa_opt_state = uvfa_optimizer.init(intrinsic_uvfa_initial_params)
        extrinsic_uvfa_initial_params = uvfa_unroll.init(extrinsic_key_init, extrinsic_uvfa_initial_state)
        extrinsic_uvfa_opt_state = uvfa_optimizer.init(extrinsic_uvfa_initial_params)

        random_key, key_init = jax.random.split(random_key)
        idm_initial_params = idm_action_pred.init(key_init)
        idm_opt_state = idm_optimizer.init(idm_initial_params)

        random_key, key_init, key_init_random = jax.random.split(random_key, 3)
        distillation_initial_params = distillation_embed.init(key_init)
        distillation_initial_random_params = distillation_embed.init(key_init_random)
        distillation_opt_state = distillation_optimizer.init(distillation_initial_params)

        initial_params = DRLearnerNetworksParams(
            intrinsic_uvfa_params=intrinsic_uvfa_initial_params,
            extrinsic_uvfa_params=extrinsic_uvfa_initial_params,
            intrinsic_uvfa_target_params=intrinsic_uvfa_initial_params,
            extrinsic_uvfa_target_params=extrinsic_uvfa_initial_params,
            idm_params=idm_initial_params,
            distillation_params=distillation_initial_params,
            distillation_random_params=distillation_initial_random_params)

        opt_state = DRLearnerNetworksOptStates(
            intrinsic_uvfa_opt_state=intrinsic_uvfa_opt_state,
            extrinsic_uvfa_opt_state=extrinsic_uvfa_opt_state,
            idm_opt_state=idm_opt_state,
            distillation_opt_state=distillation_opt_state)

        state = TrainingState(
            params=initial_params,
            opt_state=opt_state,
            steps=jnp.array(0),
            random_key=random_key)
        # Replicate parameters.
        self._state = utils.replicate_in_all_devices(state)

        # Shard multiple inputs with on-device prefetching.
        # We split samples in two outputs, the keys which need to be kept on-host
        # since int64 arrays are not supported in TPUs, and the entire sample
        # separately so it can be sent to the sgd_step method.
        def split_sample(sample: reverb.ReplaySample) -> utils.PrefetchingSplit:
            return utils.PrefetchingSplit(host=sample.info.key, device=sample)

        self._prefetched_iterator = utils.sharded_prefetch(
            iterator,
            buffer_size=prefetch_size,
            num_threads=jax.local_device_count(),
            split_fn=split_sample)

    def step(self):
        prefetching_split = next(self._prefetched_iterator)
        # The split_sample method passed to utils.sharded_prefetch specifies what
        # parts of the objects returned by the original iterator are kept in the
        # host and what parts are prefetched on-device.
        # In this case the host property of the prefetching split contains only the
        # replay keys and the device property is the prefetched full original
        # sample.
        keys, samples = prefetching_split.host, prefetching_split.device

        data = samples.data

        # Do a batch of SGD.
        start = time.time()
        self._state, priorities, metrics = self._sgd_step(self._state, samples)
        # Take metrics from first replica.
        metrics = utils.get_from_first_device(metrics)
        # Update our counts and record it.
        counts = self._counter.increment(steps=1, time_elapsed=time.time() - start)

        # Update priorities in replay.
        if self._replay_client:
            self._async_priority_updater.put((keys, priorities))

        # Attempt to write logs.
        self._logger.write({**metrics, **counts})

    def get_variables(self, names: List[str]) -> Sequence[DRLearnerNetworksParams]:
        # Return first replica of parameters.
        return [utils.get_from_first_device(self._state.params)]

    def save(self) -> TrainingState:
        # Serialize only the first replica of parameters and optimizer state.
        return jax.tree_map(utils.get_from_first_device, self._state)

    def restore(self, state: TrainingState):
        self._state = utils.replicate_in_all_devices(state)
