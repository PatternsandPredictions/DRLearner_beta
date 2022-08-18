import dataclasses
from typing import Tuple, NamedTuple

import haiku as hk
import jax.numpy as jnp
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.wrappers.observation_action_reward import OAR


@dataclasses.dataclass
class UVFANetwork:
    """Pure functions for DRLearner Universal Value Function Approximator"""
    initial_state: networks_lib.FeedForwardNetwork
    forward: networks_lib.FeedForwardNetwork
    unroll: networks_lib.FeedForwardNetwork


class UVFANetworkInput(NamedTuple):
    """Wrap input specific to DRLearner Recurrent Q-network"""
    oar: OAR  # observation_t, action_tm1, reward_tm1
    intrinsic_reward: jnp.ndarray  # ri_tm1
    mixture_idx: jnp.ndarray  # beta_idx_tm1


def make_uvfa_net(
        make_uvfa_modules,
        batch_size: int,
        env_spec) -> UVFANetwork:
    def initial_state(batch_size: int):
        _, recurrent_core, _ = make_uvfa_modules()
        return recurrent_core.initial_state(batch_size)

    def forward(input: UVFANetworkInput,
                state: hk.LSTMState) -> Tuple[networks_lib.NetworkOutput, hk.LSTMState]:
        """
        Estimate action values for batch of inputs
        Args:
            input: batch of observations, actions, rewards, intrinsic rewards
                   and mixture indices (beta param labels)
            state: recurrent state
        Returns:
            q_values: predicted action values
            new_state: new recurrent state after prediction
        """
        embedding_torso, recurrent_core, head = make_uvfa_modules()

        embeddings = embedding_torso(input)
        embeddings, new_state = recurrent_core(embeddings, state)
        q_values = head(embeddings)
        return q_values, new_state

    def unroll(input: UVFANetworkInput,
               state: hk.LSTMState) -> Tuple[networks_lib.NetworkOutput, hk.LSTMState]:
        """
        Estimate action values for batch of input sequences
        Args:
            input: batch of observations, actions, rewards, intrinsic rewards
                   and mixture indices (beta param labels) sequences
            state: recurrent state
        Returns:
            q_values: predicted action values
            new_state: new recurrent state after prediction
        """
        embedding_torso, recurrent_core, head = make_uvfa_modules()

        embeddings = hk.BatchApply(embedding_torso)(input)
        embeddings, new_states = hk.static_unroll(recurrent_core, embeddings, state)
        q_values = hk.BatchApply(head)(embeddings)
        return q_values, new_states

    # transform functions
    initial_state_hk = hk.transform(initial_state)
    forward_hk = hk.transform(forward)
    unroll_hk = hk.transform(unroll)

    # create dummy batches for networks initialization
    observation = utils.zeros_like(env_spec.observations)
    intrinsic_reward = utils.zeros_like(env_spec.rewards)
    mixture_idxs = utils.zeros_like(env_spec.rewards, dtype=jnp.int32)
    uvfa_input_sequences = utils.add_batch_dim(
        utils.tile_nested(
            UVFANetworkInput(observation, intrinsic_reward, mixture_idxs),
            batch_size
        )
    )

    def initial_state_init(rng, batch_size: int):
        return initial_state_hk.init(rng, batch_size)

    def unroll_init(rng, initial_state):
        return unroll_hk.init(rng, uvfa_input_sequences, initial_state)

    # create FeedForwardNetworks corresponding to UVFA pure functions
    initial_state = networks_lib.FeedForwardNetwork(
        init=initial_state_init, apply=initial_state_hk.apply
    )
    forward = networks_lib.FeedForwardNetwork(
        init=forward_hk.init, apply=forward_hk.apply
    )
    unroll = networks_lib.FeedForwardNetwork(
        init=unroll_init, apply=unroll_hk.apply
    )

    return UVFANetwork(initial_state, forward, unroll)
