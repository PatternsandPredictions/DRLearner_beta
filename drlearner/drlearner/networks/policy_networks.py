import dataclasses
from typing import Callable

import jax.nn
import jax.numpy as jnp
import jax.random
import rlax
from acme import types
from acme.jax import networks as networks_lib

from .networks import DRLearnerNetworks
from ..config import DRLearnerConfig


@dataclasses.dataclass
class DRLearnerPolicyNetworks:
    """Pure functions used by DRLearner actors"""
    select_action: Callable
    embed_observation: Callable
    distillation_embed_observation: Callable


def make_policy_networks(
        networks: DRLearnerNetworks,
        config: DRLearnerConfig,
        evaluation: bool = False):
    def select_action(intrinsic_params: networks_lib.Params,
                      extrinsic_params: networks_lib.Params,
                      key: networks_lib.PRNGKey,
                      observation: types.NestedArray,
                      intrinsic_core_state: types.NestedArray,
                      extrinsic_core_state: types.NestedArray,
                      epsilon, beta):
        intrinsic_key_qnet, extrinsic_key_qnet, key_sample = jax.random.split(key, 3)
        intrinsic_q_values, intrinsic_core_state = networks.uvfa_net.forward.apply(
            intrinsic_params, intrinsic_key_qnet, observation, intrinsic_core_state)
        extrinsic_q_values, extrinsic_core_state = networks.uvfa_net.forward.apply(
            extrinsic_params, extrinsic_key_qnet, observation, extrinsic_core_state)

        q_values = config.tx_pair.apply(beta * config.tx_pair.apply_inv(intrinsic_q_values) +
                                        config.tx_pair.apply_inv(extrinsic_q_values))
        epsilon = config.evaluation_epsilon if evaluation else epsilon
        action_dist = rlax.epsilon_greedy(epsilon)
        action = action_dist.sample(key_sample, q_values)
        action_prob = action_dist.probs(
            jax.nn.one_hot(jnp.argmax(q_values, axis=-1), num_classes=q_values.shape[-1])
        )
        action_prob = jnp.squeeze(action_prob[:, action], axis=-1)
        return action, action_prob, intrinsic_core_state, extrinsic_core_state

    return DRLearnerPolicyNetworks(
        select_action=select_action,
        embed_observation=networks.embedding_net.embed.apply,
        distillation_embed_observation=networks.distillation_net.embed.apply
    )
