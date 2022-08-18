import dataclasses

import haiku as hk
import jax
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.wrappers.observation_action_reward import OAR


@dataclasses.dataclass
class DistillationNetwork:
    """Pure functions for DRLearner distillation network"""
    embed: networks_lib.FeedForwardNetwork
    embed_sequence: networks_lib.FeedForwardNetwork


def make_distillation_net(
        make_distillation_modules,
        env_spec) -> DistillationNetwork:
    def embed_fn(observation: OAR) -> networks_lib.NetworkOutput:
        """
        Embed batch of observations
        Args:
            observation: jnp.array representing a batch of observations [B, ...]

        Returns:
            embedding vectors [B, D]
        """
        embedding_torso = make_distillation_modules()
        return embedding_torso(observation.observation)

    # transform functions
    embed_hk = hk.transform(embed_fn)

    # create dummy batches for networks initialization
    observation_batch = utils.add_batch_dim(
        utils.zeros_like(env_spec.observations)
    )  # [B=1, ...]

    def embed_init(rng):
        return embed_hk.init(rng, observation_batch)

    embed = networks_lib.FeedForwardNetwork(
        init=embed_init, apply=embed_hk.apply
    )
    embed_sequence = networks_lib.FeedForwardNetwork(
        init=embed_init,
        # vmap over 1-st parameter: apply(params, random_key, data, ...)
        apply=jax.vmap(embed_hk.apply, in_axes=(None, None, 0), out_axes=0)
    )

    return DistillationNetwork(embed, embed_sequence)
