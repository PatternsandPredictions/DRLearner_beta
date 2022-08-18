import dataclasses

import haiku as hk
import jax.numpy as jnp
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.wrappers.observation_action_reward import OAR


@dataclasses.dataclass
class EmbeddingNetwork:
    """Pure functions for DRLearner embedding network"""
    predict_action: networks_lib.FeedForwardNetwork
    embed: networks_lib.FeedForwardNetwork


def make_embedding_net(
        make_embedding_modules,
        env_spec) -> EmbeddingNetwork:
    def embed_fn(observation: OAR) -> networks_lib.NetworkOutput:
        """
        Embed batch of observations
        Args:
            observation: jnp.array representing a batch of observations [B, ...]

        Returns:
            embedding vectors [B, D]
        """
        embedding_torso, _ = make_embedding_modules()
        return embedding_torso(observation.observation)

    def predict_action_fn(observation_tm1: OAR, observation_t: OAR) -> networks_lib.NetworkOutput:
        """
        Embed batch of sequences two consecutive observations x_{t_1} and x{t} and predict batch of actions a_{t-1}
        Args:
            observation_tm1: observation x_{t-1} [T, B, ...]
            observation_t:  observation x_{t} [T, B, ...]

        Returns:
            prediction logits for discrete action a_{t_1} [T, B, N]
        """
        embedding_torso, pred_head = make_embedding_modules()
        emb_tm1 = hk.BatchApply(embedding_torso)(observation_tm1.observation)
        emb_t = hk.BatchApply(embedding_torso)(observation_t.observation)
        return hk.BatchApply(pred_head)(jnp.concatenate([emb_tm1, emb_t], axis=-1))

    # transform functions
    embed_hk = hk.transform(embed_fn)
    predict_action_hk = hk.transform(predict_action_fn)

    # create dummy batches for networks initialization
    observation_sequences = utils.add_batch_dim(
        utils.add_batch_dim(
            utils.zeros_like(env_spec.observations)
        )
    )  # [T=1, B=1, ...]

    def predict_action_init(rng):
        return predict_action_hk.init(rng, observation_sequences, observation_sequences)

    # create FeedForwardNetworks corresponding to embed and action prediction functions
    predict_action = networks_lib.FeedForwardNetwork(
        init=predict_action_init, apply=predict_action_hk.apply
    )
    embed = networks_lib.FeedForwardNetwork(
        init=embed_hk.init, apply=embed_hk.apply
    )

    return EmbeddingNetwork(predict_action, embed)
