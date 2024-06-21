import haiku as hk
import jax
import jax.numpy as jnp


class UVFATorso(hk.Module):
    def __init__(self,
                 observation_embedding_torso: hk.Module,
                 num_actions: int,
                 num_mixtures: int,
                 name: str):
        super().__init__(name=name)
        self._embed = observation_embedding_torso

        self._num_actions = num_actions
        self._num_mixtures = num_mixtures

    def __call__(self, input):
        oar_t, intrinsic_reward_tm1, mixture_idx_tm1 = input.oar, input.intrinsic_reward, input.mixture_idx,
        observation_t, action_tm1, reward_tm1 = oar_t.observation, oar_t.action, oar_t.reward

        features_t = self._embed(observation_t)  # [T?, B, D]
        action_tm1 = jax.nn.one_hot(
            action_tm1,
            num_classes=self._num_actions
        )  # [T?, B, A]
        mixture_idx_tm1 = jax.nn.one_hot(
            mixture_idx_tm1,
            num_classes=self._num_mixtures
        )  # [T?, B, M]

        reward_tm1 = jnp.tanh(reward_tm1)
        intrinsic_reward_tm1 = jnp.tanh(intrinsic_reward_tm1)
        # Add dummy trailing dimensions to the rewards if necessary.
        while reward_tm1.ndim < action_tm1.ndim:
            reward_tm1 = jnp.expand_dims(reward_tm1, axis=-1)

        while intrinsic_reward_tm1.ndim < action_tm1.ndim:
            intrinsic_reward_tm1 = jnp.expand_dims(intrinsic_reward_tm1, axis=-1)

        embedding = jnp.concatenate(
            [features_t, action_tm1, reward_tm1, intrinsic_reward_tm1, mixture_idx_tm1],
            axis=-1
        )  # [T?, B, D+A+M+2]
        return embedding
