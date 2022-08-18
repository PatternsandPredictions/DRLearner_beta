from typing import Optional

import chex
import jax.numpy as jnp


@chex.dataclass
class LifelongCuriosityModulationState:
    distance_mean: chex.Scalar = 0.
    distance_var: chex.Scalar = 1.


def lifelong_curiosity_modulation(
        learnt_embeddings: chex.Array,
        random_embeddings: chex.Array,
        max_modulation: float = 5.0,
        lifelong_modulation_state: Optional[LifelongCuriosityModulationState] = None,
        ma_coef: float = 0.0001):
    if not lifelong_modulation_state:
        lifelong_modulation_state = LifelongCuriosityModulationState()

    error = jnp.sum((learnt_embeddings - random_embeddings) ** 2, axis=-1)

    distance_mean = lifelong_modulation_state.distance_mean
    distance_var = lifelong_modulation_state.distance_var
    # exponentially weighted moving average and std
    distance_var = (1 - ma_coef) * (distance_var + ma_coef * jnp.mean(error - distance_mean) ** 2)
    distance_mean = ma_coef * jnp.mean(error) + (1 - ma_coef) * distance_mean

    alpha = 1 + (error - distance_mean) / jnp.sqrt(distance_var)
    alpha = jnp.clip(alpha, 1., max_modulation)

    lifelong_modulation_state = LifelongCuriosityModulationState(
        distance_mean=distance_mean,
        distance_var=distance_var
    )

    return alpha, lifelong_modulation_state
