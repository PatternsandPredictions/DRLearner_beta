from typing import NamedTuple, Optional

import chex
import jax.numpy as jnp
import optax
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from rlax._src.exploration import IntrinsicRewardState

from .lifelong_curiosity import LifelongCuriosityModulationState


@chex.dataclass(frozen=True, mappable_dataclass=False)
class DRLearnerNetworksParams:
    """Collection of all parameters of Neural Networks used by DRLearner Agent"""
    intrinsic_uvfa_params: networks_lib.Params  # intrinsic Universal Value-Function Approximator
    extrinsic_uvfa_params: networks_lib.Params  # extrinsic Universal Value-Function Approximator
    intrinsic_uvfa_target_params: networks_lib.Params  # intrinsic UVFA target network
    extrinsic_uvfa_target_params: networks_lib.Params  # extrinsic UVFA target network
    idm_params: networks_lib.Params  # Inverse Dynamics Model
    distillation_params: networks_lib.Params  # Distillation Network
    distillation_random_params: networks_lib.Params  # Random Distillation Network


@chex.dataclass(frozen=True, mappable_dataclass=False)
class DRLearnerNetworksOptStates:
    """Collection of optimizer states for all networks trained by the Learner"""
    intrinsic_uvfa_opt_state: optax.OptState
    extrinsic_uvfa_opt_state: optax.OptState
    idm_opt_state: optax.OptState
    distillation_opt_state: optax.OptState


class TrainingState(NamedTuple):
    """DRLearner Learner training state"""
    params: DRLearnerNetworksParams
    opt_state: DRLearnerNetworksOptStates
    steps: jnp.ndarray
    random_key: networks_lib.PRNGKey


@chex.dataclass(frozen=True, mappable_dataclass=False)
class MetaControllerState:
    episode_returns_history: jnp.ndarray
    episode_count: jnp.ndarray
    current_episode_return: jnp.ndarray
    mixture_idx_history: jnp.ndarray
    beta: jnp.ndarray
    gamma: jnp.ndarray
    is_eval: bool
    num_eval_episodes: jnp.ndarray


@chex.dataclass(frozen=True, mappable_dataclass=False)
class DRLearnerActorState:
    rng: networks_lib.PRNGKey
    epsilon: jnp.ndarray
    mixture_idx: jnp.ndarray
    intrinsic_recurrent_state: actor_core_lib.RecurrentState
    extrinsic_recurrent_state: actor_core_lib.RecurrentState
    prev_intrinsic_reward: jnp.ndarray
    prev_action_prob: jnp.ndarray
    prev_alpha: jnp.ndarray
    meta_controller_state: MetaControllerState
    lifelong_modulation_state: Optional[LifelongCuriosityModulationState] = None
    intrinsic_reward_state: Optional[IntrinsicRewardState] = None

