"""DRLearner actor."""
import dataclasses
from typing import Callable
from typing import Mapping, Optional

import chex
import dm_env
import jax
import jax.numpy as jnp
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils


from rlax import episodic_memory_intrinsic_rewards
from rlax._src.exploration import IntrinsicRewardState

import tree
import operator

from .networks import DRLearnerPolicyNetworks, UVFANetworkInput
from .drlearner_types import DRLearnerNetworksParams, DRLearnerActorState, MetaControllerState
from .config import DRLearnerConfig
from .utils import get_beta_ngu, get_gamma_ngu, get_epsilon
from .lifelong_curiosity import lifelong_curiosity_modulation, LifelongCuriosityModulationState

@dataclasses.dataclass
class DRLearnerActorCore(actor_core_lib.ActorCore):
    observe: Callable
    observe_first: Callable


def get_actor_core(
        policy_networks: DRLearnerPolicyNetworks,
        intrinsic_initial_core_state: actor_core_lib.RecurrentState,
        extrinsic_initial_core_state: actor_core_lib.RecurrentState,
        actor_id: int,
        num_actors_per_mixture: int,
        config: DRLearnerConfig,
        jit: bool = True
) -> DRLearnerActorCore:
    """Returns ActorCore for DRLearner."""

    def select_action(params: DRLearnerNetworksParams,
                      observation: networks_lib.Observation,
                      state: DRLearnerActorState):
        # TODO(b/161332815): Make JAX Actor work with batched or unbatched inputs.
        rng, policy_rng = jax.random.split(state.rng)
        intrinsic_params = params.intrinsic_uvfa_params
        extrinsic_params = params.extrinsic_uvfa_params

        policy_input = UVFANetworkInput(observation, state.prev_intrinsic_reward, state.mixture_idx)
        policy_input = utils.add_batch_dim(policy_input)
        intrinsic_recurrent_state = utils.add_batch_dim(state.intrinsic_recurrent_state)
        extrinsic_recurrent_state = utils.add_batch_dim(state.extrinsic_recurrent_state)

        beta = get_beta_ngu(state.mixture_idx, config.beta_min, config.beta_max, config.num_mixtures)

        action, action_prob, intrinsic_new_recurrent_state, extrinsic_new_recurrent_state = utils.squeeze_batch_dim(
            policy_networks.select_action(intrinsic_params, extrinsic_params, policy_rng, policy_input,
                                          intrinsic_recurrent_state, extrinsic_recurrent_state, state.epsilon, beta))

        return action, dataclasses.replace(state, rng=rng,
                                           prev_action_prob=action_prob,
                                           intrinsic_recurrent_state=intrinsic_new_recurrent_state,
                                           extrinsic_recurrent_state=extrinsic_new_recurrent_state)

    intrinsic_initial_core_state = utils.squeeze_batch_dim(intrinsic_initial_core_state)
    extrinsic_initial_core_state = utils.squeeze_batch_dim(extrinsic_initial_core_state)

    def update_meta_controller(params: DRLearnerNetworksParams,
                               timestep: dm_env.TimeStep,
                               state: DRLearnerActorState) -> DRLearnerActorState:

        def get_ucb_bounds(rewards, mixture_idx_history):
            bound = jnp.array([jnp.mean(rewards[jnp.where(mixture_idx_history == a)[0]]) +
                       config.mc_beta * jnp.sqrt(1 / (len(jnp.where(mixture_idx_history == a)[0]) + 1))
                       for a in range(config.num_mixtures)])
            bound = bound.at[jnp.where(jnp.isnan(bound))[0]].set(-jnp.inf)
            return bound

        def get_greedy_bounds(rewards):
            bound = jnp.array([jnp.mean(rewards[jnp.where(mixture_idx_history == a)[0]])
                               for a in range(config.num_mixtures)])
            bound = bound.at[jnp.where(jnp.isnan(bound))[0]].set(-jnp.inf)
            return bound

        # if jit:
        #     get_ucb_bounds = jax.jit(get_ucb_bounds)
        #     get_greedy_bounds = jax.jit(get_greedy_bounds)

        rng, bandit_rng = jax.random.split(state.rng, 2)
        is_eval = state.meta_controller_state.is_eval
        returns = state.meta_controller_state.episode_returns_history
        mixture_idx_history = state.meta_controller_state.mixture_idx_history

        episode_count = state.meta_controller_state.episode_count + 1

        if episode_count > 0 and not is_eval:  # Actor has played at least 1 episode
            mixture_idx_history = jnp.concatenate(
                            (mixture_idx_history[1:], jnp.expand_dims(state.mixture_idx, axis=0)),
                                           axis=0)
            returns = jnp.concatenate(
                             (returns[1:], jnp.expand_dims(state.meta_controller_state.current_episode_return, axis=0)),
                               axis=0)
        if episode_count % state.meta_controller_state.num_eval_episodes == 0:
            is_eval = not is_eval

        if is_eval:  # Greedy arm selection and no random policy
            if episode_count < config.num_mixtures:
                policy_indx = jnp.array(episode_count, dtype=jnp.int32)
            else:
                bounds = get_greedy_bounds(returns)
                policy_indx = jnp.argmax(bounds).astype(jnp.int32)
        else:  # UCB policy selection
            if episode_count < config.num_mixtures:
                policy_indx = jnp.array(episode_count, dtype=jnp.int32)
            elif jax.random.uniform(bandit_rng) < config.mc_epsilon:
                policy_indx = jax.random.choice(bandit_rng, a=config.num_mixtures).astype(jnp.int32)
            else:
                bounds = get_ucb_bounds(returns, mixture_idx_history)
                policy_indx = jnp.argmax(bounds).astype(jnp.int32)
            
        new_beta = get_beta_ngu(policy_indx, config.beta_min, config.beta_max, config.num_mixtures)

        new_gamma = get_gamma_ngu(policy_indx, config.gamma_min, config.gamma_max, config.num_mixtures)

        mc_state = dataclasses.replace(state.meta_controller_state, beta=new_beta, gamma=new_gamma,
                                       episode_count=episode_count,
                                       episode_returns_history=returns,
                                       current_episode_return=jnp.array(0),
                                       mixture_idx_history=mixture_idx_history,
                                       is_eval=is_eval,
                                       )
        state = dataclasses.replace(state, rng=rng, mixture_idx=policy_indx, meta_controller_state=mc_state,)
        return state

    def update_meta_contoller_rewards(params: DRLearnerNetworksParams,
                                      timestep: dm_env.TimeStep,
                                      state: DRLearnerActorState) -> DRLearnerActorState:

        episode_return = tree.map_structure(operator.iadd,
                                            state.meta_controller_state.current_episode_return,
                                            timestep.reward
                                            )
        mc_state = dataclasses.replace(state.meta_controller_state, current_episode_return=episode_return)

        state = dataclasses.replace(state, meta_controller_state=mc_state)
        return state

    def compute_intrinsic_reward(params: DRLearnerNetworksParams,
                                 timestep: dm_env.TimeStep,
                                 state: DRLearnerActorState) -> DRLearnerActorState:
        observation = utils.add_batch_dim(timestep.observation)

        # compute modulation of reward
        rng, random_net_rng, learnt_net_rng = jax.random.split(state.rng, 3)
        learnt_embedding = policy_networks.distillation_embed_observation(
            params.distillation_params, learnt_net_rng, observation
        )
        random_embedding = policy_networks.distillation_embed_observation(
            params.distillation_random_params, random_net_rng, observation
        )
        alpha, lifelong_modulation_state = lifelong_curiosity_modulation(
            learnt_embedding, random_embedding,
            max_modulation=config.max_lifelong_modulation,
            lifelong_modulation_state=state.lifelong_modulation_state,
            ma_coef=config.distillation_moving_average_coef
        )
        alpha = utils.squeeze_batch_dim(alpha)

        # compute episodic intrinsic reward
        rng, embed_net_rng = jax.random.split(rng)
        embedding = policy_networks.embed_observation(params.idm_params, embed_net_rng, observation)
        intrinsic_reward, intrinsic_reward_state = episodic_memory_intrinsic_rewards(
            embedding,
            num_neighbors=config.episodic_memory_num_neighbors,
            reward_scale=alpha,
            intrinsic_reward_state=state.intrinsic_reward_state,
            constant=config.episodic_memory_pseudo_counts,
            epsilon=config.episodic_memory_epsilon,
            cluster_distance=config.episodic_memory_cluster_distance,
            max_similarity=config.episodic_memory_max_similarity,
            max_memory_size=config.episodic_memory_max_size
        )
        intrinsic_reward = utils.squeeze_batch_dim(intrinsic_reward)

        state = dataclasses.replace(
            state, rng=rng,
            prev_intrinsic_reward=intrinsic_reward,
            intrinsic_reward_state=intrinsic_reward_state,
            prev_alpha=alpha,
            lifelong_modulation_state=lifelong_modulation_state,
        )
        return state

    def observe_first(params: DRLearnerNetworksParams,
                      timestep: dm_env.TimeStep,
                      state: DRLearnerActorState) -> DRLearnerActorState:
        state = update_meta_controller(params, timestep, state)
        if jit:
            # Fix for jit compatibility.
            return jax.jit(compute_intrinsic_reward, device=jax.devices('cpu')[0])(params, timestep, state)
        else:
            return compute_intrinsic_reward(params, timestep, state)

    def observe(params: DRLearnerNetworksParams,
                action: Optional[networks_lib.Action],
                next_timestep: dm_env.TimeStep,
                state: DRLearnerActorState) -> DRLearnerActorState:
        state = update_meta_contoller_rewards(params, next_timestep, state)
        return compute_intrinsic_reward(params, next_timestep, state)

    def init(rng: networks_lib.PRNGKey, mixture_idx: jnp.ndarray, state: Optional[DRLearnerActorState]) -> DRLearnerActorState:
        """Initialize actor state at the beginning of the episode.
        Optionally, use some parameters from the state at the end of previous  episode"""
        epsilon = get_epsilon(actor_id, config.epsilon, num_actors=num_actors_per_mixture * config.num_mixtures)

        if state is None:
            mc_state = MetaControllerState(
                episode_returns_history=jnp.empty((config.window,)),
                mixture_idx_history=jnp.full((config.window,), fill_value=-1),
                episode_count=-1,
                current_episode_return=jnp.array(-1000000),
                beta=jnp.array(0.),
                gamma=jnp.array(0.),
                is_eval=True,
                num_eval_episodes=jnp.array(1.7)  # No eval episodes for actors
            )
            state = DRLearnerActorState(
                rng, epsilon,
                intrinsic_recurrent_state=intrinsic_initial_core_state,
                extrinsic_recurrent_state=extrinsic_initial_core_state,
                mixture_idx=jnp.array(0., jnp.int32),
                prev_intrinsic_reward=jnp.array(0.),
                prev_action_prob=jnp.array(0.),
                prev_alpha=jnp.array(1.),
                meta_controller_state=mc_state,
            )
        else:
            state = DRLearnerActorState(
                rng, epsilon,
                intrinsic_recurrent_state=intrinsic_initial_core_state,
                extrinsic_recurrent_state=extrinsic_initial_core_state,
                mixture_idx=state.mixture_idx,
                prev_intrinsic_reward=jnp.array(0.),
                prev_action_prob=jnp.array(0.),
                prev_alpha=jnp.array(1.),
                meta_controller_state=state.meta_controller_state,
                lifelong_modulation_state=state.lifelong_modulation_state,  # keep lifelong state,
                intrinsic_reward_state=None,  # restart episodic memory

            )
        return state

    def get_extras(state: DRLearnerActorState) -> Mapping[str, jnp.ndarray]:
        return {
            'intrinsic_core_state': state.intrinsic_recurrent_state,
            'extrinsic_core_state': state.extrinsic_recurrent_state,
            'intrinsic_reward': state.prev_intrinsic_reward,
            'mixture_idx': state.mixture_idx,
            'behavior_action_prob': state.prev_action_prob,
            'is_eval': jnp.array(state.meta_controller_state.is_eval, dtype=jnp.int32),
            'alpha': state.prev_alpha
        }

    return DRLearnerActorCore(init=init, select_action=select_action,
                              observe=observe, observe_first=observe_first,
                              get_extras=get_extras)
