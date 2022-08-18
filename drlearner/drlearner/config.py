"""DRLearner config."""
import dataclasses

import rlax
from acme.adders import reverb as adders_reverb


@dataclasses.dataclass
class DRLearnerConfig:
    """Configuration options for DRLearner agent."""
    gamma_min: float = 0.99
    gamma_max: float = 0.997
    num_mixtures: int = 32
    target_update_period: int = 400
    evaluation_epsilon: float = 0.01
    epsilon: float = 0.01
    actor_epsilon: float = 0.01
    target_epsilon: float = 0.01
    variable_update_period: int = 400

    # Learner options
    retrace_lambda: float = 0.95
    burn_in_length: int = 40
    trace_length: int = 80
    sequence_period: int = 40
    num_sgd_steps_per_step: int = 1
    uvfa_learning_rate: float = 1e-4
    idm_learning_rate: float = 5e-4
    distillation_learning_rate: float = 5e-4
    idm_weight_decay: float = 1e-5
    distillation_weight_decay: float = 1e-5
    idm_clip_steps: int = 5
    distillation_clip_steps: int = 5
    clip_rewards: bool = False
    max_absolute_reward: float = 1.0
    tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR
    distillation_moving_average_coef: float = 1e-3

    # Intrinsic reward multipliers
    beta_min: float = 0.
    beta_max: float = 0.3

    # Embedding network options
    observation_embed_dim: int = 128
    episodic_memory_num_neighbors: int = 10
    episodic_memory_max_size: int = 30_000
    episodic_memory_max_similarity: float = 8.
    episodic_memory_cluster_distance: float = 8e-3
    episodic_memory_pseudo_counts: float = 1e-3
    episodic_memory_epsilon: float = 1e-4

    # Distillation network
    distillation_embed_dim: int = 128
    max_lifelong_modulation: float = 5.0

    # Replay options
    samples_per_insert_tolerance_rate: float = 0.1
    samples_per_insert: float = 4.0
    min_replay_size: int = 50_000
    max_replay_size: int = 100_000
    batch_size: int = 64
    prefetch_size: int = 2
    num_parallel_calls: int = 16
    replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE

    # Priority options
    importance_sampling_exponent: float = 0.6
    priority_exponent: float = 0.9
    max_priority_weight: float = 0.9

    # Meta Controller options
    window: int = 160
    actor_window: int = 160
    evaluation_window: int = 3600
    n_arms: int = 32
    mc_epsilon: float = 0.5  # Value is set from actor_mc_espilon or evaluation_mc_epsilon depending on whether the actor acts as evaluator
    actor_mc_epsilon: float = 0.5
    evaluation_mc_epsilon: float = 0.01
    mc_beta: float = 1.

    # Agent's video logging options
    env_library: str = None
    video_log_period: int = 10
    actions_log_period: int = 1
    logs_dir: str = 'experiments/default'
    num_episodes: int = 50

