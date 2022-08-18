import rlax
from acme.adders import reverb as adders_reverb

from drlearner.drlearner.config import DRLearnerConfig

DiscomazeDRLearnerConfig = DRLearnerConfig(
    gamma_min=0.99,
    gamma_max=0.99,
    num_mixtures=3,
    target_update_period=100,
    evaluation_epsilon=0.,
    actor_epsilon=0.05,
    target_epsilon=0.0,
    variable_update_period=1000,

    # Learner options
    retrace_lambda=0.97,
    burn_in_length=0,
    trace_length=30,
    sequence_period=30,
    num_sgd_steps_per_step=1,
    uvfa_learning_rate=1e-3,
    idm_learning_rate=1e-3,
    distillation_learning_rate=1e-3,
    idm_weight_decay=1e-5,
    distillation_weight_decay=1e-5,
    idm_clip_steps=5,
    distillation_clip_steps=5,
    clip_rewards=False,
    max_absolute_reward=1.0,
    tx_pair=rlax.SIGNED_HYPERBOLIC_PAIR,

    # Intrinsic reward multipliers
    beta_min=0.,
    beta_max=0.5,

    # Embedding network options
    observation_embed_dim=16,
    episodic_memory_num_neighbors=10,
    episodic_memory_max_size=5_000,
    episodic_memory_max_similarity=8.,
    episodic_memory_cluster_distance=8e-3,
    episodic_memory_pseudo_counts=1e-3,
    episodic_memory_epsilon=1e-2,

    # Distillation network
    distillation_embed_dim=32,
    max_lifelong_modulation=5.0,

    # Replay options
    samples_per_insert_tolerance_rate=1.0,
    samples_per_insert=0.0,
    min_replay_size=1,
    max_replay_size=100_000,
    batch_size=64,
    prefetch_size=1,
    num_parallel_calls=16,
    replay_table_name=adders_reverb.DEFAULT_PRIORITY_TABLE,

    # Priority options
    importance_sampling_exponent=0.6,
    priority_exponent=0.9,
    max_priority_weight=0.9,

    # Meta Controller options
    actor_window=160,
    evaluation_window=1000,
    n_arms=3,
    actor_mc_epsilon=0.3,
    evaluation_mc_epsilon=0.01,
    mc_beta=1.,

    # Agent video logging options
    env_library='discomaze',
    video_log_period=10,
    actions_log_period=1,
    logs_dir='experiments/emb_size_4_nn1_less_learning_steps',
    num_episodes=50,
)
