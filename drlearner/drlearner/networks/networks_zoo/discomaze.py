import haiku as hk
import jax.nn
from acme import specs
from acme.jax import networks as networks_lib

from ..distillation_network import make_distillation_net
from ..embedding_network import make_embedding_net
from ..networks import DRLearnerNetworks
from ..uvfa_network import make_uvfa_net
from ..uvfa_torso import UVFATorso
from ...config import DRLearnerConfig


def make_discomaze_nets(config: DRLearnerConfig, env_spec: specs.EnvironmentSpec) -> DRLearnerNetworks:
    uvfa_net = make_discomaze_uvfa_net(env_spec, num_mixtures=config.num_mixtures, batch_size=config.batch_size)
    embedding_network = make_discomaze_embedding_net(env_spec, config.observation_embed_dim)
    distillation_network = make_discomaze_distillation_net(env_spec, config.distillation_embed_dim)
    return DRLearnerNetworks(uvfa_net, embedding_network, distillation_network)


def make_discomaze_embedding_net(env_spec, embedding_dim):
    def make_discomaze_embedding_modules():
        embedding_torso = hk.Sequential([
            hk.Conv2D(16, kernel_shape=3, stride=1),
            jax.nn.relu,
            hk.Conv2D(32, kernel_shape=3, stride=1),
            jax.nn.relu,
            hk.Flatten(preserve_dims=-3),
            hk.Linear(embedding_dim),
            jax.nn.relu
        ])
        pred_head = hk.Sequential([
            hk.Linear(32),
            jax.nn.relu,
            hk.Linear(env_spec.actions.num_values)
        ], name='action_pred_head')
        return embedding_torso, pred_head

    return make_embedding_net(
        make_embedding_modules=make_discomaze_embedding_modules,
        env_spec=env_spec
    )


def make_discomaze_distillation_net(env_spec, embedding_dim):
    def make_discomaze_distillation_modules():
        embedding_torso = hk.Sequential([
            hk.Conv2D(16, kernel_shape=3, stride=1),
            jax.nn.relu,
            hk.Conv2D(32, kernel_shape=3, stride=1),
            jax.nn.relu,
            hk.Flatten(preserve_dims=-3),
            hk.Linear(embedding_dim),
            jax.nn.relu
        ])
        return embedding_torso

    return make_distillation_net(
        make_distillation_modules=make_discomaze_distillation_modules,
        env_spec=env_spec
    )


def make_discomaze_uvfa_net(env_spec, num_mixtures: int, batch_size: int):
    def make_discomaze_uvfa_modules():
        embedding_torso = make_uvfa_discomaze_torso(env_spec.actions.num_values, num_mixtures)
        recurrent_core = hk.LSTM(256)
        head = networks_lib.DuellingMLP(
            num_actions=env_spec.actions.num_values,
            hidden_sizes=[256]
        )
        return embedding_torso, recurrent_core, head

    return make_uvfa_net(
        make_uvfa_modules=make_discomaze_uvfa_modules,
        batch_size=batch_size,
        env_spec=env_spec
    )


def make_uvfa_discomaze_torso(num_actions: int, num_mixtures: int):
    observation_embedding_torso = hk.Sequential([
        hk.Conv2D(16, kernel_shape=3, stride=1),
        jax.nn.relu,
        hk.Conv2D(32, kernel_shape=3, stride=1),
        jax.nn.relu,
        hk.Flatten(preserve_dims=-3),
        hk.Linear(256),
        jax.nn.relu
    ])
    return UVFATorso(
        observation_embedding_torso,
        num_actions, num_mixtures,
        name='discomaze_uvfa_torso'
    )
