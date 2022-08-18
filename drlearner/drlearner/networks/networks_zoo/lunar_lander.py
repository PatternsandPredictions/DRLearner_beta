import haiku as hk
from acme import specs
from acme.jax import networks as networks_lib

from ..distillation_network import make_distillation_net
from ..embedding_network import make_embedding_net
from ..networks import DRLearnerNetworks
from ..uvfa_network import make_uvfa_net
from ..uvfa_torso import UVFATorso
from ...config import DRLearnerConfig


def make_lunar_lander_nets(config: DRLearnerConfig, env_spec: specs.EnvironmentSpec) -> DRLearnerNetworks:
    uvfa_net = make_lunar_lander_uvfa_net(env_spec, num_mixtures=config.num_mixtures, batch_size=config.batch_size)
    embedding_network = make_lunar_lander_embedding_net(env_spec, config.observation_embed_dim)
    distillation_network = make_lunar_lander_distillation_net(env_spec, config.distillation_embed_dim)
    return DRLearnerNetworks(uvfa_net, embedding_network, distillation_network)


def make_lunar_lander_embedding_net(env_spec, embedding_dim):
    def make_mlp_embedding_modules():
        embedding_torso = hk.nets.MLP([16, 32, embedding_dim], name='mlp_embedding_torso')
        pred_head = hk.Linear(env_spec.actions.num_values, name='action_pred_head')
        return embedding_torso, pred_head

    return make_embedding_net(
        make_embedding_modules=make_mlp_embedding_modules,
        env_spec=env_spec
    )


def make_lunar_lander_distillation_net(env_spec, embedding_dim):
    def make_mlp_distillation_modules():
        embedding_torso = hk.nets.MLP([16, 32, embedding_dim], name='mlp_embedding_torso')
        return embedding_torso

    return make_distillation_net(
        make_distillation_modules=make_mlp_distillation_modules,
        env_spec=env_spec
    )


def make_lunar_lander_uvfa_net(env_spec, num_mixtures: int, batch_size: int):
    def make_mlp_uvfa_modules():
        embedding_torso = make_uvfa_lunar_lander_torso(env_spec.actions.num_values, num_mixtures)
        recurrent_core = hk.LSTM(32)
        head = networks_lib.DuellingMLP(
            num_actions=env_spec.actions.num_values,
            hidden_sizes=[32]
        )
        return embedding_torso, recurrent_core, head

    return make_uvfa_net(
        make_uvfa_modules=make_mlp_uvfa_modules,
        batch_size=batch_size,
        env_spec=env_spec
    )


def make_uvfa_lunar_lander_torso(num_actions: int, num_mixtures: int):
    observation_embedding_torso = hk.nets.MLP([16, 32, 16])
    return UVFATorso(
        observation_embedding_torso,
        num_actions, num_mixtures,
        name='mlp_uvfa_torso'
    )
