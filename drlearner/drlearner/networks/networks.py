import dataclasses

from .distillation_network import DistillationNetwork
from .embedding_network import EmbeddingNetwork
from .uvfa_network import UVFANetwork


@dataclasses.dataclass
class DRLearnerNetworks:
    """Wrapper for all DRLearner learnable networks"""
    uvfa_net: UVFANetwork
    embedding_net: EmbeddingNetwork
    distillation_net: DistillationNetwork
