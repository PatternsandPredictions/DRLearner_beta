"""Defines local DRLearner agent, using JAX."""

from typing import Optional

from acme import specs
from acme.utils import counting

from ..core import local_layout
from .builder import DRLearnerBuilder
from .config import DRLearnerConfig
from .networks import make_policy_networks, DRLearnerNetworks


class DRLearner(local_layout.LocalLayout):
    """Local agent for DRLearner.

    This implements a single-process DRLearner agent.
    """

    def __init__(
            self,
            spec: specs.EnvironmentSpec,
            networks: DRLearnerNetworks,
            config: DRLearnerConfig,
            seed: int,
            workdir: Optional[str] = '~/acme',
            counter: Optional[counting.Counter] = None,
    ):
        ngu_builder = DRLearnerBuilder(networks, config, num_actors_per_mixture=1)
        super().__init__(
            seed=seed,
            environment_spec=spec,
            builder=ngu_builder,
            networks=networks,
            policy_network=make_policy_networks(networks, config),
            workdir=workdir,
            min_replay_size=config.min_replay_size,
            samples_per_insert=config.samples_per_insert if config.samples_per_insert \
                else 10 / (config.burn_in_length + config.trace_length),
            batch_size=config.batch_size,
            num_sgd_steps_per_step=config.num_sgd_steps_per_step,
            counter=counter,
        )

    def get_extras(self):
        return self._actor.get_extras()
