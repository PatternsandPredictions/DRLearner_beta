import inspect
import logging
import time
import os

import jax

from acme import specs
from acme import core
from acme.utils import counting
from acme.utils import observers as observers_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import loggers
from acme.utils.loggers.tf_summary import TFSummaryLogger
from acme.utils.loggers import base


from ..core import distributed_layout
from ..core.environment_loop import EnvironmentLoop

from typing import Optional, Callable, Any, Mapping, Sequence, TextIO, Union


def from_dict_to_dataclass(cls, data):
    return cls(
        **{
            key: (data[key] if val.default == val.empty else data.get(key, val.default))
            for key, val in inspect.signature(cls).parameters.items()
        }
    )


class CloudCSVLogger:
    def __init__(
            self,
            directory_or_file: Union[str, TextIO] = '~/acme',
            label: str = '',
            time_delta: float = 0.,
            add_uid: bool = True,
            flush_every: int = 30,
    ):
        """Instantiates the logger.

        Args:
          directory_or_file: Either a directory path as a string, or a file TextIO
            object.
          label: Extra label to add to logger. This is added as a suffix to the
            directory.
          time_delta: Interval in seconds between which writes are dropped to
            throttle throughput.
          add_uid: Whether to add a UID to the file path. See `paths.process_path`
            for details.
          flush_every: Interval (in writes) between flushes.
        """

        if flush_every <= 0:
            raise ValueError(
                f'`flush_every` must be a positive integer (got {flush_every}).')

        self._last_log_time = time.time() - time_delta
        self._time_delta = time_delta
        self._flush_every = flush_every
        self._add_uid = add_uid
        self._writes = 0
        self.file_path = os.path.join(directory_or_file, f'{label}_logs.csv')
        self._keys = []
        logging.info('Logging to %s', self.file_path)

    def write(self, data: base.LoggingData):
        """Writes a `data` into a row of comma-separated values."""
        # Only log if `time_delta` seconds have passed since last logging event.
        now = time.time()

        elapsed = now - self._last_log_time
        if elapsed < self._time_delta:
            logging.debug('Not due to log for another %.2f seconds, dropping data.',
                          self._time_delta - elapsed)
            return
        self._last_log_time = now

        # Append row to CSV.
        data = base.to_numpy(data)
        if self._writes == 0:
            self._keys = data.keys()
            with open(self.file_path, 'w') as f:
                f.write(','.join(self._keys))
                f.write('\n')
                f.write(','.join(list(map(str, [data[k] for k in self._keys]))))
                f.write('\n')
        else:
            with open(self.file_path, 'a') as f:
                f.write(','.join(list(map(str, [data[k] for k in self._keys]))))
                f.write('\n')
        self._writes += 1


def make_tf_logger(
        workdir: str = '~/acme/',
        label: str = 'learner',
        save_data: bool = True,
        time_delta: float = 0.,
        asynchronous: bool = False,
        print_fn: Optional[Callable[[str], None]] = print,
        serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = loggers.base.to_numpy,
        steps_key: str = 'steps',

) -> loggers.base.Logger:
    del steps_key
    if not print_fn:
        print_fn = logging.info

    terminal_logger = loggers.terminal.TerminalLogger(label=label, print_fn=print_fn)

    all_loggers = [terminal_logger]

    if save_data:
        if '/gcs/' in workdir:
            all_loggers.append(CloudCSVLogger(directory_or_file=workdir, label=label, time_delta=time_delta))
        else:
            all_loggers.append(loggers.csv.CSVLogger(directory_or_file=workdir, label=label, time_delta=time_delta))

    tb_workdir = workdir
    if '/gcs/' in tb_workdir:
        tb_workdir = tb_workdir.replace('/gcs/', 'gs://')
    all_loggers.append(TFSummaryLogger(logdir=tb_workdir, label=label))

    logger = loggers.aggregators.Dispatcher(all_loggers, serialize_fn)
    logger = loggers.filters.NoneFilter(logger)

    logger = loggers.filters.TimeFilter(logger, time_delta)
    return logger


def evaluator_factory_logger_choice(environment_factory: distributed_layout.EnvironmentFactory,
                                    network_factory: distributed_layout.NetworkFactory,
                                    policy_factory: distributed_layout.PolicyFactory,
                                    logger_fn: Callable,
                                    observers: Sequence[observers_lib.EnvLoopObserver] = (),
                                    actor_id: int = 0,
                      ) -> distributed_layout.EvaluatorFactory:

    """Returns an evaluator process with customizable log function."""

    def evaluator(
      random_key: networks_lib.PRNGKey,
      variable_source: core.VariableSource,
      counter: counting.Counter,
      make_actor: distributed_layout.MakeActorFn
    ):
        """The evaluation process."""

        # Create environment and evaluator networks
        environment_key, actor_key = jax.random.split(random_key)
        environment = environment_factory(utils.sample_uint32(environment_key))
        networks = network_factory(specs.make_environment_spec(environment))

        actor = make_actor(random_key, policy_factory(networks), variable_source=variable_source)

        # Create logger and counter.
        counter = counting.Counter(counter, 'evaluator')

        logger = logger_fn()

        # Create the run loop and return it.
        return EnvironmentLoop(environment, actor, counter, logger, observers=observers)
    return evaluator
