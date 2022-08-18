from collections import Counter
from io import BytesIO
from typing import Dict

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from PIL import Image
import dm_env

from drlearner.core.loggers import ImageLogger


class ActionsObserver:
    def __init__(self, config):
        self.config = config
        self.image_logger = ImageLogger(config.logs_dir)

        self.unique_actions = set()
        self.ratios = list()
        self.episode_actions = list()

    def _log_episode_action(self, timestamp: dm_env.TimeStep, action: np.array):
        if not timestamp.last():
            self.episode_actions.append(action)
        else:
            episode_ratios = self.calculate_actions_ratio()
            self.ratios.append(episode_ratios)

            self.episode_actions = list()

    def observe(
            self,
            *args,
            **kwargs,
    ) -> None:
        env, timestamp, action, actor_extras = args

        action = np.asscalar(action)
        self.unique_actions.add(action)

        episode_count = kwargs['episode_count']
        log_action = episode_count % self.config.actions_log_period == 0
        if log_action:
            self._log_episode_action(timestamp, action)

    def calculate_actions_ratio(self) -> Dict:
        """
        Calculates actions ratio per episode

        Returns
            ratios: list of action ratios per action type
            unique_actions: set of possible actions
        """
        counter = Counter(self.episode_actions)
        episode_ratios = dict()

        for action in self.unique_actions:
            count = counter.get(action, 0)
            ratio = count / len(self.episode_actions)
            episode_ratios[str(action)] = ratio

        return episode_ratios

    def _plot_actions(self):
        """
        Creates actions plot and logs it into tensorboard
        """
        for action in self.unique_actions:
            values = [ratio.get(str(action), 0) for ratio in self.ratios]

            n = len(self.ratios)
            steps = list(range(n))

            plt.plot(steps, values, c=np.random.rand(3), label=action)
            plt.legend()

        plt.title('action ratios per episode')
        plt.xlabel('episode')
        plt.ylabel('action ratio')

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        img = Image.open(buffer)
        self.image_logger.write_image('actions_ratio', tf.convert_to_tensor(np.array(img)))

    def get_metrics(self, **kwargs):
        episode_count = kwargs['episode_count']
        last_episode = episode_count == self.config.num_episodes - 1

        if last_episode:
            self._plot_actions()

        return dict()
