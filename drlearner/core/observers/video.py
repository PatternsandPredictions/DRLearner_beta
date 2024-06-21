import os
from abc import ABC, abstractmethod
import platform

from skvideo import io
import numpy as np
import tensorflow as tf
import dm_env

from drlearner.core.loggers import ImageLogger
from drlearner.core.observers.lazy_dict import LazyDictionary


class VideoObserver(ABC):
    def __init__(self, config):
        self.config = config
        self.env_library = config.env_library
        self.log_period = config.video_log_period

        self.platform = platform.system().lower()

    def render(self, env):
        """
        Renders current frame
        """
        render_funcs = LazyDictionary(
            {
                'dm_control': lambda: env.physics.render(camera_id=0),
                'gym': lambda: env.environment.render(mode='rgb_array'),
                'discomaze': lambda: env.render(mode='state_pixels'),
            },
        )
        env_lib = self.env_library

        if env_lib in render_funcs.keys():
            return render_funcs[env_lib]
        else:
            raise ValueError(
                f"Unknown environment library: {env_lib}; choose among {list(render_funcs.keys())}",
            )

    def _log_video(self, episode_count):
        return True if (episode_count + 1) % self.log_period == 0 else False

    @abstractmethod
    def observe(self, env: dm_env.Environment, *args, **kwargs):
        pass


class StorageVideoObserver(VideoObserver):
    def __init__(self, config):
        print(f'INIT: {self.__class__.__name__}')
        super().__init__(config)

        self.frames = list()
        self.videos_dir = self._create_videos_dir()

    def observe(self, env: dm_env.Environment, *args, **kwargs):
        frame = self.render(env)
        self.frames.append(frame.astype('uint8'))

    def get_metrics(self, **kwargs):
        episode = kwargs['episode']

        if self._log_video(episode):
            video_dir = os.path.join(self.videos_dir, f'episode_{episode + 1}.mp4')
            io.vwrite(video_dir, np.array(self.frames))

        self.frames = list()

        return dict()

    def _create_videos_dir(self):
        video_dir = os.path.join(self.config.logs_dir, 'episodes')

        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)

        return video_dir

    observe_first = observe


class TBVideoObserver(VideoObserver):
    def __init__(self, config):
        super().__init__(config)
        self.image_logger = ImageLogger(config.logs_dir)

    def log_frame(self, env: dm_env.Environment, episode=None, step=None):
        frame = self.render(env)

        self.image_logger.write_image(
            f'video_{episode + 1}',
            tf.convert_to_tensor(np.array(frame)),
            step=step,
        )

    def observe(self, env: dm_env.Environment, *args, **kwargs) -> None:
        episode = kwargs['episode']
        step = kwargs['step']

        if self._log_video(episode):
            self.log_frame(env, episode=episode, step=step)

    observe_first = observe