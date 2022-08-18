import tensorflow as tf
from acme.utils.loggers.tf_summary import TFSummaryLogger


def disable_view_window():
    """
    Disables gym view window
    """
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=False)

    rendering.Viewer.__init__ = constructor


class ImageLogger(TFSummaryLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def write_image(self, name, image, step=0):
        with self.summary.as_default():
            tf.summary.image(name, [image], step=step)