import dm_env
import numpy as np

MAZE_PATH_COLOR = (0., 0., 0.)
AGENT_COLOR = (1., 1., 1.)


def mask_color_on_rgb(image, color) -> np.ndarray:
    """
    Given `image` of shape (H, W, C=3) and `color` pf shape (3,) return
    mask of shape (H, W) where pixel on the image have the same color as `color`
    """
    return np.isclose(image[..., 0], color[0]) & \
           np.isclose(image[..., 1], color[1]) & \
           np.isclose(image[..., 2], color[2])


class UniqueStatesVisitsCounter:
    def __init__(self, total):
        self.__total = total
        self.__visited = set()
        self.__reward_first_visit = []
        self.__reward_repeated_visit = []

    def add(self, state, intrinsic_reward):
        coords = self.get_xy_from_state(state)

        if coords in self.__visited:
            self.__reward_repeated_visit.append(float(intrinsic_reward))
        else:
            self.__visited.add(coords)
            self.__reward_first_visit.append(float(intrinsic_reward))

    @staticmethod
    def get_xy_from_state(state):
        mask = mask_color_on_rgb(state, AGENT_COLOR)
        coords = np.where(mask)
        x, y = coords[1][0], coords[0][0]
        return x, y

    def get_number_of_visited(self):
        return len(self.__visited)

    def get_fraction_of_visited(self):
        return len(self.__visited) / self.__total

    def get_mean_first_visit_reward(self):
        return np.mean(self.__reward_first_visit)

    def get_mean_repeated_visit_reward(self):
        return np.mean(self.__reward_repeated_visit)


class UniqueStatesDiscoMazeObserver:
    def __init__(self):
        self._state_visit_counter: UniqueStatesVisitsCounter = None

    def reset(self, states_total):
        self._state_visit_counter = UniqueStatesVisitsCounter(states_total)

    def observe_first(self, *args, **kwargs) -> None:
        env, timestamp, actor_extras = args

        states_total = mask_color_on_rgb(
            timestamp.observation.observation,
            color=MAZE_PATH_COLOR
        ).sum() + 1  # +1 for current position of agent
        self.reset(states_total)

    def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
                action: np.ndarray, actor_extras, **kwargs) -> None:
        self._state_visit_counter.add(
            timestep.observation.observation,
            actor_extras['intrinsic_reward']
        )

    def get_metrics(self, **kwargs):
        metrics = {
            "unique_fraction": self._state_visit_counter.get_fraction_of_visited(),
            "first_visit_mean_reward": self._state_visit_counter.get_mean_first_visit_reward(),
            "repeated_visit_mean_reward": self._state_visit_counter.get_mean_repeated_visit_reward()
        }
        return metrics
