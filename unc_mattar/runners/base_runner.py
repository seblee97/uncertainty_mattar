from key_door import key_door_env
from key_door import visualisation_env

import os

from unc_mattar import constants
from unc_mattar.agents import q_learner, dyna_learner

import abc
import numpy as np


class BaseRunner(abc.ABC):

    def __init__(
        self,
        config,
        unique_id: str = "",
    ):

        self._learning_rate = config.learning_rate
        self._beta = config.beta
        self._gamma = config.gamma
        self._num_episodes = config.num_episodes
        self._train_episode_timeout = config.train_episode_timeout
        self._test_episode_timeout = config.test_episode_timeout

        if config.initialisation_strategy == constants.RANDOM_NORMAL:
            self._initialisation_strategy = {
                "random_normal": {"mean": 0, "variance": 1},
            }
        elif config.initialisation_strategy == "zeros":
            self._initialisation_strategy = {"zeros"}
        else:
            raise ValueError(
                f"Initialisation strategy {config.initialisation_strategy} not recognised."
            )

        self._checkpoint_path = config.checkpoint_path
        self._heatmap_path = os.path.join(self._checkpoint_path, constants.HEATMAPS)
        os.makedirs(self._heatmap_path, exist_ok=True)
        self._video_path = os.path.join(self._checkpoint_path, constants.VIDEOS)
        os.makedirs(self._video_path, exist_ok=True)

        self._pre_episode_planning_steps = config.pre_episode_planning_steps
        self._post_episode_planning_steps = config.post_episode_planning_steps

        _train_env = key_door_env.KeyDoorEnv(
            map_ascii_path=config.map_path,
            map_yaml_path=config.map_yaml_path,
            representation="agent_position",
            episode_timeout=self._train_episode_timeout,
        )
        self._train_env = visualisation_env.VisualisationEnv(_train_env)

        _test_env = key_door_env.KeyDoorEnv(
            map_ascii_path=config.map_path,
            map_yaml_path=config.test_map_yaml_path,
            representation="agent_position",
            episode_timeout=self._test_episode_timeout,
        )
        self._test_env = visualisation_env.VisualisationEnv(_test_env)

    def train(self):
        train_episode_returns = []
        train_episode_lengths = []

        test_episode_returns = []
        test_episode_lengths = []

        for i in range(self._num_episodes):

            if i % 25 == 0:
                print(f"Episode {i}")
                if i != 0:
                    self._train_env.visualise_episode_history(
                        os.path.join(self._video_path, f"train_{i}.mp4"),
                        history="train",
                    )
                    self._test_env.visualise_episode_history(
                        os.path.join(self._video_path, f"test_{i}.mp4"), history="test"
                    )
                    averaged_heatmap = (
                        self._test_env.average_values_over_positional_states(
                            values={
                                k: np.max(v)
                                for k, v in self._agent.state_action_values.items()
                            },
                        )
                    )
                    self._test_env.plot_heatmap_over_env(
                        averaged_heatmap,
                        save_name=os.path.join(self._heatmap_path, f"heatmap_{i}.png"),
                    )

            train_episode_return, train_episode_length = self._train_episode()
            test_episode_return, test_episode_length = self._test_episode()

            train_episode_returns.append(train_episode_return)
            train_episode_lengths.append(train_episode_length)

            test_episode_returns.append(test_episode_return)
            test_episode_lengths.append(test_episode_length)

        return (
            train_episode_returns,
            train_episode_lengths,
            test_episode_returns,
            test_episode_lengths,
        )

    @abc.abstractmethod
    def _train_episode(self):
        pass

    def _test_episode(self):

        episode_return = 0
        episode_length = 0

        state = self._test_env.reset_environment(train=False)

        while self._test_env.active:
            action = self._agent.select_greedy_action(state)
            reward, new_state = self._test_env.step(action)
            state = new_state
            episode_return += reward
            episode_length += 1

        return episode_return, episode_length
