from key_door import key_door_env
from key_door import visualisation_env

from run_modes import base_runner

import os
import copy

from typing import List

from unc_mattar import constants, utils
from unc_mattar.agents import random_dyna_learner, evb_dyna_learner

import abc
import numpy as np
import pandas as pd


class Runner(base_runner.BaseRunner):

    def __init__(
        self,
        config,
        unique_id: str = "",
    ):

        base_runner.BaseRunner.__init__(self, config=config, unique_id=unique_id)

        self._learning_rate = config.learning_rate
        self._beta = config.beta
        self._gamma = config.gamma
        self._num_episodes = config.num_episodes
        self._train_episode_timeout = config.train_episode_timeout
        self._test_episode_timeout = config.test_episode_timeout

        self._transition_learning_rate = config.transition_learning_rate
        self._pre_episode_planning_steps = config.pre_episode_planning_steps
        self._post_episode_planning_steps = config.post_episode_planning_steps
        self._k_additional_planning_steps = config.k_additional_planning_steps

        self._initialisation_strategy = self._setup_initialisation(config)
        self._setup_logging(config)

        self._train_env, self._test_env = self._setup_envs(config)

        self._agent = self._setup_agent(config)

        self._populate_transition_matrix()

    @utils.timer
    def _setup_logging(self, config):
        self._checkpoint_frequency = config.checkpoint_frequency
        self._data_columns = self._setup_data_columns()
        self._log_columns = self._get_data_columns()

        self._heatmap_path = os.path.join(self._checkpoint_path, constants.HEATMAPS)
        os.makedirs(self._heatmap_path, exist_ok=True)
        self._video_path = os.path.join(self._checkpoint_path, constants.VIDEOS)
        os.makedirs(self._video_path, exist_ok=True)

    @utils.timer
    def _setup_initialisation(self, config):
        if config.initialisation_strategy == constants.RANDOM_NORMAL:
            initialisation_strategy = {
                "random_normal": {"mean": config.mean, "variance": config.variance},
            }
        elif config.initialisation_strategy == "zeros":
            initialisation_strategy = {"zeros"}
        else:
            raise ValueError(
                f"Initialisation strategy {config.initialisation_strategy} not recognised."
            )
        return initialisation_strategy

    @utils.timer
    def _setup_envs(self, config):
        _train_env = key_door_env.KeyDoorEnv(
            map_ascii_path=config.map_path,
            map_yaml_path=config.map_yaml_path,
            representation="agent_position",
            episode_timeout=self._train_episode_timeout,
        )
        _train_env = visualisation_env.VisualisationEnv(_train_env)

        _test_env = key_door_env.KeyDoorEnv(
            map_ascii_path=config.map_path,
            map_yaml_path=config.test_map_yaml_path,
            representation="agent_position",
            episode_timeout=self._test_episode_timeout,
        )
        _test_env = visualisation_env.VisualisationEnv(_test_env)

        return _train_env, _test_env

    @utils.timer
    def _setup_agent(self, config):
        if config.runner == constants.DYNA:
            agent = random_dyna_learner.RandomDynaLearner(
                action_space=self._train_env.action_space,
                state_space=self._train_env.state_space,
                learning_rate=self._learning_rate,
                transition_learning_rate=self._transition_learning_rate,
                gamma=self._gamma,
                beta=self._beta,
                initialisation_strategy=self._initialisation_strategy,
            )
        elif config.runner == constants.EVB:
            agent = evb_dyna_learner.EVBDynaLearner(
                action_space=self._train_env.action_space,
                state_space=self._train_env.state_space,
                learning_rate=self._learning_rate,
                transition_learning_rate=self._transition_learning_rate,
                gamma=self._gamma,
                beta=self._beta,
                initialisation_strategy=self._initialisation_strategy,
            )
        else:
            raise ValueError(
                f"Runner {config.runner} not recognised. " "Please use 'dyna' or 'evb'."
            )
        return agent

    @utils.timer
    def _populate_transition_matrix(self):
        dummy_env = copy.deepcopy(self._train_env)

        for state in dummy_env.state_space:
            for action in dummy_env.action_space:
                dummy_env.reset_environment(train=True)
                dummy_env._agent_position = state
                _, new_state = dummy_env.step(action)
                self._agent.increment_transition_matrix(state, new_state)
                self._agent.add_to_replay_buffer(
                    state, action, 0, new_state, self._train_env.active
                )

        self._agent.normalise_transition_matrix()
        # TODO: transition from reward to start state

    def _get_data_columns(self):
        """Output data columns to be logged by runner."""
        columns = [
            constants.TRAIN_EPISODE_RETURN,
            constants.TRAIN_EPISODE_LENGTH,
            constants.TEST_EPISODE_RETURN,
            constants.TEST_EPISODE_LENGTH,
        ]
        return columns

    def _setup_data_columns(self):
        data_columns = {}
        for key in self._get_data_columns():
            arr = np.empty(self._checkpoint_frequency)
            arr[:] = np.nan
            data_columns[key] = arr

        return data_columns

    def _checkpoint_data(self):
        log_dict = {k: self._data_columns[k] for k in self._log_columns}
        self._data_logger.logger_data = pd.DataFrame.from_dict(log_dict)
        self._data_logger.checkpoint()
        self._data_columns = self._setup_data_columns()
        self._data_index = 0

    def train(self):

        self._data_index = 0

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

            self._data_columns[constants.TRAIN_EPISODE_RETURN][
                self._data_index
            ] = train_episode_return
            self._data_columns[constants.TRAIN_EPISODE_LENGTH][
                self._data_index
            ] = train_episode_length
            self._data_columns[constants.TEST_EPISODE_RETURN][
                self._data_index
            ] = test_episode_return
            self._data_columns[constants.TEST_EPISODE_LENGTH][
                self._data_index
            ] = test_episode_length

            self._data_index += 1

            if i % self._checkpoint_frequency == 0:
                self._checkpoint_data()

    def _train_episode(self):
        episode_return = 0
        episode_length = 0

        state = self._train_env.reset_environment(train=True)

        for _ in range(self._pre_episode_planning_steps):
            self._agent.plan()

        while self._train_env.active:

            action = self._agent.select_action(state)
            reward, new_state = self._train_env.step(action)

            self._agent.step(
                state=state,
                action=action,
                reward=reward,
                new_state=new_state,
                active=self._train_env.active,
            )

            state = new_state
            episode_return += reward
            episode_length += 1

            for _ in range(self._k_additional_planning_steps):
                self._agent.plan()

        for _ in range(self._post_episode_planning_steps):
            self._agent.plan()

        return episode_return, episode_length

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
