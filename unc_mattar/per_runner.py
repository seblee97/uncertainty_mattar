from key_door import key_door_env
from key_door import visualisation_env

from unc_mattar.agents import q_learner

import numpy as np


class Runner:

    def __init__(
        self,
        learning_rate,
        beta,
        gamma,
        num_episodes,
        train_episode_timeout,
        test_episode_timeout,
        map_path,
        map_yaml_path,
        test_map_yaml_path,
        initialisation_strategy,
    ):

        self._learning_rate = learning_rate
        self._beta = beta
        self._gamma = gamma
        self._num_episodes = num_episodes
        self._train_episode_timeout = train_episode_timeout
        self._test_episode_timeout = test_episode_timeout

        _train_env = key_door_env.KeyDoorEnv(
            map_ascii_path=map_path,
            map_yaml_path=map_yaml_path,
            representation="agent_position",
            episode_timeout=self._train_episode_timeout,
        )
        self._train_env = visualisation_env.VisualisationEnv(_train_env)

        _test_env = key_door_env.KeyDoorEnv(
            map_ascii_path=map_path,
            map_yaml_path=test_map_yaml_path,
            representation="agent_position",
            episode_timeout=self._test_episode_timeout,
        )
        self._test_env = visualisation_env.VisualisationEnv(_test_env)

        self._agent = q_learner.QLearner(
            action_space=self._train_env.action_space,
            state_space=self._train_env.state_space,
            learning_rate=self._learning_rate,
            gamma=self._gamma,
            beta=self._beta,
            initialisation_strategy=initialisation_strategy,
        )

    def train(self):
        train_episode_returns = []
        train_episode_lengths = []

        test_episode_returns = []
        test_episode_lengths = []

        for i in range(self._num_episodes):

            if i % 500 == 0:
                print(f"Episode {i}")
                if i != 0:
                    self._train_env.visualise_episode_history(f"train_{i}.mp4")
                    self._test_env.visualise_episode_history(
                        f"test_{i}.mp4", history="test"
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
                        save_name=f"heatmap_{i}.png",
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

    def _train_episode(self):
        episode_return = 0
        episode_length = 0

        state = self._train_env.reset_environment(train=True)

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
