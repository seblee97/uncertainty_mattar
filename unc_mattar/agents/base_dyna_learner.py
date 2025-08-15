from unc_mattar.agents import base_agent

import numpy as np
import abc

from typing import Dict, List, Tuple


class DynaLearner(base_agent.BaseAgent, abc.ABC):

    def __init__(
        self,
        action_space,
        state_space,
        # positional_state_space,
        learning_rate,
        transition_learning_rate,
        gamma,
        beta,
        initialisation_strategy,
    ):
        super().__init__(
            action_space,
            state_space,
            learning_rate,
            gamma,
            beta,
            initialisation_strategy,
        )

        self._transition_lr = transition_learning_rate
        # self._positional_state_space = positional_state_space

        self._transition_matrix = np.zeros(
            (len(self._state_space), len(self._state_space))
        )

        self._replay_buffer = []

    def add_to_replay_buffer(self, state, action, reward, new_state, active):
        state_id = self._state_id_mapping[state]
        new_state_id = self._state_id_mapping[new_state]

        self._replay_buffer.append((state_id, action, reward, new_state_id, active))

    def increment_transition_matrix(self, state, new_state):
        state_id = self._state_id_mapping[state]
        new_state_id = self._state_id_mapping[new_state]

        self._transition_matrix[state_id][new_state_id] += 1

    def normalise_transition_matrix(self):
        self._transition_matrix /= np.sum(self._transition_matrix, axis=1)[:, None]
        self._transition_matrix[np.isnan(self._transition_matrix)] = 0

    def _step_transition_matrix(self, state_id, new_state_id):
        target_vector = np.zeros((len(self._state_space)))
        target_vector[new_state_id] = 1

        initial_transition_vector = self._transition_matrix[state_id]
        updated_transition_vector = initial_transition_vector + self._transition_lr * (
            target_vector - initial_transition_vector
        )
        self._transition_matrix[state_id] = updated_transition_vector

    def _get_successor_matrix(self):
        return np.linalg.inv(
            np.eye(len(self._state_space)) - self._gamma * self._transition_matrix
        )

    @abc.abstractmethod
    def plan(self):
        pass

    def step(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        new_state: Tuple[int, int],
        active: bool,
    ):

        state_id = self._state_id_mapping[state]
        new_state_id = self._state_id_mapping[new_state]

        self._step_transition_matrix(state_id, new_state_id)
        self._replay_buffer.append((state_id, action, reward, new_state_id, active))

        self._step(state_id, action, reward, new_state_id, active, self._learning_rate)

    def _step(self, state_id, action, reward, new_state_id, active, learning_rate):
        if active:
            discount = self._gamma
        else:
            discount = 0

        initial_value = self._state_action_values[state_id][action]
        new_sate_values = self._state_action_values[new_state_id]

        updated_value = initial_value + learning_rate * (
            reward + discount * np.max(new_sate_values) - initial_value
        )
        self._state_action_values[state_id][action] = updated_value
