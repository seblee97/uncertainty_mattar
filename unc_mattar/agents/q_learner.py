from unc_mattar.agents import base_agent

import numpy as np

from typing import Dict, List, Tuple


class QLearner(base_agent.BaseAgent):

    def __init__(
        self,
        action_space,
        state_space,
        learning_rate,
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

        if active:
            discount = self._gamma
        else:
            discount = 0

        initial_value = self._state_action_values[state_id][action]
        new_sate_values = self._state_action_values[new_state_id]

        updated_value = initial_value + self._learning_rate * (
            reward + discount * np.max(new_sate_values) - initial_value
        )
        self._state_action_values[state_id][action] = updated_value
