from unc_mattar.agents import base_dyna_learner

import numpy as np


class EVBDynaLearner(base_dyna_learner.DynaLearner):
    """EVB as criterion for planning sampling. Replicates Mattar & Daw 2018."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plan(self, current_state):

        # Convert replay buffer to numpy arrays for vectorized operations
        replay_arr = np.array(self._replay_buffer, dtype=object)

        # Implement the planning step using the EVB criterion approach
        sr_matrix = self._get_successor_matrix()
        current_state_id = self._state_id_mapping[current_state]
        sr_row = sr_matrix[current_state_id]

        state_ids = replay_arr[:, 0].astype(int)
        needs = sr_row[state_ids]

        actions = replay_arr[:, 1].astype(int)
        rewards = replay_arr[:, 2].astype(float)
        new_state_ids = replay_arr[:, 3].astype(int)
        actives = replay_arr[:, 4].astype(bool)

        discounts = np.where(actives, self._gamma, 0.0)
        q_current = self._state_action_values[state_ids, actions]
        q_next_max = np.max(self._state_action_values[new_state_ids], axis=1)
        q_target = rewards + discounts * q_next_max
        q_updated = q_current + self._learning_rate * (q_target - q_current)
        gains = np.abs(q_updated - q_current)

        evbs = gains * needs

        idx = np.argmax(evbs)
        transition_sample = tuple(replay_arr[idx]) + (self._planning_lr,)
        self._step(*transition_sample)
