from unc_mattar.agents import base_dyna_learner

import numpy as np


class EVBDynaLearner(base_dyna_learner.DynaLearner):
    """EVB as criterion for planning sampling. Replicates Mattar & Daw 2018."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plan(self, current_state):

        # Convert replay buffer to numpy arrays for vectorized operations
        replay_arr = np.array(self._replay_buffer, dtype=object)

        # SR matrix for need term
        sr_matrix = self._get_successor_matrix()
        current_state_id = self._state_id_mapping[current_state]
        sr_row = sr_matrix[current_state_id]

        state_ids = replay_arr[:, 0].astype(int)
        needs = sr_row[state_ids]

        actions = replay_arr[:, 1].astype(int)
        rewards = replay_arr[:, 2].astype(float)
        new_state_ids = replay_arr[:, 3].astype(int)
        actives = replay_arr[:, 4].astype(bool)

        # Hypothetical Q-learning update for all transitions in buffer
        discounts = np.where(actives, self._gamma, 0.0)
        q_current = self._state_action_values[state_ids, actions]
        q_next_max = np.max(self._state_action_values[new_state_ids], axis=1)
        q_target = rewards + discounts * q_next_max
        q_updated = q_current + self._learning_rate * (q_target - q_current)

        # Compute gains from hypothetical updates
        old_softmax_denominator = np.sum(
            np.exp(self._beta * self._state_action_values[state_ids]), axis=1
        )
        new_softmax_denominator = (
            old_softmax_denominator
            - np.exp(self._beta * q_current)
            + np.exp(self._beta * q_updated)
        )

        old_q = self._state_action_values[state_ids]
        new_q = old_q.copy()
        new_q[np.arange(len(q_updated)), actions] = q_updated

        old_policy = np.exp(self._beta * old_q) / old_softmax_denominator[:, None]
        new_policy = np.exp(self._beta * new_q) / new_softmax_denominator[:, None]

        v_new = np.sum(new_policy * new_q, axis=1)
        v_old = np.sum(old_policy * old_q, axis=1)
        gains = v_new - v_old

        evbs = gains * needs

        idx = np.argmax(evbs)
        transition_sample = tuple(replay_arr[idx]) + (self._planning_lr,)
        self._step(*transition_sample)
