from unc_mattar.agents import base_dyna_learner

import numpy as np


class EVBDynaLearner(base_dyna_learner.DynaLearner):
    """EVB as criterion for planning sampling. Replicates Mattar & Daw 2018."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plan(self, current_state):
        # Implement the planning step using the EVB criterion approach
        sr_matrix = self._get_successor_matrix()
        current_state_id = self._state_id_mapping[current_state]
        sr_row = sr_matrix[current_state_id]
        needs = [sr_row[exp[0]] for exp in self._replay_buffer]

        gains = []

        for exp in self._replay_buffer:
            state_id, action, reward, new_state_id, active = exp

            if active:
                discount = self._gamma
            else:
                discount = 0

            q_current = self._state_action_values[state_id][action]
            q_target = reward + discount * np.max(
                self._state_action_values[new_state_id]
            )
            q_updated = q_current + self._learning_rate * (q_target - q_current)
            # unclear whether to use planning or regular lr here
            # I assume its not even distinct in the original case.

            gain = abs(q_updated - q_current)
            gains.append(gain)
        evbs = [g * n for g, n in zip(gains, needs)]
        transition_sample = self._replay_buffer[np.argmax(evbs)]
        transition_sample = transition_sample + (self._planning_lr,)
        self._step(*transition_sample)
