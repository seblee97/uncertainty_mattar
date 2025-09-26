from unc_mattar.agents import base_dyna_learner

import numpy as np


class RandomDynaLearner(base_dyna_learner.DynaLearner):
    """A Dyna learner that samples experiences randomly for planning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plan(self, current_state):

        # Sample a random transition from the replay buffer
        transition_sample = self._replay_buffer.get_random()
        transition_sample = transition_sample + (self._planning_lr,)

        self._state_planning_counts[self._id_state_mapping[transition_sample[0]]] += 1
        self._per_state_state_planning_counts[current_state][
            self._id_state_mapping[transition_sample[0]]
        ] += 1
        self._episode_state_planning_counts[
            self._id_state_mapping[transition_sample[0]]
        ] += 1
        self._episode_per_state_state_planning_counts[current_state][
            self._id_state_mapping[transition_sample[0]]
        ] += 1

        self._step(*transition_sample)
