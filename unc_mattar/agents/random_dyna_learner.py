from unc_mattar.agents import base_dyna_learner

import numpy as np


class RandomDynaLearner(base_dyna_learner.DynaLearner):
    """A Dyna learner that samples experiences randomly for planning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plan(self, current_state):
        # Sample a random transition from the replay buffer
        transition_sample = self._replay_buffer[
            np.random.choice(range(len(self._replay_buffer)))
        ]
        transition_sample = transition_sample + (self._planning_lr,)
        self._step(*transition_sample)
