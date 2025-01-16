import abc

from typing import Dict, List, Tuple

import numpy as np

from unc_mattar import constants


class BaseAgent(abc.ABC):

    def __init__(
        self,
        action_space,
        state_space,
        learning_rate,
        gamma,
        beta,
        initialisation_strategy,
    ):

        self._action_space = action_space
        self._state_space = state_space

        self._state_id_mapping = {state: i for i, state in enumerate(self._state_space)}
        self._id_state_mapping = {i: state for i, state in enumerate(self._state_space)}

        self._state_action_values = self._initialise_values(
            initialisation_strategy=initialisation_strategy
        )

        self._state_visitation_counts = {s: 0 for s in self._state_space}

        self._learning_rate = learning_rate
        self._gamma = gamma
        self._beta = beta

    @property
    def action_space(self) -> List[int]:
        return self._action_space

    @property
    def state_id_mapping(self) -> Dict:
        return self._state_id_mapping

    @property
    def id_state_mapping(self) -> Dict:
        return self._id_state_mapping

    @property
    def state_visitation_counts(self) -> Dict[Tuple[int, int], int]:
        return self._state_visitation_counts

    @property
    def state_action_values(self) -> Dict[Tuple[int, int], np.ndarray]:
        values = {
            self._id_state_mapping[i]: action_values
            for i, action_values in enumerate(self._state_action_values)
        }
        return values

    def _initialise_values(
        self, initialisation_strategy: Dict[str, Dict]
    ) -> np.ndarray:
        """Initialise values for each state, action pair in state-action space.

        Args:
            initialisation_strategy: name of method used to initialise.

        Returns:
            initial_values: matrix containing state-action id / value mapping.
        """
        initialisation_strategy_name = list(initialisation_strategy.keys())[0]
        initialisation_parameters = initialisation_strategy[
            initialisation_strategy_name
        ]
        if isinstance(initialisation_strategy_name, (int, float)):
            return initialisation_strategy_name * np.ones(
                (len(self._state_space), len(self._action_space))
            )
        elif initialisation_strategy_name == constants.RANDOM_UNIFORM:
            return np.random.rand(len(self._state_space), len(self._action_space))
        elif initialisation_strategy_name == constants.RANDOM_NORMAL:
            return np.random.normal(
                loc=initialisation_parameters["mean"],
                scale=initialisation_parameters["variance"],
                size=(len(self._state_space), len(self._action_space)),
            )
        elif initialisation_strategy_name == constants.ZEROS:
            return np.zeros((len(self._state_space), len(self._action_space)))
        elif initialisation_strategy_name == constants.ONES:
            return np.ones((len(self._state_space), len(self._action_space)))

    def select_action(self, state):
        state_id = self._state_id_mapping[state]
        _softmax_values = np.exp(self._beta * self._state_action_values[state_id])
        softmax_values = _softmax_values / np.sum(_softmax_values)
        action = np.random.choice(a=range(len(self._action_space)), p=softmax_values)
        return action

    def select_greedy_action(self, state):
        state_id = self._state_id_mapping[state]
        action = np.argmax(self._state_action_values[state_id])
        return action

    @abc.abstractmethod
    def step(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        new_state: Tuple[int, int],
        active: bool,
    ):
        pass
