import time
from collections import deque

import numpy as np

from typing import Dict, List


def timer(function):
    def f(*args, **kwargs):
        before = time.time()
        function_return = function(*args, **kwargs)
        after = time.time()
        print(f"Time for function {function.__name__}: {round(after - before, 4)}s")
        return function_return

    return f


class ReplayBuffer:

    def __init__(self, max_size: int):

        self._max_size = max_size

        self._states = np.empty(max_size, dtype=np.int32)
        self._actions = np.empty(max_size, dtype=np.int32)
        self._rewards = np.empty(max_size, dtype=np.float32)
        self._next_states = np.empty(max_size, dtype=np.int32)
        self._actives = np.empty(max_size, dtype=np.bool_)

        self._predecessors: Dict[int, List] = {}
        self._predecessors_inv: Dict[int, int] = {}

        self._index = 0
        self._current_size = 0  # Track the actual size of the buffer

    def add(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        active: bool,
    ) -> None:
        overwritten_index = self._index  # Track the index being overwritten

        if overwritten_index in self._predecessors_inv:
            # Remove the overwritten index from predecessors
            old_next_state = self._predecessors_inv[overwritten_index]
            self._predecessors[old_next_state].remove(overwritten_index)
            if not self._predecessors[old_next_state]:  # Clean up empty lists
                del self._predecessors[old_next_state]
            del self._predecessors_inv[overwritten_index]

        self._states[self._index] = state
        self._actions[self._index] = action
        self._rewards[self._index] = reward
        self._next_states[self._index] = next_state
        self._actives[self._index] = active

        # Update predecessors
        if next_state not in self._predecessors:
            self._predecessors[next_state] = []
        self._predecessors[next_state].append(self._index)
        self._predecessors_inv[self._index] = next_state

        # Update the index and size
        self._index = (self._index + 1) % self._max_size
        self._current_size = min(self._current_size + 1, self._max_size)

    def get(self, index: int):
        return (
            self._states[index],
            self._actions[index],
            self._rewards[index],
            self._next_states[index],
            self._actives[index],
        )

    def get_predecessors(self, state_id: int) -> List[int]:
        if state_id not in self._predecessors:
            print(f"Warning: State ID {state_id} has no predecessors.")
        return self._predecessors.get(state_id, [])

    @property
    def buffer(self):
        if len(self) == 0:
            return None  # Handle empty buffer case
        size = len(self)
        return (
            self._states[:size],
            self._actions[:size],
            self._rewards[:size],
            self._next_states[:size],
            self._actives[:size],
        )

    def __len__(self):
        return self._current_size
