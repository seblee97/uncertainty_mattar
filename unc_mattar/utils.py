import time
from collections import deque

import numpy as np

from typing import Dict, List, Tuple


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

    def get_random(self):
        """Return a random (s, a, r, s_next, done) tuple from the buffer."""
        return self.get(np.random.choice(len(self)))

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


class ModelBuffer:
    """
    Fixed model-style replay buffer for tabular RL.
    Exactly one entry per (s,a). Overwrites the previous entry for that (s,a)
    (In contrast to above buffer).

    Data stored per (s,a): r, s_next, done.
    Efficient predecessor lookup via intrusive linked lists per successor state:
      head[s] -> first index i with s_next[i] == s (or -1)
      next[i], prev[i] -> pointers within that bucket
    """

    def __init__(self, num_states: int, num_actions: int):

        self._num_states = num_states
        self._num_actions = num_actions
        self._N = self._num_states * self._num_actions

        # model tables
        self._rewards = np.zeros(self._N)
        self._next_states = np.full(self._N, -1)
        self._actives = np.zeros(self._N)

        # intrusive linked lists per successor state (for predecessors())
        self._heads = np.full(self._num_states, -1)
        self._nexts = np.full(self._N, -1)
        self._prevs = np.full(self._N, -1)

        # convenience arrays for decoding (s,a) from idx without div/mod each time
        self._state_of = np.arange(self._N) // self._num_actions
        self._action_of = np.arange(self._N) % self._num_actions

    def _state_action_to_idx(self, state_index: int, action: int) -> int:
        """
        Convert (s,a) to flat index idx = s*A + a.
        """
        return state_index * self._num_actions + action

    def _insert_into_bucket(self, idx: int, succ: int):
        """Insert idx at the head of bucket for successor 'succ'."""
        head = self._heads[succ]
        self._prevs[idx] = -1
        self._nexts[idx] = head
        if head != -1:
            self._prevs[head] = idx
        self._heads[succ] = idx

    def _remove_from_bucket(self, idx: int, succ: int):
        """Remove idx from bucket for successor 'succ' if linked."""
        prev, next = self._prevs[idx], self._nexts[idx]
        if prev != -1:
            self._nexts[prev] = next
        else:
            self._heads[succ] = next
        if next != -1:
            self._prevs[next] = prev
        self._prevs[idx] = -1
        self._nexts[idx] = -1

    def add(
        self, state: int, action: int, reward: float, next_state: int, active: bool
    ):
        """
        Overwrite the (s,a) slot with new (r, s_next, done).
        Maintains predecessor buckets in O(1).
        """
        idx = self._state_action_to_idx(state, action)

        old_succ = int(self._next_states[idx])
        if 0 <= old_succ < self._num_states:
            self._remove_from_bucket(idx, old_succ)

        # write new data
        self._rewards[idx] = reward
        self._next_states[idx] = next_state
        self._actives[idx] = active

        # link into new bucket if successor
        if 0 <= next_state < self._num_states:
            self._insert_into_bucket(idx, next_state)

    def get_by_state_action(
        self, state: int, action: int
    ) -> Tuple[bool, float, int, bool]:
        """Return (r, s_next, done) for (s,a)."""
        idx = self._state_action_to_idx(state, action)
        return (
            self._rewards[idx],
            self._next_states[idx],
            self._actives[idx],
        )

    def get(self, idx: int) -> Tuple[int, int, bool, float, int, bool]:
        """Return (s, a, r, s_next, done) for flat index idx."""
        state = self._state_of[idx]
        action = self._action_of[idx]
        return (
            state,
            action,
            self._rewards[idx],
            self._next_states[idx],
            self._actives[idx],
        )

    def get_random(self):
        """Return a random (s, a, r, s_next, done) tuple from the buffer."""
        valid_indices = np.where(self._next_states != -1)[0]
        if len(valid_indices) == 0:
            raise ValueError("The buffer is empty.")
        random_idx = np.random.choice(valid_indices)
        return self.get(random_idx)

    def predecessors(self, succ: int) -> List[int]:
        """
        Return flat indices of all (s,a) whose stored successor equals 'succ'.
        """
        out = []
        i = self._heads[succ]
        while i != -1:
            if self._next_states[i] == succ:
                out.append(int(i))
            i = self._nexts[i]
        return out

    # def arrays_for_valid(self):
    #     """
    #     Vectorized view of all valid entries: returns (idx, s, a, r, s_next, done).
    #     Useful for bulk EVB computations without Python loops.
    #     """
    #     idx = self.valid_indices()
    #     return (
    #         idx,
    #         self._s_of[idx],
    #         self._a_of[idx],
    #         self.r[idx],
    #         self.s_next[idx],
    #         self.done[idx],
    #     )

    def clear(self):
        self._rewards.fill(0)
        self._next_states.fill(-1)
        self._actives.fill(False)
        self._heads.fill(-1)
        self._nexts.fill(-1)
        self._prevs.fill(-1)

    @property
    def buffer(self):
        states = self._state_of[np.arange(self._N)]
        actions = self._action_of[np.arange(self._N)]
        return (states, actions, self._rewards, self._next_states, self._actives)

    def __len__(self):
        return np.sum(self._next_states != -1)
