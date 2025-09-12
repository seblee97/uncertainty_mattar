from unc_mattar.agents import base_dyna_learner

import numpy as np


class EVBDynaLearner(base_dyna_learner.DynaLearner):
    """EVB as criterion for planning sampling. Replicates Mattar & Daw 2018."""

    def __init__(
        self,
        action_space,
        state_space,
        # positional_state_space,
        learning_rate,
        transition_learning_rate,
        planning_learning_rate,
        gamma,
        beta,
        initialisation_strategy,
        max_buffer_size,
        top_k,
        max_chain,
    ):
        super().__init__(
            action_space=action_space,
            state_space=state_space,
            learning_rate=learning_rate,
            transition_learning_rate=transition_learning_rate,
            planning_learning_rate=planning_learning_rate,
            gamma=gamma,
            beta=beta,
            initialisation_strategy=initialisation_strategy,
            max_buffer_size=max_buffer_size,
        )

        self._top_k = top_k
        self._max_chain = max_chain

    def _get_best_evb_transitions(self, buffer, sr_row):
        """
        TODO
        """

        state_ids = buffer[0]
        actions = buffer[1]
        rewards = buffer[2]
        next_state_ids = buffer[3]
        actives = buffer[4]

        needs = sr_row[state_ids]

        # Hypothetical Q-learning update for all transitions in buffer
        discounts = np.where(actives, self._gamma, 0.0)
        q_current = self._state_action_values[state_ids, actions]
        q_next_max = np.max(self._state_action_values[next_state_ids], axis=1)
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

        idx = np.argsort(evbs)[: self._top_k]

        return idx

    def _get_continuation_chain(self, replay_element):
        """
        # TODO
        """
        # n-step chains
        if replay_element[1] != np.argmax(self._state_action_values[replay_element[0]]):
            return [replay_element[:3], replay_element[0]]

        chain = [replay_element[:3]]

        while len(chain) < self._max_chain:
            predecessors = self._replay_buffer.get_predecessors(chain[0][0])

            if not predecessors:
                break

            link = None

            for predecessor_idx in predecessors:
                predecessor = self._replay_buffer.get(predecessor_idx)

                s_pred, a_pred, r_pred, s_next_pred, _ = predecessor

                if s_next_pred != chain[0][0]:
                    continue

                greedy_a_pred = np.argmax(self._state_action_values[s_pred])
                if a_pred != greedy_a_pred:
                    continue
                greedy_a_chain = np.argmax(self._state_action_values[chain[0][0]])
                if chain[0][1] != greedy_a_chain:
                    continue

                link = (s_pred, a_pred, r_pred)
                break

            if link is None:
                break

            chain.insert(0, link)

        return chain + [replay_element[3]]

    def _get_n_step_target(self, trajectory):
        """
        # TODO
        """
        G, g = 0.0, 1.0
        for _, _, r in trajectory[:-1]:
            G += g * r
            g *= self._gamma
        G += g * np.max(self._state_action_values[trajectory[-1]])
        return G

    def _score_episode_Q1_evb(self, episode_chain, sr_row):
        # setup eligibility trace
        e = np.zeros_like(self._state_action_values)

        Q_h = self._state_action_values.copy()
        accumulated_gain = 0.0

        if len(episode_chain) < 2:
            return 0.0, 0.0

        if isinstance(episode_chain[-1], (int, np.integer)):
            s_k = episode_chain[-1]
        else:
            s_k = episode_chain[-1][0]

        T = len(episode_chain) - 1

        for t in range(T):
            s_t, a_t, r_t = episode_chain[t]
            if isinstance(episode_chain[t + 1], (int, np.integer)):
                s_tp = episode_chain[t + 1]
            else:
                s_tp = episode_chain[t + 1][0]

            if a_t != np.argmax(Q_h[s_t]):
                e[:] = 0.0
                break

            value_old = np.max(Q_h[s_t])

            # TD error with greedy bootstrap (Watkins' Q) and λ=1
            if t == T - 1:
                value_tp1 = 0.0
            else:
                value_tp1 = np.max(Q_h[s_tp])
            delta = r_t + self._gamma * value_tp1 - Q_h[s_t, a_t]

            # traces: accumulate then decay by gamma (since λ=1)
            e[s_t, a_t] += 1.0
            Q_h += self._learning_rate * delta * e
            e *= self._gamma

            # local gain at s_t
            local_gain = np.max(Q_h[s_t]) - value_old
            if local_gain > 0:
                accumulated_gain += local_gain

            if t < T - 1:
                a_tp1 = episode_chain[t + 1][1]
                if a_tp1 != np.argmax(Q_h[s_tp]):
                    e[:] = 0.0
                    break

        end_need = sr_row[s_k]
        episode_evb = accumulated_gain * end_need

        return episode_evb, accumulated_gain

    def _apply_episode_Q1(self, episode_chain):
        # setup eligibility trace
        e = np.zeros_like(self._state_action_values)

        accumulated_gain = 0.0

        if len(episode_chain) < 2:
            return 0.0

        T = len(episode_chain) - 1
        for t in range(T):
            s_t, a_t, r_t = episode_chain[t]
            if isinstance(episode_chain[t + 1], (int, np.integer)):
                s_tp = episode_chain[t + 1]
            else:
                s_tp = episode_chain[t + 1][0]

            value_old = np.max(self._state_action_values[s_t])

            # TD error with greedy bootstrap (Watkins' Q) and λ=1
            if t == T - 1:
                value_tp1 = 0.0
            else:
                value_tp1 = np.max(self._state_action_values[s_tp])
            delta = r_t + self._gamma * value_tp1 - self._state_action_values[s_t, a_t]

            # traces: accumulate then decay by gamma (since λ=1)
            e[s_t, a_t] += 1.0
            self._state_action_values += self._planning_lr * delta * e
            e *= self._gamma

            # local gain at s_t
            local_gain = np.max(self._state_action_values[s_t]) - value_old
            if local_gain > 0:
                accumulated_gain += local_gain

            if t < T - 1:
                a_tp1 = episode_chain[t + 1][1]
                if a_tp1 != np.argmax(self._state_action_values[s_tp]):
                    e[:] = 0.0
                    break

        return accumulated_gain

    def plan(self, current_state):

        # SR matrix for need term
        sr_matrix = self._get_successor_matrix()
        current_state_id = self._state_id_mapping[current_state]
        sr_row = sr_matrix[current_state_id]

        buffer = self._replay_buffer.buffer

        # 1-step EVB for all transitions in buffer, identify best
        idx = self._get_best_evb_transitions(buffer, sr_row)

        # use best as seed tail to build n-step chain
        episode_chains = [
            self._get_continuation_chain(self._replay_buffer.get(idx_i))
            for idx_i in idx
        ]

        best_evb, best_i, best_gain = -0.0, None, 0.0
        for i, episode_chain in enumerate(episode_chains):
            episode_evb, gain = self._score_episode_Q1_evb(episode_chain, sr_row)
            if episode_evb > best_evb:
                best_evb = episode_evb
                best_i = i
                best_gain = gain

        if best_i is None or best_evb <= 0:
            return {"applied": False, "evb": None, "gain": None}

        episode_chain = episode_chains[best_i]

        # apply best n-step update
        self._apply_episode_Q1(episode_chain)

        return {"applied": True, "evb": best_evb, "gain": best_gain}

        # # TODO: n-step updates

        # transition_sample = (
        #     state_ids[idx],
        #     actions[idx],
        #     rewards[idx],
        #     next_state_ids[idx],
        #     actives[idx],
        #     self._planning_lr,
        # )
        # self._step(*transition_sample)
