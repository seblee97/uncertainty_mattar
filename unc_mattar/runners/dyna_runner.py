from unc_mattar.runners import base_runner
from unc_mattar.agents import dyna_learner

import copy


class DynaRunner(base_runner.BaseRunner):

    def __init__(
        self,
        learning_rate,
        transition_learning_rate,
        beta,
        gamma,
        num_episodes,
        train_episode_timeout,
        test_episode_timeout,
        pre_episode_planning_steps,
        post_episode_planning_steps,
        map_path,
        map_yaml_path,
        test_map_yaml_path,
        initialisation_strategy,
    ):
        super().__init__(
            learning_rate,
            beta,
            gamma,
            num_episodes,
            train_episode_timeout,
            test_episode_timeout,
            map_path,
            map_yaml_path,
            test_map_yaml_path,
        )

        self._agent = dyna_learner.DynaLearner(
            action_space=self._train_env.action_space,
            state_space=self._train_env.state_space,
            learning_rate=self._learning_rate,
            transition_learning_rate=self._transition_learning_rate,
            gamma=self._gamma,
            beta=self._beta,
            initialisation_strategy=initialisation_strategy,
        )
        self._populate_transition_matrix()

        self._pre_episode_planning_steps = pre_episode_planning_steps
        self._post_episode_planning_steps = post_episode_planning_steps

        self._transition_learning_rate = transition_learning_rate

    def _populate_transition_matrix(self):
        dummy_env = copy.deepcopy(self._train_env)

        for state in dummy_env.state_space:
            for action in dummy_env.action_space:
                dummy_env.reset_environment(train=True)
                dummy_env._agent_position = state
                _, new_state = dummy_env.step(action)
                self._agent.increment_transition_matrix(state, new_state)
                self._agent.add_to_replay_buffer(
                    state, action, 0, new_state, self._train_env.active
                )

        self._agent.normalise_transition_matrix()

        # TODO: transition from reward to start state

    def _train_episode(self):
        episode_return = 0
        episode_length = 0

        state = self._train_env.reset_environment(train=True)

        for _ in range(self._pre_episode_planning_steps):
            self._agent.plan()

        while self._train_env.active:

            action = self._agent.select_action(state)
            reward, new_state = self._train_env.step(action)

            self._agent.step(
                state=state,
                action=action,
                reward=reward,
                new_state=new_state,
                active=self._train_env.active,
            )

            state = new_state
            episode_return += reward
            episode_length += 1

        for _ in range(self._post_episode_planning_steps):
            self._agent.plan()

        return episode_return, episode_length
