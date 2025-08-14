from unc_mattar.runners import base_runner
from unc_mattar.agents import random_dyna_learner, evb_dyna_learner

from unc_mattar import constants, experiments

import copy


class DynaRunner(base_runner.BaseRunner):

    def __init__(
        self,
        config: experiments.config.Config,
        unique_id: str = "",
    ):

        super().__init__(config=config, unique_id=unique_id)

        self._transition_learning_rate = config.transition_learning_rate

        self._pre_episode_planning_steps = config.pre_episode_planning_steps
        self._post_episode_planning_steps = config.post_episode_planning_steps
        self._k_additional_planning_steps = config.k_additional_planning_steps

        print(config.runner)

        if config.runner == constants.DYNA:
            self._agent = random_dyna_learner.RandomDynaLearner(
                action_space=self._train_env.action_space,
                state_space=self._train_env.state_space,
                learning_rate=self._learning_rate,
                transition_learning_rate=self._transition_learning_rate,
                gamma=self._gamma,
                beta=self._beta,
                initialisation_strategy=self._initialisation_strategy,
            )
        elif config.runner == constants.EVB:
            self._agent = evb_dyna_learner.EVBDynaLearner(
                action_space=self._train_env.action_space,
                state_space=self._train_env.state_space,
                learning_rate=self._learning_rate,
                transition_learning_rate=self._transition_learning_rate,
                gamma=self._gamma,
                beta=self._beta,
                initialisation_strategy=self._initialisation_strategy,
            )
        else:
            raise ValueError(
                f"Runner {config.runner} not recognised. " "Please use 'dyna' or 'evb'."
            )

        self._populate_transition_matrix()

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

            for _ in range(self._k_additional_planning_steps):
                self._agent.plan()

        for _ in range(self._post_episode_planning_steps):
            self._agent.plan()

        return episode_return, episode_length
