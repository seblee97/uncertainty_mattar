from unc_mattar.runners import base_runner
from unc_mattar.agents import q_learner


class QRunner(base_runner.BaseRunner):

    def __init__(
        self,
        learning_rate,
        beta,
        gamma,
        num_episodes,
        train_episode_timeout,
        test_episode_timeout,
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

        self._agent = q_learner.QLearner(
            action_space=self._train_env.action_space,
            state_space=self._train_env.state_space,
            learning_rate=self._learning_rate,
            gamma=self._gamma,
            beta=self._beta,
            initialisation_strategy=initialisation_strategy,
        )

    def _train_episode(self):
        episode_return = 0
        episode_length = 0

        state = self._train_env.reset_environment(train=True)

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

        return episode_return, episode_length
