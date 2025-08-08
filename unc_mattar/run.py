from unc_mattar.runners import q_runner, dyna_runner

import os
import matplotlib.pyplot as plt
import numpy as np

import time
import datetime

NUM_REPEATS = 100
NUM_EPISODES = 50

LEARNING_RATE = 1
TRANSITION_LEARNING_RATE = 0.9
BETA = 5.0
GAMMA = 0.9
TRAIN_EPISODE_TIMEOUT = 100000
TEST_EPISODE_TIMEOUT = 100000
PRE_EPISODE_PLANNING_STEPS = 20
POST_EPISODE_PLANNING_STEPS = 20
MAP_PATH = "maps/maze.txt"
MAP_YAML_PATH = "maps/maze.yaml"
TEST_MAP_YAML_PATH = "maps/test_maze.yaml"
INITIALISATION_STRATEGY = {
    "zeros": None
}  # {"random_normal": {"mean": 2, "variance": 0.1}}

RUNNER = "q"

raw_datetime = datetime.datetime.fromtimestamp(time.time())
exp_path = os.path.join("results", raw_datetime.strftime("%Y-%m-%d-%H-%M-%S"))

os.makedirs(exp_path, exist_ok=True)


def single_run():
    if RUNNER == "dyna":
        runner = dyna_runner.DynaRunner(
            learning_rate=LEARNING_RATE,
            transition_learning_rate=TRANSITION_LEARNING_RATE,
            beta=BETA,
            gamma=GAMMA,
            num_episodes=NUM_EPISODES,
            train_episode_timeout=TRAIN_EPISODE_TIMEOUT,
            test_episode_timeout=TEST_EPISODE_TIMEOUT,
            pre_episode_planning_steps=PRE_EPISODE_PLANNING_STEPS,
            post_episode_planning_steps=POST_EPISODE_PLANNING_STEPS,
            map_path=MAP_PATH,
            map_yaml_path=MAP_YAML_PATH,
            test_map_yaml_path=TEST_MAP_YAML_PATH,
            initialisation_strategy=INITIALISATION_STRATEGY,
            exp_path=exp_path,
        )
    elif RUNNER == "q":
        runner = q_runner.QRunner(
            learning_rate=LEARNING_RATE,
            beta=BETA,
            gamma=GAMMA,
            num_episodes=NUM_EPISODES,
            train_episode_timeout=TRAIN_EPISODE_TIMEOUT,
            test_episode_timeout=TEST_EPISODE_TIMEOUT,
            map_path=MAP_PATH,
            map_yaml_path=MAP_YAML_PATH,
            test_map_yaml_path=TEST_MAP_YAML_PATH,
            initialisation_strategy=INITIALISATION_STRATEGY,
            exp_path=exp_path,
        )

    return runner.train()


res = np.array([single_run() for _ in range(NUM_REPEATS)])

fig = plt.figure()
plt.plot(np.mean(res[:, 0], axis=0))
fig.savefig(os.path.join(exp_path, "train_returns.png"))

fig = plt.figure()
plt.plot(np.mean(res[:, 1], axis=0))
fig.savefig(os.path.join(exp_path, "train_lengths.png"))

fig = plt.figure()
plt.plot(np.mean(res[:, 2], axis=0))
fig.savefig(os.path.join(exp_path, "test_returns.png"))

fig = plt.figure()
plt.plot(np.mean(res[:, 3], axis=0))
fig.savefig(os.path.join(exp_path, "test_lengths.png"))
