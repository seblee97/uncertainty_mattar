from unc_mattar import per_runner

import matplotlib.pyplot as plt

LEARNING_RATE = 0.01
BETA = 5.0
GAMMA = 0.9
NUM_EPISODES = 5000
TRAIN_EPISODE_TIMEOUT = 200
TEST_EPISODE_TIMEOUT = 1000
MAP_PATH = "maze.txt"
MAP_YAML_PATH = "maze.yaml"
TEST_MAP_YAML_PATH = "test_maze.yaml"
INITIALISATION_STRATEGY = {"random_normal": {"mean": 0, "variance": 0.1}}

runner = per_runner.Runner(
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
)

res = runner.train()

fig = plt.figure()
plt.plot(res[0])
fig.savefig("train_returns.png")

fig = plt.figure()
plt.plot(res[1])
fig.savefig("train_lengths.png")

fig = plt.figure()
plt.plot(res[2])
fig.savefig("test_returns.png")

fig = plt.figure()
plt.plot(res[3])
fig.savefig("test_lengths.png")
