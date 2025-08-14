import itertools
import numpy as np


CONFIG_CHANGES = {
    f"{model}_{seed}_{p}": [
        {
            "seed": seed,
            "runner": model,
            "learning": {
                "pre_episode_planning_steps": p,
                "post_episode_planning_steps": p,
            },
        }
    ]
    for model, seed, p in itertools.product(["dyna", "q_learning"], [0], [1, 2, 5])
}
