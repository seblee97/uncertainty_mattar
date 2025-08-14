import itertools
import numpy as np


CONFIG_CHANGES = {
    f"{model}_{seed}_{p}": [
        {
            "seed": seed,
            "runner": model,
            "learning": {
                "pre_episode_planning_steps": 0,
                "post_episode_planning_steps": 0,
                "k_additional_planning_steps": p,
            },
        }
    ]
    for model, seed, p in itertools.product(["dyna"], [0, 1, 2, 3, 4, 5], [0, 1, 2, 5])
}
