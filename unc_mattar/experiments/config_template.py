from config_manager import config_field, config_template

from unc_mattar import constants


class ConfigTemplate:

    _experiment_template = config_template.Template(
        fields=[
            config_field.Field(
                constants.NUM_EPISODES, types=[int], requirements=[lambda x: x > 0]
            ),
            config_field.Field(name=constants.MAP_PATH, types=[str]),
            config_field.Field(name=constants.MAP_YAML_PATH, types=[str]),
            config_field.Field(name=constants.TEST_MAP_YAML_PATH, types=[str]),
            config_field.Field(
                name=constants.TRAIN_EPISODE_TIMEOUT,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.TEST_EPISODE_TIMEOUT,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.EXPERIMENT],
    )

    _logging_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.CHECKPOINT_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.STDOUT_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.LOGGING],
    )

    _random_initialisation_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.MEAN,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.VARIANCE,
                types=[float, int],
                requirements=[lambda x: x >= 0],
            ),
        ],
        level=[constants.LEARNING, constants.RANDOM_INITIALISATION],
        dependent_variables=[constants.INITIALISATION_STRATEGY],
        dependent_variables_required_values=[[constants.RANDOM_NORMAL]],
    )

    _learning_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.LEARNING_RATE,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.TRANSITION_LEARNING_RATE,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.PLANNING_LEARNING_RATE,
                types=[float, int],
            ),
            config_field.Field(
                name=constants.BETA,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.GAMMA,
                types=[int, float],
                requirements=[lambda x: (x > 0) and (x <= 1)],
            ),
            config_field.Field(
                name=constants.PRE_EPISODE_PLANNING_STEPS,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.POST_EPISODE_PLANNING_STEPS,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.K_ADDITIONAL_PLANNING_STEPS,
                types=[int],
                requirements=[lambda x: x >= 0],
            ),
            config_field.Field(
                name=constants.INITIALISATION_STRATEGY,
                types=[str],
                requirements=[
                    lambda x: x in [constants.RANDOM_NORMAL, constants.ZEROS]
                ],
            ),
        ],
        level=[constants.LEARNING],
        nested_templates=[_random_initialisation_template],
    )

    base_config_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.SEED, types=[int]),
            config_field.Field(name=constants.GPU_ID, types=[type(None), int]),
            config_field.Field(
                name=constants.RUNNER,
                types=[str],
                requirements=[
                    lambda x: x in [constants.DYNA, constants.Q_LEARNING, constants.EVB]
                ],
            ),
        ],
        nested_templates=[
            _experiment_template,
            _logging_template,
            _learning_template,
        ],
    )
