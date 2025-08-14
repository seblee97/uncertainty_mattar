from unc_mattar.agents import base_dyna_learner


class EVBDynaLearner(base_dyna_learner.DynaLearner):
    """EVB as criterion for planning sampling. Replicates Mattar & Daw 2018."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plan(self):
        # Implement the planning step using the EVB criterion approach
        import pdb

        pdb.set_trace()
