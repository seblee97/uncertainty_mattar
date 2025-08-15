# Normative Models for Prioritized Experience Replay

This codebase implements a variety of normative models for prioritized replay. The basis of the work can be found in [this](https://pubmed.ncbi.nlm.nih.gov/30349103/) seminal paper by Marcelo Mattar and Nathaniel Daw. Here we extend their formulation to include concepts related to uncertainty. 

# Setup

It is recommended to set up this package within a virtual environment e.g. using conda. The easiest method is to use the configuration file provided via:

```conda env create -f env.yaml```

This should create a virtual environment, install the necessary package requirements, and install this repository as a package within the environment. 

### Current minor snags

There is a minor bug in one of the upstream packages that will be installed as part of the above process. A quick fix for this (until the upstream package is remedied) is to change.

# Structure and Usage

Experiments are run from ```uncertainty_mattar/unc_mattar/experiments``` via ```python run.py```. The primary interface for changing the parameters of an experiment are under the ```config.yaml``` file and the structure of this file is restricted by a template specified in ```config_template.py```; both are in the experiments folder. The gridworld maps are stored under ```uncertainty_mattar/unc_mattar/maps```, although they can be stored anywhere in principle. Results for experiments are stored under ```uncertainty_mattar/unc_mattar/results``` by default. 

### Run Modes

By default the run command above will trigger an experiment corresponding to the configuration specified by the config yaml file. For ablation studies or more systematic experiments, a different mode can be used via the mode flag i.e. ```python run.py --mode X```, where X can be _single_ (default), _serial_ (runs multiple experiments in serial), _parallel_ (runs multiple experiments in parallel), or _cluster_ (runs an array of jobs on clusters using slurm or univa schedulers). The array of experiments that will run will be determined by the config yaml file along with modifications specified in the ```config_changes.py``` file. 