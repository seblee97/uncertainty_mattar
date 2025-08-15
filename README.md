# Normative Models for Prioritized Experience Replay

This codebase implements a variety of normative models for prioritized replay. The basis of the work can be found in [this](https://pubmed.ncbi.nlm.nih.gov/30349103/) seminal paper by Marcelo Mattar and Nathaniel Daw. Here we extend their formulation to include concepts related to uncertainty. 

# Setup

It is recommended to set up this package within a virtual environment e.g. using conda. The easiest method is to use the configuration file provided via:

```conda env create -f env.yaml```

This should create a virtual environment, install the necessary package requirements, and install this repository as a package within the environment. 

### Current minor snags

There is a minor bug in one of the upstream packages that will be installed as part of the above process. A quick fix for this (until the upstream package is remedied) is to change.

# Structure and Usage

