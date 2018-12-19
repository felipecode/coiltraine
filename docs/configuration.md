# Configuration


The COiLTRAiNE framework works with a central global module
with all the default configurations. The configurations
set data related things (dataset, input format), network
architecture, training hyperparameters and others.

When executing any process, it first merges a configuration file
with the central global module.

### Files/Batches

The configuration file is defined as  a YAML files.
Each file is associated with a *exp batch*, a folder grouping
a set of *exp alias* that is the name of the YAML file. All
exp batches are located inside the *configs* folder.


A fully commented configuration file can be found at
[configs/sample/resnet34.yaml](../configs/sample/resnet34.yaml)

