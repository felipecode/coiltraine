Logger and Printer
==================

The [processes used](docs/main_modules.md) on the COiLTRAiNE framework
write outputs about their current status.
So, by parsing the logs, with a single
glance, the user could update himself about the status of
the experiments.



The logs have the following organization

```
_logs
│
└───<exp_batch>
    │
    └──<exp_alias1>
    │   └─ checkpoints
    │
    └──<exp_alias2>
    ...


```

Where <exp batch> is a folder
of experiments and <exp alias> is a single experiment


The logger is global and accessible at any moment.

The experiments have a set of status for each part of
the logging. The general status is the last one that stopped.
For our system the following message types exist.

* Not Started
* Loading
* Iterating
* Error
* Finished


### Training Messages

### Validation Messages

### Driving Messages

