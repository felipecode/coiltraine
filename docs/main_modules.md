
Main Modules
============

There are three kinds of process that are executed on
COiLTRAiNE.

### Train

Trains a controller model given a configuration file.
The training process produces checkpoints with respect to
the SAVE_SCHEDULE attribute. These checkpoints are used
to evaluate the model on a dataset or in some driving benchmark.

The training produces logs to be printed on the screen but it
also produces tensorboard logs.

To configure the training you can adjust
some [network configurations](docs/network.md).


### Validation


### Drive

The driving process is executed over the run_drive.py script.
A different process is executed for every driving environment that
is passed as parameter to the execution. For instance to run the
[CoRL 2017 benchmark] over 4 CARLA processes on the sample experiments,
run:

    python3 coiltraine.py --folder sample -de CorlTraining_Town01 CorlNewWeather_Town01 CorlNewTown_Town02 CorlNewWeatherTown_Town02

There are two types of execution:

    * Regular
    * Docker



What it does is to