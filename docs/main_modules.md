
Main Modules
============

There are three kinds of process that are executed on
COiLTRAiNE.

### Train

Runs a training for process based on a configuration file.

### Validation


### Drive

The driving process is executed over the run_drive.py script.
A different process is executed for every driving environment that
is passed as parameter to the execution. For instance to run the
CoRL 2017 benchmark over 4 CARLA process on the sample experiments,
run:

    python3 coiltraine.py --folder sample -de CorlTraining_Town01 CorlNewWeather_Town01 CorlNewTown_Town02 CorlNewWeatherTown_Town02

There are two types of execution, over docker



What it does is to