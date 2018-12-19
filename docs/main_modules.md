
Core Modules
============

There are three kinds of process that are executed on
COiLTRAiNE.

### Train

Trains a controller model given a configuration file.
The training process produces checkpoints with respect to
the SAVE_SCHEDULE attribute. These checkpoints are used
to evaluate the model on a dataset or in some driving benchmark.

The training produces logs to be printed on the screen and
also tensorboard logs.

To configure the training you can adjust [model configurations](docs/network.md),
such as network architecture and loss function and also the
 [input data](docs/input.md) format and distribuction.
To train a single process:

    python3 coiltraine.py --single-process train -e coil_icra --folder sample --gpus 0

To train all the models specified on the sample folder:

    python3 coiltraine.py --folder sample --gpus 0



### Validation

The validation module produces one output for each of the data
inputs from a some validation dataset. This is stored inside csv
files located at _logs/[<exp_batches>/<exp_alias>](docs/configuration.md/#files/batches)
/<validation_dataset_name>_csv.

To run the validation for the  as a single process, configs/sample/icra_model.yaml,
 with a certain validation dataset name:

    python3 coiltraine.py --single-process validation -e coil_icra --folder sample --gpus 0 -vd <validation_dataset>

Note that the <validation_dataset> must be inside the folder defined
 in the COIL_DATASET_PATH env variable. To perform validation on
 the two sample validation datasets:

     python3 coiltraine.py --single-process validation -e coil_icra --folder sample --gpus 0 -vd CoILVal1 CoILVal2


 The CoILVal1 an CoILVal2 datasets can be downloaded running
 the "tools/get_sample_datasets.py" script.
 Note that the validation will not execute if the model has not been
 trained yet.



### Drive

The driving process is executed over the run_drive.py script.
A different process is executed for every driving environment that
is passed as parameter to the execution. For instance to run the
[CoRL 2017 benchmark](https://github.com/carla-simulator/driving-benchmarks/blob/master/Docs/benchmark_start.md/#corl-2017)
over 4 CARLA processes on the sample experiments, run:

    python3 coiltraine.py --folder sample -de CorlTraining_Town01 CorlNewWeather_Town01 CorlNewTown_Town02 CorlNewWeatherTown_Town02

There are two types of execution:

* Regular: It starts carla process. for that running mode.
* Docker



What it does is to