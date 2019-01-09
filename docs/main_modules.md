
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

To run a single experiment, we use the flag single-process train
and the experiment name.

To train the [configs/sample/icra_model.yaml](configs/sample/coil_icra.yaml) model, using the GPU 0, run:

    python3 coiltraine.py --single-process train -e coil_icra --folder sample --gpus 0

To train all the models specified on the sample folder:

    python3 coiltraine.py --folder sample --gpus 0

With COiLTRAiNE you can also do simultaneous driving evaluation and validation
on some static dataset.

Also note that the training dataset must be set on the [experiment configuration file](docs/configuration.md) directly,
since training data is strictly associated with the experiment.



#### Validation Curve Dependency Mode

For some experiments we can condition the driving test and the amount
of training into a stop of improvement in validation.

    python3 coiltraine.py --single-process train -e coil_icra --folder val_based_sample --gpus 0

In this example it will train until the slope of the validation
curve becomes positive. After that, it will use this checkpoint
as the checkpoint for testing drive.



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
is passed as parameter to the execution.  Requires the [docker with carla installation](https://carla.readthedocs.io/en/latest/carla_docker/)

For regular execution:

    python3 coiltraine.py --folder sample --gpus 0 -de <DrivingScenario_Town0X> --docker carlasim/carla:version


Where carlasim/carla:version is the installed docker version of CARLA.
The DrivingScenario is a suite class defined inside the drive/suites folder.
Each scenario defines the start and end positions for driving, the number of cars and pedestrians,
weathers etc. The Town0X is the town used on the scenario and must
 mach its definition, for now it is either Town01 or Town02.


Fianally, multiple driving evaluation instances are also allowed,
 so you can also to run the
[CoRL 2017 benchmark](https://github.com/carla-simulator/driving-benchmarks/blob/master/Docs/benchmark_start.md/#corl-2017)
over 4 CARLA processes on the sample experiments, run:

    python3 coiltraine.py --folder sample -de CorlTraining_Town01 CorlNewWeather_Town01 CorlNewTown_Town02   CorlNewWeatherTown_Town02 --docker carlasim/carla:version

