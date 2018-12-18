COiLTRAiNE: Conditional Imitation Learning Training Framework
-------------------------------------------------------------

!!! Note: The old imitation learning repo is temporarily inside
the "old imitation folder"

This repository can be used to easily train and manage the trainings of imitation
learning networks jointly with evaluations  on the CARLA simulator.
Objectives:

 * To enable the user to perform several trainings with a single command.
 * To automatically test the trained systems using CARLA.
 * Allow the user to monitor several trainings
   and testings on CARLA with a single glance.
 * Allows [to perform the testing methodology proposed](docs/on_offline_evaluation.md)
 on the paper "On Offline Evaluation of Vision-based Driving Models"
 * Model ZOO of some published imitation learning approaches. (New
 pull requests accepted)



### System Overview


![COIL Diagram](docs/img/CoIL.png?raw=true )

The idea of the system is to, by using a single command, train several
experiments with different [model conditions](docs/network.md) and
[input data configurations](docs/input.md).

For a given [experiment configuration file](docs/configuration.md), the main executer function
can perform:  [training](docs/training.md), measurement of the [model prediction error](docs/main_modules.md) on some
dataset and evaluation of the model performance on a [driving benchmark](docs/main_modules.md).


The training, prediction error, and driving benchmarks are performed
simultaneusly on different process. To perform evaluation
the validation and driving modules wait for the
training part to produce checkpoints. All of this is controlled
by the [executer module](docs/executer.md).


During the execution, all the information is [logged and
printed](docs/logger.md) on the screen to summarize the experiment
 training or evaluation status to be checked by a single glance
 of a user.




### Getting started

#### Installation

To install COiLTRAiNE, we provide a conda environment requirements file.
Start by cloning the repository on some folder and then to
install just run:

    conda env create -f requirements.yml



#### Setting Environment/ Getting Data

The first thing you need to do is define the datasets folder.
This is the folder that will contain your training and validation datasets

    export COIL_DATASET_PATH=<Path to where your dataset folders are>

You can also download a sample dataset pack, with one training
and two validations, by running

    python3 tools/get_sample_datasets.py

The datasets; CoilTrain , CoilVal1 and CoilVal2; will be stored at
 the COIL_DATASET_PATH folder.

To collect other datasets please check the data collector repository.
https://github.com/carla-simulator/data-collector



#### Single experiment mode

To run a single experiment, we use the flag single-process train
and the experiment name.

To train the [configs/sample/icra_model.yaml](configs/sample/coil_icra.yaml) model, using the GPU 0, run: 

    python3 coiltraine.py --single-process train -e coil_icra --folder sample --gpus 0

There other two process that could be run: [drive](docs/main_modules.md/#drive)
and [validation](docs/main_modules.md/#validation).


#### Folder execution mode

Experiments are defined in config files inside [CoIL/configs](docs/configuration.md).
You can run all the experiments in a folder using:

    python3 coiltraine.py --folder <my_folder> --gpus 0 1 -de <DrivingEnvironmentClass_Town0X> -vd <validation_dataset>

Where the DrivingEnvironmentClass is one of the classes defined in the
modules at [CoIL/drive/suites](docs/suites.md). Those driving environments
define the start and end positions for driving, the number of cars and pedestrians, etc.
 That information will define the scenario to be driven the by model inside CARLA.
 Town0X is either Town01 or Town02.
Note that the training dataset must be set on the [experiment configuration file](docs/configuration.md) directly,
since training is strictly associated with the experiment.
The validation datasets are passed as parameter with -vd <validation_dataset_list>,
 for instance, using -vd CoilVal1 CoilVal2




#### Conditional Models Zoo

* Conditional Imitation Learning
* Conditional Imitation Learning CARLA paper
* On Offline Evaluation of Vision-based Driving Models
* New One (Soon)





