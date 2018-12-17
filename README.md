COiLTRAiNE: Conditional Imitation Learning Training Framework
-------------------------------------------------------------

!!! Note: The old imitation learning repo is temporarely inside
the "old imitation folder"

This repository can be used to easily train and manage the trainings of imitation
learning networks toguether with the CARLA simulator.
Objectives:

 * To enable the user to perform several trainings with a single command.
 * To automatically test the trained system using CARLA.
 * Allow the user to monitor several trainings
   and testings on CARLA with a single glance.
 * Allows [to perform the testing methodology proposed](docs/on_offline_evaluation.md)
 on the paper "On Offline Evaluation of Vision-based Driving Models"
 * Model ZOO of some published imitation learning approaches. (New
 pull requests accepted)



#### System Overview


![COIL Diagram](docs/img/CoIL.png?raw=true )

The idea of the system is to, by using a single command, train several
experiments with different [network conditions](docs/network.md) and
[input data configurations](docs/input.md).

For a given [experiment configuration file](docs/configuration.md), the main executer function
can perform a [training](docs/training.md), measure the [model prediction error](docs/main_modules.md) on some
dataset and evaluate the model performance on a [driving benchmark](docs/main_modules.md).


The training, prediction error, and driving benchmarks are performed
simultaneusly on different process. To perform evaluation
the validation and driving modules wait for the
training part to produce checkpoints. All of this is controlled
by the [execution module](). The user can execute a single configuration
file but also an entire folder.

During the execution, all the information is [logged and
printed](docs/logger.md) on the screen to summarize the experiment
 training or evaluation status
with a single glance.





#### Getting started

#### Installation

To install the CoIL large scale training framework, we provide a conda environment requirements file.
Basically just do:

    conda env create -f requirements.yml



#### Execution


Assuming you collected and post-processed the data at:

    ~/CARLA/CARLA100 ---
            --/episode_00001
            --/episode_00002
            …

To collect datasets please check the data collector repository.
https://github.com/carla-simulator/data-collector


The first thing you need to do is define the datasets folder.
This is the folder that contains your training and validation datasets

    export COIL_DATASET_PATH=<Path to where your dataset folders are>


##### Single experiment mode

To run a single experiment, we use the flag single-process train
and the experiment name.

A full example to train a ResNet_34  is shown below. There
is also further documentation about ( CIte 3 modules)

    python3 coiltraine.py --single-process train -e resnet34 --folder sample --gpus 0 1 2



##### Folder execution mode

Experiments are defined in config files inside [CoIL/configs](docs/configuration.md).
You can run all the experiments in a folder using:

    python3 coiltraine.py --folder <my_folder> --gpus 0 1 -de <DrivingEnvironmentClass_Town0X> -vd <validation_dataset>

Where the DrivingEnvironmentClass is one of the classes defined in the
modules at [CoIL/drive/suites](docs/suites.md). Those driving environments
define the start and end positions for driving, the number of cars, people, etc.
 That information will define the Benchmark to test the model driving in CARLA in parallel to training.
 Town0X is either Town01 or Town02.
 The
Note that the training dataset must be set on the experiment file directly,
 since training is strictly associated with the experiment.
 The validation datasets are passed as parameter.

 validation datasets ...


#### Paper Results reproduction

* Conditional Imitiation learning
* Conditional IMitation learning CARLA paper
* ECCV
* On