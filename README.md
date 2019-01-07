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


The [COiLTRAiNE](docs/coiltraine.md) framework allows simultaneous [training](docs/main_modules.md/#train), driving on [scenarios in CARLA](docs/main_modules.md/#drive) and [prediction of controls](docs/main_modules.md/#validation) on some static dataset. This process can be done on several experiments at the same time. 



### Getting started

#### Installation

To install COiLTRAiNE, we provide a conda environment requirements file.
Start by cloning the repository on some folder and then, to
install, just run:

    conda env create -f requirements.yml

#### Setting Environment/ Getting Data

The first thing you need to do is define the datasets folder.
This is the folder that will contain your training and validation datasets

    export COIL_DATASET_PATH=<Path to where your dataset folders are>

Download a sample dataset pack, with one training
and two validations, by running

    python3 tools/get_sample_datasets.py

The datasets; CoILTrain , CoILVal1 and CoILVal2; will be stored at
 the COIL_DATASET_PATH folder.

To collect other datasets please check the data collector repository.
https://github.com/carla-simulator/data-collector

For doing scenario evaluation in CARLA you must download CARLA 0.8.4 or CARLA 0.8.2
and unpack it in some directory. After that, you should set the CARLA_PATH
variable with the path to reach the CARLA root directory:

    export CARLA_PATH=<carla_root_directory>

#### Executing

 Assuming that you set the CARLA_PATH, you can run the coiltraine system by running:
     
    python3 coiltraine.py --folder sample --gpus 0 -de CorlNewWeather_Town01 -vd CoILVal1

Where the --folder sample is the [experiment batch](https://github.com/felipecode/CoIL/blob/master/docs/configuration.md)
containg all the experiments taht are going to 
be trained and validated.
The CorlTraining is a driving scenario on Town01, defined as one of the classes on the
drive/suites folder. The validation datasets are passed as parameter with -vd  and should be placed 
at the COIL_DATASET_PATH folder.

Finally, note that the execution of the driving scenario on CARLA can also be done [using docker](docs/main_modules.md/#drive),
 option which we recommend.




#### Conditional Models Zoo

* Conditional Imitation Learning
* Conditional Imitation Learning CARLA paper
* [On Offline Evaluation of Vision-based Driving Models](docs/on_offline_evaluation.md) [paper](https://arxiv.org/abs/1809.04843)
* New One (Soon)




