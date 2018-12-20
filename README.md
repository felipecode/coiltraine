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
can perform:  [training](docs/main_modules.md/#train), measurement of the [model prediction error](docs/main_modules.md/#validation) on some
dataset and evaluation of the model performance on a [driving benchmark](docs/main_modules.md/#drive).


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

For doing scenario evaluation in CARLA you must download CARLA 0.8.4 or CARLA 0.8.2
and unpack it in some directory. After that, you should set the CARLA_PATH
variable with the path to reach the CARLA root directory:

    export CARLA_PATH=<carla_root_directory>


#### Single experiment mode

To run a single experiment, we use the flag single-process train
and the experiment name.

To train the [configs/sample/icra_model.yaml](configs/sample/coil_icra.yaml) model, using the GPU 0, run: 

    python3 coiltraine.py --single-process train -e coil_icra --folder sample --gpus 0

There are other two process that could be run: [drive](docs/main_modules.md/#drive)
and [validation](docs/main_modules.md/#validation).


#### Folder execution mode

Experiments are defined in config files inside [CoIL/configs](docs/configuration.md).
You can train all the experiments in a folder using:

    python3 coiltraine.py --folder sample --gpus 0 

With COiLTRAiNE you can also do simultaneous driving evaluation and validation
on some static dataset. Assuming that you set the CARLA_PATH, 
to add evaluation on a CARLA scenario and also some evaluation in
a sample dataset run:
     
    python3 coiltraine.py --folder sample --gpus 0 -de CorlTraining_Town01 -vd CoILVal1

Where the CorlTraining is a driving scenario on Town01, defined as one of the classes on the
drive/suites folder.  
Also note that the training dataset must be set on the [experiment configuration file](docs/configuration.md) directly,
since training data is strictly associated with the experiment.

The validation datasets are passed as parameter with -vd  and should be placed 
at the COIL_DATASET_PATH folder.

Finally, note the execution of the driving scenario on CARLA can also be done [using docker](docs/main_modules.md/#drive),
 option which we recommend.




#### Conditional Models Zoo

* Conditional Imitation Learning
* Conditional Imitation Learning CARLA paper
* [On Offline Evaluation of Vision-based Driving Models](docs/on_offline_evaluation.md)
* New One (Soon)





