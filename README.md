Conditional Imitation Learning Training Framework
-------------

This code can be used to easily train and manage the trainings of imitation
learning networks, toguether with the CARLA simulator.


#### General System view

The system has a few modules:
[![COIL Diagram](docs/img/CoIL.png?raw=true )]()

#### Installation

To install the CoIL large scale training framework,we provide a conda environment requirements file.
Basically just do:

    conda env create -f requirements.yml


#### Getting started


Assuming you collected and post-processed the data at 

~/CARLA/CARLA100 ---
    --/episode_00001
    --/episode_00002
    …


The first thing you need to do is define the datasets folder.
This is the folder that contains your training and validation datasets

    export COIL_DATASET_PATH=<Path to where your dataset folders are>
 
Experiments are defined in config files inside CoIL/configs. You can run all the experiments in a folder using:

    python3 run_CoIL.py --folder <my_folder> --gpus 0 1 -de <DrivingEnvironmentClass_Town0X> 

Where DrivingEnvironmentClass is one of the classes defined in the modules at CoIL/drive/suites. Those driving environments define the start and end positions for driving, the number of cars, people, etc. That information will define the Benchmark to test the model driving in CARLA in parallel to training. Town0X is either Town01 or Town02. 
Note that the training dataset must be set on the experiment file directly. Since training is strictly associated with the experiment. The validation datasets are passed as parameter. There are two modes of running.

To run a single experiment, we use the flag single-process train and the experiment name. A full example to train a ResNet_34 with attention losses is shown below:

    python3 run_CoIL.py --folder attention_tests --gpus 0 1 2 3 4 5 6 7 -de ExredTraining_Town01 --single-process train -e resnet_attention --docker




