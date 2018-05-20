Conditional Imitation Learning Training Framework
-------------

This code can be used to easily train and manage the trainings of imitation
learning networks, toguether with the CARLA simulator.



####Getting started


##### RequiredPackages

Imgaug
h5py
pytorch 0.4


#### General System view

The system has a few modules

#### Getting started


Train/Validation mode:

The first thing you need to do is define the datasets folder.
This is the folder that contains your training and validation datasets.

    export COIL_DATASET_PATH=<path to the datasetfolders>


The training dataset must be set on the experiment file directly. 
Since training is strictly associated with the experiment.
The validation datasets are passed as parameter.
There are two modes of running.

Running a simple process:
This mode is for basically just running a single process that can be
a training/ validation or drive proccess.

Running a folder

    python3 --folder <folder_name>  --gpus < list of gpus > -vd < list of validation datasets >
     -de < List of driving environments >


Drive Mode:

For driving and testing in CARLA the path to the CARLA
folder must be specified.

    export CARLA_PATH=<pathtocarla/CARLA/
    
