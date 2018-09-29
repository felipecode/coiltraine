Conditional Imitation Learning Training Framework
-------------

This code can be used to easily train and manage the trainings of imitation
learning networks, toguether with the CARLA simulator.



#### Getting started


##### RequiredPackages

Imgaug
h5py
pytorch 0.4


#### General System view

Diagram 


#### Getting started


Train/Validation mode:

The first thing you need to do is define the datasets folder.
This is the folder that contains your training and validation datasets.

    export COIL_DATASET_PATH=<path to the datasetfolders>


The training dataset must be set on the experiment file directly. 
Since training is strictly associated with the experiment.
The validation datasets are passed as parameter.
There are two modes for running.

Drive Mode:

For driving and testing in CARLA the path to the CARLA
folder must be specified.

    export CARLA_PATH=<pathtocarla/CARLA/


Running a folder

The system works by running a full folder that has a list of all the experiments.

    python3 --folder <folder_name>  --gpus < list of gpus > -vd < list of validation datasets >
     -de < List of driving environments >

To get all the eccv models and results you should run:
    


Running a simple process:

This mode is for basically just running a single process that can be
a training/ validation or a driving proccess.


Running plot:

To run the plotting run

    python3 run_plotting.py --folder eccv_debug -p plotting_all_cameras
    
Where the "folder" is the folder of the experiments. "-p" is the plotting configuration file
that is localized at the visulization/plotting_params    


