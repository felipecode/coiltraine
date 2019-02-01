Sample Agents
============


Here we present some sample conditional imitation learning
agents driving on the CARLA towns.


Getting Started
-------------
### Preparation

First [install the system](../README.md/#installation) and set the environment.

You can download the pytorch checkpoints by running the following script:

    python3 tools/download_sample_models.py

The checkpoints should now be allocated already on the proper folders.

Make sure you set the python path for the CARLA egg:

     export PYTHONPATH=<Path-To-CARLA-93>/PythonAPI/carla-0.9.3-py3.5-linux-x86_64.egg


### Town03 Agent

This agent is specialized on the Town03.

#### Inference

First have a CARLA 0.93 executing at some terminal at 40 fps (Recommend)

    sh CarlaUE4.sh Town03 -windowed -world-port=2000  -benchmark -fps=40
 

To run the and visualize the model run:

    python3 view_model.py  -f town03 -e resnet34imnet -cp 200000 -cv 0.9

After running, you will see on the botton corner the activations of resnet intermediate
layers. You can command a destination for the agent by using the arrow keys from the keyboard.


#### Training


Download the dataset. Make sure you have set the COIL_DATASET_PATH variable before:

    python3 tools/get_town03_dataset.py

You can learn how to use the framework on the following [main tutorial](../README.md)
However you can also do a single train of the model for town03 using the
town03 dataset:

    python3 coiltraine.py --single-process train -e resnet34imnet4 --folder town03 --gpus 0

To check images and train curves there is also a tensorboard log
being saved at "_logs" folder on the repository root.



###  Town01/2 Agent

This agent is much more powerfull since it had much more
time for development, but if follows the same principle as the
agent for town03. Town03 is more complicated and probably
require a different set of high level commands.

#### Inference

First have a CARLA 0.93 executing at some terminal at 40 fps (Recommend)

    sh CarlaUE4.sh Town01 -windowed -world-port=2000  -benchmark -fps=40
 
To run the and visualize the model run:

    python3 view_model.py  -f nocrash -e resnet34imnet10 -cp 320000 -cv 0.9

### Adding Vehicles and Weather Changes

If you want to make the life of the agents harder, you can use the pythonAPI sample
scripts to add dynamic objects and changes on the weather conditions.

In another terminal change the directory to where CARLA 0.9.3 is:

    cd <Path-To-CARLA-93>/

Then to add 40 vehicles for instance run:

    python3 spawn_npc.py -n 40
    
To add dynamic weather change:

    python3 dynamic_weather.py
    







