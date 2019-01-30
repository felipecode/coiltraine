Sample Agents
============


Here we present some sample conditional imitation learning
agents driving on the CARLA towns.


Getting Started
-------------
### Preparation

First [Install the system](../README.md/#installation) and set the environment.

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

    python3 view_model.py  -f town03 -e resnet34imnet -cp 100000 -cv 0.9

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

Comming Soon

