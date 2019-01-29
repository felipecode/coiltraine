Sample Agents
============


Here we show some sample conditional imitation learning
agents driving on the CARLA towns.


Getting Started
-------------


[Install the system](../README.md/#installation) and set the environment.

You can download the pytorch checkpoints by running the following script:

    python3 tools/download_sample_models.py

They should now be already allocated into the proper folder.

Make sure you have the python path for the CARLA egg set

export the python path


### Town03 Agent


#### Inference
 
    
 


##### Executing


First have a CARLA 0.92 executing at some terminal at 40 fps (Recommend)


    sh CarlaUE4.sh Town03 -windowed -world-port=2000  -benchmark -fps=40
 

Assuming


To run the and visualize the model run:

    python3 view_model.py  -f town03 -e resnet34imnet -cp 20000 -cv 0.9

Visualize

To train use this data ()



### Training


Download the dataset





Town02/1 Agent
----------------

Download
Visualize.


#### Training


Data available soon !



Town03 fine tunned.
---------------