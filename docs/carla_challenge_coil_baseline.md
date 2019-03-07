CARLA Challenge Track 2 Baseline - Conditional Imitation Learning
============



[![CARLA Video](img/thumbnail.png)](https://youtu.be/fX1omU4IRwI)




CARLA Challenge Test Results
-----------------------------
We keep an updated score for the current challenge tasks:


| Challenge Basic:  | AVG Score 34.70 |
|-------------------|--------|



Running the Baseline
----------

### Preparation


Clone the repository:

    git clone https://github.com/felipecode/coiltraine.git 
    cd coiltraine

We provide a conda environment requirements file, to
install and activate, just run:

    conda env create -f requirements.yaml
    conda activate coiltraine

Download the agent pytorch checkpoint by running the following script:

    python3 tools/download_sample_models.py

The checkpoints should now be allocated already on the proper folders.
Make sure you set the PYTHONPATH with the CARLA egg and the Python API:

     export PYTHONPATH=<Path-To-CARLA-Latest>/PythonAPI/carla-0.9.3-py3.5-linux-x86_64.egg:<Path-To-CARLA-Latest>/PythonAPI:$PYTHONPATH
     

### Visualize the agent results 

First have the latest version of CARLA executing at some terminal at 40 fps (Recommend)

    sh CarlaUE4.sh Town03 -windowed -world-port=2000  -benchmark -fps=40
 

To run the and visualize the model run:

    python3 view_model.py  -f baselines -e resnet34imnet -cp 180000 -cv 0.9

After running, you will see on the botton corner the activations of resnet intermediate
layers. You can command a destination for the agent by using the arrow keys from the keyboard.


### Get the agent performance on the CARLA Challenge



Clone the scenario  runner repository:
    
    cd
    git clone https://github.com/carla-simulator/scenario_runner.git

Setup the scenario runner challenge repository by setting the path to your CARLA root
folder.

    cd scenario_runner
    bash setup_environment --carla-root <path_to_carla_root_folder>


Export the coiltraine path to the PYTHONPATH:

    cd ~/coitraine
    export PYTHONPATH=`pwd`:$PYTHONPATH


Execute the challenge with the conditional imitation learning baseline

    python3  srunner/challenge/challenge_evaluator.py --file --scenario=group:ChallengeBasic --agent=../coiltraine/drive/CoILBaseline.py --config ../coiltraine/drive/sample_agent.json


Watch the results.


Training
---------

Define the datasets folder.
This is the folder that will contain your training and validation datasets

    export COIL_DATASET_PATH=<Path to where your dataset folders are>


Download the dataset:

    python3 tools/get_baseline_dataset.py

You can learn how to use the framework on the following [main tutorial](../README.md)
However, you can also do a single train of the model  using the
basic dataset:

    python3 coiltraine.py --single-process train -e resnet34imnet --folder baselines --gpus 0

To check images and train curves there is also a tensorboard log
being saved at "_logs" folder on the repository root.









