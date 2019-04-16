Exploring the Limitations of Behavior Cloning for Autonomous Driving
====================================================================

### Reproducing the results

#### Downloading Checkpoints

The configs/nocrash folder has some of the basic models that can be found on the paper.
Soon we will post how to reproduce the full results, including the plots. 

The following script will download
all the needed files.

    python3 tools/download_nocrash_models.py
    

#### Models present

 * *resnet34imnet10S1*: is the model with the random seed 1 from Figure 6 and it is also our best model
 * *resnet34imnet10S2*: is the model with the random seed 2 from Figure 6
 * *resnetmodel without the speed prediction and ten hours of training (Yellow model Fig. 5)
 * 




#### View our best model driving 




If you use any of our baselines please cite the paper