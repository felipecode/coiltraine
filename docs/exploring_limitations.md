Exploring the Limitations of Behavior Cloning for Autonomous Driving
====================================================================


#### Downloading Checkpoints

The configs/nocrash folder has some of the relevant models that can be found on the paper.

The following script will download
all the needed files.

    python3 tools/download_nocrash_models.py
    

#### Models available

 * *resnet34imnet10S1*: is the model with the random seed 1 from Figure 6 .
 * *resnet34imnet10S2*: is the model with the random seed 2 from Figure 6 and it is also our best model (Green Model Fig. 5).
 * *resnet34imnet10-nospeed*: without the speed prediction and ten hours of training (Yellow model Fig. 5).
 * *resnet34imnet100*: the model with 100 hours of demonstrations (Blue model Fig. 5).
 * *resnet34imnet100-nospeed*: the model with 100 hours of demonstrations and no speed prediction (Red model Fig. 5).



#### Reproducing the results

To reproduce all  of the available models, using the gpu 0, run:

    python3 coiltraine.py --gpus 0 --folder nocrash -de NocrashNewWeatherTown_Town02 NocrashNewWeather_Town01\
     NocrashTraining_Town01 NocrashNewTown_Town02 --docker carlagear
      

Note, the models use the CARLA single gear version of 0.8.4. This is discussed
on this repository:

https://github.com/carla-simulator/data-collector

To test our best model on the hardest condition: 

    python3 coiltraine.py --gpus 0 --single-process drive -e resnet34imnet10S2 --folder nocrash \
    -de NocrashNewWeatherTown_Town02 --docker carlagear



If you use any of our baselines please cite our paper:

*Add Soon*
