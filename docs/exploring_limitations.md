Exploring the Limitations of Behavior Cloning for Autonomous Driving
====================================================================


    

#### Models available

 * *resnet34imnet10S1*: is the model with the random seed 2 from Figure 6 and it is also our best model (Green Model Fig. 5).
 * *resnet34imnet10S2*: is the model with the random seed 1 from Figure 6.
 * *resnet34imnet10-nospeed*: without the speed prediction and ten hours of training (Yellow model Fig. 5).
 * *resnet34imnet100*: the model with 100 hours of demonstrations (Blue model Fig. 5).
 * *resnet34imnet100-nospeed*: the model with 100 hours of demonstrations and no speed prediction (Red model Fig. 5).
 
##### Downloading checkpoints

The configs/nocrash folder has some of the relevant models that can be found on the paper.

The following script will download
all the needed files.

    python3 tools/download_nocrash_models.py

#### Expert demonstrator

The expert demonstration present on the paper is available at this repository:
https://github.com/carla-simulator/data-collector

#### CARLA100 Dataset
Links for downloading.

http://datasets.cvc.uab.es/CVPR2019-CARLA100/CVPR2019-CARLA100_01.zip
http://datasets.cvc.uab.es/CVPR2019-CARLA100/CVPR2019-CARLA100_02.zip
http://datasets.cvc.uab.es/CVPR2019-CARLA100/CVPR2019-CARLA100_03.zip
http://datasets.cvc.uab.es/CVPR2019-CARLA100/CVPR2019-CARLA100_04.zip
http://datasets.cvc.uab.es/CVPR2019-CARLA100/CVPR2019-CARLA100_05.zip
http://datasets.cvc.uab.es/CVPR2019-CARLA100/CVPR2019-CARLA100_06.zip
http://datasets.cvc.uab.es/CVPR2019-CARLA100/CVPR2019-CARLA100_07.zip
http://datasets.cvc.uab.es/CVPR2019-CARLA100/CVPR2019-CARLA100_08.zip
http://datasets.cvc.uab.es/CVPR2019-CARLA100/CVPR2019-CARLA100_09.zip
http://datasets.cvc.uab.es/CVPR2019-CARLA100/CVPR2019-CARLA100_10.zip
http://datasets.cvc.uab.es/CVPR2019-CARLA100/CVPR2019-CARLA100_11.zip
http://datasets.cvc.uab.es/CVPR2019-CARLA100/CVPR2019-CARLA100_12.zip
http://datasets.cvc.uab.es/CVPR2019-CARLA100/CVPR2019-CARLA100_13.zip
http://datasets.cvc.uab.es/CVPR2019-CARLA100/CVPR2019-CARLA100_14.zip

You can also collect a similar dataset using the data collector.
https://github.com/carla-simulator/data-collector


#### NoCrash Benchmark

It can be find seperately on the repository above:


However it is also integrated with the coiltraine system.


#### Reproducing the baseline results

To reproduce all  of the available models, using the gpu 0, run:

    python3 coiltraine.py --gpus 0 --folder nocrash -de NocrashNewWeatherTown_Town02 NocrashNewWeather_Town01\
     NocrashTraining_Town01 NocrashNewTown_Town02 --docker carlagear
      

Note, the models use the CARLA single gear version of 0.8.4. This is discussed
on this repository:

https://github.com/carla-simulator/data-collector

To test our best model on the hardest condition: 

    python3 coiltraine.py --gpus 0 --single-process drive -e resnet34imnet10S2 --folder nocrash \
    -de NocrashNewWeatherTown_Town02 --docker carlagear


If you use any of our baselines or benchmark please cite our paper:

```
@article{codevilla2019exploring,
  title={Exploring the Limitations of Behavior Cloning for Autonomous Driving},
  author={Codevilla, Felipe and Santana, Eder and L{\'o}pez, Antonio M and Gaidon, Adrien},
  journal={International Conference on Computer Vision(ICCV)},
  year={2019}
}

```
