On Offline Evaluation of Vision-based Driving Models
====================================================


The evaluation proposed on the paper comprises
using a a validation dataset toguether with some benchmark on
CARLA. Considering that the validation
dataset and the benchmark were computed on a similar
scenario, we compute the correlation between driving and prediction
on some dataset.

### Reproducing the results



#### Download the Models

First thing is to download all the 64 trained models.
For that we provide a script that makes the download and store
it on the appropriete folder. Run:

    python3 tools/get_offline_online_data.py

#### Plotting

To test computing the scatter plots for a single experiment,
you should run:

    python3 run_plotting.py -p sample_plot -e experiment_64


The folder _logs/eccv/plots/sample_plots should show
the following plot:

[ Figure]

Note, for a single experiment, containing 12 models from
different checkpoints, the correlation is usually high.

To compute all the plots:

    python3 run_plotting.py -p eccv_online_offline_plots

A loading bar should show up. It takes aproximatelly 4 hours
to complete the plotting process.

#### Re-Training
To re run the trainings:


Note: there are non determinism on the training and evaluation, the
plots when retraining all the models will not be the same.




Paper
-----

If you use this evaluation methodology, please cite our ECCV’18 paper.

_On Offline Evaluation of Vision-based Driving Models_<br>Felipe Codevilla,
 Antonio M. López, Vladlen Koltun, Alexey Dosovitskiy;
[[PDF](https://arxiv.org/pdf/1809.04843.pdf)]


```
@inproceedings{codevilla2018offline,
  title={On Offline Evaluation of Vision-Based Driving Models},
  author={Codevilla, Felipe and L{\'o}pez, Antonio M and Koltun, Vladlen and Dosovitskiy, Alexey},
  booktitle={European Conference on Computer Vision},
  pages={246--262},
  year={2018},
  organization={Springer}
}

```
