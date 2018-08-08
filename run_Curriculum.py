import os
import json
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
from glob import glob

from curriculum_core import execute_train, execute_validation, MODEL_TYPE, MODEL_CONFIGURATION
from utils.general import create_log_folder, create_exp_path, erase_logs, fix_driving_environments,\
                          erase_wrong_plotting_summaries, erase_validations
from configs import g_conf
from input import CoILDataset, Augmenter, splitter
from network import CoILModel

from es import OpenES
from visualization import plot_scatter

# You could send the module to be executed and they could have the same interface.


def softmax(x):
   e_x = np.exp(x - np.max(x))
   return e_x / e_x.sum()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--gpus',
        nargs='+',
        dest='gpus',
        type=str
    )
    argparser.add_argument(
        '-f',
        '--folder',
        type=str
    )
    argparser.add_argument(
        '-e',
        '--exp',
        type=str
    )
    argparser.add_argument(
        '-vd',
        '--val-datasets',
        dest='validation_datasets',
        nargs='+',
        default=[]
    )
    argparser.add_argument(
        '--no-train',
        dest='is_training',
        action='store_false'
    )
    argparser.add_argument(
        '-de',
        '--drive-envs',
        dest='driving_environments',
        nargs='+',
        default=[]
    )
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')

    argparser.add_argument(
        '-ns', '--no-screen',
        action='store_true',
        dest='no_screen',
        help='Set to carla to run offscreen'
    )
    argparser.add_argument(
        '-ebv', '--erase-bad-validations',
        action='store_true',
        dest='erase_bad_validations',
        help='Set to carla to run offscreen'
    )
    # TODO: add the posibility to delete a subset
    argparser.add_argument(
        '-rv', '--restart-validations',
        action='store_true',
        dest='restart_validations',
        help='Set to carla to run offscreen'
    )
    argparser.add_argument(
        '-gv',
        '--gpu-value',
        dest='gpu_value',
        type=float,
        default=4.0
    )
    argparser.add_argument(
        '-dk', '--docker',
        type = str,
        dest='docker',
        default=None,
        help='Set to run carla using docker'
    )
    args = argparser.parse_args()

    gpus = []
    for gpu in args.gpus:
        try:
            gpus.append(int(gpu))
        except:
            raise ValueError(" Gpu is not a valid int number")

    # create sampler
    print(g_conf.TRAIN_DATASET_NAME)
    full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME)
    augmenter = Augmenter(g_conf.AUGMENTATION)
    dataset = CoILDataset(full_dataset, transform=augmenter)
    keys = splitter.full_split(dataset)

    # define checkpoint list
    n_agents = len(gpus)
    checkpoints = []
    for n in range(n_agents):
        c = os.path.join("_curriculum_checkpoints", "checkpoint_{}.pth".format(n))
        checkpoints.append(c)

    # initialize checkpoints
    model = CoILModel(MODEL_TYPE, MODEL_CONFIGURATION)
    state = {'iteration': 0, 'state_dict': model.state_dict(), 'total_time': 0}
    for c in checkpoints:
        torch.save(state, c)

    es = OpenES(len(keys), popsize=len(gpus), weight_decay=0, rank_fitness=False, forget_best=False)

    # ask, tell, update loop
    for cc in tqdm(range(2)):
        weights = es.ask()
        for w, g, c in zip(weights, gpus, checkpoints):
            print("Launching iteration {} at GPU {}: {}".format(cc, g, c))
            execute_train(softmax(w), keys, cc, c, g)

        this_c = 0
        cfiles = glob('./_curriculum_checkpoints/*.pth')
        while len(cfiles) != len(checkpoints):
            cfiles = glob('./_curriculum_checkpoints/*.pth')
            # time.sleep(1)
            print("waiting for checkpoints" + "." * this_c, end="\r")
            this_c = (this_c + 1) % 4

        for g, c in zip(gpus, checkpoints):
            print("Validating checkpoint {} at GPU {}".format(c, g))
            execute_validation(c, '_validation_results/{}.json'.format(c.split('/')[1]), g)

        this_c = 0
        jfiles = glob('./_validation_results/*.json')
        while len(jfiles) != len(checkpoints):
            jfiles = glob('./_validation_results/*.json')
            print(len(jfiles), len(checkpoints))
            # time.sleep(1)
            print("waiting for validation results" + "." * this_c, end="\r")
            this_c = (this_c + 1) % 4

        rewards = []
        for j in jfiles:
            with open(j) as J:
                JJ = json.load(J)
                rewards.append(-JJ['avg_error'])

        es.tell(rewards)
        print("Fit: ", es.result()[1])

        # Get best model and sync
        idx = np.argmax(rewards)
        best_ckpt = checkpoints[idx]
        for c in checkpoints:
            os.system("cp {} {}".format(best_ckpt, c))

    np.save("best_weights", es.result()[0])
