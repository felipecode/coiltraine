import os

import json
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from collections import deque

from curriculum_core import execute_train, execute_validation, MODEL_TYPE, MODEL_CONFIGURATION
from utils.general import create_log_folder, create_exp_path, erase_logs, fix_driving_environments,\
                          erase_wrong_plotting_summaries, erase_validations
from configs import g_conf
from input import CoILDataset, Augmenter, splitter
from network import CoILModel

from es import OpenES, CMAES
from visualization import plot_scatter

# You could send the module to be executed and they could have the same interface.
# GPUS_P100 = "0001111222233334444555566667777"
GPUS_P100 = "01234567" * 6


def number_alive_process(process_deque):
    isalive = [p.is_alive() for p in process_deque]
    return sum(isalive)


def softmax(x):
   e_x = np.exp(x - np.max(x))
   return e_x / e_x.sum()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--gpus',
        nargs='+',
        dest='gpus',
        default=GPUS_P100,
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
    argparser.add_argument(
        '-mp', '--max_process',
        type = int,
        dest='max_process',
        default=16,
        help='Max number of parallel processes to train or validate'
    )
    args = argparser.parse_args()

    gpus = []
    for gpu in args.gpus:
        try:
            gpus.append(int(gpu))
        except:
            raise ValueError(" Gpu is not a valid int number")

    # create sampler
    # print(g_conf.TRAIN_DATASET_NAME)
    # full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], "80HoursW1-3-6-8")
    # augmenter = Augmenter(g_conf.AUGMENTATION)
    # dataset = CoILDataset(full_dataset, transform=augmenter)
    # keys = splitter.full_split(dataset)
    # np.save('full_split_keys', keys)
    keys = np.load('full_split_keys.npy')
    keys = [k['keys'] for k in keys]

    # define checkpoint list
    n_agents = len(gpus)
    checkpoints = []
    for n in range(n_agents):
        c = os.path.join("_curriculum_checkpoints", "checkpoint_{}.pth".format(n))
        checkpoints.append(c)

    # initialize checkpoints and warm up weights
    w = -5*np.ones(len(keys))
    w[-1] = 1
    model = CoILModel(MODEL_TYPE, MODEL_CONFIGURATION)
    state = {'iteration': 0, 'state_dict': model.state_dict(), 'total_time': 0}
    torch.save(state, checkpoints[0])
    p = execute_train(softmax(w), keys, 0, checkpoints[0], 0, n_batches=5000)
    this_c = 0
    while p.is_alive():
        print(" "*100, end="\r")
        print("waiting for warmup" + "." * this_c, end="\r")
        this_c = (this_c + 1) % 4
        time.sleep(.5)
    # sync checkpoints
    for c in checkpoints[1:]:
        os.system("cp {} {}".format(checkpoints[0], c))

    # es = OpenES(len(keys), popsize=len(gpus), weight_decay=0, rank_fitness=False, forget_best=False)
    es = CMAES(len(keys), popsize=len(gpus), weight_decay=0, initial_w=softmax(w))

    best_reward = -1000000
    # ask, tell, update loop
    process_deque = deque(maxlen=args.max_process)
    for cc in tqdm(range(1000)):
        weights = es.ask()
        for w, g, c in zip(weights, gpus, checkpoints):
            while number_alive_process(process_deque) >= args.max_process:
                # wait
                time.sleep(1)
            print("Launching iteration {} at GPU {}: {}".format(cc, g, c))
            p = execute_train(softmax(w), keys, cc, c, g)
            time.sleep(.5)
            while not p.is_alive():
                print("Launching iteration {} at GPU {}: {}".format(cc, g, c))
                p = execute_train(softmax(w), keys, cc, c, g)
                time.sleep(1)
            process_deque.append(p)

        this_c = 0
        cfiles = glob('./_curriculum_checkpoints/*.pth')
        while len(cfiles) != len(checkpoints) and number_alive_process(process_deque) != 0:  # waiting for training to fully sync before trying to valid
            cfiles = glob('./_curriculum_checkpoints/*.pth')
            time.sleep(1)
            print(" "*100, end="\r")
            print("{}/{} waiting for checkpoints".format(len(cfiles), len(checkpoints)) + "." * this_c, end="\r")
            this_c = (this_c + 1) % 4

        print()
        os.system("rm _validation_results/*")  # delete old results
        for g, c in zip(gpus, checkpoints):
            while number_alive_process(process_deque) >= args.max_process:
                # wait
                time.sleep(1)
            print("Validating checkpoint {} at GPU {}".format(c, g))
            p = execute_validation(c, '_validation_results/{}.json'.format(c.split('/')[1]), g)
            while not p.is_alive():
                print("Validating checkpoint {} at GPU {}".format(c, g))
                p = execute_validation(c, '_validation_results/{}.json'.format(c.split('/')[1]), g)
                time.sleep(1)
            process_deque.append(p)

        this_c = 0
        jfiles = glob('./_validation_results/*.json')
        while len(jfiles) != len(checkpoints) and number_alive_process(process_deque) != 0:
            jfiles = glob('./_validation_results/*.json')
            time.sleep(1)
            print(" "*100, end="\r")
            print("{}/{} waiting for validation results".format(len(jfiles), len(checkpoints)) + "." * this_c, end="\r")
            this_c = (this_c + 1) % 4

        rewards = []
        for j in jfiles:
            with open(j) as J:
                JJ = json.load(J)
                rewards.append(-JJ['avg_error'])
        rewards = np.asarray(rewards)
        # rewards = (rewards - rewards.mean())/rewards.std()

        this_best, best_idx = np.max(rewards), np.argmax(rewards)
        es.tell(rewards)
        print("Fit: ", es.result()[1], "rewards min, max", rewards.min(), rewards.max())
        if this_best > best_reward:
            best_reward = this_best
            print("FOUND NEW BEST CHECKPOINT ON ITER {}!!!".format(cc))
            os.system("cp {} _curriculum_checkpoints/checkpoint.best".format(checkpoints[best_idx]))
            np.save("_evolved_weights/{}_weights".format(cc), es.result())
