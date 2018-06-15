# ***** main loop *****
import argparse
import numpy as np
import h5py

import os

import random
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path viewer')
    # parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('-pt', '--path', default="")

    args = parser.parse_args()
    path = args.path

    first_time = True
    count = 0
    steering_pred = []
    steering_gt = []



    files = glob.glob(os.path.join(path, 'data_*.h5'))
    for f in files:

        try:
            data = h5py.File(f, "r+")
        except Exception as e:
            continue

        print (data['targets'][0][31])
        print (data['targets'][0][32])
        print (data['targets'][0][33])
        print (data['targets'][0][34])
