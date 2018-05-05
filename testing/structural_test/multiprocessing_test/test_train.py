import os
import time
import tensorflow as tf


def execute(gpu, exp_alias):

    print("Running ", __file__, " On GPU ", gpu, "of experiment name ", exp_alias)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


    print (os.environ["COIL_DATASET_PATH"])

    tf.Session()
    time.sleep(10)
