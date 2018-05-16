from __future__ import unicode_literals

import json
import logging
import os

from .json_formatter import filelogger
from .tensorboard_logger import Logger


g_logger = filelogger('None')

# We keep the file names saved here in the glogger to avoid including global

EXPERIMENT_NAME = ''
EXPERIMENT_BATCH_NAME = ''
PROCESS_NAME = ''
LOG_FREQUENCY = 1
tl = ''
# This next bit is to ensure the script runs unchanged on 2.x and 3.x





#logging.info(SM('message 1', set_value={1, 2, 3}, snowman='\u2603'))

def create_log(exp_batch_name, exp_name, process_name, log_frequency=1):

    """

    Arguments
        exp_batch_name: The name of the experiments folder
        exp_name: the name of the current folder that is being used.
        process_name: The name of the process, if it is some kind of evaluation or training or test.
    """

    global g_logger
    global EXPERIMENT_BATCH_NAME
    global EXPERIMENT_NAME
    global PROCESS_NAME
    global LOG_FREQUENCY
    global tl

    root_path = "_logs"
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    if not os.path.exists(os.path.join(root_path, exp_batch_name)):
        os.mkdir(os.path.join(root_path, exp_batch_name))

    if not os.path.exists(os.path.join(root_path, exp_batch_name, exp_name)):
        os.mkdir(os.path.join(root_path, exp_batch_name, exp_name))

    dir_name = os.path.join(root_path, exp_batch_name, exp_name)
    print ("logger dir name ", dir_name)
    full_name = os.path.join(dir_name, process_name)

    flog = filelogger(exp_name + '_' + process_name , [], full_name)



    # TODO: This needs to be updated after a while. ???
    g_logger = flog
    EXPERIMENT_BATCH_NAME = exp_batch_name
    EXPERIMENT_NAME = exp_name
    PROCESS_NAME = process_name
    LOG_FREQUENCY = log_frequency
    tl = Logger(os.path.join(root_path, exp_batch_name, exp_name, 'tensorboard_logs_'+process_name))


# TODO: iteration is required ! Assert that
def add_message(phase, message, iteration=None):
    """
    For the normal case
    Args:
        phase: The phase this message corresponds
        message: The dictionary with the message

    Returns:


    """


    if phase == 'Iterating' and iteration is None:
        raise ValueError(" Iterating messages should have the iteration/checkpoint.")

    if iteration is not None:
        if iteration % LOG_FREQUENCY == 0:

            g_logger.info({phase: message})

    else:
        g_logger.info({phase: message})

    # What if it is an error message ?
    # We can monitor the status based on error message. An error should mean the exp is not working





# TODO: the logger should also interface with tensorboard.

# TODO: Maybe an add scalar, message ??


def write_on_csv(checkpoint_name, output):
    """
    We also create the posibility to write on a csv file. So it is faster to load
    and check. Just using this to write the network outputs
    Args
        checkpoint_name: the name of the checkpoint being writen
        output: what is being written on the file


    Returns:

    """
    root_path = "_logs"

    full_path_name = os.path.join(root_path, EXPERIMENT_BATCH_NAME,
                                  EXPERIMENT_NAME, PROCESS_NAME + '_csv')

    file_name = os.path.join(full_path_name, str(checkpoint_name) + '.csv')

    #print (file_name)

    with open(file_name, 'a+') as f:
        f.write("%f" % output[0])
        for i in range(1, len(output)):
            f.write(',%f' % output[i])
        f.write("\n")









def add_scalar(tag, value, iteration=None):

    """
    For raw output  logging on tensorboard.


    """
    if iteration is not None:
        if iteration % LOG_FREQUENCY == 0:
            tl.scalar_summary(tag, value, iteration + 1)

    else:
        tl.scalar_summary(tag, value, 0)







def add_image(tag, images, iteration=None):
    # Add the image to a log, the monitorer is the module responsible by checking this
    # and eventually put some of the images to tensorboard.


    if iteration is not None:
        if iteration % LOG_FREQUENCY == 0:
            images = images.view(-1, 28, 28)[:10].cpu().numpy()
            tl.image_summary(tag, images, iteration + 1)


    else:
        images = images.view(-1, 28, 28)[:10].cpu().numpy()
        tl.image_summary(tag, images, iteration + 1)


