from __future__ import unicode_literals

import json
import logging
import os

from .json_formatter import filelogger

g_logger = filelogger('None')


# This next bit is to ensure the script runs unchanged on 2.x and 3.x





#logging.info(SM('message 1', set_value={1, 2, 3}, snowman='\u2603'))

def create_log(exp_batch_name, exp_name, process_name):
    global g_logger
    """

    Arguments
        exp_batch_name: The name of the experiments folder
        exp_name: the name of the current folder that is being used.
        process_name: The name of the process, if it is some kind of evaluation or training or test.
    """


    #fh = logging.FileHandler(os.path.join(exp_batch_name,exp_name,process_name))

    root_path = "_logs"
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    if not os.path.exists(os.path.join(root_path, exp_batch_name)):
        os.mkdir(os.path.join(root_path, exp_batch_name))

    if not os.path.exists(os.path.join(root_path, exp_batch_name, exp_name)):
        os.mkdir(os.path.join(root_path, exp_batch_name, exp_name))

    dir_name = os.path.join(root_path, exp_batch_name, exp_name)

    full_name = os.path.join(dir_name, process_name)

    flog = filelogger(exp_name + '_' + process_name , [], full_name)

    # TODO: This needs to be updated after a while.
    g_logger = flog


def add_message(phase, message):
    """
    For the normal case
    Args:
        phase: The phase this message corresponds
        message: The dictionary with the message

    Returns:

    """

    if phase != 'Loading' and 'Iteration' not in message:
        raise ValueError(" Non loading messages should have the iteration.")

    # What if it is an error message ?
    # We can monitor the status based on error message. An error should mean the exp is not working
    g_logger.info({phase: message})

    pass

# TODO: the logger should also interface with tensorboard.

# TODO: Maybe an add scalar, message ??

#def add_scalar() ???


def add_image(some_image):
    # Add the image to a log, the monitorer is the module responsible by checking this
    # and eventually put some of the images to tensorboard.
    pass



