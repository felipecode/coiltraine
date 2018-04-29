from __future__ import unicode_literals

import json
import logging
import os

from configs import g_conf

# This next bit is to ensure the script runs unchanged on 2.x and 3.x
try:
    unicode
except NameError:
    unicode = str

class Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, set):
            return tuple(o)
        elif isinstance(o, unicode):
            return o.encode('unicode_escape').decode('ascii')
        return super(Encoder, self).default(o)

class StructuredMessage(object):
    def __init__(self, message, **kwargs):
        self.message = message
        self.kwargs = kwargs

    def __str__(self):
        s = Encoder().encode(self.kwargs)
        return '%s >>> %s' % (self.message, s)




#logging.info(SM('message 1', set_value={1, 2, 3}, snowman='\u2603'))

#TODO: eventually create a wrapper for the logging.

def create_log(exp_batch_name, exp_name, process_name):
    """

    Arguments
        exp_batch_name: The name of the experiments folder
        exp_name: the name of the current folder that is being used.
        process_name: The name of the process, if it is some kind of evaluation or training or test.
    """


    fh = logging.FileHandler(os.path.join(exp_batch_name,exp_name,process_name))



    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def add_message(module, message):

    # What if it is an error message ?
    # We can monitor the status based on error message. An error should mean the exp is not working

    pass

# TODO: the logger should also interface with tensorboard.


def add_image(some_image):
    # Add the image to a log, the monitorer is the module responsible by checking this
    # and eventually put some of the images to tensorboard.
    pass