import re
import os
from PIL import Image

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


# TODO: there should be a more natural way to do that
def command_number_to_index(command_vector):

    return command_vector-2


def plot_test_image(image, name):

    os.makedirs(name)

    image_to_plot = Image.fromarray(image)
    image_to_plot.save(name)





