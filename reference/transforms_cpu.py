

import numpy as np
import numbers
import six
import six.moves as sm
import torch


from input.augmenter.transforms import BaseTransform

from input.augmenter import functional as F
from input.augmenter import randomizer
from utils import checking as ch




class MultiplyCPU(BaseTransform):
    """
    Multiply all pixels in an image with a specific value. Adapted from IMAUG
    This augmenter can be used to make images lighter or darker.
    Function adapted to perform on CPU

    Parameters
    ----------
    mul : float or tuple of two floats or StochasticParameter, optional(default=1.0)
        The value with which to multiply the pixel values in each
        image.
            * If a float, then that value will always be used.
            * If a tuple (a, b), then a value from the range a <= x <= b will
              be sampled per image and used for all pixels.


    per_channel : bool or float, optional(default=False)
        Whether to use the same multiplier per pixel for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Multiply(2.0)

    would multiply all images by a factor of 2, making the images
    significantly brighter.

    >>> aug = iaa.Multiply((0.5, 1.5))

    would multiply images by a random value from the range 0.5 <= x <= 1.5,
    making some images darker and others brighter.

    """

    def __init__(self, mul=1.0, per_channel=False, name=None,
                 deterministic=False, random_state=None):
        super(MultiplyCPU, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if isinstance(mul, numbers.Real) or isinstance(mul, numbers.Integral):
            ch.do_assert(mul >= 0.0, "Expected multiplier to have range [0, inf), got value %.4f." % (mul,))
            self.mul = randomizer.Deterministic(mul)
        else:
            ch.do_assert(len(mul) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(mul),))
            self.mul = randomizer.Uniform(mul[0], mul[1])

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = randomizer.Deterministic(int(per_channel))
        else:
            ch.do_assert(0 <= per_channel <= 1.0)
            self.per_channel = randomizer.Binomial(per_channel)

    def __call__(self, images):

        result = images
        nb_images = len(images)
        seeds = randomizer.current_random_state().randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i]
            rs_image = randomizer.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                nb_channels = image.shape[2]
                samples = self.mul.draw_samples((nb_channels,), random_state=rs_image)
                for c, sample in enumerate(samples):
                    ch.do_assert(sample >= 0)

                    image = F.multiply(image[c, ...], torch.ones(1) * float(sample))
            else:
                sample = self.mul.draw_sample(random_state=rs_image)
                ch.do_assert(sample >= 0)
                image = F.multiply(image, torch.ones(1) * float(sample))


            #image = meta.clip_augmented_image_(image, 0, 255) # TODO make value range more flexible
            #image = meta.restore_augmented_image_dtype_(image, input_dtypes[i])

            result[i] = image

        return result

    def get_parameters(self):
        return [self.mul]




class AddCPU(BaseTransform):
    """
    Add a value to all pixels in an image.

    Parameters
    ----------
    value : int or iterable of two ints or StochasticParameter, optional(default=0)
        Value to add to all
        pixels.
            * If an int, then that value will be used for all images.
            * If a tuple (a, b), then a value from the discrete range [a .. b]
              will be used.
            * If a StochasticParameter, then a value will be sampled per image
              from that parameter.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Add(10)

    always adds a value of 10 to all pixels in the image.

    >>> aug = iaa.Add((-10, 10))

    adds a value from the discrete range [-10 .. 10] to all pixels of
    the input images. The exact value is sampled per image.

    >>> aug = iaa.Add((-10, 10), per_channel=True)

    adds a value from the discrete range [-10 .. 10] to all pixels of
    the input images. The exact value is sampled per image AND channel,
    i.e. to a red-channel it might add 5 while subtracting 7 from the
    blue channel of the same image.

    >>> aug = iaa.Add((-10, 10), per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """

    def __init__(self, value=0, per_channel=False, name=None,
                 deterministic=False, random_state=None):
        super(AddCPU, self).__init__(name=name, deterministic=deterministic, random_state=random_state)


        #elif ia.is_iterable(value):
        #    ia.do_assert(len(value) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(value),))

        if isinstance(value, numbers.Integral):
            ch.do_assert(-255 <= value <= 255, "Expected value to have range [-255, 255], got value %d." % (value,))
            self.value = randomizer.Deterministic(value)
        else:
            self.value = randomizer.DiscreteUniform(value[0], value[1])
        #elif isinstance(value, StochasticParameter):
        #    self.value = value
        #else:
        #    raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(value),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = randomizer.Deterministic(int(per_channel))
        else:# ia.is_single_number(per_channel):
            #ia.do_assert(0 <= per_channel <= 1.0, "Expected bool, or number in range [0, 1.0] for per_channel, got %s." % (type(per_channel),))
            self.per_channel = randomizer.Binomial(per_channel)

        #else:
        #    raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

    def __call__(self, images):
        #input_dtypes = copy_dtypes_for_restore(images, force_list=True)

        result = images
        nb_images = len(images)
        random_seeds = randomizer.current_random_state().randint(0, 10 ** 6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i]
            rs_image = randomizer.new_random_state(random_seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                nb_channels = image.shape[2]
                samples = self.value.draw_samples((nb_channels,), random_state=rs_image)
                for c, sample in enumerate(samples):
                    # TODO make value range more flexible

                    ch.do_assert(-255 <= sample <= 255)

                    image = F.add(image[c, ...], torch.ones(1) * float(sample))
            else:
                sample = self.value.draw_sample(random_state=rs_image)
                ch.do_assert(-255 <= sample <= 255) # TODO make value range more flexible

                image = F.add(image, torch.ones(1) * float(sample))

            #image = meta.clip_augmented_image_(image, 0, 255) # TODO make value range more flexible
            #image = meta.restore_augmented_image_dtype_(image, input_dtypes[i])

            result[i] = image

        return result

    def get_parameters(self):
        return [self.value]

