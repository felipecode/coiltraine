
import numpy as np
import numbers



import torch
from torchvision import transforms

import six

import six.moves as sm

from input.augmenter import functional as F
from abc import ABCMeta

from . import randomizer

def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def copy_dtypes_for_restore(images, force_list=False):
    if ia.is_np_array(images):
        if force_list:
            return [images.dtype for _ in sm.xrange(len(images))]
        else:
            return images.dtype
    else:
        return [image.dtype for image in images]


def do_assert(condition, message="Assertion failed."):
    """
    Function that behaves equally to an `assert` statement, but raises an
    Exception.

    This is added because `assert` statements are removed in optimized code.
    It replaces `assert` statements throughout the library that should be
    kept even in optimized code.

    Parameters
    ----------
    condition : bool
        If False, an exception is raised.

    message : string, optional(default="Assertion failed.")
        Error message.

    """
    if not condition:
        raise AssertionError(str(message))




#TODO :

@six.add_metaclass(ABCMeta)
class BaseTransform(object): # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Base class for Augmenter objects.
    All augmenters derive from this class.
    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        """
        Create a new Augmenter instance.

        Parameters
        ----------
        name : None or string, optional(default=None)
            Name given to an Augmenter object. This name is used in print()
            statements as well as find and remove functions.
            If None, `UnnamedX` will be used as the name, where X is the
            Augmenter's class name.

        deterministic : bool, optional(default=False)
            Whether the augmenter instance's random state will be saved before
            augmenting images and then reset to that saved state after an
            augmentation (of multiple images/keypoints) is finished.
            I.e. if set to True, each batch of images will be augmented in the
            same way (e.g. first image might always be flipped horizontally,
            second image will never be flipped etc.).
            This is useful when you want to transform multiple batches of images
            in the same way, or when you want to augment images and keypoints
            on these images.
            Usually, there is no need to set this variable by hand. Instead,
            instantiate the augmenter with the defaults and then use
            `augmenter.to_deterministic()`.

        random_state : None or int or np.random.RandomState, optional(default=None)
            The random state to use for this
            augmenter.
                * If int, a new np.random.RandomState will be created using this
                  value as the seed.
                * If np.random.RandomState instance, the instance will be used directly.
                * If None, imgaug's default RandomState will be used, which's state can
                  be controlled using imgaug.seed(int).
            Usually there is no need to set this variable by hand. Instead,
            instantiate the augmenter with the defaults and then use
            `augmenter.to_deterministic()`.

        """
        super(BaseTransform, self).__init__()

        if name is None:
            self.name = "Unnamed%s" % (self.__class__.__name__,)
        else:
            self.name = name

        self.deterministic = deterministic

        if random_state is None:
            if self.deterministic:
                self.random_state = randomizer.new_random_state()
            else:
                self.random_state = randomizer.current_random_state()
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        self.activated = True



class Add(BaseTransform):
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
        super(Add, self).__init__(name=name, deterministic=deterministic, random_state=random_state)


        #elif ia.is_iterable(value):
        #    ia.do_assert(len(value) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(value),))

        if isinstance(value, numbers.Integral):
            do_assert(-255 <= value <= 255, "Expected value to have range [-255, 255], got value %d." % (value,))
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

                    do_assert(-255 <= sample <= 255)


                    F.add(image[c, ...].cuda(), torch.ones(1).cuda() * float(sample))
            else:
                sample = self.value.draw_sample(random_state=rs_image)
                do_assert(-255 <= sample <= 255) # TODO make value range more flexible

                F.add(image.cuda(), torch.ones(1).cuda() * float(sample))

            #image = meta.clip_augmented_image_(image, 0, 255) # TODO make value range more flexible
            #image = meta.restore_augmented_image_dtype_(image, input_dtypes[i])

            result[i] = image

        return result

    def get_parameters(self):
        return [self.value]





class Multiply(BaseTransform):
    """
    Multiply all pixels in an image with a specific value.

    This augmenter can be used to make images lighter or darker.

    Parameters
    ----------
    mul : float or tuple of two floats or StochasticParameter, optional(default=1.0)
        The value with which to multiply the pixel values in each
        image.
            * If a float, then that value will always be used.
            * If a tuple (a, b), then a value from the range a <= x <= b will
              be sampled per image and used for all pixels.
            * If a StochasticParameter, then that parameter will be used to
              sample a new value per image.

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
        super(Multiply, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if isinstance(mul, numbers.Real) or isinstance(mul, numbers.Integral):
            do_assert(mul >= 0.0, "Expected multiplier to have range [0, inf), got value %.4f." % (mul,))
            self.mul = randomizer.Deterministic(mul)
        else:#elif ia.is_iterable(mul):
            do_assert(len(mul) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(mul),))
            self.mul = randomizer.Uniform(mul[0], mul[1])


        #elif isinstance(mul, StochasticParameter):
        #    self.mul = mul
        #else:
        #    raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(mul),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = randomizer.Deterministic(int(per_channel))
        else: #ia.is_single_number(per_channel):
            do_assert(0 <= per_channel <= 1.0)
            self.per_channel = randomizer.Binomial(per_channel)
        #else:
        #    raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

    def __call__(self, images):
        #input_dtypes = meta.copy_dtypes_for_restore(images, force_list=True)
        # TODO: next step do the multiplication in batches ( GPU efficient)

        result = images
        nb_images = len(images)
        seeds = randomizer.current_random_state().randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i].astype(np.float32)
            rs_image = randomizer.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                nb_channels = image.shape[2]
                samples = self.mul.draw_samples((nb_channels,), random_state=rs_image)
                for c, sample in enumerate(samples):
                    do_assert(sample >= 0)

                    F.multiply(image[c, ...].cuda(), torch.ones(1).cuda() * float(sample))
            else:
                sample = self.mul.draw_sample(random_state=rs_image)
                do_assert(sample >= 0)
                F.multiply(image.cuda(), torch.ones(1).cuda() * float(sample))


            #image = meta.clip_augmented_image_(image, 0, 255) # TODO make value range more flexible
            #image = meta.restore_augmented_image_dtype_(image, input_dtypes[i])

            result[i] = image

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.mul]




def Dropout(p=0, per_channel=False, name=None, deterministic=False,
            random_state=None):
    """
    Augmenter that sets a certain fraction of pixels in images to zero.

    Parameters
    ----------
    p : float or tuple of two floats or StochasticParameter, optional(default=0)
        The probability of any pixel being dropped (i.e. set to
        zero).
            * If a float, then that value will be used for all images. A value
              of 1.0 would mean that all pixels will be dropped and 0.0 that
              no pixels would be dropped. A value of 0.05 corresponds to 5
              percent of all pixels dropped.
            * If a tuple (a, b), then a value p will be sampled from the
              range a <= p <= b per image and be used as the pixel's dropout
              probability.
            * If a StochasticParameter, then this parameter will be used to
              determine per pixel whether it should be dropped (sampled value
              of 0) or shouldn't (sampled value of 1).

    per_channel : bool or float, optional(default=False)
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
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
    >>> aug = iaa.Dropout(0.02)

    drops 2 percent of all pixels.

    >>> aug = iaa.Dropout((0.0, 0.05))

    drops in each image a random fraction of all pixels, where the fraction
    is in the range 0.0 <= x <= 0.05.

    >>> aug = iaa.Dropout(0.02, per_channel=True)

    drops 2 percent of all pixels in a channel-wise fashion, i.e. it is unlikely
    for any pixel to have all channels set to zero (black pixels).

    >>> aug = iaa.Dropout(0.02, per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """
    if ia.is_single_number(p):
        p2 = Binomial(1 - p)
    elif ia.is_iterable(p):
        ia.do_assert(len(p) == 2)
        ia.do_assert(p[0] < p[1])
        ia.do_assert(0 <= p[0] <= 1.0)
        ia.do_assert(0 <= p[1] <= 1.0)
        p2 = Binomial(Uniform(1 - p[1], 1 - p[0]))
    elif isinstance(p, StochasticParameter):
        p2 = p
    else:
        raise Exception("Expected p to be float or int or StochasticParameter, got %s." % (type(p),))
    return MultiplyElementwise(p2, per_channel=per_channel, name=name, deterministic=deterministic, random_state=random_state)




def CoarseDropout(p=0, size_px=None, size_percent=None,
                  per_channel=False, min_size=4, name=None, deterministic=False,
                  random_state=None):
    """
    Augmenter that sets rectangular areas within images to zero.

    In contrast to Dropout, these areas can have larger sizes.
    (E.g. you might end up with three large black rectangles in an image.)
    Note that the current implementation leads to correlated sizes,
    so when there is one large area that is dropped, there is a high likelihood
    that all other dropped areas are also large.

    This method is implemented by generating the dropout mask at a
    lower resolution (than the image has) and then upsampling the mask
    before dropping the pixels.

    Parameters
    ----------
    p : float or tuple of two floats or StochasticParameter, optional(default=0)
        The probability of any pixel being dropped (i.e. set to
        zero).
            * If a float, then that value will be used for all pixels. A value
              of 1.0 would mean, that all pixels will be dropped. A value of
              0.0 would lead to no pixels being dropped.
            * If a tuple (a, b), then a value p will be sampled from the
              range a <= p <= b per image and be used as the pixel's dropout
              probability.
            * If a StochasticParameter, then this parameter will be used to
              determine per pixel whether it should be dropped (sampled value
              of 0) or shouldn't (sampled value of 1).

    size_px : int or tuple of two ints or StochasticParameter, optional(default=None)
        The size of the lower resolution image from which to sample the dropout
        mask in absolute pixel dimensions.
            * If an integer, then that size will be used for both height and
              width. E.g. a value of 3 would lead to a 3x3 mask, which is then
              upsampled to HxW, where H is the image size and W the image width.
            * If a tuple (a, b), then two values M, N will be sampled from the
              range [a..b] and the mask will be generated at size MxN, then
              upsampled to HxW.
            * If a StochasticParameter, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of two floats or StochasticParameter, optional(default=None)
        The size of the lower resolution image from which to sample the dropout
        mask *in percent* of the input image.
            * If a float, then that value will be used as the percentage of the
              height and width (relative to the original size). E.g. for value
              p, the mask will be sampled from (p*H)x(p*W) and later upsampled
              to HxW.
            * If a tuple (a, b), then two values m, n will be sampled from the
              interval (a, b) and used as the percentages, i.e the mask size
              will be (m*H)x(n*W).
            * If a StochasticParameter, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value (is dropped / is not dropped)
        for all channels of a pixel (False) or to sample a new value for each
        channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    min_size : int, optional(default=4)
        Minimum size of the low resolution mask, both width and height. If
        `size_percent` or `size_px` leads to a lower value than this, `min_size`
        will be used instead. This should never have a value of less than 2,
        otherwise one may end up with a 1x1 low resolution mask, leading easily
        to the whole image being dropped.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Dropout(0.02, size_percent=0.5)

    drops 2 percent of all pixels on an lower-resolution image that has
    50 percent of the original image's size, leading to dropped areas that
    have roughly 2x2 pixels size.


    >>> aug = iaa.Dropout((0.0, 0.05), size_percent=(0.05, 0.5))

    generates a dropout mask at 5 to 50 percent of image's size. In that mask,
    0 to 5 percent of all pixels are dropped (random per image).

    >>> aug = iaa.Dropout((0.0, 0.05), size_px=(2, 16))

    same as previous example, but the lower resolution image has 2 to 16 pixels
    size.

    >>> aug = iaa.Dropout(0.02, size_percent=0.5, per_channel=True)

    drops 2 percent of all pixels at 50 percent resolution (2x2 sizes)
    in a channel-wise fashion, i.e. it is unlikely
    for any pixel to have all channels set to zero (black pixels).

    >>> aug = iaa.Dropout(0.02, size_percent=0.5, per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """
    if ia.is_single_number(p):
        p2 = Binomial(1 - p)
    elif ia.is_iterable(p):
        ia.do_assert(len(p) == 2)
        ia.do_assert(p[0] < p[1])
        ia.do_assert(0 <= p[0] <= 1.0)
        ia.do_assert(0 <= p[1] <= 1.0)
        p2 = Binomial(Uniform(1 - p[1], 1 - p[0]))
    elif isinstance(p, StochasticParameter):
        p2 = p
    else:
        raise Exception("Expected p to be float or int or StochasticParameter, got %s." % (type(p),))

    if size_px is not None:
        p3 = FromLowerResolution(other_param=p2, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        p3 = FromLowerResolution(other_param=p2, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    return MultiplyElementwise(p3, per_channel=per_channel, name=name, deterministic=deterministic, random_state=random_state)




class ContrastNormalization(BaseTransform):
    """
    Augmenter that changes the contrast of images.

    Parameters
    ----------
    alpha : float or tuple of two floats or StochasticParameter, optional(default=1.0)
        Strength of the contrast normalization. Higher values than 1.0
        lead to higher contrast, lower values decrease the contrast.
            * If a float, then that value will be used for all images.
            * If a tuple (a, b), then a value will be sampled per image from
              the range a <= x <= b and be used as the alpha value.
            * If a StochasticParameter, then this parameter will be used to
              sample the alpha value per image.

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
    >>> iaa.ContrastNormalization((0.5, 1.5))

    Decreases oder improves contrast per image by a random factor between
    0.5 and 1.5. The factor 0.5 means that any difference from the center value
    (i.e. 128) will be halved, leading to less contrast.

    >>> iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)

    Same as before, but for 50 percent of all images the normalization is done
    independently per channel (i.e. factors can vary per channel for the same
    image). In the other 50 percent of all images, the factor is the same for
    all channels.

    """

    def __init__(self, alpha=1.0, per_channel=False, name=None, deterministic=False, random_state=None):
        super(ContrastNormalization, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_number(alpha):
            ia.do_assert(alpha >= 0.0, "Expected alpha to have range (0, inf), got value %.4f." % (alpha,))
            self.alpha = Deterministic(alpha)
        elif ia.is_iterable(alpha):
            ia.do_assert(len(alpha) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(alpha),))
            self.alpha = Uniform(alpha[0], alpha[1])
        elif isinstance(alpha, StochasticParameter):
            self.alpha = alpha
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(alpha),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            ia.do_assert(0 <= per_channel <= 1.0)
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

    def _augment_images(self, images, random_state, parents, hooks):
        input_dtypes = meta.copy_dtypes_for_restore(images, force_list=True)

        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i].astype(np.float32)
            rs_image = ia.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel:
                nb_channels = images[i].shape[2]
                alphas = self.alpha.draw_samples((nb_channels,), random_state=rs_image)
                for c, alpha in enumerate(alphas):
                    image[..., c] = alpha * (image[..., c] - 128) + 128
            else:
                alpha = self.alpha.draw_sample(random_state=rs_image)
                image = alpha * (image - 128) + 128

            image = meta.clip_augmented_image_(image, 0, 255) # TODO make value range more flexible
            image = meta.restore_augmented_image_dtype_(image, input_dtypes[i])

            result[i] = image

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.alpha]






