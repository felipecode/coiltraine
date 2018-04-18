
import numpy as np

import torch
from torchvision import transforms

from . import augmenter_functions as F


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})




#TODO :


class Augmenter(object):


    def __init__(self, multiply_pos, multiply_neg):
        pass
    def __call__(self, tensor):
        pass

        # We define the callable to compose all the







#TODO change this augmenter to something more generic.

class Add(Augmenter):
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

        if ia.is_single_integer(value):
            ia.do_assert(-255 <= value <= 255, "Expected value to have range [-255, 255], got value %d." % (value,))
            self.value = Deterministic(value)
        elif ia.is_iterable(value):
            ia.do_assert(len(value) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(value),))
            self.value = DiscreteUniform(value[0], value[1])
        elif isinstance(value, StochasticParameter):
            self.value = value
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(value),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif ia.is_single_number(per_channel):
            ia.do_assert(0 <= per_channel <= 1.0, "Expected bool, or number in range [0, 1.0] for per_channel, got %s." % (type(per_channel),))
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

    def _augment_images(self, images, random_state, parents, hooks):
        input_dtypes = meta.copy_dtypes_for_restore(images, force_list=True)

        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in sm.xrange(nb_images):
            image = images[i].astype(np.int32)
            rs_image = ia.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                nb_channels = image.shape[2]
                samples = self.value.draw_samples((nb_channels,), random_state=rs_image)
                for c, sample in enumerate(samples):
                    # TODO make value range more flexible
                    ia.do_assert(-255 <= sample <= 255)
                    image[..., c] += sample
            else:
                sample = self.value.draw_sample(random_state=rs_image)
                ia.do_assert(-255 <= sample <= 255) # TODO make value range more flexible
                image += sample

            image = meta.clip_augmented_image_(image, 0, 255) # TODO make value range more flexible
            image = meta.restore_augmented_image_dtype_(image, input_dtypes[i])

            result[i] = image

        return result

    def get_parameters(self):
        return [self.value]





class Multiply(Augmenter):
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

        if ia.is_single_number(mul):
            ia.do_assert(mul >= 0.0, "Expected multiplier to have range [0, inf), got value %.4f." % (mul,))
            self.mul = Deterministic(mul)
        elif ia.is_iterable(mul):
            ia.do_assert(len(mul) == 2, "Expected tuple/list with 2 entries, got %d entries." % (len(mul),))
            self.mul = Uniform(mul[0], mul[1])
        elif isinstance(mul, StochasticParameter):
            self.mul = mul
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(mul),))

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
            if per_channel == 1:
                nb_channels = image.shape[2]
                samples = self.mul.draw_samples((nb_channels,), random_state=rs_image)
                for c, sample in enumerate(samples):
                    ia.do_assert(sample >= 0)
                    image[..., c] *= sample
            else:
                sample = self.mul.draw_sample(random_state=rs_image)
                ia.do_assert(sample >= 0)
                image *= sample

            image = meta.clip_augmented_image_(image, 0, 255) # TODO make value range more flexible
            image = meta.restore_augmented_image_dtype_(image, input_dtypes[i])

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




class ContrastNormalization(Augmenter):
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



class Multiply(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, multiply_pos, multiply_neg):
        self.multiply = multiply_pos
        self.multiply_neg = multiply_neg

    def __call__(self, tensor):

        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be added.
            numpy array: A numpy array directly (C, H, W) to be multiplied
        Returns:
            Tensor: Normalized Tensor image.
        """


        return F.multiply(tensor.cuda(), torch.ones(1).cuda()*self.multiply)




    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class MultiplyCPU(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, multiply_pos, multiply_neg):
        self.multiply = multiply_pos
        self.multiply_neg = multiply_neg

    def __call__(self, tensor):

        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be added.
            numpy array: A numpy array directly (C, H, W) to be multiplied
        Returns:
            Tensor: Normalized Tensor image.
        """
        if not _is_tensor_image(tensor):
            raise NotImplementedError

        return F.multiply(tensor, torch.ones(1) * self.multiply)




    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



















aug = transforms.Compose([

        Multiply(0.15, 2.5),
Multiply(0.15, 2.5),
Multiply(0.15, 2.5),
Multiply(0.15, 2.5),
Multiply(0.15, 2.5)

    ])

aug_cpu = transforms.Compose([
        transforms.ToTensor(),
    MultiplyCPU(0.15, 2.5),
    MultiplyCPU(0.15, 2.5),
    MultiplyCPU(0.15, 2.5),
    MultiplyCPU(0.15, 2.5),
    MultiplyCPU(0.15, 2.5)

    ])
