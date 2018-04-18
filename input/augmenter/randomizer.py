import numpy as np
import numbers

import six
from abc import ABCMeta, abstractmethod
from utils import checking  as ch


def handle_continuous_param(param, name, value_range=None, tuple_to_uniform=True, list_to_choice=True):
    def check_value_range(v):
        if value_range is None:
            return True
        elif isinstance(value_range, tuple):
            ch.do_assert(len(value_range) == 2)
            if value_range[0] is None and value_range[1] is None:
                return True
            elif value_range[0] is None:
                ch.do_assert(v <= value_range[1], "Parameter '%s' is outside of the expected value range (x <= %.4f)" % (name, value_range[1]))
                return True
            elif value_range[1] is None:
                ch.do_assert(value_range[0] <= v, "Parameter '%s' is outside of the expected value range (%.4f <= x)" % (name, value_range[0]))
                return True
            else:
                ch.do_assert(value_range[0] <= v <= value_range[1], "Parameter '%s' is outside of the expected value range (%.4f <= x <= %.4f)" % (name, value_range[0], value_range[1]))
                return True
        elif ch.is_callable(value_range):
            value_range(v)
            return True
        else:
            raise Exception("Unexpected input for value_range, got %s." % (str(value_range),))

    if ch.is_single_number(param):
        check_value_range(param)
        return Deterministic(param)
    elif tuple_to_uniform and isinstance(param, tuple):
        ch.do_assert(len(param) == 2)
        check_value_range(param[0])
        check_value_range(param[1])
        return Uniform(param[0], param[1])
    elif list_to_choice and isinstance(param, (tuple, list)):
        for param_i in param:
            check_value_range(param_i)
        return Choice(param)
    elif isinstance(param, StochasticParameter):
        return param
    else:
        raise Exception("Expected number, tuple of two number, list of number or StochasticParameter for %s, got %s." % (name, type(param),))


def handle_discrete_param(param, name, value_range=None, tuple_to_uniform=True, list_to_choice=True, allow_floats=True):
    def check_value_range(v):
        if value_range is None:
            return True
        elif isinstance(value_range, tuple):
            ch.do_assert(len(value_range) == 2)
            if value_range[0] is None and value_range[1] is None:
                return True
            elif value_range[0] is None:
                ch.do_assert(v <= value_range[1], "Parameter '%s' is outside of the expected value range (x <= %.4f)" % (name, value_range[1]))
                return True
            elif value_range[1] is None:
                ch.do_assert(value_range[0] <= v, "Parameter '%s' is outside of the expected value range (%.4f <= x)" % (name, value_range[0]))
                return True
            else:
                ch.do_assert(value_range[0] <= v <= value_range[1], "Parameter '%s' is outside of the expected value range (%.4f <= x <= %.4f)" % (name, value_range[0], value_range[1]))
                return True
        elif ch.is_callable(value_range):
            value_range(v)
            return True
        else:
            raise Exception("Unexpected input for value_range, got %s." % (str(value_range),))

    if isinstance(param, numbers.Integral) or (allow_floats and isinstance(param, numbers.Real)):
        check_value_range(param)
        return Deterministic(int(param))
    elif tuple_to_uniform and isinstance(param, tuple):
        ch.do_assert(len(param) == 2)
        if allow_floats:
            ch.do_assert(ch.is_single_number(param[0]), "Expected number, got %s." % (type(param[0]),))
            ch.do_assert(ch.is_single_number(param[1]), "Expected number, got %s." % (type(param[1]),))
        else:
            ch.do_assert(isinstance(param[0], numbers.Integral), "Expected integer, got %s." % (type(param[0]),))
            ch.do_assert(isinstance(param[1], numbers.Integral), "Expected integer, got %s." % (type(param[1]),))
        check_value_range(param[0])
        check_value_range(param[1])
        return DiscreteUniform(int(param[0]), int(param[1]))
    elif list_to_choice and isinstance(param, (tuple, list)):
        for param_i in param:
            check_value_range(param_i)
        return Choice([int(param_i) for param_i in param])
    elif isinstance(param, StochasticParameter):
        return param
    else:
        if allow_floats:
            raise Exception("Expected number, tuple of two number, list of number or StochasticParameter for %s, got %s." % (name, type(param),))
        else:
            raise Exception("Expected int, tuple of two int, list of int or StochasticParameter for %s, got %s." % (name, type(param),))


CURRENT_RANDOM_STATE = np.random.RandomState(42)

def new_random_state(seed=None, fully_random=False):
    """
    Returns a new random state.

    Parameters
    ----------
    seed : None or int, optional(default=None)
        Optional seed value to use.
        The same datatypes are allowed as for np.random.RandomState(seed).

    fully_random : bool, optional(default=False)
        Whether to use numpy's random initialization for the
        RandomState (used if set to True). If False, a seed is sampled from
        the global random state, which is a bit faster and hence the default.

    Returns
    -------
    out : np.random.RandomState
        The new random state.

    """
    if seed is None:
        if not fully_random:
            # sample manually a seed instead of just RandomState(),
            # because the latter one
            # is way slower.
            seed = CURRENT_RANDOM_STATE.randint(0, 10**6, 1)[0]
    return np.random.RandomState(seed)

def current_random_state():

    # We instantiate a current/global random state here once.
    # One can also call np.random, but that is (in contrast to np.random.RandomState)
    # a module and hence cannot be copied via deepcopy. That's why we use RandomState
    # here (and in all augmenters) instead of np.random.
    return CURRENT_RANDOM_STATE


def forward_random_state(random_state):
    random_state.uniform()



@six.add_metaclass(ABCMeta)
class StochasticParameter(object): # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Abstract parent class for all stochastic parameters.

    Stochastic parameters are here all parameters from which values are
    supposed to be sampled. Usually the sampled values are to a degree random.
    E.g. a stochastic parameter may be the range [-10, 10], with sampled
    values being 5.2, -3.7, -9.7 and 6.4.

    """

    def __init__(self):
        super(StochasticParameter, self).__init__()

    def draw_sample(self, random_state=None):
        """
        Draws a single sample value from this parameter.

        Parameters
        ----------
        random_state : None or np.random.RandomState, optional(default=None)
            A random state to use during the sampling process.
            If None, the libraries global random state will be used.

        Returns
        -------
        out : anything
            A single sample value.

        """
        return self.draw_samples(1, random_state=random_state)[0]

    def draw_samples(self, size, random_state=None):
        """
        Draws one or more sample values from the parameter.

        Parameters
        ----------
        size : tuple of int
            Number of sample values by
            dimension.

        random_state : None or np.random.RandomState, optional(default=None)
            A random state to use during the sampling process.
            If None, the libraries global random state will be used.

        Returns
        -------
        out : (size) iterable
            Sampled values. Usually a numpy ndarray of basically any dtype,
            though not strictly limited to numpy arrays.

        """
        random_state = random_state if random_state is not None else CURRENT_RANDOM_STATE
        samples = self._draw_samples(size, random_state)
        forward_random_state(random_state)

        return samples


class Deterministic(StochasticParameter):
    """
    Parameter that resembles a constant value.

    If N values are sampled from this parameter, it will return N times V,
    where V is the constant value.

    Parameters
    ----------
    value : number or string or StochasticParameter
        A constant value to use.
        A string may be provided to generate arrays of strings.
        If this is a StochasticParameter, a single value will be sampled
        from it exactly once and then used as the constant value.

    Examples
    --------
    >>> param = Deterministic(10)

    Will always sample the value 10.

    """
    def __init__(self, value):
        super(Deterministic, self).__init__()

        if isinstance(value, StochasticParameter):
            self.value = value.draw_sample()
        elif isinstance(value, numbers.Integral) or isinstance(value, numbers.Real) or isinstance(value, six.string_types):
            self.value = value
        else:
            raise Exception("Expected StochasticParameter object or number or string, got %s." % (type(value),))

    def _draw_samples(self, size, random_state):
        return np.tile(np.array([self.value]), size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.value, numbers.Integral):
            return "Deterministic(int %d)" % (self.value,)
        elif isinstance(self.value, numbers.Real):
            return "Deterministic(float %.8f)" % (self.value,)
        else:
            return "Deterministic(%s)" % (str(self.value),)


class Binomial(StochasticParameter):
    """
    Binomial distribution.

    Parameters
    ----------
    p : number or tuple of two number or list of number or StochasticParameter
        Probability of the binomial distribution. Expected to be in the
        range [0, 1]. If this is a StochasticParameter, the value will be
        sampled once per call to _draw_samples().

    Examples
    --------
    >>> param = Binomial(Uniform(0.01, 0.2))

    Uses a varying probability `p` between 0.01 and 0.2 per sampling.

    """

    def __init__(self, p):
        super(Binomial, self).__init__()

        """
        if isinstance(p, StochasticParameter):
            self.p = p
        elif ia.is_single_number(p):
            ia.do_assert(0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,))
            self.p = Deterministic(float(p))
        else:
            raise Exception("Expected StochasticParameter or float/int value, got %s." % (type(p),))
        """

        self.p = handle_continuous_param(p, "p")

    def _draw_samples(self, size, random_state):
        p = self.p.draw_sample(random_state=random_state)
        ia.do_assert(0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,))
        return random_state.binomial(1, p, size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.p, float):
            return "Binomial(%.4f)" % (self.p,)
        else:
            return "Binomial(%s)" % (self.p,)




class Uniform(StochasticParameter):
    """
    Parameter that resembles a (continuous) uniform range [a, b).

    Parameters
    ----------
    {a, b} : number or tuple of two number or list of number or StochasticParameter
        Lower and upper bound of the sampling range. Values will be sampled
        from a <= x < b. All sampled values will be continuous. If a or b is
        a StochasticParameter, it will be queried once per sampling to
        estimate the value of a/b. If a>b, the values will automatically be
        flipped. If a==b, all generated values will be identical to a.

    Examples
    --------
    >>> param = Uniform(0, 10.0)

    Samples random values from the range [0, 10.0).

    """
    def __init__(self, a, b):
        super(Uniform, self).__init__()

        """
        ia.do_assert(isinstance(a, (int, float, StochasticParameter)), "Expected a to be int, float or StochasticParameter, got %s" % (type(a),))
        ia.do_assert(isinstance(b, (int, float, StochasticParameter)), "Expected b to be int, float or StochasticParameter, got %s" % (type(b),))

        if ia.is_single_number(a):
            self.a = Deterministic(a)
        else:
            self.a = a

        if ia.is_single_number(b):
            self.b = Deterministic(b)
        else:
            self.b = b
        """

        self.a = handle_continuous_param(a, "a")
        self.b = handle_continuous_param(b, "b")

    def _draw_samples(self, size, random_state):
        a = self.a.draw_sample(random_state=random_state)
        b = self.b.draw_sample(random_state=random_state)
        if a > b:
            a, b = b, a
        elif a == b:
            return np.tile(np.array([a]), size)
        return random_state.uniform(a, b, size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Uniform(%s, %s)" % (self.a, self.b)



class DiscreteUniform(StochasticParameter):
    """
    Parameter that resembles a discrete range of values [a .. b].

    Parameters
    ----------
    {a, b} : int or StochasticParameter
        Lower and upper bound of the sampling range. Values will be sampled
        from a <= x <= b. All sampled values will be discrete. If a or b is
        a StochasticParameter, it will be queried once per sampling to
        estimate the value of a/b. If a>b, the values will automatically be
        flipped. If a==b, all generated values will be identical to a.

    Examples
    --------
    >>> param = DiscreteUniform(10, Choice([20, 30, 40]))

    Sampled values will be discrete and come from the either [10..20] or
    [10..30] or [10..40].

    """

    def __init__(self, a, b):
        super(DiscreteUniform, self).__init__()

        """
        # for two ints the samples will be from range a <= x <= b
        ia.do_assert(isinstance(a, (int, StochasticParameter)), "Expected a to be int or StochasticParameter, got %s" % (type(a),))
        ia.do_assert(isinstance(b, (int, StochasticParameter)), "Expected b to be int or StochasticParameter, got %s" % (type(b),))

        if ia.is_single_integer(a):
            self.a = Deterministic(a)
        else:
            self.a = a

        if ia.is_single_integer(b):
            self.b = Deterministic(b)
        else:
            self.b = b
        """
        self.a = handle_discrete_param(a, "a")
        self.b = handle_discrete_param(b, "b")

    def _draw_samples(self, size, random_state):
        a = self.a.draw_sample(random_state=random_state)
        b = self.b.draw_sample(random_state=random_state)
        if a > b:
            a, b = b, a
        elif a == b:
            return np.tile(np.array([a]), size)
        return random_state.randint(a, b + 1, size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "DiscreteUniform(%s, %s)" % (self.a, self.b)


class Choice(StochasticParameter):
    """
    Parameter that samples value from a list of allowed values.

    Parameters
    ----------
    a : iterable
        List of allowed values.
        Usually expected to be integers, floats or strings.

    replace : bool, optional(default=True)
        Whether to perform sampling with or without
        replacing.

    p : None or iterable, optional(default=None)
        Optional probabilities of each element in `a`.
        Must have the same length as `a` (if provided).

    Examples
    --------
    >>> param = Choice([0.25, 0.5, 0.75], p=[0.25, 0.5, 0.25])

    Parameter of which 50 pecent of all sampled values will be 0.5.
    The other 50 percent will be either 0.25 or 0.75.

    """
    def __init__(self, a, replace=True, p=None):
        super(Choice, self).__init__()

        self.a = a
        self.replace = replace
        self.p = p

    def _draw_samples(self, size, random_state):
        if any([isinstance(a_i, StochasticParameter) for a_i in self.a]):
            seed = random_state.randint(0, 10**6, 1)[0]
            samples = ia.new_random_state(seed).choice(self.a, np.prod(size), replace=self.replace, p=self.p)

            # collect the sampled parameters and how many samples must be taken
            # from each of them
            params_counter = defaultdict(lambda: 0)
            #params_keys = set()
            for sample in samples:
                if isinstance(sample, StochasticParameter):
                    key = str(sample)
                    params_counter[key] += 1
                    #params_keys.add(key)

            # collect per parameter once the required number of samples
            # iterate here over self.a to always use the same seed for
            # the same parameter
            # TODO this might fail if the same parameter is added
            # multiple times to self.a?
            # TODO this will fail if a parameter cant handle size=(N,)
            param_to_samples = dict()
            for i, param in enumerate(self.a):
                key = str(param)
                if key in params_counter:
                    #print("[Choice] sampling %d from %s" % (params_counter[key], key))
                    param_to_samples[key] = param.draw_samples(
                        size=(params_counter[key],),
                        random_state=ia.new_random_state(seed+1+i)
                    )

            # assign the values sampled from the parameters to the `samples`
            # array by replacing the respective parameter
            param_to_readcount = defaultdict(lambda: 0)
            for i, sample in enumerate(samples):
                #if i%10 == 0:
                #    print("[Choice] assigning sample %d" % (i,))
                if isinstance(sample, StochasticParameter):
                    key = str(sample)
                    readcount = param_to_readcount[key]
                    #if readcount%10==0:
                    #    print("[Choice] readcount %d for %s" % (readcount, key))
                    samples[i] = param_to_samples[key][readcount]
                    param_to_readcount[key] += 1

            samples = samples.reshape(size)
        else:
            samples = random_state.choice(self.a, size, replace=self.replace, p=self.p)
        return samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Choice(a=%s, replace=%s, p=%s)" % (str(self.a), str(self.replace), str(self.p),)



#
# class Sometimes(Augmenter):
#     """
#     Augment only p percent of all images with one or more augmenters.
#
#     Let C be one or more child augmenters given to Sometimes.
#     Let p be the percent of images to augment.
#     Let I be the input images.
#     Then (on average) p percent of all images in I will be augmented using C.
#
#     Parameters
#     ----------
#     p : float or StochasticParameter, optional(default=0.5)
#         Sets the probability with which the given augmenters will be applied to
#         input images. E.g. a value of 0.5 will result in 50 percent of all
#         input images being augmented.
#
#     then_list : None or Augmenter or list of Augmenters, optional(default=None)
#         Augmenter(s) to apply to p percent of all images.
#
#     else_list : None or Augmenter or list of Augmenters, optional(default=None)
#         Augmenter(s) to apply to (1-p) percent of all images.
#         These augmenters will be applied only when the ones in then_list
#         are NOT applied (either-or-relationship).
#
#     name : string, optional(default=None)
#         See `Augmenter.__init__()`
#
#     deterministic : bool, optional(default=False)
#         See `Augmenter.__init__()`
#
#     random_state : int or np.random.RandomState or None, optional(default=None)
#         See `Augmenter.__init__()`
#
#     Examples
#     --------
#     >>> aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3))
#
#     when calling `aug.augment_images()`, only (on average) 50 percent of
#     all images will be blurred.
#
#     >>> aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3), iaa.Fliplr(1.0))
#
#     when calling `aug.augment_images()`, (on average) 50 percent of all images
#     will be blurred, the other (again, on average) 50 percent will be
#     horizontally flipped.
#
#     """
#
#     def __init__(self, p=0.5, then_list=None, else_list=None, name=None, deterministic=False, random_state=None):
#         super(Sometimes, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
#
#         if ia.is_single_float(p) or ia.is_single_integer(p):
#             ia.do_assert(0 <= p <= 1)
#             self.p = Binomial(p)
#         elif isinstance(p, StochasticParameter):
#             self.p = p
#         else:
#             raise Exception("Expected float/int in range [0, 1] or StochasticParameter as p, got %s." % (type(p),))
#
#         if then_list is None:
#             self.then_list = Sequential([], name="%s-then" % (self.name,))
#         elif ia.is_iterable(then_list):
#             # TODO does this work with SomeOf(), Sequential(), ... ?
#             self.then_list = Sequential(then_list, name="%s-then" % (self.name,))
#         elif isinstance(then_list, Augmenter):
#             self.then_list = Sequential([then_list], name="%s-then" % (self.name,))
#         else:
#             raise Exception("Expected None, Augmenter or list/tuple as then_list, got %s." % (type(then_list),))
#
#         if else_list is None:
#             self.else_list = Sequential([], name="%s-else" % (self.name,))
#         elif ia.is_iterable(else_list):
#             self.else_list = Sequential(else_list, name="%s-else" % (self.name,))
#         elif isinstance(else_list, Augmenter):
#             self.else_list = Sequential([else_list], name="%s-else" % (self.name,))
#         else:
#             raise Exception("Expected None, Augmenter or list/tuple as else_list, got %s." % (type(else_list),))
#
#     def _augment_images(self, images, random_state, parents, hooks):
#         if hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
#             input_is_np_array = ia.is_np_array(images)
#             if input_is_np_array:
#                 input_dtype = images.dtype
#
#             nb_images = len(images)
#             samples = self.p.draw_samples((nb_images,), random_state=random_state)
#
#             # create lists/arrays of images for if and else lists (one for each)
#             indices_then_list = np.where(samples == 1)[0] # np.where returns tuple(array([0, 5, 9, ...])) or tuple(array([]))
#             indices_else_list = np.where(samples == 0)[0]
#             if isinstance(images, list):
#                 images_then_list = [images[i] for i in indices_then_list]
#                 images_else_list = [images[i] for i in indices_else_list]
#             else:
#                 images_then_list = images[indices_then_list]
#                 images_else_list = images[indices_else_list]
#
#             # augment according to if and else list
#             result_then_list = self.then_list.augment_images(
#                 images=images_then_list,
#                 parents=parents + [self],
#                 hooks=hooks
#             )
#             result_else_list = self.else_list.augment_images(
#                 images=images_else_list,
#                 parents=parents + [self],
#                 hooks=hooks
#             )
#
#             # map results of if/else lists back to their initial positions (in "images" variable)
#             result = [None] * len(images)
#             for idx_result_then_list, idx_images in enumerate(indices_then_list):
#                 result[idx_images] = result_then_list[idx_result_then_list]
#             for idx_result_else_list, idx_images in enumerate(indices_else_list):
#                 result[idx_images] = result_else_list[idx_result_else_list]
#
#             # if input was a list, keep the output as a list too,
#             # otherwise it was a numpy array, so make the output a numpy array too
#             if input_is_np_array:
#                 result = np.array(result, dtype=input_dtype)
#         else:
#             result = images
#
#         return result
#
#
#
#     def _to_deterministic(self):
#         aug = self.copy()
#         aug.then_list = aug.then_list.to_deterministic()
#         aug.else_list = aug.else_list.to_deterministic()
#         aug.deterministic = True
#         aug.random_state = ia.new_random_state()
#         return aug
#
#     def get_parameters(self):
#         return [self.p]
#
#     def get_children_lists(self):
#         return [self.then_list, self.else_list]
#
#     def __str__(self):
#         return "Sometimes(p=%s, name=%s, then_list=[%s], else_list=[%s], deterministic=%s)" % (self.p, self.name, self.then_list, self.else_list, self.deterministic)
