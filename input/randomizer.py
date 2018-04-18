import numpy as np

# We instantiate a current/global random state here once.
# One can also call np.random, but that is (in contrast to np.random.RandomState)
# a module and hence cannot be copied via deepcopy. That's why we use RandomState
# here (and in all augmenters) instead of np.random.
CURRENT_RANDOM_STATE = np.random.RandomState(42)


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




class Sometimes(Augmenter):
    """
    Augment only p percent of all images with one or more augmenters.

    Let C be one or more child augmenters given to Sometimes.
    Let p be the percent of images to augment.
    Let I be the input images.
    Then (on average) p percent of all images in I will be augmented using C.

    Parameters
    ----------
    p : float or StochasticParameter, optional(default=0.5)
        Sets the probability with which the given augmenters will be applied to
        input images. E.g. a value of 0.5 will result in 50 percent of all
        input images being augmented.

    then_list : None or Augmenter or list of Augmenters, optional(default=None)
        Augmenter(s) to apply to p percent of all images.

    else_list : None or Augmenter or list of Augmenters, optional(default=None)
        Augmenter(s) to apply to (1-p) percent of all images.
        These augmenters will be applied only when the ones in then_list
        are NOT applied (either-or-relationship).

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3))

    when calling `aug.augment_images()`, only (on average) 50 percent of
    all images will be blurred.

    >>> aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3), iaa.Fliplr(1.0))

    when calling `aug.augment_images()`, (on average) 50 percent of all images
    will be blurred, the other (again, on average) 50 percent will be
    horizontally flipped.

    """

    def __init__(self, p=0.5, then_list=None, else_list=None, name=None, deterministic=False, random_state=None):
        super(Sometimes, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if ia.is_single_float(p) or ia.is_single_integer(p):
            ia.do_assert(0 <= p <= 1)
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected float/int in range [0, 1] or StochasticParameter as p, got %s." % (type(p),))

        if then_list is None:
            self.then_list = Sequential([], name="%s-then" % (self.name,))
        elif ia.is_iterable(then_list):
            # TODO does this work with SomeOf(), Sequential(), ... ?
            self.then_list = Sequential(then_list, name="%s-then" % (self.name,))
        elif isinstance(then_list, Augmenter):
            self.then_list = Sequential([then_list], name="%s-then" % (self.name,))
        else:
            raise Exception("Expected None, Augmenter or list/tuple as then_list, got %s." % (type(then_list),))

        if else_list is None:
            self.else_list = Sequential([], name="%s-else" % (self.name,))
        elif ia.is_iterable(else_list):
            self.else_list = Sequential(else_list, name="%s-else" % (self.name,))
        elif isinstance(else_list, Augmenter):
            self.else_list = Sequential([else_list], name="%s-else" % (self.name,))
        else:
            raise Exception("Expected None, Augmenter or list/tuple as else_list, got %s." % (type(else_list),))

    def _augment_images(self, images, random_state, parents, hooks):
        if hooks.is_propagating(images, augmenter=self, parents=parents, default=True):
            input_is_np_array = ia.is_np_array(images)
            if input_is_np_array:
                input_dtype = images.dtype

            nb_images = len(images)
            samples = self.p.draw_samples((nb_images,), random_state=random_state)

            # create lists/arrays of images for if and else lists (one for each)
            indices_then_list = np.where(samples == 1)[0] # np.where returns tuple(array([0, 5, 9, ...])) or tuple(array([]))
            indices_else_list = np.where(samples == 0)[0]
            if isinstance(images, list):
                images_then_list = [images[i] for i in indices_then_list]
                images_else_list = [images[i] for i in indices_else_list]
            else:
                images_then_list = images[indices_then_list]
                images_else_list = images[indices_else_list]

            # augment according to if and else list
            result_then_list = self.then_list.augment_images(
                images=images_then_list,
                parents=parents + [self],
                hooks=hooks
            )
            result_else_list = self.else_list.augment_images(
                images=images_else_list,
                parents=parents + [self],
                hooks=hooks
            )

            # map results of if/else lists back to their initial positions (in "images" variable)
            result = [None] * len(images)
            for idx_result_then_list, idx_images in enumerate(indices_then_list):
                result[idx_images] = result_then_list[idx_result_then_list]
            for idx_result_else_list, idx_images in enumerate(indices_else_list):
                result[idx_images] = result_else_list[idx_result_else_list]

            # if input was a list, keep the output as a list too,
            # otherwise it was a numpy array, so make the output a numpy array too
            if input_is_np_array:
                result = np.array(result, dtype=input_dtype)
        else:
            result = images

        return result



    def _to_deterministic(self):
        aug = self.copy()
        aug.then_list = aug.then_list.to_deterministic()
        aug.else_list = aug.else_list.to_deterministic()
        aug.deterministic = True
        aug.random_state = ia.new_random_state()
        return aug

    def get_parameters(self):
        return [self.p]

    def get_children_lists(self):
        return [self.then_list, self.else_list]

    def __str__(self):
        return "Sometimes(p=%s, name=%s, then_list=[%s], else_list=[%s], deterministic=%s)" % (self.p, self.name, self.then_list, self.else_list, self.deterministic)
