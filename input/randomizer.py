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