


class Multiply(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, multiply_neg, multiply_pos):

        self.multiply_neg = multiply_neg
        self.multiply = multiply_pos

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

    def __init__(self, multiply_neg, multiply_pos):
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

        Multiply(0.15, 2.5)


    ])

aug_cpu = transforms.Compose([
        transforms.ToTensor(),
    MultiplyCPU(0.15, 2.5)


    ])


