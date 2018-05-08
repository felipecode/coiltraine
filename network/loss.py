from logger import coil_logger
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
import torch


class Loss(object):

    def __init__(self):
        pass

    # TODO: iteration control should go inside the logger, somehow
    def __call__(self, tensor, output_tensor):
        coil_logger.add_message('Loss', {
            "Iteration": 765,
            "LossValue": [0.232,  0.232,  0.332,  0.2322,  0.232,  0.232, 0.232]}
        )
        return tensor



    def MSELoss(self, branches, targets, controls, size_average=True, reduce=True):
        """
        Args:
              branches - A list contains 5 branches results
              targets - The target (here are steer, gas and brake)
              controls - The control directions
              size_average - By default, the losses are averaged over observations for each minibatch.
                             However, if the field size_average is set to ``False``, the losses are instead
                             summed for each minibatch. Only applies when reduce is ``True``. Default: ``True``
              reduce - By default, the losses are averaged over observations for each minibatch, or summed,
                       depending on size_average. When reduce is ``False``, returns a loss per input/target
                       element instead and ignores size_average. Default: ``True``

        return: MSE Loss

        """
        # command flags: 2 - follow lane; 3 - turn left; 4 - turn right; 5 - go strange

        # when command = 2, branch 1 (follow lane) is activated
        controls_b1 = (controls == 2)
        controls_b1 = torch.tensor(controls_b1, dtype = torch.float32).cuda()
        controls_b1 = torch.cat([controls_b1,controls_b1,controls_b1],1)    # activation for steer, gas and brake
        # when command = 3, branch 2 (turn left) is activated
        controls_b2 = (controls == 3)
        controls_b2 = torch.tensor(controls_b2, dtype = torch.float32).cuda()
        controls_b2 = torch.cat([controls_b2, controls_b2, controls_b2], 1)
        # when command = 4, branch 3 (turn right) is activated
        controls_b3 = (controls == 4)
        controls_b3 = torch.tensor(controls_b3, dtype = torch.float32).cuda()
        controls_b3 = torch.cat([controls_b3, controls_b3, controls_b3], 1)
        # when command = 5, branch 4 (go strange) is activated
        controls_b4 = (controls == 5)
        controls_b4 = torch.tensor(controls_b4, dtype = torch.float32).cuda()
        controls_b4 = torch.cat([controls_b4, controls_b4, controls_b4], 1)

        # calculate loss for each branch with specific activation
        loss_b1 = ((branches[0] - targets) * controls_b1) ** 2
        loss_b2 = (branches[1] * controls_b2 - targets * controls_b2) ** 2
        loss_b3 = (branches[2] * controls_b3 - targets * controls_b3) ** 2
        loss_b4 = (branches[3] * controls_b4 - targets * controls_b4) ** 2

        # add all branches losses together
        mse_loss = loss_b1 + loss_b2 + loss_b3 + loss_b4

        if reduce:
            if size_average:
                mse_loss = torch.sum(mse_loss)/(mse_loss.shape[0] * mse_loss.shape[1])
            else:
                mse_loss = torch.sum(mse_loss)
        else:
            if size_average:
                raise RuntimeError(" size_average can not be applies when reduce is set to 'False' ")
            else:
                mse_loss = mse_loss

        return mse_loss
