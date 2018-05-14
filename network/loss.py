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

        return tensor



    def MSELoss(self, branches, targets, controls, speed_gt, size_average=True, reduce=True, weights=None):
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
              *argv: weights - By default, the weights are all set to 1.0. To set different weights for different
                               outputs, a list containing different lambda for each target item is required.
                               The number of lambdas should be the same as the target items.

        return: MSE Loss

        """

        # weight different target items with lambdas
        if weights:
            if len(weights) != targets.shape[1]:
                raise ValueError('The input number of weight lambdas is '+ str(len(weights)) + ', while the number of the target items is '+ str(targets.shape[1]))
            else:
                lambda_matrix = torch.zeros(targets.shape).cuda()
                for i in range(targets.shape[1]):
                    lambda_matrix[:, i] = weights[i]
        else:
            lambda_matrix = torch.ones(targets.shape).cuda()


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

        # Normalize with the maximum speed from the training set (40 km/h)
        speed_gt = speed_gt / 40.0 # TODO: this we take from the loss function

        # calculate loss for each branch with specific activation
        loss_b1 = ((branches[0] - targets) * controls_b1) ** 2 * lambda_matrix
        loss_b2 = ((branches[1] - targets) * controls_b2) ** 2 * lambda_matrix
        loss_b3 = ((branches[2] - targets) * controls_b3) ** 2 * lambda_matrix
        loss_b4 = ((branches[3] - targets) * controls_b4) ** 2 * lambda_matrix
        loss_b5 = (branches[4] - speed_gt) ** 2

        # add all branches losses together
        mse_loss = loss_b1 + loss_b2 + loss_b3 + loss_b4


        if reduce:
            if size_average:
                mse_loss = torch.sum(mse_loss)/(mse_loss.shape[0] * mse_loss.shape[1]) + torch.sum(loss_b5)/mse_loss.shape[0]
            else:
                mse_loss = torch.sum(mse_loss) + torch.sum(loss_b5)
        else:
            if size_average:
                raise RuntimeError(" size_average can not be applies when reduce is set to 'False' ")
            else:
                mse_loss =torch.cat([mse_loss,loss_b5],1)

        return mse_loss

    def MELoss(self, branches, targets, controls, speed_gt, size_average=True, reduce=True, weights=None):
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
              *argv: weights - By default, the weights are all set to 1.0. To set different weights for different
                               outputs, a list containing different lambda for each target item is required.
                               The number of lambdas should be the same as the target items.

        return: MSE Loss

        """

        # weight different target items with lambdas
        if weights:
            if len(weights) != targets.shape[1]:
                raise ValueError('The input number of weight lambdas is '+ str(len(weights)) + ', while the number of the target items is '+ str(targets.shape[1]))
            else:
                lambda_matrix = torch.zeros(targets.shape).cuda()
                for i in range(targets.shape[1]):
                    lambda_matrix[:, i] = weights[i]
        else:
            lambda_matrix = torch.ones(targets.shape).cuda()


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

        # Normalize with the maximum speed from the training set (40 km/h)
        speed_gt = speed_gt / 40.0 # TODO: this we take from the loss function

        # calculate loss for each branch with specific activation
        loss_b1 = ((branches[0] - targets) * controls_b1)  * lambda_matrix
        loss_b2 = ((branches[1] - targets) * controls_b2)  * lambda_matrix
        loss_b3 = ((branches[2] - targets) * controls_b3)  * lambda_matrix
        loss_b4 = ((branches[3] - targets) * controls_b4)  * lambda_matrix
        loss_b5 = (branches[4] - speed_gt)

        # add all branches losses together
        mse_loss = loss_b1 + loss_b2 + loss_b3 + loss_b4


        if reduce:
            if size_average:
                mse_loss = torch.sum(mse_loss)/(mse_loss.shape[0] * mse_loss.shape[1]) + torch.sum(loss_b5)/mse_loss.shape[0]
            else:
                mse_loss = torch.sum(mse_loss) + torch.sum(loss_b5)
        else:
            if size_average:
                raise RuntimeError(" size_average can not be applies when reduce is set to 'False' ")
            else:
                mse_loss =torch.cat([mse_loss,loss_b5],1)

        return mse_loss
