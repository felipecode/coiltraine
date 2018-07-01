from logger import coil_logger
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
import torch
from configs import g_conf


# TODO: needs some severe refactor to avoid hardcoding and repetition

def L2(branches, targets, controls, speed_gt, size_average=True,
            reduce=True, variable_weights=None, branch_weights=None):
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
    if variable_weights:
        if len(variable_weights) != targets.shape[1]:
            raise ValueError('The input number of weight lambdas is '
                             + str(len(variable_weights)) +
                             ', while the number of branches items is '
                             + str(targets.shape[1]))
    else:

        variable_weights = {'Gas': 1.0, 'Steer': 1.0, 'Brake': 1.0}

    if branch_weights:
        if len(branch_weights) != len(branches):
            raise ValueError('The input number of branch weight lambdas is '
                             + str(len(branch_weights)) +
                             ', while the number of branches items is '
                             + str(len(branches)))


    else:
        branch_weights = [1.0] * len(branches)

    # TODO: improve this code quality
    # command flags: 2 - follow lane; 3 - turn left; 4 - turn right; 5 - go strange

    # when command = 2, branch 1 (follow lane) is activated
    controls_b1 = (controls == 2)
    controls_b1 = torch.tensor(controls_b1, dtype=torch.float32).cuda()
    controls_b1 = torch.cat([controls_b1] * len(g_conf.TARGETS), 1)

    # when command = 3, branch 2 (turn left) is activated
    controls_b2 = (controls == 3)
    controls_b2 = torch.tensor(controls_b2, dtype=torch.float32).cuda()
    controls_b2 = torch.cat([controls_b2] *  len(g_conf.TARGETS), 1)
    # when command = 4, branch 3 (turn right) is activated
    controls_b3 = (controls == 4)
    controls_b3 = torch.tensor(controls_b3, dtype=torch.float32).cuda()
    controls_b3 = torch.cat([controls_b3] * len(g_conf.TARGETS), 1)
    # when command = 5, branch 4 (go strange) is activated
    controls_b4 = (controls == 5)
    controls_b4 = torch.tensor(controls_b4, dtype=torch.float32).cuda()
    controls_b4 = torch.cat([controls_b4] * len(g_conf.TARGETS), 1)

    # calculate loss for each branch with specific activation
    loss_b1 = ((branches[0] - targets) * controls_b1) ** 2 * branch_weights[0]
    loss_b2 = ((branches[1] - targets) * controls_b2) ** 2 * branch_weights[1]
    loss_b3 = ((branches[2] - targets) * controls_b3) ** 2 * branch_weights[2]
    loss_b4 = ((branches[3] - targets) * controls_b4) ** 2 * branch_weights[3]
    loss_b5 = (branches[4] - speed_gt) ** 2 * branch_weights[4]

    # Apply the variable weights
    # TODO; the variable and its weigths should be sincronized in the same variable.
    # TODO: very dangerous part. Instead of indexing it should use variable names
    if 'W1A' in variable_weights:   # TODO: FIX this hardcodedism
        loss_b1 = loss_b1[:, 0] * variable_weights['Steer'] + loss_b1[:, 1] * variable_weights['Gas'] \
                  + loss_b1[:, 2] * variable_weights['Brake'] \
                  + loss_b1[:, 3] * variable_weights['W1A'] \
                  + loss_b1[:, 4] * variable_weights['W2A']
        loss_b2 = loss_b2[:, 0] * variable_weights['Steer'] + loss_b2[:, 1] * variable_weights['Gas'] \
                  + loss_b2[:, 2] * variable_weights['Brake'] \
                  + loss_b2[:, 3] * variable_weights['W1A'] \
                  + loss_b2[:, 4] * variable_weights['W2A']

        loss_b3 = loss_b3[:, 0] * variable_weights['Steer'] + loss_b3[:, 1] * variable_weights['Gas'] \
                  + loss_b3[:, 2] * variable_weights['Brake'] \
                  + loss_b3[:, 3] * variable_weights['W1A'] \
                  + loss_b3[:, 4] * variable_weights['W2A']

        loss_b4 = loss_b4[:, 0] * variable_weights['Steer'] + loss_b4[:, 1] * variable_weights['Gas'] \
                  + loss_b4[:, 2] * variable_weights['Brake'] \
                  + loss_b4[:, 3] * variable_weights['W1A'] \
                  + loss_b4[:, 4] * variable_weights['W2A']
    else:
        loss_b1 = loss_b1[:, 0] * variable_weights['Steer'] + loss_b1[:, 1] * variable_weights['Gas'] \
                  + loss_b1[:, 2] * variable_weights['Brake']
        loss_b2 = loss_b2[:, 0] * variable_weights['Steer'] + loss_b2[:, 1] * variable_weights['Gas'] \
                  + loss_b2[:, 2] * variable_weights['Brake']
        loss_b3 = loss_b3[:, 0] * variable_weights['Steer'] + loss_b3[:, 1] * variable_weights['Gas'] \
                  + loss_b3[:, 2] * variable_weights['Brake']
        loss_b4 = loss_b4[:, 0] * variable_weights['Steer'] + loss_b4[:, 1] * variable_weights['Gas'] \
                  + loss_b4[:, 2] * variable_weights['Brake']
    # add all branches losses together
    mse_loss = loss_b1 + loss_b2 + loss_b3 + loss_b4

    if reduce:
        if size_average:
            mse_loss = torch.sum(mse_loss) / (mse_loss.shape[0]) \
                       + torch.sum(loss_b5) / mse_loss.shape[0]
        else:
            mse_loss = torch.sum(mse_loss) + torch.sum(loss_b5)
    else:
        if size_average:
            raise RuntimeError(" size_average can not be applies when reduce is set to 'False' ")
        else:
            mse_loss = torch.cat([mse_loss, loss_b5], 1)

    return mse_loss

def L1(branches, targets, controls, speed_gt, size_average=True,
       reduce=True, variable_weights=None, branch_weights=None):
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
    if variable_weights:
        if len(variable_weights) != targets.shape[1]:
            raise ValueError('The input number of weight lambdas is '
                             + str(len(branch_weights)) +
                             ', while the number of branches items is '
                             + str(targets.shape[1]))
    else:

        variable_weights = {'Gas': 1.0, 'Steer': 1.0, 'Brake': 1.0}

    if branch_weights:
        if len(branch_weights) != len(branches):
            raise ValueError('The input number of branch weight lambdas is '
                             + str(len(branch_weights)) +
                             ', while the number of branches items is '
                             + str(len(branches)))


    else:
        branch_weights = [1.0] * len(branches)

    # TODO: improve this code quality
    # command flags: 2 - follow lane; 3 - turn left; 4 - turn right; 5 - go strange

    # when command = 2, branch 1 (follow lane) is activated
    controls_b1 = (controls == 2)
    controls_b1 = torch.tensor(controls_b1, dtype=torch.float32).cuda()
    controls_b1 = torch.cat([controls_b1, controls_b1, controls_b1],
                            1)  # activation for steer, gas and brake
    # when command = 3, branch 2 (turn left) is activated
    controls_b2 = (controls == 3)
    controls_b2 = torch.tensor(controls_b2, dtype=torch.float32).cuda()
    controls_b2 = torch.cat([controls_b2, controls_b2, controls_b2], 1)
    # when command = 4, branch 3 (turn right) is activated
    controls_b3 = (controls == 4)
    controls_b3 = torch.tensor(controls_b3, dtype=torch.float32).cuda()
    controls_b3 = torch.cat([controls_b3, controls_b3, controls_b3], 1)
    # when command = 5, branch 4 (go strange) is activated
    controls_b4 = (controls == 5)
    controls_b4 = torch.tensor(controls_b4, dtype=torch.float32).cuda()
    controls_b4 = torch.cat([controls_b4, controls_b4, controls_b4], 1)

    # calculate loss for each branch with specific activation
    loss_b1 = torch.abs((branches[0] - targets) * controls_b1) * branch_weights[0]
    loss_b2 = torch.abs((branches[1] - targets) * controls_b2) * branch_weights[1]
    loss_b3 = torch.abs((branches[2] - targets) * controls_b3) * branch_weights[2]
    loss_b4 = torch.abs((branches[3] - targets) * controls_b4) * branch_weights[3]
    loss_b5 = torch.abs(branches[4] - speed_gt) * branch_weights[4]

    # Apply the variable weights
    loss_b1 = loss_b1[:, 0] * variable_weights['Steer'] + loss_b1[:, 1] * variable_weights[
        'Gas'] \
              + loss_b1[:, 2] * variable_weights['Brake']
    loss_b2 = loss_b2[:, 0] * variable_weights['Steer'] + loss_b2[:, 1] * variable_weights[
        'Gas'] \
              + loss_b2[:, 2] * variable_weights['Brake']
    loss_b3 = loss_b3[:, 0] * variable_weights['Steer'] + loss_b3[:, 1] * variable_weights[
        'Gas'] \
              + loss_b3[:, 2] * variable_weights['Brake']
    loss_b4 = loss_b4[:, 0] * variable_weights['Steer'] + loss_b4[:, 1] * variable_weights[
        'Gas'] \
              + loss_b4[:, 2] * variable_weights['Brake']
    # add all branches losses together
    mse_loss = loss_b1 + loss_b2 + loss_b3 + loss_b4

    if reduce:
        if size_average:
            mse_loss = torch.sum(mse_loss) / (mse_loss.shape[0]) \
                       + torch.sum(loss_b5) / mse_loss.shape[0]
        else:
            mse_loss = torch.sum(mse_loss) + torch.sum(loss_b5)
    else:
        if size_average:
            raise RuntimeError(
                " size_average can not be applies when reduce is set to 'False' ")
        else:
            mse_loss = torch.cat([mse_loss, loss_b5], 1)

    return mse_loss


def Loss(loss_name):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """
    # TODO: this could be extended to some more arbitrary definition

    if loss_name == 'L1':

        return L1

    elif loss_name == 'L2':

        return L2
    else:

        raise ValueError(" Not found architecture name")
