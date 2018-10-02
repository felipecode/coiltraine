from logger import coil_logger
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
import torch
from configs import g_conf


def normalize(x, dim):
    x_normed = x / x.max(dim, keepdim=True)[0]
    return x_normed

# TODO: needs some severe refactor to avoid hardcoding and repetition

def compute_attention_map_L2(il):

    """
    THis compute the attention map that is actually viewable for L2
    :param il:
    :return:
    """

    L2 = torch.pow(il, 2)
    L2 = L2.mean(1)  # channel pooling
    max_value = torch.max(L2.view(L2.shape[0], -1))
    print (" max L2 ", max_value.mean())
    L2 = torch.div(L2, max_value)

    return L2

def compute_attention_map_L1(il):
    """
    THis compute the attention map that is actually viewable for L1
    :param il:
    :return:
    """

    L1 = il.mean(1)
    l1_max_value = torch.max(L1.view(L1.shape[0], -1))
    print (" max L1 ", l1_max_value.mean())
    L1 = torch.div(L1, l1_max_value)

    return L1

def compute_attention_loss(inter_layers, variable_weights, intention_factors):

    """ Take the batch size from the number of channels on the attention maps"""
    print (inter_layers[0].shape)
    print (intention_factors.shape)
    loss = torch.zeros([intention_factors.shape[0]], dtype=torch.float32).cuda()

    intention, _ = torch.min(intention_factors, 1)
    intention = (1. > intention).float()
    print (loss.shape)
    print (intention.shape)

    count = 0
    for il in inter_layers:
        """ We compute the square ( L2) for each of the maps and them take the mean"""
        L2 = compute_attention_map_L2(il)
        L2 = F.avg_pool2d(L2, variable_weights['AVGP_Kernel_Size'], 1)
        L2 = L2.mean(1).mean(1)

        """ We compute the square (L1) for each of the maps and them take the mean"""
        L1 = compute_attention_map_L1(il)
        L1 = F.avg_pool2d(L1, variable_weights['AVGP_Kernel_Size'], 1)
        L1 = L1.mean(1).mean(1)

        #print (" atention ", count)
        #print (" intention ", intention)
        #print (" L1 ", L1.shape)
        #print (" L2", L2.shape)
        """ We take the measurements used as attention important and weight"""
        # This part should have dimension second dimension 1
        loss += (variable_weights['L2']*L2 * intention + variable_weights['L1']*L1*(1-intention))\
                    / len(inter_layers)

        print (" Partial Loss ", loss)


    return loss, L1, L2



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

    print ( "variable ", variable_weights, ' Targets ', targets.shape[1], 'conf targets',  g_conf.TARGETS)
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
    controls_b2 = torch.cat([controls_b2] * len(g_conf.TARGETS), 1)
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


def L1_attention(branches, targets, controls, speed_gt, inter_layers =None, intention_factors=None, size_average=True,
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
    if inter_layers is None or intention_factors is None:
        raise ValueError(" L1 atttention requires inter_layers and interest factor to be passed")

    # weight different target items with lambdas
    # if variable_weights:
    #     if len(variable_weights) != targets.shape[1]:
    #         raise ValueError('The input number of weight lambdas is '
    #                          + str(len(branch_weights)) +
    #                          ', while the number of branches items is '
    #                          + str(targets.shape[1]))
    # else:

    #     variable_weights = {'Gas': 1.0, 'Steer': 1.0, 'Brake': 1.0}

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

    loss, L1, L2 = compute_attention_loss(inter_layers, variable_weights, intention_factors)
    mse_loss = loss_b1 + loss_b2 + loss_b3 + loss_b4 + loss




    if reduce:
        if size_average:
            mse_loss = torch.sum(mse_loss) / (mse_loss.shape[0]) \
                       + torch.sum(loss_b5) / mse_loss.shape[0]

            L1 = torch.sum(L1) / (L1.shape[0])
            L2 = torch.sum(L2) / (L2.shape[0])

        else:
            mse_loss = torch.sum(mse_loss) + torch.sum(loss_b5)
    else:
        if size_average:
            raise RuntimeError(
                " size_average can not be applies when reduce is set to 'False' ")
        else:
            mse_loss = torch.cat([mse_loss, loss_b5], 1)


    #L1 = 0
    #L2 = 0
    return mse_loss, L1, L2


# TODO: CLEAN THIS, PRE DEADLINE HARD CODE !

def L1_no_brake(branches, targets, controls, speed_gt, size_average=True,
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
    controls_b1 = torch.cat([controls_b1, controls_b1], 1)  # activation for steer, gas and brake
    # when command = 3, branch 2 (turn left) is activated
    controls_b2 = (controls == 3)
    controls_b2 = torch.tensor(controls_b2, dtype=torch.float32).cuda()
    controls_b2 = torch.cat([controls_b2, controls_b2], 1)
    # when command = 4, branch 3 (turn right) is activated
    controls_b3 = (controls == 4)
    controls_b3 = torch.tensor(controls_b3, dtype=torch.float32).cuda()
    controls_b3 = torch.cat([controls_b3, controls_b3], 1)
    # when command = 5, branch 4 (go strange) is activated
    controls_b4 = (controls == 5)
    controls_b4 = torch.tensor(controls_b4, dtype=torch.float32).cuda()
    controls_b4 = torch.cat([controls_b4, controls_b4], 1)

    # calculate loss for each branch with specific activation
    loss_b1 = torch.abs((branches[0] - targets) * controls_b1) * branch_weights[0]
    loss_b2 = torch.abs((branches[1] - targets) * controls_b2) * branch_weights[1]
    loss_b3 = torch.abs((branches[2] - targets) * controls_b3) * branch_weights[2]
    loss_b4 = torch.abs((branches[3] - targets) * controls_b4) * branch_weights[3]
    loss_b5 = torch.abs(branches[4] - speed_gt) * branch_weights[4]

    # Apply the variable weights
    loss_b1 = loss_b1[:, 0] * variable_weights['Steer'] + loss_b1[:, 1] * variable_weights[
        'Gas']
    loss_b2 = loss_b2[:, 0] * variable_weights['Steer'] + loss_b2[:, 1] * variable_weights[
        'Gas']
    loss_b3 = loss_b3[:, 0] * variable_weights['Steer'] + loss_b3[:, 1] * variable_weights[
        'Gas']
    loss_b4 = loss_b4[:, 0] * variable_weights['Steer'] + loss_b4[:, 1] * variable_weights[
        'Gas']
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






def L3(branches, targets, controls, speed_gt, size_average=True,
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

    print ( "variable ", variable_weights, ' Targets ', targets.shape[1], 'conf targets',  g_conf.TARGETS)
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
    controls_b2 = torch.cat([controls_b2] * len(g_conf.TARGETS), 1)
    # when command = 4, branch 3 (turn right) is activated
    controls_b3 = (controls == 4)
    controls_b3 = torch.tensor(controls_b3, dtype=torch.float32).cuda()
    controls_b3 = torch.cat([controls_b3] * len(g_conf.TARGETS), 1)
    # when command = 5, branch 4 (go strange) is activated
    controls_b4 = (controls == 5)
    controls_b4 = torch.tensor(controls_b4, dtype=torch.float32).cuda()
    controls_b4 = torch.cat([controls_b4] * len(g_conf.TARGETS), 1)

    # calculate loss for each branch with specific activation
    loss_b1 = torch.abs(((branches[0] - targets) * controls_b1) ** 3) * branch_weights[0]
    loss_b2 = torch.abs(((branches[1] - targets) * controls_b2) ** 3) * branch_weights[1]
    loss_b3 = torch.abs(((branches[2] - targets) * controls_b3) ** 3) * branch_weights[2]
    loss_b4 = torch.abs(((branches[3] - targets) * controls_b4) ** 3) * branch_weights[3]
    loss_b5 = torch.abs((branches[4] - speed_gt) ** 3) * branch_weights[4]

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


def L1_regularization(branches, targets, controls, speed_gt, inter_layers =None, intention_factors=None, size_average=True,
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
    intention_factors = 1. + 0. * intention_factors
    if inter_layers is None or intention_factors is None:
        raise ValueError(" L1 atttention requires inter_layers and interest factor to be passed")

    # weight different target items with lambdas
    # if variable_weights:
    #     if len(variable_weights) != targets.shape[1]:
    #         raise ValueError('The input number of weight lambdas is '
    #                          + str(len(branch_weights)) +
    #                          ', while the number of branches items is '
    #                          + str(targets.shape[1]))
    # else:

    #     variable_weights = {'Gas': 1.0, 'Steer': 1.0, 'Brake': 1.0}

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

    loss, L1, L2 = compute_attention_loss(inter_layers, variable_weights, intention_factors)
    mse_loss = loss_b1 + loss_b2 + loss_b3 + loss_b4 + loss

    if reduce:
        if size_average:
            mse_loss = torch.sum(mse_loss) / (mse_loss.shape[0]) \
                       + torch.sum(loss_b5) / mse_loss.shape[0]

            L1 = torch.sum(L1) / (L1.shape[0])
            L2 = torch.sum(L2) / (L2.shape[0])

        else:
            mse_loss = torch.sum(mse_loss) + torch.sum(loss_b5)
    else:
        if size_average:
            raise RuntimeError(
                " size_average can not be applies when reduce is set to 'False' ")
        else:
            mse_loss = torch.cat([mse_loss, loss_b5], 1)


    #L1 = 0
    #L2 = 0
    return mse_loss, L1, L2


def Loss(loss_name):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """
    # TODO: this could be extended to some more arbitrary definition

    if loss_name == 'L1':

        return L1
    elif loss_name == 'L1_no_brake':

        return L1_no_brake

    elif loss_name == 'L2':

        return L2
    elif loss_name == 'L1_attention':

        return L1_attention
    elif loss_name == 'L3':

        return L3

    elif loss_name == 'L1_regularization':
        return L1_regularization

    else:
        raise ValueError(" Not found Loss name")


