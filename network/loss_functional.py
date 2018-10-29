from torch.nn import functional as F
import torch


def normalize(x, dim):
    x_normed = x / x.max(dim, keepdim=True)[0]
    return x_normed


def compute_attention_map_l2(il):
    """
    THis compute the attention map that is actually viewable for L2

    """
    L2 = torch.pow(il, 2)
    L2 = L2.mean(1)  # channel pooling
    max_value, _ = torch.max(L2.view(L2.shape[0], -1), 1, keepdim=True)
    max_value = max_value.view(-1, 1, 1)
    # print (" max L2 ", max_value.mean())
    L2 = torch.div(L2, max_value)

    return L2


def compute_attention_map_l1(il):
    """
    THis compute the attention map that is actually viewable for L1

    """

    L1 = il.mean(1)
    l1_max_value, _ = torch.max(L1.view(L1.shape[0], -1), 1, keepdim=True)
    l1_max_value = l1_max_value.view(-1, 1, 1)
    L1 = torch.div(L1, l1_max_value)


    return L1


def compute_attention_loss(inter_layers, variable_weights, intention_factors):

    """ Take the batch size from the number of channels on the attention maps"""

    loss = torch.zeros([intention_factors.shape[0]], dtype=torch.float32).cuda()

    intention, _ = torch.min(intention_factors, 1)
    intention = (1. > intention).float()

    count = 0
    for il in inter_layers:
        """ We compute the square ( L2) for each of the maps and them take the mean"""
        L2 = compute_attention_map_l2(il)
        L2 = F.avg_pool2d(L2, variable_weights['AVGP_Kernel_Size'],
                          padding=int(variable_weights['AVGP_Kernel_Size']/2))
        L2 = L2.mean(1).mean(1)

        """ We compute the square (L1) for each of the maps and them take the mean"""
        L1 = compute_attention_map_l1(il)
        L1 = F.avg_pool2d(L1, variable_weights['AVGP_Kernel_Size'],
                          padding=int(variable_weights['AVGP_Kernel_Size']/2))
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

def weight_decay_l1(loss, model, intention_factors, alpha, gating):

    wdecay = 0
    for w in model.parameters():
        if w.requires_grad:
            wdecay = torch.add(torch.sum(torch.abs(w)), wdecay)

    if intention_factors is not None:

        intention, _ = torch.min(intention_factors, 1)
        intention = (1. > intention).float()
        if gating == 'hard':
            # Multiply by a factor proportional to the size of the number of non 1
            wdecay = wdecay * intention.shape[0]/torch.sum(intention)

        elif gating == 'easy':
            wdecay = wdecay * torch.sum(intention)/intention.shape[0]

    loss = torch.add(loss, alpha * wdecay)
    return loss


def weight_decay_l2(loss, model, intention_factors, alpha, gating):

    wdecay = 0
    for w in model.parameters():
        if w.requires_grad:
            wdecay = torch.add(torch.sum(w**2), wdecay)

    if intention_factors is not None:

        intention, _ = torch.min(intention_factors, 1)
        intention = (1. > intention).float()
        if gating == 'hard':
            # Multiply by a factor proportional to the size of the number of non 1
            wdecay = wdecay * intention.shape[0]/torch.sum(intention)

        elif gating == 'easy':
            wdecay = wdecay * torch.sum(intention)/intention.shape[0]

    loss = torch.add(loss, alpha * wdecay)
    return loss

def compute_branches_masks(controls, number_targets):
    """
        Args
            controls
            the control values that have the following structure
            command flags: 2 - follow lane; 3 - turn left; 4 - turn right; 5 - go straight
            size of targets:
            How many targets is produced by the network so we can produce the masks properly
        Returns
            a mask to have the loss function applied
            only on over the correct branch.
    """

    """ A vector with a mask for each of the control branches"""
    controls_masks = []

    # when command = 2, branch 1 (follow lane) is activated
    controls_b1 = (controls == 2)
    controls_b1 = torch.tensor(controls_b1, dtype=torch.float32).cuda()
    controls_b1 = torch.cat([controls_b1] * number_targets, 1)
    controls_masks.append(controls_b1)
    # when command = 3, branch 2 (turn left) is activated
    controls_b2 = (controls == 3)
    controls_b2 = torch.tensor(controls_b2, dtype=torch.float32).cuda()
    controls_b2 = torch.cat([controls_b2] * number_targets, 1)
    controls_masks.append(controls_b2)
    # when command = 4, branch 3 (turn right) is activated
    controls_b3 = (controls == 4)
    controls_b3 = torch.tensor(controls_b3, dtype=torch.float32).cuda()
    controls_b3 = torch.cat([controls_b3] * number_targets, 1)
    controls_masks.append(controls_b3)
    # when command = 5, branch 4 (go strange) is activated
    controls_b4 = (controls == 5)
    controls_b4 = torch.tensor(controls_b4, dtype=torch.float32).cuda()
    controls_b4 = torch.cat([controls_b4] * number_targets, 1)
    controls_masks.append(controls_b4)


    return controls_masks

def l2_loss(params):
    """
        Functional LOSS L2
        Args
            params dictionary that should include:
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls_mask: the masked already expliciting the branches tha are going to be used
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points

        Returns
            A vector with the loss function

    """
    """ It is a vec for each branch"""
    loss_branches_vec = []
    # TODO This is hardcoded but all our cases rigth now uses four branches
    for i in range(len(params['branches']) -1):
        loss_branches_vec.append(((params['branches'][i] - params['targets']) **2
                                           * params['controls_mask'][i])
                                 * params['branch_weights'][i])
    """ The last branch is a speed branch"""
    # TODO: Activate or deactivate speed branch loss
    loss_branches_vec.append((params['branches'][-1] - params['inputs']) ** 2
                             * params['branch_weights'][-1])
    return loss_branches_vec, {}


def l1_loss(params):
    """
        Functional LOSS L1
        Args
            params dictionary that should include:
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls_mask: the masked already expliciting the branches tha are going to be used
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points

        Returns
            A vector with the loss function

    """
    """ It is a vec for each branch"""
    loss_branches_vec = []
    # TODO This is hardcoded but all our cases rigth now uses four branches
    for i in range(len(params['branches']) -1):
        loss_branches_vec.append(torch.abs((params['branches'][i] - params['targets'])
                                           * params['controls_mask'][i])
                                 * params['branch_weights'][i])
    """ The last branch is a speed branch"""
    # TODO: Activate or deactivate speed branch loss
    loss_branches_vec.append(torch.abs(params['branches'][-1] - params['inputs'])
                             * params['branch_weights'][-1])
    return loss_branches_vec, {}


def l1_attention_loss(params):
    """
        Functional LOSS L1 attention
        Args
            params dictionary that should include:
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls_mask: the masked already expliciting the branches tha are going to be used
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points
                inter_layers: The intermediate layers used to compute the attention
                intention_factors: The factors used to compute to weight the attention used.


        Returns
            A vector with the loss function
            a dictionary with all the intermediary values that are plotable, for this case the
            L1 And L2 computed attention

    """
    if 'inter_layers' not in params:
        raise ValueError(" Missing Intermediate layer (inter_layers) Parameters ")
    if 'variable_weights' not in params:
        raise ValueError(" Missing Variable Weights (variable_weights) Parameters ")
    if 'intention_factors' not in params:
        raise ValueError(" Missing Intention Factors (intention_factors) Parameters ")

    """ It is a vec for each branch"""
    loss_branches_vec = []

    # TODO This is hardcoded but all our cases rigth now uses four branches
    for i in range(len(params['branches']) -1):
        loss_branches_vec.append(torch.abs((params['branches'][i] - params['targets'])
                                           * params['controls_mask'][i])
                                 * params['branch_weights'][i])
    """ The last branch is a speed branch"""

    att_loss, l1, l2 = compute_attention_loss(params['inter_layers'],
                                              params['variable_weights'],
                                              params['intention_factors'])
    loss_branches_vec.append(att_loss)

    # We pre process the plotable params to make them plotable
    l1 = torch.sum(l1) / (l1.shape[0])
    l2 = torch.sum(l2) / (l2.shape[0])
    plotable_params = {'L1': l1, 'L2': l2}

    # TODO: Activate or deactivate speed branch loss
    loss_branches_vec.append(torch.abs(params['branches'][-1] - params['inputs'])
                             * params['branch_weights'][-1])
    return loss_branches_vec, plotable_params

def l1_ground_truth_attention_loss(params):
    """
        Functional LOSS L1 attention
        Args
            params dictionary that should include:
                branches: The tensor containing all the branches branches output from the network
                targets: The ground truth targets that the network should produce
                controls_mask: the masked already expliciting the branches tha are going to be used
                branches weights: the weigths that each branch will have on the loss function
                speed_gt: the ground truth speed for these data points
                inter_layers: The intermediate layers used to compute the attention
                intention_factors: The factors used to compute to weight the attention used.


        Returns
            A vector with the loss function
            a dictionary with all the intermediary values that are plotable, for this case the
            L1 And L2 computed attention

    """
    if 'inter_layers' not in params:
        raise ValueError(" Missing Intermediate layer (inter_layers) Parameters ")
    if 'variable_weights' not in params:
        raise ValueError(" Missing Variable Weights (variable_weights) Parameters ")
    if 'intention_factors' not in params:
        raise ValueError(" Missing Intention Factors (intention_factors) Parameters ")

    """ It is a vec for each branch"""
    loss_branches_vec = []

    # TODO This is hardcoded but all our cases rigth now uses four branches
    for i in range(len(params['branches']) -1):

        print ((torch.abs((params['branches'][i] - params['targets'])
                                           * params['controls_mask'][i])
                                 * params['branch_weights'][i]).shape)
        loss_branches_vec.append(torch.abs((params['branches'][i] - params['targets'])
                                           * params['controls_mask'][i])
                                 * params['branch_weights'][i])
    """ The last branch is a speed branch"""

    att_loss, l1, l2 = compute_attention_loss(params['inter_layers'],
                                              params['variable_weights'],
                                              params['intention_factors'])
    loss_branches_vec.append(att_loss)

    # We pre process the plotable params to make them plotable
    l1 = torch.sum(l1) / (l1.shape[0])
    l2 = torch.sum(l2) / (l2.shape[0])
    plotable_params = {'L1': l1, 'L2': l2}

    # TODO: Activate or deactivate speed branch loss
    loss_branches_vec.append(torch.abs(params['branches'][-1] - params['inputs'])
                             * params['branch_weights'][-1])
    return loss_branches_vec, plotable_params
