import os
import sys
import random
import time
import traceback
import numpy as np
import torch
import torch.optim as optim
import collections


# What do we define as a parameter what not.

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss, adjust_learning_rate
from network.loss import compute_attention_map_L2, compute_attention_map_L1
from input import CoILDataset, PreSplittedSampler, splitter, Augmenter, RandomSampler
from logger import monitorer, coil_logger
from utils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint
from utils.general import softmax

from torchvision import transforms


def get_attention_vec(tensors, func=torch.abs, layer=0):
    att = tensors[layer]
    att = func(att).sum(1)  # channel pooling
    

# TODO: check a smarter way to do this
def select_data_old(dataset, keys):

    """
    Given a dataset with the float data and a set of keys, get the subset of these keys.
    Args:
        dataset:
        keys:

    Returns:
    """

    # The angle of each datapoint camera.
    print (dataset)
    camera_names =  [dpoint['angle'] for dpoint in dataset]

    if g_conf.DATA_USED == 'central':

        keys = splitter.label_split(camera_names, keys, [[0]])[0]


    elif g_conf.DATA_USED == 'sides':

        keys = splitter.label_split(camera_names, keys, [[-30.0, 30.0]])[0]

    elif g_conf.DATA_USED != 'all':
        raise ValueError(" Invalid data used keyname")


    return keys



def select_data(dataset):

    """
    Given a dataset with the float data and a set of keys, get the subset of these keys.
    Args:
        dataset:
        keys:

    Returns:
    """

    # The angle of each datapoint camera.
    # This can be updated to enable many more configurations

    if g_conf.REMOVE is not None and g_conf.REMOVE is not "None":
        name, params =  parse_remove_configuration(g_conf.REMOVE)
        splitter_function = getattr(splitter, name)

        print(" Function to remove", name)
        print(" params ", params)

        return  splitter_function(dataset, params)
    else:
        return range(0, len(dataset) - g_conf.NUMBER_IMAGES_SEQUENCE)

def parse_split_configuration(configuration):
    """
    Turns the configuration line of sliptting into a name and a set of params.
    """

    if configuration is None:
        return "None", None
    print ('conf', configuration)
    conf_dict = collections.OrderedDict(configuration)

    name = 'split'
    for key in conf_dict.keys():
        if key != 'weights' and key != 'boost':
            name += '_'
            name += key



    return name, conf_dict

# ToDO: Probably this could go to another code module.

def get_inverse_freq_weights(keys, dataset_size):

    invers_freq_weights = []
    print (" frequency")
    for key_vec in keys:
        print ((len(key_vec)/dataset_size))
        invers_freq_weights.append((len(key_vec)/dataset_size))

    return softmax(np.array(invers_freq_weights))








# TODO: for now is not posible to maybe balance just labels or just steering. Is either all or nothing
def select_balancing_strategy(dataset, iteration, number_of_workers):



    # Creates the sampler, this part is responsible for managing the keys. It divides
    # all keys depending on the measurements and produces a set of keys for each bach.

    #keys = select_data(dataset.measurements)   I WILL TEST SELECTING DATA ON OTHER PART
    keys = range(0, len(dataset) - g_conf.NUMBER_IMAGES_SEQUENCE)
    print (" ALL THE KEYS ")
    print (len(keys))

    # In the case we are using the balancing
    print(" Split is ", g_conf.SPLIT)

    if g_conf.SPLIT is not None and g_conf.SPLIT is not "None":
        name, params = parse_split_configuration(g_conf.SPLIT)
        splitter_function = getattr(splitter, name)

        print (" Function to split", name)
        print (" params ", params)
        print (" Weights ", params['weights'])
        keys_splitted = splitter_function(dataset.measurements, params)
        print (keys_splitted)

        for i in range(len(keys_splitted)):

            keys_splitted[i] = np.array(list(set(keys_splitted[i]).intersection(set(keys))))

        print (keys_splitted)
        print (" number of kleys",len(keys_splitted))

        if params['weights'] == 'inverse':
            weights = get_inverse_freq_weights(keys_splitted, len(dataset.measurements) - g_conf.NUMBER_IMAGES_SEQUENCE)
        else:
            weights = params['weights']

        print ( " final weights ")
        print ( weights )
        sampler = PreSplittedSampler(keys_splitted, iteration * g_conf.BATCH_SIZE, weights)
    else:

        print (" Random Splitter ")

        #print(keys)
        sampler = RandomSampler(keys, iteration * g_conf.BATCH_SIZE)


    print ( "Getting dataloader")

    # The data loader is the multi threaded module from pytorch that release a number of
    # workers to get all the data.
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=g_conf.BATCH_SIZE,
                                              sampler=sampler,
                                              num_workers=number_of_workers,
                                              pin_memory=True)



    return data_loader


# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias, suppress_output=True, number_of_workers=12):
    # We set the visible cuda devices


    # TODO: probable race condition, the train has to be started before.
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        g_conf.VARIABLE_WEIGHT = {}
        print("BEFOE MERGET variable ", g_conf.VARIABLE_WEIGHT, 'conf targets',
              g_conf.TARGETS)

        # At this point the log file with the correct naming is created.
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'))
        set_type_of_process('train')

        coil_logger.add_message('Loading', {'GPU': gpu})

        if not os.path.exists('_output_logs'):
            os.mkdir('_output_logs')

        print("variable ", g_conf.VARIABLE_WEIGHT, 'conf targets',
              g_conf.TARGETS)

        # Put the output to a separate file
        if suppress_output:
            sys.stdout = open(os.path.join('_output_logs', exp_alias + '_' +
                              g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"), "a", buffering=1)
            sys.stderr = open(os.path.join('_output_logs',
                              exp_alias + '_err_'+g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"),
                              "a", buffering=1)


        checkpoint_file = get_latest_saved_checkpoint()
        print ( " LOADING  ", checkpoint_file)
        if checkpoint_file is not None:
            checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias,
                                     'checkpoints', str(get_latest_saved_checkpoint())))
            iteration = checkpoint['iteration']
            best_loss = checkpoint['best_loss']
            best_loss_iter = checkpoint['best_loss_iter']

        else:
            iteration = 0
            best_loss = 10000.0
            best_loss_iter = 0

        # TODO: The checkpoint will continue, so it should erase everything up to the iteration on tensorboard
        # Define the dataset. This structure is has the __get_item__ redefined in a way
        # that you can access the HD_FILES positions from the root directory as a in a vector.
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME)

        # augmenter_cpu = iag.AugmenterCPU(g_conf.AUGMENTATION_SUITE_CPU)

        # By instanciating the augmenter we get a callable that augment images and transform them
        # into tensors.
        augmenter = Augmenter(g_conf.AUGMENTATION)

        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=str(g_conf.NUMBER_OF_HOURS) + 'hours_' + g_conf.TRAIN_DATASET_NAME)


        data_loader = select_balancing_strategy(dataset, iteration, number_of_workers)


        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        model.cuda()

        if checkpoint_file is not None:
            model.load_state_dict(checkpoint['state_dict'])

        print(model)

        criterion = Loss(g_conf.LOSS_FUNCTION)

        optimizer = optim.Adam(model.parameters(), lr=g_conf.LEARNING_RATE)


        if checkpoint_file is not None:
            accumulated_time = checkpoint['total_time']
        else:
            accumulated_time = 0  # We accumulate iteration time and keep the average speed


        #TODO: test experiment continuation. Is the data sampler going to continue were it started.. ?
        capture_time = time.time()
        for data in data_loader:
            print ("READ TIME ", time.time() - capture_time)
            # get the control commands from float_data, size = [120,1]

            capture_time = time.time()
            controls = data['directions']



            # The output(branches) is a list of 5 branches results, each branch is with size [120,3]

            model.zero_grad()
            branches = model(torch.squeeze(data['rgb'].cuda()),
                             dataset.extract_inputs(data).cuda())

            # Make use of attention more general.



            #TODO: This requires some cleaning, there is two selection points for the loss
            if 'attention' in g_conf.LOSS_FUNCTION or 'regularization' in g_conf.LOSS_FUNCTION:
                inter_layers = [model.intermediate_layers[ula] for ula in g_conf.USED_LAYERS_ATT]
                loss, loss_L1, loss_L2 = criterion(branches, dataset.extract_targets(data).cuda(),
                                 controls.cuda(), dataset.extract_inputs(data).cuda(),
                                 branch_weights=g_conf.BRANCH_LOSS_WEIGHT,
                                 variable_weights=g_conf.VARIABLE_WEIGHT,
                                 inter_layers=inter_layers,
                                 intention_factors=dataset.extract_intentions(data).cuda())

                coil_logger.add_scalar('L1', loss_L1.data, iteration)
                coil_logger.add_scalar('L2', loss_L2.data, iteration)

                count = 0
                for il in inter_layers:
                    coil_logger.add_image('Attention L1 ' + str(g_conf.USED_LAYERS_ATT[count]),
                                          compute_attention_map_L1(il).unsqueeze(1), iteration)
                    coil_logger.add_image('Attention L2 ' + str(g_conf.USED_LAYERS_ATT[count]),
                                          compute_attention_map_L2(il).unsqueeze(1), iteration)
                    count += 1

            else:
                loss = criterion(branches, dataset.extract_targets(data).cuda(),
                                 controls.cuda(), dataset.extract_inputs(data).cuda(),
                                 branch_weights=g_conf.BRANCH_LOSS_WEIGHT,
                                 variable_weights=g_conf.VARIABLE_WEIGHT)




            # TODO: All these logging things could go out to clean up the main
            if loss.data < best_loss:
                best_loss = loss.data.tolist()
                best_loss_iter = iteration

            # Log a random position
            position = random.randint(0, len(data)-1)

            output = model.extract_branch(torch.stack(branches[0:4]), controls)
            error = torch.abs(output - dataset.extract_targets(data).cuda())




            # TODO: For now we are computing the error for just the correct branch, it could be multi- branch,
            print (" The produced loss")

            coil_logger.add_scalar('Loss', loss.data, iteration)
            print ("RGB")
            coil_logger.add_image('Image', torch.squeeze(data['rgb']), iteration)






            loss.backward()
            optimizer.step()

            print ("Inference + opt + TIME ", time.time() - capture_time)
            accumulated_time += time.time() - capture_time
            capture_time = time.time()


            # TODO: Get only the  float_data that are actually generating output
            # TODO: itearation is repeating , and that is dumb
            coil_logger.add_message('Iterating',
                                    {'Iteration': iteration,
                                     'Loss': loss.data.tolist(),
                                     'Images/s': (iteration*g_conf.BATCH_SIZE)/accumulated_time,
                                     'BestLoss': best_loss, 'BestLossIteration': best_loss_iter,
                                     'Output': output[position].data.tolist(),
                                     'GroundTruth': dataset.extract_targets(data)[position].data.tolist(),
                                     'Error': error[position].data.tolist(),
                                     'Inputs': dataset.extract_inputs(data)[position].data.tolist()},
                                    iteration)

            # TODO: For now we are computing the error for just the correct branch, it could be multi-branch,


            # TODO: save also the optimizer state dictionary
            if is_ready_to_save(iteration):

                state = {
                    'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'total_time': accumulated_time,
                    'best_loss_iter': best_loss_iter

                }
                # TODO : maybe already summarize the best model ???
                torch.save(state, os.path.join('_logs', exp_batch, exp_alias
                                               , 'checkpoints', str(iteration) + '.pth'))

            iteration += 1
            print (iteration)

            if iteration % 1000 == 0:
                adjust_learning_rate(optimizer, iteration)

            del data

            print ("The REST", time.time() - capture_time)
            capture_time = time.time()

        coil_logger.add_message('Finished', {})




    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})


