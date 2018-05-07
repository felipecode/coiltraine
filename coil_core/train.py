import os

import torch
import torch.optim as optim
import imgauggpu as iag
# What do we define as a parameter what not.

from configs import g_conf
from network import CoILModel, Loss
from input import CoILDataset, CoILSampler, splitter
from logger import monitorer
from utils.checkpoint_schedule import is_iteration_for_saving, get_latest_saved_checkpoint
from torchvision import transforms


# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias):
    # We set the visible cuda devices
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # At this point the log file with the correct naming is created.
    g_conf.merge_with_yaml(os.path.join(exp_batch, exp_alias+'.yaml'))
    g_conf.set_type_of_process('train')



    if monitorer.get_status(exp_batch, exp_alias, g_conf.param.PROCESS_NAME)[0] == "Finished":
        # TODO: print some cool summary or not ?
        return

    #Define the dataset. This structure is has the __get_item__ redefined in a way
    #that you can access the HDFILES positions from the root directory as a in a vector.
    full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.param.INPUT.DATASET_NAME)

    dataset = CoILDataset(full_dataset, transform=transforms.Compose([transforms.ToTensor()]))

    # Creates the sampler, this part is responsible for managing the keys. It divides
    # all keys depending on the measurements and produces a set of keys for each bach.
    sampler = CoILSampler(splitter.control_steer_split(dataset.measurements, dataset.meta_data))

    # The data loader is the multi threaded module from pytorch that release a number of
    # workers to get all the data.
    # TODO: batch size an number of workers go to some configuration file
    data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=120,
                                              shuffle=False, num_workers=12, pin_memory=True)
    # By instanciating the augmenter we get a callable that augment images and transform them
    # into tensors.
    augmenter = iag.Augmenter(g_conf.param.INPUT.AUGMENTATION_SUITE)

    # TODO: here there is clearly a posibility to make a cool "conditioning" system.
    model = CoILModel(g_conf.param.NETWORK.MODEL_DEFINITION)

    criterion = Loss()

    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    #TODO: Probably there is more differences between train and validation that justify a new file.

    checkpoint_file = get_latest_saved_checkpoint()
    if checkpoint_file != None:
        checkpoint = torch.load(get_latest_saved_checkpoint())
        iteration = checkpoint['iteration']
    else:
        iteration = 0

    # TODO: The checkpoint will continue, so the logs should restart ??? OR continue were it was

    print (dataset.meta_data)
    for data in data_loader:

        input_data, labels = data
        #TODO we have to divide the input with other data.
        print (input_data['rgb'].shape)

        # TODO, ADD ITERATION SCHEDULE
        print (labels.shape)
        input_rgb_data = augmenter(0, input_data['rgb'])

        output = model(input_rgb_data, labels[:, 11])

        loss = criterion(output, labels)

        #loss.backward()

        #optimizer.step()

        # TODO: save also the optimizer state dictionary
        if is_iteration_for_saving(iteration):

            state = {
                'iteration': iteration,
                'state_dict': model.state_dict()
            }
            # TODO : maybe already summarize the best model ???
            torch.save(state, os.path.join(exp_batch, exp_alias, str(iteration) + '.pth'))

        iteration += 1

        #shutil.copyfile(filename, 'model_best.pth.tar')

    # TODO: DO ALL THE AMAZING LOGGING HERE, as a way to very the status in paralell.
    # THIS SHOULD BE AN INTERELY PARALLEL PROCESS

