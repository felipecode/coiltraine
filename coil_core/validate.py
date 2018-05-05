import os

import torch
import torch.optim as optim
# What do we define as a parameter what not.

from configs import g_conf
from network import Model, Loss
from input import CoILDataset, CoILSampler, splitter
import imgauggpu as iag
from logger import monitorer


# TODO: Maybe avoid replication from the pre loop phase in a future refactor
# The main function maybe we could call it with a default name
def execute(gpu, exp_alias, compute_loss=True):

    # TODO: VALIDATION SHOULD BE SEQUENTIAL OR RANDOM DEPENDING ON WHAT YAML says
    # TODO: How many validations can we put per GPU ??
    # We set the visible cuda devices
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # At this point the log file with the correct naming is created.
    g_conf.merge_with_parameters(exp_alias)
    g_conf.set_type_of_process('validate')

    #TODO: Get THe experiment folder somehow

    if monitorer.get_status(exp_alias) == "Finished":
        return

    #Define the dataset. This structure is has the __get_item__ redefined in a way
    #that you can access the HDFILES positions from the root directory as a in a vector.
    full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.param.dataset_folder_name)

    dataset = CoILDataset(full_dataset)

    # Creates the sampler, this part is responsible for managing the keys. It divides
    # all keys depending on the measurements and produces a set of keys for each bach.
    sampler = CoILSampler(splitter.control_steer_split(dataset.measurements, dataset.meta_data))

    # The data loader is the multi threaded module from pytorch that release a number of
    # workers to get all the data.
    # TODO: batch size an number of workers go to
    data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=120,
                                              shuffle=False, num_workers=12, pin_memory=True)
    # By instanciating the augmenter we get a callable that augment images and transform them
    # into tensors.
    augmenter = iag.Augmenter(g_conf.param.INPUT.AUGMENTATION_SUITE)

    # TODO: here there is clearly a posibility to make a cool "conditioning" system.
    model = Model(g_conf.param.NETWORK.MODEL_DEFINITION)

    criterion = Loss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    #TODO: Probably there is more differences between train and validation that justify a new file.

    for data in data_loader:

        input_data, labels = data
        #TODO we have to divide the input with other data.

        input_data = augmenter(input_data)

        output = model(input_data)



    # TODO: DO ALL THE AMAZING LOGGING HERE, as a way to very the status in paralell.
    # THIS SHOULD BE AN INTERELY PARALLEL PROCESS

