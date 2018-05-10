import os
import time
import sys

import numpy as np
import torch
import torch.optim as optim

# What do we define as a parameter what not.

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss
from input import CoILDataset, CoILSampler, splitter
from logger import monitorer, coil_logger
from utils.checkpoint_schedule import get_latest_evaluated_checkpoint, is_next_checkpoint_ready,\
    maximun_checkpoint_reach, get_next_checkpoint, get_latest_saved_checkpoint
from torchvision import transforms


# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias):
    # We set the visible cuda devices
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # At this point the log file with the correct naming is created.
    merge_with_yaml(os.path.join(exp_batch, exp_alias+'.yaml'))
    set_type_of_process('validation')




    sys.stdout = open(str(os.getpid()) + ".out", "a", buffering=1)

    if monitorer.get_status(exp_batch, exp_alias, g_conf.PROCESS_NAME)[0] == "Finished":
        # TODO: print some cool summary or not ?
        return

    #Define the dataset. This structure is has the __get_item__ redefined in a way
    #that you can access the HDFILES positions from the root directory as a in a vector.
    full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.DATASET_NAME)

    dataset = CoILDataset(full_dataset, transform=transforms.Compose([transforms.ToTensor()]))

    # Creates the sampler, this part is responsible for managing the keys. It divides
    # all keys depending on the measurements and produces a set of keys for each bach.

    # The data loader is the multi threaded module from pytorch that release a number of
    # workers to get all the data.
    # TODO: batch size an number of workers go to some configuration file
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
                                              shuffle=False, num_workers=12, pin_memory=True)


    # TODO: here there is clearly a posibility to make a cool "conditioning" system.
    model = CoILModel(g_conf.MODEL_NAME)
    model.cuda()



    # TODO: The checkpoint will continue, so the logs should restart ??? OR continue were it was

    latest = get_latest_evaluated_checkpoint()
    if latest is None:  # When nothing was tested, get latest returns none, we fix that.
        latest = 0

    print (dataset.meta_data)

    while not maximun_checkpoint_reach(latest, g_conf.TEST_SCHEDULE):

        if is_next_checkpoint_ready(g_conf.TEST_SCHEDULE):

            latest = get_next_checkpoint(g_conf.TEST_SCHEDULE)

            checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias
                                    , 'checkpoints', str(latest) + '.pth'))
            checkpoint_iteration = checkpoint['iteration']
            print ("Validation loaded ", checkpoint_iteration)

            for data in data_loader:

                input_data, labels = data
                control_position = np.where(dataset.meta_data[:, 0] == 'control')[0][0]
                speed_position = np.where(dataset.meta_data[:, 0] == 'speed_module')[0][0]
                print (torch.squeeze(input_data['rgb']).shape)

                print (control_position)
                print (speed_position)
                # Obs : Maybe we could also check for other branches ??
                output = model.forward_branch(torch.squeeze(input_data['rgb']).cuda(),
                                              labels[:, speed_position, :].cuda(),
                                              labels[:, control_position, :].cuda())
                # TODO: clean this squeeze and dimension things


                for i in range(input_data['rgb'].shape[0]):

                    coil_logger.write_on_csv(checkpoint_iteration, [output[i][0],
                                                                    output[i][1],
                                                                    output[i][2]])




                #loss = criterion(output, labels)

                #loss.backward()

                #optimizer.step()

                #shutil.copyfile(filename, 'model_best.pth.tar')
        else:
            time.sleep(1)
            print ("Waiting for the next Validation")

    # TODO: DO ALL THE AMAZING LOGGING HERE, as a way to very the status in paralell.
    # THIS SHOULD BE AN INTERELY PARALLEL PROCESS

