import os
import sys

import torch
import torch.optim as optim
import imgauggpu as iag
# What do we define as a parameter what not.

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss
from input import CoILDataset, CoILSampler, splitter
from logger import monitorer
from utils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint
from torchvision import transforms


# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias):
    # We set the visible cuda devices
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # At this point the log file with the correct naming is created.
    merge_with_yaml(os.path.join(exp_batch, exp_alias+'.yaml'))
    set_type_of_process('train')


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
    sampler = CoILSampler(splitter.control_steer_split(dataset.measurements, dataset.meta_data))

    # The data loader is the multi threaded module from pytorch that release a number of
    # workers to get all the data.
    # TODO: batch size an number of workers go to some configuration file
    data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=120,
                                              shuffle=False, num_workers=12, pin_memory=True)
    # By instanciating the augmenter we get a callable that augment images and transform them
    # into tensors.
    augmenter = iag.Augmenter(g_conf.AUGMENTATION_SUITE)

    # TODO: here there is clearly a posibility to make a cool "conditioning" system.

    model = CoILModel(g_conf.MODEL_NAME)
    model.cuda()
    print(model)

    criterion = Loss()

    # TODO: DATASET SIZE SEEMS WEIRD
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


    checkpoint_file = get_latest_saved_checkpoint()
    if checkpoint_file != None:
        checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias,
                                 'checkpoints', str(get_latest_saved_checkpoint())))
        iteration = checkpoint['iteration']
    else:
        iteration = 0

    # TODO: The checkpoint will continue, so the logs should restart ??? OR continue were it was



    print (dataset.meta_data)

    print (model)

    for data in data_loader:

        input_data, labels = data

        #TODO we have to divide the input with other data.

        #TODO, ADD ITERATION SCHEDULE
        input_rgb_data = augmenter(0, input_data['rgb'])

        # get the control commands from labels, size = [120,1]
        controls = labels[:, 24, :]

        # The output(branches) is a list of 5 branches results, each branch is with size [120,3]
        model.zero_grad()
        branches = model(input_rgb_data, labels[:, 10, :].cuda())

        #print ("len ",len(branches))

        # get the steer, gas and brake ground truth from labels
        steer_gt = labels[:, 0, :]
        gas_gt = labels[:, 1, :]
        brake_gt = labels[:, 2, :]
        speed_gt = labels[:, 10, :]

        targets = torch.cat([steer_gt, gas_gt, brake_gt], 1)

        loss = criterion.MSELoss(branches, targets.cuda(), controls.cuda(), speed_gt.cuda())

        loss.backward()
        optimizer.step()

        # TODO: save also the optimizer state dictionary
        if is_ready_to_save(iteration):

            state = {
                'iteration': iteration,
                'state_dict': model.state_dict()
            }
            # TODO : maybe already summarize the best model ???
            torch.save(state, os.path.join('_logs', exp_batch, exp_alias
                                           , 'checkpoints', str(iteration) + '.pth'))
        iteration += 1

        #shutil.copyfile(filename, 'model_best.pth.tar')

    # TODO: DO ALL THE AMAZING LOGGING HERE, as a way to very the status in paralell.
    # THIS SHOULD BE AN INTERELY PARALLEL PROCESS

    #torch.save(model, os.path.join(os.environ["COIL_TRAINED_MODEL_PATH"], exp_alias))
    #print('------------------- Trainind Done! --------------------------')
    #print('The trained model has been saved in ' + os.path.join(os.environ["COIL_TRAINED_MODEL_PATH"], exp_alias))


