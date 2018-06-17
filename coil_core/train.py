import os
import sys
import random
import time
import traceback

import torch
import torch.optim as optim
import imgauggpu as iag


# What do we define as a parameter what not.

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss
from input import CoILDataset, BatchSequenceSampler, splitter, Augmenter
from logger import monitorer, coil_logger
from utils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint
from optmizer import adjust_learning_rate
from torchvision import transforms







# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias, suppress_output=True):
    # We set the visible cuda devices


    # TODO: probable race condition, the train has to be started before.
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        # At this point the log file with the correct naming is created.
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'))
        set_type_of_process('train')

        coil_logger.add_message('Loading', {'GPU': gpu})

        if not os.path.exists('_output_logs'):
            os.mkdir('_output_logs')

        # Put the output to a separate file
        if suppress_output:
            sys.stdout = open(os.path.join('_output_logs',
                              g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"), "a", buffering=1)



        if monitorer.get_status(exp_batch, exp_alias + '.yaml', g_conf.PROCESS_NAME)[0] == "Finished":
            # TODO: print some cool summary or not ?
            return



        checkpoint_file = get_latest_saved_checkpoint()
        if checkpoint_file is not None:
            checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias,
                                     'checkpoints', str(get_latest_saved_checkpoint())))
            iteration = checkpoint['iteration']
            accumulated_time = checkpoint['total_time']
            best_loss = checkpoint['best_loss']
            best_loss_iter = checkpoint['best_loss_iter']

        else:
            iteration = 0
            best_loss = 10000.0
            accumulated_time = 0  # We accumulate iteration time and keep the average speed
            best_loss_iter = 0

        # TODO: The checkpoint will continue, so it should erase everything up to the iteration

        # Define the dataset. This structure is has the __get_item__ redefined in a way
        # that you can access the HD_FILES positions from the root directory as a in a vector.
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME)

        #augmenter_cpu = iag.AugmenterCPU(g_conf.AUGMENTATION_SUITE_CPU)

        # By instanciating the augmenter we get a callable that augment images and transform them
        # into tensors.
        augmenter = Augmenter(g_conf.AUGMENTATION)

        dataset = CoILDataset(full_dataset, transform=augmenter)

        # Creates the sampler, this part is responsible for managing the keys. It divides
        # all keys depending on the measurements and produces a set of keys for each bach.
        keys = range(0, len(dataset.measurements[0, :]) - g_conf.NUMBER_IMAGES_SEQUENCE)
        sampler = BatchSequenceSampler(
                splitter.control_steer_split(dataset.measurements, dataset.meta_data, keys),
                iteration * g_conf.BATCH_SIZE,
                g_conf.BATCH_SIZE, g_conf.NUMBER_IMAGES_SEQUENCE, g_conf.SEQUENCE_STRIDE
        )

        # The data loader is the multi threaded module from pytorch that release a number of
        # workers to get all the data.
        # TODO: batch size an number of workers go to some configuration file
        data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                                                  shuffle=False, num_workers=12,
                                                  pin_memory=True)


        # TODO: here there is clearly a posibility to make a cool "conditioning" system.

        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        model.cuda()

        if checkpoint_file is not None:
            model.load_state_dict(checkpoint['state_dict'])

        print(model)

        criterion = Loss()

        optimizer = optim.Adam(model.parameters(), lr=g_conf.LEARNING_RATE)

        print (dataset.meta_data)

        print (model)
        if checkpoint_file is not None:
            accumulated_time = checkpoint['total_time']
        else:
            accumulated_time = 0  # We accumulate iteration time and keep the average speed


        #TODO: test experiment continuation. Is the data sampler going to continue were it started.. ?
        capture_time = time.time()
        for data in data_loader:


            input_data, float_data = data



            # get the control commands from float_data, size = [120,1]

            controls = float_data[:, dataset.controls_position(), :]

            # The output(branches) is a list of 5 branches results, each branch is with size [120,3]

            model.zero_grad()
            print (input_data['rgb'].shape)

            branches = model(torch.squeeze(input_data['rgb'].cuda()), dataset.extract_inputs(float_data).cuda())


            print ("Weights", g_conf.VARIABLE_WEIGHT.values())

            loss = criterion.MSELoss(branches, dataset.extract_targets(float_data).cuda(),
                                     controls.cuda(), dataset.extract_inputs(float_data).cuda(),
                                     branch_weights=g_conf.BRANCH_LOSS_WEIGHT,
                                     variable_weights=g_conf.VARIABLE_WEIGHT)
            # TODO: All these logging things could go out to clean up the main
            if loss.data < best_loss:
                best_loss = loss.data.tolist()
                best_loss_iter = iteration

            # Log a random position
            position = random.randint(0, len(float_data)-1)

            output = model.extract_branch(torch.stack(branches[0:4]), controls)
            error = torch.abs(output - dataset.extract_targets(float_data).cuda())




            # TODO: For now we are computing the error for just the correct branch, it could be multi- branch,

            coil_logger.add_scalar('Loss', loss.data, iteration)
            coil_logger.add_image('Image', torch.squeeze(input_data['rgb']), iteration)


            loss.backward()
            optimizer.step()

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
                                     'GroundTruth': dataset.extract_targets(float_data)[position].data.tolist(),
                                     'Error': error[position].data.tolist(),
                                     'Inputs': dataset.extract_inputs(float_data)[position].data.tolist()},
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

            if iteration % 1000 == 0:
                adjust_learning_rate(optimizer, iteration)

        coil_logger.add_message('Finished', {})




    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except:
        traceback.print_exc()

        coil_logger.add_message('Error', {'Message': 'Something Happened'})



