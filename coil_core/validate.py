import os
import time
import sys
import random


import torch
import traceback
import dlib

# What do we define as a parameter what not.

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel
from input import CoILDataset, Augmenter
from logger import coil_logger
from coilutils.checkpoint_schedule import get_latest_evaluated_checkpoint, is_next_checkpoint_ready,\
    maximun_checkpoint_reach, get_next_checkpoint


def write_waypoints_output(iteration, output):

    for i in range(g_conf.BATCH_SIZE):
        steer = 0.7 * output[i][3]

        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        coil_logger.write_on_csv(iteration, [steer,
                                            output[i][1],
                                            output[i][2]])


def write_regular_output(iteration, output):
    for i in range(len(output)):
        coil_logger.write_on_csv(iteration, [output[i][0],
                                            output[i][1],
                                            output[i][2]])




# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias, dataset_name, suppress_output):
    latest = None
    try:
        # We set the visible cuda devices
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        # At this point the log file with the correct naming is created.
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias+'.yaml'))
        # The validation dataset is always fully loaded, so we fix a very high number of hours
        g_conf.NUMBER_OF_HOURS = 10000
        set_type_of_process('validation', dataset_name)

        if not os.path.exists('_output_logs'):
            os.mkdir('_output_logs')

        if suppress_output:
            sys.stdout = open(os.path.join('_output_logs',
                                           exp_alias + '_' + g_conf.PROCESS_NAME + '_'
                                           + str(os.getpid()) + ".out"),
                              "a", buffering=1)
            sys.stderr = open(os.path.join('_output_logs',
                              exp_alias + '_err_' + g_conf.PROCESS_NAME + '_'
                                           + str(os.getpid()) + ".out"),
                              "a", buffering=1)


        # Define the dataset. This structure is has the __get_item__ redefined in a way
        # that you can access the HDFILES positions from the root directory as a in a vector.
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name)
        augmenter = Augmenter(None)
        # Definition of the dataset to be used. Preload name is just the validation data name
        dataset = CoILDataset(full_dataset, transform=augmenter,
                              preload_name=dataset_name)

        # Creates the sampler, this part is responsible for managing the keys. It divides
        # all keys depending on the measurements and produces a set of keys for each bach.

        # The data loader is the multi threaded module from pytorch that release a number of
        # workers to get all the data.
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=g_conf.BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=g_conf.NUMBER_OF_LOADING_WORKERS,
                                                  pin_memory=True)

        model = CoILModel(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
        # The window used to keep track of the trainings
        l1_window = []
        latest = get_latest_evaluated_checkpoint()
        if latest is not None:  # When latest is noe
            l1_window = coil_logger.recover_loss_window(dataset_name, None)

        model.cuda()

        best_mse = 1000
        best_error = 1000
        best_mse_iter = 0
        best_error_iter = 0

        while not maximun_checkpoint_reach(latest, g_conf.TEST_SCHEDULE):

            if is_next_checkpoint_ready(g_conf.TEST_SCHEDULE):

                latest = get_next_checkpoint(g_conf.TEST_SCHEDULE)

                checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias
                                        , 'checkpoints', str(latest) + '.pth'))
                checkpoint_iteration = checkpoint['iteration']
                print("Validation loaded ", checkpoint_iteration)

                model.load_state_dict(checkpoint['state_dict'])

                model.eval()
                accumulated_mse = 0
                accumulated_error = 0
                iteration_on_checkpoint = 0
                for data in data_loader:

                    # Compute the forward pass on a batch from  the validation dataset
                    controls = data['directions']
                    output = model.forward_branch(torch.squeeze(data['rgb']).cuda(),
                                                  dataset.extract_inputs(data).cuda(),
                                                  controls)
                    # It could be either waypoints or direct control
                    if 'waypoint1_angle' in g_conf.TARGETS:
                        write_waypoints_output(checkpoint_iteration, output)
                    else:
                        write_regular_output(checkpoint_iteration, output)

                    mse = torch.mean((output -
                                      dataset.extract_targets(data).cuda())**2).data.tolist()
                    mean_error = torch.mean(
                                    torch.abs(output -
                                              dataset.extract_targets(data).cuda())).data.tolist()

                    accumulated_error += mean_error
                    accumulated_mse += mse
                    error = torch.abs(output - dataset.extract_targets(data).cuda())

                    # Log a random position
                    position = random.randint(0, len(output.data.tolist())-1)

                    coil_logger.add_message('Iterating',
                         {'Checkpoint': latest,
                          'Iteration': (str(iteration_on_checkpoint*120)+'/'+str(len(dataset))),
                          'MeanError': mean_error,
                          'MSE': mse,
                          'Output': output[position].data.tolist(),
                          'GroundTruth': dataset.extract_targets(data)[position].data.tolist(),
                          'Error': error[position].data.tolist(),
                          'Inputs': dataset.extract_inputs(data)[position].data.tolist()},
                          latest)
                    iteration_on_checkpoint += 1
                    print("Iteration %d  on Checkpoint %d : Error %f" % (iteration_on_checkpoint,
                                                                checkpoint_iteration, mean_error))

                """
                    ########
                    Finish a round of validation, write results, wait for the next
                    ########
                """

                checkpoint_average_mse = accumulated_mse/(len(data_loader))
                checkpoint_average_error = accumulated_error/(len(data_loader))
                coil_logger.add_scalar('Loss', checkpoint_average_mse, latest, True)
                coil_logger.add_scalar('Error', checkpoint_average_error, latest, True)

                if checkpoint_average_mse < best_mse:
                    best_mse = checkpoint_average_mse
                    best_mse_iter = latest

                if checkpoint_average_error < best_error:
                    best_error = checkpoint_average_error
                    best_error_iter = latest

                coil_logger.add_message('Iterating',
                     {'Summary':
                         {
                          'Error': checkpoint_average_error,
                          'Loss': checkpoint_average_mse,
                          'BestError': best_error,
                          'BestMSE': best_mse,
                          'BestMSECheckpoint': best_mse_iter,
                          'BestErrorCheckpoint': best_error_iter
                         },

                      'Checkpoint': latest},
                                        latest)

                l1_window.append(checkpoint_average_error)
                coil_logger.write_on_error_csv(dataset_name, checkpoint_average_error)

                # If we are using the finish when validation stops, we check the current
                if g_conf.FINISH_ON_VALIDATION_STALE is not None:
                    if dlib.count_steps_without_decrease(l1_window) > 3 and \
                            dlib.count_steps_without_decrease_robust(l1_window) > 3:
                        coil_logger.write_stop(dataset_name, latest)
                        break

            else:

                latest = get_latest_evaluated_checkpoint()
                time.sleep(1)

                coil_logger.add_message('Loading', {'Message': 'Waiting Checkpoint'})
                print("Waiting for the next Validation")

        coil_logger.add_message('Finished', {})

    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})
        # We erase the output that was unfinished due to some process stop.
        if latest is not None:
            coil_logger.erase_csv(latest)

    except RuntimeError as e:
        if latest is not None:
            coil_logger.erase_csv(latest)
        coil_logger.add_message('Error', {'Message': str(e)})

    except:
        traceback.print_exc()
        coil_logger.add_message('Error', {'Message': 'Something Happened'})
        # We erase the output that was unfinished due to some process stop.
        if latest is not None:
            coil_logger.erase_csv(latest)
