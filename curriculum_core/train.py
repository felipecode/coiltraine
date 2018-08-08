import os
import time
import traceback
import torch
import multiprocessing
from torch import optim

from configs import g_conf
from input.splitter import full_split
from input import CoILDataset, Augmenter
from input.coil_sampler import PreSplittedSampler
from network import CoILModel, Loss, adjust_learning_rate
from utils.checkpoint_schedule import is_ready_to_save
from .configs import MODEL_TYPE, MODEL_CONFIGURATION


def execute_train(weights, keys, iteration, checkpoint, gpu):
    # p = multiprocessing.Process(target=execute,
    #                             args=(weights, keys, iteration, checkpoint, gpu))
    # p.start()
    execute(weights, keys, iteration, checkpoint, gpu)


def execute(weights, keys, iteration, checkpoint, gpu):
    try:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        g_conf.VARIABLE_WEIGHT = {}
        # print("BEFORE MERGE variable ", g_conf.VARIABLE_WEIGHT, 'conf targets',
        #       g_conf.TARGETS)

        if not os.path.exists('_output_logs'):
            os.mkdir('_output_logs')

        # print("variable ", g_conf.VARIABLE_WEIGHT, 'conf targets', g_conf.TARGETS)

        # Setup sampler and data loader
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME)
        augmenter = Augmenter(g_conf.AUGMENTATION)
        sampler = PreSplittedSampler(keys, iteration*g_conf.BATCH_SIZE, weights)
        dataset = CoILDataset(full_dataset, transform=augmenter)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=g_conf.BATCH_SIZE,
                                                  shuffle=False,
                                                  sampler=sampler,
                                                  num_workers=2,
                                                  pin_memory=True)

        model = CoILModel(MODEL_TYPE, MODEL_CONFIGURATION)
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['state_dict'])
        model = model.cuda()

        # print(model)

        criterion = Loss(g_conf.LOSS_FUNCTION)

        optimizer = optim.Adam(model.parameters(), lr=g_conf.LEARNING_RATE)

        # print (dataset.meta_data)
        accumulated_time = ckpt['total_time']

        #TODO: test experiment continuation. Is the data sampler going to continue were it started.. ?
        capture_time = time.time()
        for data in data_loader:
            input_data, float_data = data
            # get the control commands from float_data, size = [120,1]
            controls = float_data[:, dataset.controls_position(), :]
            # The output(branches) is a list of 5 branches results, each branch is with size [120,3]
            model.zero_grad()
            branches = model(torch.squeeze(input_data['rgb'].cuda()),
                             dataset.extract_inputs(float_data).cuda())
            loss = criterion(branches, dataset.extract_targets(float_data).cuda(),
                             controls.cuda(), dataset.extract_inputs(float_data).cuda(),
                             branch_weights=g_conf.BRANCH_LOSS_WEIGHT,
                             variable_weights=g_conf.VARIABLE_WEIGHT)
            loss.backward()
            optimizer.step()
            accumulated_time += time.time() - capture_time
            capture_time = time.time()
            if is_ready_to_save(iteration):
                model = model.cpu()
                state = {
                    'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'total_time': accumulated_time,
                }
                # TODO : maybe already summarize the best model ???
                torch.save(state, checkpoint)
                model = model.cuda()

            iteration += 1

            if iteration % 1000 == 0:
                adjust_learning_rate(optimizer, iteration)
            del data

            if iteration % 100 == 0:
                break

    except KeyboardInterrupt:
        print('Error', 'Message: Killed By User')

    except:
        traceback.print_exc()
        print('Error', 'Message: Something Happened')
        exit()
