
import numpy as np

import random
import bisect
import os.path

import traceback
import time
import math
import spliter
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
#from codification import *
import torch

from torch.utils.data.sampler import Sampler



class CoILSampler(Sampler):

    def __init__(self, measurements):

        control = measurements
        steering = measurements
        keys =
        keys_splited = spliter.split_by_value(spliter.split_by_label(keys, split_style, labels),
                                              spliting_bins)










             #initialIteration,  , , , ,):











             if not hasattr(config, 'just_validating'):
                 file_step = len(file_names) / config.number_data_divisions

             print
             len(file_names)

             spliter_vec = []

             images_vec = []

             dataset_vec = []
             for div in range(config.number_data_divisions):
                 """ Read all hdf5_files """
             self._images_train, self._datasets_train = self.read_all_files(
                 file_names[(div) * file_step:(div + 1) * file_step], config.sensor_names,
             config.dataset_names)

             images_vec.append(self._images_train)
             dataset_vec.append(self._datasets_train)

             # print self._datasets_train[0][config.variable_names.index("Steer")][:]
             divided_keys_train = spliter_control.divide_keys_by_labels(

            self._datasets_train[0][config.variable_names.index("Control")][:], config.labels_per_division)


                # The vector that says if there is noise or not

                np.set_printoptions(threshold=np.nan)

                # print len(divided_keys_train)
                if hasattr(config, 'labels_angle_per_division') and len(config.labels_angle_per_division) > 0:
                    correct_angle = spliter_angle.divide_keys_by_labels(
                self._datasets_train[0][config.variable_names.index("Angle")][:],
                config.labels_angle_per_division)

                for i in range(len(divided_keys_train)):
                    divided_keys_train[i] = list(set(divided_keys_train[i]).intersection(set(correct_angle[0])))

                if hasattr(config, 'labels_noise_per_division') and len(config.labels_noise_per_division) > 0:
                    noise_vec = self._datasets_train[0][config.variable_names.index("Steer")][:] != \
                                self._datasets_train[0][config.variable_names.index("Steer_N")][:]

                # print noise_vec
                # print np.where(noise_vec==True)
                correct_angle = spliter_angle.divide_keys_by_labels(noise_vec,
                config.labels_noise_per_division)

                for i in range(len(divided_keys_train)):
                    divided_keys_train[i] = list(set(divided_keys_train[i]).intersection(set(correct_angle[0])))

                self._splited_keys_train = spliter_control.split_by_output(


            self._datasets_train[0][config.variable_names.index("Steer")][:], divided_keys_train)

            spliter_vec.append(self._splited_keys_train)














        # Parameters related to files and sequence size. That is used for using sequences instead of images



        self._perform_sequential = perform_sequential

        self._number_frames_fused = config_input.number_frames_fused
        self._number_frames_sequenced = config_input.number_frames_sequenced
        self._inputs_per_sequence = config_input.number_frames_sequenced  # config_input.number_frames_fused + config_input.number_frames_sequenced
        self._inputs_per_file = config_input.inputs_per_file
        self._resample_stride = config_input.resample_stride



        self._current_position_on_dataset = 0

        self._eliminate_not_central = hasattr(config_input, 'labels_angle_per_division') and \
                                      math.fabs(config_input.labels_angle_per_division[0][0]) == 0.0

        # For it to validate on just center it needs to have eliminate not central and to have more than 1 frame to be
        # sequenced

        self._splited_keys = splited_keys
        self._images = images
        if perform_sequential:
            self._variables = np.concatenate(tuple(datasets), axis=0)  # Cat the datasets

        else:
            self._variables = datasets


        self._positions_to_train = range(0,
                                         config_input.number_steering_bins)  # WARNING THIS NEED TO BE A MULTIPLE OF THE NUMBER OF CLIPS

        # The iteration may be neccessary.
        #self._iteration = initialIteration


        if hasattr(config_input, 'augmentation_function'):
            self._augmenting_function = getattr(augmenter, config_input.augmentation_function)
        else:
            self._augmenting_function = getattr(augmenter, 'build_augmenter_cont')




        self._config = config_input
        self._batch_size = config_input.batch_size


    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def get_batch_tensor(self):
        return self._batch_tensor

    def get_data_by_ids(self, generated_ids, batch_size):

        X_batch = np.zeros((batch_size, 1, self._input_size[0], self._input_size[1], self._input_size[2]),
                           dtype='uint8')
        count = 0
        # print generated_ids
        for i in generated_ids:
            i = int(i)
            for es, ee, x in self._images:
                # print es
                # print i
                # print ee
                # print x.shape
                if i >= es and i < ee:
                    # print x[]

                    image = np.array(x[i - es - 1 + 1:i - es + 1, :, :, :])
                    # print image

                    X_batch[count] = image
                    break

            count += 1
        return X_batch

    def sample_positions_to_train(self, number_of_samples):
        def sample_from_vec(vector):

            sample_number = random.choice(vector)

            # Remove the sampled position from the main list and the splited_list
            del self._positions_to_train[self._positions_to_train.index(sample_number)]
            del vector[vector.index(sample_number)]
            # Refil if is the case

            return sample_number, vector

        # Divide it into 3 equal parts
        sample_positions = []
        splited_list = []
        if len(self._positions_to_train) == 0:
            self._positions_to_train = range(0, self._config.number_steering_bins)

        for i in range(0, 3):
            position = i * (len(self._positions_to_train) / 3)
            if i == 2:
                splited_list.append(self._positions_to_train[position:])
            else:
                splited_list.append(self._positions_to_train[position:position + len(self._positions_to_train) / 3])
        # splited_list[2].append(splited_list[3][0])
        # del splited_list[3]
        # print splited_list

        sample_id = 0
        # print number_of_samples
        # print "Positions to Train"
        # print len(self._positions_to_train)
        while sample_id < number_of_samples:
            # Sample Mid
            if len(splited_list[1]) > 0:

                sampled_value, splited_list[1] = sample_from_vec(splited_list[1])
                # print sampled_value,splited_list[1]

                sample_positions.append(sampled_value)
                sample_id += 1
                if sample_id >= number_of_samples:
                    break

                if len(self._positions_to_train) == 0:
                    self._positions_to_train = range(0, self._number_steering_levels)
                    splited_list = []
                    for i in range(0, 3):
                        position = i * (len(self._positions_to_train) / 3)
                        if i == 2:
                            splited_list.append(self._positions_to_train[position:])
                        else:
                            splited_list.append(
                                self._positions_to_train[position:position + len(self._positions_to_train) / 3])

            # Sample Left

            if len(splited_list[0]) > 0:

                sampled_value, splited_list[0] = sample_from_vec(splited_list[0])
                sample_positions.append(sampled_value)
                sample_id += 1
                if sample_id >= number_of_samples:
                    break

                if len(self._positions_to_train) == 0:
                    self._positions_to_train = range(0, self._config.number_steering_bins)
                    splited_list = []
                    for i in range(0, 3):
                        position = i * (len(self._positions_to_train) / 3)
                        if i == 2:
                            splited_list.append(self._positions_to_train[position:])
                        else:
                            splited_list.append(
                                self._positions_to_train[position:position + len(self._positions_to_train) / 3])

            # Sample Right
            if len(splited_list[2]) > 0:

                sampled_value, splited_list[2] = sample_from_vec(splited_list[2])
                sample_positions.append(sampled_value)
                sample_id += 1
                if sample_id >= number_of_samples:
                    break

                if len(self._positions_to_train) == 0:
                    self._positions_to_train = range(0, self._config.number_steering_bins)
                    splited_list = []
                    for i in range(0, 3):
                        position = i * (len(self._positions_to_train) / 3)
                        if i == 2:
                            splited_list.append(self._positions_to_train[position:])
                        else:
                            splited_list.append(
                                self._positions_to_train[position:position + len(self._positions_to_train) / 3])

        return sample_positions

    def _get_position(self, chosen_id):

        return chosen_id % self._inputs_per_file

    def _read_images2(self,images, batch_size, used_ids):
        # sensors_batch = []
        # for i in range(len(self._images)):
        #    sensors_batch.append(np.zeros((batch_size, self._config.sensors_size[i][0],
        #                                   self._config.sensors_size[i][1], self._config.sensors_size[i][2]),
        #                                  dtype='uint8'))        for s in range(len(self._images)):

        # We test here directly and include the other images here.
        sensors_batch = []
        for i in range(len(images)):
            sensors_batch.append(np.zeros(
                (batch_size,
                 self._config.sensors_size[i][0],
                 self._config.sensors_size[i][1],
                 self._config.sensors_size[i][2] * self._number_frames_fused), dtype='uint8'
            ))

        for s in range(len(images)):
            count = 0

            for chosen_key in used_ids:

                count_seq = 0
                first_enter = True

                for i in range(self._number_frames_fused):
                    chosen_key = chosen_key + i * 3

                    for es, ee, x in images[s]:

                        if chosen_key >= es and chosen_key < ee:
                            """ We found the part of the data to open """
                            # print x[]
                            first_enter = False

                            pos_inside = chosen_key - es

                            # print 'el i'
                            # print chosen_key
                            # print pos_inside
                            # print x[chosen_key - es - 1 + 1:chosen_key - es + 1,:,:,:].shape

                            sensors_batch[s][count, :, :,
                            (i * 3):((i + 1) * 3)] = np.array(x[pos_inside, :, :, :])

                            # print sensors_batch[s][count].shape
                            #if not self._perform_sequential:
                            #img = Image.fromarray(sensors_batch[s][count])
                            #img.save('test' + str(self._current_position_on_dataset +count) + '_0_.jpg')
                count += 1

        return sensors_batch

    def _reshape_for_sequences(self, sensors):

        """


        :param data_to_reshape:

        First Step:
        The frames fused are made to stay in the end

        Second Step:

        Reshape to have the sequences, remember that the fused frames are replicated

        Seems to be fine. have to do the same thing for targets inputs

        :return:
        """

        # TODO THis function is clearly not working for N not equal to 3

        final_sensor_list = []

        for data_to_reshape in sensors:

            input_shape = data_to_reshape.shape

            # print input_shape

            # print data_to_reshape

            data_reshaped = np.zeros((input_shape[0] - self._number_frames_fused + 1
                                      , self._config.sensors_size[0][0], self._config.sensors_size[0][1]
                                      , self._config.sensors_size[0][2] * self._number_frames_fused))

            for i in range(self._number_frames_fused):
                for j in range(self._number_frames_fused):

                    final_index = -self._number_frames_fused + i + 1

                    if final_index < 0:
                        array = data_to_reshape[j + i:final_index:self._number_frames_fused, :, :, :]

                    else:
                        array = data_to_reshape[j + i::self._number_frames_fused, :, :, :]

                    data_reshaped[j::self._number_frames_fused, :, :,
                    (i * 3):((i + 1) * 3)] = array

            # final_array = np.reshape(data_reshaped, (
            #    shape_r[0] / (self._number_frames_sequenced+1), (self._number_frames_sequenced+1)
            #   , shape_r[1], shape_r[2], shape_r[3]))
            final_sensor_list.append(data_reshaped)

        # for

        return final_sensor_list


    def out_of_border(self,position):



        if position % 200 < 20 or position % 200 > 180:
            return True

        return False



    def generate_batch_keys(self,variables, batch_size,splited_keys,number_control_divisions):

        """
        :param batch_size:
        :param number_control_divisions:
        :param resample_stride: We need to know how much inputs were skiped ( FPS reduction)
        :return:
        """

        # generated_ids = np.zeros((batch_size/self._inputs_per_sequence, self._inputs_per_sequence), dtype='int32')
        # The batch size right now is fixed thinking on a number of frames equal to three

        generated_ids = np.zeros((batch_size), dtype='int32')

        # while True:
        chosen_keys_vec = []


        try:
            count = 0



            for control_part in range(0, number_control_divisions):



                sampled_positions = self.sample_positions_to_train(
                    int(batch_size / (self._inputs_per_sequence * 3)))





                # print sampled_positions
                # print ' We Sampled : ', sampled_positions, 'That is ',len(sampled_positions)

                for outer_n in sampled_positions:

                    # Key that was chosen by chance to be added to the batch.

                    chosen_key = variables.shape[1]
                    # print chosen_key
                    while chosen_key > variables.shape[1] - self._inputs_per_sequence * 3:

                        chosen_key = random.choice(splited_keys[control_part][outer_n])


                    if self._inputs_per_sequence > 1 and self._config.bdd_data:
                        while self.out_of_border(chosen_key):
                            chosen_key = random.choice(splited_keys[control_part][outer_n])



                    #while not self.small_diff(chosen_key, chosen_keys_vec):

                    #    chosen_key = random.choice(self._splited_keys[control_part][outer_n])

                        # print 'That was the key chosen as startpoint ', chosen_key


                    #chosen_keys_vec.append(chosen_key)

                    # We need to know on which position this key is inside the hdf5 files

                    #position_on_file = self._get_position(chosen_key)
                    # print ' This key was inside a file on position: ', position_on_file

                    # Get the number of files to take on the file used to get the images.
                    # inputs_left_on_file = max(self._inputs_per_sequence - position_on_file, self._inputs_per_sequence)

                    # print ' That is why we are first looking at ',self._inputs_per_sequence,' in a pace of ', self._resample_stride
                    if self._eliminate_not_central:
                        for j in range(0, self._inputs_per_sequence * 3, self._resample_stride * 3):
                            # Take the choosen key + The strided version.
                            generated_ids[count] = int(chosen_key + j)
                            # output[count,j/resample_stride,:] = self._outputs[:,i + j]
                            count += 1
                    else:
                        for j in range(0, self._inputs_per_sequence, self._resample_stride):
                            # Take the choosen key + The strided version.
                            generated_ids[count] = int(chosen_key + j)
                            # output[count,j/resample_stride,:] = self._outputs[:,i + j]
                            count += 1


            return generated_ids
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            pass



    def generate_sequential_batch_keys(self, batch_size):

        """
        :param batch_size:
        :param number_control_divisions:
        :param resample_stride: We need to know how much inputs were skiped ( FPS reduction)
        :return:
        """



        # generated_ids = np.zeros((batch_size/self._inputs_per_sequence, self._inputs_per_sequence), dtype='int32')
        # The batch size right now is fixed thinking on a number of frames equal to three


        # while True:
        try:





            if self._eliminate_not_central and (self._number_frames_sequenced >1 or self._number_frames_fused > 1) :


                count=0

                generated_ids = []
                while count < batch_size:
                    if self._variables[self._config.variable_names.index('Angle'), self._current_position_on_dataset] == 0.0:
                        generated_ids.append(self._current_position_on_dataset)
                        count += 1

                    self._current_position_on_dataset += 1
                    if self._current_position_on_dataset == len(self._variables[0, :]):
                        self._current_position_on_dataset = 0

                generated_ids= np.array(generated_ids)

            else:
                if self._current_position_on_dataset + batch_size >= len(self._variables[0, :]) + 1:
                    # Take from the end

                    self._current_position_on_dataset = 0



                generated_ids = np.array(
                            range(self._current_position_on_dataset, self._current_position_on_dataset + batch_size))


                self._current_position_on_dataset += batch_size


            return generated_ids
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            pass




    """Return the next `batch_size` examples from this data set."""

    def next_batch(self):

        batch_size = self._batch_size
        if self._perform_sequential:
            variables = self._variables
            images = self._images
        else:
            selected_dataset = random.randint(0,len(self._splited_keys)-1)

            variables = np.concatenate(tuple(self._variables[selected_dataset]), axis=0)
            images = self._images[selected_dataset]

        if self._perform_sequential:
            generated_ids = self.generate_sequential_batch_keys(batch_size)
        else:
            generated_ids = self.generate_batch_keys(variables,batch_size,self._splited_keys[selected_dataset], 3)


        # print generated_ids
        # print 'generated_ids'
        # print generated_ids.shape
        sensors = self._read_images2(images ,batch_size, generated_ids)

        # sensors, generated_ids = self.datagen(1, batch_size, len(self._splited_keys))

        if hasattr(self._config, 'augmentation_is_continuous'):


            augmenter = self._augmenting_function(self._iteration)

            for i in range(len(sensors)):
                sensors[i] = np.array((sensors[i]))

                if self._augmenter_sched[0] is not None and self._augmenter_sched[0][i+1] == True:
                    if i == 1:
                        sensors[i][np.where(sensors[i] == 0)] = 6

                    for j in range(self._number_frames_fused):
                        sensors[i][:,:,:,(i*3):(i+1)*3] = augmenter.augment_images(
                                                        sensors[i][:,:,:,(i*3):(i+1)*3])


                    if i == 1:
                        sensors[i][np.where(sensors[i] == 0)] = 2 * (255 / 4)
                        sensors[i][np.where(sensors[i] == 6)] = 0

                sensors[i] = sensors[i].astype(np.float32)

        else:
            augmenter_vec = [None] * len(sensors)
            if self._augmenter_sched[0] is not None:
                for position in self._augmenter_sched:
                    if self._iteration < position[0]:  # already got to this iteration
                        break

                    else:
                        for i in range(1, len(sensors) + 1):
                            augmenter_vec[i - 1] = position[i]

                # Get the images
                # print 'LEN SENSORS ',len(senso)
            for i in range(len(sensors)):
                sensors[i] = np.array((sensors[i]))


                if augmenter_vec[i] != None:
                    if i == 1:
                        sensors[i][np.where(sensors[i] == 0)] = 6

                    sensors[i] = augmenter_vec[i].augment_images(sensors[i])
                    if i == 1:
                        sensors[i][np.where(sensors[i] == 0)] = 2 * (255 / 4)
                        sensors[i][np.where(sensors[i] == 6)] = 0

                sensors[i] = sensors[i].astype(np.float32)



        # Get the targets
        float_data = variables[:, generated_ids]
        targets = []
        for i in range(len(self._config.targets_names)):
            targets.append(np.zeros((batch_size, self._config.targets_sizes[i])))

        # Get the inputs
        inputs = []
        for i in range(len(self._config.inputs_names)):
            inputs.append(np.zeros((batch_size, self._config.inputs_sizes[i])))

        for i in range(0, batch_size):

            for j in range(len(images)):
                # if self._config.sensor_names[j] == 'labels':
                # sensors[j][i,:,:,:] = sensors[j][i,:,:,:]

                sensors[j][i, :, :, :] = np.multiply(sensors[j][i, :, :, :], 1.0 / 255.0)

        for i in range(0, batch_size):

            count = 0
            for j in range(len(self._config.targets_names)):
                k = self._config.variable_names.index(self._config.targets_names[j])
                targets[count][i] = float_data[k, i]

                if self._config.targets_names[j] == "Speed":
                    targets[count][i] /= self._config.speed_factor

                if self._config.targets_names[j] == "Command":


                    targets[count][i] = bdd_command_to_logits(float_data[k, i])



                if self._config.targets_names[j] == "Gas":
                    targets[count][i] = max(0, targets[count][i])

                if self._config.targets_names[j] == "Steer":
                    if hasattr(self._config, 'bdd_data'):
                        targets[count][i] = min(1.0, max(-1.0, 30 * targets[count][i]))
                    else:
                        targets[count][i] = min(1.0, max(-1.0,  targets[count][i]))



                if hasattr(self._config, 'extra_augment_factor') and self._config.targets_names[j] == "Steer":
                    camera_pos = self._config.variable_names.index('Camera')
                    speed_pos = self._config.variable_names.index('Speed')

                    # angle = self._config.variable_names.index('Angle') == 15
                    angle = float_data[self._config.variable_names.index('Angle'), i]

                    #########Augmentation!!!!
                    time_use = 1.0
                    car_lenght = 6.0
                    speed = math.fabs(float_data[speed_pos, i])
                    if angle > 0.0:
                        angle = math.radians(math.fabs(angle))
                        targets[count][i] -= min(self._config.extra_augment_factor * (
                            math.atan((angle * car_lenght) / (time_use * speed + 0.05))) / 3.1415, 0.3)
                    else:
                        angle = math.radians(math.fabs(angle))
                        targets[count][i] += min(self._config.extra_augment_factor * (
                            math.atan((angle * car_lenght) / (time_use * speed + 0.05))) / 3.1415, 0.3)

                if hasattr(self._config, 'saturated_factor') and self._config.targets_names[j] == "Steer":
                    angle = float_data[self._config.variable_names.index('Angle'), i]
                    # print angle
                    #########Augmentation!!!!
                    if angle < 0.0:

                        targets[count][i] = 1.0
                    elif angle > 0.0:
                        targets[count][i] = -1.0

                count += 1

            count = 0
            for j in range(len(self._config.inputs_names)):
                k = self._config.variable_names.index(self._config.inputs_names[j])

                if self._config.inputs_names[j] == "Camera":
                    inputs[count][i] = float_data[k, i]

                if self._config.inputs_names[j] == "Control":
                    inputs[count][i] = encode(float_data[k, i])

                if self._config.inputs_names[j] == "Speed":
                    inputs[count][i] = float_data[k, i] / self._config.speed_factor

                if self._config.inputs_names[j] == "Distance":
                    inputs[count][i] = check_distance(float_data[k, i])

                if self._config.inputs_names[j] == "Goal":
                    module = math.sqrt(
                        float_data[k, i] * float_data[k, i] + float_data[k + 1, i] * float_data[k + 1, i])
                    # print 'k ',k
                    # print 'module',module
                    # print 'float_data',float_data[k,i],float_data[k+1,i]
                    float_data[k, i] = float_data[k, i] / module
                    float_data[k + 1, i] = float_data[k + 1, i] / module

                    inputs[count][i] = float_data[k:k + 2, i]

                count += 1

        self._iteration += 1

        # if self._number_frames_fused >1:
        #    sensors = self._reshape_for_sequences(sensors)

        #print ' SHARKS IN THE WATER AND THE WATER IS DEEP'


        # print targets
        try:  # We have both labels and rgb
            labels_pos = self._config.sensor_names.index('labels')
            rgb_pos = self._config.sensor_names.index('rgb')

            fused_rgb_labels = np.concatenate((sensors[rgb_pos], sensors[labels_pos]), axis=3)
            # print 'CONCATENATE'
            return fused_rgb_labels, targets, inputs

        except:

            try:  # We have labels
                labels_pos = self._config.sensor_names.index('labels')
                return sensors[labels_pos], targets, inputs
            except:  # We have RGB
                return sensors[self._config.sensor_names.index('rgb')], targets, inputs




    def process_run(self, sess, data_loaded):

        queue_feed_dict = {self._queue_image_input: data_loaded[0]}  # images we already put by default

        for i in range(len(self._config.targets_names)):
            queue_feed_dict.update({self._queue_targets[i]: data_loaded[1][i]})

        for i in range(len(self._config.inputs_names)):
            queue_feed_dict.update({self._queue_inputs[i]: data_loaded[2][i]})

        # print queue_feed_dict
        sess.run(self._enqueue_op, feed_dict=queue_feed_dict)

    def enqueue(self, sess):

        while True:
            # print("starting to write into queue")
            queue_time = time.time()

            data_loaded = self.next_batch()

            self.process_run(sess, data_loaded)

