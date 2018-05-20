import  numpy as np
import os
import math
import traceback
import collections


"""

speed_labels_1 = np.array(map(int, map(float, open('speed_file_Town01_1.txt'))))


speed_labels_1_noise = np.array(map(int, map(float, open('speed_file_Town01_1_noise.txt'))))

speed_labels_2 = np.array(map(int, map(float, open('speed_file_Town02_14.txt'))))


speed_labels_2_noise = np.array(map(int, map(float, open('speed_file_Town02_14_noise.txt'))))
"""


def read_control_csv(control_csv_file):





    f = open(control_csv_file, "r")
    header = f.readline()
    header = header.split(',')
    header[-1] = header[-1][:-2]
    f.close()

    data_matrix = np.loadtxt(open(control_csv_file, "rb"), delimiter=",", skiprows=1)
    control_results_dic = {}
    count = 0
    # print data_matrix
    for step in data_matrix[:,0]:

        control_results_dic.update({step: data_matrix[count, 1:]})
        count += 1


    return control_results_dic




def _read_step_data_wp(step_path):
    """
    For this case we automatically get the steering angles from the waypoints.
    :param step_path:
    :return:
    """



    step_dictionary = {}


    steer_pred = 0.8 * np.loadtxt(step_path + '_seq_output_val.csv', delimiter=" ", skiprows=0, usecols=([3]))

    step_dictionary.update({'steer_pred': steer_pred})
    steer_gt = 0.8 * np.loadtxt(step_path + '_seq_gt_val.csv', delimiter=" ", skiprows=0, usecols=([3]))
    step_dictionary.update({'steer_gt': steer_gt})

    steer_error =  0.8 * np.loadtxt(step_path + '_seq_error_val.csv', delimiter=" ", skiprows=0, usecols=([3]))
    step_dictionary.update({'steer_error': steer_error})

    if 'Town01' in step_path and 'noise' in step_path:
        speed_input = speed_labels_1_noise
    elif 'Town01' in step_path:
        speed_input = speed_labels_1
    elif 'Town02' in step_path and 'noise' in step_path:
        speed_input = speed_labels_2_noise
    else:
        speed_input = speed_labels_2

    step_dictionary.update({'speed_input': speed_input})

    #speed_input =np.loadtxt(step_path + '_seq_B_4_input_val.csv', delimiter=" ", skiprows=0, usecols=([0]))
    #step_dictionary.update({'speed_input': speed_input})


    return step_dictionary



def _read_step_data(step_path):

    step_dictionary = {}


    steer_pred = np.loadtxt(step_path + '_seq_output_val.csv', delimiter=" ", skiprows=0, usecols=([0]))
    step_dictionary.update({'steer_pred': steer_pred})
    steer_gt = np.loadtxt(step_path + '_seq_gt_val.csv', delimiter=" ", skiprows=0, usecols=([0]))
    step_dictionary.update({'steer_gt': steer_gt})

    steer_error = np.loadtxt(step_path + '_seq_error_val.csv', delimiter=" ", skiprows=0, usecols=([0]))
    step_dictionary.update({'steer_error': steer_error})



    if 'Town01' in step_path and 'noise' in step_path:
        speed_input = speed_labels_1_noise
    elif 'Town01' in step_path:
        speed_input = speed_labels_1
    elif 'Town02' in step_path and 'noise' in step_path:
        speed_input = speed_labels_2_noise
    else:
        speed_input = speed_labels_2



    step_dictionary.update({'speed_input': speed_input})


    #speed_input =np.loadtxt(step_path + '_seq_B_4_input_val.csv', delimiter=" ", skiprows=0, usecols=([0]))
    #step_dictionary.update({'speed_input': speed_input})


    return step_dictionary





def _read_town_data(train_town_path,control_to_use):
    town_dictionary = {}

    # The word auto refers to the use of the autopilot

    # Try to read the controls
    # Some flags to check the control files found




    try:
        control = read_control_csv(os.path.join(train_town_path,'control_summary'+control_to_use+'.csv'))
    except KeyboardInterrupt:
        raise
    except:
        # HACKYYYYYY
        try:
            control = read_control_csv(os.path.join(train_town_path[:-6], 'control_summary' + control_to_use + '.csv'))
        except:
            traceback.print_exc()
            return None
    # Now we get the steps
    benchmarked_steps = control.items()

    # Simple extra counter




    for i in range(len(benchmarked_steps)):

        print (benchmarked_steps[i][0])
        step = int(benchmarked_steps[i][0])


        step_path = os.path.join(train_town_path,'raw', str(step))

        # First we try to add prediction data
        try:
            if '_wp_' in train_town_path:  # We test if the model is a waypoints based model, then we read a different part
                prediction_data = {step: _read_step_data_wp(step_path)}
            else:
                prediction_data = {step: _read_step_data(step_path)}

            town_dictionary.update(prediction_data)
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            return None


        print ('control'+control_to_use)
        print (benchmarked_steps[i][1])
        town_dictionary[step].update({'control': benchmarked_steps[i][1]})

    town_dictionary_ordered = collections.OrderedDict(sorted(town_dictionary.items()))

    return town_dictionary_ordered

