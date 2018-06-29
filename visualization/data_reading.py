import  numpy as np
import os
import math
import traceback
import collections

from configs import g_conf
from utils.general import static_vars

"""

speed_labels_1 = np.array(map(int, map(float, open('speed_file_Town01_1.txt'))))


speed_labels_1_noise = np.array(map(int, map(float, open('speed_file_Town01_1_noise.txt'))))

speed_labels_2 = np.array(map(int, map(float, open('speed_file_Town02_14.txt'))))


speed_labels_2_noise = np.array(map(int, map(float, open('speed_file_Town02_14_noise.txt'))))
"""

def read_summary_csv(control_csv_file):



    f = open(control_csv_file, "rU")
    header = f.readline()
    header = header.split(',')
    header[-1] = header[-1][:-2]
    f.close()

    data_matrix = np.loadtxt(control_csv_file, delimiter=",", skiprows=1)
    summary_dict = {}

    if len(data_matrix) == 0:
        return None

    if len(data_matrix.shape) == 1:
        data_matrix = np.expand_dims(data_matrix, axis=0)

    count = 0
    for _ in header:
        summary_dict.update({header[count]: data_matrix[:, count]})
        count += 1



    return summary_dict




def read_control_csv(control_csv_file):

    # If the file does not exist, return None,None, to point out that data is missing
    if not os.path.exists(control_csv_file):
        return None, None

    f = open(control_csv_file, "r")
    header = f.readline()
    header = header.split(',')
    header[-1] = header[-1][:-2]
    f.close()

    data_matrix = np.loadtxt(open(control_csv_file, "rb"), delimiter=",", skiprows=1)
    control_results_dic = {}
    count = 0

    if len(data_matrix) == 0:
        return None, None
    if len(data_matrix.shape) == 1:
        data_matrix = np.expand_dims(data_matrix, axis=0)

    for step in data_matrix[:, 0]:

        control_results_dic.update({step: data_matrix[count, 1:]})
        count += 1


    return control_results_dic, header




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

# Add a static variable to avoid re-reading
@static_vars(previous_ground_truth={})
def get_ground_truth(dataset_name):

    if dataset_name not in get_ground_truth.previous_ground_truth:
        full_path = os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name, 'ground_truth.csv')
        ground_truth = np.loadtxt(full_path, delimiter=",", skiprows=0, usecols=([0]))
        get_ground_truth.previous_ground_truth.update({dataset_name :ground_truth})

    return get_ground_truth.previous_ground_truth[dataset_name]


@static_vars(previous_speed_ground_truth=None)
def get_speed_ground_truth(dataset_name):
    if get_speed_ground_truth.previous_speed_ground_truth is None:
        full_path = os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name,
                                 'speed_ground_truth.csv')
        speed_ground_truth = np.loadtxt(full_path, delimiter=",", skiprows=0, usecols=([0]))
        get_speed_ground_truth.previous_speed_ground_truth = speed_ground_truth

    return get_speed_ground_truth.previous_speed_ground_truth

@static_vars(previous_camera_labels={})
def get_camera_labels(dataset_name):

    if dataset_name not in get_camera_labels.previous_camera_labels:
        full_path = os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name,
                                 'camera_labels.csv')
        camera_labels = np.loadtxt(full_path, delimiter=",", skiprows=0, usecols=([0]))
        get_camera_labels.previous_camera_labels.update({dataset_name: camera_labels })

    return get_camera_labels.previous_camera_labels[dataset_name]




def _read_step_data(step_path):


    val_dataset_name = step_path.split('/')[-2].split('_')[-2]

    print ("val_dataset_name", val_dataset_name)

    step_dictionary = {}

    # On this step we read all predictions for this benchmark step.
    predictions = np.loadtxt(step_path, delimiter=",", skiprows=0, usecols=([0]))

    # Get the ground truth directly from the datasets path with the already generated steer and speed
    ground_truth = get_ground_truth(val_dataset_name)

    step_dictionary.update({'steer_pred': predictions})



    #steer_gt = np.loadtxt(step_path + '_seq_gt_val.csv', delimiter=" ", skiprows=0, usecols=([0]))
    step_dictionary.update({'steer_gt': ground_truth})

    #steer_error = np.loadtxt(step_path + '_seq_error_val.csv', delimiter=" ", skiprows=0, usecols=([0]))
    #step_dictionary.update({'steer_error': compute_error(predictions, ground_truth)})



    """
    if 'Town01' in step_path and 'noise' in step_path:
        speed_input = speed_labels_1_noise
    elif 'Town01' in step_path:
        speed_input = speed_labels_1
    elif 'Town02' in step_path and 'noise' in step_path:
        speed_input = speed_labels_2_noise
    else:
        speed_input = speed_labels_2
    """


    step_dictionary.update({'speed_input': get_speed_ground_truth(val_dataset_name)})




    return step_dictionary





def _read_control_data(full_path, control_to_use):


    # The word auto refers to the use of the autopilot

    # Try to read the controls
    # Some flags to check the control files found

    try:
        control, _ = read_control_csv(os.path.join(full_path, 'control_output' + control_to_use +
                                                   '.csv'))
    except:
        raise ValueError("exception on control_csv reading full_path = %s,  " % (full_path))


    # resend the none to eliminate this dataset
    if control is None:
        return control

    return list(control.items())
    # Simple extra counter


def _read_data(full_path, benchmarked_steps):
    town_dictionary = {}

    print (benchmarked_steps)

    for i in range(len(benchmarked_steps)):

        print (benchmarked_steps[i][0])
        step = int(benchmarked_steps[i][0])


        step_path = os.path.join(full_path, str(step) + '.csv')
        print (step_path)

        # First we try to add prediction data
        try:
            # Check for waypoints.
            if '_wp_' in full_path:  # We test if the model is a waypoints based model, then we read a different part
                prediction_data = {step: _read_step_data_wp(step_path)}
            else:
                prediction_data = {step: _read_step_data(step_path)}


            town_dictionary.update(prediction_data)
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            return None


        town_dictionary[step].update({'control': benchmarked_steps[i][1]})

    town_dictionary_ordered = collections.OrderedDict(sorted(town_dictionary.items()))

    return town_dictionary_ordered

