import numpy as np
import os
import math
import traceback
import collections

from coilutils.general import static_vars


def augment_steering(camera_angle, steer, speed):
    """
        Apply the steering physical equation to augment for the lateral cameras.
    Args:
        camera_angle_batch:
        steer_batch:
        speed_batch:
    Returns:
        the augmented steering
    """

    time_use = 1.0
    car_length = 6.0
    old_steer = steer
    pos = camera_angle > 0.0
    neg = camera_angle <= 0.0
    # You should use the absolute value of speed
    speed = math.fabs(speed)
    rad_camera_angle = math.radians(math.fabs(camera_angle))
    val = 6 * (
        math.atan((rad_camera_angle * car_length) / (time_use * speed + 0.05))) / 3.1415
    steer -= pos * min(val, 0.3)
    steer += neg * min(val, 0.3)

    steer = min(1.0, max(-1.0, steer))

    # print('Angle', camera_angle, ' Steer ', old_steer, ' speed ', speed, 'new steer', steer)
    return steer


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


def read_summary_tasks_csv(control_csv_file):


    f = open(control_csv_file, "rU")
    header = f.readline()
    header = header.split(',')
    header[-1] = header[-1][:-2]
    f.close()

    print (header)

    data_matrix = np.loadtxt(control_csv_file, delimiter=",", skiprows=1)
    summary_dict = {}

    if len(data_matrix) == 0:
        return None

    if len(data_matrix.shape) == 1:
        data_matrix = np.expand_dims(data_matrix, axis=0)


    task_list = []
    for task in range(len(data_matrix)):
        task_list.append(data_matrix[task, header.index('task')])

    count = 0

    for key in set(task_list):
        task_dict = {}
        for name in header:
            if name == 'task':
                  continue

            # The list of values from this collumn of the matrix
            value_list = []
            for step in range(len(data_matrix)):
                if data_matrix[step, header.index('task')] == key:

                    value_list.append(data_matrix[step, count])

                task_dict.update({header[count]: value_list})

            count += 1

        summary_dict.update({key: task_dict})



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


# TODO add csv generation directly here, if it was not generated.

# Add a static variable to avoid re-reading
@static_vars(previous_ground_truth={})
def get_ground_truth(dataset_name):

    if dataset_name not in get_ground_truth.previous_ground_truth:
        full_path = os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name, 'ground_truth.csv')
        ground_truth = np.loadtxt(full_path, delimiter=",", skiprows=0, usecols=([0]))
        get_ground_truth.previous_ground_truth.update({dataset_name :ground_truth})

    return get_ground_truth.previous_ground_truth[dataset_name]


@static_vars(previous_speed_ground_truth={})
def get_speed_ground_truth(dataset_name):


    if dataset_name not in get_speed_ground_truth.previous_speed_ground_truth:
        full_path = os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name,
                                 'speed_ground_truth.csv')
        speed_ground_truth = np.loadtxt(full_path, delimiter=",", skiprows=0, usecols=([0]))
        get_speed_ground_truth.previous_speed_ground_truth.update({dataset_name: speed_ground_truth})


    return get_speed_ground_truth.previous_speed_ground_truth[dataset_name]

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
        raise ValueError("exception on control_csv reading full_path = %s,  " % full_path)

    # resend the none to eliminate this dataset
    if control is None:
        return control
    return list(control.items())
    # Simple extra counter




def _read_data(full_path, benchmarked_steps):
    town_dictionary = {}

    for i in range(len(benchmarked_steps)):
        step = int(benchmarked_steps[i][0])
        step_path = os.path.join(full_path, str(step) + '.csv')
        # First we try to add prediction data
        try:
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

