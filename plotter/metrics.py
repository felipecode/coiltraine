import numpy as np
import math

def compute_and_aggregate(metric_func, data, param):
    # metric_func = locals()['compute_' + metric_name]
    metric_results = []
    for step, data_item in data['values'].items():
            # print(metric_func)
            metric_raw = metric_func(data_item, param)
            # print(metric_raw)
            if 'aggregate' not in param:
                param['aggregate'] = {}
            metric_results.append(aggregate_metric(metric_raw, param['aggregate']))
    return metric_results


def aggregate_metric(metric_raw, param):
    if np.isscalar(metric_raw):
        assert not param , 'There should be no aggregate param for metrics which are already scalar'
        metric_result = metric_raw
    elif np.any(np.isnan(metric_raw)) or np.any(np.isinf(metric_raw)):
        metric_result = np.nan
    else:
        # by default we average
        if not param:
            param['type'] = 'mean'

        metric_raw_np = np.array(metric_raw)

        if 'condition' in param:
            metric_mask = param['condition'](metric_raw_np)
        else:
            metric_mask = np.ones(len(metric_raw_np), dtype=np.bool)

        if param['type'] == 'mean':
            metric_result = np.mean(metric_raw_np[metric_mask])
        elif param['type'] == 'percentile':
            metric_result = np.percentile(metric_raw_np[metric_mask], param['percentile'])
        elif param['type'] == 'count':
            assert ('condition' in param), 'Need a condition to compute a counting metric'
            metric_result = float(np.sum(metric_mask)) / float(len(metric_raw_np)) # param['condition'] should be a lambda expression
        else:
            raise Exception('Unknown aggregation type', param['type'])
    return metric_result


def compute_id(data, param):
    if data['town'] == 'Town01':
        return [0]*len(data['values'].items())
    elif data['town'] == 'Town02':
        return [1]*len(data['values'].items())
    else:
        raise Exception('Unknown town', data['town'])


def compute_experiment(data, param):
    return [hash(data['experiment']) % 50] * len(data['values'].items())


def compute_step(data, param):
    step_data = []
    for step, data_item in data['values'].items():
        step_data.append(float(step)/1000.0)
    return step_data

def compute_control_accuracy(data_item, param):
    print('Better use the compute_control_success_rate function - it does the same but has a better name')
    return data_item['control'][-2]

# this is a copy of compute_control_accuracy but with a better name
def compute_control_success_rate(data_item, param):
    return data_item['control'][-2]

def compute_control_average_completion(data_item, param):
    # print('data_item[\'control\']', data_item['control'])
    return data_item['control'][0]

def compute_km_per_infraction(data_item, param):
    label_to_index = {'intersection_offroad':1, 'intersection_otherlane': 2, 'collision_pedestrians': 3, 'collision_vehicles': 4}

    km_run = data_item['control'][-1]
    infraction_vec = np.array(data_item['control'][2:6])
    infraction_vec = infraction_vec/km_run

    infraction_vec[infraction_vec==0] = 0.01
    km_per_infraction = np.sum(infraction_vec)*km_run
    return km_per_infraction


float_formatter = lambda x: "%.2f" % x

def compute_steering_error(data_item, param):
    #np.savetxt('80maug.csv', data_item['steer_gt'] - data_item['steer_pred'], delimiter='', fmt='%-10.06f')

    #np.set_printoptions(threshold=np.nan, formatter={'float_kind': float_formatter})
    #print (abs(data_item['steer_gt'] - data_item['steer_pred'])[0:100])
    return abs(data_item['steer_gt'] - data_item['steer_pred'])

def compute_steering_error_filter_gt(data_item, param):
    error_np = np.array(abs(data_item['steer_gt'] - data_item['steer_pred']))
    gt_np = np.array(data_item['steer_gt'])
    gt_mask = param['gt_condition'](gt_np)
    return error_np[gt_mask]

def compute_steering_accuracy(data_item, param):
    pred_np = np.array(data_item['steer_pred'])
    gt_np = np.array(data_item['steer_gt'])
    thresh = param['threshold']
    pred_np = np.digitize(pred_np, [-thresh, thresh])
    gt_np = np.digitize(gt_np, [-thresh, thresh])
    matches = (pred_np == gt_np).astype(np.float)
    return matches

def compute_steering_classification_error(data_item, param):
    pred_np = np.array(data_item['steer_pred'])
    gt_np = np.array(data_item['steer_gt'])
    thresh = param['threshold']
    pred_np = np.digitize(pred_np, [-thresh, thresh])
    gt_np = np.digitize(gt_np, [-thresh, thresh])
    matches = 1. - (pred_np == gt_np).astype(np.float)
    return matches

def compute_steering_accuracy_filter_gt(data_item, param):
    pred_np = np.array(data_item['steer_pred'])
    gt_np = np.array(data_item['steer_gt'])
    thresh = param['threshold']
    gt_mask = param['gt_condition'](gt_np)
    pred_np = np.digitize(pred_np[gt_mask], [-thresh, thresh])
    gt_np = np.digitize(gt_np[gt_mask], [-thresh, thresh])
    matches = (pred_np == gt_np).astype(np.float)
    return matches

def compute_steering_avg_l1(data_item, param):
    return abs(data_item['steer_gt'] - data_item['steer_pred'])

def compute_steering_avg_l1_speed(data_item, param):
    steer_gt =data_item['steer_gt'][np.where(data_item['speed_input'] > (param['thresh_speed']*40))]
    steer_pred = data_item['steer_pred'][np.where(data_item['speed_input'] > (param['thresh_speed']*40))]

    return np.mean(abs(steer_gt - steer_pred))

def compute_steering_avg_mse(data_item, param):
    return (data_item['steer_gt'] - data_item['steer_pred'])**2

def compute_steering_avg_mse_filter_gt(data_item, param):
    error_np = np.array(abs(data_item['steer_gt'] - data_item['steer_pred']))
    gt_np = np.array(data_item['steer_gt'])
    gt_mask = param['gt_condition'](gt_np)
    return error_np[gt_mask]**2

def compute_displacement(data_item, param):
    return np.multiply(abs(data_item['steer_gt'] - data_item['steer_pred']),np.absolute(data_item['speed_input']))

def compute_displacement_steer(data_item, param):
    steer_gt =data_item['steer_gt'][np.where(abs(data_item['steer_gt']) > (param['thresh_steer']))]
    steer_pred = data_item['steer_pred'][np.where(abs(data_item['steer_gt']) > (param['thresh_steer']))]

    return np.mean(abs(steer_gt - steer_pred))

def compute_cumulative_displacement(data_item, param):
    np.set_printoptions(threshold=np.nan)
    cummulative_displacement_vec = []
    for i in range(0, len(data_item['steer_pred']) - param['window']):
        displacement_vec_pred = [steer * math.fabs(speed) * param['timestep'] for steer, speed in
                            zip(data_item['steer_pred'][(i):(i + param['window'])], data_item['speed_input'][(i):(i + param['window'])])]
        displacement_vec_gt = [steer * math.fabs(speed) * param['timestep'] for steer, speed in
                            zip(data_item['steer_gt'][(i):(i + param['window'])], data_item['speed_input'][(i):(i + param['window'])])]
        cummulative_displacement_vec.append(math.fabs(sum(displacement_vec_pred)-sum(displacement_vec_gt)))

    return cummulative_displacement_vec


def compute_correlation(data_item, param):
    def calc_score(gts_ann, res_ann):
        """
        Computer CC score. A simple implementation
        :param gts_ann : ground-truth fixation map
        :param res_ann : predicted saliency map
        :return score: int : score
        """

        fixation_map = gts_ann - np.mean(gts_ann)
        if np.max(fixation_map) > 0:
            fixation_map = fixation_map / np.std(fixation_map)
        sal_map = res_ann - np.mean(res_ann)
        if np.max(sal_map) > 0:
            sal_map = sal_map / np.std(sal_map)

        return np.corrcoef(sal_map.reshape(-1), fixation_map.reshape(-1))[0][1]

    return calc_score(data_item['steer_pred']*np.absolute(data_item['speed_input'])
                      , data_item['steer_gt']*np.absolute(data_item['speed_input']))



def compute_count_errors_weighted(data_item,param):
    weighted_gt = np.multiply(np.abs(data_item['steer_gt']), param['coeff'])

    count = float(sum(abs(data_item['steer_gt'] - data_item['steer_pred']) > weighted_gt)) / float(len(weighted_gt))

    if np.isnan(np.sum(abs(data_item['steer_gt'] - data_item['steer_pred']))):
        count = 1.0

    #nans = np.isnan(abs(data_item['steer_gt'] - data_item['steer_pred']))
    #abs(data_item['steer_gt'] - data_item['steer_pred']) = abs(data_item['steer_gt'] - data_item['steer_pred'])[np.invert(nans)]
    # For sure, temporality is important. Even taking into account this naive idea already can be very good

    return count

def compute_relative_error_smoothed(data_item,param):
    assert param['steer_smooth'] > 1e-6, 'Smooth parameter must be at least 1e-8 to avoid numerical problems'

    return abs(data_item['steer_gt'] - data_item['steer_pred']) / (np.abs(data_item['steer_gt']) + param['steer_smooth'])


def compute_count_errors_weighted_speed(data_item,param):
    speed_weighted_coeff = param['coeff'] * np.sqrt(np.absolute(data_item['speed_input'])* (1.0/40.0)) # YOu divide by the maximun speed

    weighted_gt = np.multiply(data_item['steer_gt'], speed_weighted_coeff)


    count = float(sum(abs(data_item['steer_gt'] - data_item['steer_pred']) > weighted_gt)) / float(len(weighted_gt))
    if np.isnan(np.sum(abs(data_item['steer_gt'] - data_item['steer_pred']))):
        count = 1.0

    return count




def compute_count_cumulative_displacement(data_item, param):


    # We get the list of cameras that were previously computed

    # Remember 0 must be set as central camera

    #print ('list_cameras'+'_' +  data['town'])
    #print param['list_cameras'+'_' +  data['town']]

    # COULD BE SPEED UP


    central_steer_gt = data_item['steer_gt']

    central_steer_pred = data_item['steer_pred']
    central_speed_input = data_item['speed_input']


    central_steer_error = central_steer_pred - central_steer_gt


    cummulative_displacement_vec = []
    cummulative_ground_truth_disp = []
    for i in range(0, len(central_steer_error) - param['window']):

        displacement_vec = [steer * speed * param['timestep']  for steer, speed in   zip(central_steer_pred[(i):(i + param['window'])],
                           central_speed_input[(i):(i + param['window'])])]

        ground_truth_disp = [steer * speed * param['timestep']  for steer, speed in   zip(central_steer_gt[(i):(i + param['window'])],
                           central_speed_input[(i):(i + param['window'])])]


        cummulative_displacement_vec.append(sum(displacement_vec))
        cummulative_ground_truth_disp.append(sum(ground_truth_disp))


    return float(sum(cummulative_displacement_vec > (cummulative_ground_truth_disp*param['coeff']))) / float(len(cummulative_displacement_vec))
