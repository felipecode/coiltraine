import pprint as pp
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style('whitegrid')
from data_reading import compute_avg_result_dict_ours, add_data_for_method


def plot_table(compared_results):


    for task, methods in compared_results.items():

        print("Task ", task)
        print("")
        # The matrix with all std values


        for method, envs in methods.items():


            print("  Method ", method)
            for env, metrics in envs.items():
                print("    env ", env)
                print("   ", end='')
                for metric, runs in metrics.items():
                    print(" $%.2f$ &" % np.mean(runs), end='')
                print("")



if __name__ == "__main__":
    scenario_dict = {'episodes_fully_completed': [],
                     'end_pedestrian_collision': [],
                     'end_vehicle_collision': [],
                     'end_other_collision': [],
                     'timeout': [],
                     'stopping': []
                     }

    compared_results = {
        'Empty':
            {"MT": {"Training": copy.deepcopy(scenario_dict),
                    "NewWeather": copy.deepcopy(scenario_dict),
                    "NewTown": copy.deepcopy(scenario_dict),
                    "NewWeatherTown": copy.deepcopy(scenario_dict)
                    },
             "CAL": {"Training": copy.deepcopy(scenario_dict),
                     "NewWeather": copy.deepcopy(scenario_dict),
                     "NewTown": copy.deepcopy(scenario_dict),
                     "NewWeatherTown": copy.deepcopy(scenario_dict)
                     },
             "CoIL": {"Training": copy.deepcopy(scenario_dict),
                      "NewWeather": copy.deepcopy(scenario_dict),
                      "NewTown": copy.deepcopy(scenario_dict),
                      "NewWeatherTown": copy.deepcopy(scenario_dict)
                      },
             },
        'Normal':
            {"MT": {"Training": copy.deepcopy(scenario_dict),
                    "NewWeather": copy.deepcopy(scenario_dict),
                    "NewTown": copy.deepcopy(scenario_dict),
                    "NewWeatherTown": copy.deepcopy(scenario_dict)
                    },
             "CAL": {"Training": copy.deepcopy(scenario_dict),
                     "NewWeather": copy.deepcopy(scenario_dict),
                     "NewTown": copy.deepcopy(scenario_dict),
                     "NewWeatherTown": copy.deepcopy(scenario_dict)
                     },
             "CoIL": {"Training": copy.deepcopy(scenario_dict),
                      "NewWeather": copy.deepcopy(scenario_dict),
                      "NewTown": copy.deepcopy(scenario_dict),
                      "NewWeatherTown": copy.deepcopy(scenario_dict)
                      },
             },
        'Cluttered':
            {"MT": {"Training": copy.deepcopy(scenario_dict),
                    "NewWeather": copy.deepcopy(scenario_dict),
                    "NewTown": copy.deepcopy(scenario_dict),
                    "NewWeatherTown": copy.deepcopy(scenario_dict)
                    },
             "CAL": {"Training": copy.deepcopy(scenario_dict),
                     "NewWeather": copy.deepcopy(scenario_dict),
                     "NewTown": copy.deepcopy(scenario_dict),
                     "NewWeatherTown": copy.deepcopy(scenario_dict)
                     },
             "CoIL": {"Training": copy.deepcopy(scenario_dict),
                      "NewWeather": copy.deepcopy(scenario_dict),
                      "NewTown": copy.deepcopy(scenario_dict),
                      "NewWeatherTown": copy.deepcopy(scenario_dict)
                      },
             },
    }

    MT_root = '/home/felipe/CoIL_TRI/_results_MT/_benchmarks_results'
    # Translate scenarios to weathers

    MT_bench_paths = {
        'Training': [os.path.join(MT_root, 'town02_test_LongitudinalControl2018_Town01')
                     # os.path.join(MT_root, 'test3_test_LongitudinalControl2018_Town01')
                     ],
        'NewWeather': [os.path.join(MT_root, 'town02_test_LongitudinalControl2018_Town01')
                       # os.path.join(MT_root, 'test3_test_LongitudinalControl2018_Town01')
                       ],
        'NewTown': [os.path.join(MT_root, 'town02_test_LongitudinalControl2018_Town02'),
                    os.path.join(MT_root, 'test2_LongitudinalControl2018_Town02'),
                    os.path.join(MT_root, 'test3_LongitudinalControl2018_Town02')
                    ],
        'NewWeatherTown': [os.path.join(MT_root, 'town02_test_LongitudinalControl2018_Town02'),
                           os.path.join(MT_root, 'test2_LongitudinalControl2018_Town02'),
                           os.path.join(MT_root, 'test3_LongitudinalControl2018_Town02')
                           ],
    }

    CAL_root = '/home/felipe/CoIL_TRI/_results_CAL/_benchmarks_results'
    CAL_bench_paths = {'Training': [os.path.join(CAL_root, 'test_LongitudinalControl2018_Town01'),
                                    os.path.join(CAL_root,
                                                 'test110_LongitudinalControl2018_Town01'),
                                    os.path.join(CAL_root, 'test320_LongitudinalControl2018_Town01')
                                    ],
                       'NewWeather': [os.path.join(CAL_root, 'test_LongitudinalControl2018_Town01'),
                                      os.path.join(CAL_root,
                                                   'test110_LongitudinalControl2018_Town01'),
                                      os.path.join(CAL_root,
                                                   'test320_LongitudinalControl2018_Town01')
                                      ],
                       'NewTown': [os.path.join(CAL_root, 'test_LongitudinalControl2018_Town02'),
                                   os.path.join(CAL_root, 'test220_LongitudinalControl2018_Town02'),
                                   os.path.join(CAL_root, 'test320_LongitudinalControl2018_Town02')
                                   ],
                       'NewWeatherTown': [
                           os.path.join(CAL_root, 'test_LongitudinalControl2018_Town02'),
                           os.path.join(CAL_root, 'test220_LongitudinalControl2018_Town02'),
                           os.path.join(CAL_root, 'test320_LongitudinalControl2018_Town02')
                       ],
                       }

    IL_root = '/home/felipe/CoIL_TRI/_results_IL/_benchmarks_results'
    IL_bench_paths = {'Training': [os.path.join(IL_root, 'test_LongitudinalControl2018_Town01'),
                                   os.path.join(IL_root, 'test2_LongitudinalControl2018_Town01')
                                   # os.path.join(IL_root, 'test110_LongitudinalControl2018_Town01'),
                                   # os.path.join(IL_root, 'test320_LongitudinalControl2018_Town01')
                                   ],
                      'NewWeather': [os.path.join(IL_root, 'test_LongitudinalControl2018_Town01'),
                                     os.path.join(IL_root, 'test2_LongitudinalControl2018_Town01')
                                     # os.path.join(IL_root, 'test110_LongitudinalControl2018_Town01'),
                                     # os.path.join(IL_root, 'test320_LongitudinalControl2018_Town01')
                                     ],
                      'NewTown': [os.path.join(IL_root, 'test_LongitudinalControl2018_Town02'),
                                  os.path.join(IL_root, 'test2_LongitudinalControl2018_Town02'),
                                  os.path.join(IL_root, 'test3_LongitudinalControl2018_Town02')
                                  ],
                      'NewWeatherTown': [
                          os.path.join(IL_root, 'test_LongitudinalControl2018_Town02'),
                          os.path.join(IL_root, 'test2_LongitudinalControl2018_Town02'),
                          os.path.join(IL_root, 'test3_LongitudinalControl2018_Town02')
                      ],
                      }


    root_path = '/home/felipe/CoIL_TRI/_logs'




    res34_10 = [
                'valstop_ablationspeed_seed1/' + 'res34-10-lowdropout-imnet',
                'valstop_ablationspeed_seed2/' + 'res34-10-lowdropout-imnet',
                'valstop_ablationspeed_seed3/' + 'res34-10-lowdropout-imnet',
                'valstop_ablationspeed_seed4/' + 'res34-10-lowdropout-imnet' ]

    res34_100 = [
        'cvprfinal_valstop_seed1/' + 'res34-100-lowdropout-imnet',
        'cvprfinal_valstop_seed2/' + 'res34-100-lowdropout-imnet',
        #'cvprfinal_valstop_seed3/' + 'res34-100-lowdropout-imnet',
        'cvprfinal_valstop_seed4/' + 'res34-100-lowdropout-imnet',
        'cvprfinal_valstop_seed5/' + 'res34-100-lowdropout-imnet']
    ns_10 = [
                'valstop_abspeed3_seed1/' + 'res34-10-lowdropout-imnet-nospeed',
                'valstop_abspeed3_seed2/' + 'res34-10-lowdropout-imnet-nospeed',
                'valstop_abspeed3_seed3/' + 'res34-10-lowdropout-imnet-nospeed',
                'valstop_abspeed3_seed4/' + 'res34-10-lowdropout-imnet-nospeed' ]


    res34_10 = [os.path.join(root_path, name) for name in res34_10]
    res34_100 = [os.path.join(root_path, name) for name in res34_100]
    ns_10 = [os.path.join(root_path, name) for name in ns_10]

    res34_10_results_max = compute_avg_result_dict_ours(res34_10, 'Long3', maximum=True)
    res34_10_results_min = compute_avg_result_dict_ours(res34_10, 'Long3', maximum=False)
    ns_10_results_max = compute_avg_result_dict_ours(ns_10, 'Long', maximum=True)
    ns_10_results_min = compute_avg_result_dict_ours(ns_10, 'Long', maximum=False)
    res34_100_results_max = compute_avg_result_dict_ours(res34_100, 'Long3', maximum=True)
    res34_100_results_min = compute_avg_result_dict_ours(res34_100, 'Long3', maximum=False)

    for task in compared_results.keys():
        compared_results[task].update({'res34_10_max': res34_10_results_max[task]})
        compared_results[task].update({'res34_10_min': res34_10_results_min[task]})
        compared_results[task].update({'res34_100_max': res34_100_results_max[task]})
        compared_results[task].update({'res34_100_min': res34_100_results_min[task]})
        compared_results[task].update({'ns_10_max': ns_10_results_max[task]})
        compared_results[task].update({'ns_10_min': ns_10_results_min[task]})

    add_data_for_method(compared_results, paths_dict=MT_bench_paths, method_name='MT')
    add_data_for_method(compared_results, paths_dict=CAL_bench_paths, method_name='CAL')
    add_data_for_method(compared_results, paths_dict=IL_bench_paths, method_name='CoIL')

    plot_table(compared_results)