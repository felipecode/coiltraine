import os
import h5py
import cv2
import json
import numpy as np
import sys
from glob import glob


def process_and_save(h5, out):
    with h5py.File(h5, 'r') as H:
        cc = 0
        D = {}
        for t, r in zip(H['targets'], H['rgb']):
            t = t.astype('float64')
            D['steer'] = t[0]
            D['throttle'] = t[1]
            D['brake'] = t[2]
            D['hand_brake'] = t[3]
            D['reverse_gear'] = t[4]
            D['steer_noise'] = t[5]
            D['throttle_noise'] = t[6]
            D['brake_noise'] = t[7]
            D["playerMeasurements"] = {
             'forward_speed': 0.277778 * t[10]
            }
            #    "transform" = {
            #        "location" = {
            #            "x" =
            #            "y" =
            #        }
            #    }
            # }
            D['collision_other'] = t[12]
            D['collision_pedestrian'] = t[13]
            D['collision_car'] = t[14]
            D['opposite_lane_iter'] = t[15]
            D['platform_time'] = t[19]
            D['game_time'] = t[20]
            D['directions'] = t[24]
            D['noise'] = t[25]

            if t[26] == -30:
                img_name = "LeftRGB_{:05d}.png".format(cc)
            elif t[26] == 0:
                img_name = "CentralRGB_{:05d}.png".format(cc)
            elif t[26] == 30:
                img_name = "RightRGB_{:05d}.png".format(cc)
            else:
                print('something went wrong!')
                break
            img_name = os.path.join(out, img_name)
            m_name = os.path.join(out, "measurements_{:05d}.json".format(cc))

            # save
            cv2.imwrite(img_name, r)
            with open(m_name, 'w') as fp:
                json.dump(D, fp, indent=4, sort_keys=True)
            cc += 1


if __name__ == '__main__':
    assert len(sys.argv) == 3

    input_folder = sys.argv[1]  # folder with h5 files
    output_folder = sys.argv[2]  # final folder with episode folders
    os.system('mkdir {}'.format(output_folder))

    all_h5 = glob(os.path.join(input_folder, '*.h5'))

    for ep, h5 in enumerate(all_h5):
        out = os.path.join(output_folder, 'episode_{:05d}'.format(ep))
        os.system('mkdir {}'.format(out))
        process_and_save(h5, out)