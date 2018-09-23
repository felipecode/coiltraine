import os
import h5py
import cv2
import json
import re
import numpy as np
import sys
from glob import glob


def tryint(s):
    try:
        return int(s)
    except:
        return s

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)



def process_and_save(h5, out, cc):
    read_left = 0
    read_right = 0
    read_center = 0
    with h5py.File(h5, 'r') as H:
        D = {}

        for t, r, d, s in zip(H['targets'], H['rgb'], H['depth'], H['labels']):
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
                "forwardSpeed": 0.277778 * t[10]
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
            D['stop_pedestrian'] = 1
            D['stop_vehicle'] = 1
            D['stop_traffic_lights'] = 1

            if t[26] == -30:
                read_left += 1
                img_name = "LeftRGB_{:05d}.png".format(cc)

            elif t[26] == 0:
                read_center += 1
                img_name = "CentralRGB_{:05d}.png".format(cc)
            elif t[26] == 30:
                read_right += 1
                img_name = "RightRGB_{:05d}.png".format(cc)
            else:
                print('something went wrong!')
                break
            img_name = os.path.join(out, img_name)
            m_name = os.path.join(out, "measurements_{:05d}.json".format(cc))
            r = r[:, :, ::-1]
            # save
            cv2.imwrite(img_name, r)
            with open(m_name, 'w') as fp:
                json.dump(D, fp, indent=4, sort_keys=True)
            if read_right == read_left and read_right == read_center:
                cc += 1

    return cc

if __name__ == '__main__':
    assert len(sys.argv) == 3

    input_folder = sys.argv[1]  # folder with h5 files
    output_folder = sys.argv[2]  # final folder with episode folders
    os.system('mkdir {}'.format(output_folder))

    all_h5 = glob(os.path.join(input_folder, '*.h5'))
    sort_nicely(all_h5)
    c_ep = 0
    for ep, h5 in enumerate(all_h5):
        print(ep)
        #if ep < 15000:

        #    if ep % 20 == 0:
        #        c_ep += 1

        #    continue

        if ep % 20 == 0:

            cc = 0
            out = os.path.join(output_folder, 'episode_{:05d}'.format(c_ep))
            os.system('mkdir {}'.format(out))
            c_ep += 1

        try:
            cc = process_and_save(h5, out, cc)
        except KeyboardInterrupt:
            break
        except:
            pass
