import argparse
from PIL import Image
import numpy as np
import os

from shutil import copyfile




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NPY viewer')
    # parser.add_argument('model', type=str, help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('-f', '--file', default="")

    test_images_write_path = 'tools/_test_images_'


    args = parser.parse_args()
    preload_name = args.file

    sensor_data_names, measurements = np.load(os.path.join('../_preloads', preload_name + '.npy'))

    if not os.path.exists(test_images_write_path + preload_name):
        os.mkdir(test_images_write_path + preload_name)


    for i in range(len(measurements)):

        img_path = os.path.join(os.environ["COIL_DATASET_PATH"], preload_name,  # Make this preload name better
                                sensor_data_names[i].split('/')[-2],
                                sensor_data_names[i].split('/')[-1])

        copyfile(img_path,
                 os.path.join(test_images_write_path + preload_name, str(i) + '.png'))
        print (' imager ', i)

