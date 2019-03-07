
import os
import sys

import tarfile

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from tools.download_tools import download_file_from_google_drive

if __name__ == "__main__":

    # Download the full data from the models
    print ("Downloading the coil models checkpoints  224 MB")
    file_id = '1NQuGRGb0b8zV916rQaAmG2bptJsbS3Ry'
    destination_pack = 'track2_baseline.tar.gz'

    download_file_from_google_drive(file_id, destination_pack)
    destination_final = '_logs/'
    if not os.path.exists(destination_final):
        os.makedirs(destination_final)

    tf = tarfile.open("track2_baseline.tar.gz")
    tf.extractall(destination_final)
    # Remove both the original and the file after moving.
    os.remove("track2_baseline.tar.gz")

    # Now you move the two models for their respective folders
    # The 180000.pth model is the checkpoint
    distination_town02 = '_logs/baselines/resnet34imnet/checkpoints/'
    if not os.path.exists(distination_town02):
        os.makedirs(distination_town02)
    os.rename("_logs/180000.pth", distination_town02 + '180000.pth')


