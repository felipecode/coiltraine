import requests
import os
import sys

import tarfile

PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from .download_tools import download_file_from_google_drive

if __name__ == "__main__":

    # Download the full data from the models
    print ("Downloading the coil models checkpoints  500 MB")
    file_id = '1ynh2V6FMpC7NLXX2kbuxSHb3ymu7fp-r'
    destination_pack = 'coil_view_models.tar.gz'

    download_file_from_google_drive(file_id, destination_pack)
    destination_final = '_logs/'
    if not os.path.exists(destination_final):
        os.makedirs(destination_final)

    tf = tarfile.open("coil_view_models.tar.gz")
    tf.extractall(destination_final)
    # Remove both the original and the file after moving.
    os.remove("coil_view_models.tar.gz")

    # Now you move the two models for their respective folders
    # The 320000.pth model is from the town01/02 model
    distination_town02 = '_logs/nocrash/resnet34imnet10/checkpoints/'
    if not os.path.exists(distination_town02):
        os.makedirs(distination_town02)
    os.rename("_logs/320000.pth", distination_town02 + '320000.pth')

    # The 200000.pth  is from the
    distination_town03 = '_logs/town03/resnet34imnet/checkpoints/'
    if not os.path.exists(distination_town03):
        os.makedirs(distination_town03)
    os.rename("_logs/200000.pth", distination_town03 + '200000.pth')

