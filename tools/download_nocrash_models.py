
import os
import sys

import tarfile

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from tools.download_tools import download_file_from_google_drive

if __name__ == "__main__":

    # Download the full data from the models
    print ("Downloading the coil models checkpoints   1.1 GB")
    file_id = '1AzSIkmGETGSNLBtWxoTLGjqLlXNVgkEH'
    destination_pack = 'nocrash_basic.tar.xz'

    download_file_from_google_drive(file_id, destination_pack)
    destination_final = '_logs/'
    if not os.path.exists(destination_final):
        os.makedirs(destination_final)

    tf = tarfile.open("nocrash_basic.tar.xz")
    tf.extractall(destination_final)
    # Remove both the original and the file after moving.
    os.remove("nocrash_basic.tar.xz")

    # Now you copy each of the models for their respective folder.

