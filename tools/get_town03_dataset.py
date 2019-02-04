import os
import tarfile

from .download_tools import download_file_from_google_drive



if __name__ == "__main__":
    try:
        path = os.environ["COIL_DATASET_PATH"]
    except KeyError as e:
        print("")
        print("COIL_DATASET_PATH env variable must be defined.")
        print("")
        raise e

    # Download the datasets
    file_id = '1IrG6i61kgVgygUgnpP2e6kF2-JtpoisW'
    destination_pack = 'CoILTrainTown03_2.tar.gz'

    print("Downloading the training dataset for Town03  (6GB total)")
    download_file_from_google_drive(file_id, destination_pack)
    destination_final = os.path.join("~/", os.environ["COIL_DATASET_PATH"])
    if not os.path.exists(destination_final):
        os.makedirs(destination_final)

    print("Unpacking the dataset (takes a few minutes)")

    tf = tarfile.open("CoILTrainTown03_2.tar.gz")
    tf.extractall(destination_final)

    os.remove("CoILTrainTown03_2.tar.gz")