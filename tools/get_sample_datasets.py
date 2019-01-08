import requests
import os
import sys

import tarfile


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)    


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    total_downloaded = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                total_downloaded += CHUNK_SIZE
                sys.stdout.write("Downloaded " + str(total_downloaded))
                sys.stdout.flush()
                sys.stdout.write("\r")
                f.write(chunk)


if __name__ == "__main__":
    try:
        path = os.environ["COIL_DATASET_PATH"]
    except KeyError as e:
        print("")
        print("COIL_DATASET_PATH env variable must be defined.")
        print("")
        raise e

    # Download the datasets
    file_id = '1cXX7Pdxfkz5MD6oMjbmNlXDkUIAxPd6m'
    destination_pack = 'COiLTRAiNESampleDatasets.tar.gz'

    print("Downloading on training an two validations datasets (7GB total)")
    download_file_from_google_drive(file_id, destination_pack)
    destination_final = os.path.join("~/", os.environ["COIL_DATASET_PATH"])
    if not os.path.exists(destination_final):
        os.makedirs(destination_final)

    print("Unpacking the dataset")

    tf = tarfile.open("COiLTRAiNESampleDatasets.tar.gz")
    tf.extractall(destination_final)

    os.remove("COiLTRAiNESampleDatasets.tar.gz")
