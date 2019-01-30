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

    # Download the full data from the models
    print ("Downloading the visualizer models checkpoints  253 MB")
    file_id = '1CT37umZ9lMzR-Nz8IfEi3qetSM30MtQV'
    destination_pack = 'town03_model.tar.gz'

    download_file_from_google_drive(file_id, destination_pack)
    destination_final = '_logs/town03/resnet34imnet/checkpoints'
    if not os.path.exists(destination_final):
        os.makedirs(destination_final)

    tf = tarfile.open("town03_model.tar.gz")
    tf.extractall(destination_final)
    # Remove both the original and the file after moving.
    os.remove("town03_model.tar.gz")