import wget
import os
import zipfile
import tarfile
import argparse
import requests

parser = argparse.ArgumentParser(description='Process some dataset')
parser.add_argument('--dataset_folder', type=str, default='dataset', help='Path to dataset folder')
args = parser.parse_args()
URL = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz"
# 2. download the data behind the URL
response = requests.get(URL)
# 3. Open the response into a new file called instagram.ico
open("file.tgz", "wb").write(response.content)
folder_path = args.dataset_folder
file = tarfile.open('file.tgz')
# extracting file
file.extractall(folder_path)
file.close()


