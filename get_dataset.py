import wget
import os
import zipfile
import tarfile
import argparse

parser = argparse.ArgumentParser(description='Process some dataset')
parser.add_argument('--dataset_folder', type=str, default='dataset', help='Path to dataset folder')
args = parser.parse_args()

    
folder_path = args.dataset_folder

url = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz"
wget.download(url, "file.tgz")

file = tarfile.open('file.tgz')
  
# extracting file
file.extractall(folder_path)

folder_path = 'dataset'
image_ext = ('jpg', 'jpeg', 'png')
img_paths = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if f.endswith(image_ext)]

def get_folder():
    return img_paths

file.close()


