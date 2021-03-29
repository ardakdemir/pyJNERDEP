import subprocess
import sys
import gdown
import tqdm
import zipdir
from zipfile import ZipFile
import argparse
import os

id_model_map = {"SA": "1Yt6MVemyNDlv0bMrjEEgGSjZ7P42bW0t",
                "NER": "",
                "DEP": "",
                "FLAT": ""}
names = {"SA": "Sentiment Analysis",
         "NER": "Named Entity Recognition",
         "DEP": "Dependency Parsing",
         "FLAT": "Multi-task Learning"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default='SA',
                        help='key for downloading the models')
    parser.add_argument('--save_folder', type=str, default=None,
                        help='Destination to save downloaded models')
    args = parser.parse_args()
    return args


def unzip(src, dest):
    with ZipFile(src, 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        zipObj.extractall(dest)
        print('File is unzipped in {} folder'.format(dest))


def download_link_generator(id):
    return "https://drive.google.com/uc?id={}".format(id)


def load_download_models(key, save_folder=None):
    id = id_model_map[key]
    print("\n===Downloading trained {} models to replicate the result===\n".format(names[key]))
    link = download_link_generator(id)
    dest = "../{}_models".format(key)
    unzip_path = "../{}".format(dest) if not save_folder else save_folder
    if not os.path.exists(unzip_path):
        print("{} not found. Downloading trained models for {}".format(unzip_path,key))
        gdown.download(link, dest)
        unzip(dest, unzip_path)
        print("Trained models are stored in {}".format(unzip_path))
        return unzip_path
    else:
        print("Models for {} are already downloaded.".format(key))
        return unzip_path


def main():
    args = parse_args()
    key = args.key
    save_folder = args.save_folder
    content = os.listdir(save_folder)
    print("Content of the save model folder: {}".format(content))
    load_download_models(key, save_folder)


if __name__ == "__main__":
    main()
