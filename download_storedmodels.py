import subprocess
import sys
import gdown
import tqdm
import zipdir
from zipfile import ZipFile
import argparse

id_model_map = {"sa": "1Yt6MVemyNDlv0bMrjEEgGSjZ7P42bW0t",
                "ner": "",
                "dep": "",
                "flat": ""}
names = {"sa": "Sentiment Analysis",
         "ner": "Named Entity Recognition",
         "dep": "Dependency Parsing",
         "flat": "Multi-task Learning"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sa_train_file', type=str, default='../../datasets/sa_twitter_turkish-train.json',
                        help='training file for sa')
    parser.add_argument('--sa_dev_file', type=str, default='../../datasets/sa_twitter_turkish-dev.json',
                        help='validation file for sa')
    parser.add_argument('--sa_test_file', type=str, default='../../datasets/sa_twitter_turkish-test.json',
                        help='test file for sa'
    args = parser.parse_args()
    return args


def unzip(src, dest):
    with ZipFile(src, 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        zipObj.extractall(dest)
        print('File is unzipped in {} folder'.format(dest))


def download_link_generator(id):
    return "https://drive.google.com/uc?id={}".format(id)


def load_download_models(key):
    id = id_model_map[key]
    print("\n===Downloading trained {} models to replicate the result===\n".format(names[id]))
    link = download_link_generator(id)
    dest = "../{}_models".format(key)
    unzip_path = "../{}".format(dest)
    if not os.path.exists(unzip_path):
        print("{} not found. Downloading trained models for {}".format(key))
        gdown.download(link, dest)
        unzip(dest, unzip_path)
        print("Trained models are stored in {}".format(unzip_path))
        return unzip_path
    else:
        print("Models for {} are already downloaded.".format(key))
        return unzip_path


def main():
    args = parse_args()


if __name__ == "__main__":
    main()
