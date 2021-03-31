import subprocess
import sys
import gdown
import tqdm
from zipfile import ZipFile
import argparse
import os

id_model_map = {"SA": {"twitter_turkish": "10mOwZGp4-NTo9K_bJkE2KlWa4HUudW07",
                       "movie_turkish": "1IoQhYijlWVlK0vHnVUn0e1ohwahePO7R",
                       "movie_english": "1t2XgkbfxGPjvThEg-wkO_ejTOlikIzv7"},
                "NER": {"bert": "1M9-JWPL535IIDUSoDNDpMRzTex8tBbN7",
                        "mbert": "",
                        "bert_en": "",
                        "fastext": "1JWHSHDmTxsZoYwkc6_K8Wz76JSUmJVhA",
                        "random_init": "",
                        "word2vec": "1E5jGGlhbevjSg-oprf_e0vU2y89_zhHJ"
                        },
                "DEP": {"bert": "",
                        "mbert": "",
                        "bert_en": "",
                        "fastext": "",
                        "random_init": "",
                        "word2vec": ""
                        },
                "FLAT_NER": {"bert": "",
                             "mbert": "",
                             "bert_en": "",
                             "fastext": "",
                             "random_init": "",
                             "word2vec": ""
                             },
                "FLAT_DEP": {"bert": "",
                             "mbert": "",
                             "bert_en": "",
                             "fastext": "",
                             "random_init": "",
                             "word2vec": ""
                             }
                }

names = {"SA": "Sentiment Analysis",
         "NER": "Named Entity Recognition",
         "DEP": "Dependency Parsing",
         "FLAT_DEP": "Multi-task Learning DEP",
         "FLAT_NER": "Multi-task Learning NER"}

word2vec_driveIds = {"jp": "1dYISBXsgK3yR6mw-LRGfjGrcN3aVme2q",
                     "tr": "14WH-amhKXn4ayqi2lugUSIoS7b8q0U9H",
                     "hu": "1dmEC0-7Zkmc4p9OmTIw3JKMJ7aNyGYIt",
                     "en": "1avdWgjq138lrfJnIZVRpa9EaLJrU4HVj",
                     "fi": "1wtqAc4FZ6wl4w4_kSozWbDjgUUV22Fjr",
                     "cs": "1ibFwJ6B01Kpm6k6qdI1cJ64wLyaJy-s3"}


def drive_download_w2v(lang, save_path):
    print("\nDownloading word2Vec model for {} to {}".format(lang,save_path))
    id = word2vec_driveIds[lang]
    link = download_link_generator(id)
    gdown.download(link, save_path, quiet=False)
    return save_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='NER',
                        help='key for downloading the models')
    parser.add_argument('--word_type', type=str, default='fastext',
                        help='key for downloading the models')
    parser.add_argument('--save_folder', type=str, default=None,
                        help='Destination to save downloaded models')
    parser.add_argument('--id', type=str, default=None,
                        help='Id of the model to be downloaded...')
    args = parser.parse_args()
    return args


def unzip(src, dest):
    with ZipFile(src, 'r') as zipObj:
        # Extract all the contents of zip file in different directory
        zipObj.extractall(dest)
        print('File is unzipped in {} folder'.format(dest))


def download_link_generator(id):
    return "https://drive.google.com/uc?id={}".format(id)


def load_download_models(model_type, word_type, save_folder=None):
    id = id_model_map[model_type][word_type]
    print("\n===Downloading trained {} models to replicate the result===\n".format(names[model_type]))
    link = download_link_generator(id)
    dest = "../{}_{}_models.zip".format(model_type, word_type)
    unzip_path = "../{}".format(os.path.split(dest)[-1].split(".")[0]) if not save_folder else save_folder
    if not os.path.exists(unzip_path):
        print("{} not found. Downloading trained models for {} {}".format(unzip_path, model_type, word_type))
        gdown.download(link, dest, quiet=False)
        unzip(dest, unzip_path)
        print("Trained models are stored in {}".format(unzip_path))
        return unzip_path
    else:
        print("Models for {} {} are already downloaded.".format(model_type, word_type))
        return unzip_path

def download_by_id():
    args = parse_args()
    id = args.id
    link = download_link_generator(id)
    gdown.download(link, "{}_downloaded".format(id), quiet=False)

def main():
    args = parse_args()
    model_type = args.model_type
    word_type = args.word_type
    save_folder = args.save_folder

    # download_by_id()
    load_download_models(model_type, word_type, save_folder)
    content = os.listdir(save_folder)
    print("Content of the save model folder: {}".format(content))


if __name__ == "__main__":
    main()
