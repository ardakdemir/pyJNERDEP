import subprocess
import sys
import gdown
import tqdm
import zipdir

def download_link_generator(id):
    return "https://drive.google.com/uc?id={}".format(id)


id_model_map = {"sa": "1Yt6MVemyNDlv0bMrjEEgGSjZ7P42bW0t",
                "ner": "",
                "dep": "",
                "flat": ""}
names = {"sa": "Sentiment Analysis",
         "ner": "Named Entity Recognition",
         "dep": "Dependency Parsing",
         "flat": "Multi-task Learning"}

id = id_model_map[key]
print("\n===Downloading trained {} models to replicate the result===\n".format(names[id]))
link = download_link_generator(id)
dest = "../{}_models".format(key)
gdown.download(link, dest)
