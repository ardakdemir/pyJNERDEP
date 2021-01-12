from datareader import DataReader
from transformers import AutoTokenizer, AutoModel, BertForPreTraining, BertForTokenClassification
import os

encoding_map = {"cs": "latin-1",
                "tr": "utf-8",
                "hu": "utf-8",
                "jp": "utf-8",
                "fi": "utf-8"}

model_name_dict = {"jp": "cl-tohoku/bert-base-japanese",
                   "tr": "dbmdz/bert-base-turkish-cased",
                   "hu": "/home/aakdemir/bert_models/hubert",
                   "fi": "TurkuNLP/bert-base-finnish-cased-v1",
                   "cs": "DeepPavlov/bert-base-bg-cs-pl-ru-cased"}

lang_abs = {"fi": "finnish", "hu": "hungarian", "cs": "czech", "tr": "turkish", "jp": "japanese"}

batch_size = 10
data_folder = "/home/aakdemir/datasets"

# config
lang = "jp"
encoding = encoding_map[lang]
model_name = model_name_dict[lang]
task = "NER"
language = lang_abs[lang]
file_name = os.path.join(data_folder, "myner_{}-train.txt".format(language))

bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
data_reader = DataReader(file_name, task, batch_size=batch_size,
                         tokenizer=bert_tokenizer)
l = len(data_reader)
print("Read {} batches ".format(l))
