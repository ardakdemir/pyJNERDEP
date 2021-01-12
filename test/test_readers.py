from datareader import DataReader
from transformers import AutoTokenizer, AutoModel, BertForPreTraining, BertForTokenClassification
import os
from parser.parsereader import DepDataset
import logging

log_path = "test/test_log_file.txt"
logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler(log_path, 'w', 'utf-8')],
                    format='%(levelname)s - %(message)s')

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

batch_size = 300
data_folder = "/home/aakdemir/datasets"

# config
for lang in model_name_dict.keys():
    print("\nTrying to read {} datasets".format(lang))
    for task in ["NER", "DEP"]:
        print("Reading {} dataset".format(task))
        lang = "jp"
        encoding = encoding_map[lang]
        model_name = model_name_dict[lang]
        language = lang_abs[lang]

        if task == "NER":
            file_name = os.path.join(data_folder, "myner_{}-train.txt".format(language))
        else:
            file_name = os.path.join(data_folder, "dep_{}_train.conllu".format(language))

        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_tokenizer.add_tokens(['[SOS]', '[EOS]', '[ROOT]', '[PAD]'])
        if task == "NER":
            data_reader = DataReader(file_name, task, batch_size=batch_size,
                                     tokenizer=bert_tokenizer)
        elif task == "DEP":
            data_reader = DepDataset(file_name, batch_size=12,
                                     tokenizer=bert_tokenizer)
        batch = data_reader[0]
        logging.info(batch)
        l = len(data_reader)

        print("Read {} batches ".format(l))
