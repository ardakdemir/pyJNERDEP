"""
    Check OOVs for word2vec
"""

from transformers import AutoTokenizer
import os

from test.test_models import model_name_dict, load_word2vec, lang_abs
from parser.parsereader import DepDataset
from datareader import DataReader

data_folder = "../../datasets"

l = "jp"
lang = lang_abs[l]
model_type = "bert"


def init_tokenizer(lang, model_type):
    if model_type in ["mbert", "bert_en"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name_dict[model_type])
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_dict[lang])
    return tokenizer


bert_tokenizer = init_tokenizer(l, model_type)

dep_train_name = os.path.join(data_folder, "dep_{}_train.conllu".format(lang))
dep_dev_name = os.path.join(data_folder, "dep_{}_dev.conllu".format(lang))
dep_test_name = os.path.join(data_folder, "dep_{}_test.conllu".format(lang))
ner_train_name = os.path.join(data_folder, "myner_{}-train.txt".format(lang))
ner_dev_name = os.path.join(data_folder, "myner_{}-dev.txt".format(lang))
ner_test_name = os.path.join(data_folder, "myner_{}-test.txt".format(lang))

nertrainreader = DataReader(ner_train_name, "NER", batch_size=200,
                            tokenizer=bert_tokenizer)
deptraindataset = DepDataset(dep_train_name, batch_size=200,
                             tokenizer=bert_tokenizer)

w2v_model = load_word2vec(l)

dep_vocab = deptraindataset.vocabs["tok_vocab"].w2ind
ner_vocab = nertrainreader.word_voc.w2ind

