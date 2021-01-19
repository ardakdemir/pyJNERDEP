"""
    Check OOVs for word2vec
"""

from transformers import AutoTokenizer, AutoModel, BertForPreTraining, BertForTokenClassification
import io
import argparse
import torch.nn as nn
import numpy as np
import torch
from gensim.models import Word2Vec
import os
import fasttext as ft
import fasttext.util

from sequence_classifier import SequenceClassifier
from sareader import SentReader
from test.test_models import model_name_dict, word2vec_dict, load_word2vec
from parser.parsereader import DepDataset
from parser.utils import conll_writer, sort_dataset, unsort_dataset, score, convert2IOB2new
from datareader import DataReader

data_folder = "~/datasets"

lang = "jp"
model_type = "bert"


def init_tokenizer(lang, model_type):
    if model_type in ["mbert", "bert_en"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name_dict[model_type])
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_dict[lang])
    return tokenizer


bert_tokenizer = init_tokenizer(lang, model_type)

dep_train_name = os.path.join(data_folder, "dep_{}_train.conllu".format(lang))
dep_dev_name = os.path.join(data_folder, "dep_{}_dev.conllu".format(lang))
dep_test_name = os.path.join(data_folder, "dep_{}_test.conllu".format(lang))
ner_train_name = os.path.join(data_folder, "myner_{}-train.txt".format(lang))
ner_dev_name = os.path.join(data_folder, "myner_{}-dev.txt".format(lang))
ner_test_name = os.path.join(data_folder, "myner_{}-test.txt".format(lang))

self.nertrainreader = DataReader(ner_train_name, "NER", batch_size=200,
                                 tokenizer=bert_tokenizer)
self.deptraindataset = DepDataset(dep_train_name, batch_size=200,
                                  tokenizer=bert_tokenizer)

w2v_model = load_word2vec(lang)


