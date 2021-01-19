import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import time
import os
import copy
from transformers import AutoTokenizer, AutoModel, BertForPreTraining, BertForTokenClassification
from sequence_classifier import SequenceClassifier
from sareader import SentReader

model_name_dict = {"jp": "cl-tohoku/bert-base-japanese",
                   "tr": "dbmdz/bert-base-turkish-cased",
                   "hu": "/home/aakdemir/bert_models/hubert",
                   "fi": "TurkuNLP/bert-base-finnish-cased-v1",
                   "cs": "DeepPavlov/bert-base-bg-cs-pl-ru-cased",
                   "en": "bert-base-cased",
                   "mbert": "bert-base-multilingual-cased",
                   "bert_en": "bert-base-cased"}

data_path = '../datasets/sa_movie_turkish-test.json'
lang, model_type, num_cats = "tr", "bert", 2

tokenizer = AutoTokenizer.from_pretrained(model_name_dict[lang])
reader = SentReader(data_path, tokenizer=tokenizer)

word_vocab = reader.word_vocab
seq_classifier = SequenceClassifier(lang, word_vocab, model_type, num_cats)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Vocab size: {}".format(len(reader.word_vocab.w2ind)))


def train():
    seq_classifier.train()
    seq_classifier.to(device)

    seq_classifier.classifier_optimizer.zero_grad()
    seq_classifier.base_optimizer.zero_grad()

    data = reader[0]
    class_logits = seq_classifier(data)


train()
