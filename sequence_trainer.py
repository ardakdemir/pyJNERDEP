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


data_path = '../datasets/sa_movie_turkish-test.json'
reader = SentReader(data_path)
lang, word_vocab, model_type, num_cats = "tr", reader.word_vocab,"bert",2
seq_classifier = SequenceClassifier(lang, word_vocab, model_type, num_cats)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():

    seq_classifier.train()
    seq_classifier.to(device)

    seq_classifier.classifier_optimizer.zero_grad()
    seq_classifier.base_optimizer.zero_grad()


    data = reader[0]
    class_logits = seq_classifier(data)

train()





