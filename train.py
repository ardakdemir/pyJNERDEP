from __future__ import print_function
from __future__ import division
import torch
from skimage import io, transform
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from pdb import set_trace
import unidecode
from pytorch_transformers import *

from datareader import DataReader
from bertner import BertNER
from evaluate import Evaluate

import logging


def ner_train(data_path):
    evaluator = Evaluate("NER")
    logging.basicConfig(level=logging.DEBUG, filename='trainer.log', filemode='w', format='%(levelname)s - %(message)s')
    datareader = DataReader(data_path, "NER")
    logging.info("Data is read from %s"%data_path)
    vocab_size = datareader.vocab_size
    l2ind = datareader.l2ind
    num_cat = len(l2ind)
    model = BertNER(lstm_hidden = 10, vocab_size=vocab_size, l2ind = l2ind, num_cat = num_cat)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    EPOCH = 1
    B_S =  1
    L = len(datareader.dataset)
    for i in range(EPOCH):
        l = 0
        c,t,p_tot = 0,0,0
        train_loss = 0
        for l in tqdm(range(L)):
            data = datareader.get_bert_input()
            my_tokens, bert_tokens, ids, enc_ids, seq_ids, bert2tok, labels = data[0]
            #print(my_tokens)
            #print(labels)
            if len(labels)==1:
                continue
            optimizer.zero_grad()
            loss = model._bert_crf_neg_loss(my_tokens, bert_tokens, ids, seq_ids,labels, bert2tok)
            loss.backward()
            logging.info("Loss {}".format(loss.item()))
            optimizer.step()
            l += B_S
            train_loss+= loss.item()
            if l%10 == 0:
                data = datareader.get_bert_input()
                my_tokens, bert_tokens, ids, enc_ids, seq_ids, bert2tok, labels = data[0]
                if len(labels)==1:
                    continue
                decoded_path, score = model(my_tokens, bert_tokens, ids,seq_ids, bert2tok)
                evaluator.f_1(decoded_path, labels.numpy())
if __name__ == "__main__":
    data_path = '../datasets/turkish-ner-train.tsv'
    ner_train(data_path)
