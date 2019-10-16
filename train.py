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


def ner_train(data_path, val_path, save_path):
    evaluator = Evaluate("NER")
    logging.basicConfig(level=logging.DEBUG, filename='trainer.log', filemode='w', format='%(levelname)s - %(message)s')
    datareader = DataReader(data_path, "NER")
    valreader = DataReader(val_path,"NER")
    logging.info("Data is read from %s"%data_path)
    vocab_size = datareader.vocab_size
    l2ind = datareader.l2ind
    num_cat = len(l2ind)
    model = BertNER(lstm_hidden = 100, vocab_size=vocab_size, l2ind = l2ind, num_cat = num_cat)

    if os.path.isfile(save_path):
        logging.info("Model loaded %s"%save_path)
        model.load_state_dict(torch.load(save_path))

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    EPOCH =5
    B_S =  1
    best_loss = -1
    best_model = 0
    L = len(datareader.dataset)
    for i in range(EPOCH):
        l = 0
        c,t,p_tot = 0,1,1
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
            #logging.info("Loss {}".format(loss.item()))
            optimizer.step()
            train_loss+= loss.item()
            if l%100 == 9:
                logging.info("AVERAGE TRAIN LOSS : {} after {} examples ".format( train_loss/l,l))
                model.eval()
                for x in range(10):
                    valdata = valreader.get_bert_input(for_eval=True)
                    my_tokens, bert_tokens, ids, enc_ids, seq_ids, bert2tok, labels = valdata[0]
                    if len(labels)==1:
                        continue
                    decoded_path, score = model(my_tokens, bert_tokens, ids,seq_ids, bert2tok)
                    c_,p_,tot = evaluator.f_1(decoded_path, labels.numpy())
                    c+=c_
                    p_tot+=p_
                    t+=tot
                logging.info("Precision : {}  Recall {} Total labels: {} Total predictions : {}".format((c+1)/p_tot,(c+1)/t ,t,p_tot))
                model.train()
        if i==0 or train_loss < best_loss:
            torch.save(model.state_dict(), save_path)
            best_loss = train_loss

if __name__ == "__main__":
    save_path = "best_model.pth"
    data_path = '../datasets/turkish-ner-train.tsv'
    val_path = '../datasets/turkish-ner-dev.tsv'
    ner_train(data_path, val_path, save_path)
