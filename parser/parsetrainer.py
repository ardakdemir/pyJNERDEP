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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from pdb import set_trace
import unidecode
from pytorch_transformers import *
from parsereader import *
from parser import *
import sys
import logging
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler('reader.log','w','utf-8')], format='%(levelname)s - %(message)s')
def main():
    print("Working on : {}".format(device))
    file_name = "../../../datasets/tr_imst-ud-train.conllu"
    depdataset = DepDataset(file_name,batch_size = 300)
    voc = depdataset.dep_vocab.w2ind
    sent = depdataset.dataset[0][1]
    parser = Parser(len(voc))
    optimizer = optim.SGD([{"params":parser.bilstm.parameters()},\
        {"params": parser.dep_rel.parameters()},{"params":parser.unlabeled.parameters()}], lr=0.1,weight_decay=0.001) 
    param_optimizer = list(parser.bert_model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
     ]
    bert_optimizer =AdamW(optimizer_grouped_parameters,
                     lr=2e-5)
    EPOCH = 30
    parser.train().to(device)
    L = len(depdataset)
    for i in range(EPOCH):
        train_loss = 0
        for j in range(L):
            optimizer.zero_grad()
            bert_optimizer.zero_grad()
            tokens, sent_lens, masks, tok_inds, pos, dep_inds, dep_rels, bert_batch_after_padding,\
                bert_batch_ids, bert_seq_ids, bert2toks = depdataset[j]
            sent_lens = sent_lens.to(device)
            masks = masks.to(device)
            tok_inds = tok_inds.to(device)
            dep_inds = dep_inds.to(device)
            dep_rels = dep_rels.to(device)
            bert2toks = bert2toks.to(device)
            bert_batch_ids = bert_batch_ids.to(device)
            bert_seq_ids = bert_seq_ids.to(device)
            loss = parser(bert_batch_ids, masks, dep_inds, dep_rels, bert_seq_ids,sent_lens, bert2toks)
            loss = loss/ torch.sum(sent_lens)
            loss.backward()
            train_loss += loss
            #print(loss.item())
            #print(parser.dep_rel.W1.weight.grad)
            #print(parser.unlabeled.scorer.W_bilin.weight.grad)
            if j%10 == 1:
                print("Average Train Loss  {}".format(train_loss/j))
            optimizer.step()
            bert_optimizer.step()
        print("Final train loss of epoch {} :  {}".format(i+1,train_loss/L))
if __name__ == "__main__":

    main()


