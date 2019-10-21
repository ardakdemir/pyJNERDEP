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
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
import time
import os
import copy
from pdb import set_trace
import unidecode
from pytorch_transformers import *
from parsereader import *
from biaffine import *
import sys
import logging
import time

PAD = "[PAD]"
PAD_IND = 0
ROOT = "[ROOT]"
ROOT_IND = 1

## not sure if root is needed at this stage
VOCAB_PREF = {PAD : PAD_IND, ROOT : ROOT_IND}


class Parser(nn.Module):
    def __init__(self,tag_size):
        super(Parser,self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
        self.w_dim = self.bert_model.encoder.layer[11].output.dense.out_features
        self.lstm_hidden = 100
        self.biaffine_hidden = 30
        self.num_cat = tag_size
        self.bilstm  = nn.LSTM(self.w_dim,self.lstm_hidden, bidirectional=True, num_layers=1, batch_first=True)
        self.unlabeled = DeepBiaffineScorer(2*self.lstm_hidden,2*self.lstm_hidden,self.biaffine_hidden,1,pairwise=True)
        self.dep_rel = DeepBiaffineScorer(2*self.lstm_hidden,2*self.lstm_hidden, self.biaffine_hidden, self.num_cat,pairwise=True)
        self.crit = nn.CrossEntropyLoss(ignore_index=-1, reduction= 'sum')## ignore paddings
    def forward(self, ids,heads, dep_rels, seq_ids, sent_lens, bert2toks):
        bert_output = self.bert_model(ids,seq_ids)
        bert_out = self._get_bert_batch_hidden(bert_output[2],bert2toks)
        packed_sequence = pack_padded_sequence(bert_out,sent_lens, batch_first=True)
        lstm_out, hidden = self.bilstm(packed_sequence)
        unpacked, _ = pad_packed_sequence(lstm_out,batch_first=True)
        unlabeled_scores = self.unlabeled(unpacked,unpacked)
        deprel_scores  = self.dep_rel(unpacked,unpacked)

        return unlabeled_scores, deprel_scores
    def _get_bert_batch_hidden(self, hiddens , bert2toks, layers=[-2,-3,-4]):
        meanss = torch.mean(torch.stack([hiddens[i] for i in layers]),0)
        batch_my_hiddens = []
        for means,bert2tok in zip(meanss,bert2toks):
            my_token_hids = []
            my_hiddens = []
            for i,b2t in enumerate(bert2tok):
                if i>0 and b2t!=bert2tok[i-1]:
                    my_hiddens.append(torch.mean(torch.cat(my_token_hids),0).view(1,-1))
                    my_token_hids = [means[i+1].view(1,-1)] ## we skip the CLS token
                else:
                    my_token_hids.append(means[i+1].view(1,-1))
            my_hiddens.append(torch.mean(torch.cat(my_token_hids),0).view(1,-1))
            batch_my_hiddens.append(torch.cat(my_hiddens))
        return torch.stack(batch_my_hiddens)
class Vocab:

    def __init__(self,w2ind):
        self.w2ind =  w2ind
        self.ind2w = [x for x in w2ind.keys()]

    def map(self,units):
        return [self.w2ind[x] for x in units]

    def unmap(self,idx):
        return [self.ind2w[i] for i in idx]

if __name__=="__main__":

    depdataset = DepDataset("../../datasets/tr_imst-ud-train.conllu", batch_size = 300)
    tokens, sent_lens, tok_inds, pos, dep_inds, dep_rels, bert_batch_after_padding,\
        bert_batch_ids, bert_seq_ids, bert2toks = depdataset[0]
    voc = depdataset.dep_vocab.w2ind
    print(len(voc))
    #scores = parser.crf(lstm_out)
    #viterbi_loss = CRFLoss(voc)
    #feats = parser.fc(lstm_out)
    #print(loss)
    #print(dep_rels)
