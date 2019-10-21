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


logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler('reader.log','w','utf-8')], format='%(levelname)s - %(message)s')
def main():
    file_name = "../../../datasets/tr_imst-ud-train.conllu"
    depdataset = DepDataset(file_name,batch_size = 300)
    voc = depdataset.dep_vocab.w2ind
    sent = depdataset.dataset[0][1]
    parser = Parser(len(voc))
    optimizer = optim.SGD(parser.parameters(), lr=0.001) 
    tokens, sent_lens, masks, tok_inds, pos, dep_inds, dep_rels, bert_batch_after_padding,\
        bert_batch_ids, bert_seq_ids, bert2toks = depdataset[0]
    loss = parser(bert_batch_ids, masks, dep_inds, dep_rels, bert_seq_ids,sent_lens, bert2toks)
    loss.backward()

    print(loss.item()) 
    print("Size of the words ")
    optimizer.step()
if __name__ == "__main__":

    main()


