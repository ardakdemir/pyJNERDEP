"""
    Joint Trainer for dependency parsing and NER

    Gets two datasets and defines two separate models
    Both models share a common feature extraction layer
    The loss is calculated separately for both models and send backward
"""
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
from parser.parsereader import bert2token, pad_trunc_batch
import matplotlib.pyplot as plt
import time
import os
import copy
from pdb import set_trace
import unidecode
from pytorch_transformers import *
from torch import autograd

from parser.parsereader import DepDataset
from parser.utils import conll_writer
from datareader import DataReader
from bertner import BertNER
from evaluate import Evaluate
from conll_eval import evaluate_conll_file
import sys
import logging
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/depparse', help='Root dir for saving models.')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/word2vec', help='Directory of word vectors')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--gold_file', type=str, default=None, help='Output CoNLL-U file.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--char_hidden_dim', type=int, default=400)
    parser.add_argument('--deep_biaff_hidden_dim', type=int, default=400)
    parser.add_argument('--composite_deep_biaff_hidden_dim', type=int, default=100)
    parser.add_argument('--word_emb_dim', type=int, default=75)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--tag_emb_dim', type=int, default=50)
    parser.add_argument('--transformed_dim', type=int, default=125)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=-1)
    parser.add_argument('--word_dropout', type=float, default=0.33)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    parser.add_argument('--no_pretrain', dest='pretrain', action='store_false', help="Turn off pretrained embeddings.")
    parser.add_argument('--no_linearization', dest='linearization', action='store_false', help="Turn off linearization term.")
    parser.add_argument('--no_distance', dest='distance', action='store_false', help="Turn off distance term.")

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--beta2', type=float, default=0.95)

    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--max_steps_before_stop', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='saved_models/depparse', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    args = parser.parse_args()
    return args


class BaseModel(nn.Module):
    
    def __init__(self,args):
        super(BaseModel,self).__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
        self.w_dim = self.bert_model.encoder.layer[11].output.dense.out_features
        self.vocab_size = args.vocab_size
        self.cap_types = 4
        self.lstm_hidden = 20
        self.cap_dim = 10
        self.cap_embeds  = nn.Embedding(self.cap_types,self.cap_dim)
        self.word_embeds = nn.Embedding(self.vocab_size, self.w_dim)
        self.bilstm  = nn.LSTM(self.w_dim+self.cap_dim, self.lstm_hidden, bidirectional=True, num_layers=1, batch_first=True)
    
        
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

    def forward(self,batch_bert_ids, batch_seq_ids, bert2toks, cap_inds):
        bert_out = self.bert_model(batch_bert_ids,batch_seq_ids)
        bert_hiddens = self._get_bert_batch_hidden(bert_out[2],bert2toks)
        cap_embedding = self.cap_embeds(cap_inds)
        lstm_out,_ = self.bilstm(torch.cat((bert_hiddens,cap_embedding),dim=2))
        return lstm_out

class JointTrainer:
    
    def __init__(self):
        self.args = parse_args()
        self.getdatasets()
        ## feature extraction
        self.basemodel = BaseModel(self.args)
    def getdatasets(self):
        
        self.nertrainreader = DataReader(self.args.ner_train_file,"NER",batch_size = self.args.batch_size)
        self.nervalreader = DataReader(self.args.ner_dep_file,"NER", batch_size = self.args.batch_size)
        self.nervalreader.label_voc = self.nertrainreader.label_voc
        
        self.deptraindataset = DepDataset(self.args.dep_train_file,batch_size = self.args.batch_size)
        self.depvaldataset  = DepDataset(self.args.dep_val_file, batch_size = self.args.batch_size, vocabs = self.deptraindataset.vocabs, for_eval=True)
    
    def forward(self, batch, task= "DEP"):
        features = self.basemodel(batch)
        return features   
            
        



