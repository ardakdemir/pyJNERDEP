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
from decoder import *
import sys
import logging
import time

PAD = "[PAD]"
PAD_IND = 0
ROOT = "[ROOT]"
ROOT_IND = 1
UNK = "[UNK]"
UNK_IND = 2
## not sure if root is needed at this stage
VOCAB_PREF = {PAD : PAD_IND, ROOT : ROOT_IND}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Parser(nn.Module):
    def __init__(self, args, tag_size, vocabs):
        super(Parser,self).__init__()
        
        self.lstm_hidden = 10
        self.biaffine_hidden = 30
        self.vocabs = vocabs
        self.num_cat = tag_size
        
        self.args = args 
        
        self.pos_embed = nn.Embedding(len(vocabs['pos_vocab']), self.args['pos_dim'], padding_idx=0)
        

        self.bert_model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
        self.w_dim = self.bert_model.encoder.layer[11].output.dense.out_features
        self.bilstm  = nn.LSTM(self.w_dim,self.lstm_hidden, bidirectional=True, num_layers=1, batch_first=True)
        
        
        self.unlabeled = DeepBiaffineScorer(2*self.lstm_hidden,2*self.lstm_hidden,self.biaffine_hidden,1,pairwise=True)
        self.dep_rel = DeepBiaffineScorer(2*self.lstm_hidden,2*self.lstm_hidden, self.biaffine_hidden, self.num_cat,pairwise=True)
        
        self.dep_rel_crit = nn.CrossEntropyLoss(ignore_index=0, reduction= 'sum')## ignore paddings
        self.dep_ind_crit = nn.CrossEntropyLoss(ignore_index=-1, reduction = 'sum')## ignore paddings at -1 including root
    def decode(self,edge_preds, label_preds, sent_lens,verbose=False):
        trees = []
        dep_rels = []
        dep_tokens = []
        dep_tokens2 = []
        
        s = 0
        
        for l, rel, edge in zip(sent_lens, label_preds, edge_preds):
            logging.info("Rel shape {}".format(rel.shape))
            head_seq = list(chuliu_edmonds_one_root(edge[:l,:l]))
            logging.info("Head seg uzunlugu nedir {} sent_len :  {}".format(len(head_seq),l))

            dep_rel = [rel[i+1][h] for i, h in enumerate(head_seq[1:l])]
            dep_tokens2.append(self.vocabs['dep_vocab'].unmap(dep_rel))
            #print(head_seq.shape)
            trees.append(head_seq+[0 for i in range(sent_lens[0]-l)])
            dep_rels.append(dep_rel)
        #trees = torch.tensor(trees,dtype=torch.long).to(device)       
        #deprel_scores = torch.gather(label_preds,2,trees.unsqueeze(2).unsqueeze(3).\
        #    expand(-1,-1,1,self.num_cat)).squeeze(2).transpose(1,2)
            #.view(len(sent_lens), self.num_cat,sent_lens[0])
        #deprel_preds = torch.argmax(deprel_scores,dim=1)
        #trees = trees.detach().cpu().numpy()
        #for d,l in zip(deprel_preds, sent_lens):
        #    x = d[1:l].detach().cpu().numpy()
        #    #print(self.vocabs['dep_vocab'].unmap(x))
        #    dep_tokens.append(self.vocabs['dep_vocab'].unmap(x))
        #    dep_rels.append(x)
        
        outputs = []
        for l, t, d in zip(sent_lens,trees, dep_tokens2):
            outputs.append([[str(t_), d_] for t_,d_ in zip(t[1:l],d)])
        return trees, dep_rels, outputs
    
    def predict(self, ids, masks, heads, dep_rels, seq_ids, sent_lens, bert2toks):
        #logging.info("Neler oluyor yahu")
        #logging.info(heads[1])
        #logging.info(dep_rels[1])
        batch_size = masks.size()[0]
        word_size = masks.size()[1]
        
        bert_output = self.bert_model(ids,seq_ids)
        bert_out = self._get_bert_batch_hidden(bert_output[2],bert2toks)
        packed_sequence = pack_padded_sequence(bert_out,sent_lens, batch_first=True)
        lstm_out, hidden = self.bilstm(packed_sequence)
        unpacked, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        unlabeled_scores = self.unlabeled(unpacked,unpacked).squeeze(3)
        deprel_scores  = self.dep_rel(unpacked,unpacked) 
        
        mask = torch.zeros(deprel_scores.size(),dtype=torch.long).to(device)
        mask[:,:,:,:2] = 1 
        mask = mask.bool()
        deprel_save = deprel_scores.clone()
        deprel_save = deprel_save.masked_fill(mask,-float('inf'))
        preds = []
        preds.append(F.log_softmax(unlabeled_scores,2).detach().cpu().numpy())
        preds.append(deprel_save)
        #head_preds = torch.argmax(unlabeled_scores,dim=2) 
        #deprel_scores = torch.gather(deprel_scores,2,heads.unsqueeze(2).unsqueeze(3).\
        #    expand(-1,-1,1,self.num_cat)).squeeze(2).transpose(1,2)
            #.view(batch_size, self.num_cat,word_size)
        #deprel_preds = torch.argmax(deprel_scores,dim=1)
        #return preds, deprel_preds,deprel_scores, heads, deprel_save
        return  preds
    
    def forward(self, ids, masks, heads, dep_rels, seq_ids, sent_lens, bert2toks):
        batch_size = masks.size()[0]
        word_size = masks.size()[1]
        
        bert_output = self.bert_model(ids,seq_ids)
        bert_out = self._get_bert_batch_hidden(bert_output[2],bert2toks)
        packed_sequence = pack_padded_sequence(bert_out,sent_lens, batch_first=True)
        
        lstm_out, hidden = self.bilstm(packed_sequence)
        unpacked, _ = pad_packed_sequence(lstm_out,batch_first=True)
        
        unlabeled_scores = self.unlabeled(unpacked,unpacked).squeeze(3)
        deprel_scores  = self.dep_rel(unpacked,unpacked) 
        
        with torch.no_grad():
            #set [PAD] [ROOT] predictions to -infinity
            mask = torch.zeros(deprel_scores.size(),dtype=torch.long).to(device)
            mask[:,:,:,:2] = 1 
            mask = mask.bool()
            deprel_save = deprel_scores.clone()
            deprel_save = deprel_save.masked_fill(mask,-float('inf'))
            preds = []
            preds.append(F.log_softmax(unlabeled_scores,2).detach().cpu().numpy())
            preds.append(torch.argmax(deprel_save,dim = 3).detach().cpu().numpy())
            #preds.append(deprel_save)
        
        diag = torch.eye(heads.size()[1]).to(device)
        diag = diag.bool()
        ## set self-edge scores to negative infinity
        unlabeled_scores = unlabeled_scores.masked_fill(diag,-float('inf'))
        head_scores = torch.gather(deprel_scores,2,heads.unsqueeze(2).\
            unsqueeze(3).expand(-1,-1,1,self.num_cat)).squeeze(2).transpose(1,2)
            #.view(batch_size, self.num_cat,word_size)
        heads_ = heads.masked_fill(masks,-1)
        
        deprel_loss = self.dep_rel_crit(head_scores,dep_rels)
        depind_loss = self.dep_ind_crit(unlabeled_scores[:,1:].transpose(1,2),heads_[:,1:])
        loss = deprel_loss + depind_loss
        
        return preds, loss, deprel_loss, depind_loss
    

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
    def __len__(self):
        return len(self.w2ind)
    def map(self,units):
        return [self.w2ind.get(x,UNK_IND) for x in units]

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
