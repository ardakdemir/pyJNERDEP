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

from hlstm import HighwayLSTM
from parser.parsereader import *
from parser.biaffine import *
from parser.decoder import *

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


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
This class implements the dependency parser that will be used in a multi-task setting
Assumes that the model will receive pos-tag embeddings (if available) and bert embeddings
"""
class JointParser(nn.Module):
    def __init__(self, args ):
        super(JointParser,self).__init__()
        
        self.args = args 
        self.biaffine_hidden = self.args['biaffine_hidden']
        #self.vocabs = vocabs
        self.num_cat = self.args['dep_cats']
        self.device = self.args['device']
        self.pos_dim = self.args['pos_dim']
        
        self.pos_embed = nn.Embedding(self.args['pos_vocab_size'], self.args['pos_dim'], padding_idx=0)
        self.dep_embed = nn.Embedding(self.num_cat, self.args['dep_dim'], padding_idx=0)
       
        self.dep_lr = self.args['dep_lr']
        self.weight_decay = self.args['weight_decay']
        self.pos_drop = self.args['word_drop']
        self.lstm_drop = self.args['lstm_drop']
        self.parser_drop = self.args['parser_drop']
        self.lstm_layers = self.args['lstm_layers']
        self.lstm_hidden = self.args['lstm_hidden']
        if self.args['model_type'] == 'NERDEP': 
            ## output of the ner layer
            if self.args['inner']:
                self.lstm_input_dim = self.args['lstm_input_size'] + 2 * self.lstm_hidden
            else:
                self.lstm_input_dim = self.args['ner_dim'] + self.args['lstm_input_size']
        else:
            self.lstm_input_dim = self.args['lstm_input_size']
        
        
        self.parserlstm  = nn.LSTM(self.lstm_input_dim,self.lstm_hidden, bidirectional=True, num_layers=self.lstm_layers, batch_first=True)
        self.highwaylstm = HighwayLSTM(self.lstm_input_dim,self.lstm_hidden, bidirectional=True,num_layers=self.lstm_layers,batch_first=True,dropout=self.lstm_drop,pad=True )
        
        
        self.pos_dropout = nn.Dropout(self.pos_drop)
        self.lstm_dropout = nn.Dropout(self.lstm_drop)
        self.parser_dropout = nn.Dropout(self.parser_drop)
        
        self.unlabeled = DeepBiaffineScorer(2*self.lstm_hidden,2*self.lstm_hidden,self.biaffine_hidden,1,pairwise=True,dropout=self.parser_drop)
        self.dep_rel = DeepBiaffineScorer(2*self.lstm_hidden,2*self.lstm_hidden, self.biaffine_hidden, self.num_cat,pairwise=True,dropout=self.parser_drop)
        
        self.dep_rel_crit = nn.CrossEntropyLoss(ignore_index=0, reduction= 'sum')## ignore paddings
        self.dep_ind_crit = nn.CrossEntropyLoss(ignore_index=-1, reduction = 'sum')## ignore paddings at -1 including root
    
        
        self.optimizer = optim.AdamW([{"params":self.parserlstm.parameters()},\
            {"params": self.dep_rel.parameters()},\
            {"params": self.highwaylstm.parameters()},\
            {"params":self.unlabeled.parameters()},\
            {"params": self.pos_embed.parameters()},\
            {"params": self.dep_embed.parameters()}],\
             lr=self.dep_lr, weight_decay=0.03, betas=(0.9,self.args['beta2']), eps=1e-6)    
    
    
    def decode(self,edge_preds, label_preds, sent_lens, verbose=False):
        trees = []
        dep_rels = []
        dep_tokens = []
        dep_tokens2 = []
        
        s = 0
        
        for l, rel, edge in zip(sent_lens, label_preds, edge_preds):
            head_seq = list(chuliu_edmonds_one_root(edge[:l,:l]))
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
    
    
    def forward(self, masks, bert_out,  heads, dep_rels, pos_ids, sent_lens, training=True, task="DEP"):
        """
            heads and dep_rels are used during training
            must be initialized to empty if ner-epoch or during prediction mode
        """
        batch_size = masks.size()[0]
        word_size = masks.size()[1]
        
        ## pos tag information added 
        ## may consider multi-learning this one as well!!
        ## if dependency epoch we have pos tags so use this information as well
        ## if ner epoch we must get pos_ids from somewhere like a separate model
        
        #pos_embeds = self.pos_dropout(self.pos_embed(pos_ids))
        #x = torch.cat([bert_out,pos_embeds],dim=2)
        x = bert_out
        #x = self.lstm_dropout(bert_out)
        #packed_sequence = pack_padded_sequence(bert_out,sent_lens, batch_first=True) 
        packed_sequence = pack_padded_sequence(x,sent_lens, batch_first=True)
        
        #lstm_out, hidden = self.parserlstm(packed_sequence)
        highway_out,_ = self.highwaylstm(x,sent_lens)
        #logging.info(highway_out.shape)
        unpacked = self.lstm_dropout(highway_out)
        #unpacked, _ = pad_packed_sequence(lstm_out,batch_first=True)
        #unpacked = self.lstm_dropout(unpacked)
        if task=="DEPNER" and training and self.args['inner']:
            return torch.cat([x,unpacked],dim=2)
        unlabeled_scores = self.unlabeled(unpacked,unpacked).squeeze(3)
        deprel_scores  = self.dep_rel(unpacked,unpacked) 
        
        preds = []
        acc = 0
        if (task=="NERDEP" or task=="DEP") and training:
            diag = torch.eye(heads.size()[1]).to(self.device)
            diag = diag.bool()
            ## set self-edge scores to negative infinity
            unlabeled_scores = unlabeled_scores.masked_fill(diag,-float('inf'))
            head_scores = torch.gather(deprel_scores,2,heads.unsqueeze(2).\
                unsqueeze(3).expand(-1,-1,1,self.num_cat)).squeeze(2).transpose(1,2)
                #.view(batch_size, self.num_cat,word_size)
            with torch.no_grad():
                #set [PAD] [ROOT] [EOS] [UNK] predictions to -infinity
                mask = torch.zeros(deprel_scores.size(),dtype=torch.long).to(self.args['device'])
                mask[:,:,:,:4] = 1 
                mask = mask.bool()
                deprel_save = deprel_scores.clone()
                deprel_save = deprel_save.masked_fill(mask,-float('inf'))
                preds.append(F.log_softmax(unlabeled_scores,2).detach().cpu().numpy())
                preds.append(torch.argmax(deprel_save,dim = 3).detach().cpu().numpy())
                
                arc_scores = torch.argmax(unlabeled_scores,dim=2)
                dep_scores = torch.argmax(deprel_save,dim=3) 
                dep_ind_preds = torch.gather(dep_scores, 2, arc_scores.unsqueeze(2)).squeeze(2)
                
                #logging.info("Input nasil birseydi : ")
                #logging.info(x.shape)
                #logging.info("Dep preds nasil birsey")
                #logging.info(preds[1].shape)
                #logging.info("Head preds nasil birsey")
                #logging.info(preds[0].shape)
                head_eq = sum(sum(arc_scores[:,1:].detach().cpu().numpy()==heads[:,1:].detach().cpu().numpy()))
                rel_eq = sum(sum(dep_ind_preds.detach().cpu().numpy()==dep_rels.detach().cpu().numpy()))
                acc = rel_eq*1.0/sum(sent_lens).item()
                head_acc = head_eq/sum(sent_lens).item()
            heads_ = heads.masked_fill(masks,-1)            
            deprel_loss = self.dep_rel_crit(head_scores,dep_rels)
            depind_loss = self.dep_ind_crit(unlabeled_scores[:,1:].transpose(1,2),heads_[:,1:])
            loss = deprel_loss + depind_loss
            loss = loss/torch.sum(sent_lens)
            deprel_loss = deprel_loss/torch.sum(sent_lens)
            depind_loss = depind_loss/torch.sum(sent_lens)
            return preds, loss, deprel_loss, depind_loss, acc , head_acc
        
        elif task=="DEPNER" and training:   
            
            if self.args['inner']:
                return torch.cat([x,unpacked],dim=2)
            
            mask = torch.zeros(deprel_scores.size(),dtype=torch.long).to(self.device)
            mask[:,:,:,:4] = 1 
            mask = mask.bool()

            deprel_save = deprel_scores.clone()
            deprel_save = deprel_save.masked_fill(mask,-float('inf'))
            
            preds = []
            arc_scores = torch.argmax(unlabeled_scores,dim = 2)
            rel_scores = torch.argmax(deprel_save, dim = 3 )
            
            preds.append(arc_scores.detach().cpu().numpy())
            preds.append(rel_scores.detach().cpu().numpy())
            
            dep_ind_preds = torch.gather(rel_scores,2,arc_scores.unsqueeze(2)).squeeze(2)
            dep_embeddings = self.dep_embed(dep_ind_preds)
            dep_embeddings = self.pos_dropout(dep_embeddings)
            
            return torch.cat([x,dep_embeddings],dim=2)
        
        else:
            with torch.no_grad():
                #set [PAD] [ROOT] predictions to -infinity
                mask = torch.zeros(deprel_scores.size(),dtype=torch.long).to(self.args['device'])
                mask[:,:,:,:4] = 1 
                mask = mask.bool()
                
                deprel_save = deprel_scores.clone()
                deprel_save = deprel_save.masked_fill(mask,-float('inf'))
                preds.append(F.log_softmax(unlabeled_scores,2).detach().cpu().numpy())
                preds.append(torch.argmax(deprel_save,dim = 3).detach().cpu().numpy())
                #preds.append(deprel_save)
                
                arc_scores = torch.argmax(unlabeled_scores,dim=2)
                dep_scores = torch.argmax(deprel_save,dim=3) 
                dep_ind_preds = torch.gather(dep_scores, 2, arc_scores.unsqueeze(2)).squeeze(2)
                head_eq = sum(sum(arc_scores[:,1:].detach().cpu().numpy()==heads[:,1:].detach().cpu().numpy()))
                rel_eq = sum(sum(dep_ind_preds.detach().cpu().numpy()==dep_rels.detach().cpu().numpy()))
                acc = rel_eq*1.0/sum(sent_lens).item()
                head_acc = head_eq/sum(sent_lens).item()
            return preds,0, 0, 0, acc , head_acc
        
    

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
    device = 'cpu'
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    depdataset = DepDataset("../../datasets/tr_imst-ud-train.conllu", batch_size = 300)
    masks = torch.zeros(3,5)
    bert_out = torch.randn(3,5,100)
    lens = torch.tensor([5,5,5])
    args = {'bert_dim' : 100,'pos_dim' : 12, 'pos_vocab_size': 23,'lstm_hidden' : 10,'device':device}
    parser= JointParser(args)
    parser.vocabs = depdataset.vocabs
    pos_ids = torch.zeros(3,5,dtype=torch.long)
    preds, _ = parser.forward(masks, bert_out, None, None, pos_ids, lens, training=False, dep=True)
    print(preds[0].shape)
    print(preds[1].shape)
    trees, dep_rels, output = parser.decode(preds[0],preds[1],lens)
    print(trees)
    print(dep_rels)
    print(output)
