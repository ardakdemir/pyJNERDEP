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

from parser.parsereader import *
from parser.parser import *

from parser.utils import score, conll_writer, unsort_dataset, sort_dataset
import sys
import logging
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler('parsing.log','w','utf-8')], format='%(levelname)s - %(message)s')

class DepTrainer:
    def __init__(self,args, dataset, load_file = None):
        self.depdataset = dataset
        self.args = args
        if load_file:
            self.parser = load_model(load_file)
        else:
            self.parser = Parser(self.args, self.depdataset.num_rels, self.depdataset.vocabs)
        
        
        self.parser.to(device) 
        
        param_optimizer = list(self.parser.bert_model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
         ]
        
        self.bert_optimizer =AdamW(optimizer_grouped_parameters,
                         lr=2e-5)
        self.optimizer = optim.SGD([{"params":self.parser.bilstm.parameters()},\
            {"params": self.parser.dep_rel.parameters()},\
            {"params":self.parser.unlabeled.parameters()}, {"params": self.parser.pos_embed.parameters()}], lr=0.1,weight_decay=0.001)    
    
    
    def forward(self,batch,train=True):
        
        tokens,bert_batch_after_padding, sent_lens, masks, tok_inds, pos, dep_inds, dep_rels, \
            bert_batch_ids, bert_seq_ids, bert2toks, cap_inds = batch
        sent_lens = sent_lens.to(device)
        masks = masks.to(device)
        tok_inds = tok_inds.to(device)
        dep_inds = dep_inds.to(device)
        dep_rels = dep_rels.to(device)
        bert2toks = bert2toks.to(device)
        bert_batch_ids = bert_batch_ids.to(device)
        bert_seq_ids = bert_seq_ids.to(device)
        pos = pos.to(device)
        
        if train: 
            _, loss, deprel, depind = self.parser(bert_batch_ids, masks, dep_inds, dep_rels, pos, bert_seq_ids,sent_lens, bert2toks) 
            return loss/torch.sum(sent_lens), deprel/torch.sum(sent_lens), depind/torch.sum(sent_lens)
        
        else:
            self.parser.eval()
            #preds, deprel_preds, true_scores, headtrues, deprel_scores = self.parser.predict(bert_batch_ids, masks, dep_inds, dep_rels, bert_seq_ids, sent_lens, bert2toks)
            #preds  = self.parser.predict(bert_batch_ids, masks, dep_inds, dep_rels, bert_seq_ids, sent_lens, bert2toks)
            preds ,_,_,_= self.parser(bert_batch_ids, masks, dep_inds, dep_rels, pos, bert_seq_ids, sent_lens, bert2toks)
            #return preds, deprel_preds, true_scores, headtrues, deprel_scores
            return preds


    def update(self,batch,eval=False): 
        
        self.optimizer.zero_grad()
        self.bert_optimizer.zero_grad()
        
        loss, deprel, depind =  self.forward(batch)
        loss_val = loss.item()
        loss = loss.backward()
        
        self.optimizer.step()
        self.bert_optimizer.step()
        
        return loss_val, deprel.item(), depind.item()


    def evaluate(self, dataset): 
        logging.info("Evaluating performance on {}".format(dataset.file_name))
        self.parser.eval()
        
        orig_idx = dataset.orig_idx
        data = []
        
        field_names = ["word", "head", "deprel"]
        gold_file = dataset.file_name
        pred_file = "pred_"+gold_file.split("/")[-1]    
        start_id = orig_idx[0]
        for x in tqdm(range(len(dataset)),desc = "Evaluation"):
            batch = dataset[x]
            sent_lens = batch[2]
            tokens = batch[0]
            
            preds = self.forward(batch,train=False)
            heads, dep_rels , output = self.parser.decode(preds[0], preds[1], sent_lens,verbose=True)
            for outs,sent,l in zip(output,tokens,sent_lens):
                new_sent = []
                assert len(sent[1:l]) == len(outs), "Sizes do not match"
                for pred,tok in zip(outs,sent[1:l]):
                    new_sent.append([tok]+pred)
                data.append(new_sent)
         
        data = unsort_dataset(data,orig_idx)
        
        conll_writer(pred_file, data, field_names,task_name = "dep")
        p, r, f1 = score(pred_file, gold_file,verbose=False)
        #p,r, f1 = 0,0,0
        logging.info("LAS Precision : {}  Recall {} F1 {}".format(p,r,f1))
        self.parser.train() 
        
        return p, r, f1

def main(args = None):
    
    print("Working on : {}".format(device))
    
    file_name = "../../datasets/tr_imst-ud-train.conllu"
    val_name = "../../datasets/tr_imst-ud-dev.conllu"
    args = {"pos_dim" : 50}
    depdataset = DepDataset(file_name,batch_size = 300)
    dep_valid = DepDataset(val_name,batch_size = 300 , vocabs=depdataset.vocabs, for_eval=True)
    
    ParseTrainer = DepTrainer(args, depdataset)
    
    EPOCH =  30
    L = len(depdataset)
    logging.info("Pos vocab nasil ")
    logging.info(depdataset.vocabs['pos_vocab'].w2ind)
    logging.info(depdataset.vocabs['dep_vocab'].w2ind)
    for i in tqdm(range(EPOCH),desc= "Epochs"):
        train_loss = 0
        ex = 0
        dep_loss = 0 
        for j in tqdm(range(L), unit="batch",desc = "Training"):
            batch = depdataset[j]
            loss, deprel, depind = ParseTrainer.update(batch)
            train_loss += loss
            dep_loss += deprel
            ex += len(batch[1])
            if j%10 == 1:
                logging.info("Average Train Loss  {} after {} sentences ".format(train_loss/(j+1),ex))
                logging.info("Dep rel losses {}".format(dep_loss/(j+1)))
        logging.info("Final train loss of epoch {} :  {}".format(i+1,train_loss/L))
        p, r, f1 = ParseTrainer.evaluate(dep_valid)
if __name__ == "__main__":

    main()


