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

from parser.utils import sort_dataset, unsort_dataset
from parser.utils import conll_writer
from datareader import DataReader
from bertner import BertNER
from evaluate import Evaluate
from conll_eval import evaluate_conll_file
import sys
import logging
import time

logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler('trainread.log','w','utf-8')], format='%(levelname)s - %(message)s')


class NERTrainer:
    def __init__(self,train_file, val_file = None, save_path='ner_best_model.pkt', output_file='ner_out.txt',gpu=1, load = 0):
        self.save_path = save_path
        self.batch_size = 300
        self.gpu = gpu
        self.load = load
        self.get_data(train_file, val_file,batch_size=self.batch_size)##initializes some attributes
        self.val_file = val_file
        self.output_file = output_file
        self.lstm_hidden = 200
        device = torch.device("cpu")

        if gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.args = {}
        self.model = BertNER(self.args, l2ind = self.l2ind,lstm_hidden = self.lstm_hidden,  num_cat = self.num_cat, device = self.device)
        
        self.optimizer = optim.SGD([{"params": self.model.fc.parameters()},\
        {"params": self.model.bilstm.parameters()},\
        {"params": self.model.cap_embeds.parameters()},\
        {"params":self.model.crf.parameters()}],\
        lr=0.1, weight_decay = 1e-3)

        param_optimizer = list(self.model.bert_model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
         ]
        bert_optimizer =AdamW(optimizer_grouped_parameters,
                         lr=2e-5)
        self.bert_optimizer = bert_optimizer

    def get_data(self,train_file,val_file,batch_size=3000):
        datareader = DataReader(train_file, "NER",batch_size=batch_size)
        self.trainreader = datareader
        if val_file:
            valreader = DataReader(val_file,"NER", batch_size=batch_size)
            valreader.l2ind = datareader.l2ind
            valreader.word2ind  = datareader.word2ind
            valreader.label_voc = datareader.label_voc
            valreader.word_voc  = datareader.label_voc
            self.valreader = valreader
        logging.info("Data is read from %s"%train_file)
        vocab_size = datareader.vocab_size
        l2ind = datareader.l2ind
        num_cat = len(l2ind)
        self.vocab_size = vocab_size
        self.l2ind = l2ind
        self.num_cat = num_cat


    def update(self,batch,bert=True):
        if bert:
            self.bert_optimizer.zero_grad()
        self.optimizer.zero_grad()
        loss = self.forward(batch)
        loss.backward()
        self.optimizer.step()
        if bert:
            self.bert_optimizer.step()
        return loss.item()

    def forward(self,batch,train=True):
        tokens, bert_batch_after_padding, data = batch
        inputs = []
        for d in data:
            inputs.append(d.to(self.device))
        lens, tok_inds, ner_inds,\
             bert_batch_ids,  bert_seq_ids, bert2toks, cap_inds = inputs
        #sent_1 = tok_inds[-1].detach().cpu().numpy()
        #lens = lens.to(device)
        #ner_inds = ner_inds.to(device)
        #bert_batch_ids = bert_batch_ids.to(device)
        #bert_seq_ids = bert_seq_ids.to(device)
        #bert2toks = bert2toks.to(device)
        if train :
            feats = self.model._get_feats(bert_batch_ids, bert_seq_ids, bert2toks, cap_inds, lens)
            crf_scores = self.model.crf(feats)
            loss = self.model.crf_loss(crf_scores, ner_inds, lens)
            return loss/torch.sum(lens).item()
        else:
            with torch.no_grad():
                feats = self.model._get_feats(bert_batch_ids, bert_seq_ids, bert2toks, cap_inds, lens)
                crf_scores = self.model.crf(feats)
            return crf_scores
    
    def evaluate(self):
        self.model.eval()
        self.valreader.label_voc = self.trainreader.label_voc
        sents = []
        preds = []
        truths = []
        ## for each batch
        for i in tqdm(range(len(self.valreader))):
            d = self.valreader[i]
            tokens = d[0]
            lens = d[2][0]
            ner_inds = d[2][2]
            with torch.no_grad():
                crf_scores = self.forward(d,train=False)
                for i in range(crf_scores.shape[0]):
                    path, score = self.model._viterbi_decode3(crf_scores[i,:,:,:],lens[i])
                    truth = ner_inds[i].detach().cpu().numpy()//self.num_cat##converting 1d labels
                    sents.append(tokens[i])
                    preds.append(path)
                    truths.append(truth) 
        
        content = generate_pred_content(sents, preds, truths, label_voc = self.trainreader.label_voc)
        orig_idx = self.valreader.orig_idx
        content = unsort_dataset(content,orig_idx)
        field_names = ["token","truth", "ner_tag"]
        out_file = self.output_file
        conll_writer(out_file,content,field_names,"ner")
        prec, rec, f1 = evaluate_conll_file(open(out_file,encoding='utf-8').readlines())
        logging.info("{} {} {} ".format(prec,rec,f1))
        self.model.train()
        return prec, rec, f1
    
    
    def train(self):        
        logging.info("GPU : {}".format(self.gpu))
        print(self.device)
        
        self.model.to(self.device)
        if os.path.isfile(self.save_path) and self.load:
            logging.info("Model loaded %s"%self.save_path)
            self.model.load_state_dict(torch.load(self.save_path))
        
        EPOCH = 30
        best_loss = 1<<31
        best_f1 = 0
        best_epoch = -1
        
        ##repeat
        for i in range(EPOCH):
            epoch_loss = 0
            self.model.train()
            logging.info("Best epoch {} with f1 {}".format(best_epoch,best_f1))
            ##train
            for j in tqdm(range(len(self.trainreader))):
                loss = self.update(self.trainreader[j])
                epoch_loss +=loss
                if j%100 == 99:
                    logging.info("Average loss after {} examples:  {}".format(j,epoch_loss/j))
            ##info
            if epoch_loss < best_loss:
                logging.info("Best epoch for training loss : {} since {}".format(i,best_epoch))
                best_epoch = i
            elif i - best_epoch > 4:
                logging.info("No improvement in crf_loss for the last {} epochs ".format(i-best_epoch))
            ##evaluate
            self.model.eval()
            prec, rec, f1= self.evaluate()
            if f1 > best_f1:
                logging.info("Best f1 achieved!! pre : {}  rec : {}  f1 : {}".format(prec,rec,f1))
                best_f1 = f1
                self.save_model()
    
    
    def save_model(self,weights = True): 
        if weights:
            logging.info("Saving best model to {}".format(self.save_path))
            torch.save(self.model.state_dict(), self.save_path)

def generate_pred_content(tokens, preds, truths=None, label_voc=None):
    
    sents = []
    if truths:
        for sent,pred,truth in zip(tokens,preds,truths):
            sents.append(list(zip(sent[:-1], label_voc.unmap(truth[:-1]),label_voc.unmap(pred[:-1]))))
    else:
        for sent,pred in zip(tokens,preds):
            sents.append(list(zip(sent[:-1],label_voc.unmap(pred[:-1]))))
    
    return sents

def ner_train(data_path, val_path, save_path, load = True, gpu = True):
    evaluator = Evaluate("NER")
    logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler('trainread.log','w','utf-8')], format='%(levelname)s - %(message)s')


    logging.info("GPU : {}".format(gpu))
    batch_size = 300
    datareader = DataReader(data_path, "NER",batch_size=batch_size)
    valreader = DataReader(val_path,"NER", batch_size=batch_size)
    valreader.l2ind = datareader.l2ind
    valreader.word2ind  = datareader.word2ind
    valreader.label_voc = datareader.label_voc
    valreader.word_voc  = datareader.label_voc
    logging.info("Data is read from %s"%data_path)
    
    vocab_size = datareader.vocab_size
    l2ind = datareader.l2ind
    num_cat = len(l2ind)
    device = torch.device("cpu")
    
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model = BertNER(lstm_hidden = 200, vocab_size=vocab_size, l2ind = l2ind, num_cat = num_cat, device = device)
    logging.info("Training on : %s"%device)
    #model = model.to(device)
    model.to(device)
    if os.path.isfile(save_path) and load:
        logging.info("Model loaded %s"%save_path)
        model.load_state_dict(torch.load(save_path))

    optimizer = optim.SGD([{"params": model.fc.parameters()},\
        {"params": model.bilstm.parameters()},\
        {"params": model.cap_embeds.parameters()},\
        {"params":model.crf.parameters()}],\
        lr=0.01, weight_decay = 1e-4)
    param_optimizer = list(model.bert_model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
     ]
    bert_optimizer =AdamW(optimizer_grouped_parameters,
                     lr=2e-5)
    EPOCH = 10
    B_S = 1
    best_loss = -1
    best_model = 0
    L = len(datareader.dataset)
    num_labels = len(datareader.label_voc)
    model.train()
    os.system('nvidia-smi')
    for i in range(EPOCH):
        s = time.time()
        train_loss = 0
        for l in tqdm(range(len(datareader.batched_dataset))):
            optimizer.zero_grad()
            bert_optimizer.zero_grad()
            #my_tokens, bert_tokens, data = datareader.get_bert_input()
            tokens, bert_batch_after_padding, data = datareader[l]

            inputs = []
            for d in data:
                inputs.append(d.to(device))
            lens, tok_inds, ner_inds,\
                 bert_batch_ids,  bert_seq_ids, bert2toks, cap_inds = inputs
            #sent_1 = tok_inds[-1].detach().cpu().numpy()
            #lens = lens.to(device)
            #ner_inds = ner_inds.to(device)
            #bert_batch_ids = bert_batch_ids.to(device)
            #bert_seq_ids = bert_seq_ids.to(device)
            #bert2toks = bert2toks.to(device)
            feats = model._get_feats(bert_batch_ids,bert_seq_ids,bert2toks, cap_inds,lens)
            crf_scores = model.crf(feats)
            loss = model.crf_loss(crf_scores,ner_inds,lens)
            loss = loss/batch_size
            #print(bert_out[2][0].shape)
            loss.to(device)
            loss.backward()
                #print(model.crf.emission.weight.grad)
            optimizer.step()
            bert_optimizer.step()
            train_loss+= loss.item()
            if  l%100 == 10:
                print("Train loss average : {}".format(train_loss/l))
                logging.info("Average training loss : {}".format(train_loss/l))
            
        e = time.time()
        d = round(e-s,3)
        logging.info("AVERAGE TRAIN LOSS : {} after {} examples took {} seconds".format( train_loss/l,l , d))
        model.eval()
        sents = []
        preds = []
        truths = []
        for l in tqdm(range(len(valreader.batched_dataset))):
            tokens, bert_batch_after_padding, data = valreader[l]
            inputs = []
            for d in data:
                inputs.append(d.to(device))
            lens, tok_inds, ner_inds,\
                 bert_batch_ids,  bert_seq_ids, bert2toks,cap_inds = inputs
            with torch.no_grad():
                feats = model._get_feats(bert_batch_ids,bert_seq_ids,bert2toks, cap_inds)
                crf_scores = model.crf(feats)
                for i in range(crf_scores.shape[0]):
                    path, score = model._viterbi_decode3(crf_scores[i,:,:,:],lens[i])
                    truth = ner_inds[i].detach().cpu().numpy()//num_labels
                    sents.append(tokens[i])
                    preds.append(path)
                    truths.append(truth)
                #decoded_path, score = model(ids,seq_ids, bert2tok)
                #c_,p_,tot = evaluator.f_1(decoded_path,labels.numpy())
            #logging.info("preds:  {}  true :  {} ".format(decoded_path,labels.numpy()))
            #c+=c_
            #p_tot+=p_
            #t+=tot
        #logging.info("Precision : {}  Recall {} Total labels: {} Total predictions : {}".format((c+1)/p_tot,(c+1)/t ,t,p_tot))
        
        
        content = generate_pred_content(sents,preds,truths, label_voc = valreader.label_voc)
        field_names = ["token","truth", "ner_tag"]
        out_file = "ner_out.txt"
        conll_writer(out_file,content,field_names,"ner")
        evaluate_conll_file(open(out_file,encoding='utf-8').readlines())
        model.train()
    
        if i==0 or train_loss < best_loss:
            logging.info("Saving best model to {}".format(save_path))
            torch.save(model.state_dict(), save_path)
            best_loss = train_loss

if __name__ == "__main__":
    args = sys.argv
    gpu =args[1]
    load = args[2]
    save_path = "../best_model_batch300_test.pth"
    data_path = '../../datasets/turkish-ner-train.tsv'
    val_path = '../../datasets/turkish-ner-test.tsv'
    #ner_train(data_path, val_path, save_path, load = int(load), gpu = int(gpu))
    nertrainer = NERTrainer(data_path, val_path, save_path, gpu = int(gpu), load = int(load))
    nertrainer.train()
