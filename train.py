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

import sys
import logging
import time

def ner_train(data_path, val_path, save_path, load = True, gpu = True):
    evaluator = Evaluate("NER")
    logging.basicConfig(level=logging.DEBUG, filename='../trainer2.log', filemode='w', format='%(levelname)s - %(message)s')
    logging.info("GPU : {}".format(gpu))
    datareader = DataReader(data_path, "NER")
    valreader = DataReader(val_path,"NER")
    valreader.l2ind = datareader.l2ind
    valreader.word2ind  = datareader.word2ind
    logging.info("Data is read from %s"%data_path)
    vocab_size = datareader.vocab_size
    l2ind = datareader.l2ind
    num_cat = len(l2ind)
    device = torch.device("cpu")
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertNER(lstm_hidden = 100, vocab_size=vocab_size, l2ind = l2ind, num_cat = num_cat, device = device)
    logging.info("Training on : %s"%device)
    #model = model.to(device)
    model.to(device)
    if os.path.isfile(save_path) and load:
        logging.info("Model loaded %s"%save_path)
        model.load_state_dict(torch.load(save_path))

    optimizer = optim.SGD([{"params": model.fc.parameters()},{"params": model.bilstm.parameters()},\
          {"params": model.cap_embed.parameters()}], lr=0.001, weight_decay = 1e-5)
    param_optimizer = list(model.bert_model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
     ]
    bert_optimizer =AdamW(optimizer_grouped_parameters,
                     lr=2e-5,
                     )
    EPOCH =5
    B_S =  1
    best_loss = -1
    best_model = 0
    L = len(datareader.dataset)
    model.train()
    os.system('nvidia-smi')
    for i in range(EPOCH):
        l = 0
        c,t,p_tot = 0,1,1
        train_loss = 0
        s = time.time()
        for l in tqdm(range(L)):
            my_tokens, bert_tokens, data = datareader.get_bert_input()
            #data = torch.tensor(data)
            #data.to(device)
            for d in data[0]:
                d = d.to(device)
            ids, enc_ids, seq_ids, bert2tok, labels, ortho = data[0]
            #print(my_tokens)
            ids = ids.to(device)
            seq_ids = seq_ids.to(device)
            bert2tok = bert2tok.to(device)
            labels = labels.to(device)
            ortho = ortho.to(device)
            if l==0:
                logging.info(" Device var mi :%s" %bert2tok.device)
                with torch.no_grad():
                    embs = model.cap_embed(ortho).view(-1,self.cap_dim).to("cpu").numpy()
                    logging.info(embs)
                    for tok, ortho in zip(my_tokens,ortho):    
                        logging.info("Word : {}  ortho : {} ".format(tok, ortho.item()))

            #print(labels)
            if len(labels)==1:
                continue
            optimizer.zero_grad()
            bert_optimizer.zero_grad()
            loss = model._bert_crf_neg_loss(ids, seq_ids,labels, bert3tok, ortho)
            loss.to(device)
            loss.backward()
            #logging.info("Loss {}".format(loss.item()))
            #logging.info(model.fc.weight.grad)
            optimizer.step()
            bert_optimizer.step()
            train_loss+= loss.item()
            if l%100 == 99:
                e = time.time()
                d = round(e-s,3)
                logging.info("AVERAGE TRAIN LOSS : {} after {} examples took {} seconds".format( train_loss/l,l , d))
                model.eval()
                for x in range(10):
                    my_tokens, bert_tokens, valdata = valreader.get_bert_input(for_eval=True)
                    ids, enc_ids, seq_ids, bert2tok, labels, ortho = valdata[0]
                    ids = ids.to(device)
                    seq_ids = seq_ids.to(device)
                    bert2tok = bert2tok.to(device)
                    ortho = ortho.to(device)
                    if len(labels)==1:
                        continue
                    with torch.no_grad():
                        decoded_path, score = model(ids,seq_ids, bert2tok, ortho)
                        c_,p_,tot = evaluator.f_1(decoded_path,labels.numpy())
                    logging.info("preds:  {}  true :  {} ".format(decoded_path,labels.numpy()))
                    c+=c_
                    p_tot+=p_
                    t+=tot
                logging.info("Precision : {}  Recall {} Total labels: {} Total predictions : {}".format((c+1)/p_tot,(c+1)/t ,t,p_tot))
                model.train()
        if i==0 or train_loss < best_loss:
            torch.save(model.state_dict(), save_path)
            best_loss = train_loss

if __name__ == "__main__":
    args = sys.argv
    gpu =args[1]
    load = args[2]
    save_path = "../best_model_lstm100.pth"
    data_path = '../datasets/turkish-ner-train.tsv'
    val_path = '../datasets/turkish-ner-dev.tsv'
    ner_train(data_path, val_path, save_path, load = int(load), gpu = int(gpu))
