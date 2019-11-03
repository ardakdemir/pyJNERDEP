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

from datareader import DataReader
from bertner import BertNER
from evaluate import Evaluate

import sys
import logging
import time

def ner_train(data_path, val_path, save_path, load = True, gpu = True):
    evaluator = Evaluate("NER")
    logging.basicConfig(level=logging.DEBUG, encoding= 'utf-8', filename='trainer_batch.log', filemode='w', format='%(levelname)s - %(message)s')
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
    print(device)
    
    model = BertNER(lstm_hidden = 10, vocab_size=vocab_size, l2ind = l2ind, num_cat = num_cat, device = device)
    logging.info("Training on : %s"%device)
    #model = model.to(device)
    model.to(device)
    logging.info(model.parameters())
    if os.path.isfile(save_path) and load:
        logging.info("Model loaded %s"%save_path)
        model.load_state_dict(torch.load(save_path))

    optimizer = optim.SGD([{"params": model.fc.parameters()},\
        {"params": model.bilstm.parameters()},\
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
    EPOCH = 5
    B_S = 1
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
        for l in tqdm(range(len(datareader.batched_dataset))):
            optimizer.zero_grad()
            bert_optimizer.zero_grad()
            #my_tokens, bert_tokens, data = datareader.get_bert_input()
            tokens, bert_batch_after_padding, data = datareader[l]
            logging.info(tokens[0])

            inputs = []
            for d in data:
                inputs.append(d.to(device))
            lens, tok_inds, ner_inds,\
                 bert_batch_ids,  bert_seq_ids, bert2toks = inputs
            sent_1 = tok_inds[0].detach().cpu().numpy()
            logging.info(" ".join(datareader.word_voc.unmap(sent_1)))
            #lens = lens.to(device)
            #ner_inds = ner_inds.to(device)
            #bert_batch_ids = bert_batch_ids.to(device)
            #bert_seq_ids = bert_seq_ids.to(device)
            #bert2toks = bert2toks.to(device)
            feats = model._get_feats(bert_batch_ids,bert_seq_ids,bert2toks)
            logging.info("Feat shape {} ".format(feats.shape))
            crf_scores = model.crf(feats)
            loss = model.crf_loss(crf_scores,ner_inds,lens)
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

            #continue
            
            if l%100 == 10:
                e = time.time()
                d = round(e-s,3)
                logging.info("AVERAGE TRAIN LOSS : {} after {} examples took {} seconds".format( train_loss/l,l , d))
                model.eval()
                for x in range(10):
                    tokens, bert_batch_after_padding, data = valreader[x]
                    inputs = []
                    for d in data:
                        inputs.append(d.to(device))
                    lens, tok_inds, ner_inds,\
                         bert_batch_ids,  bert_seq_ids, bert2toks = inputs
                    with torch.no_grad():
                        feats = model._get_feats(bert_batch_ids,bert_seq_ids,bert2toks)
                        logging.info("Feat shape {} ".format(feats.shape))
                        crf_scores = model.crf(feats)
                        logging.info("CRF Scores shape nedir : {}".format(crf_scores.shape) )
                        for i in range(crf_scores.shape[0]):
                            path, score = model._viterbi_decode3(crf_scores[0,:,:,:])
                            print("Path nasil birsey : {} for word : {} ".format(len(path), crf_scores[0].shape))
                            print(path)
                        #decoded_path, score = model(ids,seq_ids, bert2tok)
                        #c_,p_,tot = evaluator.f_1(decoded_path,labels.numpy())
                    #logging.info("preds:  {}  true :  {} ".format(decoded_path,labels.numpy()))
                    #c+=c_
                    #p_tot+=p_
                    #t+=tot
                #logging.info("Precision : {}  Recall {} Total labels: {} Total predictions : {}".format((c+1)/p_tot,(c+1)/t ,t,p_tot))
                model.train()
        if i==0 or train_loss < best_loss:
            torch.save(model.state_dict(), save_path)
            best_loss = train_loss

if __name__ == "__main__":
    args = sys.argv
    gpu =args[1]
    load = args[2]
    save_path = "../best_model_batch.pth"
    data_path = '../../datasets/turkish-ner-train.tsv'
    val_path = '../../datasets/turkish-ner-dev.tsv'
    ner_train(data_path, val_path, save_path, load = int(load), gpu = int(gpu))
