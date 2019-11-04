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

from parser.utils import conll_writer
from datareader import DataReader
from bertner import BertNER
from evaluate import Evaluate
from conll_eval import evaluate_conll_file
import sys
import logging
import time


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
    
    datareader = DataReader(data_path, "NER")
    valreader = DataReader(val_path,"NER")
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
                 bert_batch_ids,  bert_seq_ids, bert2toks = inputs
            #sent_1 = tok_inds[-1].detach().cpu().numpy()
            #lens = lens.to(device)
            #ner_inds = ner_inds.to(device)
            #bert_batch_ids = bert_batch_ids.to(device)
            #bert_seq_ids = bert_seq_ids.to(device)
            #bert2toks = bert2toks.to(device)
            feats = model._get_feats(bert_batch_ids,bert_seq_ids,bert2toks)
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
                 bert_batch_ids,  bert_seq_ids, bert2toks = inputs
            with torch.no_grad():
                logging.info("Berte giden idler nedir : ")
                logging.info(bert_batch_ids[0].detach().cpu().numpy()[:10])
                feats = model._get_feats(bert_batch_ids,bert_seq_ids,bert2toks)
                crf_scores = model.crf(feats)
                for i in range(crf_scores.shape[0]):
                    path, score = model._viterbi_decode3(crf_scores[i,:,:,:],lens[i])
                    logging.info("Path nasil birsey : {} for word : {} with score {}".format(len(path), crf_scores[0].shape,score))
                    truth = ner_inds[i].detach().cpu().numpy()//num_labels
                    logging.info(path)
                    logging.info("Path uzunlugum : {} - truth length: {}".format(len(path),len(truth[:lens[i]])))
                    logging.info("Gercek indexler: ")
                    for j in range(lens[i]):
                        logging.info("Word : {} pred : {} tag : {} ".format(tokens[i][j],path[j],truth[j]))
                    logging.info(truth[:lens[i]])
                    logging.info("Transitionlar nasil degisiyor end - start tag")
                    logging.info(model.crf.transition.data[2,:])
                    logging.info(model.crf.transition.data[:,1])
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
    save_path = "../best_model_batch.pth"
    data_path = '../../datasets/turkish-ner-train.tsv'
    val_path = '../../datasets/turkish-ner-dev.tsv'
    ner_train(data_path, val_path, save_path, load = int(load), gpu = int(gpu))
