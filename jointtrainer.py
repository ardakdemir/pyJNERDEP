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
from torch.nn.utils import clip_grad_norm_
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from jointner import JointNer
from jointparser import JointParser
from parser.parsereader import DepDataset
from parser.utils import conll_writer, sort_dataset, unsort_dataset, score, convert2IOB2new
from datareader import DataReader
from bertner import BertNER
from evaluate import Evaluate
from conll_eval import evaluate_conll_file

import sys
import logging
import time
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_pred_content(tokens, preds, truths=None, label_voc=None):
    ## this is where the end token gets eliminated!!
    sents = []
    if truths:
        for sent,pred,truth in zip(tokens,preds,truths):
            l = list(zip(sent[1:-1], label_voc.unmap(truth[1:-1]),label_voc.unmap(pred[1:-1])))
            sents.append(l)
    else:
        for sent,pred in zip(tokens,preds):
            sents.append(list(zip(sent[:-1],label_voc.unmap(pred[:-1]))))
    
    return sents
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/depparse', help='Root dir for models.')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/word2vec', help='Directory of word vectors')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--gold_file', type=str, default=None, help='Output CoNLL-U file.')
    
    
    parser.add_argument('--log_file', type=str, default='jointtraining.log', help='Input file for data loader.')
    parser.add_argument('--ner_train_file', type=str, default='../../datasets/train_pos.tsv', help='training file for ner')
    parser.add_argument('--dep_train_file', type=str, default="../../datasets/tr_imst-ud-train.conllu", help='training file for dep')
    parser.add_argument('--ner_val_file', type=str, default="../../datasets/test_pos.tsv", help='validation file for ner')
    parser.add_argument('--dep_val_file', type=str, default="../../datasets/tr_imst-ud-test.conllu", help='validation file for dep')
    parser.add_argument('--ner_output_file', type=str, default="joint_ner_out.txt", help='Output file for named entity recognition')
    parser.add_argument('--dep_output_file', type=str, default="joint_dep_out.txt", help='Output file for dependency parsing')
    parser.add_argument('--conll_file_name', type=str, default='conll_ner_output', help='Output file name in conll bio format')
    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--lstm_hidden', type=int, default=256)
    parser.add_argument('--char_hidden_dim', type=int, default=200)
    parser.add_argument('--biaffine_hidden', type=int, default=400)
    parser.add_argument('--composite_deep_biaff_hidden_dim', type=int, default=100)
    parser.add_argument('--cap_dim', type=int, default=32)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--cap_types', type=int, default=6)
    parser.add_argument('--pos_dim', type=int, default=64)
    parser.add_argument('--dep_dim', type=int, default=128)
    parser.add_argument('--transformed_dim', type=int, default=125)
    
    parser.add_argument('--lstm_layers', type=int, default=3)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=-1)
    
    parser.add_argument('--word_drop', type=float, default = 0.33)
    parser.add_argument('--embed_drop', type=float, default = 0.33)
    parser.add_argument('--lstm_drop', type=float, default = 0.5)
    parser.add_argument('--crf_drop', type=float, default=0.3)
    parser.add_argument('--parser_drop', type=float, default=0.40)
    
    parser.add_argument('--rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    parser.add_argument('--no_pretrain', dest='pretrain', action='store_false', help="Turn off pretrained embeddings.")
    parser.add_argument('--no_linearization', dest='linearization', action='store_false', help="Turn off linearization term.")
    parser.add_argument('--no_distance', dest='distance', action='store_false', help="Turn off distance term.")

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--ner_lr', type=float, default=0.002, help='Learning rate for ner lstm')
    parser.add_argument('--embed_lr', type=float, default=0.002, help='Learning rate for embeddiing')
    parser.add_argument('--dep_lr', type=float, default=0.002, help='Learning rate dependency lstm')
    parser.add_argument('--lr_decay', type=float, default=0.999, help='Learning rate decay')
    
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0.90)
    
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--dep_warmup', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--max_steps_before_stop', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--max_depgrad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='..', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default='best_joint_model.pkh', help="File name to save the model")
    parser.add_argument('--save_ner_name', type=str, default='best_ner_model.pkh', help="File name to save the model")
    parser.add_argument('--save_dep_name', type=str, default='best_dep_model.pkh', help="File name to save the model")
    parser.add_argument('--load_model', type=int, default=0, help='Binary for loading previous model')

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    parser.add_argument('--hierarchical', type=int, default=0, help=' Choose whether to train a hiearchical or flat model')
    args = parser.parse_args()
    return args


class JointModel(nn.Module):

    def __init__(self,args):
        super(JointModel,self).__init__()
        self.args = args
        #base model for generating bert output
        self.base_model = BaseModel(self.args)
        
        self.args['lstm_input_size'] = self.base_model.lstm_input_size  
        
        self.depparser= JointParser(self.args)
        
        self.nermodel = JointNer(self.args)





class BaseModel(nn.Module):
    
    def __init__(self,args):

        super(BaseModel,self).__init__()
        
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
        
        self.w_dim = self.bert_model.encoder.layer[11].output.dense.out_features
        #self.vocab_size = args['vocab_size']
        self.args = args        
        self.cap_types = self.args['cap_types']
        self.cap_dim = args['cap_dim']
        self.pos_dim = self.args['pos_dim']
        self.embed_drop = args['embed_drop']
        self.lstm_drop = args['lstm_drop']
        self.weight_decay = self.args['weight_decay']
        self.lstm_input_size = self.w_dim + self.cap_dim + self.pos_dim
      
        self.cap_embeds  = nn.Embedding(self.cap_types, self.cap_dim)
        self.pos_embeds  = nn.Embedding(self.args['pos_vocab_size'], self.pos_dim)
        #self.word_embeds = nn.Embedding(self.vocab_size, self.w_dim)
        
        #self.bilstm  = nn.LSTM(self.lstm_input, self.lstm_hidden, bidirectional=True, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(self.lstm_drop)
        self.embed_dropout = nn.Dropout(self.embed_drop)
        

        self.embed_optimizer = optim.AdamW([
        {"params": self.cap_embeds.parameters()},\
        {"params": self.pos_embeds.parameters()}],\
        lr=self.args['embed_lr'],betas=(0.9,self.args['beta2']), eps=1e-6 )
             
        param_optimizer = list(self.bert_model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
         ]
        bert_optimizer = AdamW(optimizer_grouped_parameters,
                         lr=2e-5)
        self.bert_optimizer = bert_optimizer
   
    def _get_bert_batch_hidden(self, hiddens , bert2toks, layers=[-2,-3,-4]):
        meanss = torch.mean(torch.stack([hiddens[i] for i in layers]),0)
        batch_my_hiddens = []
        for means,bert2tok in zip(meanss,bert2toks):
            my_token_hids = []
            my_hiddens = []
            for i,b2t in enumerate(bert2tok):
                if i>0 and b2t!=bert2tok[i-1]:
                    my_hiddens.append(torch.mean(torch.stack(my_token_hids),0))
                    my_token_hids = [means[i+1]] ## we skip the CLS token
                else:
                    my_token_hids.append(means[i+1])
            my_hiddens.append(torch.mean(torch.stack(my_token_hids),0))
            sent_hiddens = torch.stack(my_hiddens)
            batch_my_hiddens.append(sent_hiddens)
        return torch.stack(batch_my_hiddens)
    
    def _get_bert_batch_hidden2(self, hiddens , bert2toks, layers=[-2,-3,-4]):
        meanss = torch.mean(torch.stack([hiddens[i] for i in layers]),0)
        batch_my_hiddens = []
        for means,bert2tok in zip(meanss,bert2toks):
            my_token_hids = []
            my_hiddens = []
            for i,b2t in enumerate(bert2tok):
                if i>0 and b2t!=bert2tok[i-1]:
                    my_hiddens.append(torch.mean(torch.cat(my_token_hids),0).view(1,-1))
                    my_token_hids.append(means[i+1].view(1,-1))
            my_hiddens.append(torch.mean(torch.cat(my_token_hids),0).view(1,-1))
            batch_my_hiddens.append(torch.cat(my_hiddens))
        return torch.stack(batch_my_hiddens)

    def forward(self,pos_ids, batch_bert_ids, batch_seq_ids, bert2toks, cap_inds, sent_lens):
        
        bert_out = self.bert_model(batch_bert_ids,batch_seq_ids)
        bert_hiddens = self._get_bert_batch_hidden(bert_out[2],bert2toks)
        
        cap_embedding = self.embed_dropout(self.cap_embeds(cap_inds))
        pos_embedding = self.embed_dropout(self.pos_embeds(pos_ids))
        bert_hiddens = self.dropout(bert_hiddens)
        
        concat = torch.cat((bert_hiddens,cap_embedding,pos_embedding),dim=2) 
        #bilstms are separate for each task
        #padded = pack_padded_sequence(concat,sent_lens,batch_first=True)
        #lstm_out,_ = self.bilstm(padded)
        #unpacked, _ = pad_packed_sequence(lstm_out, batch_first=True)
        #unpacked = self.dropout(unpacked)
        
        return concat

class JointTrainer:
    
    def __init__(self,args):
        self.args = args
        #args2 = {'bert_dim' : 100,'pos_dim' : 12, 'pos_vocab_size': 23,'lstm_hidden' : 10,'device':device}
        #self.args  = {**self.args,**args2}
        self.getdatasets()
        self.args['ner_cats'] = len(self.nertrainreader.label_voc)
        ## feature extraction
        self.device = self.args['device']
        print("Joint Trainer initialized on {}".format(self.device))
        self.jointmodel=JointModel(self.args)
        self.jointmodel.base_model.pos_vocab = self.pos_vocab
        self.nertrainreader.pos_vocab = self.pos_vocab
        self.nervalreader.pos_vocab = self.pos_vocab
        self.jointmodel.depparser.vocabs = self.deptraindataset.vocabs 
        self.jointmodel.to(self.device)
        self.nerevaluator = Evaluate("NER")
        
    def update_lr(self):
        for param_group in self.jointmodel.base_model.embed_optimizer.param_groups:
            param_group['lr']*=self.args['lr_decay']
        for param_group in self.jointmodel.depparser.optimizer.param_groups:
            param_group['lr']*=self.args['lr_decay']
        for param_group in self.jointmodel.nermodel.ner_optimizer.param_groups:
            param_group['lr']*=self.args['lr_decay']

    def getdatasets(self):
        
        self.nertrainreader = DataReader(self.args['ner_train_file'],"NER",batch_size = self.args['batch_size'])
        self.nervalreader = DataReader(self.args['ner_val_file'],"NER", batch_size = self.args['batch_size'])
        self.nervalreader.label_voc = self.nertrainreader.label_voc
        
        self.deptraindataset = DepDataset(self.args['dep_train_file'],batch_size = self.args['batch_size'])
        self.depvaldataset  = DepDataset(self.args['dep_val_file'], batch_size = self.args['batch_size'], vocabs = self.deptraindataset.vocabs, for_eval=True)
        self.args['vocab_size'] = len(self.nertrainreader.word_voc)
        self.args['pos_vocab_size'] = len(self.deptraindataset.vocabs['pos_vocab'])
        self.pos_vocab = self.deptraindataset.vocabs['pos_vocab']
        self.args['dep_cats'] = len(self.deptraindataset.vocabs['dep_vocab'])
        assert self.args['dep_cats'] == self.deptraindataset.num_rels, 'Dependency types do not match'
        assert self.args['pos_vocab_size'] == self.deptraindataset.num_pos," Pos vocab size do not match "

    def forward(self, batch, task= "DEP"):
        tokens, bert_batch_after_padding, data = batch
        inputs = []
        for d in data:
            inputs.append(d.to(self.device))
        sent_lens, masks, tok_inds, ner_inds, pos_inds,_, _,\
             bert_batch_ids,  bert_seq_ids, bert2toks, cap_inds = inputs
        features = self.jointmodel.base_model(pos_inds, bert_batch_ids, bert_seq_ids, bert2toks, cap_inds, sent_lens) 
        return features
    
    def ner_loss(self,batch):
        sent_lens = batch[2][0].to(self.device)
        ner_inds = batch[2][3].to(self.device)
        bert_feats = self.forward(batch, task="NER") 
        crf_scores = self.jointmodel.nermodel(bert_feats, sent_lens)
        loss = self.jointmodel.nermodel.loss(crf_scores, ner_inds, sent_lens)
        loss = loss/ self.args['batch_size']
        return loss 
    
        
    def dep_forward(self, dep_batch, task = "DEP", training = True):
        bert_feats = self.forward(dep_batch)
        inputs = []
        data = dep_batch[2]
        for d in data:
            inputs.append(d.to(self.device))
        sent_lens, masks, _, _, pos, dep_inds, dep_rels, \
            _ , _ , _ , _  = inputs
        tokens = dep_batch[0]
        
        if task=="NER":
            dep_out = self.jointmodel.depparser(masks,bert_feats, dep_inds, dep_rels, pos, sent_lens, training=True, task=task)
            return dep_out
        #logging.info("Head predictions ")
        #logging.info(preds[0][-1])
        #logging.info(dep_inds[-1])
        else:
            preds, loss, deprel_loss, depind_loss, acc, head_acc = self.jointmodel.depparser(masks,bert_feats, dep_inds, dep_rels, pos, sent_lens, training=training, task=task)
            return  loss, preds, deprel_loss, depind_loss, acc , head_acc 
    
    def ner_update(self,batch):
        
        self.jointmodel.base_model.embed_optimizer.zero_grad()
        self.jointmodel.base_model.bert_optimizer.zero_grad()
        self.jointmodel.nermodel.ner_optimizer.zero_grad()
        
        if self.args['hierarchical']==1: 
            self.jointmodel.depparser.optimizer.zero_grad()
        
        #batch = self.nertrainreader[i]
        sent_lens = batch[2][0].to(self.device)
        ner_inds = batch[2][3].to(self.device)
        
        ## if hierarchical dep embeddings are appended to the bert outputs
        if self.args['hierarchical']==1: 
            bert_feats = self.dep_forward(batch, task="NER")

        else:
            bert_feats = self.forward(batch, task="NER") 
        crf_scores = self.jointmodel.nermodel(bert_feats, sent_lens)
        

        loss = self.jointmodel.nermodel.loss(crf_scores, ner_inds, sent_lens)
        loss = loss/ sum(sent_lens).item()
        loss.backward()
        pos_inds = batch[2][4]
         
        clip_grad_norm_(self.jointmodel.nermodel.parameters(),self.args['max_grad_norm'])
        clip_grad_norm_(self.jointmodel.base_model.parameters(),self.args['max_grad_norm'])
        if self.args['hierarchical']==1: 
            clip_grad_norm_(self.jointmodel.depparser.parameters(),self.args['max_grad_norm'])
        
        self.jointmodel.base_model.embed_optimizer.step()
        self.jointmodel.base_model.bert_optimizer.step()
        self.jointmodel.nermodel.ner_optimizer.step()
        if self.args['hierarchical']==1: 
            self.jointmodel.depparser.optimizer.step()
    
        return loss.item()
    
    def dep_update(self,dep_batch):
        
        self.jointmodel.base_model.embed_optimizer.zero_grad()
        self.jointmodel.base_model.bert_optimizer.zero_grad()
        self.jointmodel.depparser.optimizer.zero_grad()
        
        dep_loss,_,deprel_loss, depind_loss, acc, head_acc = self.dep_forward(dep_batch)
        dep_loss = dep_loss/self.args['batch_size']
        deprel_loss /=self.args['batch_size']
        depind_loss /= self.args['batch_size']
        dep_loss.backward()
        
        clip_grad_norm_(self.jointmodel.base_model.parameters(),self.args['max_depgrad_norm'])
        clip_grad_norm_(self.jointmodel.depparser.parameters(),self.args['max_depgrad_norm'])
        
        self.jointmodel.base_model.embed_optimizer.step()
        self.jointmodel.base_model.bert_optimizer.step()
        self.jointmodel.depparser.optimizer.step()
        
        return dep_loss.item(), deprel_loss.item(), depind_loss.item(), acc, head_acc
    
    def train(self):
        logging.info("Training on {} ".format(self.args['device']))
        logging.info("Dependency pos vocab : {} ".format(self.deptraindataset.vocabs['pos_vocab'].w2ind))
        logging.info("Dependency dep vocab : {} ".format(self.deptraindataset.vocabs['dep_vocab'].w2ind))
        epoch = self.args['max_steps']//self.args['eval_interval']
        self.jointmodel.train()
        best_ner_f1 = 0
        best_dep_f1 = 0

        if self.args['load_model']==1: 
            save_path = os.path.join(self.args['save_dir'],self.args['save_name'])
            logging.info("Model loaded %s"%save_path)
            self.jointmodel.load_state_dict(torch.load(save_path))
        print("Pos tag vocab for dependency dataset")
        print(self.deptraindataset.vocabs['pos_vocab'].w2ind)
        print("Pos tag vocab for named entity dataset")
        print(self.nertrainreader.pos_vocab.w2ind)
        print("Dep vocab for dependency dataset")
        print(self.deptraindataset.vocabs['dep_vocab'].w2ind)
        for e in range(epoch):
            train_loss = 0
            ner_losses = 0
            dep_losses = 0
            depind_losses = 0
            deprel_losses = 0
            deprel_acc = 0
            uas_epoch = 0
            self.nertrainreader.for_eval = False
            self.jointmodel.train()
            for i in tqdm(range(self.args['eval_interval'])):
    
                if e >= self.args['dep_warmup']:
                    
                    ner_batch = self.nertrainreader[i]
                    #logging.info(ner_batch[0])
                    ner_loss = self.ner_update(ner_batch)
                    ner_losses += ner_loss
                    train_loss +=ner_loss
                
                dep_batch = self.deptraindataset[i]
                dep_loss, deprel_loss, depind_loss, acc, uas = self.dep_update(dep_batch)
                dep_losses += dep_loss
                train_loss +=dep_loss
                deprel_losses += deprel_loss
                depind_losses += depind_loss
                deprel_acc += acc
                uas_epoch += uas
            logging.info("Results for epoch : {}".format(e+1))
            logging.info("Unlabeled attachment score : {} ".format(uas_epoch/(i+1)))
            self.jointmodel.eval()
            dep_pre, dep_rec, dep_f1 = self.dep_evaluate()
            ner_f1 = 0
            if e >= self.args['dep_warmup']:
                ner_pre, ner_rec, ner_f1 = self.ner_evaluate()
                logging.info("NER Results -- f1 : {} ".format(ner_f1))
            logging.info("Dependency Results -- f1 : {} ".format(dep_f1))
            
            if dep_f1 > best_dep_f1:
                self.save_model(self.args['save_dep_name'])
                best_dep_f1 = dep_f1
                best_dep_epoch = e
            
            else:
                logging.info("Best LAS of {} achieved at {}".format(best_dep_f1, best_dep_epoch))
                for param_group in self.jointmodel.depparser.optimizer.param_groups:
                    param_group['lr']*=self.args['lr_decay']
                
                for param_group in self.jointmodel.base_model.embed_optimizer.param_groups:
                    param_group['lr']*=self.args['lr_decay']
            
            if ner_f1 > best_ner_f1:
                best_ner_epoch = e
                self.save_model(self.args['save_ner_name'])
                best_ner_f1 = ner_f1
            
            else: 
                logging.info("Best F-1 for NER  of {} achieved at {}".format(best_ner_f1, best_ner_epoch))
                for param_group in self.jointmodel.nermodel.ner_optimizer.param_groups:
                    param_group['lr']*=self.args['lr_decay']
                for param_group in self.jointmodel.base_model.embed_optimizer.param_groups:
                    param_group['lr']*=self.args['lr_decay']
            
            
            if ner_f1 > best_ner_f1 and dep_f1 > best_dep_f1:
                self.save_model(self.args['save_name'])
            self.jointmodel.train()
    
    
    def save_model(self,save_name,weights = True): 
        save_name = os.path.join(self.args['save_dir'],save_name)
        if weights:
            logging.info("Saving best model to {}".format(save_name))
            torch.save(self.jointmodel.state_dict(), save_name)
    def dep_evaluate(self): 
        
        logging.info("Evaluating dependency performance on {}".format(self.depvaldataset.file_name))
        self.jointmodel.eval()
        dataset = self.depvaldataset
        orig_idx = self.depvaldataset.orig_idx
        data = []
        
        field_names = ["word", "head", "deprel"]
        gold_file = dataset.file_name
        pred_file = "pred_"+gold_file.split("/")[-1]    
        start_id = orig_idx[0]
        #for x in tqdm(range(len(self.depvaldataset)),desc = "Evaluation"):
        rel_accs = 0
        head_accs = 0
        self.depvaldataset.for_eval=True
        for x in tqdm(range(len(self.depvaldataset)),desc = "Evaluation"):
            #batch = dataset[x]
            batch = self.depvaldataset[x]
            sent_lens = batch[2][0]
            tokens = batch[0]
            
            loss, preds, _ , _, rel_acc , head_acc = self.dep_forward(batch,training=False)
            rel_accs += rel_acc
            head_accs += head_acc
            heads, dep_rels , output = self.jointmodel.depparser.decode(preds[0], preds[1], sent_lens,verbose=True)
            for outs,sent,l in zip(output,tokens,sent_lens):
                new_sent = []
                assert len(sent[1:l]) == len(outs), "Sizes do not match"
                for pred,tok in zip(outs,sent[1:l]):
                    new_sent.append([tok]+pred)
                data.append(new_sent)
        #print(orig_idx)
        #print(len(data))
        head_accs /= len(self.depvaldataset)
        rel_accs /= len(self.depvaldataset)
        print("Head prediction accuracy {}  rel prediction accuracy {}".format(head_accs,rel_accs))
        data = unsort_dataset(data,orig_idx)
        pred_file = os.path.join(self.args['save_dir'],pred_file)
        conll_writer(pred_file, data, field_names,task_name = "dep")
        print("Predictions can be observed from {}".format(pred_file))
        p, r, f1 = score(pred_file, gold_file,verbose=False)
        #p,r, f1 = 0,0,0
        logging.info("LAS Precision : {}  Recall {} F1 {}".format(p,r,f1))
        print("Dependency output:")
        print("LAS Precision : {}  Recall {} F1 {}".format(p,r,f1))
        #self.parser.train() 
        
        return p, r, f1
    
    
    def ner_evaluate(self):
        self.jointmodel.eval()
        sents = []
        preds = []
        truths = []
        ## for each batch
        dataset = self.nervalreader
        dataset.for_eval = True
        for i in tqdm(range(len(dataset))):
            d = dataset[i]
            tokens = d[0]
            sent_lens = d[2][0]
            ner_inds = d[2][3]
            with torch.no_grad():
                if self.args['hierarchical'] == 0:
                    bert_out = self.forward(d,task="NER")
                else:
                    bert_out = self.dep_forward(d,task="NER")
                crf_scores = self.jointmodel.nermodel(bert_out, sent_lens,train=False)
                paths, scores = self.jointmodel.nermodel.batch_viterbi_decode(crf_scores, sent_lens)
                for i in range(crf_scores.shape[0]):
                    truth = ner_inds[i].detach().cpu().numpy()//self.args['ner_cats']##converting 1d labels
                    sents.append(tokens[i])
                    preds.append(paths[i])
                    truths.append(truth) 
        
        content = generate_pred_content(sents, preds, truths, label_voc = self.nertrainreader.label_voc)
        orig_idx = dataset.orig_idx
        content = unsort_dataset(content,orig_idx)
        
        field_names = ["token","truth", "ner_tag"]
        out_file = os.path.join(self.args['save_dir'],self.args['ner_output_file'])

        conll_writer(out_file,content,field_names,"ner")
        conll_file = os.path.join(self.args['save_dir'],self.args['conll_file_name'])
        convert2IOB2new(out_file,conll_file)
        prec, rec, f1 = evaluate_conll_file(open(conll_file,encoding='utf-8').readlines())
        #logging.info("{} {} {} ".format(prec,rec,f1))
        my_pre, my_rec, my_f1 = self.nerevaluator.conll_eval(out_file)
        logging.info("My values ignoring the boundaries.\npre: {}  rec: {}  f1: {} ".format(my_pre,my_rec,my_f1))
        print(" NER results : pre: {}  rec: {}  f1: {}".format(my_pre,my_rec,my_f1))
        #self.model.train()
        return my_pre, my_rec, my_f1




if __name__ == '__main__':
    args = parse_args()
    print(args)
    vars(args)['device'] = device
    args = vars(args)
    if not os.path.isdir(args['save_dir']):
        os.mkdir(args['save_dir'])
    log_path = os.path.join(args['save_dir'],args['log_file'])
    logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler(log_path, 'w', 'utf-8')], format='%(levelname)s - %(message)s')
    jointtrainer = JointTrainer(args)
    jointtrainer.train()
