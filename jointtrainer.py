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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from jointner import JointNer
from jointparser import JointParser
from parser.parsereader import DepDataset
from parser.utils import conll_writer
from datareader import DataReader
from bertner import BertNER
from evaluate import Evaluate
from conll_eval import evaluate_conll_file

import sys
import logging
import time
import argparse
logging.basicConfig(level=logging.DEBUG,handlers= [logging.FileHandler('jointtrainer.log','w','utf-8')], format='%(levelname)s - %(message)s')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_pred_content(tokens, preds, truths=None, label_voc=None):
    
    sents = []
    if truths:
        for sent,pred,truth in zip(tokens,preds,truths):
            sents.append(list(zip(sent[:-1], label_voc.unmap(truth[:-1]),label_voc.unmap(pred[:-1]))))
    else:
        for sent,pred in zip(tokens,preds):
            sents.append(list(zip(sent[:-1],label_voc.unmap(pred[:-1]))))
    
    return sents
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/depparse', help='Root dir for saving models.')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/word2vec', help='Directory of word vectors')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--gold_file', type=str, default=None, help='Output CoNLL-U file.')
    
    
    parser.add_argument('--ner_train_file', type=str, default='../../datasets/turkish-ner-train.tsv', help='training file for ner')
    parser.add_argument('--dep_train_file', type=str, default="../../datasets/tr_imst-ud-train.conllu", help='training file for dep')
    parser.add_argument('--ner_val_file', type=str, default='../../datasets/turkish-ner-test.tsv', help='validation file for ner')
    parser.add_argument('--dep_val_file', type=str, default="../../datasets/tr_imst-ud-dev.conllu", help='validation file for dep')
    parser.add_argument('--ner_output_file', type=str, default="joint_ner_out.txt", help='Output file for named entity recognition')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--lstm_hidden', type=int, default=200)
    parser.add_argument('--char_hidden_dim', type=int, default=400)
    parser.add_argument('--biaffine_hidden', type=int, default=400)
    parser.add_argument('--composite_deep_biaff_hidden_dim', type=int, default=100)
    parser.add_argument('--cap_dim', type=int, default=50)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--pos_dim', type=int, default=50)
    parser.add_argument('--transformed_dim', type=int, default=125)
    parser.add_argument('--lstm_layers', type=int, default=3)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=-1)
    parser.add_argument('--word_dropout', type=float, default=0.33)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    parser.add_argument('--no_pretrain', dest='pretrain', action='store_false', help="Turn off pretrained embeddings.")
    parser.add_argument('--no_linearization', dest='linearization', action='store_false', help="Turn off linearization term.")
    parser.add_argument('--no_distance', dest='distance', action='store_false', help="Turn off distance term.")

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--ner_lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--beta2', type=float, default=0.95)

    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--max_steps_before_stop', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='saved_models/depparse', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
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
        self.vocab_size = args['vocab_size']
        
        self.cap_types = 4
        self.cap_dim = args['cap_dim']
        self.drop_prob = args['drop_prob']
        
        self.lstm_input_size = self.w_dim + self.cap_dim
        self.cap_embeds  = nn.Embedding(self.cap_types, self.cap_dim)
        self.word_embeds = nn.Embedding(self.vocab_size, self.w_dim)
        
        #self.bilstm  = nn.LSTM(self.lstm_input, self.lstm_hidden, bidirectional=True, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(self.drop_prob)
        

        self.embed_optimizer = optim.SGD([
        {"params": self.cap_embeds.parameters()}],\
        lr=0.1, weight_decay = 1e-3)
    
        param_optimizer = list(self.bert_model.named_parameters())
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

    def forward(self,batch_bert_ids, batch_seq_ids, bert2toks, cap_inds, sent_lens):
        
        bert_out = self.bert_model(batch_bert_ids,batch_seq_ids)
        bert_hiddens = self._get_bert_batch_hidden(bert_out[2],bert2toks)
        
        cap_embedding = self.cap_embeds(cap_inds)
        concat = torch.cat((bert_hiddens,cap_embedding),dim=2)
        
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
        self.jointmodel.depparser.vocabs = self.deptraindataset.vocabs 
        self.jointmodel.to(self.device)
    def getdatasets(self):
        
        self.nertrainreader = DataReader(self.args['ner_train_file'],"NER",batch_size = self.args['batch_size'])
        self.nervalreader = DataReader(self.args['ner_val_file'],"NER", batch_size = self.args['batch_size'])
        self.nervalreader.label_voc = self.nertrainreader.label_voc
        
        self.deptraindataset = DepDataset(self.args['dep_train_file'],batch_size = self.args['batch_size'])
        self.depvaldataset  = DepDataset(self.args['dep_val_file'], batch_size = self.args['batch_size'], vocabs = self.deptraindataset.vocabs, for_eval=True)
        self.args['vocab_size'] = len(self.nertrainreader.word_voc)
        self.args['pos_vocab_size'] = len(self.deptraindataset.vocabs['pos_vocab'])
        self.args['dep_cats'] = len(self.deptraindataset.vocabs['dep_vocab'])
        assert self.args['dep_cats'] == self.deptraindataset.num_rels, 'Dependency types do not match'
        assert self.args['pos_vocab_size'] == self.deptraindataset.num_pos," Pos vocab size do not match "

    def forward(self, batch, task= "DEP"):
        if task=="NER":
            tokens, bert_batch_after_padding, data = batch
            inputs = []
            for d in data:
                inputs.append(d.to(self.device))
            sent_lens, tok_inds, ner_inds,\
                 bert_batch_ids,  bert_seq_ids, bert2toks, cap_inds = inputs
            features = self.jointmodel.base_model(bert_batch_ids, bert_seq_ids, bert2toks, cap_inds, sent_lens)

            return features   
            
    def train(self):
        logging.info("Training on {} ".format(self.args['device']))

        epoch = 100
        self.jointmodel.train()
        for e in range(epoch):
            train_loss = 0
            for i in tqdm(range(100)):
                 
                batch = self.nertrainreader[0]
                sent_lens = batch[2][0].to(self.device)
                ner_inds = batch[2][2].to(self.device)
                
                self.jointmodel.base_model.embed_optimizer.zero_grad()
                self.jointmodel.base_model.bert_optimizer.zero_grad()
                self.jointmodel.nermodel.ner_optimizer.zero_grad()

                bert_feats = self.forward(batch, task="NER") 
                crf_scores = self.jointmodel.nermodel(bert_feats, sent_lens)
                loss = self.jointmodel.nermodel.loss(crf_scores, ner_inds, sent_lens)
                loss = loss/ self.args['batch_size']
                train_loss += loss.item()
                loss.backward()
                self.jointmodel.base_model.embed_optimizer.step()
                self.jointmodel.base_model.bert_optimizer.step()
                self.jointmodel.nermodel.ner_optimizer.step()
                if i%10 == 9:
                    logging.info("Train loss average : {} after {} examples".format(train_loss/(i+1),i+1))
                    logging.info(self.jointmodel.nermodel.crf.transition)
                    logging.info("Cap embeds: ")
                    logging.info(self.jointmodel.base_model.cap_embeds.weight)
            self.ner_evaluate()
            self.jointmodel.train()
    def ner_evaluate(self):
        self.jointmodel.eval()
        sents = []
        preds = []
        truths = []
        ## for each batch
        for i in tqdm(range(1)):
            d = self.nertrainreader[0]
            tokens = d[0]
            sent_lens = d[2][0]
            ner_inds = d[2][2]
            with torch.no_grad():
                bert_out = self.forward(d,task="NER")
                crf_scores = self.jointmodel.nermodel(bert_out, sent_lens,train=False)
                paths, scores = self.jointmodel.nermodel.batch_viterbi_decode(crf_scores, sent_lens)
                for i in range(crf_scores.shape[0]):
                    truth = ner_inds[i].detach().cpu().numpy()//self.args['ner_cats']##converting 1d labels
                    sents.append(tokens[i])
                    preds.append(paths[i])
                    truths.append(truth) 
        
        content = generate_pred_content(sents, preds, truths, label_voc = self.nertrainreader.label_voc)
        
        field_names = ["token","truth", "ner_tag"]
        out_file = self.args['ner_output_file']
        conll_writer(out_file,content,field_names,"ner")
        prec, rec, f1 = evaluate_conll_file(open(out_file,encoding='utf-8').readlines())
        logging.info("{} {} {} ".format(prec,rec,f1))
        #self.model.train()
        return prec, rec, f1


if __name__ == '__main__':
    args = parse_args()
    print(args)
    vars(args)['device'] = device
    args = vars(args)
    jointtrainer = JointTrainer(args)
    jointtrainer.train()
