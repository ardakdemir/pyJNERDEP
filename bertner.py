from __future__ import print_function
from __future__ import division
import torch
from skimage import io, transform
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
import os
import copy
from pdb import set_trace
from crf import CRF, CRFLoss
import unidecode
import logging
from pytorch_transformers import *

from datareader import DataReader, START_TAG, END_TAG, PAD_IND, START_IND, END_IND



class BertNER(nn.Module):
    def __init__(self,args,l2ind, lstm_hidden=100,channel_size=400,num_cat=5, device='cpu'):
        super(BertNER,self).__init__()
        self.args = args
        self.args['ner_cats'] = num_cat
        self.num_cat = num_cat
        self.l2ind = l2ind
        print(l2ind)
        print(num_cat)
        self.START_TAG = "[SOS]"
        self.END_TAG = "[EOS]"
        self.device = device
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.bert_model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
        self.w_dim = self.bert_model.encoder.layer[11].output.dense.out_features
        #self.vocab_size = vocab_size
        self.cap_types = 4
        self.cap_dim = 10
        self.drop_prob = 0.5
        self.lstm_input_size = self.w_dim + self.cap_dim
        self.cap_embeds  = nn.Embedding(self.cap_types,self.cap_dim)
        #self.word_embeds = nn.Embedding(self.vocab_size, self.w_dim)
        self.bilstm  = nn.LSTM(self.lstm_input_size, lstm_hidden, bidirectional=True, num_layers=1, batch_first=True)
        
        self.fc = nn.Linear(lstm_hidden*2, self.num_cat)
        self.crf = CRF(lstm_hidden*2, self.num_cat,self.device)
        assert self.START_TAG in l2ind, "Add the start and end  tags!!"
        self.crf_loss = CRFLoss(self.args,device =self.device)
        self.transitions = nn.Parameter(torch.randn(self.num_cat, self.num_cat,dtype=torch.float,device=device))
        self.dropout = nn.Dropout(self.drop_prob)
        self.drop_replacement = nn.Parameter(torch.randn(self.lstm_input_size) / np.sqrt(self.lstm_input_size))
        #self.crf.transition[self.l2ind[self.START_TAG],:] = torch.tensor([-10000]).expand(1,self.num_cat)
        #self.crf.transition[:,self.l2ind[self.END_TAG]] = torch.tensor([-10000]).expand(1,self.num_cat)
    
    def _viterbi_decode3(self,feats,sent_len):
        start_ind = self.l2ind[self.START_TAG]
        end_ind = self.l2ind[self.END_TAG]
        #feats = feats[:,end_ind+1:,end_ind+1:]
        parents = [[start_ind for x in range(feats.size()[1])]]
        layer_scores = feats[0,:,start_ind] 
        for feat in feats[1:sent_len,:,:]:
            #layer_scores =feat[:,:start_ind,:start_ind] + layer_scores.unsqueeze(1).expand(1,layer_scores.shape[1],layer_scores.shape[2])
            layer_scores =feat + layer_scores.unsqueeze(0).expand(layer_scores.shape[0],layer_scores.shape[0])
            layer_scores, parent = torch.max(layer_scores,dim=1)
            parents.append(parent)
        #layer_scores = layer_scores + self.crf.transitions[self.l2ind[END_TAG],:] 
        
        path = [end_ind]
        path_score = layer_scores[end_ind]
        parent = path[0]
        #parents.reverse()
        for p in range(len(parents)-1,0,-1):
            path.append(parents[p][parent].item())
            parent = parents[p][parent]
        path.reverse()
        return path, path_score.item()
    
    
    def _viterbi_decode2(self,feats):
        feats = feats[:self.l2ind[self.START_TAG]]
        parents = [[self.l2ind[START_TAG] for x in range(feats.size()[1])]]
        layer_scores = feats[0,:] + self.transitions[:self.l2ind[self.START_TAG],self.l2ind[self.START_TAG]]
        for t in range(1,feats.size()[0]):
            time_step = feats[t].view(1,-1)
            new_layer_scores = []
            layer_parent = []
            for i in range(time_step.size()[1]):
                tag_score = time_step[0,i].view(1,-1).expand(1,self.num_cat)\
                    + self.transitions[i,:self.l2ind[START_TAG]] + layer_scores
                parent = tag_score.argmax()
                tag_score = tag_score.max().view(1)
                layer_parent.append(parent.item())
                new_layer_scores.append(tag_score)
            layer_scores = torch.tensor(new_layer_scores).view(1,-1)
            parents.append(layer_parent)
        layer_scores += self.transitions[self.l2ind[self.END_TAG],:self.l2ind[self.START_TAG]].view(1,-1)
        ##backtrack
        path = []
        index = layer_scores[-1].argmax().item()
        path.append(index)
        path_score = layer_scores[-1].max()
        #score = layer_scores[-1].max()
        for i in range(len(parents)-1,-1,-1):
            index = parents[i][index]
            path.append(index)
        assert path.pop()==self.l2ind[self.START_TAG]
        path.reverse()
        return path, path_score

    def _viterbi_decode(self,feats):
        parents = [[self.l2ind[self.START_TAG] for x in range(feats.size()[1])]]
        layer_scores = feats[0,:] + self.transitions[:,self.l2ind[self.START_TAG]]
        for t in range(1,feats.size()[0]):
            time_step = feats[t].view(1,-1)
            new_layer_scores = []
            layer_parent = []
            for i in range(time_step.size()[1]):
                tag_score = time_step[0,i].view(1,-1).expand(1,self.num_cat)\
                    + self.transitions[i] + layer_scores
                parent = tag_score.argmax()
                tag_score = tag_score.max().view(1)
                layer_parent.append(parent.item())
                new_layer_scores.append(tag_score)
            layer_scores = torch.tensor(new_layer_scores,device=self.device).view(1,-1)
            parents.append(layer_parent)
        layer_scores += self.transitions[self.l2ind[self.END_TAG]].view(1,-1)
        ##backtrack
        path = []
        index = layer_scores[-1].argmax().item()
        path.append(index)
        path_score = layer_scores[-1].max()
        #score = layer_scores[-1].max()
        for i in range(len(parents)-1,-1,-1):
            index = parents[i][index]
            path.append(index)
        assert path.pop()==self.l2ind[self.START_TAG]
        path.reverse()
        return path, path_score

    def argmax(self,vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()
    
    
    def _log_sum_exp(self,scores):
        max_score = scores[0, self.argmax(scores)]
        max_broad = max_score.view(1,-1).expand(1,len(scores))
        return max_score + torch.log(torch.sum(torch.exp(scores-max_score)))

    def _forward_score(self,scores):
        forward_score = torch.tensor([-10000 for i in range(len(self.l2ind))],dtype=torch.float,device=self.device)
        forward_score[self.l2ind[self.START_TAG]] = 0
        #forward_score = a
        for time_step in scores :
            time_step_scores = []
            for i, tag in enumerate(time_step):
                tag_score = time_step[i].view(1,-1).expand(1,len(forward_score))\
                    + forward_score + self.transitions[i]
                s = self._log_sum_exp(tag_score).view(1)
                time_step_scores.append(s)
            forward_score = torch.cat(time_step_scores).view(1,-1)
        terminal_score = forward_score + self.transitions[self.l2ind[self.END_TAG]]
        return self._log_sum_exp(terminal_score)

    def _get_bert_score(self, ids, seq_ids, bert2tok):
        bert_output = self.bert_model(ids,seq_ids)
        bert_hiddens = self._get_bert_hiddens(bert_output[2], bert2tok)
        #print(bert_hiddens.view(1,-1,self.w_dim).shape)
        bert_hiddens = bert_hiddens.view(1,-1,self.w_dim)
        out,hidden = self.bilstm(bert_hiddens) #out gives the h_t from last layer for each t
        #print("lstm: ", out.shape)
        scores = self.fc(out.view(bert_hiddens.shape[1],-1))
        return scores

    def _get_bert_hiddens(self, all_hiddens, bert2token, layers=[-2,-3,-4]):
        means = torch.cat([all_hiddens[layer] for layer in layers])
        means = torch.mean(means,0)
        my_hiddens = []
        hidden_ind = 1
        my_token_hids = []
        for i,b2t in enumerate(bert2token):
            if i>0 and b2t!=bert2token[i-1]:
                my_hiddens.append(torch.mean(torch.cat(my_token_hids),0).view(1,-1))
                my_token_hids = [means[i+1].view(1,-1)] ## we skip the CLS token
            else:
                my_token_hids.append(means[i+1].view(1,-1))
        my_hiddens.append(torch.mean(torch.cat(my_token_hids),0).view(1,-1))
        return torch.cat(my_hiddens)

    def _get_scores(self,sent):
        embeds = self.word_embeds(sent).view(1,-1,self.w_dim)
        out,hidden = self.bilstm(embeds) #out gives the h_t from last layer for each t
        scores = self.fc(out.view(len(sent),-1))
        #print(scores.shape)
        return scores

    def _gold_score(self,scores,tags):
        path_score = torch.zeros(1,device = self.device)
        prev_tag = self.l2ind[self.START_TAG]
        end_tag = self.l2ind[self.END_TAG]
        for score,tag in zip(scores,tags):
            path_score += score[tag] + self.transitions[tag,prev_tag]
            prev_tag = tag
        path_score += self.transitions[self.l2ind[self.END_TAG],prev_tag]
        return path_score
    
    
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

    def _get_feats(self,batch_bert_ids, batch_seq_ids, bert2toks, cap_inds, sent_lens):
        bert_out = self.bert_model(batch_bert_ids,batch_seq_ids)
        bert_hiddens = self.dropout(self._get_bert_batch_hidden(bert_out[2],bert2toks))
        cap_embedding = self.cap_embeds(cap_inds)
        concat = self.dropout(torch.cat((bert_hiddens,cap_embedding),dim=2))
        padded = pack_padded_sequence(concat,sent_lens,batch_first=True)
        lstm_out,_ = self.bilstm(padded)
        unpacked , _ = pad_packed_sequence(lstm_out, batch_first=True)
        return self.dropout(unpacked)


    def _crf_neg_loss(self,sent,true_tags):
        scores = self._get_scores(sent)
        gold_score = self._gold_score(scores,true_tags)
        assert scores.shape[0]==true_tags.shape[0]
        forward_score = self._forward_score(scores)
        #print("Gold:",gold_score, " Forw : ", forward_score, "Diff : ", forward_score - gold_score)
        return forward_score - gold_score

    def _bert_crf_neg_loss(self, ids, seq_ids,true_tags, bert2tok):
        scores = self._get_bert_score(ids, seq_ids,  bert2tok)
        #print(scores.shape)
        gold_score = self._gold_score(scores,true_tags)
        assert scores.shape[0]==true_tags.shape[0]
        forward_score = self._forward_score(scores)
        #print("Gold:",gold_score, " Forw : ", forward_score, "Diff : ", forward_score - gold_score)
        return forward_score - gold_score

    def forward(self, ids, seq_ids, bert2tok):
        scores = self._get_bert_score(ids, seq_ids, bert2tok)
        #tag_scores = F.log_softmax(scores, dim=1)
        decoded_path, path_score = self._viterbi_decode(scores)
        return decoded_path, path_score
