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
# from jointtrainer import embedding_initializer
from datareader import DataReader, START_TAG, END_TAG, PAD_IND, START_IND, END_IND
from hlstm import HighwayLSTM


def embedding_initializer(dim, num_labels):
    embed = nn.Embedding(num_labels, dim)
    nn.init.uniform_(embed.weight, -np.sqrt(6 / (dim + num_labels)), np.sqrt(6 / (dim + num_labels)))
    return embed


class JointNer(nn.Module):
    def __init__(self, args):
        super(JointNer, self).__init__()
        self.args = args
        self.num_cat = self.args['ner_cats']
        self.START_TAG = "[SOS]"
        self.END_TAG = "[EOS]"
        self.device = self.args['device']

        # self.vocab_size = vocab_size
        self.lstm_drop = self.args['lstm_drop']
        self.embed_drop = self.args['embed_drop']
        if self.args['model_type'] == "DEPNER":
            if self.args['inner'] == 1:
                self.lstm_input_dim = int(self.args['lstm_input_size'] + self.args['lstm_hidden'] * 2)
            else:
                self.lstm_input_dim = int(self.args['lstm_input_size'] + self.args['dep_dim'])
        else:
            self.lstm_input_dim = int(self.args['lstm_input_size'])
        self.lstm_hidden = int(self.args['lstm_hidden'])
        self.lstm_layers = int(self.args['lstm_layers'])
        self.lr = self.args['ner_lr']
        self.weight_decay = self.args['weight_decay']
        self.rec_dropout = self.args['rec_dropout']
        # self.ner_embeds = nn.Embedding(self.num_cat,self.args['ner_dim'])
        self.ner_embeds = embedding_initializer(self.args['ner_dim'], self.num_cat)
        # self.cap_embeds  = nn.Embedding(self.cap_types,self.cap_dim)
        # self.word_embeds = nn.Embedding(self.vocab_size, self.w_dim)

        # self.nerlstm  = nn.LSTM(self.lstm_input_dim, self.lstm_hidden, bidirectional=True, num_layers=self.lstm_layers, batch_first=True)

        self.highwaylstm = HighwayLSTM(self.lstm_input_dim, self.lstm_hidden, bidirectional=True,
                                       num_layers=self.lstm_layers, batch_first=True, dropout=self.lstm_drop, pad=True)

        self.crf = CRF(self.lstm_hidden * 2, self.num_cat, self.device)
        self.crf_loss = CRFLoss(args, device=self.device)

        self.dropout = nn.Dropout(self.lstm_drop)
        self.embed_dropout = nn.Dropout(self.embed_drop)
        self.drop_replacement = nn.Parameter(torch.randn(self.lstm_input_dim) / np.sqrt(self.lstm_input_dim))

        self.ner_optimizer = optim.AdamW([{"params": self.highwaylstm.parameters()}, \
                                          {"params": self.crf.parameters()}, \
                                          {"params": self.ner_embeds.parameters()}], \
                                         lr=self.lr, betas=(0.9, self.args['beta2']), eps=1e-6)

    def batch_viterbi_decode(self, feats, sent_lens):
        paths = []
        scores = []
        sent_lens = sent_lens
        for i in range(feats.size()[0]):
            feat = feats[i]
            sent_len = sent_lens[i]
            path, score = self._viterbi_decode(feat[:], sent_len)
            paths.append(path)
            scores.append(score)
        return paths, scores

    def _viterbi_decode(self, feats, sent_len):
        start_ind = START_IND
        end_ind = END_IND
        # feats = feats[:,end_ind+1:,end_ind+1:]
        parents = [[torch.tensor(start_ind) for x in range(feats.size()[1])]]
        layer_scores = feats[1, :, start_ind]

        for feat in feats[2:sent_len, :, :]:
            # layer_scores =feat[:,:start_ind,:start_ind] + layer_scores.unsqueeze(1).expand(1,layer_scores.shape[1],layer_scores.shape[2])
            layer_scores = feat + layer_scores.unsqueeze(0).expand(layer_scores.shape[0], layer_scores.shape[0])
            layer_scores, parent = torch.max(layer_scores, dim=1)
            parents.append(parent)
        # layer_scores = layer_scores + self.crf.transitions[self.l2ind[END_TAG],:]

        path = [end_ind]
        path_score = layer_scores[end_ind]
        parent = path[0]
        # parents.reverse()
        for p in range(len(parents) - 1, -1, -1):
            path.append(parents[p][parent].item())
            parent = parents[p][parent]
        path.reverse()
        return path, path_score.item()

    def soft_embedding(self, scores):
        embeds = torch.stack(
            [self.ner_embeds(torch.tensor(i, dtype=torch.long).to(self.device)) for i in range(self.num_cat)])
        merge_dim = len(scores.shape) - 1
        probs = F.softmax(scores, dim=merge_dim).unsqueeze(len(scores.shape))

        # logging.info("NER index probs")
        # logging.info(probs[-1])
        # logging.info("Probs shape {}".format(probs.shape))
        soft_embed = torch.sum(probs * embeds, dim=merge_dim)
        return soft_embed

    def argmax(self, vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()

    def _log_sum_exp(self, scores):
        max_score = scores[0, self.argmax(scores)]
        max_broad = max_score.view(1, -1).expand(1, len(scores))
        return max_score + torch.log(torch.sum(torch.exp(scores - max_score)))

    def _get_hlstm_out(self, highway_out):
        if self.args['relu'] == 1:
            return self.dropout(torch.relu(highway_out))
        else:
            return self.dropout(highway_out)

    def _get_feats(self, bert_out, sent_lens, task="NER"):

        padded = pack_padded_sequence(bert_out, sent_lens, batch_first=True)
        # lstm_out,_ = self.nerlstm(padded)
        # bert_out = self.dropout(bert_out)
        highway_out, _ = self.highwaylstm(bert_out, sent_lens)
        hlstm_out = self._get_hlstm_out(highway_out)
        # unpacked , _ = pad_packed_sequence(lstm_out, batch_first=True)
        if task == "NERDEP":
            if self.args['inner'] == 1:
                feats = torch.cat([bert_out, hlstm_out], dim=2)
            else:
                scores = self.crf.emission(hlstm_out)
                for i, s in enumerate(sent_lens):
                    scores[i, 1:s, [PAD_IND, START_IND, END_IND]] = -100
                if self.args['soft'] == 1:
                    soft_embeds = self.soft_embedding(scores)
                    ner_embeds = soft_embeds
                else:
                    # logging.info("Hard embeds")
                    ner_indexes = torch.argmax(scores, dim=2)
                    ner_embeds = self.ner_embeds(ner_indexes)

                # logging.info("Max indexes nedir??")
                # logging.info(ner_indexes[-1])
                # logging.info(ner_embeds.shape)
                feats = torch.cat([bert_out, self.embed_dropout(ner_embeds)], dim=2)
            return feats
        return self.dropout(hlstm_out)

        # return self.dropout(unpacked)
        # return unpacked

    def loss(self, crf_scores, ner_inds, sent_lens):
        return self.crf_loss(crf_scores, ner_inds, sent_lens)

    def predict(self, crf_scores, sent_lens):
        paths = []
        scores = []
        for i in range(crf_scores.shape[0]):
            path, score = self._viterbi_decode(crf_scores[i], sent_lens[i])
            paths.append(path)
            scores.append(score)
        return paths, scores

    def forward(self, bert_out, sent_lens, train=True, task="NER"):
        feats = self._get_feats(bert_out, sent_lens, task=task)
        if task == "NERDEP":
            return feats
        if train:

            with torch.no_grad():
                label_scores = self.crf.emission(feats)
            for i, s in enumerate(sent_lens):
                label_scores[i, 1:s, [PAD_IND, START_IND, END_IND]] = -100
            crf_scores = self.crf(feats)
            # soft_embeds = self.soft_embedding(label_scores)
            # logging.info("Soft embeds nasil seyler")
            # logging.info(soft_embeds[0,0,:10])
            # logging.info(label_scores[0,0])
            # logging.info(self.ner_embeds.weight[:,:10])
            ner_indexes = torch.argmax(label_scores, dim=2)
            # logging.info("Transitions nasil?")
            # logging.info(self.crf.transition.data)
        else:
            with torch.no_grad():
                crf_scores = self.crf(feats)

        return crf_scores
