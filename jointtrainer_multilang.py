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
import fasttext
import fasttext.util
import torch.optim as optim
import copy
import json
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import gensim
import pickle
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torchvision
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from transformers import AutoTokenizer, AutoModel, BertForPreTraining, BertForTokenClassification
from parser.parsereader import bert2token, pad_trunc_batch
import matplotlib.pyplot as plt
import time
import os
import copy
from pdb import set_trace
import unidecode
# from pytorch_transformers import *
from torch import autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from datetime import date

from jointner import JointNer
from jointparser import JointParser
from parser.parsereader import DepDataset
from parser.utils import conll_writer, sort_dataset, unsort_dataset, score, convert2IOB2new
from datareader import DataReader
from bertner import BertNER
from evaluate import Evaluate
from conll_eval import evaluate_conll_file
import hyperopt
from hyperopt import fmin, tpe, hp, Trials
import sys
import logging
import time
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PAD = "[PAD]"
PAD_IND = 0
START_TAG = "[SOS]"
END_TAG = "[EOS]"
START_IND = 1
END_IND = 2
ROOT_TAG = "[ROOT]"
ROOT_IND = 1
special_labels = [PAD, START_TAG, END_TAG, ROOT_TAG]

lang_abs = {"fi": "finnish", "hu": "hungarian", "cs": "czech", "tr": "turkish", "jp": "japanese"}

model_name_dict = {"jp": "cl-tohoku/bert-base-japanese",
                   "tr": "dbmdz/bert-base-turkish-cased",
                   "hu": "/home/aakdemir/bert_models/hubert",
                   "fi": "TurkuNLP/bert-base-finnish-cased-v1",
                   "cs": "DeepPavlov/bert-base-bg-cs-pl-ru-cased"}

encoding_map = {"cs": "latin-1",
                "tr": "utf-8",
                "hu": "utf-8",
                "fi": "utf-8"}

word2vec_dict = {"jp": "../word_vecs/jp/jp.bin",
                 "tr": "../word_vecs/tr/tr.bin",
                 "hu": "../word_vecs/hu/hu.bin",
                 "fi": "../word_vecs/fi/fi.bin",
                 "cs": "../word_vecs/cs/cs.txt"}

fasttext_dict = {"jp": "../word_vecs/jp/cc.jp.300.bin",
                 "tr": "../word_vecs/tr/cc.tr.300.bin",
                 "hu": "../word_vecs/hu/cc.hu.300.bin",
                 "fi": "../word_vecs/fi/cc.fi.300.bin",
                 "cs": "../word_vecs/cs/cc.cs.300.bin"}

word2vec_lens = {"tr": 200,
                 "hu": 300,
                 "fi": 300,
                 "cs": 100,
                 "jp": 300}


def embedding_initializer(dim, num_labels):
    embed = nn.Embedding(num_labels, dim)
    nn.init.uniform_(embed.weight, -np.sqrt(6 / (dim + num_labels)), np.sqrt(6 / (dim + num_labels)))
    return embed


def load_word2vec(lang):
    model_name = word2vec_dict[lang]
    if lang == "cs":
        model = MyWord2Vec(model_name, lang)
    else:
        model = Word2Vec.load(model_name)
    return model


class MyDict():
    """
        My simple dictionary to allow wv.vocab access to vocab attribute
    """

    def __init__(self, w2v):
        self.w2v = w2v
        self.vocab = set(w2v.keys())

    def __getitem__(self, word):
        return self.w2v[word]

    def __setitem__(self, word, val):
        self.w2v[word] = val

    def __len__(self):
        return len(self.w2v)


class MyWord2Vec():
    """
        My word2Vec that is initialized from a file
    """

    def __init__(self, file_name, lang):
        self.file_name = file_name
        self.lang = lang
        self.vocab, self.wv, self.dim = self.get_vectors(file_name)
        self.encoding_map

    def get_vectors(self, file_name):
        with open(file_name, "r", encoding=encoding_map[self.lang]) as f:
            f = f.read().split("\n")
            wv = {}
            my_len = 0
            for l in f:  # s
                w, v = l.split(" ", 1)
                vec = [float(v_) for v_ in v]
                if len(vec) < 10:
                    continue  # skip not a proper vector
                wv[w] = vec
                length = len(vec)
                if length > 1:
                    my_len = length
        return wv.keys(), wv, length


def get_pretrained_word_embeddings(w2ind, lang='tr', dim='768', word_vec_root="../word_vecs", load_w2v=False):
    vocab_size = len(w2ind)
    # load_path = embedding_path+"_cached.pk"
    print("Getting pretrained embeddings with dim:{}".format(dim))
    start = time.time()
    from_model = True
    pretrained_type = 'fastext' if not load_w2v else 'word2vec'
    if pretrained_type == 'word2vec':
        word_vec_path = os.path.join(word_vec_root, lang, "w2v_{}_{}".format(lang, dim))
        gensim_model = load_word2vec(lang)
        # word_vec_path = os.path.join(word_vec_root,lang,"{}.{}.tsv".format(lang,dim))
        embed = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(embed.weight, -np.sqrt(6 / (dim + vocab_size)), np.sqrt(6 / (dim + vocab_size)))
        c = 0
        for word in list(w2ind.keys()):
            if word in gensim_model.wv.vocab:
                ind = w2ind[word]
                vec = gensim_model.wv[word]
                embed.weight.data[ind].copy_(torch.tensor(vec, requires_grad=True))
                c += 1
        print("Found {} out of {} words in word2vec for {} ".format(c, len(w2ind), lang))
        return embed
    if not from_model:
        load_path = ''
        print("Loading embeddings from {}".format(load_path))
        emb_dict = pickle.load(open(load_path, 'rb'))
        # embed = nn.Embedding(embed_mat.shape[0],embed_mat.shape[1])
        dim = len(list(emb_dict.values())[0])
        # embed.weight.data.copy_(torch.from_numpy(embed_mat))
    else:
        print("Generating embedding from scratch using fasttext model (not .txt file)")

        fastext_path = fasttext_dict[lang]
        if not os.path.exists(fastext_path):
            fasttext.util.download_model(lang, if_exists='ignore')
            ft = fasttext.load_model('cc.{}.300.bin'.format(lang))
        else:
            ft = fasttext.load_model(fastext_path)
        s = time.time()
        # embeddings = open(embedding_path,encoding='utf-8').read().strip().split("\n")
        # print("First line of embeddings")
        # print(embeddings[0])
        # print("Number of words in embeddings {} ".format(len(embeddings)))
        # if len(embeddings[0].split())==2:## with header
        #    embeddings = embeddings[1:]
        # dim = len(embeddings[0].split())-1 ## embedding dimension
        # emb_dict = {l.split()[0]: [float(x) for x in l.split()[1:] ] for l in embeddings[:-1]} ## last index problematic
        # e = time.time()
        # print("Read all embeddings in {} seconds".format(round(e-s,4)))
        # print("Saving embedding dictionary to {}".format(load_path))

        # pickle.dump(emb_dict,open(load_path,'wb'))
        c = 0
        embed = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(embed.weight, -np.sqrt(6 / (dim + vocab_size)), np.sqrt(6 / (dim + vocab_size)))
        for word in list(w2ind.keys()):
            # if emb_dict.get(word) is not None:
            #    ind = w2ind[word]
            # embed.weight.data[ind].copy_(torch.tensor(emb_dict[word],requires_grad=True))
            #    embed.weight.data[ind].copy_(ft_vec)
            #    c +=1
            c += 1
        ind = w2ind[word]
        vec = ft.get_word_vector(word)
        ft_vec = torch.tensor(vec, requires_grad=True)
        embed.weight.data[ind].copy_(ft_vec)
        print("Initialized {} out of {} words from fastext".format(c, vocab_size))
        end = time.time()
        print("Word embeddings initialized in {} seconds ".format(round(end - start, 4)))
    return embed


def generate_pred_content(tokens, preds, truths=None, lens=None, label_voc=None):
    ## this is where the start token and  end token get eliminated!!
    sents = []
    if truths:
        for sent_len, sent, pred, truth in zip(lens, tokens, preds, truths):
            s_ind = 0
            e_ind = 0
            if sent[0] == START_TAG:
                s_ind = 1
            if sent[sent_len - 1] == END_TAG:
                e_ind = 1
            preds = label_voc.unmap(pred[1:-1])
            preds = list(map(lambda x: x if x not in special_labels else "O", preds))
            truth_unmapped = label_voc.unmap(truth[s_ind:sent_len - e_ind])
            # print("Unmapped truths")
            # print(truth_unmapped)
            l = list(zip(sent[s_ind:sent_len - e_ind], truth_unmapped, preds))
            sents.append(l)
    else:
        for sent, pred in zip(tokens, preds):
            end_ind = len(sent) + e_ind
            sents.append(list(zip(sent[s_ind:end_ind], label_voc.unmap(pred[1:-1]))))

    return sents


def read_config(args):
    config_file = args['config_file']
    with open(config_file) as json_file:
        data = json.load(json_file)
        print(data)
        for d in data:
            args[d] = data[d]
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/depparse', help='Root dir for models.')
    parser.add_argument('--data_folder', type=str, default='../../datasets', help='Root directory for all datasets')
    parser.add_argument('--wordvec_dir', type=str, default='../word_vecs', help='Directory of word vectors')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--gold_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--ner_result_out_file', type=str, default="ner_results", help='Output CoNLL-U file.')

    parser.add_argument('--log_file', type=str, default='jointtraining.log', help='Input file for data loader.')
    parser.add_argument('--ner_train_file', type=str, default='../../datasets/traindev_pos.tsv',
                        help='training file for ner')
    parser.add_argument('--dep_train_file', type=str, default="../../datasets/tr_imst-ud-traindev.conllu",
                        help='training file for dep')
    parser.add_argument('--ner_val_file', type=str, default="../../datasets/dev_pos.tsv",
                        help='validation file for ner')
    parser.add_argument('--dep_val_file', type=str, default="../../datasets/tr_imst-ud-train.conllu",
                        help='validation file for dep')
    parser.add_argument('--ner_test_file', type=str, default="../../datasets/test_pos.tsv", help='test file for ner')
    parser.add_argument('--dep_test_file', type=str, default="../../datasets/tr_imst-ud-test.conllu",
                        help='test file for dep')
    parser.add_argument('--ner_output_file', type=str, default="joint_ner_out.txt",
                        help='Output file for named entity recognition')
    parser.add_argument('--dep_output_file', type=str, default="joint_dep_out.txt",
                        help='Output file for dependency parsing')
    parser.add_argument('--conll_file_name', type=str, default='conll_ner_output',
                        help='Output file name in conll bio format')
    parser.add_argument('--config_file', type=str, default='config.json', help='Output file name in conll bio format')
    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--load_config', default=0, type=int)
    parser.add_argument('--lang', default='tr', type=str, help='Language', choices=['en', 'jp', 'tr', 'cs', 'fi', 'hu'])
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--lstm_hidden', type=int, default=400)
    parser.add_argument('--char_hidden_dim', type=int, default=200)
    parser.add_argument('--biaffine_hidden', type=int, default=400)
    parser.add_argument('--composite_deep_biaff_hidden_dim', type=int, default=100)
    parser.add_argument('--cap_dim', type=int, default=32)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--cap_types', type=int, default=6)
    parser.add_argument('--pos_dim', type=int, default=64)
    parser.add_argument('--dep_dim', type=int, default=128)
    parser.add_argument('--ner_dim', type=int, default=128)
    parser.add_argument('--transformed_dim', type=int, default=125)
    parser.add_argument('--word_embed_type', default='bert', choices=['bert', 'random_init', 'fastext', 'word2vec'],
                        help='Word embedding type to be used')

    parser.add_argument('--fix_embed', default=False, action='store_true', help='Word embedding type to be used')
    parser.add_argument('--word_only', default=False, action='store_true',
                        help='If true, pos/cap embeddings are not used')

    parser.add_argument('--lstm_layers', type=int, default=3)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=-1)

    parser.add_argument('--word_drop', type=float, default=0.3)
    parser.add_argument('--embed_drop', type=float, default=0.3)
    parser.add_argument('--lstm_drop', type=float, default=0.3)
    parser.add_argument('--crf_drop', type=float, default=0.3)
    parser.add_argument('--parser_drop', type=float, default=0.3)

    parser.add_argument('--rec_dropout', type=float, default=0.2, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Char Recurrent dropout")
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    parser.add_argument('--no_pretrain', dest='pretrain', action='store_false', help="Turn off pretrained embeddings.")
    parser.add_argument('--no_linearization', dest='linearization', action='store_false',
                        help="Turn off linearization term.")
    parser.add_argument('--no_distance', dest='distance', action='store_false', help="Turn off distance term.")

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--ner_lr', type=float, default=0.0015, help='Learning rate for ner lstm')
    parser.add_argument('--embed_lr', type=float, default=0.015, help='Learning rate for embeddiing')
    parser.add_argument('--dep_lr', type=float, default=0.0015, help='Learning rate dependency lstm')
    parser.add_argument('--lr_decay', type=float, default=0.6, help='Learning rate decay')
    parser.add_argument('--min_lr', type=float, default=2e-6, help='minimum value for learning rate')
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--multiple', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--dep_warmup', type=int, default=-1)
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--ner_warmup', type=int, default=-1)
    parser.add_argument('--eval_interval', type=int, default=None)
    parser.add_argument('--max_steps_before_stop', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--max_depgrad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='../model_save_dir', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default='best_joint_model.pkh', help="File name to save the model")
    parser.add_argument('--save_ner_name', type=str, default='best_ner_model.pkh', help="File name to save the model")
    parser.add_argument('--save_dep_name', type=str, default='best_dep_model.pkh', help="File name to save the model")
    parser.add_argument('--load_model', type=int, default=0, help='Binary for loading previous model')
    parser.add_argument('--load_path', type=str, default='best_joint_model.pkh', help="File name to load the model")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    parser.add_argument('--inner', type=int, default=1,
                        help=' Choose whether to give a hidden output or prediction embedding')
    parser.add_argument('--soft', type=int, default=1,
                        help=' Choose whether to give max prediction or weighted average prediction ')
    parser.add_argument('--relu', type=int, default=1, help=' Choose whether to apply relu or not')
    parser.add_argument('--model_type', default='FLAT', help=' Choose model type to be trained')
    parser.add_argument('--hyper', type=int, default=0, help=' Hyperparam optimization mode')
    parser.add_argument('--max_evals', type=int, default=50, help=' Hyperparam optimization mode')

    parser.add_argument('--ner_only', type=int, default=0, help=' Choose whether to train a ner only model')
    parser.add_argument('--dep_only', type=int, default=0, help=' Choose whether to train a dep only model')
    parser.add_argument('--depner', type=int, default=0, help=' Hierarchical model type : dep helping ner')
    parser.add_argument('--nerdep', type=int, default=0, help=' Hierarchical model type : ner helping dep')
    args = parser.parse_args()

    vars(args)['device'] = device
    args = vars(args)
    if args['load_config'] == 1:
        args = read_config(args)
    return args


class JointModel(nn.Module):

    def __init__(self, args, tokenizer):
        super(JointModel, self).__init__()
        self.args = args
        # base model for generating bert output
        self.base_model = BaseModel(self.args, tokenizer)

        self.args['lstm_input_size'] = self.base_model.lstm_input_size

        self.depparser = JointParser(self.args)

        self.nermodel = JointNer(self.args)


def load_bert_model(lang):
    model_name = model_name_dict[lang]
    if lang == "hu":
        model = BertForPreTraining.from_pretrained(model_name, from_tf=True)
        return model
    else:
        model = AutoModel.from_pretrained(model_name)
        return model


class BertModelforJoint(nn.Module):

    def __init__(self, lang):
        super(BertModelforJoint, self).__init__()
        self.model = self.load_bert_model(lang)
        self.lang = lang
        # base model for generating bert output

    def load_bert_model(self, lang):
        model_name = model_name_dict[lang]
        if lang == "hu":
            model = BertForPreTraining.from_pretrained(model_name, from_tf=True, output_hidden_states=True)
        else:
            model = BertForTokenClassification.from_pretrained(model_name)
            model.classifier = nn.Identity()
        return model

    def forward(self, input, attention_mask, **kwargs):
        """
            Output the logits of the last layer for each word...
        :param input:
        :return:
        """
        if self.lang == "hu":
            output = self.model(input, attention_mask)[2][-1]
        else:
            output = self.model(input, attention_mask)[0]
        return output


class BaseModel(nn.Module):

    def __init__(self, args, tokenizer):

        super(BaseModel, self).__init__()
        self.bert_tokenizer = tokenizer
        # self.bert_model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
        self.bert_model = BertModelforJoint(args["lang"])
        self.bert_model.model.resize_token_embeddings(len(tokenizer))
        self.w_dim = 768
        # self.vocab_size = args['vocab_size']
        self.args = args
        self.lang = args["lang"]

        self.embed_drop = args['embed_drop']
        self.lstm_drop = args['lstm_drop']
        self.weight_decay = self.args['weight_decay']
        if self.args['word_embed_type'] == "random_init":
            print("Whole vocab size {}".format(len(self.args['vocab'])))
            self.w_dim = word2vec_lens[self.lang]
            self.word_embeds = embedding_initializer(self.w_dim, len(self.args['vocab']))
            if self.args['fix_embed']:
                self.word_embeds.weight.requires_grad = False
                print("Fixing the pretrained embeddings ")

        if self.args['word_embed_type'] in ["fastext", 'word2vec']:
            print("Whole vocab size {}".format(len(self.args['vocab'])))
            self.w_dim = word2vec_lens[self.lang] if self.args['word_embed_type'] == "word2vec" else 300
            load_w2v = True if self.args['word_embed_type'] == 'word2vec' else False
            self.word_embeds = get_pretrained_word_embeddings(self.args['vocab'], self.args['lang'],
                                                              self.w_dim, self.args['wordvec_dir'],
                                                              load_w2v=load_w2v)

            print("Initialized word embeddings from {}".format(self.args['word_embed_type']))
            self.w_dim = len(self.word_embeds.weight[0])
            print("Embeddings fixed? {} ".format(self.args['fix_embed']))
            if self.args['fix_embed']:
                self.word_embeds.weight.requires_grad = False
            print("Requires grad {}".format(self.word_embeds.weight.requires_grad))
        # self.cap_embeds  = nn.Embedding(self.cap_types, self.cap_dim)
        # self.pos_embeds  = nn.Embedding(self.args['pos_vocab_size'], self.pos_dim)
        self.lstm_input_size = self.w_dim
        if not self.args['word_only']:
            self.cap_types = self.args['cap_types']
            self.cap_dim = args['cap_dim']
            self.pos_dim = self.args['pos_dim']
            self.cap_embeds = embedding_initializer(self.cap_dim, self.cap_types)
            self.pos_embeds = embedding_initializer(self.pos_dim, self.args['pos_vocab_size'])
            self.lstm_input_size = self.w_dim + self.cap_dim + self.pos_dim
            # self.bilstm  = nn.LSTM(self.lstm_input, self.lstm_hidden, bidirectional=True, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(self.lstm_drop)
        self.embed_dropout = nn.Dropout(self.embed_drop)

        if self.args['word_embed_type'] in ['random_init', 'fastext', 'word2vec']:
            self.embed_optimizer = optim.AdamW([
                {"params": self.cap_embeds.parameters()}, \
                {"params": self.word_embeds.parameters(),
                 'lr': self.args['embed_lr'] if self.args['word_embed_type'] == 'random_init' \
                     else 2e-3}, \
                {"params": self.pos_embeds.parameters()}], \
                lr=self.args['embed_lr'], betas=(0.9, self.args['beta2']), eps=1e-6)
        else:
            self.embed_optimizer = optim.AdamW([
                {"params": self.cap_embeds.parameters()}, \
                {"params": self.pos_embeds.parameters()}], \
                lr=self.args['embed_lr'], betas=(0.9, self.args['beta2']), eps=1e-6)

        param_optimizer = list(self.bert_model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        bert_optimizer = optim.AdamW(optimizer_grouped_parameters,
                                     lr=2e-5)
        self.bert_optimizer = bert_optimizer

    def _get_bert_batch_hidden(self, hiddens, bert2toks, layers=[-2, -3, -4]):
        # meanss = torch.mean(torch.stack([hiddens[i] for i in layers]), 0)
        meanss = hiddens  # just use hiddens without averaging
        batch_my_hiddens = []

        for means, bert2tok in zip(meanss, bert2toks):
            my_token_hids = []
            my_hiddens = []

            for i, b2t in enumerate(bert2tok):
                if i > 0 and b2t != bert2tok[i - 1]:
                    my_hiddens.append(torch.mean(torch.stack(my_token_hids), 0))
                    my_token_hids = [means[i + 1]]  ## we skip the CLS token
                else:
                    my_token_hids.append(means[i + 1])
            my_hiddens.append(torch.mean(torch.stack(my_token_hids), 0))
            sent_hiddens = torch.stack(my_hiddens)
            batch_my_hiddens.append(sent_hiddens)
        try:
            return torch.stack(batch_my_hiddens)
        except:
            print("Problem in batch")
            for x in batch_my_hiddens:
                print("Shape {} ".format(x.shape))

    def _get_bert_batch_hidden2(self, hiddens, bert2toks, layers=[-2, -3, -4]):
        # meanss = torch.mean(torch.stack([hiddens[i] for i in layers]), 0)
        meanss = hiddens
        batch_my_hiddens = []
        for means, bert2tok in zip(meanss, bert2toks):
            my_token_hids = []
            my_hiddens = []
            for i, b2t in enumerate(bert2tok):
                if i > 0 and b2t != bert2tok[i - 1]:
                    my_hiddens.append(torch.mean(torch.cat(my_token_hids), 0).view(1, -1))
                    my_token_hids.append(means[i + 1].view(1, -1))
            my_hiddens.append(torch.mean(torch.cat(my_token_hids), 0).view(1, -1))
            batch_my_hiddens.append(torch.cat(my_hiddens))
        return torch.stack(batch_my_hiddens)

    def get_word_embedding(self, word_embed_input, type="bert"):
        if type == "bert":
            batch_bert_ids, batch_seq_ids, bert2toks = word_embed_input
            bert_out = self.bert_model(batch_bert_ids, batch_seq_ids)
            # print(bert2toks)
            print("Bert shape: {}".format(bert_out.shape))
            bert_hiddens = self._get_bert_batch_hidden(bert_out, bert2toks)
            bert_hiddens = self.dropout(bert_hiddens)
            return bert_hiddens
        else:
            word_inds = word_embed_input
            word_embeds = self.word_embeds(word_inds)
            word_embeds = self.embed_dropout(word_embeds)
            # print("Word embeddings shape {}".format(word_embeds.shape))
            return word_embeds

    def forward(self, tok_inds, pos_ids, batch_bert_ids, batch_seq_ids, bert2toks, cap_inds, sent_lens):

        if self.args['word_embed_type'] == 'bert':
            word_embed = self.get_word_embedding([batch_bert_ids, batch_seq_ids, bert2toks],
                                                 type=self.args['word_embed_type'])
        if self.args['word_embed_type'] in ['fastext', 'random_init', 'word2vec']:
            # print("Before weights of fourth word{}".format(self.word_embeds.weight.data[tok_inds[0,3]]))
            word_embed = self.get_word_embedding(tok_inds, type=self.args['word_embed_type'])
            print("Word embedding output dim shape {}".format(word_embed.shape))
        # bert_out = self.bert_model(batch_bert_ids,batch_seq_ids)
        # bert_hiddens = self._get_bert_batch_hidden(bert_out[2],bert2toks)
        # bert_hiddens = self.dropout(bert_hiddens)
        if not self.args['word_only']:
            cap_embedding = self.embed_dropout(self.cap_embeds(cap_inds))
            pos_embedding = self.embed_dropout(self.pos_embeds(pos_ids))
            word_embed = torch.cat((word_embed, cap_embedding, pos_embedding), dim=2)

        # bilstms are separate for each task
        # padded = pack_padded_sequence(concat,sent_lens,batch_first=True)
        # lstm_out,_ = self.bilstm(padded)
        # unpacked, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # unpacked = self.dropout(unpacked)
        # print("Word representation output shape: {}".format(word_embed.shape))
        return word_embed


class JointTrainer:

    def hyperparamoptimize(self):
        def objective(args):
            logging.info("Parameters being tried!")
            logging.info(args)
            for arg in args:
                self.args[arg] = args[arg]
                print(arg, args[arg], self.args[arg])
            return -self.run_training()

        ner_space = {"ner_lr": hp.uniform('a', 0.001, 5e-4), "lr_decay": hp.uniform('b', 0.1, 0.9),
                     "lstm_drop": hp.uniform('c', 0.1, 0.5), 'lstm_hidden': 200 + hp.randint('d', 200)}
        space = ner_space
        if self.args['model_type'] == "DEP":
            dep_space = {"dep_lr": hp.uniform('a', 0.001, 5e-4), "lr_decay": hp.uniform('b', 0.1, 0.9),
                         "lstm_drop": hp.uniform('c', 0.3, 0.5), 'lstm_hidden': 200 + hp.randint('d', 200)}
            space = dep_space
        trials = Trials()
        best = fmin(objective, space, algo=tpe.suggest, max_evals=self.args['max_evals'], trials=trials)
        val = hyperopt.space_eval(space, best)
        results = []
        for trial in trials:
            r = -trial['result']['loss']
            results.append(r)
            logging.info("Obtained loss")
            logging.info(r)
            logging.info("Parameter config : ")
            logging.info(trial['misc']['vals'])
        self.plot_f1(results, "hyperparam", self.args['model_type'])
        logging.info("Results for the hyperoptimization {}".format(best))
        logging.info("Results for the hyperoptimization {}".format(val))
        return val

    def __init__(self, args):
        self.args = args
        # args2 = {'bert_dim' : 100,'pos_dim' : 12, 'pos_vocab_size': 23,'lstm_hidden' : 10,'device':device}
        # self.args  = {**self.args,**args2}
        # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        self.lang = self.args['lang']
        model_name = model_name_dict[self.lang]
        print(self.lang, " ", model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("BERT TOkenigezer")
        print(self.bert_tokenizer)
        self.bert_tokenizer.add_tokens(['[SOS]', '[EOS]', '[ROOT]', '[PAD]'])
        self.exp_name = "{}_{}_{}".format(self.args['model_type'], self.args['word_embed_type'], self.args['lang'])
        print("Experiment name {} ".format(self.exp_name))
        self.getdatasets()
        print("Ner dataset contains {} batches".format(len(self.nertrainreader)))
        print("Dep dataset contains {}  batches ".format(len(self.deptraindataset)))
        if self.args['eval_interval'] is None:
            if self.args['model_type'] not in ['DEP', 'NER']:
                self.args['eval_interval'] = min(len(self.nertrainreader), len(self.deptraindataset))
            elif self.args['model_type'] == 'DEP':
                self.args['eval_interval'] = len(self.deptraindataset)

            elif self.args['model_type'] == 'NER':
                self.args['eval_interval'] = len(self.nertrainreader)
            print("Eval interval is set to {} ".format(self.args['eval_interval']))
        self.args['ner_cats'] = len(self.nertrainreader.label_voc)
        print("Word vocabulary size  : {}".format(len(self.nertrainreader.word_voc)))
        self.args['vocab'] = self.whole_vocab.w2ind

        ## feature extraction
        self.device = self.args['device']
        print("Joint Trainer initialized on {}".format(self.device))

        self.update_funcs = {"DEPNER": self.depner_update, "NERDEP": self.nerdep_update, "FLAT": self.flat_update,
                             "DEP": self.dep_update_caller, "NER": self.ner_update_caller}

        self.best_global_ner_f1 = 0
        self.best_global_dep_f1 = 0

    def run_multiple(self):
        ner_results = []
        dep_results = []
        for i in range(self.args['repeat']):
            self.init_models()
            ner_f1, dep_las = self.train2()
            ner_results.append(ner_f1)
            dep_results.append(dep_las)
            logging.info("Results for repeat {}".format(i))
            logging.info("NER {}  DEP {} ".format(ner_f1, dep_las))
        logging.info("All results for ner ")
        logging.info(ner_results)
        logging.info("All results for dep")
        logging.info(dep_results)
        logging.info("Average results === NER : {}  DEP  : {}".format(sum(ner_results) / self.args['repeat'],
                                                                      sum(dep_results) / self.args['repeat']))

    def predict(self):
        assert (self.args['load_model'] == 1), 'Model must be loaded in predict mode'

        self.init_models()
        ner_pre, ner_rec, ner_f1 = self.ner_evaluate()
        dep_pre, dep_rec, dep_f1, uas_f1 = self.dep_evaluate()

    def run_training(self):
        self.init_models()
        ner_f1, dep_las = self.train2()
        return ner_f1 + dep_las

    def init_models(self):
        logging.info("Initializing the model  from start with the following configuration")
        # logging.info(self.args)

        self.jointmodel = JointModel(self.args, self.bert_tokenizer)

        if self.args['load_model'] == 1:
            load_path = self.args['load_path']
            # save_path = os.path.join(self.args['save_dir'],self.args['save_name'])
            logging.info("Model loaded %s" % load_path)
            print("Model will be loaded from {} ".format(load_path))
            self.jointmodel.load_state_dict(torch.load(load_path))

        self.jointmodel.base_model.pos_vocab = self.pos_vocab

        self.nertrainreader.pos_vocab = self.pos_vocab
        self.nervalreader.pos_vocab = self.pos_vocab

        self.jointmodel.depparser.vocabs = self.deptraindataset.vocabs
        self.jointmodel.to(self.device)
        self.nerevaluator = Evaluate("NER")

    def plot_f1(self, f1_array, model_name="FLAT", task="NER"):
        today = date.today()
        plt.figure(model_name + task)
        plt.plot([i + 1 for i in range(len(f1_array))], f1_array, label=task)
        plt.legend()
        plt.title("{} F1-Score for the {} model on the development set".format(task, model_name))
        plt.savefig(os.path.join(self.args['save_dir'],
                                 "{}_{}_{}_{}devf1.png".format(model_name, task, today.day, today.month)))
        plt.show()

    def lr_updater(self, ner_patience, dep_patience):
        t = 0
        ams = 0
        if ner_patience > self.args['lr_patience'] * 5:
            # self.jointmodel.nermodel.ner_optimizer['amsgrad']=True
            ams = 1
        if ner_patience > self.args['lr_patience']:
            self.update_lr(task="NER")
            t = 1
            ner_patience = 0
        if dep_patience > self.args['lr_patience'] * 5:
            # self.jointmodel.depparser.optimizer['amsgrad']=True
            ams = 1
        if dep_patience > self.args['lr_patience']:
            self.update_lr(task="DEP")
            t = 1
            dep_patience = 0
        # if ams!=0:
        # self.jointmodel.base_model.embed_optimizer['amsgrad']=True
        if t != 0:
            self.update_base_lr()
        return ner_patience, dep_patience

    def update_base_lr(self):
        for param_group in self.jointmodel.base_model.embed_optimizer.param_groups:
            param_group['lr'] = max(self.args['min_lr'], param_group['lr'] * self.args['lr_decay'])

    def update_lr(self, task="NER"):
        if task == "NER":
            for param_group in self.jointmodel.nermodel.ner_optimizer.param_groups:
                param_group['lr'] = max(self.args['min_lr'], param_group['lr'] * self.args['lr_decay'])
        else:
            for param_group in self.jointmodel.depparser.optimizer.param_groups:
                param_group['lr'] = max(self.args['min_lr'], param_group['lr'] * self.args['lr_decay'])

    ## for finnish-hungarian-czech datasets dep and ner are together dayoo!!!!
    def get_joint_dataset():
        self.nerdeptraindataset = DepDataset(self.args['dep_train_file'], batch_size=self.args['batch_size'],
                                             tokenizer=self.bert_tokenizer)
        if self.args['mode'] == 'predict':
            self.nerdepvaldataset = DepDataset(self.args['dep_test_file'], batch_size=self.args['batch_size'],
                                               vocabs=self.deptraindataset.vocabs, for_eval=True,
                                               tokenizer=self.bert_tokenizer)
        else:
            self.nerdepvaldataset = DepDataset(self.args['dep_val_file'], batch_size=self.args['batch_size'],
                                               vocabs=self.deptraindataset.vocabs, for_eval=True,
                                               tokenizer=self.bert_tokenizer)

    def getdatasets(self):
        lang = lang_abs[self.args['lang']]
        dep_train_name = os.path.join(self.args['data_folder'], "dep_{}_train.conllu".format(lang))
        dep_dev_name = os.path.join(self.args['data_folder'], "dep_{}_dev.conllu".format(lang))
        dep_test_name = os.path.join(self.args['data_folder'], "dep_{}_test.conllu".format(lang))
        ner_train_name = os.path.join(self.args['data_folder'], "myner_{}-train.txt".format(lang))
        ner_dev_name = os.path.join(self.args['data_folder'], "myner_{}-dev.txt".format(lang))
        ner_test_name = os.path.join(self.args['data_folder'], "myner_{}-test.txt".format(lang))

        self.nertrainreader = DataReader(ner_train_name, "NER", batch_size=self.args['batch_size'],
                                         tokenizer=self.bert_tokenizer)
        self.deptraindataset = DepDataset(dep_train_name, batch_size=self.args['batch_size'],
                                          tokenizer=self.bert_tokenizer)
        if self.args['mode'] == 'predict':
            self.nervalreader = DataReader(ner_test_name, "NER", batch_size=self.args['batch_size'],
                                           tokenizer=self.bert_tokenizer)
            self.depvaldataset = DepDataset(dep_test_name, batch_size=self.args['batch_size'],
                                            vocabs=self.deptraindataset.vocabs, for_eval=True,
                                            tokenizer=self.bert_tokenizer)
        else:
            self.nervalreader = DataReader(ner_dev_name, "NER", batch_size=self.args['batch_size'],
                                           tokenizer=self.bert_tokenizer)
            self.depvaldataset = DepDataset(dep_dev_name, batch_size=self.args['batch_size'],
                                            vocabs=self.deptraindataset.vocabs, for_eval=True,
                                            tokenizer=self.bert_tokenizer)
        # self.nervalreader.label_voc = self.nertrainreader.label_voc
        diff = set(self.nertrainreader.label_voc.w2ind) - set(self.nervalreader.label_voc.w2ind)
        diff = set(self.nervalreader.label_voc.w2ind) - set(self.nertrainreader.label_voc.w2ind)
        print(self.nervalreader.label_voc.w2ind)
        print(self.nertrainreader.label_voc.w2ind)
        print("DEP REL VOCABS")
        print(self.deptraindataset.vocabs['dep_vocab'].w2ind)
        print(self.depvaldataset.vocabs['dep_vocab'].w2ind)

        def merge_vocabs(voc1, voc2):
            for v in voc2.w2ind:
                if v not in voc1.w2ind:
                    voc1.w2ind[v] = len(voc1.w2ind)
            return voc1

        print('Before merging')
        print("Dependency pos tags ")
        print(self.deptraindataset.vocabs['pos_vocab'].w2ind)
        print("NER pos tags")
        print(self.nertrainreader.pos_voc.w2ind)

        merged_pos = merge_vocabs(self.nertrainreader.pos_voc, self.deptraindataset.vocabs['pos_vocab'])
        merged_tok = merge_vocabs(self.nertrainreader.word_voc, self.deptraindataset.vocabs['tok_vocab'])
        self.nertrainreader.word_voc = merged_tok
        self.deptraindataset.vocabs['tok_vocab'] = merged_tok
        self.nertrainreader.pos_voc = merged_pos
        self.deptraindataset.vocabs['pos_vocab'] = merged_pos
        self.whole_vocab = merged_tok
        print("After merging")
        print("Dependency pos tags ")
        print(self.deptraindataset.vocabs['pos_vocab'].w2ind)
        print("NER pos tags")
        print(self.nertrainreader.pos_voc.w2ind)
        print("NER vocab size {} ".format(len(self.nertrainreader.word_voc.w2ind)))
        print("DEP vocab size {} ".format(len(self.deptraindataset.vocabs['tok_vocab'].w2ind)))

        self.depvaldataset.vocabs = self.deptraindataset.vocabs
        self.nervalreader.word_voc = merged_tok
        self.nervalreader.label_voc = self.nertrainreader.label_voc
        self.nervalreader.pos_voc = self.nertrainreader.pos_voc
        self.nervalreader.num_cats = self.nertrainreader.num_cats
        self.args['vocab_size'] = len(self.nertrainreader.word_voc)
        self.args['pos_vocab_size'] = len(self.deptraindataset.vocabs['pos_vocab'])
        self.deptraindataset.num_pos = len(self.deptraindataset.vocabs['pos_vocab'])
        self.pos_vocab = self.deptraindataset.vocabs['pos_vocab']
        self.args['dep_cats'] = len(self.deptraindataset.vocabs['dep_vocab'])
        assert self.args['dep_cats'] == self.deptraindataset.num_rels, 'Dependency types do not match'
        assert self.args['pos_vocab_size'] == self.deptraindataset.num_pos, " Pos vocab size do not match "

        print("Training pos vocab : {}".format(self.nertrainreader.pos_voc.w2ind))
        print("Testing  pos vocab : {}".format(self.nervalreader.pos_voc.w2ind))

        print("Training vocab size : {}".format(len(self.nertrainreader.word_voc)))
        print("Test vocab size : {}".format(len(self.nervalreader.word_voc)))
        diff = set(self.nervalreader.word_voc.w2ind) - set(self.nertrainreader.word_voc.w2ind)
        print("{} words not in training set ".format(len(diff)))
        self.nervalreader.word_voc.w2ind = self.nertrainreader.word_voc.w2ind

    def forward(self, batch):
        tokens, bert_batch_after_padding, data = batch
        inputs = []
        for d in data:
            inputs.append(d.to(self.device))
        sent_lens, masks, tok_inds, ner_inds, pos_inds, _, _, \
        bert_batch_ids, bert_seq_ids, bert2toks, cap_inds = inputs
        features = self.jointmodel.base_model(tok_inds, pos_inds, bert_batch_ids, bert_seq_ids, bert2toks, cap_inds,
                                              sent_lens)
        return features

    def ner_loss(self, batch):
        sent_lens = batch[2][0].to(self.device)
        ner_inds = batch[2][3].to(self.device)
        bert_feats = self.forward(batch)
        crf_scores = self.jointmodel.nermodel(bert_feats, sent_lens)
        loss = self.jointmodel.nermodel.loss(crf_scores, ner_inds, sent_lens)
        loss = loss / torch.sum(sent_lens)
        return loss

    def dep_forward(self, dep_batch, task="DEP", training=True, word_embed_type='bert'):
        logging.info("Eval data ")
        ## get the output of ner layer
        if task == "NERDEP":
            bert_feats = self.forward(dep_batch)
            sent_lens = dep_batch[2][0].to(self.device)
            ner_inds = dep_batch[2][3].to(self.device)
            bert_feats = self.jointmodel.nermodel(bert_feats, sent_lens, task=task)
        else:
            bert_feats = self.forward(dep_batch)
        inputs = []
        data = dep_batch[2]
        for d in data:
            inputs.append(d.to(self.device))
        sent_lens, masks, _, _, pos, dep_inds, dep_rels, \
        _, _, _, _ = inputs
        tokens = dep_batch[0]
        # logging.info("Nedir tokenlar")
        # logging.info(dep_batch[0][-1])
        # logging.info("Nedir dep indler")
        # logging.info(dep_batch[2][5][-1])
        if task == "DEPNER":

            dep_out = self.jointmodel.depparser(masks, bert_feats, dep_inds, dep_rels, pos, sent_lens, training=True,
                                                task=task)
            return dep_out
        # logging.info("Head predictions ")
        # logging.info(preds[0][-1])
        # logging.info(dep_inds[-1])
        else:
            preds, loss, deprel_loss, depind_loss, acc, head_acc = self.jointmodel.depparser(masks, bert_feats,
                                                                                             dep_inds, dep_rels, pos,
                                                                                             sent_lens,
                                                                                             training=training,
                                                                                             task=task)
            # logging.info("Rel loss :  {} Index  loss : {} ".format(deprel_loss,depind_loss))
            return loss, preds, deprel_loss, depind_loss, acc, head_acc

    def ner_update(self, batch):

        self.jointmodel.base_model.embed_optimizer.zero_grad()
        self.jointmodel.base_model.bert_optimizer.zero_grad()
        self.jointmodel.nermodel.ner_optimizer.zero_grad()

        if self.args['model_type'] == 'DEPNER':
            self.jointmodel.depparser.optimizer.zero_grad()

        # batch = self.nertrainreader[i]
        sent_lens = batch[2][0].to(self.device)
        ner_inds = batch[2][3].to(self.device)

        ## if hierarchical dep embeddings are appended to the bert outputs
        if self.args['model_type'] == 'DEPNER':
            bert_feats = self.dep_forward(batch, task="DEPNER")

        else:
            bert_feats = self.forward(batch)
            # logging.info("Token lengths : {} sent_lens : {} bert_lens : {}".format(len(batch[0][-1]),sent_lens[-1],bert_feats.shape))

        # logging.info(batch[0][-1])
        crf_scores = self.jointmodel.nermodel(bert_feats, sent_lens)
        loss = self.jointmodel.nermodel.loss(crf_scores, ner_inds, sent_lens)
        # logging.info("Loss {}".format(loss/sum(sent_lens)))
        if loss > 1000 or loss < 0:
            logging.info("Problematic batch!!")
            logging.info(batch[0])
            logging.info(loss)
        # loss = loss/ sum(sent_lens-1)
        loss = loss / sum(sent_lens)
        loss.backward()
        pos_inds = batch[2][4]

        clip_grad_norm_(self.jointmodel.nermodel.parameters(), self.args['max_grad_norm'])
        clip_grad_norm_(self.jointmodel.base_model.parameters(), self.args['max_grad_norm'])
        if self.args['model_type'] == 'DEPNER':
            clip_grad_norm_(self.jointmodel.depparser.parameters(), self.args['max_grad_norm'])

        self.jointmodel.base_model.embed_optimizer.step()
        self.jointmodel.base_model.bert_optimizer.step()
        self.jointmodel.nermodel.ner_optimizer.step()
        if self.args['model_type'] == 'DEPNER':
            self.jointmodel.depparser.optimizer.step()
        return loss.item()

    def dep_update_caller(self, index, epoch):
        dep_batch = self.deptraindataset[index]
        dep_loss, deprel_loss, depind_loss, acc, uas = self.dep_update(dep_batch)
        return 0, dep_loss

    def dep_update(self, dep_batch, task="DEP"):

        self.jointmodel.base_model.embed_optimizer.zero_grad()
        self.jointmodel.base_model.bert_optimizer.zero_grad()
        self.jointmodel.depparser.optimizer.zero_grad()

        if task == "NERDEP":
            self.jointmodel.nermodel.ner_optimizer.zero_grad()
        dep_loss, _, deprel_loss, depind_loss, acc, head_acc = self.dep_forward(dep_batch, task=task)
        dep_loss.backward()

        clip_grad_norm_(self.jointmodel.base_model.parameters(), self.args['max_depgrad_norm'])
        clip_grad_norm_(self.jointmodel.depparser.parameters(), self.args['max_depgrad_norm'])
        if task == "NERDEP":
            clip_grad_norm_(self.jointmodel.nermodel.parameters(), self.args['max_grad_norm'])

        self.jointmodel.base_model.embed_optimizer.step()
        self.jointmodel.base_model.bert_optimizer.step()
        self.jointmodel.depparser.optimizer.step()
        if task == "NERDEP":
            self.jointmodel.nermodel.ner_optimizer.step()

        return dep_loss.item(), deprel_loss.item(), depind_loss.item(), acc, head_acc

    def ner_update_caller(self, index, epoch):
        ner_batch = self.nertrainreader[index]
        ind = ner_batch[2][2].to(self.device)
        ner_loss = self.ner_update(ner_batch)
        return ner_loss, 0

    def nerdep_update(self, index, epoch):

        dep_loss = 0
        if epoch >= self.args['ner_warmup']:
            dep_batch = self.deptraindataset[index]
            dep_loss, deprel_loss, depind_loss, acc, uas = self.dep_update(dep_batch, task="NERDEP")

        ner_batch = self.nertrainreader[index]
        ner_loss = self.ner_update(ner_batch)

        return ner_loss, dep_loss

    def depner_update(self, index, epoch):

        ner_loss = 0
        if epoch >= self.args['dep_warmup']:
            ner_batch = self.nertrainreader[index]
            ner_loss = self.ner_update(ner_batch)

        dep_batch = self.deptraindataset[index]
        dep_loss, deprel_loss, depind_loss, acc, uas = self.dep_update(dep_batch)
        # logging.info("rel loss {}".format(deprel_loss))
        # logging.info("ind loss {}".format(depind_loss))
        return ner_loss, dep_loss

    def flat_update(self, index, epoch):

        ner_batch = self.nertrainreader[index]
        # logging.info("NER batch: ")
        # logging.info(ner_batch[0][-1])
        # logging.info(ner_batch[2][0][-1])
        # logging.info(ner_batch[2][3][-1])
        # logging.info("NER POS tag inds ")
        # logging.info(ner_batch[2][4][-1])
        ner_loss = self.ner_update(ner_batch)

        dep_batch = self.deptraindataset[index]
        # logging.info("DEP batch: ")
        # logging.info(dep_batch[0][-1])
        # logging.info(dep_batch[2][0][-1])
        # logging.info(dep_batch[2][5][-1])
        # logging.info("DEP POS tag inds ")
        # logging.info(dep_batch[2][4][-1])
        dep_loss, deprel_loss, depind_loss, acc, uas = self.dep_update(dep_batch)

        return ner_loss, dep_loss

    def train2(self):
        logging.info("Training on {} ".format(self.args['device']))

        logging.info("Dependency pos vocab : {} ".format(self.deptraindataset.vocabs['pos_vocab'].w2ind))
        logging.info("Dependency dep vocab : {} ".format(self.deptraindataset.vocabs['dep_vocab'].w2ind))
        epoch = self.args['max_steps'] // self.args['eval_interval'] if self.args['epochs'] is None else self.args[
            'epochs']
        self.jointmodel.train()
        experiment_log_name = "experiment_log_" + self.args['lang'] + "_" + self.args['word_embed_type'] + ".json"
        experiment_log = {"ner_f1": [],
                          "dep_f1": [],
                          "dep_uas_f1": [],
                          "ner_loss": [],
                          "dep_loss": []}

        save_ner_name = self.args['lang'] + "_" + self.args['word_embed_type'] + "_" + self.args['save_ner_name']
        save_name = self.args['lang'] + "_" + self.args['word_embed_type'] + "_" + self.args['save_name']
        save_dep_name = self.args['lang'] + "_" + self.args['word_embed_type'] + "_" + self.args['save_dep_name']
        save_ner_name = self.exp_name + "_" + self.args['save_ner_name']
        save__name = self.exp_name + "_" + self.args['save_name']
        save_dep_name = self.exp_name + "_" + self.args['save_dep_name']
        best_ner_f1 = 0
        best_dep_f1 = 0
        best_uas_f1 = 0
        best_model_nerpre = 0
        best_model_nerrec = 0
        best_ner_epoch = 0
        best_dep_epoch = 0
        ner_patience = 0
        dep_patience = 0
        dep_val_f1 = []
        ner_val_f1 = []
        logging.info("Pos tag vocab for dependency dataset")
        logging.info(self.deptraindataset.vocabs['pos_vocab'].w2ind)
        logging.info("Pos tag vocab for named entity dataset")
        logging.info(self.nertrainreader.pos_vocab.w2ind)
        logging.info("Dep vocab for dependency dataset")
        logging.info(self.deptraindataset.vocabs['dep_vocab'].w2ind)
        logging.info("NER label vocab")
        logging.info(self.nertrainreader.label_voc.w2ind)
        ## get the updating function depending on the model type
        model_func = self.update_funcs[self.args['model_type']]
        model_type = self.args['model_type']
        logging.info("Training on : {} type ".format(self.args['word_embed_type']))
        if self.args['word_embed_type'] in ['random_init', 'fastext']:
            print("Word embed type {} ".format(self.args['word_embed_type']))
            for param_group in self.jointmodel.base_model.embed_optimizer.param_groups:
                print("Word embedding learning rates : {}".format(param_group['lr']))

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
                ner_loss, dep_loss = model_func(i, e)
                ner_losses += ner_loss
                dep_losses += dep_loss
                if i % 10 == 9 and (not self.args['hyper'] == 1):
                    logging.info("Average dep loss {} ".format(dep_losses / (i + 1)))
                    logging.info("Average ner loss {} ".format(ner_losses / (i + 1)))

            logging.info("Results for epoch : {}".format(e + 1))
            self.jointmodel.eval()
            dep_f1 = 0
            uas_f1 = 0

            if (model_type != "NER" and model_type != "NERDEP") or (
                    model_type == "NERDEP" and e > self.args['ner_warmup']):
                dep_pre, dep_rec, dep_f1, uas_f1 = self.dep_evaluate()
            ner_f1 = 0

            if (model_type == "DEPNER" and e >= self.args['dep_warmup']) or (
                    model_type != "DEPNER" and model_type != "DEP"):
                ner_pre, ner_rec, ner_f1 = self.ner_evaluate()
                logging.info("NER Results -- pre : {}  rec : {} f1 : {}  ".format(ner_pre, ner_rec, ner_f1))

            logging.info("Dependency Results -- LAS f1 : {}  UAS f1 :  {} ".format(dep_f1, uas_f1))
            ner_val_f1.append(ner_f1)
            dep_val_f1.append(dep_f1)

            logging.info("Losses -- train {}  dependency {} ner {} ".format(train_loss, dep_losses, ner_losses))
            if uas_f1 > best_uas_f1:
                best_uas_f1 = uas_f1
            if model_type != "NER":
                if dep_f1 > best_dep_f1:
                    if dep_f1 > self.best_global_dep_f1:
                        print("Saving best dep model to {} ".format(save_dep_name))
                        self.save_model(save_dep_name)
                        self.best_global_dep_f1 = dep_f1
                    best_dep_f1 = dep_f1
                    best_dep_epoch = e + 1
                else:
                    logging.info(
                        "Best LAS of {} achieved at {} with {} UAS".format(best_dep_f1, best_dep_epoch, best_uas_f1))
                    dep_patience += 1
                    if (e + 1) - best_dep_epoch > self.args['early_stop']:
                        logging.info("Early stopping!!")
                        break
            if model_type != "DEP":
                if ner_f1 > best_ner_f1:
                    best_ner_epoch = e + 1
                    if ner_f1 > self.best_global_ner_f1:
                        print("Saving best ner model to {} ".format(save_ner_name))
                        self.save_model(save_ner_name)
                        self.best_global_ner_f1 = ner_f1
                    best_ner_f1 = ner_f1
                    best_model_nerpre = ner_pre
                    best_model_nerrec = ner_rec
                else:
                    logging.info("Best F-1 for NER  of {} achieved at {}".format(best_ner_f1, best_ner_epoch))
                    ner_patience += 1
                    if (e + 1) - best_ner_epoch > self.args['early_stop']:
                        logging.info("Early stopping!!")
                        break
            ner_patience, dep_patience = self.lr_updater(ner_patience, dep_patience)

            if ner_f1 > best_ner_f1 and dep_f1 > best_dep_f1:
                self.save_model(save_name)
            self.jointmodel.train()

            experiment_log["ner_f1"].append(ner_f1)
            experiment_log["dep_f1"].append(dep_f1)
            experiment_log["dep_uas_f1"].append(uas_f1)
            experiment_log["ner_loss"].append(ner_loss)
            experiment_log["dep_loss"].append(dep_loss)

        logging.info("Best results : ")
        logging.info("NER : {}  LAS : {} UAS : {}".format(best_ner_f1, best_dep_f1, best_uas_f1))
        logging.info(
            "Best NER results : pre : {} rec : {}  f1 : {} ".format(best_model_nerpre, best_model_nerrec, best_ner_f1))
        # self.plot_f1(ner_val_f1, self.args['model_type'], "NER")
        # self.plot_f1(dep_val_f1, self.args['model_type'], "DEP")
        logging.info("NER val f1s ")
        logging.info(ner_val_f1)
        logging.info("DEP val f1s ")
        logging.info(dep_val_f1)
        o_f = self.args['ner_result_out_file']
        with open(o_f, "a") as o:
            o.write("NER Results on {} embed_type : {} fixed : {} \n pre : {} rec : {} f1 : {}\n".format(
                self.args['ner_val_file'], self.args['word_embed_type'], self.args['fix_embed'], best_model_nerpre,
                best_model_nerrec, best_ner_f1))

        logging.info("Experiment log")
        logging.info(experiment_log)
        with open(experiment_log_name, "w") as o:
            json.dump(experiment_log, o)
        return best_ner_f1, best_dep_f1

    def train(self):
        logging.info("Training on {} ".format(self.args['device']))
        logging.info("Dependency pos vocab : {} ".format(self.deptraindataset.vocabs['pos_vocab'].w2ind))
        logging.info("Dependency dep vocab : {} ".format(self.deptraindataset.vocabs['dep_vocab'].w2ind))
        epoch = self.args['max_steps'] // self.args['eval_interval']
        self.jointmodel.train()
        best_ner_f1 = 0
        best_dep_f1 = 0
        best_uas_f1 = 0
        best_ner_epoch = 0
        best_dep_epoch = 0
        dep_val_loss = []
        ner_val_loss = []
        if self.args['load_model'] == 1:
            save_path = os.path.join(self.args['save_dir'], self.args['save_name'])
            logging.info("Model loaded %s" % save_path)
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

                if not self.args['dep_only'] and e >= self.args['dep_warmup']:
                    ner_batch = self.nertrainreader[i]
                    # logging.info(ner_batch[0])
                    ner_loss = self.ner_update(ner_batch)
                    ner_losses += ner_loss
                    train_loss += ner_loss

                if not self.args['ner_only']:
                    dep_batch = self.deptraindataset[i]
                    dep_loss, deprel_loss, depind_loss, acc, uas = self.dep_update(dep_batch)
                    dep_losses += dep_loss
                    train_loss += dep_loss
                    deprel_losses += deprel_loss
                    depind_losses += depind_loss
                    deprel_acc += acc
                    uas_epoch += uas
            logging.info("Results for epoch : {}".format(e + 1))
            self.jointmodel.eval()
            dep_f1 = 0
            uas_f1 = 0
            if not self.args['ner_only']:
                dep_pre, dep_rec, dep_f1, uas_f1 = self.dep_evaluate()
            ner_f1 = 0
            logging.info("Losses -- train {}  dependency {} ner {} ".format(train_loss, dep_losses, ner_losses))
            if e >= self.args['dep_warmup'] and not self.args['dep_only']:
                ner_pre, ner_rec, ner_f1 = self.ner_evaluate()
                logging.info("NER Results -- f1 : {} ".format(ner_f1))

            logging.info("Dependency Results -- LAS f1 : {}  UAS f1 :  {} ".format(dep_f1, uas_f1))

            if uas_f1 > best_uas_f1:
                best_uas_f1 = uas_f1

            if dep_f1 > best_dep_f1:
                self.save_model(self.args['save_dep_name'])
                best_dep_f1 = dep_f1
                best_dep_epoch = e
            else:
                logging.info("Best LAS of {} achieved at {}".format(best_dep_f1, best_dep_epoch))
                for param_group in self.jointmodel.depparser.optimizer.param_groups:
                    param_group['lr'] *= self.args['lr_decay']

                for param_group in self.jointmodel.base_model.embed_optimizer.param_groups:
                    param_group['lr'] *= self.args['lr_decay']

            if ner_f1 > best_ner_f1:
                best_ner_epoch = e
                self.save_model(self.args['save_ner_name'])
                best_ner_f1 = ner_f1

            else:
                logging.info("Best F-1 for NER  of {} achieved at {}".format(best_ner_f1, best_ner_epoch))
                for param_group in self.jointmodel.nermodel.ner_optimizer.param_groups:
                    param_group['lr'] *= self.args['lr_decay']
                for param_group in self.jointmodel.base_model.embed_optimizer.param_groups:
                    param_group['lr'] *= self.args['lr_decay']

            if ner_f1 > best_ner_f1 and dep_f1 > best_dep_f1:
                self.save_model(self.args['save_name'])
            self.jointmodel.train()

        logging.info("Best results : ")
        logging.info("NER : {}  LAS : {} UAS : {}".format(best_ner_f1, best_dep_f1, best_uas_f1))

    def save_model(self, save_name, weights=True):
        save_name = os.path.join(self.args['save_dir'], save_name)
        if weights:
            logging.info("Saving best model to {}".format(save_name))
            torch.save(self.jointmodel.state_dict(), save_name)
        config_path = os.path.join(self.args['save_dir'], self.args['config_file'])
        arg = copy.deepcopy(self.args)
        del arg['device']
        with open(config_path, 'w') as outfile:
            json.dump(arg, outfile)

    def dep_evaluate(self):

        logging.info("Evaluating dependency performance on {}".format(self.depvaldataset.file_name))
        self.jointmodel.eval()
        dataset = self.depvaldataset
        orig_idx = self.depvaldataset.orig_idx
        data = []

        field_names = ["word", "head", "deprel"]
        gold_file = dataset.file_name
        pred_file = "pred_{}_{}_{}_".format(self.args['model_type'], self.args['word_embed_type'], self.args['lang']) + \
                    gold_file.split("/")[-1]
        start_id = orig_idx[0]
        # for x in tqdm(range(len(self.depvaldataset)),desc = "Evaluation"):
        rel_accs = 0
        head_accs = 0
        self.depvaldataset.for_eval = True
        for x in tqdm(range(len(self.depvaldataset)), desc="Evaluation"):
            # batch = dataset[x]
            batch = self.depvaldataset[x]
            sent_lens = batch[2][0]
            tokens = batch[0]
            if self.args['model_type'] == "NERDEP":
                loss, preds, _, _, rel_acc, head_acc = self.dep_forward(batch, training=False,
                                                                        task=self.args['model_type'])
            else:
                loss, preds, _, _, rel_acc, head_acc = self.dep_forward(batch, training=False, task="DEP")
            rel_accs += rel_acc
            head_accs += head_acc
            heads, dep_rels, output = self.jointmodel.depparser.decode(preds[0], preds[1], sent_lens, verbose=True)
            for outs, sent, l in zip(output, tokens, sent_lens):
                new_sent = []
                assert len(sent[1:l]) == len(outs), "Sizes do not match"
                for pred, tok in zip(outs, sent[1:l]):
                    new_sent.append([tok] + pred)
                data.append(new_sent)
        # print(orig_idx)
        # print(len(data))
        head_accs /= len(self.depvaldataset)
        rel_accs /= len(self.depvaldataset)
        data = unsort_dataset(data, orig_idx)
        pred_file = os.path.join(self.args['save_dir'], pred_file)
        conll_writer(pred_file, data, field_names, task_name="dep")
        print("Predictions can be observed from {}".format(pred_file))
        p, r, f1, uas_f1 = score(pred_file, gold_file, verbose=False)
        # p,r, f1 = 0,0,0
        logging.info("LAS F1 {}  ====    UAS F1 {}".format(f1 * 100, uas_f1 * 100))
        print("Dependency results")
        print("LAS F1 {}  ====    UAS F1 {}".format(f1 * 100, uas_f1 * 100))
        # self.parser.train()

        return p, r, f1 * 100, uas_f1 * 100

    def ner_evaluate(self):
        self.jointmodel.eval()
        sents = []
        preds = []
        truths = []
        ## for each batch
        dataset = self.nervalreader
        dataset.for_eval = True
        all_lens = []
        for i in tqdm(range(len(dataset))):
            d = dataset[i]
            tokens = d[0]
            sent_lens = d[2][0]
            ner_inds = d[2][3][:, :]
            # print(ner_inds)
            with torch.no_grad():
                if self.args['model_type'] != "DEPNER":
                    bert_out = self.forward(d)
                else:
                    bert_out = self.dep_forward(d, task="DEPNER")
                crf_scores = self.jointmodel.nermodel(bert_out, sent_lens, train=False)

                paths, scores = self.jointmodel.nermodel.batch_viterbi_decode(crf_scores, sent_lens)
                for i in range(crf_scores.shape[0]):
                    truth = ner_inds[i].detach().cpu().numpy() // self.args['ner_cats']  ##converting 1d labels
                    sents.append(tokens[i])
                    preds.append(paths[i])
                    truths.append(truth)
                    all_lens.append(sent_lens[i].detach().cpu().numpy())

        content = generate_pred_content(sents, preds, truths, lens=all_lens, label_voc=self.nertrainreader.label_voc)
        orig_idx = dataset.orig_idx
        content = unsort_dataset(content, orig_idx)

        field_names = ["token", "truth", "ner_tag"]
        ner_out_name = "{}_{}_{}_{}".format(self.args['model_type'], self.args['word_embed_type'], self.args['lang'],
                                            self.args['ner_output_file'])
        out_file = os.path.join(self.args['save_dir'], ner_out_name)
        print("Ner output will be written to {}".format(out_file))
        conll_writer(out_file, content, field_names, "ner")
        conll_ner_name = "{}_{}_{}_{}".format(self.args['model_type'], self.args['word_embed_type'], self.args['lang'],
                                              self.args['conll_file_name'])
        conll_file = os.path.join(self.args['save_dir'], conll_ner_name)
        # convert2IOB2new(out_file,conll_file)
        prec, rec, f1 = 0, 0, 0
        try:
            prec, rec, f1 = evaluate_conll_file(open(out_file, encoding='utf-8').readlines())
        except:
            f1 = 0
        # logging.info("{} {} {} ".format(prec,rec,f1))
        my_pre, my_rec, my_f1 = self.nerevaluator.conll_eval(out_file)
        logging.info("My values ignoring the boundaries.\npre: {}  rec: {}  f1: {} ".format(my_pre, my_rec, my_f1))
        logging.info("NER f1 : {} ".format(f1))
        # self.model.train()
        return prec, rec, f1


def main(args):
    jointtrainer = JointTrainer(args)

    if args['hyper'] == 1:
        best_params = jointtrainer.hyperparamoptimize()
        print(best_params)
        logging.info(best_params)
    else:
        if args['mode'] == 'predict':
            print("Running in prediction mode ")
            jointtrainer.predict()
        else:
            if args['multiple'] == 1:
                jointtrainer.run_multiple()
            else:
                jointtrainer.init_models()
                score = jointtrainer.train2()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if not os.path.isdir(args['save_dir']):
        os.mkdir(args['save_dir'])
    log_path = os.path.join(args['save_dir'], args['log_file'])
    logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler(log_path, 'w', 'utf-8')],
                        format='%(levelname)s - %(message)s')
    main(args)
