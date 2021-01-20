import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import numpy as np
import time
import os
import copy
import fasttext
import fasttext.util
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel, BertForPreTraining, BertForTokenClassification


def embedding_initializer(dim, num_labels):
    embed = nn.Embedding(num_labels, dim)
    nn.init.uniform_(embed.weight, -np.sqrt(6 / (dim + num_labels)), np.sqrt(6 / (dim + num_labels)))
    return embed


RANDOM_DIM = 300

lang_abs = {"fi": "finnish", "en": "english", "hu": "hungarian", "cs": "czech", "tr": "turkish", "jp": "japanese"}

model_name_dict = {"jp": "cl-tohoku/bert-base-japanese",
                   "tr": "dbmdz/bert-base-turkish-cased",
                   "hu": "/home/aakdemir/bert_models/hubert",
                   "fi": "TurkuNLP/bert-base-finnish-cased-v1",
                   "cs": "DeepPavlov/bert-base-bg-cs-pl-ru-cased",
                   "en": "bert-base-cased",
                   "mbert": "bert-base-multilingual-cased",
                   "bert_en": "bert-base-cased"}

word2vec_dict = {"jp": "../word_vecs/jp/jp.bin",
                 "tr": "../word_vecs/tr/tr.bin",
                 "hu": "../word_vecs/hu/hu.bin",
                 "en": "../word_vecs/en/en.txt",
                 "fi": "../word_vecs/fi/fi.bin",
                 "cs": "../word_vecs/cs/cs.txt"}

fasttext_dict = {"jp": "../word_vecs/jp/cc.jp.300.bin",
                 "tr": "../word_vecs/tr/cc.tr.300.bin",
                 "hu": "../word_vecs/hu/cc.hu.300.bin",
                 "en": "../word_vecs/en/cc.en.300.bin",
                 "fi": "../word_vecs/fi/cc.fi.300.bin",
                 "cs": "../word_vecs/cs/cc.cs.300.bin"}

word2vec_lens = {"tr": 200,
                 "hu": 300,
                 "fi": 300,
                 "en": 100,
                 "cs": 100,
                 "jp": 300}


def load_bert_model(lang):
    model_name = model_name_dict[lang]
    if lang == "hu":
        model = BertForPreTraining.from_pretrained(model_name, from_tf=True)
        return model
    else:
        model = AutoModel.from_pretrained(model_name)
        return model


def load_word2vec(lang):
    model_name = word2vec_dict[lang]
    if lang == "cs" or lang == "en":
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

    def get_vectors(self, file_name):
        with open(file_name, "r", encoding=encoding_map[self.lang]) as f:
            f = f.read().split("\n")
            wv = {}
            my_len = 0
            c = 0
            for l in f:  # s
                s = l.split(" ")
                if len(s) < 2:
                    continue
                w = s[0]
                v = s[1:]
                vec = [float(v_) for v_ in v if len(v_) > 1]
                if len(vec) < 10:
                    continue  # skip not a proper vector
                wv[w] = vec
                length = len(vec)
                if length > 1:
                    my_len = length
        vocab, wv, length = wv.keys(), MyDict(wv), my_len
        return vocab, wv, length


class MyBertModel(nn.Module):

    def __init__(self, lang):
        super(MyBertModel, self).__init__()
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


class SequenceClassifier(nn.Module):
    def __init__(self, lang, word_vocab, model_type, num_cats, device):
        super(SequenceClassifier, self).__init__()
        self.lang = lang
        self.device = device
        self.word_vocab = word_vocab
        self.vocab_size = len(word_vocab)
        self.hidden_dim = 128
        self.bidirectional = True
        self.model_type = model_type

        self.num_cat = num_cats
        self.init_base_model()

        self.classifier_input_dim = 2 * self.hidden_dim if self.bidirectional else self.hidden_dim
        if "bert" in model_type:
            self.classifier_input_dim = self.vector_dim
        self.classifier = nn.Linear(self.classifier_input_dim, num_cats)
        self.classifier_optimizer = optim.AdamW([{"params": self.classifier.parameters()}], \
                                                lr=0.0015, eps=1e-6)
        self.soft = nn.Softmax(dim=1)
        self.criterion = CrossEntropyLoss()
        # self.criterion = BCEWithLogitsLoss()
    def init_bert(self):
        if self.model_type in ["bert_en", "mbert"]:
            print("Initializing {} model".format(self.model_type))
            self.base_model = MyBertModel(self.model_type)
        else:
            print("Initializing lang specific bert model")
            self.base_model = MyBertModel(self.lang)
        self.vector_dim = 768
        bert_optimizer = optim.AdamW(self.base_model.parameters(),
                                     lr=2e-5)
        self.base_optimizer = bert_optimizer

    def init_word2vec(self):
        self.vector_dim = word2vec_lens[self.lang]
        dim = self.vector_dim
        gensim_model = load_word2vec(self.lang)
        embed = nn.Embedding(self.vocab_size, self.vector_dim)
        nn.init.uniform_(embed.weight, -np.sqrt(6 / (dim + self.vocab_size)), np.sqrt(6 / (dim + self.vocab_size)))
        w2ind = self.word_vocab.w2ind
        c = 0
        for word in list(w2ind.keys()):
            if word in gensim_model.wv.vocab:
                ind = w2ind[word]
                vec = gensim_model.wv[word]
                embed.weight.data[ind].copy_(torch.tensor(vec, requires_grad=True))
                c += 1
        print("Found {} out of {} words in word2vec for {} ".format(c, len(w2ind), self.lang))
        self.base_model = embed
        self.base_optimizer = optim.AdamW([{"params": self.base_model.parameters(),
                                            'lr': 2e-3}])
        self.init_lstm()

    def init_fastext(self):
        self.vector_dim = 300
        dim = self.vector_dim
        fastext_path = fasttext_dict[self.lang]
        ft = fasttext.load_model(fastext_path)
        embed = nn.Embedding(self.vocab_size, self.vector_dim)
        nn.init.uniform_(embed.weight, -np.sqrt(6 / (dim + self.vocab_size)), np.sqrt(6 / (dim + self.vocab_size)))
        w2ind = self.word_vocab.w2ind
        c = 0
        for word in w2ind:
            c += 1
            ind = w2ind[word]
            vec = ft.get_word_vector(word)
            ft_vec = torch.tensor(vec, requires_grad=True)
            embed.weight.data[ind].copy_(ft_vec)
        print("Found {} out of {} words in fastext for {} ".format(c, len(w2ind), self.lang))
        self.base_model = embed
        self.base_optimizer = optim.AdamW([{"params": self.base_model.parameters(),
                                            'lr': 2e-3}])
        self.init_lstm()

    def init_randominit(self):
        self.vector_dim = RANDOM_DIM
        dim = self.vector_dim
        embed = nn.Embedding(self.vocab_size, self.vector_dim)
        nn.init.uniform_(embed.weight, -np.sqrt(6 / (dim + self.vocab_size)), np.sqrt(6 / (dim + self.vocab_size)))
        self.base_model = embed
        self.base_optimizer = optim.AdamW([{"params": self.base_model.parameters(),
                                            'lr': 2e-3}])
        self.init_lstm()

    def init_lstm(self):
        self.lstm = nn.LSTM(self.vector_dim, self.hidden_dim, bidirectional=self.bidirectional)
        self.hidden_optimizer = optim.AdamW([{"params": self.lstm.parameters(),
                                              'lr': 2e-3}])

    def init_base_model(self):
        if self.model_type == "word2vec":
            self.init_word2vec()
        elif self.model_type == "random_init":
            self.init_randominit()
        elif self.model_type == "fastext":
            self.init_fastext()
        elif self.model_type in ["bert", "mbert", "bert_en"]:
            self.init_bert()

    def predict(self, input):
        tokens, tok_inds, bert_batch_after_padding, data = input
        bert_lens, masks, padded_tok_inds, labels, bert_batch_ids, bert_seq_ids = data
        labels.to(self.device)
        class_logits = self.forward(input)
        loss = self.loss(class_logits, labels)
        preds = torch.argmax(class_logits, dim=1)
        return preds, loss

    def get_embed_output(self, input):
        tokens, tok_inds, bert_batch_after_padding, data = input
        padded_tok_inds = data[2]
        padded_tok_inds = padded_tok_inds.to(self.device)
        embed_outs = self.base_model(padded_tok_inds)
        return embed_outs

    def get_bert_output(self, input):
        tokens, tok_inds, bert_batch_after_padding, data = input
        bert_lens, masks, padded_tok_inds, labels, bert_batch_ids, bert_seq_ids = data
        bert_batch_ids = bert_batch_ids.to(self.device)
        bert_seq_ids = bert_seq_ids.to(self.device)
        bert_out = self.base_model(bert_batch_ids, bert_seq_ids)
        return bert_out

    def zero_grad(self):
        self.base_optimizer.zero_grad()
        self.classifier_optimizer.zero_grad()
        if hasattr(self, "hidden_optimizer"):
            self.hidden_optimizer.zero_grad()

    def optimizer_step(self):
        self.base_optimizer.step()
        self.classifier_optimizer.step()
        if hasattr(self, "hidden_optimizer"):
            self.hidden_optimizer.step()

    def loss(self, class_logits, labels):
        labels = labels.to(self.device)
        probs = self.soft(class_logits)
        return self.criterion(probs, labels)

    def forward(self, input):
        if "bert" in self.model_type:
            bert_output = self.get_bert_output(input)
            hidden_out = bert_output[:, 0, :]
        else:
            embed_out = self.get_embed_output(input)
            hidden, _ = self.lstm(embed_out)
            hidden_out = hidden[:, 0, :]
            print("LSTM out shape: {}".format(hidden_out.shape))
        class_logits = self.classifier(hidden_out)
        return class_logits
