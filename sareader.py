"""
DataReader for Sentiment Analysis

take as input a json file and
"""

from collections import Counter
import glob
import pandas as pd
import torch
import numpy as np
import logging
from vocab import Vocab
from parser.utils import sort_dataset, unsort_dataset
from transformers import AutoTokenizer, AutoModel, BertForPreTraining, BertForTokenClassification
import json

PAD = "[PAD]"
PAD_IND = 0
CLS = "[CLS]"
CLS_IND = 1


def pad_batch(batch, pad=PAD):
    padded_batch = []
    sent_lens = []
    sent_lens = [len(x) for x in batch]
    max_len = max(sent_lens)
    for l, sent in zip(sent_lens, batch):
        padded_sent = sent + [pad] * (max_len - l)
        padded_batch.append(padded_sent)
    if not all([len(x) == len(padded_batch[0]) for x in padded_batch]):
        print("Problematic batch")
        logging.info("Problematic batch")
        logging.info(padded_batch)
    return padded_batch, sent_lens


def group_into_batch(dataset, batch_size):
    """

        Batch size is given in word length so that some batches do not contain
        too many examples!!!

        Do not naively batch by number of sentences!!

    """

    batched_dataset = []
    sentence_lens = []
    current_len = 0
    i = 0

    ## they are already in sorted order
    current = []
    max_len = 0
    for k, data in dataset:
        text = data["text"].split(" ")
        my_len = len(text)
        current.append(data)
        current_len += my_len
        if current_len > batch_size:
            batched_dataset.append(current)
            current = []
            current_len = 0
    if len(current) > 0:
        batched_dataset.append(current)
    return batched_dataset


def all_num(token):
    n = "0123456789."
    for c in token:
        if c not in n:
            return False
    return True


class SentReader():

    def __init__(self, file_path, batch_size=200, tokenizer=None, encoding="utf-8"):
        """
            Reader for sentence classification task!
        :param file_path:
        :param task_name:
        :param batch_size:  is in number of tokens???
        :param tokenizer:
        :param encoding:
        """
        self.encoding = encoding
        self.file_path = file_path
        self.batch_size = batch_size
        self.get_dataset()
        print("Dataset size : {}".format(len(self.dataset)))
        self.data_len = len(self.dataset)
        self.word2ind, self.l2ind, self.vocab_size = self.get_vocabs()
        self.word_vocab = Vocab(self.word2ind)
        self.label_vocab = Vocab(self.l2ind)

        self.batched_dataset = group_into_batch(self.dataset, batch_size=self.batch_size)
        self.bert_token_limit = 500
        self.for_eval = False
        self.num_cats = len(self.l2ind)
        if tokenizer is None:
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.bert_tokenizer.add_tokens(PAD, CLS)
        else:
            self.bert_tokenizer = tokenizer
        self.val_index = 0

    def get_ind2sent(self, sent):
        return " ".join([self.word2ind[w] for w in sent])

    def get_vocabs(self):
        word2ix = {PAD: PAD_IND, CLS: CLS_IND}
        l2ind = {}
        for k, data in self.dataset:
            for word in data["text"].split(" "):
                if word not in word2ix:
                    word2ix[word] = len(word2ix)
            label = data["label"]
            if label not in l2ind:
                l2ind[label] = len(l2ind)
        vocab_size = len(word2ix)
        return word2ix, l2ind, vocab_size

    def get_dataset(self):
        print("Reading from {} ".format(self.file_path))
        dataset = json.load(open(self.file_path, "r"))
        dataset = [(k, v) for k, v in dataset.items()]
        dataset.sort(key=lambda x: len(x[1]["text"].split(" ")),reverse=True)  # Sort by of tokens
        sentence_lengths = [len(x[1]["text"].split(" ")) for x in dataset]
        data = [x[1] for x in dataset]
        sentence_ids = [x[0] for x in dataset]
        self.dataset = dataset
        self.sentence_ids = sentence_ids

    ## compatible with getSent and for word embeddings
    def prepare_sent(self, sent, word2ix):
        idx = [word2ix[word[0]] for word in sent]
        return torch.tensor(idx, dtype=torch.long)

    def prepare_label(self, labs, l2ix):
        idx = [l2ix[lab] for lab in labs]
        return torch.tensor(idx, dtype=torch.long)

    def __len__(self):
        return len(self.batched_dataset)

    def __getitem__(self, idx):
        """
            Indexing for the DepDataset
            converts all the input into tensor before passing

            input is of form :
                word_ids  (Batch_size, Sentence_lengths)
        """
        if not self.for_eval:
            idx = np.random.randint(len(self.batched_dataset))
            idx = idx % len(self.batched_dataset)
        batch = self.batched_dataset[idx]  # array of data_dictionaries
        tok_inds = []
        labels = []
        tokens = []
        for data in batch:
            sentence = data["text"]
            toks = sentence.split(" ")
            label = data["label"]
            labels.append(self.label_vocab.map([label])[0]) # To enforce correct shape
            tok_inds.append(self.word_vocab.map(toks))
            tokens.append(toks)
        assert len(tok_inds) == len(tokens) == len(batch)

        padded_tok_inds, _ = pad_batch(tok_inds, PAD_IND)
        padded_tok_inds = torch.LongTensor(padded_tok_inds)
        labels = torch.LongTensor(labels)
        bert_batch_before_padding = []
        max_bert_len = 0
        i = 0
        for toks in tokens:
            sentence = " ".join(toks)
            btok = self.bert_tokenizer.tokenize(sentence)
            if len(btok) > self.bert_token_limit:
                print("{} is too long pruning to {}".format(len(btok),self.bert_token_limit))
                btok = btok[:self.bert_token_limit]
            btok = [CLS] + btok
            bert_batch_before_padding.append(btok)
        bert_batch_after_padding, bert_lens = pad_batch(bert_batch_before_padding)
        bert_batch_ids = torch.LongTensor([self.bert_tokenizer.convert_tokens_to_ids(sent) for \
                                           sent in bert_batch_after_padding])
        masks = torch.ones(bert_batch_ids.shape, dtype=torch.bool)
        for i, l in enumerate(bert_lens):
            masks[i, :l] = torch.tensor([0] * l, dtype=torch.bool)
        bert_seq_ids = torch.LongTensor([[1 for i in range(len(bert_batch_after_padding[0]))] \
                                         for j in range(len(bert_batch_after_padding))])
        data = torch.tensor(bert_lens), masks, padded_tok_inds, labels, bert_batch_ids, bert_seq_ids
        return tokens, tok_inds, bert_batch_after_padding, data


if __name__ == "__main__":
    data_path = '../datasets/sa_movie_turkish-test.json'
    reader = SentReader(data_path)
    tokens, tok_inds, bert_batch_after_padding, data = reader[0]
    print(data[4])
