from collections import Counter, defaultdict
import glob
import pandas as pd
import torch
import numpy as np
from pytorch_transformers import BertTokenizer
from parser import Parser, Vocab, PAD, PAD_IND, VOCAB_PREF, ROOT, ROOT_IND
from torch.utils.data import Dataset, DataLoader
import logging
import time

def bert2token(my_tokens, bert_tokens, bert_ind = 1):
    inds = []
    token_sum =""
    bert_ind = bert_ind
    for ind in range(len(my_tokens)):
        my_token = my_tokens[ind]

        while len(token_sum)!=len(my_token) and bert_ind<len(bert_tokens):
            token = bert_tokens[bert_ind]
            if token.startswith("#"):
                token_sum+=token[2:]
            else:
                token_sum+=token
            inds.append(ind)
            bert_ind+=1
        assert len(token_sum)==len(my_token), print(my_tokens)
        token_sum=""

    return inds, ind+1

def pad_trunc_batch(batch, max_len, pad = PAD, pad_ind = PAD_IND, bert = False,b2t=False):
    padded_batch = []
    sent_lens = []
    for sent in batch:
        sent_lens.append(len(sent))
        if len(sent)>=max_len:
            if bert:
                if b2t:
                    padded_batch.append(sent)
                else:
                    padded_batch.append(sent + ["[SEP]"])
            else:
                padded_batch.append(sent)
        else:
            l = len(sent)
            if not bert:
                index_len = len(sent[0])
            padded_sent = sent
            for i in range(max_len-l):
                if bert:
                    if b2t:
                        padded_sent.append(padded_sent[-1])
                    else:
                        padded_sent = padded_sent + [PAD]
                else:
                    padded_sent.append([PAD for x in range(index_len)]) ## PAD ALL FIELDS WITH [PAD]
            if bert:
                if not b2t:
                    padded_sent = padded_sent + ["[SEP]"]
            padded_batch.append(padded_sent)
    return padded_batch, sent_lens

def group_into_batch(dataset, batch_size):
    """
        Batch size is given word length so that some batches do not contain
        too many examples!!!

        Do not naively batch by number of sentences!!
    """
    dataset.sort(key = lambda x: len(x))
    dataset.reverse()
    batched_dataset = []
    sentence_lens = []
    current_len = 0
    i = 0
    ## they are already in sorted order
    current = []
    max_len = 0
    for x in dataset:
        if current_len + len(x) > batch_size:
            #print(current)
            current, lens  = pad_trunc_batch(current, max_len)
            batched_dataset.append(current)
            sentence_lens.append(lens)
            current = []
            current_len = 0
            max_len = 0
        max_len = max(len(x),max_len) ##
        current.append(x)
        current_len +=len(x)
    if len(current) > 0:
        current,lens  = pad_trunc_batch(current, max_len)
        sentence_lens.append(lens)
        batched_dataset.append(current)
    return batched_dataset, sentence_lens


FIELD_TO_IDX = {'id': 0, 'word': 1, 'lemma': 2, 'upos': 3, 'xpos': 4, 'feats': 5, 'head': 6, 'deprel': 7, 'deps': 8, 'misc': 9}

def read_conllu(file_name, cols = ['word','upos','head','deprel']):
    """
        Reads a conllu file and generates the vocabularies
    """
    assert file_name.endswith("conllu"), "File must a .conllu type"
    file = open(file_name, encoding = "utf-8").read().rstrip().split("\n")
    dataset = []
    sentence = []
    tok2ind = {PAD : PAD_IND, ROOT : ROOT_IND}
    pos2ind = {PAD : PAD_IND, ROOT : ROOT_IND}
    dep2ind = {PAD : PAD_IND, ROOT : ROOT_IND}
    total_word_size = 0
    root = [[ROOT for _ in range(len(cols))]]
    for line in file:
        if line.startswith("#"):
            continue
        elif line=="":
            sentence = root + sentence
            dataset.append(sentence)
            sentence = []
        else:
            line = line.split("\t")
            if "-" in line[0]: #skip expanded words
                continue
            total_word_size+=1
            sentence.append([line[FIELD_TO_IDX[x.lower()]] for x in cols])
            if line[1] not in tok2ind:
                tok2ind[line[1]] = len(tok2ind)
            if line[3] not in pos2ind:
                pos2ind[line[3]] = len(pos2ind)
            if line[7] not in dep2ind:
                dep2ind[line[7]] = len(dep2ind)
    if len(sentence):
        sentence = root + sentence
        dataset.append(sentence)
    ##
    dataset.sort(key = lambda x : len(x))
    dataset.reverse()
    assert all([len(d)>=len(d_) for d,d_ in zip(dataset,dataset[1:])]),\
        "Dataset is not sorted properly"
    tok_vocab = Vocab(tok2ind)

    dep_vocab = Vocab(dep2ind)
    pos_vocab = Vocab(pos2ind)
    return dataset, tok_vocab, dep_vocab, pos_vocab, total_word_size


def count_words_in_batches(dataset):
    c = 0
    for batch in dataset:
        for sent in batch:
            c+=len(sent)
    return c

class DepDataset(Dataset):
    """
        Dependency Parsing Dataset

        Batch size refers to the number of words in a batch
        Not the number of samples!!

    """
    def __init__(self, data_file, for_eval = False, vocabs = None,  transform = None, batch_size = 500):
        self.file_name = data_file
        self.for_eval = for_eval
        self.batch_size = batch_size
        if for_eval and not vocabs:
            raise AssertionError("Evaluation mode requires vocab")
        self.dataset, self.tok_vocab, self.dep_vocab, self.pos_vocab\
        ,self.total_word_size = read_conllu(self.file_name)
        self.average_length = self.total_word_size/len(self.dataset)
        print("Dataset size before batching {} and number of words : {}".format(len(self.dataset),self.total_word_size))
        self.dataset, self.sent_lens = group_into_batch(self.dataset, batch_size)
        print("{} batches created for {}.".format(len(self.dataset), self.file_name))
        print("{} words inside the batches".format(sum([sum(l) for l in self.sent_lens])))
        if for_eval :
            self.tok_vocab, self.dep_vocab, self.pos_vocab = vocabs
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.data_len = len(self.dataset)
        self.index = 0
        self.num_rels = len(self.dep_vocab.w2ind)
        self.num_pos  = len(self.pos_vocab.w2ind)

    def __len__(self):
        return len(self.dataset)

    def get_sents(self, sent_inds, feats = False, label_index = -1):
        sents = []
        labels = []
        for i in sent_inds:
            sent = []
            label = []
            for y in self.dataset[i]:
                if feats:
                    sent.append([y[0],y[1]])
                else:
                    sent.append([y[0]])
                label.append(y[label_index])
            sents.append(sent)
            labels.append(label)
        return sents,labels

    def __getitem__(self,idx):
        """
            Indexing for the DepDataset
            converts all the input into tensor before passing

            input is of form :
                word_ids  (Batch_size, Sentence_lengths)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ## batch contains multiple sentences
        ## as list of lists --> [word pos dep_ind dep_rel]
        batch = self.dataset[idx]
        lens = self.sent_lens[idx]
        ## create the bert tokens and pad each sentence to match the longest
        ## bert tokenization
        ## requires additional padding for bert
        tok_inds = []
        pos = []
        dep_rels = []
        dep_inds = []
        tokens = []
        for x in batch:
            t, p, d_i, d_r = zip(*x) ##unzip the batch
            tokens.append(t)
            pos.append(self.pos_vocab.map(p))
            dep_rels.append(self.dep_vocab.map(d_r))
            dep_inds.append([-1 if d=="[PAD]" or d=="[ROOT]" else int(d) for d in d_i])
            tok_inds.append(self.tok_vocab.map(t))
        assert len(tok_inds)== len(pos) == len(dep_rels) == len(dep_inds)
        tok_inds = torch.LongTensor(tok_inds)
        dep_inds = torch.LongTensor(dep_inds)
        dep_rels = torch.LongTensor(dep_rels)
        pos = torch.LongTensor(pos)
        bert_batch_before_padding = []
        bert_lens = []
        max_bert_len = 0
        bert2toks = []

        for sent, l in zip(batch,lens):
            my_tokens = [x[0] for x in sent]
            sentence = " ".join(my_tokens)
            bert_tokens = self.bert_tokenizer.tokenize(sentence)
            bert_lens.append(len(bert_tokens))
            bert_tokens = ["[CLS]"] + bert_tokens
            max_bert_len = max(max_bert_len,len(bert_tokens))
            ## bert_ind = 0 since we did not put [CLS] yet
            b2tok, ind = bert2token(my_tokens, bert_tokens, bert_ind = 1)
            assert ind == len(my_tokens), "Bert ids do not match token size"
            bert_batch_before_padding.append(bert_tokens)
            bert2toks.append(b2tok)
        bert_batch_after_padding, bert_lens = \
            pad_trunc_batch(bert_batch_before_padding, max_len = max_bert_len, bert = True)
        #print(bert_batch_after_padding)
        bert2tokens_padded, _ = pad_trunc_batch(bert2toks,max_len = max_bert_len, bert = True, b2t=True)
        bert2toks = torch.LongTensor(bert2tokens_padded)
        bert_batch_ids = torch.LongTensor([self.bert_tokenizer.convert_tokens_to_ids(sent) for \
            sent in bert_batch_after_padding])
        bert_seq_ids = torch.LongTensor([[1 for i in range(len(bert_batch_after_padding[0]))]\
            for j in range(len(bert_batch_after_padding))])
        return tokens, torch.tensor(lens), tok_inds, pos, dep_inds, dep_rels,\
            bert_batch_after_padding, bert_batch_ids,  bert_seq_ids, bert2toks

if __name__=="__main__":
    depdataset = DepDataset("../../datasets/tr_imst-ud-train.conllu", batch_size = 300)
    print(depdataset.average_length)
    i = 0
    b = time.time()
    tokens, sent_lens, tok_inds, pos, dep_inds, dep_rels,\
        bert_batch_after_padding, bert_batch_ids, bert_seq_ids, bert2toks = depdataset[i]
    e = time.time()
    print(" {} seconds for reading a batch ".format(e-b))
    for t, p, di, dr, b in zip(tokens, pos, dep_inds, dep_rels, bert_batch_ids):
        print(t,p,di,dr,b)
    for i in range(len(bert2toks)):
        print(len(bert2toks[i]),bert2toks[i][-1])
        print(len(bert_batch_after_padding[i]))
    #print(bert_batch_after_padding[0])
