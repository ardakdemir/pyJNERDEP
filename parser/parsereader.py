from collections import Counter, defaultdict
import glob
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


from parser.parser import Vocab, VOCAB_PREF
from parser.utils import sort_dataset, unsort_dataset

import logging
import time
import random

UNK = "[UNK]"
UNK_IND = 3
PAD = "[PAD]"
PAD_IND = 0
START_TAG = "[SOS]"
END_TAG   = "[EOS]"
START_IND = 1
END_IND = 2
ROOT_TAG = "[ROOT]"
ROOT_IND = 1
def all_num(token):
    n = "0123456789."
    for c in token:
        if c not in n:
            return False
    return True

def get_orthographic_feat(token):
    if token=="[SOS]" or token==END_TAG or token==PAD :
        return 5
    if "'" in token:
        return 4
    if all_num(token):
        return 3
    if token.isupper():
        return 2
    if token.istitle():
        return 1 
    if token.islower():
        return 0
    return 0

def bert2token(my_tokens, bert_tokens, bert_ind = 1):
    inds = []
    token_sum =""
    bert_ind = bert_ind
    for ind in range(len(my_tokens)):
        my_token = my_tokens[ind]
        #token_sum = bert_tokens[bert_ind]
        #bert_ind = bert_ind +1
        #inds.append(ind)
        b_len = len(bert_tokens)
        if bert_tokens[bert_ind]=='[UNK]':
            inds.append(ind)
            bert_ind+=1
            continue
        while bert_ind < b_len and  len(token_sum)!=len(my_token):
            if bert_tokens[bert_ind].startswith("##"):
                token_sum +=bert_tokens[bert_ind][2:]
            elif bert_tokens[bert_ind].endswith("##"):
                token_sum+=bert_tokens[bert_ind][:-2]
            else:
                token_sum +=bert_tokens[bert_ind]
            bert_ind = bert_ind +1
            inds.append(ind)
        #while len(token_sum)!=len(my_token) and bert_ind<len(bert_tokens):
        #    token = bert_tokens[bert_ind]
        #    if token.startswith("##"):
        #        token_sum+=token[2:]
        #    else:
        #        token_sum+=token
        #    inds.append(ind)
        #    bert_ind+=1
        #logging.info("my {} bert {}".format(token_sum,my_token))
        if len(token_sum)!=len(my_token):
            
            logging.info("Problem\n bert {} \n my {}".format(bert_tokens,my_tokens))
            logging.info('index : {} my token  {}  bert token {} '.format(ind, my_token,token_sum))
            raise AssertionError
        token_sum=""

    return inds, ind+1

def pad_trunc_batch(batch, max_len,batch_size = None , pad = PAD, pad_ind = PAD_IND, bert = False,b2t=False):
    padded_batch = []
    sent_lens = []
    for sent in batch:
        sent_lens.append(min([len(sent),max_len,batch_size if batch_size is not None else 512]))
        if len(sent)>=max_len:
            if bert:
                if b2t:
                    padded_batch.append(sent)
                else:
                    padded_batch.append(sent + ["[SEP]"])
            else:
                ## Changed this!!! 
                padded_batch.append(sent[:batch_size])
        else:
            l = len(sent)
            if not bert:
                index_len = len(sent[0])
            padded_sent = sent
            if not bert:
                max_len = min(max_len,batch_size if batch_size is not None else 512)
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
    if not all([len(x)==len(padded_batch[0]) for x in padded_batch]):
        print("Problematic batch")
        logging.info("Problematicv batch")
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
    for x in dataset:
        current.append(x)
        max_len = max(len(x),max_len) ##
        current_len +=len(x)
        if current_len  > batch_size:
            #print(current)
            current, lens  = pad_trunc_batch(current, max_len,batch_size)
            batched_dataset.append(current)
            sentence_lens.append(lens)
            current = []
            current_len = 0
            max_len = 0
    if len(current) > 0:
        current,lens  = pad_trunc_batch(current, max_len,batch_size)
        sentence_lens.append(lens)
        batched_dataset.append(current)

    return batched_dataset, sentence_lens


FIELD_TO_IDX = {'id': 0, 'word': 1, 'lemma': 2, 'upos': 3, 'xpos': 4, 'feats': 5, 'head': 6, 'deprel': 7, 'deps': 8, 'misc': 9}

def read_conllu(file_name, cols = ['word','upos','head','deprel'], get_ner=False,batch_size=None):
    """
        Reads a conllu file and generates the vocabularies
    """
    assert file_name.endswith("conllu") or 'conllu' in file_name, "File must a .conllu type"
    file = open(file_name, encoding = "utf-8").read().rstrip().split("\n")
    dataset = []
    sentence = []
    tok2ind = {PAD : PAD_IND, START_TAG : START_IND, UNK:UNK_IND, END_TAG: END_IND}
    pos2ind = {PAD : PAD_IND, START_TAG : START_IND, END_TAG: END_IND, UNK:UNK_IND}
    dep2ind = {PAD : PAD_IND, START_TAG : START_IND, END_TAG : END_IND, UNK:UNK_IND}
    #ner2ind = {PAD : PAD_IND, START_TAG : START_IND, END_TAG : END_IND, UNK:UNK_IND}
    total_word_size = 0
    root = [[START_TAG for _ in range(len(cols))]]
    for line in file:
        if line.startswith("#"):
            continue

        elif line=="":
            sentence = root + sentence
            if batch_size is not None:
                if len(sentence) < batch_size:
                    dataset.append(sentence)
            else:
                dataset.append(sentence)
            sentence = []
        else:
            line = line.split("\t")
            if "-" in line[0] or  "." in line[0]: #skip expanded words
                continue

            total_word_size += 1
            fields = [line[FIELD_TO_IDX[x.lower()]] for x in cols]
            if fields[-1] == "_":
                print("Problemliii")
                print(" ".join(line).encode("utf-8"))
                fields[-1] = line[-2].split(":")[-1]
                fields[-2] = int(line[-2].split(":")[0])
            if 'misc' in fields and get_ner:
                j = json.loads(fields[-1])
                ner_tag =  j["NER_TAG"]
                fields[-1] =  ner_tag
            sentence.append(fields)
            if fields[0] not in tok2ind:
                tok2ind[fields[0]] = len(tok2ind)
            if fields[1] not in pos2ind:
                pos2ind[fields[1]] = len(pos2ind)
            if fields[-1] not in dep2ind:
                dep2ind[fields[-1]] = len(dep2ind)
            #if line[FIELD_TO_IDX['misc']] not in ner2ind:
            #    ner2ind[line[FIELD_TO_IDX['misc']]] = len(ner2ind)

    
    if len(sentence):
        sentence = root + sentence
        dataset.append(sentence)
    ##
    sort = True
    logging.info("Sorting dataset is set to {}".format(sort))
    dataset, orig_idx = sort_dataset(dataset, sort = sort)
    if sort :
        assert all([len(d)>=len(d_) for d,d_ in zip(dataset,dataset[1:])]),\
            "Dataset is not sorted properly"
    tok_vocab = Vocab(tok2ind)
    if 'misc' in cols and get_ner:
        ner_vocab = Vocab(ner2ind)
    dep_vocab = Vocab(dep2ind)
    pos_vocab = Vocab(pos2ind)
    vocabs = {"tok_vocab": tok_vocab, "dep_vocab": dep_vocab, "pos_vocab": pos_vocab}
    return dataset, orig_idx, vocabs,  total_word_size



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
    def __init__(self, data_file, for_eval = False, vocabs = None,  transform = None, batch_size = 500, tokenizer = None, type="dep"):
        self.file_name = data_file
        self.for_eval = for_eval
        self.batch_size = batch_size
        if for_eval and not vocabs:
            raise AssertionError("Evaluation mode requires vocab")
        self.dataset, self.orig_idx, self.vocabs\
        ,self.total_word_size = read_conllu(self.file_name,batch_size=self.batch_size)
        self.average_length = self.total_word_size/len(self.dataset)
        print("DEP dataset number of sents : {}".format(len(self.dataset)))
        print("Dataset size before batching {} and number of words : {}".format(len(self.dataset),self.total_word_size))
        self.dataset, self.sent_lens = group_into_batch(self.dataset, batch_size)
        print("{} batches created for {}.".format(len(self.dataset), self.file_name))
        print("{} words inside the batches".format(sum([sum(l) for l in self.sent_lens])))
        if for_eval :
            self.vocabs = vocabs
        if tokenizer is None:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        else:
            self.bert_tokenizer = tokenizer
        self.data_len = len(self.dataset)
        self.index = 0
        self.num_rels = len(self.vocabs['dep_vocab'].w2ind)
        self.num_pos  = len(self.vocabs['pos_vocab'].w2ind)
        if type == 'joint':
            self.num_ner = len(self.vocabs['ner_vocab'].w2ind)

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
        #x = [i for i in range(len(self.dataset))]
        #random.shuffle(x)
        #idx = x[idx]
        #print(idx)
        if not self.for_eval:
            idx = np.random.randint(len(self.dataset))
        idx = idx% len(self.dataset)
        
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
            pos.append(self.vocabs['pos_vocab'].map(p))
            dep_rels.append(self.vocabs['dep_vocab'].map(d_r))
            dep_inds.append([0 if d=="[PAD]" or d=="[SOS]" else int(d) for d in d_i])
            tok_inds.append(self.vocabs['tok_vocab'].map(t))
        assert len(tok_inds)== len(pos) == len(dep_rels) == len(dep_inds)
        tok_inds = torch.LongTensor(tok_inds)
        dep_inds = torch.LongTensor(dep_inds)
        dep_rels = torch.LongTensor(dep_rels)
        pos = torch.LongTensor(pos)
        bert_batch_before_padding = []
        bert_lens = []
        max_bert_len = 0
        bert2toks = []
        masks = torch.ones(tok_inds.shape,dtype=torch.bool)
        i = 0
        cap_types = []
        for sent, l in zip(batch,lens):
            my_tokens = [x[0] for x in sent]
            cap = [get_orthographic_feat(x[0]) for x in sent]
            cap_types.append(torch.tensor(cap))
            masks[i,:l] = torch.tensor([0]*l,dtype=torch.bool)    
            #sentence = " ".join(my_tokens)
            #bert_tokens = self.bert_tokenizer.tokenize(sentence)
            #bert_tokens = ["[CLS]"] + bert_tokens
            #b2tok, ind = bert2token(my_tokens, bert_tokens, bert_ind = 1)
            bert_tokens = []
            b2tok = []
            for j, token in enumerate(my_tokens):
                btok = self.bert_tokenizer.tokenize(token)
                if len(btok)==0:
                    bert_tokens = bert_tokens + ['[UNK]']
                    b2tok = b2tok + [j]
                bert_tokens = bert_tokens + btok
                b2tok = b2tok + [j for _ in range(len(btok))]
            bert_tokens = ["[CLS]"] + bert_tokens
            bert_lens.append(len(bert_tokens))
            ind = b2tok[-1]+1
            max_bert_len = max(max_bert_len,len(bert_tokens))
            ## bert_ind = 0 since we did not put [CLS] yet
            assert ind == len(my_tokens), "Bert ids do not match token size"
            bert_batch_before_padding.append(bert_tokens)
            bert2toks.append(b2tok)
            i+=1
        bert_batch_after_padding, bert_lens = \
            pad_trunc_batch(bert_batch_before_padding, max_len = max_bert_len, bert = True)
        #print(bert_batch_after_padding)
        bert2tokens_padded, _ = pad_trunc_batch(bert2toks,max_len = max_bert_len, bert = True, b2t=True)
        bert2toks = torch.LongTensor(bert2tokens_padded)
        bert_batch_ids = torch.LongTensor([self.bert_tokenizer.convert_tokens_to_ids(sent) for \
            sent in bert_batch_after_padding])
        bert_seq_ids = torch.LongTensor([[1 for i in range(len(bert_batch_after_padding[0]))]\
            for j in range(len(bert_batch_after_padding))])
        ner_inds = torch.tensor([])
        data = torch.tensor(lens),masks, tok_inds, ner_inds, pos, dep_inds, dep_rels,\
        bert_batch_ids,  bert_seq_ids, bert2toks, torch.stack(cap_types)
        return tokens,bert_batch_after_padding, data 

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
