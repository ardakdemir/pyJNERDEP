from collections import Counter
import glob
import pandas as pd
import torch
import numpy as np
from pytorch_transformers import BertTokenizer
class DataReader():

    def __init__(self,file_path,task_name,*kwargs):
        self.file_path = file_path
        self.task_name = task_name
        self.dataset, self.label_counts = self.get_dataset()
        self.data_len = len(self.dataset)
        self.l2ind, self.word2ind, self.vocab_size = self.get_vocabs()
        self.num_cats = len(self.l2ind)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.val_index = 0


    def get_bert_input(self,batch_size = 1, morp = False,for_eval=False):
        if  for_eval:
            indexes = [i for i in range(self.val_index,self.val_index+batch_size)]
            self.val_index += batch_size
        else:
            indexes = np.random.permutation([i for i in range(self.data_len)])
            indexes = indexes[:batch_size]

        sents, labels = self.get_sents(indexes, feats = morp)
        bert_inputs = []
        for sent, label in zip(sents, labels):
            my_tokens = [x[0] for x in sent]
            sentence = " ".join(my_tokens)
            marked_sent = "[CLS]" + sentence + "[SEP]"
            bert_tokens = self.bert_tokenizer.tokenize(marked_sent)
            ids = self.bert_tokenizer.convert_tokens_to_ids(bert_tokens)
            enc_ids = self.bert_tokenizer.encode(sentence)
            seq_ids = [1 for i in range(len(bert_tokens))]
            bert2tok, final_len = self.bert2token(my_tokens, bert_tokens)
            lab = self.prepare_label(label,self.l2ind)
            bert_inputs.append([ torch.tensor([ids],dtype=torch.long), torch.tensor(enc_ids,dtype=torch.long),\
torch.tensor([seq_ids],dtype=torch.long), torch.tensor(bert2tok), lab])
        return my_tokens, bert_tokens, bert_inputs


    def bert2token(self, my_tokens, bert_tokens):
        inds = []
        token_sum =""
        bert_ind = 1
        for ind in range(len(my_tokens)):
            my_token = my_tokens[ind]
            while len(token_sum)!=len(my_token) and bert_ind<len(bert_tokens)-1:
                token = bert_tokens[bert_ind]
                if token.startswith("#"):
                    token_sum+=token[2:]
                else:
                    token_sum+=token
                inds.append(ind)
                bert_ind+=1
            assert len(token_sum)==len(my_token), print(token_sum)
            token_sum=""
        return inds, ind+1

    def get_vocabs(self):
        l2ind = {}
        START_TAG = "SOS"
        END_TAG   = "EOS"

        for i,x in enumerate(self.label_counts):
            l2ind[x] = i
        l2ind[START_TAG] = len(l2ind)
        l2ind[END_TAG] = len(l2ind)
        word2ix = {}
        for sent in self.dataset:
            for word in sent:
                if word[0] not in word2ix:
                    word2ix[word[0]]=len(word2ix)
        vocab_size = len(word2ix)
        return l2ind, word2ix, vocab_size

    def get_dataset(self):
        dataset = open(self.file_path,encoding='utf-8').readlines()
        new_dataset = []
        sent = []
        label_counts = Counter()
        for line in dataset:
            if line.rstrip()=='':
                if len(sent)>0:
                    new_dataset.append(sent)
                    sent = []
            else:
                row = line.rstrip().split()
                sent.append(row)
                label_counts.update([row[-1]])
        return new_dataset, label_counts



    def pad_trunc(self,sent,max_len, pad_len):
        if len(sent)>max_len:
            return sent[:max_len]
        else:
            l = len(sent)
            for i in range(max_len-l):
                if pad_len ==1:
                    sent.append(0)
                else:
                    sent.append([0 for i in range(pad_len)])
            return sent

    def get_next_data(self,sent_inds, data_len=-1,feats = True,padding=False):
        sents, labels = self.get_sents(sent_inds,feats = feats)
        datas = []
        labs = []
        for sent, label in zip(sents,labels):
            sent_vector = []
            lab_ = []
            for l,word in zip(label,sent):
                if data_len!=-1 and data_len == len(lab_):
                    break
                sent_vector.append(self.getword2vec(word))
                lab_.append(l2ind[l])
            if padding :
                sent_vector = self.pad_trunc(sent_vector, max_len=100,pad_len=400)
                lab_ = self.pad_trunc(lab_,max_len=100,pad_len=1)
            datas.append(sent_vector)
            labs.append(lab_)
        return torch.tensor(np.array(datas)).float(),torch.tensor(labs[0],dtype=torch.long).view(-1)

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

    ## compatible with getSent and for word embeddings
    def prepare_sent(self, sent, word2ix):
        idx = [word2ix[word[0]] for word in sent]
        return torch.tensor(idx,dtype=torch.long)

    def prepare_label(self, labs, l2ix):
        idx = [l2ix[lab] for lab in labs]
        return torch.tensor(idx,dtype=torch.long)

    def getword2vec(self, row):
        key = row[0].lower()
        root = row[1][:row[1].find("+")].encode().decode("unicode-escape")
        while(len(key)>0):
            if key in word_vectors:
                return word_vectors[key]
            elif root.lower() in word_vectors:
                return word_vectors[root.lower()]
            else:
                return word_vectors["OOV"]
        return 0

    def getword2vec2(self, row):
        key = row[0].lower()
        root = row[1][:row[1].find("+")].encode().decode("unicode-escape") ## for turkish special chars
        while(len(key)>0):
            if key in word_vectors:
                return 2
            elif root.lower() in word_vectors:
                return 1
            else:
                return 0
        return 0
