from collections import Counter
import glob
import pandas as pd
import torch
import numpy as np
import logging
from pytorch_transformers import BertTokenizer
from parser.parsereader import group_into_batch, bert2token, pad_trunc_batch
from parser.parser import Vocab
from parser.utils import sort_dataset, unsort_dataset
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
    if token==START_TAG or token==END_TAG or token==PAD or token==ROOT_TAG:
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

def pad_trunc(sent,max_len, pad_len, pad_ind):
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


class DataReader():

    def __init__(self,file_path, task_name, batch_size = 3000,tokenizer = None,encoding = "utf-8"):
        self.encoding = encoding
        self.with_pos = with_pos
        self.file_path = file_path
        self.task_name = task_name
        self.batch_size = batch_size
        self.dataset, self.orig_idx , self.label_counts, self.pos_counts = self.get_dataset()
        print("Dataset size : {}".format(len(self.dataset)))
        self.data_len = len(self.dataset)
        self.pos2ind, self.l2ind, self.word2ind, self.vocab_size = self.get_vocabs()
        self.pos_voc = Vocab(self.pos2ind)
        self.label_voc = Vocab(self.l2ind)
        self.word_voc = Vocab(self.word2ind)
        self.batched_dataset, self.sentence_lens = group_into_batch(self.dataset,batch_size = self.batch_size)
        self.for_eval = False
        self.num_cats = len(self.l2ind)
        if tokenizer is None:    
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
            tokenizer.add_tokens(['[SOS]','[EOS]', '[ROOT]'])
        else:
            self.bert_tokenizer = tokenizer
        self.val_index = 0

    def get_ind2sent(self,sent):
        return " ".join([self.word2ind[w] for w in sent])

    def get_bert_input(self,batch_size = 1, morp = False, for_eval = False):
        if  for_eval:
            indexes = [i%self.data_len for i in range(self.val_index, self.val_index+batch_size)]
            self.val_index += batch_size
            self.val_index %=self.data_len
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
            bert2tok, final_len = bert2token(my_tokens, bert_tokens)
            lab = self.prepare_label(label,self.l2ind)
            bert_inputs.append([ torch.tensor([ids],dtype=torch.long), torch.tensor(enc_ids,dtype=torch.long),\
torch.tensor([seq_ids],dtype=torch.long), torch.tensor(bert2tok), lab])
        return my_tokens, bert_tokens, bert_inputs


    def get_vocabs(self):
        l2ind = {PAD : PAD_IND, START_TAG:START_IND, END_TAG: END_IND }
        word2ix = {PAD : PAD_IND, START_TAG:START_IND, END_TAG: END_IND}
        pos2ind = {PAD : PAD_IND, START_TAG:START_IND, END_TAG: END_IND}
        #l2ind = {PAD : PAD_IND, START_TAG:START_IND, END_TAG: END_IND, ROOT_TAG:PAD_IND }
        #word2ix = {PAD : PAD_IND, START_TAG:START_IND, END_TAG: END_IND ,ROOT_TAG:PAD_IND}
        #pos2ind = {PAD : PAD_IND, START_TAG:START_IND, END_TAG: END_IND ,ROOT_TAG:PAD_IND}
        #print(self.label_counts)
        
        for x in self.label_counts:
            l2ind[x] = len(l2ind)
        for x in self.pos_counts:
            pos2ind[x] = len(pos2ind)

        for sent in self.dataset:
            for word in sent:
                if word[0] not in word2ix:
                    word2ix[word[0]]=len(word2ix)
        vocab_size = len(word2ix)
        return pos2ind, l2ind, word2ix, vocab_size

    def get_dataset(self):
        print("Reading from {} ".format(self.file_path))
        dataset = open(self.file_path,encoding=self.encoding).read().split("\n")
        new_dataset = []
        sent = []
        label_counts = Counter()
        pos_counts = Counter()
        root = [START_TAG, START_TAG, START_TAG, START_TAG]
        for line in dataset:
            if len(line.split())<2:
                if len(sent)>0:
                    sent.append([END_TAG, END_TAG , END_TAG, END_TAG ])
                    if len(sent)>2:
                        new_dataset.append([root]+sent)
                    #new_dataset.append(sent)
                    sent = []
            else:
                row = line.rstrip().split()
                row[0] = row[0].replace("\ufeff","")
                row[0] = row[0].replace('\u200f',"")
                row[0] = row[0].replace('\t','')
                row[0] = row[0].replace(' ','')
                sent.append(row)
                label_counts.update([row[-1]])

                pos_counts.update([row[-2]])
        if len(sent)>0:
            sent.append([END_TAG, END_TAG, END_TAG , END_TAG ])
            new_dataset.append([root]+sent)
            #new_dataset.append(sent)
        print("Number of sentences for {}  : {} ".format(self.file_path,len(new_dataset)))
        new_dataset, orig_idx = sort_dataset(new_dataset, sort = True)
        
        return new_dataset, orig_idx, label_counts, pos_counts

    def get_next_data(sent_inds, data_len=-1,feats = True, padding=False):
        sents, labels = self.get_sents(sent_inds, feats = feats)
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
                sent_vector = pad_trunc(sent_vector, max_len=100,pad_len=400)
                lab_ = pad_trunc(lab_,max_len=100,pad_len=1)
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
        return sents, labels

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

    def get_1d_targets(self,targets):
        prev_tag = self.l2ind[START_TAG]
        tagset_size = self.num_cats
        targets_1d = []
        for current_tag in targets:
            targets_1d.append(current_tag*(tagset_size)+ prev_tag)
            prev_tag = current_tag
        return targets_1d

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
    def __len__(self):
        return len(self.batched_dataset)
    def __getitem__(self,idx):
        """
            Indexing for the DepDataset
            converts all the input into tensor before passing

            input is of form :
                word_ids  (Batch_size, Sentence_lengths)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if not self.for_eval:
            idx = np.random.randint(len(self.batched_dataset))
            idx = idx%len(self.batched_dataset)
        batch = self.batched_dataset[idx]
        lens = self.sentence_lens[idx]
        tok_inds = []
        ner_inds = []
        pos_inds = []
        tokens = []
        for x in batch:
            toks, feats,poss, labels = zip(*x) ##unzip the batch
            tokens.append(toks)
            pos_inds.append(self.pos_vocab.map(poss))
            tok_inds.append(self.word_voc.map(toks))
            ner_inds.append(self.get_1d_targets(self.label_voc.map(labels)))
            #if self.for_eval:
                #print("DEV LABELS {} \n DEV INDS {} ".format(labels,ner_inds))
                #print(labels)
                #print("DEV vocab : {}".format(self.label_voc.w2ind))
                #print(ner_inds)
        assert len(tok_inds)== len(ner_inds) == len(tokens) == len(batch) == len(pos_inds)
        for toks in tokens:
            if toks[0]!="[SOS]":
                logging.info("Problemli batch")
                logging.info(tokens)
                break
        tok_inds = torch.LongTensor(tok_inds)
        ner_inds = torch.LongTensor(ner_inds)
        pos_inds = torch.LongTensor(pos_inds)
        bert_batch_before_padding = []
        bert_lens = []
        max_bert_len = 0
        bert2toks = []
        cap_types = []
        masks = torch.ones(tok_inds.shape,dtype=torch.bool)
        i = 0
        for sent, l in zip(batch,lens):
            my_tokens = [x[0] for x in sent]
            cap_types.append(torch.tensor([get_orthographic_feat(x[0]) for x in sent]))
            sentence = " ".join(my_tokens)
            masks[i,:l] = torch.tensor([0]*l,dtype=torch.bool)    
            i+=1
            #sentence = " ".join(my_tokens)
            #bert_tokens = self.bert_tokenizer.tokenize(sentence)
            #bert_tokens = ["[CLS]"] + bert_tokens
            #b2tok, ind = bert2token(my_tokens, bert_tokens, bert_ind = 1)
            bert_tokens = []
            b2tok = []
            for j, token in enumerate(my_tokens):
                btok = self.bert_tokenizer.tokenize(token)
                
                if len(btok)==0:
                    b2tok = b2tok + [j]
                    bert_tokens = bert_tokens + ['[UNK]']
                b2tok = b2tok + [j for _ in range(len(btok))]
                bert_tokens = bert_tokens + btok
            bert_tokens = ["[CLS]"] + bert_tokens
            bert_lens.append(len(bert_tokens))
            ind = b2tok[-1]+1
            max_bert_len = max(max_bert_len,len(bert_tokens))
            ## bert_ind = 0 since we did not put [CLS] yet
            assert ind == len(my_tokens), "Bert ids do not match token size {} {} ".format(ind , len(my_tokens))
            bert_batch_before_padding.append(bert_tokens)
            bert2toks.append(b2tok)
        bert_batch_after_padding, bert_lens = \
            pad_trunc_batch(bert_batch_before_padding, max_len = max_bert_len, bert = True)
        #print(bert_batch_after_padding)
        bert2tokens_padded, _ = pad_trunc_batch(bert2toks,max_len = max_bert_len, bert = True, b2t=True)
        bert_batch_ids = torch.LongTensor([self.bert_tokenizer.convert_tokens_to_ids(sent) for \
            sent in bert_batch_after_padding])
        assert all([len(x)==len(bert_batch_ids[0]) for x in bert_batch_ids])
        bert_seq_ids = torch.LongTensor([[1 for i in range(len(bert_batch_after_padding[0]))]\
            for j in range(len(bert_batch_after_padding))])
        dep_rels = torch.tensor([])
        dep_inds = torch.tensor([])
        data = torch.tensor(lens), masks, tok_inds, ner_inds,pos_inds, dep_rels,dep_inds, bert_batch_ids,  bert_seq_ids, torch.tensor(bert2tokens_padded,dtype=torch.long) , torch.stack(cap_types)
        return tokens, bert_batch_after_padding, data 
if __name__ == "__main__":
    data_path = '../datasets/turkish-ner-train.tsv'
    reader = DataReader(data_path,"NER")
    print(sum(map(len,reader.dataset))/reader.data_len)
    batched_dataset, sentence_lens = group_into_batch(reader.dataset,batch_size = 300)
