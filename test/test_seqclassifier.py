from transformers import AutoTokenizer, AutoModel, BertForPreTraining, BertForTokenClassification
import io
import argparse
import torch.nn as nn
import numpy as np
import torch
from gensim.models import Word2Vec
import os
import fasttext as ft
import fasttext.util
from sequence_classifier import SequenceClassifier
from sareader import SentReader

sent_dict = {"tr": "Fakat, temel gündem maddesi değişmiyordu: Türkiye.",
             "cs": "Potřebujete rychle poradit?",
             "hu": "Az ezredfordulós szilveszter valószínűleg az átlagos év véginél komolyabb feladatokat ró a sürgősségi betegellátás szervezeteire és a rendőrségre.",
             "jp": "為せば成る為さねば成らぬ。 \n麩菓子は 麩を主材料とした日本の菓子。",
             "en": "I am moving to Azabu-juban next year ...",
             "fi": " Showroomilla esiteltiin uusi The Garden Collection ja tarjoiltiin maukasta aamupalaa aamu-unisille muotibloggaajille."}

model_name_dict = {"jp": "cl-tohoku/bert-base-japanese",
                   "tr": "dbmdz/bert-base-turkish-cased",
                   "hu": "/home/aakdemir/bert_models/hubert",
                   "fi": "TurkuNLP/bert-base-finnish-cased-v1",
                   "cs": "DeepPavlov/bert-base-bg-cs-pl-ru-cased",
                   "en": "bert-base-cased",
                   "mbert": "bert-base-multilingual-cased",
                   "bert_en": "bert-base-cased"}

output_file = "tokenized.txt"

encoding_map = {"cs": "utf-8",
                "tr": "utf-8",
                "hu": "utf-8",
                "en": "latin-1",
                "jp": "utf-8",
                "fi": "utf-8"}

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

unks = {l: np.random.rand(word2vec_lens[l]) for l in word2vec_lens.keys()}


def test_sequence_classifiers():
    for mod in ["bert", "mbert", "bert_en", "fastext", "word2vec", "random_init"]:
        print("Testing sequence classifier with {}".format(mod))
        lang, model_type = "tr", model
        data_path = '../datasets/sa_movie_turkish-test.json'
        tokenizer = AutoTokenizer.from_pretrained(model_name_dict[lang])
        reader = SentReader(data_path, tokenizer=tokenizer)

        num = reader.num_cats
        word_vocab = reader.word_vocab
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seq_classifier = SequenceClassifier(lang, word_vocab, model_type, num_cats)

        seq_classifier.train()
        seq_classifier.to(device)

        seq_classifier.zero_grad()

        data = reader[0]
        labels = data[3]
        class_logits = seq_classifier(data)
        print("Logit shape: {} label shape: {}".format(class_logits.shape, labels.shape))
        loss = seq_classifier.loss(labels)
        loss.backward()
        seq_classifier.optimizer_step()


def main():

    model_name = "dbmdz/bert-base-turkish-cased"
    test_sequence_classifiers()
    # print(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # print(tokenizer)

    # test_fastext()


if __name__ == "__main__":
    main()
