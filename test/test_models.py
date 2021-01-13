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
                   "en":"bert-base-cased",
                   "mbert":"bert-base-multilingual-cased",
                   "bert_en":"bert-base-cased"}

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

class MyDict():
    """
        My simple dictionary to allow wv.vocab access
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


def load_bert_model(lang):
    model_name = model_name_dict[lang]
    if lang == "hu":
        model = BertForPreTraining.from_pretrained(model_name, from_tf=True, output_hidden_states=True)
    else:
        model = BertForTokenClassification.from_pretrained(model_name)
        model.classifier = nn.Identity()
    return model


def load_word2vec(lang):
    model_name = word2vec_dict[lang]
    if lang == "cs":
        model = MyWord2Vec(model_name, lang)
    else:
        model = Word2Vec.load(model_name)
    return model


def test_fastext():
    print("Testing fasttext")
    vec_dict = {}
    for lang, model_name in fasttext_dict.items():
        print("Testing {}".format(lang))
        if not os.path.exists(model_name):
            print("Model not found. SKipping {}".format(lang))
            continue
        tokens = sent_dict[lang].split(" ")
        model = fasttext.load_model(model_name)
        vecs = []
        for tok in tokens:
            vec = model.get_word_vector(tok)
            vecs.append(vec)
            assert len(vec) == 300
        print("{} vecs {} tokens".format(len(vecs), len(tokens)))
        vec_dict[lang] = vecs
    return vec_dict


def test_word2vec():
    print("Testing word2vec")
    vec_dict = {}
    for lang, model_name in word2vec_dict.items():
        print("Testing {}".format(lang))
        if not os.path.exists(model_name):
            print("Model not found. SKipping {}".format(lang))
            continue
        tokens = sent_dict[lang].split(" ")
        model = load_word2vec(lang)
        vecs = []
        c = 0
        for tok in tokens:
            if tok in model.wv.vocab:
                vec = model.wv[tok]
                vecs.append(vec)
            else:
                print("Could not find token in vocab")
                vecs.append(unks[lang])
                c += 1
        print("{}/{} oovs".format(c, len(tokens)))
        print("{} vecs {} tokens".format(len(vecs), len(tokens)))
        vec_dict[lang] = vecs
    return vec_dict


def test_models():
    for lang, sent in sent_dict.items():

        for mod in ["bert","mbert","bert_en"]:
            sent = sent_dict[lang]
            print("Testing {} with {}...".format(lang, mod))
            if mod == "bert":
                model_name = model_name_dict[lang]
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = BertModelforJoint(lang)
            else:
                model_name = model_name_dict[mod]
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = BertModelforJoint(mod)
            tokens = tokenizer.tokenize(sent)
            input = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokens)).reshape(1, -1)
            print("Token ids: {}".format(input.shape))
            attention_mask = torch.ones(*input.shape)
            output = model(input, attention_mask)
            print("Output shape: {}".format(output.shape))
            assert output.shape == (input.shape[0], input.shape[1], 768)
            # print(model)
            with open(output_file, "a", encoding=encoding_map[lang]) as o:
                o.write("{}\t{}\n".format(lang,mod))
                o.write("Sentence: " + sent)
                o.write("\n")
                o.write("Tokens: " + " ".join(tokens))
                o.write("\n")


def main():
    test_models()
    # test_word2vec()
    model_name = "dbmdz/bert-base-turkish-cased"
    # print(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # print(tokenizer)

    # test_fastext()

if __name__ == "__main__":
    main()
