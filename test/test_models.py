from transformers import AutoTokenizer, AutoModel, BertForPreTraining, BertForTokenClassification
import io
import argparse
import torch.nn as nn
import numpy as np
import torch
from gensim.models import Word2Vec
import os

sent_dict = {"tr": "Sen benim kim olduğumu biliyor musun?",
             "cs": "Potřebujete rychle poradit?",
             "hu": "Az ezredfordulós szilveszter valószínűleg az átlagos év véginél komolyabb feladatokat ró a sürgősségi betegellátás szervezeteire és a rendőrségre.",
             "jp": "為せば成る為さねば成らぬ。\n麩菓子は、麩を主材料とした日本の菓子。",
             "fi": " Showroomilla esiteltiin uusi The Garden Collection ja tarjoiltiin maukasta aamupalaa aamu-unisille muotibloggaajille."}

model_name_dict = {"jp": "cl-tohoku/bert-base-japanese",
                   "tr": "dbmdz/bert-base-turkish-cased",
                   "hu": "/home/aakdemir/bert_models/hubert",
                   "fi": "TurkuNLP/bert-base-finnish-cased-v1",
                   "cs": "DeepPavlov/bert-base-bg-cs-pl-ru-cased"}

word2vec_dict = {"jp": "../../word_vecs/jp/jp.bin",
                 "tr": "../../word_vecs/tr/tr.bin",
                 "hu": "../../word_vecs/hu/hu.bin",
                 "fi": "../../word_vecs/fi/fi.bin",
                 "cs": "../../word_vecs/cs/cs.txt"}

output_file = "tokenized.txt"

word2vec_lens = {"tr": 200,
                 "hu": 300,
                 "fi": 300,
                 "cs": 100,
                 "jp": 300}

unks = {l: np.random.rand(word2vec_lens[l]) for l in word2vec_lens.keys()}

encoding_map = {"cs": "latin-1",
                "tr": "utf-8",
                "hu": "utf-8",
                "fi": "utf-8"}


class MyWord2Vec():
    """
        My word2Vec that is initialized from a file
    """

    def __init__(self, file_name, lang):
        self.file_name = file_name
        self.vocab, self.wv, self.dim = self.get_vectors(file_name)
        self.encoding_map

    def get_vectors(self, file_name):
        with open(file_name, "r", encoding=encoding_map[lang]) as f:
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


def test_embeddings():
    print("Testing word2vec")
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
    return test_embeddings


def test_models():
    for lang, model_name in model_name_dict.items():
        sent = sent_dict[lang]
        print("Testing {}...".format(lang))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertModelforJoint(lang)
        tokens = tokenizer.tokenize(sent)
        input = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokens)).reshape(1, -1)
        print("Token ids: {}".format(input.shape))
        attention_mask = torch.ones(*input.shape)
        output = model(input, attention_mask)
        print("Output shape: {}".format(output.shape))
        assert output.shape == (input.shape[0], input.shape[1], 768)
        # print(model)
        with open(output_file, "a", encoding="utf-8") as o:
            o.write("{}\n".format(lang))
            o.write("Sentence: " + sent)
            o.write("\n")
            o.write("Tokens: " + " ".join(tokens))
            o.write("\n")


def main():
    test_models()
    test_embeddings()


if __name__ == "__main__":
    main()
