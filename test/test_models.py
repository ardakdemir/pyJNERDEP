from transformers import AutoTokenizer, AutoModel,BertForPreTraining,BertForTokenClassification

import argparse
import torch.nn as nn
import torch


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

output_file = "tokenized.txt"



class BertModelforJoint(nn.Module):

    def __init__(self, lang):
        super(BertModelforJoint, self).__init__()
        self.model = self.load_bert_model(lang)
        self.lang = lang
        # base model for generating bert output

    def load_bert_model(self,lang):
        model_name = model_name_dict[lang]
        if lang == "hu":
            model = BertForPreTraining.from_pretrained(model_name, from_tf=True, output_hidden_states=True)
        else:
            model = BertForTokenClassification.from_pretrained(model_name)
            model.classifier = nn.Identity()
        return model

    def forward(self,input,attention_mask,**kwargs):
        """
            Output the logits of the last layer for each word...
        :param input:
        :return:
        """
        if self.lang == "hu":
            output = model(input, attention_mask)[2][-1]
        else:
            output = model(input, attention_mask)[0]
        return output

def load_bert_model(lang):
    model_name = model_name_dict[lang]
    if lang == "hu":
        model = BertForPreTraining.from_pretrained(model_name, from_tf=True,output_hidden_states=True)
    else:
        model = BertForTokenClassification.from_pretrained(model_name)
        model.classifier = nn.Identity()
    return model


def test_models():
    for lang,model_name in model_name_dict.items():
        sent = sent_dict[lang]
        print("Testing {}...".format(lang))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertModelforJoint(lang)
        tokens = tokenizer.tokenize(sent)
        input = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokens)).reshape(1, -1)
        print("Token ids: {}".format(input))
        attention_mask = torch.ones(*input.shape)
        if lang == "hu":
            output = model(input, attention_mask)[2][-1]
        else:
            output = model(input, attention_mask)[0]
        assert output.shape == (*input.shape,768)
        # print(model)
        with open(output_file, "a", encoding="utf-8") as o:
            o.write("{}\n".format(lang))
            o.write("Sentence: " + sent)
            o.write("\n")
            o.write("Tokens: " + " ".join(tokens))
            o.write("\n")

def main():
    test_models()

if __name__ == "__main__":
    main()