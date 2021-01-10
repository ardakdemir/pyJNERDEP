from transformers import BertModel, AutoTokenizer, AutoModel, BertForPreTraining, BertForTokenClassification, \
    AutoModelForTokenClassification
import torch.nn as nn
import torch

model_name_dict = {"jp": "cl-tohoku/bert-base-japanese",
                   "tr": "dbmdz/bert-base-turkish-cased",
                   "hu": "hubert",
                   "fi": "TurkuNLP/bert-base-finnish-cased-v1",
                   "cs": "DeepPavlov/bert-base-bg-cs-pl-ru-cased"}


def load_bert_model(lang):
    model_name = model_name_dict[lang]
    if lang == "hu":
        model = BertForPreTraining.from_pretrained(model_name, from_tf=True,output_hidden_states=True)
    else:
        model = BertForTokenClassification.from_pretrained(model_name)
        model.classifier = nn.Identity()
    return model


lang = "jp"
model = load_bert_model(lang)
model_name = model_name_dict[lang]
# model.cls = nn.Identity()
# model = nn.Sequential(*list(model.children())[:-1])
tokenizer = AutoTokenizer.from_pretrained(model_name)
input = tokenizer.tokenize("Arda Akdemir  nereye gidiyorsunuz.")
input = torch.LongTensor([tokenizer.convert_tokens_to_ids(input)]).reshape(1,-1)

print(input.shape)
# bert_token_ids = torch.randint(0, 10, (3,12),dtype=torch.Long)
# print(bert_token_ids.shape)
attention_mask = torch.ones(*input.shape)
if lang == "hu":
    output = model(input, attention_mask)[2][-1]
else:
    output = model(input, attention_mask)[0]

print(input.shape)
print(output.shape)
assert output.shape == (input.shape[0],input.shape[1],768)