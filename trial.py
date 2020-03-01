from torch import nn 
from pytorch_transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
tokenizer.add_tokens(['[SOS]','[EOS]'])

tokens = tokenizer.tokenize("[PAD]")

emb = nn.Embedding(100,200)

loss = torch.tensor([100])-torch.sum(emb(torch.tensor([1])))
print(loss)

loss.backward()
