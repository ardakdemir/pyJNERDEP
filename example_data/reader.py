import sys
import codecs

from transformers import AutoTokenizer, AutoModelForMaskedLM

word_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
char_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

sents = "為せば成る為さねば成らぬ\n麩菓子は、麩を主材料とした日本の菓子。\nSen benim kim olduğumu biliyor musun?\nThe more you use it the more you need it."

l = sents.split("\n")[2]
#
# with codecs.open("example_sents.txt", "r", encoding="utf-8") as f:
#     l = f.read().split("\n")[2]

word_tokens = word_tokenizer.tokenize(l)
char_tokens = char_tokenizer.tokenize(l)

with open("test.txt", "w", encoding="utf-8") as o:
    o.write(l)
    o.write("\n")
    o.write(" ".join(word_tokens))
    o.write("\n")
    o.write(" ".join(char_tokens))
