import os

files = ["ner_japanese-dev.txt", "ner_japanese-test.txt", "ner_japanese-train.txt"]

for f in files:
    with open(f, "r", encoding="utf-8")as ff:
        sents = ff.read().split("\n\n")
        print(len(sents[0]))
        new_sents = "\n\n".join(["\n".join(["\t".join([x.split()[0], "_", "_", x.split()[1]]) for x in sent.split("\n") if len(x.split())>1]) for sent in sents if len(sent)>1])
        with open("my"+f,"w",encoding="utf-8") as o:
            o.write(new_sents)