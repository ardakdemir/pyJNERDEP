import os
import subprocess

word2vec_dict = {"jp": "../word_vecs/jp/jp.bin",
                 "tr": "../word_vecs/tr/tr.bin",
                 "hu": "../word_vecs/hu/hu.bin",
                 "en": "../word_vecs/en/en.txt",
                 "fi": "../word_vecs/fi/fi.bin",
                 "cs": "../word_vecs/cs/cs.txt"}

root = "parser/final_model"
save_folder = "word_vecs"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
for l,p in word2vec.items():
    dest = os.path.join(save_folder,l)
    if not os.path.exists(dest):
        os.makedirs(dest)
    cmd = "scp hgc:~/{}/{} {}".format(root,p,dest)
    subprocess.call(cmd,shell=True)
    