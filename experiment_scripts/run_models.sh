#!/bin/bash

repeat=1
eval_interval=1
epoch=1
langs=(czech turkish  japanese english finnish hungarian)
word_vec_prefs=("cs" "tr"  "jp" "en" "fi" "hu")
for model_type in NER DEP FLAT
do
  save_dir=$model_type"_experiment_results"
  for i in 0 1 2 3 4 5
  do
    lang=${langs[$i]}
    wvec_pref=${word_vec_prefs[$i]}
    for type in bert mbert bert_en fastext word2vec random_init
    do
      echo "Running for "${lang}"  "${type}"  pref  "${wvec_pref}
      python jointtrainer_multilang.py --model_type ${model_type}  --word_embed_type ${type}    --log_file ${model_type}_log_${type}_${lang}_log.txt --lang ${wvec_pref}  --epochs ${epoch} --save_dir $save_dir --repeat ${repeat} --eval_interval ${eval_interval}
    done
  done
done
