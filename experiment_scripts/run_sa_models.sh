#!/bin/bash


langs=(turkish  english)
lang_prefs=("tr" "en")
model_type="sa"
load_folder="../sa_models"
save_dir="../"${model_type}"_experiment_results2903"
for i in 0 1
do
  lang=${langs[$i]}
  lang_pref=${lang_prefs[$i]}
#    for type in bert mbert bert_en fastext word2vec random_init
  for type in bert mbert bert_en fastext word2vec random_init
  do
    echo "Running for " ${model_type}"  "${lang}"  "${type}
    python sequence_trainer.py --mode predict  --word_embed_type ${type}  --lang ${lang_pref}  --save_folder $save_dir --load_folder ${load_folder}
  done
done

