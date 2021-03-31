#!/bin/bash

model_folder=${1}
task_name=${2}
save_folder=${3}

langs=(czech turkish english finnish hungarian)
lang_prefs=("cs" "tr" "en" "fi" "hu")


#langs=(japanese  hungarian)
#lang_prefs=("jp"  "hu")

model_name=${task_name}


#for i in 0 1 2 3 4 5
for i in 0 1 2 3 4
do
  lang=${langs[$i]}
  lang_pref=${lang_prefs[$i]}
#  for type in fastext word2vec
  for type in bert mbert bert_en fastext word2vec random_init
  do
    echo "Running for " ${model_type}"  "${lang}"  "${type}
    #load_path = os.path.join(model_folder,"{}_{}_{}_best_{}_model.pkh".format(task,word_type,l_p,task_lower))
    lower_task="$(echo $task_name | tr '[A-Z]' '[a-z]')"
    load_path=$model_folder"/"${model_name}"_"${type}"_"${lang_pref}"_best_"${lower_task}"_model.pkh"
    echo "Loading model from: "${load_path}
    python jointtrainer_multilang.py --model_type ${model_name} --eval_mode ${task_name} --mode "predict" --word_embed_type ${type}   --lang ${lang_pref}  --save_dir $save_folder --load_model 1 --load_path $load_path
  done
done