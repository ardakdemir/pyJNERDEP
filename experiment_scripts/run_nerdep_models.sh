#!/bin/bash

langs=(czech turkish  japanese english finnish hungarian)
lang_prefs=("cs" "tr"  "jp" "en" "fi" "hu")
for model_type in NER DEP FLAT
do
  save_dir="../"$model_type"_experiment_results"
  for i in 0 1 2 3 4 5
  do
    lang=${langs[$i]}
    lang_pref=${lang_prefs[$i]}
    for type in bert mbert bert_en fastext word2vec random_init
    do
      model_folder="../"${model_type}"_"${type}"_models"
      echo "Downloading the stored models to: "$model_folder
      lower_model="$(echo $model_type | tr '[A-Z]' '[a-z]')"
      python download_storedmodels.py --key ${model_type} --save_folder ${model_folder}
      load_path=$model_folder"/"${model_type}"_"${type}"_"${lang}"_best_"$lower_model"_model.pkh"
      echo "Running for " ${model_type}"  "${lang}"  "${type}
      echo "Model will be loaded from: "$load_path
      echo "Results will be saved in: "${save_dir}
      python jointtrainer_multilang.py --mode "predict" --model_type ${model_type}  --load_model 1 --load_path ${load_path} --word_embed_type ${type}   --lang ${lang_pref} --save_dir $save_dir
    done
  done
done
