#!/bin/bash

model_folder=${1}
save_dir=
lower_model="$(echo $model_type | tr '[A-Z]' '[a-z]')"
load_path=$model_folder"/"${model_type}"_"${type}"_"${lang}"_best_"$lower_model"_model.pkh"
echo "Running for " ${model_type}"  "${lang}"  "${type}
echo "Model will be loaded from: "$load_path
echo "Results will be saved in: "${save_dir}
python jointtrainer_multilang.py --mode "predict" --model_type ${model_type}  --load_model 1 --load_path ${load_path} --word_embed_type ${type}   --lang ${lang_pref} --save_dir $save_dir
