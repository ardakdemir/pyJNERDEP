#!/bin/bash


langs=(turkish  english)
lang_prefs=("tr" "en")

lang_pref=${1} #tr or en
lang=${2}
domain=${3} #movie or twitter
word_type=${4}
model_folder=${5}
save_dir=${6}

model_type="sa"
exp_key=${domain}"_"${lang}
echo "Downloading trained Sentiment Analysis models to: "${model_folder}
python download_storedmodels.py --model_type SA --word_type ${exp_key} --save_folder $model_folder

echo "Running for " ${model_type}"  "${lang}"  "${word_type}
python sequence_trainer.py --mode predict  --domain ${domain} --word_embed_type ${word_type}  --lang ${lang_pref}  --save_folder $save_dir --load_folder ${model_folder}
