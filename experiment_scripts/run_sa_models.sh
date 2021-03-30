#!/bin/bash


langs=(turkish  english)
lang_prefs=("tr" "en")

lang_pref=${1} #tr or en
lang=${2}
domain=${3} #movie or twitter
model_folder=${4}
save_dir=${5}

model_type="sa"
exp_key=${domain}"_"${lang}
echo "Downloading trained Sentiment Analysis models to: "${model_folder}
python download_storedmodels.py --model_type SA --word_type ${exp_key} --save_folder $model_folder

# for type in bert mbert bert_en fastext word2vec random_init
# for type in bert mbert bert_en fastext word2vec random_init
#for type in bert mbert bert_en fastext word2vec random_init
echo "Running for " ${model_type}"  "${lang}"  "${type}
python sequence_trainer.py --mode predict  --word_embed_type ${type}  --lang ${lang_pref}  --save_folder $save_dir --load_folder ${model_folder}
