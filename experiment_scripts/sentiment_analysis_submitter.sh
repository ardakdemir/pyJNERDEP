#!/bin/bash
eval_interval=${1}
batch_size=${2}
exp_suf="$(date +"%m_%d_%H_%M")"
load_dir="../sa_model_save_dir_${exp_suf}/"
echo "EXPERIMENT WILL BE STORED IN ${load_dir}"
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G

cd ~/parser/final_model

#langs=(czech hungarian japanese english finnish  turkish)
langs=("english" "turkish")

lang_abr=("en" "tr")
domains=("movie" "twitter")

for i in 0 1
do
  lang=${langs[$i]}
  l=${lang_abr[$i]}
  for j in 0 1
  do
    domain=${domains[$j]}
    echo "Language: "${lang}" Domain: "${domain}
    train_path="../../datasets/sa_"${domain}"_"${lang}"-train.json"
    dev_path="../../datasets/sa_"${domain}"_"${lang}"-dev.json"
    test_path="../../datasets/sa_"${domain}"_"${lang}"-test.json"
    if [ -f ${train_path} ]
    then
      for type in random_init wrd2vec fastext bert_en mbert bert
      do
        exp_name=${domain}"_"${lang}"_"${type}"_"${exp_suf}
        exp_file="sa_experiment_log_"${domain}"_"${lang}"_"${type}".json"
        echo $train_path" "${lang}" "${domain}
        qsub -N exp_name experiment_scripts/sentiment_analysis_singlesubmit.sh ${eval_interval} ${batch_size} ${domain} ${l} ${type} ${load_dir}
      done
    fi
  done
done
