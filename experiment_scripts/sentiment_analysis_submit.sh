/usr/local/bin/nosh
eval_interval=${1}
batch_size=${2}
repeat=${3}
exp_suf="$(date +"%m_%d_%H_%M")"
load_dir="../sa_model_save_dir_${exp_suf}/"
echo "EXPERIMENT WILL BE STORED IN ${load_dir}"
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G

langs=(czech hungarian japanese english finnish  turkish)
word_vec_prefs=("cs" "hu" "jp" "en" "fi" "tr")
domains=("movie" "twitter")

cd ~/parser/final_model

langs=(czech hungarian japanese english finnish  turkish)
lang_abr=("cs" "hu" "jp" "en" "fi" "tr")
domains=("movie" "twitter")

for i in 0 1 2 3 4 5
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
      for type in bert mbert bert_en fastext word2vec random_init
      do
        exp_file="sa_experiment_log_"${domain}"_"${lang}"_"${type}".json"
        echo $train_path" "${lang}" "${domain}
        singularity exec --nv  --writable ~/singularity/pt-cuda-tf-tr-ft python sequence_trainer.py  --sa_train_file ${train_path} --sa_dev_file ${dev_path} --sa_test_file ${test_path}  --exp_file ${exp_file} --word_embed_type ${type}  --repeat ${repeat} --lang $l --batch_size ${batch_size} --epochs 10 --save_folder $load_dir --eval_interval ${eval_interval}
      done
    fi
  done
done
