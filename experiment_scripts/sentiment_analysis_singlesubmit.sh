/usr/local/bin/nosh

eval_interval=${1}
batch_size=${2}
domain=${3}
l=${4}
lang=${5}
type=${6}
load_dir=${7}

exp_suf="$(date +"%m_%d_%H_%M")"
echo "EXPERIMENT WILL BE STORED IN ${load_dir}"
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
train_path="../../datasets/sa_"${domain}"_"${lang}"-train.json"
dev_path="../../datasets/sa_"${domain}"_"${lang}"-dev.json"
test_path="../../datasets/sa_"${domain}"_"${lang}"-test.json"
if [ -f ${train_path} ]
then
singularity exec --nv  --writable ~/singularity/pt-cuda-tf-tr-ft python sequence_trainer.py  --domain ${domain} --sa_train_file ${train_path} --sa_dev_file ${dev_path} --sa_test_file ${test_path}   --word_embed_type ${type} --lang $l --batch_size ${batch_size} --epochs 10 --save_folder $load_dir --eval_interval ${eval_interval}
fi