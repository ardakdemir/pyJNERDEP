/usr/local/bin/nosh
word_vec_folder='../word_vecs/'
train_dataset=~/datasets/arabic_data/sentsplitted_conll_all.test.features.txt
val_dataset=~/datasets/arabic_data/sentsplitted_conll_all.dev.features.txt
test_dataset=~/datasets/arabic_data/sentsplitted_conll_all.test.features.txt
batch_size=100
model_type=${1}
word_embed_dim=768
exp_suf="$(date +"%m_%d_%H_%M")"
load_dir="../model_save_dir_${model_type}_${exp_suf}/"
#load_dir="../model_save_dir_DEP_03_14_21_26/"
echo "EXPERIMENT WILL BE STORED IN ${load_dir}"
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N ner_multilang

langs=(czech hungarian finnish  turkish)
word_vec_prefs=(cs hu fi tr)


cd ~/parser/final_model

for i in 0 1 2 3
do

    lang=${langs[$i]}

    wvec_pref=${word_vec_prefs[$i]}
    train_dataset=../../datasets/myner_${lang}-train.txt
    val_dataset=../../datasets/myner_${lang}-test.txt

    for type in bert random_init fastext word2vec
    do
        echo "Running for "${lang}"  "${type}"  pref  "${wvec_pref}
        singularity exec --nv ~/singularity/pt-cuda-tf-ft-gn python jointtrainer_multilang.py --model_type ${model_type}  --word_embed_type ${type}    --log_file ${model_type}_log_${type}_${lang} --lang ${wvec_pref} --batch_size 100 --epochs 30 --save_dir $load_dir --word_embed_dim ${word_embed_dim}
        if [ ! ${model_type} == 'NER' ]
        then
            singularity exec --nv ~/singularity/pt-cuda-tf-ft-gn python jointtrainer_multilang.py --model_type ${model_type}  --word_embed_type ${type}   --log_file ${model_type}_predlog_${type}_${lang} --lang ${wvec_pref} --mode predict --load_path  ${load_dir}${model_type}'_'$type'_'$wvec_pref'_best_dep_model.pkh' --load_model 1 --save_dir ${load_dir} >> ${load_dir}'dep_result_for_'${model_type}'_'${type}'_'${lang}'.txt' --batch_size 100  --word_embed_dim ${word_embed_dim}
        fi
        if [ ! ${model_type} == 'DEP' ]
        then
            singularity exec --nv ~/singularity/pt-cuda-tf-ft-gn python jointtrainer_multilang.py --model_type ${model_type}  --word_embed_type ${type}    --log_file ${model_type}_predlog_${type}_${lang} --lang ${wvec_pref} --mode predict --load_path  ${load_dir}${model_type}'_'${type}'_'$wvec_pref'_best_ner_model.pkh' --load_model 1 --save_dir ${load_dir} >> ${load_dir}'ner_result_for_'${model_type}'_'${type}'_'${lang}'.txt'  --batch_size 100  --word_embed_dim ${word_embed_dim}
        fi

    done
done
