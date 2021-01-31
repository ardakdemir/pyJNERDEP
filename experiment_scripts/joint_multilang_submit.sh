/usr/local/bin/nosh
model_type=${1}
repeat=${2}
eval_interval=${3}
exp_pref=${4}
exp_suf="$(date +"%m_%d_%H_%M")"
epoch=10
load_dir="../"${exp_pref}"_"${model_type}"_"${exp_suf}"/"
#load_dir="../model_save_dir_DEP_03_14_21_26/"
echo "EXPERIMENT WILL BE STORED IN ${load_dir}"
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G

langs=(czech turkish  japanese english finnish hungarian)
word_vec_prefs=("cs" "tr"  "jp" "en" "fi" "hu" )


cd ~/parser/final_model

for i in 0 1 2 3 4 5
do

    lang=${langs[$i]}

    wvec_pref=${word_vec_prefs[$i]}

#    for type in bert mbert bert_en fastext word2vec random_init
#    for type in fastext word2vec random_init
    for type in fastext word2vec random_init
    do
        echo "Running for "${lang}"  "${type}"  pref  "${wvec_pref}

        singularity exec --nv  --writable ~/singularity/pt-cuda-tf-tr-ft python jointtrainer_multilang.py --model_type ${model_type}  --word_embed_type ${type}    --log_file ${model_type}_log_${type}_${lang}_log.txt --lang ${wvec_pref}  --epochs ${epoch} --save_dir $load_dir --repeat ${repeat} --eval_interval ${eval_interval} 
#        if [ ! ${model_type} == 'NER' ]
#        then
#            singularity exec --nv ~/singularity/pt-cuda-tf-ft-gn python jointtrainer_multilang.py --model_type ${model_type}  --word_embed_type ${type}   --log_file ${model_type}_predlog_${type}_${lang} --lang ${wvec_pref} --mode predict --load_path  ${load_dir}${model_type}'_'$type'_'$wvec_pref'_best_dep_model.pkh' --load_model 1 --save_dir ${load_dir} >> ${load_dir}'dep_result_for_'${model_type}'_'${type}'_'${lang}'.txt' --batch_size 100  --word_embed_dim ${word_embed_dim}
#        fi
#        if [ ! ${model_type} == 'DEP' ]
#        then
#            singularity exec --nv ~/singularity/pt-cuda-tf-ft-gn python jointtrainer_multilang.py --model_type ${model_type}  --word_embed_type ${type}    --log_file ${model_type}_predlog_${type}_${lang} --lang ${wvec_pref} --mode predict --load_path  ${load_dir}${model_type}'_'${type}'_'$wvec_pref'_best_ner_model.pkh' --load_model 1 --save_dir ${load_dir} >> ${load_dir}'ner_result_for_'${model_type}'_'${type}'_'${lang}'.txt'  --batch_size 100  --word_embed_dim ${word_embed_dim}
#        fi

    done
done
