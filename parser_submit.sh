/usr/local/bin/nosh
#$ -cwd
#$ -l os7,v100=1,s_vmem=100G,mem_req=100G
#$ -N parser_batch -e error -o stdout
cd parser/
singularity exec --nv ~/singularity/pytorch-cuda-0.1.simg python parsetrainer.py
