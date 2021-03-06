## Subword Contextual Embeddings for Languages with Rich Morphology

This repository contains the source code for replicating the experiments in our paper, submitted for review for EMNLP 2020, Long papers.
For the double-blind review we refrain from sharing any private information.

We also share the trained models using drive links together with all the data used for both tasks.


The entry point to run the main experiments  is the jointtrainer_multilang.py file. It controls both training, prediction and evaluation steps through the command line arguments.



## Datasets and Pretrained Models

In order to train your own models using the same datasets or replicate the results, we share the trained models reported on the paper together with the datasets using external links.
Below are the required documents for training new models and replicating the results:

- Pretrained models : We provide all the models necessary to replicate the results reported in our paper. ".pkh" files containing the saved model parameters [link](https://drive.google.com/drive/folders/1I2YSW6Vzw6CrIgJlKfIm3uFod1ETd7SR?usp=sharing).

Below we give the name mappings to get the right model file name for each configuration.

Name mapping for each method :

    - Baseline : random_init
    - Word2Vec : word2vec
    - FastText : fastext
    - mBERT : bert

Name mapping for each language :

    - Turkish : tr
    - Finnish : fi
    - Hungarian : hu
    - Czech : cs


For example, for the Word2Vec-DEP-Finnish combination the relevant model file is :

```
DEP_word2vec_fi_best_dep_model.pkh
```


- *_config.json files for models that use different parameter combinations than default (NER_only model with lstm size 229 for the Turkish language) [link](https://drive.google.com/drive/folders/1I2YSW6Vzw6CrIgJlKfIm3uFod1ETd7SR?usp=sharing)

### Datasets


Training, development and test files for each language are under the 'ner_datasets' and 'dep_datasets' folders of the link(training files are required to get the vocabularies for each task and pos tags) [link](https://drive.google.com/drive/folders/1ugT4tk8FlxxOQdjp4m9pXc_6_Xhdlo2-?usp=sharing)


In addition we provide sample data under:

```
example_data/
```

which contains small portions of train and test sets for both tasks, for the Turkish language.




## Setup

To ease the work required to for doing the setup, we provide a docker file that contains all the setting for running the source code. The docker image is available under : https://hub.docker.com/r/aakdemir/pytorch-cuda/tags with the version 0.2.

Start by cloning the source code (hier2 branch) to your machine (or by downloading and unzipping it).

### Using the docker image

Make sure that docker is installed and running on your local device. Check docker [homepage](https://docs.docker.com/docker-for-windows/install/) for details.

Next run the following code to mount required data inside your local machine:

```
docker run -it --rm -v ~/pyjNERDEP:/work aakdemir/pytorch-cuda:0.2

```

This code will download the docker image and start a container which mounts the local directory ~/pyJNERDEP to the /work directory inside the container.
Make sure that you obtain and place the datasets inside ~/pyJNERDEP so that container can have access to them, in addition to the source code shared in this folder.
By default, the code runs using GPU whenever available.

A more generic way of initiating the container is as follows:

```
docker run -it --rm -v [path_to_the_datasets_in_local]:[path_in_container_for_data] -v [path_to_the_source_code_inside_local]:[path_in_container_for_source_code]  aakdemir/pytorch-cuda:0.2

```

The docker contains all the requirements and have cuda installed to allow running in GPU mode without trouble.

### Setup using pip

We highly recommend to create a virtual environment (virtualenv or anaconda etc.) for setup.
Assuming that you are inside a virtual environment run the following code from the source code directory:

```
pip3 install -r requirements.txt

```


## Training Mode

Training a hierarchical model with dependency parser as the low-level task.
By default all the files will be stored under the current directory.
Change the '--save_dir' parameters accordingly. All the information about the training process is logged into the file jointtraining.log by default.



By default, the model looks at '~/datasets' folder for all the datasets and the default setting is the 'FLAT' model where task specific components only share the common layer.

## Training-Evaluation Scripts

We used "singularity" [link](https://singularity.lbl.gov/docs-docker) to run our experiments on our cluster, which can be used with the docker image provided above. To run the training script you can build the singularity image using the following command:

```
    singularity build ~/singularity/pt-cuda-tf-ft-gn  aakdemir/pytorch-cuda:0.2
```

this will build a singularity image under "~/singularity".


To train models for each language (Czech, Hungarian, Finnish and Turkish) using each embedding method (Baseline, Word2Vec, FastText and mBERT) you can use the script under "experiment_scripts". Example run :

```
    bash experiment_scripts/joint_multilang_submit.sh NER

```

The above command will run the training and evaluation for the NER task, for each embedding method and each language, separately. Similarly for Dependency Parsing task:

```
    bash experiment_scripts/joint_multilang_submit.sh DEP

```

The evaluation scripts for NER and DEP tasks are :

```
 conll_eval.py
```

```
 parser/conll_ud_eval.py
```

the relevant functions from these scripts are called during validation/inference.


### Architectural Parameters

Below are some important parameters and their descriptions.
- --inner : If set to 1 the inner representation of the low-level component is fed as input to the high-level task (corresponds to the hier_repr setting explained in the paper). If set to 0, corresponds to hier_pred model. By default, it is 0.
- --soft : If set to 1 weighted average of embeddings are used as input to the high-level component (soft version of hier_pred). If set to 0 hard embedding variation is used.
- --relu : Binary parameter to determine whether to use relu activation to the HLSTM outputs.  
- --warmup : Denotes the number of epochs that the low-level task will be trained before starting the multi-task learning (warming up the low-level weights). Only used for hierarchical models (DEPNER and NERDEP).
- --model_type : Categorical parameter with five values.
    - DEPNER : DEP is the low-level task
    - NERDEP : NER is the low-level task
    - DEP : DEP only model
    - NER : NER only model
    - FLAT : FLAT model where task-specific components only share the three embeddings
- --save_dir : denotes the directory to save all the training related files and log files. Important information are logged by default to jointtraining.log file.


***Note.*** Use --ner_train_file --ner_test_file --ner_val_file --dep_val_file --dep_train_file --dep_test_file flags to denote the paths to the datasets. Input files must be in specific conll formats. Example data is included  under the example_data directory.

***Note2*** The models require a relatively large memory size and it is not suitable for running the models on many local devices. If you would like to run on your local device be sure to train a smaller version of the models by changing the following parameters :

--batch_size

--lstm_hidden

--biaffine_hidden

--lstm_layers

#### Multitask Model 1 (hier_repr)
Training a hierarchical model where the HLSTM output of the DEP-component is concatenated to the common layer output and given to the NER component (corresponds to DEP_low_repr and NER_high_repr models in the paper).

```
    python jointtrainer_multilang.py --model_type DEPNER
```

#### Multitask Model 2 (hier_pred)
Training a hierarchical model where the soft embeddings of the NER label predictions are concatenated to the common layer output (DEP_high_pred and NER_low_pred models in the paper). In addition warmup is set to 10, so that the NER component will be trained for 10 epochs before starting the multitask learning.

```
    python jointtrainer_multilang.py --model_type NERDEP --inner 0 --soft 1 --warmup 10
```


#### Multitask Model 3 (flat)

```
    python jointtrainer_multilang.py --model_type FLAT

```

#### Single Models (DEP_only and NER_only)
These two models are conventional DNN based models attempting a single task.

```
python jointtrainer_multilang.py --model_type DEP
```

```
python jointtrainer_multilang.py --model_type NER
```


## Inference Mode

In order to replicate the results the users need to obtain the following files in addition to the source code  available in the repository:

- .pkh files containing the saved model parameters [link](https://drive.google.com/drive/folders/1I2YSW6Vzw6CrIgJlKfIm3uFod1ETd7SR?usp=sharing)
- *_config.json files for models that use different parameter combinations than default (NER_only model with lstm size 229) [link](https://drive.google.com/drive/folders/1I2YSW6Vzw6CrIgJlKfIm3uFod1ETd7SR?usp=sharing)
- training and test files are under the 'ner_datasets' and 'dep_dataseets' folders of the link(training files are required to get the vocabularies for each task and pos tags) [link](https://drive.google.com/drive/folders/1ugT4tk8FlxxOQdjp4m9pXc_6_Xhdlo2-?usp=sharing)

We provide all these files using external links.

Example runs.

### Example run for the DEP only model:



```
python jointtrainer_multilang.py --mode predict --load_model 1 --load_path dep_only_6788.pkh --ner_train_file traindev_pos.tsv --dep_train_file tr_imst_ud_traindev.conllu --model_type DEP --save_dir predicts
```


### Example run using the config file that stores the experiment specific model configuration
```
python jointtrainer_multilang.py --mode predict --load_model 1 --load_path dep_only_6788.pkh --ner_train_file traindev_pos.tsv --dep_train_file tr_imst_ud_traindev.conllu --model_type NER --load_config 1 --config_file ner_9382_config.json  --save_dir predicts
```


## Analysis Codes

In the submitted paper, we give an analysis of the sentences that 1) other methods failed, and mBERT succeeded, and 2) other methods succeded, and mBERT failed.

All of the code to replicate the results of this analysis is contained inside the ipython Notebook :

```
error_analysis_scripts.ipynb
```


To be able to replicate the results, we also share the results of each model for all languages for the NER task through the drive link : [link](https://drive.google.com/drive/folders/1hgM4m4KGspk4kvzQqiAF6bidWi0vAgB4?usp=sharing)

This drive folder contains results for 16 cases for the NER task (4 models (baseline, word2vec, fasttext and mBERT) for 4 languages).

In addition to get the unknown-rare frequency analysis, the training files should also be downloaded and stored in the folder "datasets" under the current working directory. For example for the Czech Language,

```
datasets/myner_czech-train.txt
```
