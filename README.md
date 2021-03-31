## A Comprehensive Analysis of Subword Contextual Embeddings for Languages with Rich Morphology

This repository contains the source code for replicating the experiments in our paper titled, "A Comprehensive Analysis of Subword Contextual Embeddings for Languages with Rich Morphology".




## Running on Colab

The easiest way to replicate the results and train new models is to use Google Colab.  
We provide a Colab notebook to replicate the results: "DLA_Submission_Notebook.ipynb".  


### Data dependencies

In order to run all the experiments without additional effort, the provided Colab notebook downloads all the data required to run the models:

- All datasets used for the experiments.
- All pretrained word embeddings (e.g., pretrained Word2Vec and FastText embeddings for all languages). (required for training only)
- All trained models. (required for testing only)

If you already have any of these downloaded, you can skip the steps inside the Colab notebook.  
Please follow the instructions inside the Colab notebook and make sure to change the variable values accordingly.

- Datasets are by default under "../../datasets"


## Meta info


Below are the required information for training new models and replicating the results:

Name mapping for each method :

    - Baseline : random_init
    - Word2Vec : word2vec
    - FastText : fastext
    - Bert English: bert_en
    - mBERT : mbert
    - BERT (language specific): bert

Name mapping for each language :

    - Turkish : tr
    - Finnish : fi
    - Hungarian : hu
    - Czech : cs
    - English : en
    - Japanese : jp





## Setup

To ease the work required for the setup, we provide a docker file that contains all the setting for running the source code. The docker image is available under : https://hub.docker.com/r/aakdemir/pytorch-cuda-tensorflow with the version 0.4 or latest.

### Using the docker image

Make sure that docker is installed and running on your local device. Check docker [homepage](https://docs.docker.com/docker-for-windows/install/) for details.

Next run the following code to mount required data inside your local machine:

```
docker run -it --rm -v ~/PATH_TO_SOURCE_CODE_FOLDER:/work aakdemir/pytorch-cuda-tensorflow:0.4

```

This code will download the docker image and start a container which mounts the local directory containing the source code to the /work directory inside the container.
Make sure that you obtain and place the datasets inside ~/PATH_TO_SOURCE_CODE_FOLDER so that container can have access to them, in addition to the source code shared in this folder.
By default, the code runs using GPU whenever available.

A more generic way of initiating the container is as follows:

```
docker run -it --rm -v [path_to_the_datasets_in_local]:[path_in_container_for_data] -v [path_to_the_source_code_inside_local]:[path_in_container_for_source_code] aakdemir/pytorch-cuda-tensorflow:0.4

```

The docker contains all the requirements and have cuda installed to allow running in GPU mode without trouble.

### Setup using pip

We highly recommend to create a virtual environment (virtualenv or anaconda etc.) for setup.
Assuming that you are inside a virtual environment run the following code from the source code directory:

```
pip3 install -r requirements.txt

```


## Training Mode


By default, the model expects the datasets to be under "../../datasets" (--data_folder). You can change the data root using that flag.
For each language the data splits are expected to follow the following conventions (all shared inside the datasets folder):

```
dep_train_name = os.path.join(self.args['data_folder'], "dep_{}_train.conllu".format(lang))
dep_dev_name = os.path.join(self.args['data_folder'], "dep_{}_dev.conllu".format(lang))
dep_test_name = os.path.join(self.args['data_folder'], "dep_{}_test.conllu".format(lang))
ner_train_name = os.path.join(self.args['data_folder'], "myner_{}-train.txt".format(lang))
ner_dev_name = os.path.join(self.args['data_folder'], "myner_{}-dev.txt".format(lang))
ner_test_name = os.path.join(self.args['data_folder'], "myner_{}-test.txt".format(lang))
```

where lang is the absolute lower case language name, e.g., english for English (en).

### Architectural Parameters

Below are some important parameters and their descriptions.
- --word_only : If set model does not use the POS and casing embeddings.
- --model_type : Categorical parameter with five values.
    - DEP : DEP only model
    - NER : NER only model
    - FLAT : FLAT model for multi-task learning
- --save_dir : denotes the directory to save all the training related files and log files. Important information are logged by default to jointtraining.log file.


***Note*** The models require a relatively large memory size and it is not suitable for running the models on many local devices. If you would like to run on your local device be sure to train a smaller version of the models by changing the following parameters :

--batch_size

--lstm_hidden

--biaffine_hidden

--lstm_layers

#### Multitask Model

Training a multitask model for NER and DEP multi-task learning:

```
python jointtrainer_multilang.py --model_type FLAT  --word_embed_type fastext  --lang tr --save_dir ../flat_turkish_fastext_savedir

```

The above code would train a multi-task learning model for the Turkish language with the FastText embeddings, and store everything under "../flat_turkish_fastext_savedir".



#### Single Models (DEP_only and NER_only)
These two models are conventional DNN based models attempting a single task

Example for Japanese DEP with bert

```
python jointtrainer_multilang.py --model_type DEP  --word_embed_type bert  --lang jp
```

Example for Hungarian NER with mbert

```
python jointtrainer_multilang.py --model_type NER --word_embed_type mbert  --lang hu
```


## Analysis Codes

In the submitted paper, we give an analysis of the sentences that 1) other methods failed, and mBERT succeeded, and 2) other methods succeded, and mBERT failed.

All of the code to replicate the results of this analysis is contained inside the ipython Notebook :

```
error_analysis_scripts.ipynb
```


To be able to replicate the results, we also share the results of each model for all languages for the NER task through the drive link : [link](https://drive.google.com/drive/folders/1hgM4m4KGspk4kvzQqiAF6bidWi0vAgB4?usp=sharing)
This drive folder contains results for 36 cases for the NER task (6 models (baseline, word2vec, fastext, bert_en, mbert, and bert) for 6 languages).


Same files are also shared under the "test_outs" folder, inside the zip file for the Deep Learning Applications, Vol3 submission.



In addition to get the unknown-rare frequency analysis, the training files should also be stored in the folder "datasets" under the current working directory. For example for the Czech Language,

```
datasets/myner_czech-train.txt
```


## Sentiment Analysis Task

Domains:

- twitter
- movie

Languages:

- tr: Turkish
- en: English

We conducted experiments in three settings: Movie+Twitter for Turkish, and Movie for English.

To run the experiments for the sentiment analysis task use the below examples.

English movie using word2vec:
```
python sequence_trainer.py --lang en --eval_interval -1 --batch_size 150 --word_embed_type word2vec --sa_train_file ../../datasets/sa_movie_english-train.json --domain movie   --sa_dev_file ../../datasets/sa_movie_english-dev.json  --sa_test_file ../../datasets/sa_movie_english-test.json

```

Turkish examples:

```
python sequence_trainer.py --lang tr --eval_interval -1 --batch_size 100 --word_embed_type bert_en --save_folder sa_tr_movie_berten --domain movie --sa_train_file  ../../datasets/sa_movie_turkish-train.json  --sa_dev_file ../../datasets/sa_movie_turkish-dev.json  --sa_test_file ../../datasets/sa_movie_turkish-test.json

python sequence_trainer.py --lang tr --eval_interval -1 --batch_size 150 --word_embed_type mbert --save_folder sa_tr_twitter_mbert --domain twitter --sa_train_file  ../../datasets/sa_twitter_turkish-train.json  --sa_dev_file ../../datasets/sa_twitter_turkish-dev.json  --sa_test_file ../../datasets/sa_twitter_turkish-test.json

```


All sentiment analysis results would be stored inside the folder denoted with "--save_folder".
