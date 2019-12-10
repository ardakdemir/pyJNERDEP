## Hierarchical Multi-task Learning of Dependency Parsing and Named Entity Recognition using Subword Contextual Embeddings

This repository contains the source code for replicating the experiments in our paper, submitted for review for ACL 2020.
For the double-blind review we refrain from sharing any private information.

We also share the trained models using drive links together with all the data used for both tasks.


The entry point for the project is the jointtrainer.py which controls both training, prediction and evaluation steps through the arguments. 

## Datasets and Trained Models

In order to train your own models using the same datasets or replicate the results, we share the trained models reported on the paper together with the datasets using external links.
Below are the required documents for training new models and replicating the results: 

- .pkh files containing the saved model parameters [link](https://drive.google.com/drive/folders/1I2YSW6Vzw6CrIgJlKfIm3uFod1ETd7SR?usp=sharing)
- *_config.json files for models that use different parameter combinations than default (NER_only model with lstm size 229) [link](https://drive.google.com/drive/folders/1I2YSW6Vzw6CrIgJlKfIm3uFod1ETd7SR?usp=sharing)
- training and test files are under the 'ner_datasets' and 'dep_dataseets' folders of the link(training files are required to get the vocabularies for each task and pos tags) [link](https://drive.google.com/drive/folders/1ugT4tk8FlxxOQdjp4m9pXc_6_Xhdlo2-?usp=sharing)


## Setup

To ease the work required to for doing the setup, we provide a docker file that contains all the setting for running the source code. The docker image is available under : https://hub.docker.com/r/aakdemir/pytorch-cuda/tags with the version 0.1. 

Start by cloning the source code (hier2 branch) to your machine (or by downloading and unzipping it).

### Using the docker image

Make sure that docker is installed and running on your local device. Check docker [homepage](https://docs.docker.com/docker-for-windows/install/) for details.

Next run the following code to mount /tmp/data of your local machine: 

```
docker run -it --rm -v ~/pjNERDEP:/work aakdemir/pytorch-cuda:0.1

This code will download the docker image and start a container which mounts the local directory ~/pyJNERDEP to the /work directory inside the container.
Make sure that you obtain and place the datasets inside ~/pyJNERDEP so that container can have access to them.
By default, the code runs using GPU whenever available.

A more generic way of initiating the container is as follows:

```
docker run -it --rm -v [path_to_the_datasets_in_local]:[path_in_container_for_data] -v [path_to_the_source_code_inside_local]:[path_in_container_for_source_code]  aakdemir/pytorch-cuda:0.1
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
    python jointtrainer.py --model_type DEPNER 
```

#### Multitask Model 2 (hier_pred)
Training a hierarchical model where the soft embeddings of the NER label predictions are concatenated to the common layer output (DEP_high_pred and NER_low_pred models in the paper). In addition warmup is set to 10, so that the NER component will be trained for 10 epochs before starting the multitask learning.

```
    python jointtrainer.py --model_type NERDEP --inner 0 --soft 1 --warmup 10
```


#### Multitask Model 3 (flat)

```
    python jointtrainer.py --model_type FLAT 
   
```

#### Single Models (DEP_only and NER_only)
These two models are conventional DNN based models attempting a single task.

```
python jointtrainer.py --model_type DEP
```

```
python jointtrainer.py --model_type NER
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
python jointtrainer.py --mode predict --load_model 1 --load_path dep_only_6788.pkh --ner_train_file traindev_pos.tsv --dep_train_file tr_imst_ud_traindev.conllu --model_type DEP --save_dir predicts 
```


### Example run using the config file that stores the experiment specific model configuration
```
python jointtrainer.py --mode predict --load_model 1 --load_path dep_only_6788.pkh --ner_train_file traindev_pos.tsv --dep_train_file tr_imst_ud_traindev.conllu --model_type NER --load_config 1 --config_file ner_9382_config.json  --save_dir predicts 
```



