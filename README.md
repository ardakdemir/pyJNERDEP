## Hierarchical Multi-task Learning of Dependency Parsing and Named Entity Recognition using Subword Contextual Embeddings

This repository contains the source code for replicating the experiments in our paper.
For the double-blind review we refrain from sharing any private information.

We also share the trained models using drive links together with all the data used for both tasks.


Training a hierarchical model with dependency parser as the low-level task.

***Note.*** Use --ner_train_file --ner_test_file --dep_train_file --dep_test_file flags to denote the paths.
```
    python jointtrainer.py --model_type DEPNER 
```

## Inference Mode

In order to replicate the results the users need to obtain the following files in addition to the source code  available in the repository:

- .pkh files containing the saved model parameters 
- *_config.json files for models that use different parameter combinations than default (NER_only model with lstm size 229)
- training and test files (training files are required to get the vocabularies for each task and pos tags)

We provide all these files using external links.

Example runs.

Example run for the DEP only model:



```
python jointtrainer.py --mode predict --load_model 1 --load_path dep_only_6788.pkh --ner_train_file traindev_pos.tsv --dep_train_file tr_imst_ud_traindev.conllu --model_type DEP --save_dir predicts 
```


```
python jointtrainer.py --mode predict --load_model 1 --load_path dep_only_6788.pkh --ner_train_file traindev_pos.tsv --dep_train_file tr_imst_ud_traindev.conllu --model_type NER --load_config 1 --config_file ner_9382_config.json  --save_dir predicts 
```

