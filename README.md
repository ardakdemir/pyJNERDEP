## Hierarchical Multi-task Learning of Dependency Parsing and Named Entity Recognition using Subword Contextual Embeddings

This repository contains the source code for replicating the experiments in our paper.
For the double-blind review we refrain from sharing any private information.

We also share the trained models using drive links together with all the data used for both tasks.


Training a hierarchical model with dependency parser as the low-level task.

***Note.*** Use --ner_train_file --ner_test_file --dep_train_file --dep_test_file flags to denote the paths.
```
    python jointtrainer.py --model_type DEPNER
```
