import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import time
from tqdm import tqdm
import argparse

import os
import copy
from transformers import AutoTokenizer, AutoModel, BertForPreTraining, BertForTokenClassification
from sequence_classifier import SequenceClassifier
from sareader import SentReader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name_dict = {"jp": "cl-tohoku/bert-base-japanese",
                   "tr": "dbmdz/bert-base-turkish-cased",
                   "hu": "/home/aakdemir/bert_models/hubert",
                   "fi": "TurkuNLP/bert-base-finnish-cased-v1",
                   "cs": "DeepPavlov/bert-base-bg-cs-pl-ru-cased",
                   "en": "bert-base-cased",
                   "mbert": "bert-base-multilingual-cased",
                   "bert_en": "bert-base-cased"}


def init_tokenizer(lang, model_type):
    if model_type in ["mbert", "bert_en"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name_dict[model_type])
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_dict[lang])
    return tokenizer


def read_config(args):
    config_file = args['config_file']
    with open(config_file) as json_file:
        data = json.load(json_file)
        print(data)
        for d in data:
            args[d] = data[d]
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sa_train_file', type=str, default='../../datasets/sa_twitter_turkish-train.json',
                        help='training file for sa')
    parser.add_argument('--sa_dev_file', type=str, default='../../datasets/sa_twitter_turkish-dev.json',
                        help='validation file for sa')
    parser.add_argument('--sa_test_file', type=str, default='../../datasets/sa_twitter_turkish-test.json',
                        help='test file for sa')
    parser.add_argument('--sa_output_file', type=str, default="sa_out.txt",
                        help='Output file for named entity recognition')
    parser.add_argument('--config_file', type=str, default='config.json', help='Output file name in conll bio format')
    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--load_config', default=0, type=int)
    parser.add_argument('--lang', default='tr', type=str, help='Language', choices=['en', 'jp', 'tr', 'cs', 'fi', 'hu'])
    parser.add_argument('--word_embed_type', default='bert',
                        choices=["mbert", "bert_en", 'bert', 'random_init', 'fastext', 'word2vec'],
                        help='Word embedding type to be used')

    parser.add_argument('--fix_embed', default=False, action='store_true', help='Word embedding type to be used')

    parser.add_argument('--lstm_layers', type=int, default=3)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=-1)

    parser.add_argument('--word_drop', type=float, default=0.3)
    parser.add_argument('--embed_drop', type=float, default=0.3)
    parser.add_argument('--lstm_drop', type=float, default=0.3)
    parser.add_argument('--crf_drop', type=float, default=0.3)
    parser.add_argument('--parser_drop', type=float, default=0.3)

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--embed_lr', type=float, default=0.015, help='Learning rate for embeddiing')
    parser.add_argument('--dep_lr', type=float, default=0.0015, help='Learning rate dependency lstm')
    parser.add_argument('--lr_decay', type=float, default=0.6, help='Learning rate decay')
    parser.add_argument('--min_lr', type=float, default=2e-6, help='minimum value for learning rate')
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--model_save_name', type=str, default='best_sa_model.pkh', help="File name to save the model")
    parser.add_argument('--save_folder', type=str, default='../sa_savedir', help="Folder to save files")
    parser.add_argument('--exp_file', type=str, default='sa_experiment_log.json', help="File to store exp details")
    parser.add_argument('--load_model', type=int, default=0, help='Binary for loading previous model')
    parser.add_argument('--load_path', type=str, default='best_joint_model.pkh', help="File name to load the model")

    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    vars(args)['device'] = device
    args = vars(args)
    if args['load_config'] == 1:
        args = read_config(args)
    return args


def evaluate(model, dataset):
    dataset.for_eval = True
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    total = 0
    eval_loss = 0
    for x in tqdm(range(len(dataset)), desc="Evaluation"):
        with torch.no_grad():
            tokens, tok_inds, bert_batch_after_padding, data_tuple = dataset[x]
            labels = data_tuple[3]
            preds, loss = model.predict(dataset[x])
            preds = preds.detach().cpu().numpy()
            eval_loss += loss.item()
            for l, p in zip(labels, preds):
                total += 1
                if l == p:
                    if l == 1:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if l == 1:
                        fn += 1
                    else:
                        fp += 1
    acc = (tp + tn) / total
    recall = tp / (fn + tp)
    precision = tp / (fp + tp)
    f1 = (2 * recall * precision) / (precision + recall)
    print("TP: {} FP: {} FN: {} TN: {} === Acc: {} === Loss: {}".format(tp, fp, fn, tn, acc, eval_loss))
    return acc, f1, eval_loss


def train():
    args = parse_args()
    lang = args["lang"]
    model_type = args["word_embed_type"]
    save_folder = args["save_folder"]
    exp_file = args["exp_file"]

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    tokenizer = init_tokenizer(lang, model_type)

    file_map = {"train": args["sa_train_file"],
                "dev": args["sa_dev_file"],
                "test": args["sa_test_file"]}
    print(file_map)
    datasets = {f: SentReader(file_map[f], batch_size=args["batch_size"], tokenizer=tokenizer) for f in file_map}
    num_cats = len(datasets["train"].label_vocab.w2ind)


    for x in ["dev", "test"]:
        datasets[x].word_vocab.w2ind = datasets["train"].word_vocab.w2ind

    word_vocab = datasets["train"].word_vocab
    seq_classifier = SequenceClassifier(lang, word_vocab, model_type, num_cats, device)

    seq_classifier.train()
    seq_classifier.to(device)

    eval_interval = args["eval_interval"]
    epochs = args["epochs"]
    epochs_losses, accs, f1s, losses = [], [], [], []
    best_f1 = 0
    begin = time.time()
    for e in tqdm(range(epochs), desc="Epoch"):
        total_loss = 0
        acc = 0
        seq_classifier.train()
        for i in tqdm(range(eval_interval), "training"):
            seq_classifier.zero_grad()
            data = datasets["train"][i]
            tokens, tok_inds, bert_batch_after_padding, data_tuple = data
            labels = data_tuple[3]
            class_logits = seq_classifier(data)
            print("Logit shape: {} label shape: {}".format(class_logits.shape, labels.shape))
            loss = seq_classifier.loss(class_logits, labels)
            total_loss += loss.item()
            print("Loss: {}".format(loss))
            loss.backward()
            seq_classifier.optimizer_step()
            if i % 100 == 99:
                aver_loss = total_loss / (i + 1)
                print("Average loss at {} steps: {}".format(i + 1, aver_loss))
        epochs_losses.append(total_loss / eval_interval)
        print("Evaluating the model")
        seq_classifier.eval()
        acc, f1, loss = evaluate(seq_classifier, datasets["dev"])
        accs.append(round(acc,3))
        f1s.append(round(f1,3))
        losses.append(round(loss,3))
        if f1 > best_f1:
            best_model_weights = seq_classifier.state_dict()
            best_f1 = f1

    end = time.time()
    train_time = round(end-begin)
    print("Epoch train losses ", epochs_losses)
    print("Accuracies ", accs)
    print("F1s ", f1s)
    print("Eval losses ", losses)

    print("Evaluating on test")
    seq_classifier.load_state_dict(best_model_weights)
    acc, f1, loss = evaluate(seq_classifier, datasets["test"])
    print("=== Test results === \n Acc:\t{}\nF1\t{}\nLoss\t{}\n".format(acc, f1, loss))
    exp_log = {"dev_acc": accs,
               "dev_f1": f1,
               "dev_loss": losses,
               "test_acc": round(acc,3),
               "test_f1": round(f1,3),
               "test_loss": round(loss,3),
               "lang": lang,
               "word_embed_type": model_type,
               "test_file": file_map["test"],
               "train_file": file_map["train"]}

    exp_save_path = os.path.join(save_folder, exp_file)
    with open(exp_save_path, "w") as o:
        json.dump(exp_log,o)

if __name__ == "__main__":
    train()
