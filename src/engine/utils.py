import pandas as pd
from sklearn import model_selection
import yaml
import numpy as np
import torch
import os
from distutils.dir_util import copy_tree
from shutil import copy2
from tensorboardX import SummaryWriter

# create folds (normally won't need to be called again)
def folds_generator(nr_folds):
    df = pd.read_csv("../data/raw/train.csv")
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=nr_folds)
    
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    'train_folds_{}.csv'.format(nr_folds)
    df.to_csv('train_folds_{}.csv'.format(nr_folds), index=False)


def load_yaml(file_name):
    with open(file_name, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)

    return config


class EpochManager:
    def __init__(self, config, fold, patience=7, mode="max", delta=0.0001):
        self.model_name = config.get_model_name()+'_'+str(fold)
        self.yaml_name = config.get_yaml_name()
        self.script_name = config.get_script_name()
        self.patience = patience
        self.counter = 0
        self.epoch_n = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.writer = None
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, train_loss, valid_loss, model, model_path):
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.writer is None:
            self.writer = \
                SummaryWriter(os.path.join("../logs", self.model_name))
        self.writer.add_scalar('train/loss', train_loss, self.epoch_n)
        self.writer.add_scalar('valid/loss', valid_loss, self.epoch_n)
        self.writer.add_scalar('valid/auc', epoch_score, self.epoch_n)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EpochManager counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0
        self.epoch_n += 1

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.val_score, epoch_score
                )
            )
            state = {
                 "name_script": self.script_name,
                "epoch": self.epoch_n,
                "auc": epoch_score,
                "train_loss": self.train_loss,
                "valid_loss": self.valid_loss,
                "state_dict": model.state_dict
            }

            if not os.path.exists(os.path.join(model_path, self.model_name)):
                os.makedirs(os.path.join(model_path, self.model_name))
                os.makedirs(os.path.join(model_path, self.model_name, "src"))
            torch.save(state, os.path.join(model_path, self.model_name, self.model_name + ".pth"))
            copy_tree(os.path.abspath("../src"), os.path.abspath(os.path.join(model_path, self.model_name, "src")))
            # copy2(os.path.join(os.path.abspath("../models/yamls/"), self.yaml_name), os.path.join(model_path, self.model_name))
        self.val_score = epoch_score


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking/logging current loss only.
    Source: https://github.com/abhishekkrthakur/wtfml/blob/master/wtfml/utils/average_meter.py
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count