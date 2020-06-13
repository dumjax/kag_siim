import os
import sys
import math
import torch
import albumentations
from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics
from sklearn import model_selection
from torch.nn import functional as F

import pretrainedmodels
import timm

from config import *
from utils import *

from .models.timm_model import TimmModel


# TODO: model-dependent
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class MyDataset(Dataset):
    def __init__(self, image_paths, genders, ages, targets, augmentations=None):
        self.image_paths = image_paths
        self.genders = [0 if age == 'male' else (1 if age == 'female' else 0.5) for age in ages]  # TODO: use sklearn

        # TODO: use sklearn for this stuff. Also: try categorical?
        self.ages = [float(age) for age in ages]
        median_age = np.median([a for a in self.ages if not math.isnan(a)])
        self.ages = np.array([age if not math.isnan(age) else median_age for age in self.ages])
        self.ages /= max(ages)

        self.targets = targets
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        target = self.targets[item]
        gender = self.genders[item]
        age = self.ages[item]
        image = np.array(image)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        img = torch.tensor(image, dtype=torch.float)
        gender = torch.tensor(gender, dtype=torch.float)
        age = torch.tensor(age, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)

        return img, gender, age, target


def get_loader(df, valid: bool) -> DataLoader:
    """
    df: contains the data about the images to put in the returned loader
    is_valid: whether this is a validation set (otherwise it's a training set)
    """
    if valid:
        augmentations = albumentations.Compose(
            [
                albumentations.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True)
            ])
    else:
        augmentations = albumentations.Compose(
            [
                albumentations.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
                albumentations.Flip(p=0.5)
            ])
    
    images_fnames = list(map(lambda s: os.path.join(TRAINING_DATA_PATH, s + ".jpg"), df.image_name.values))
    genders = df.sex.values
    ages = df.age_approx.values
    targets = df.target.values

    dataset = MyDataset(
        image_paths=images_fnames,
        genders=genders,
        ages=ages,
        targets=targets,
        augmentations=augmentations,
    )

    return DataLoader(
        dataset, 
        batch_size=Config.get_valid_bs() if valid else Config.get_train_bs(), 
        shuffle=(not valid),
        num_workers=Config.get_nb_workers()
    )


def train(model, loader, optimizer, scheduler):
    losses = AverageMeter()
    predictions = []
    model.train()
    
    tk0 = tqdm(loader, total=len(loader))
    
    optimizer.zero_grad()
    for b_idx, (imgs, genders, ages, targets) in enumerate(tk0):
        imgs = imgs.to(Config.get_device())
        genders = genders.to(Config.get_device())
        ages = ages.to(Config.get_device())
        targets = targets.to(Config.get_device())

        out = model(imgs, genders, ages)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1))
        
        with torch.set_grad_enabled(True):
            
            loss.backward()
            optimizer.step()
            # scheduler.step()  # TODO: update depends on running loss (take avg?)
            optimizer.zero_grad()
        
        losses.update(loss.item(), loader.batch_size)
        tk0.set_postfix(loss=losses.avg)
    return losses.avg


def evaluate(model, loader):
    losses = AverageMeter()
    final_predictions = []
    model.eval()
    with torch.no_grad():
        tk0 = tqdm(loader, total=len(loader))
        for b_idx, (imgs, genders, ages, targets) in enumerate(tk0):
            imgs = imgs.to(Config.get_device())
            genders = genders.to(Config.get_device())
            ages = ages.to(Config.get_device())
            predictions = model(imgs, genders, ages).cpu()
            loss = nn.BCEWithLogitsLoss()(predictions, targets.view(-1, 1))
            losses.update(loss.item(), loader.batch_size)
            final_predictions.append(predictions)
            tk0.set_postfix(loss=losses.avg)
    return final_predictions, losses.avg


def build_and_train(fold):
    df = pd.read_csv(FOLDS_FILENAME)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    train_loader = get_loader(df_train, valid=False)
    valid_loader = get_loader(df_valid, valid=True)

    model = efficientnet_b3_mix_1(finetuning=True)
    model.to(Config.get_device())
    
    if Config.get_optimizer() == 'adam':
        optimizer = torch.optim.Adam(model.trainable_params(), lr=Config.get_lr())
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="max"
    )

    es = EarlyStopping(config=Config, fold=fold, patience=5, mode="max")

    for epoch in range(Config.get_nb_epochs()):
        train_loss = train(model, train_loader, optimizer, scheduler)
        predictions, valid_loss = evaluate(model, valid_loader)
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(df_valid.target.values, predictions)
        #print(f"Epoch = {epoch}, AUC = {auc}")
        scheduler.step(auc)
        es(auc, train_loss, valid_loss, model, model_path="../models/pths")
        if es.early_stop:
            print("Early stopping")
            break

if __name__ == "__main__":

    args = construct_hyper_param()
    Config.init(args)
    Config.set_script_name(__file__.split("/")[-1])

    FOLDS_FILENAME = '../train_folds_{}.csv'.format(Config.get_nb_folds())
    TRAINING_DATA_PATH = '../data/input/train{}/'.format(Config.get_input_res())
    
    for fold in Config.get_folds():
        if fold < Config.get_nb_folds():
            build_and_train(fold)

