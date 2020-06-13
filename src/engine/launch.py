import os
import math
import random
import albumentations
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics
from sklearn import model_selection
from torch.nn import functional as F

from .utils import AverageMeter


class MyDataset(Dataset):
    def __init__(self, config, valid, image_paths, genders, ages, targets):
        self.image_paths = image_paths
        self.genders = [0 if age == 'male' else (1 if age == 'female' else 0.5) for age in ages]  # TODO: use sklearn

        # TODO: use sklearn for this stuff. Also: try categorical?
        self.ages = [float(age) for age in ages]
        median_age = np.median([a for a in self.ages if not math.isnan(a)])
        self.ages = np.array([age if not math.isnan(age) else median_age for age in self.ages])
        self.ages /= max(ages)

        self.targets = targets

        if valid and 'VALID_AUGMENTATIONS' in config:
            self.augmentations = config['VALID_AUGMENTATIONS']
        elif not valid and 'TRAIN_AUGMENTATIONS' in config:
            self.augmentations = config['TRAIN_AUGMENTATIONS']
        else:
            self.augmentations = None

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


def get_loader(config, df, valid: bool) -> DataLoader:
    """
    df: contains the data about the images to put in the returned loader
    valid: whether this is a validation set (otherwise it's a training set)
    """    
    images_fnames = list(map(lambda s: os.path.join(config['TRAINING_DATA_PATH'], s + ".jpg"), 
                             df.image_name.values))
    genders = df.sex.values
    ages = df.age_approx.values
    targets = df.target.values

    dataset = MyDataset(
        config=config,
        valid=valid,
        image_paths=images_fnames,
        genders=genders,
        ages=ages,
        targets=targets,
    )

    return DataLoader(
        dataset,
        batch_size=config['VALID_BATCHSIZE'] if valid else config['TRAIN_BATCHSIZE'], 
        shuffle=(not valid),
        num_workers=4
    )


def train(config, model, loader, optimizer, scheduler):
    losses = AverageMeter()
    predictions = []
    model.train()
    
    tk0 = tqdm(loader, total=len(loader))
    
    optimizer.zero_grad()
    for b_idx, (imgs, genders, ages, targets) in enumerate(tk0):
        imgs = imgs.to(config['DEVICE'])
        genders = genders.to(config['DEVICE'])
        ages = ages.to(config['DEVICE'])
        targets = targets.to(config['DEVICE'])

        out = model(imgs, genders, ages)
        loss = config['LOSS_FN'](out, targets.view(-1, 1))
        
        with torch.set_grad_enabled(True):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None and config['APPLY_SCHEDULER_EACH_MINIBATCH']:
                scheduler.step()
        
        losses.update(loss.item(), loader.batch_size)
        tk0.set_postfix(loss=losses.avg)
    return losses.avg


def evaluate(config, model, loader):
    losses = AverageMeter()
    final_predictions = []
    model.eval()
    with torch.no_grad():
        tk0 = tqdm(loader, total=len(loader))
        for b_idx, (imgs, genders, ages, targets) in enumerate(tk0):
            imgs = imgs.to(config['DEVICE'])
            genders = genders.to(config['DEVICE'])
            ages = ages.to(config['DEVICE'])
            predictions = model(imgs, genders, ages).cpu()
            loss = config['LOSS_FN'](predictions, targets.view(-1, 1))
            losses.update(loss.item(), loader.batch_size)
            final_predictions.append(predictions)
            tk0.set_postfix(loss=losses.avg)
    return final_predictions, losses.avg


def build_and_train(config, fold):
    df = pd.read_csv(config['FOLDS_FILENAME'])
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    train_loader = get_loader(config, df_train, valid=False)
    valid_loader = get_loader(config, df_valid, valid=True)

    model = config['MODEL_CLS'](config)
    model.to(config['DEVICE'])
    
    optimizer = config['OPTIMIZER_CLS'](model.trainable_params(), **config['OPTIMIZER_KWARGS'])
    if config['SCHEDULER_CLS'] is not None:
        scheduler = config['SCHEDULER_CLS'](optimizer, **config['SCHEDULER_KWARGS'])
    else:
        scheduler = None

    for epoch in range(config['NR_EPOCHS']):
        train_loss = train(config, model, train_loader, optimizer, scheduler)
        predictions, valid_loss = evaluate(config, model, valid_loader)
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(df_valid.target.values, predictions)
        print(f"Epoch = {epoch}, AUC = {auc}")
        if scheduler is not None and not config['APPLY_SCHEDULER_EACH_MINIBATCH']:
            scheduler.step(auc)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def launch(config):
    set_seed(config['SEED'])
    df = pd.read_csv(config['FOLDS_FILENAME'])
    nr_folds = len(df['kfold'].unique())
    for fold in range(nr_folds):
        build_and_train(config, fold)

