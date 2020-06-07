import os
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


# Parameters:
DEVICE = 'cuda'
NR_EPOCHS = 20
TRAIN_BATCHSIZE = 32
VALID_BATCHSIZE = 16
NR_FOLDS = 5
FOLDS_FILENAME = 'train_folds_{}.csv'.format(NR_FOLDS)
PRETRAINED_MODEL = 'resnext50d_32x4d'

# TODO: model-dependent
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# PRETRAINED_MODEL = 'se_resnext50_32x4d'
# PRETRAINED_OPT = 'imagenet'
# MEAN = pretrainedmodels.pretrained_settings[PRETRAINED_MODEL][PRETRAINED_OPT]['mean']
# STD = pretrainedmodels.pretrained_settings[PRETRAINED_MODEL][PRETRAINED_OPT]['std']
INPUT_RESOLUTION = 224
TRAINING_DATA_PATH = '../data/input/train{}/'.format(INPUT_RESOLUTION)


# class MyModel(nn.Module):
#     def __init__(self, pretrained='imagenet'):
#         """ Based on pretrainedmodels
#         """
#         super(MyModel, self).__init__()

#         self.base_model = pretrainedmodels.__dict__[
#             "se_resnext50_32x4d"
#         ](pretrained=None)

#         # disable fine-tuning:
#         for param in self.base_model.parameters():
#             param.requires_grad = False

#         self.l0 = nn.Linear(2048, 1)

#     def trainable_params(self):
#         return self.l0.parameters()

#     def forward(self, image):
#         batch_size, _, _, _ = image.shape

#         x = self.base_model.features(image)
#         x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

#         out = self.l0(x)

#         return out


class MyModel(nn.Module):
    def __init__(self):
        """ Based on timm
        """
        super(MyModel, self).__init__()

        self.base_model = timm.create_model(PRETRAINED_MODEL, pretrained=True)

        # disable fine-tuning:
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.l0 = nn.Linear(2048, 1)

    def trainable_params(self):
        return self.l0.parameters()
        
    def forward(self, image):
        batch_size, _, _, _ = image.shape

        x = self.base_model.forward_features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        out = self.l0(x)

        return out


class MyDataset(Dataset):
    def __init__(self, image_paths, targets, augmentations=None):
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        targets = self.targets[item]
        image = np.array(image)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        imgs = torch.tensor(image, dtype=torch.float)
        targets = torch.tensor(targets, dtype=torch.float)

        return imgs, targets


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
    targets = df.target.values

    dataset = MyDataset(
        image_paths=images_fnames,
        targets=targets,
        augmentations=augmentations,
    )

    return DataLoader(
        dataset, 
        batch_size=VALID_BATCHSIZE if valid else TRAIN_BATCHSIZE, 
        shuffle=(not valid),
        num_workers=4
    )

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


def train(model, loader, optimizer, scheduler):
    losses = AverageMeter()
    predictions = []
    model.train()
    
    tk0 = tqdm(loader, total=len(loader))
    
    optimizer.zero_grad()
    for b_idx, (imgs, targets) in enumerate(tk0):
        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)

        out = model(imgs)
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
        for b_idx, (imgs, targets) in enumerate(tk0):
            imgs = imgs.to(DEVICE)
            predictions = model(imgs).cpu()
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

    model = MyModel()
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.trainable_params(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="max"
    )

    for epoch in range(NR_EPOCHS):
        train_loss = train(model, train_loader, optimizer, scheduler)
        predictions, valid_loss = evaluate(model, valid_loader)
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(df_valid.target.values, predictions)
        print(f"Epoch = {epoch}, AUC = {auc}")
        # scheduler.step(auc)


if __name__ == "__main__":
    if not os.path.exists(FOLDS_FILENAME):
        # create folds
        df = pd.read_csv("../data/raw/train.csv")
        df["kfold"] = -1
        df = df.sample(frac=1).reset_index(drop=True)
        y = df.target.values
        kf = model_selection.StratifiedKFold(n_splits=NR_FOLDS)

        for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
            df.loc[v_, 'kfold'] = f

        df.to_csv(FOLDS_FILENAME, index=False)

    for fold in range(NR_FOLDS):
        build_and_train(fold)

