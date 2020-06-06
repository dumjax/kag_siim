import os
import torch
import albumentations

import numpy as np
import pandas as pd

import torch.nn as nn
from sklearn import metrics
from sklearn import model_selection
from torch.nn import functional as F

from wtfml.utils import EarlyStopping
from wtfml.engine import Engine
from wtfml.data_loaders.image import ClassificationLoader

import pretrainedmodels

NR_FOLDS = 5
FOLDS_FILENAME = 'train_folds_{}.csv'.format(NR_FOLDS)


class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SEResnext50_32x4d, self).__init__()

        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=None)
        # if pretrained is not None:
        #     self.base_model.load_state_dict(
        #         torch.load(
        #             "../input/pretrained-model-weights-pytorch/se_resnext50_32x4d-a260b3a4.pth"
        #         )
        #     )

        self.l0 = nn.Linear(2048, 1)

    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape

        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        out = self.l0(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(x))

        return out, loss

def train(fold):

    training_data_path = "../data/input/train224/"
    df = pd.read_csv(FOLDS_FILENAME)
    device = "cuda"
    epochs = 50
    train_bs = 32
    valid_bs = 16

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    model = SEResnext50_32x4d(pretrained="imagenet")
    model.to(device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            albumentations.Flip(p=0.5)
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".jpg") for i in valid_images]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=4
    )

    valid_dataset = ClassificationLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="max"
    )

    es = EarlyStopping(patience=5, mode="max")

    for epoch in range(epochs):
        train_loss = Engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_loss = Engine.evaluate(
            valid_loader, model, device=device
        )
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        print(f"Epoch = {epoch}, AUC = {auc}")
        scheduler.step(auc)

        es(auc, model, model_path=f"model_fold_{fold}.bin")
        if es.early_stop:
            print("Early stopping")
            break


def predict(fold):
    test_data_path = "../data/input/test224/"
    df = pd.read_csv("../data/raw/test.csv")
    device = "cuda"
    model_path=f"model_fold_{fold}.bin"

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    images = df.image_name.values.tolist()
    images = [os.path.join(test_data_path, i + ".jpg") for i in images]
    targets = np.zeros(len(images))

    test_dataset = ClassificationLoader(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    model = SEResnext50_32x4d(pretrained=None)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    predictions = Engine.predict(test_loader, model, device=device)
    predictions = np.vstack((predictions)).ravel()

    return predictions

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
        train(fold)

