

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import albumentations

from copy import copy
from engine import launch
from architectures import TimmModel, EfficientNet

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

"""
TODO

* Check image scaling method depending on model (bicubic vs bilinear)
"""

config = {
    ### Global parameters
    'ABS_PATH': "?",
    'NAME': 'julien_033',
    'SCRIPT_NAME': os.path.basename(__file__),
    'SEED': 41,
    'DEVICE': 'cuda',
    'FOLDS_FILENAME': 'train_folds_5.csv',
    'NR_FOLDS': 5,  # Number of folds to complete
    'TRAINING_DATA_PATH': 'data/input/train380/',
    'TEST_DATA_PATH': 'data/input/test380/',

    ### Model parameters
    'MODEL_CLS': EfficientNet,
    'PRETRAINED_MODEL': 'efficientnet-b4',  # Don't forget to update the training data path with correct resolution
    'FINETUNING': True,
    'USE_GENDER': True,
    'USE_AGE': True,
    'USE_SITES': True,
    'HIDDEN_SIZES': [40],
    'NONLINEARITY': F.relu,

    ### Training parameters:
    'NR_EPOCHS': 50,
    'TRAIN_BATCHSIZE': 8,
    'VALID_BATCHSIZE': 16,

    'OPTIMIZER_CLS': torch.optim.Adam,
    'OPTIMIZER_KWARGS': {'lr': 5e-5},  # all arguments passed to optimizer, except model parameters
    # 'SCHEDULER_CLS': torch.optim.lr_scheduler.ExponentialLR,
    'SCHEDULER_CLS': None,
    'SCHEDULER_KWARGS': {'gamma': 0.9},  # all arguments except optimizer
    'APPLY_SCHEDULER_EACH_MINIBATCH': False,  # If False, apply each epoch. No scheduler if SCHEDULER_CLS is None

    'EARLYSTOP_PATIENCE': 7,
    'LOSS_FN': nn.BCELoss(),

    ### Pre-processing parameters:
    'TRAIN_AUGMENTATIONS': albumentations.Compose(
            [
                albumentations.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True),
                # albumentations.RGBShift(p=0.2),
                # albumentations.RandomContrast(p=0.2),
                # albumentations.GridDropout(ratio=0.1, p=0.2),
                # albumentations.RandomBrightnessContrast(p=0.5),
                # albumentations.RandomContrast(p=0.5),
                albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=90, p=0.8),
                albumentations.Flip(p=0.5)
            ]),
    'VALID_AUGMENTATIONS': albumentations.Compose(
            [
                albumentations.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True)
            ])
}

if __name__ == '__main__':
    res1 = launch(config)
    print('res1: {}'.format(res1))
