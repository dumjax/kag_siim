

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import albumentations

from engine import launch
from architectures import TimmModel2

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

"""
TODO

* Reduce overfitting (more augmentations, more randomness, Dropout?)
* Exponential/linear decrease of LR
"""

config = {
    ### Global parameters
    'NAME': 'max_6',
    'SCRIPT_NAME': os.path.basename(__file__),
    'SEED': 41,
    'DEVICE': 'cuda',
    'FOLDS_FILENAME': 'train_folds_group_5.csv',
    'NR_FOLDS': 5,  # Number of folds to complete
    'TRAINING_DATA_PATH': '../data/input/train380/',
    
    ### Model parameters
    'MODEL_CLS': TimmModel2,
    'PRETRAINED_MODEL': 'efficientnet-b4',  # Don't forget to update the training data path with correct resolution
    'FINETUNING': True,
    'USE_GENDER': True,
    'USE_AGE': True,
    'USE_SITES': True,


    ### Training parameters:
    'NR_EPOCHS': 50,
    'TRAIN_BATCHSIZE': 8,
    'VALID_BATCHSIZE': 8,

    'OPTIMIZER_CLS': torch.optim.Adam,
    'OPTIMIZER_KWARGS': {'lr': 1e-4},  # all arguments passed to optimizer, except model parameters
    'SCHEDULER_CLS': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'SCHEDULER_KWARGS': {'patience': 3, 'threshold': 0.001, 'mode': 'max'},  # all arguments except optimizer
    'APPLY_SCHEDULER_EACH_MINIBATCH': False,  # If False, apply each epoch. No scheduler if SCHEDULER_CLS is None

    'EARLYSTOP_PATIENCE': 7,
    'LOSS_FN': nn.BCELoss(),

    ### Pre-processing parameters:
    'TRAIN_AUGMENTATIONS': albumentations.Compose(
            [
                albumentations.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True),
                albumentations.RGBShift(p=0.2),
                albumentations.RandomBrightnessContrast(p=0.2),
                #albumentations.GridDropout(ratio=0.1, p=0.2),
                albumentations.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.5, rotate_limit=15, p=0.5),
                albumentations.Flip(p=0.5)
            ]),
    'VALID_AUGMENTATIONS': albumentations.Compose(
            [
                albumentations.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True)
            ])
}

if __name__ == '__main__':
    launch(config)
