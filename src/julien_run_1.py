

import torch
import torch.nn as nn
import albumentations

from engine import launch
from architectures import TimmModel

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

config = {
    ### Global parameters
    'NAME': 'julien_001',
    'SEED': 41,
    'DEVICE': 'cuda',
    'FOLDS_FILENAME': '../train_folds_5.csv',
    'TRAINING_DATA_PATH': '../data/input/train288/',
    
    ### Model parameters
    'MODEL_CLS': TimmModel,
    'PRETRAINED_MODEL': 'efficientnet_b3',
    'FINETUNING': True,
    'USE_GENDER': False,
    'USE_AGE': True,

    # TODO:
    # 'HIDDEN_LAYERS': [],
    # 'NONLINEARITY': ,

    ### Training parameters:
    'NR_EPOCHS': 20,
    'TRAIN_BATCHSIZE': 16,
    'VALID_BATCHSIZE': 16,

    'OPTIMIZER_CLS': torch.optim.Adam,
    'OPTIMIZER_KWARGS': {'lr': 1e-4},  # all arguments passed to optimizer, except model parameters
    'SCHEDULER_CLS': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'SCHEDULER_KWARGS': {'patience':3, 'threshold':0.001, 'mode':'max'},  # all arguments except optimizer
    'APPLY_SCHEDULER_EACH_MINIBATCH': False,  # If False, apply each epoch. No scheduler if SCHEDULER_CLS is None

    'LOSS_FN': nn.BCEWithLogitsLoss(),

    ### Pre-processing parameters:
    'TRAIN_AUGMENTATIONS': albumentations.Compose(
            [
                albumentations.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
                albumentations.Flip(p=0.5)
            ]),
    'VALID_AUGMENTATIONS': albumentations.Compose(
            [
                albumentations.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True)
            ])
}

if __name__ == '__main__':
    launch(config)
