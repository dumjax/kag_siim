import datetime
import os
import random

import numpy as np
import torch

import argparse
import os

from pathlib import Path

from utils import load_yaml

import warnings


class Config(object):
    #gpu_ids = []
    #multi_gpus = False
    #is_gpu = torch.cuda.is_available()
    # gpus = 0
    device = 0
    # device = torch.device("cpu")
    # local_rank = 0
    # model_name = ""
    # premodel = ""
    # args = []

    @staticmethod
    def init(args):
        Config.args = args
        
        Config.set_seed(args['SEED'])
        
        Config.set_lr(args['LEARNING_RATE'])
        Config.set_optimizer(args['OPTIMIZER'])
        Config.set_train_bs(args['TRAIN_BATCHSIZE'])
        Config.set_valid_bs(args['VALID_BATCHSIZE'])
        Config.set_archi(args['PRETRAINED_MODEL'])
        Config.set_nb_epochs(args['NB_EPOCHS'])
        Config.set_nb_folds(args['NB_FOLDS'])
        Config.set_input_res(args['INPUT_RESOLUTION'])
        Config.set_folds(args['FOLDS'])
        
        #TODO scheduler
        
        torch.cuda.empty_cache()
        warnings.filterwarnings("ignore")
        
    # @staticmethod
    # def set_gpus(gpu_ids):
    #     print("Device(s) available: " + str(torch.cuda.device_count()))
    #     Config.gpu_ids = gpu_ids
    #     Config.gpus = len(gpu_ids)
    #     Config.multi_gpus = (Config.gpus > 1)
    # 
    #     if Config.multi_gpus:
    #         Config.device = torch.device("cuda:" + str(Config.local_rank))
    #         torch.cuda.set_device(Config.local_rank)
    #         torch.distributed.init_process_group(backend="nccl", world_size=Config.gpus)
    #     elif Config.train_on_gpu():
    #         Config.device = torch.device("cuda:" + str(Config.get_device_ids()[0]))

    @staticmethod
    def get_device_ids():
        return Config.gpu_ids

    @staticmethod
    def is_multi_gpus():
        return Config.multi_gpus

    @staticmethod
    def train_on_gpu():
        return Config.is_gpu

    @staticmethod
    def get_device_status():
        if Config.is_multi_gpus():
            print(f"Training on {Config.gpus} GPUs!")
        elif Config.train_on_gpu():
            print('Training on GPU (' + str(Config.get_device()) + ')!')
        else:
            print('No GPU available, training on CPU; consider making n_epochs very small.')

    @staticmethod
    def get_device():
        return Config.device
    
    @staticmethod
    def set_device(device):
        Config.device = device

    @staticmethod
    def get_optimizer():
        return Config.optimizer

    @staticmethod
    def set_optimizer(optimizer):
        Config.optimizer = optimizer
    
    @staticmethod
    def get_lr():
        return Config.lr
    
    @staticmethod
    def set_lr(lr):
        Config.lr = lr
    
    @staticmethod
    def set_archi(archi):
        Config.archi = archi
    
    @staticmethod
    def get_archi():
        return Config.archi

    @staticmethod
    def set_train_bs(train_bs):
        Config.train_bs = train_bs

    @staticmethod
    def get_train_bs():
        return Config.train_bs
    
    @staticmethod
    def set_valid_bs(valid_bs):
        Config.valid_bs = valid_bs

    @staticmethod
    def get_valid_bs():
        return Config.valid_bs
    
    @staticmethod
    def set_nb_epochs(nb_epochs):
        Config.nb_epochs = nb_epochs

    @staticmethod
    def get_nb_epochs():
        return Config.nb_epochs
    
    @staticmethod
    def set_nb_workers(nb_workers):
        Config.nb_workers = nb_workers

    @staticmethod
    def get_nb_workers():
        return Config.nb_workers
    
    @staticmethod
    def set_folds(folds):
        Config.folds = folds

    @staticmethod
    def get_folds():
        return Config.folds
    
    @staticmethod
    def set_nb_folds(nb_folds):
        Config.nb_folds = nb_folds

    @staticmethod
    def get_nb_folds():
        return Config.nb_folds
    
    @staticmethod
    def set_input_res(input_res):
        Config.input_res = input_res

    @staticmethod
    def get_input_res():
        return Config.input_res


    @staticmethod
    def get_args():
        return Config.args

    @staticmethod
    def get_model_name():
        return Config.model_name

    @staticmethod
    def set_seed(seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def construct_hyper_param():
    parser = argparse.ArgumentParser(description='kaggle siim pipeline')

    parse_general(parser)
    args = parser.parse_args()
    config_folder = os.path.join("../models/yamls", Path(args.train_cfg.strip("/")))
    train_config = load_yaml(config_folder)
    print(train_config)

    return train_config


def parse_general(parser):
    parser.add_argument('--train_cfg')
    return parser.parse_args()


def parse_test(parser):
    pass