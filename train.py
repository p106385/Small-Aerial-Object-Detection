import logging
import argparse
import collections
import torch
import os
import numpy as np
import data_loader.data_loaders as module_data
from parse_config import ConfigParser
from trainer import COWCGANFrcnnTrainer
from utils import setup_logger

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    setup_logger('base', config['path']['log'], 'train_' + config['name'], level=logging.INFO, screen=True, tofile=True)
    setup_logger('val', config['path']['log'], 'val_' + config['name'], level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = module_data.COWCGANFrcnnDataLoader(
        '/home/Usman/Super_Resolution/Datasets/COWC/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/valid_img/',
        '/home/Usman/Super_Resolution/Datasets/COWC/DetectionPatches_256x256/Potsdam_ISPRS/LR/x4/valid_img/', 
        1, 
        training=False
    )

    trainer = COWCGANFrcnnTrainer(config=config, data_loader=data_loader, valid_data_loader=valid_data_loader)
    trainer.train()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
