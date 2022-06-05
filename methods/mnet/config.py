import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    # Default configure
    ''' # For Resnet
    cfg_dict = {
        'optim': 'Adam',
        'schedule': 'StepLR',
        'lr': 2e-5,
        'batch': 8,
        'ave_batch': 1,
        'epoch': 20,
        'step_size': '15',
        'gamma': 0.1,
        'clip_gradient': 0,
        'test_batch': 1,
    }
    '''
    # for msra
    # cfg_dict = {
    #     'optim': 'SGD', # 'Adam'
    #     'schedule': 'StepLR',
    #     'lr': 1e-1,  # '5e-5'
    #     'batch': 16,
    #     'ave_batch': 1,
    #     'epoch': 20,
    #     'step_size': '10,16',
    #     'gamma': 0.1,
    #     'clip_gradient': 0,
    #     'test_batch': 1,
    # }

    cfg_dict = {
        'optim': 'SGD', # 'Adam'
        'schedule': 'StepLR',
        'lr': 1,  # '5e-2'
        'batch': 8,
        'ave_batch': 1,
        'epoch': 20,
        'step_size': '12,16',
        'gamma': 0.1,
        'clip_gradient': 0,
        'test_batch': 1,
    }

    
    parser = base_config(cfg_dict)
    # Add custom params here
    # parser.add_argument('--size', default=320, type=int, help='Input size')
    
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    
    # Config post-process
    #config['params'] = [['encoder', config['lr'] / 10], ['decoder', config['lr']]]
    config['params'] = [['encoder', 0], ['decoder', config['lr']]]
    config['lr_decay'] = 0.9
    
    return config, None