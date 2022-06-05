import sys
import os
import time
import random
import importlib

#from thop import profile
from progress.bar import Bar
from collections import OrderedDict
from util import *
from PIL import Image
from data import Train_Dataset, Test_Dataset
from test import test_model
import torch
from torch.nn import utils
from base.framework_factory import load_framework
import cv2

torch.set_printoptions(precision=5)

def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    # Loading model
    config, model, optim, sche, model_loss, saver = load_framework(net_name)
    config['stage'] = 2

    # Loading datasets
    train_loader = Train_Dataset(config) # get_loader(config)
    test_sets = OrderedDict()
    for set_name in config['vals']:
        test_sets[set_name] = Test_Dataset(name=set_name, config=config)
    
    debug = config['debug']
    num_epoch = config['epoch']
    num_iter = train_loader.size
    ave_batch = config['ave_batch']
    trset = config['trset']
    batch_idx = 0
    model.zero_grad()
    for epoch in range(1, num_epoch + 1):
        model.train()
        torch.cuda.empty_cache()
        
        if debug:
            test_model(model, test_sets, config, epoch)

        st = time.time()
        loss_count = 0
        crf_count = 0
        optim.zero_grad()
        sche.step()
        iter_per_epoch = num_iter // config['batch']
        index_list = np.array(range(num_iter))
        random.shuffle(index_list)
        index_list = index_list[:iter_per_epoch * config['batch']]
        index_list = np.array(index_list).reshape((iter_per_epoch, config['batch']))
        
        print('Current LR: {:.6f}.'.format(optim.param_groups[1]['lr']))
        bar = Bar('{:10}-{:8} | epoch {:2}:'.format(net_name, config['sub'], epoch), max=iter_per_epoch)

        lamda = config['resdual']
        #print("lamda:", lamda)

        #for i, pack in enumerate(train_loader, start=1):
        for i, idx_list in enumerate(index_list):
            
            cur_it = i + (epoch-1) * iter_per_epoch
            total_it = num_epoch * iter_per_epoch
            
            images, gts = train_loader.images[idx_list], train_loader.gts[idx_list]

            images = images.cuda()
            gts = gts.cuda()
            
            if config['multi']:
                scales = [-1, 0, 1] 
                #scales = [-2, -1, 0, 1, 2] 
                input_size = config['size']
                input_size += int(np.random.choice(scales, 1) * 64)
                images = F.upsample(images, size=(input_size, input_size), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(input_size, input_size), mode='nearest')
            
            Y = model(images, 'train')
            loss = model_loss(Y, gts, config) / ave_batch

            loss_count += loss.data
            loss.backward()

            batch_idx += 1
            if batch_idx == ave_batch:
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
                optim.step()
                optim.zero_grad()
                batch_idx = 0
            
            Bar.suffix = '{:4}/{:4} | loss: {:1.5f}, time: {}.'.format(i, iter_per_epoch, round(float(loss_count / i), 5), round(time.time() - st, 3))
            bar.next()
            
            if epoch > 2:
                train_loader.gts[idx_list] = (gts * lamda + torch.sigmoid(Y['final'].detach()) * (1 - lamda))
                
        bar.finish()
        if trset in ('DUTS-TR', 'MSB-TR'):
            test_model(model, test_sets, config, epoch)
            

if __name__ == "__main__":
    main()