import sys
import os
import time
import random

from progress.bar import Bar
from collections import OrderedDict
from util import *
from PIL import Image
from data import get_loader, Test_Dataset
from test import test_model
import torch
from torch.nn import utils
from base.framework_factory import load_framework

torch.set_printoptions(precision=5)

def main():
    # Loading model
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    config, model, optim, sche, model_loss, saver = load_framework(net_name)
    
    # Loading datasets
    config['vals'] = [config['trset'], ]
    train_loader = get_loader(config)
    test_sets = OrderedDict()
    for set_name in config['vals']:
        test_sets[set_name] = Test_Dataset(name=set_name, config=config)
    
    debug = config['debug']
    num_epoch = config['epoch']
    num_iter = len(train_loader)
    ave_batch = config['ave_batch']
    trset = config['trset']
    batch_idx = 0
    model.zero_grad()
    for epoch in range(1, num_epoch + 1):
        model.train()
        torch.cuda.empty_cache()
        
        if epoch == 1:
            test_model(model, test_sets, config, 0)
        
        print('Current LR: {:.6f}.'.format(optim.param_groups[1]['lr']))
        bar = Bar('{:10}-{:8} | epoch {:2}:'.format(net_name, config['sub'], epoch), max=num_iter)

        st = time.time()
        loss_count = 0
        optim.zero_grad()
        sche.step()
        #print(optim.state_dict())
        for i, pack in enumerate(train_loader, start=1):
            
            cur_it = i + (epoch-1) * num_iter
            total_it = epoch * num_iter
            
            images, gts = pack
            images, gts = images.float().cuda(), gts.float().cuda()
            
            if config['multi']:
                scales = [-1, 0, 1] 
                #scales = [-2, -1, 0, 1, 2] 
                input_size = config['size']
                input_size += int(np.random.choice(scales, 1) * 64)
                #input_size += int(np.random.choice(scales, 1) * 32)
                images = F.upsample(images, size=(input_size, input_size), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(input_size, input_size), mode='nearest')
            
            Y = model(images, 'train')
            #print(torch.max(Y['final']), torch.min(Y['final']))
            loss = model_loss(Y, config) / ave_batch
            loss_count += loss.data
            loss.backward()

            batch_idx += 1
            if batch_idx == ave_batch:
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
                optim.step()
                optim.zero_grad()
                batch_idx = 0
            
            Bar.suffix = '{:4}/{:4} | loss: {:1.5f}, time: {}.'.format(i, num_iter, round(float(loss_count / i), 5), round(time.time() - st, 3))
            bar.next()

        bar.finish()

        test_model(model, test_sets, config, epoch)

if __name__ == "__main__":
    main()