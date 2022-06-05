import sys
import importlib
from data import Test_Dataset
import torch
import time
from progress.bar import Bar
import os
from collections import OrderedDict
import cv2
from PIL import Image
from util import *
import numpy as np
import argparse

from metric import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../dataset/', help='The name of network')
    parser.add_argument('--vals', default='all', help='Set the testing sets')
    
    parser.add_argument('--pre_path', default='./maps', help='Weight path of network')
    
    params = parser.parse_args()
    config = vars(params)
    config['orig_size'] = True
    
    if config['vals'] == 'all':
        vals = ['MSB-TE', 'PASCAL-S', 'ECSSD', 'HKU-IS', 'DUTS-TE', 'DUT-OMRON']
    else:
        vals = config['vals'].split(',')
        
    for val in vals:
        img_path = '{}/{}/final/'.format(config['pre_path'], val)
        #img_path = config['pre_path']
        if not os.path.exists(img_path):
            continue
        test_set = Test_Dataset(name=val, config=config)
        titer = test_set.size
        MR = MetricRecorder(titer)
        #MR = MetricRecorder()
        
        test_bar = Bar('Dataset {:10}:'.format(val), max=titer)
        for j in range(titer):
            _, gt, name = test_set.load_data(j)
            #print(img_path + name + '_stage2.png')
            pred = Image.open(img_path + name + '.png').convert('L') # 
            #print(np.max(pred))
            out_shape = gt.shape
            pred = np.array(pred.resize((out_shape[::-1])))
            
            pred, gt = normalize_pil(pred, gt)
            MR.update(pre=pred, gt=gt)
            
            #pred, gt = normalize_pil(pred, gt)
            #MR.update(pre=pred, gt=gt)
            #print(np.max(pred), np.max(gt))
            #MR.update(pre=pred.astype(np.uint8), gt=(gt * 255).astype(np.uint8))
            
                
            Bar.suffix = '{}/{}'.format(j, titer)
            test_bar.next()
        
        #scores = MR.show(bit_num=3)
        mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        print('  Max-F: {}, Maen-F: {}, Fbw: {}, MAE: {}, SM: {}, EM: {}.'.format(maxf, meanf, wfm, mae, sm, em))
        #print('  Max-F: {}, adp-F: {}, Fbw: {}, MAE: {}, SM: {}, EM: {}.'.format(scores['fm'], scores['adpFm'], scores['wFm'], scores['MAE'], scores['Sm'], scores['adpEm']))
        #mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        #print('  MAE: {}, Max-F: {}, Maen-F: {}, SM: {}, EM: {}, Fbw: {}.'.format(mae, maxf, meanf, sm, em, wfm))

    
if __name__ == "__main__":
    main()