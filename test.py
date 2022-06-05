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
from torch import nn
from metric import *

from base.framework_factory import load_framework

def test_model(model, test_sets, config, epoch=None, saver=None):
    model.eval()
    if epoch is not None:
        weight_path = os.path.join(config['weight_path'], '{}_{}_{}.pth'.format(config['model_name'], config['sub'], epoch))
        torch.save(model.state_dict(), weight_path)
    
    st = time.time()
    for set_name, test_set in test_sets.items():
        save_folder = os.path.join(config['save_path'], set_name)
        check_path(save_folder)
        
        titer = test_set.size
        MR = MetricRecorder(titer)
        ious = []
        dises = []
        
        test_bar = Bar('Dataset {:10}:'.format(set_name), max=titer)
        for j in range(titer):
            image, gt, name = test_set.load_data(j)
            out_shape = gt.shape
            Y = model(image.cuda())
            Y['final'] = nn.functional.interpolate(Y['final'], size=out_shape, mode='bilinear')
            pred = Y['final'].sigmoid_().cpu().data.numpy()[0, 0]
            
            pred = (pred * 255).astype(np.uint8)
            thre, pred = cv2.threshold(pred, 0, 255, cv2.THRESH_OTSU)
            pred, gt = normalize_pil(pred, gt)
            
            if config['crf']:
                mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
                std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
                orig_img = image[0].numpy().transpose(1, 2, 0)
                orig_img = ((orig_img * std + mean) * 255.).astype(np.uint8)
                
                pred = (pred > 0.5).astype(np.uint8)
                pred = crf_inference_label(orig_img, pred)
                pred = cv2.medianBlur(pred.astype(np.uint8), 7)
            
            iou = cal_iou(pred, gt)
            ious.append(iou)
            dis = cal_dis(pred, gt)
            dises.append(dis)
            
            MR.update(pre=pred, gt=gt)
            
            # save predictions
            if config['save']:
                if config['crf']:
                    tag = 'crf'
                else:
                    tag = 'final'
                fnl_folder = os.path.join(save_folder, tag)
                check_path(fnl_folder)
                im_path = os.path.join(fnl_folder, name + '.png')
                Image.fromarray((pred * 255)).convert('L').save(im_path)
                
                if saver is not None:
                    saver(Y, gt, name, save_folder, config)
                    pass
                
            Bar.suffix = '{}/{}'.format(j, titer)
            test_bar.next()
        
        mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        print(' Mean-F: {}, MAE: {}, IOU: {:.3f}, dis: {:.3f}.'.format(meanf, mae, np.mean(ious), np.mean(dises)))
        
    print('Test using time: {}.'.format(round(time.time() - st, 3)))

def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    config, model, _, _, _, saver = load_framework(net_name)
    
    if config['crf']:
        config['orig_size'] = True
    
    if config['weight'] != '':
        model.load_state_dict(torch.load(config['weight'], map_location='cpu'))
    else:
        print('No weight file provide!')

    test_sets = OrderedDict()
    for set_name in config['vals']:
        test_sets[set_name] = Test_Dataset(name=set_name, config=config)
    
    model = model.cuda()
    test_model(model, test_sets, config, saver=saver)
        
if __name__ == "__main__":
    main()