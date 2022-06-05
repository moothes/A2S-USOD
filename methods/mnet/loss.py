import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from util import *

import numpy as np
from math import exp

def Loss(preds, config):
    #thre = 0.7
    #pred = torch.sigmoid(preds['sal']) - thre
    #loss = torch.mean(torch.pow((pred > 0).float() * pred / (1 - thre) + (pred <= 0).float() * pred / thre, 2))
    #print(torch.max((pred > 0).float() * pred / (1 - thre)), torch.min((pred <= 0).float() * pred / thre))
    loss = 0
    smooth = 0
    ws = [1, 0.1, 0.3, 0.5] # [1, 0.1, 0.3, 0.5]
    ps = [1, 0.1, 0.]
    
    for i, sal, w in zip(range(4), preds['sal'], ws):
        p = torch.clamp(torch.sigmoid(sal), smooth, 1 - smooth)
        #print(p.shape)
        
        #area = p.size(-2) * p.size(-1)
        #pos_num = torch.sum(p.gt(0.5).float(), dim=(2,3), keepdims=True)
        #neg_num = area - pos_num
        #print(area, pos_num, neg_num)
        #print(pos_num / area, neg_num / area)
        #pw = p.gt(0.5).float() * pos_num / area
        #nw = (1 - p.gt(0.5).float()) * neg_num / area
        #wmap = pw + nw
        
        #p = torch.sigmoid(sal)
        p0 = torch.pow(p - 0.5, 2)                        # L2 loss
        p1 = torch.abs(p - 0.5)                           # L1 loss
        p2 = F.binary_cross_entropy(p, p.gt(0.5).float()) # BCE loss
        p = p0 * ps[0] + p1 * ps[1] + p2 * ps[2]
        
        #p = p * wmap * 2
        
        mask = (torch.rand(p.size()) > 0.5).float().cuda()
        #mask = (p > 0.04).float().cuda()
        
        #print(p.shape, mask.shape)
        p_mean = torch.sum(p * mask) / torch.sum(mask)
        #p_mean = torch.mean(p)
        loss += (1 - p_mean) * w
        
    return loss
