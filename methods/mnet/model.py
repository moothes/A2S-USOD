import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from .res import resnet

mode = 'bilinear' # 'nearest' # 

def up_conv(cin, cout):
    yield nn.Conv2d(cin, cout, 3, padding=1)
    #yield nn.BatchNorm2d(cout)
    yield nn.GroupNorm(1, cout)
    #yield nn.Dropout(p=0.1, inplace=True)
    yield nn.ReLU(inplace=True)

def foreground_sign(pred):
    mask = (pred > 0).float()
    pos_num = torch.sum(mask, dim=(1,2,3))
    total = mask.size()[2] * mask.size()[3]
    pos_ratio = pos_num / total
    sign = torch.sign(0.5 - pos_ratio)
    sign = sign.view(-1, 1, 1, 1)
    return sign

class SE_block(nn.Module):
    def __init__(self, feat):
        super(SE_block, self).__init__()
        self.conv = nn.Conv2d(feat, feat, 1)
        self.gn = nn.GroupNorm(feat // 2, feat)

    def forward(self, x):
        glob_x = F.adaptive_avg_pool2d(x, (1, 1))
        glob_x = torch.sigmoid(self.conv(glob_x)) # self.gn(self.conv(glob_x)) #
        x = glob_x * x
        return x

class ada_block(nn.Module):
    def __init__(self, config, feat, out_feat=64):
        super(ada_block, self).__init__()
        
        self.ad0 = nn.Sequential(*list(up_conv(feat, out_feat)))
        self.se = SE_block(out_feat)

    def forward(self, x):
        x = self.ad0(x)
        x = self.se(x)
        return x

def normalize(x):
    center = torch.mean(x, dim=(2, 3), keepdim=True)
    x = x - center
    #x = x - 0.2
    return x

class decoder(nn.Module):
    def __init__(self, config, encoder, feat):
        super(decoder, self).__init__()
        
        #self.ad0 = ada_block(config, feat[0], feat[0])
        #self.ad1 = ada_block(config, feat[1], feat[0])
        self.ad2 = ada_block(config, feat[2], feat[0])
        self.ad3 = ada_block(config, feat[3], feat[0])
        self.ad4 = ada_block(config, feat[4], feat[0])
        self.fusion = ada_block(config, feat[0] * 3, feat[0]) # fusion_block(config, feat[0])
        
        #self.refine_block = ada_block(config, feat[0], 64)
        
        #self.ref_net = resnet(True)
        #self.cross_block = ada_block(config, feat[0], feat[0])
        
        #self.fusion1 = ada_block(config, 3, 1)
        
        #self.classifier = nn.Parameter(torch.ones(feat[0]) / 64.)
        #self.classifier = nn.Conv2d(feat[0], feat[0], 3, padding=1)
        
        #print(logits.argmax(dim=-1, keepdim=True).shape)
        #print(encoder.fc.weight.shape, encoder.fc.bias.shape)
        
    def forward(self, feat_maps, ws, x_size, phase='test'):
        xs = feat_maps
        
        '''
        idxs = logits.argmax(dim=-1)
        weights = w_list[0][idxs].unsqueeze(-1).unsqueeze(-1)
        bias = w_list[1][idxs].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        cam = torch.sum(cam_maps * weights + bias, dim=1, keepdim=True)
        cam = cam - torch.mean(cam, dim=(1,2,3), keepdim=True)
        
        mask = nn.functional.interpolate(torch.sigmoid(cam), size=xs[2].size()[2:], mode=mode)
        #print(xs[2].shape, xs[-1].shape)
        fore_feat = torch.sum(mask * xs[2], dim=(2,3), keepdim=True) / torch.sum(mask, dim=(2,3), keepdim=True)
        fore_mask = torch.sum(fore_feat * xs[2], dim=1, keepdim=True)
        
        
        #cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam) + 1e-5)
        '''
        
        #print(ws[0].shape, xs[4].shape)
        
        w = torch.sum(ws[0], dim=0).view(1, -1, 1, 1)
        xs[4] = xs[4] * w
        ref_pred = nn.functional.interpolate(torch.sum(xs[4], dim=1, keepdim=True), size=xs[0].size()[2:], mode=mode)
        #print(ref_pred.shape)
        
        #x0 = self.ad0(xs[0])
        #x1 = self.ad1(xs[1])
        x2 = self.ad2(xs[2])
        x3 = self.ad3(xs[3])
        x4 = self.ad4(xs[4])
        #clser = torch.mean(self.classifier(x4), dim=(2,3), keepdim=True)
        #clser = clser / torch.norm(clser, dim=(1), keepdim=True) #.norm()

        #x1u = nn.functional.interpolate(x1, size=xs[0].size()[2:], mode=mode)
        #print(torch.mean(x0.norm(dim=1), dim=(1,2)), torch.mean(x1u.norm(dim=1), dim=(1,2)), torch.mean(x2u.norm(dim=1), dim=(1,2)))
        
        #x2u = normalize(x2u)
        #x3u = normalize(x3u)
        #x4u = normalize(x4u)
        
        #fuse = torch.cat([x0, x1, x2, x3, x4], dim=1)
        
        #x0u = nn.functional.interpolate(x0, size=xs[0].size()[2:], mode=mode)
        #x1u = nn.functional.interpolate(x1, size=xs[0].size()[2:], mode=mode)
        x2u = nn.functional.interpolate(x2, size=xs[0].size()[2:], mode=mode)
        x3u = nn.functional.interpolate(x3, size=xs[0].size()[2:], mode=mode)
        x4u = nn.functional.interpolate(x4, size=xs[0].size()[2:], mode=mode)
        
        x2p = normalize(x2)
        x2p = torch.sum(x2p, dim=1, keepdim=True)
        x3p = normalize(x3)
        x3p = torch.sum(x3p, dim=1, keepdim=True)
        x4p = normalize(x4)
        x4p = torch.sum(x4p, dim=1, keepdim=True)
        
        
        fuse = torch.cat([x2u, x3u, x4u], dim=1)
        #fuse = torch.cat([x2u, x3u, x4u], dim=1)
        #fuse = torch.cat([x4u], dim=1)
        pred = self.fusion(fuse) # x4u #
        pred = normalize(pred)
        #kernel = self.refine_block(pred)
        
        pred = torch.sum(pred, dim=1, keepdim=True)
        
        # Find the foreground and background
        pred = pred * foreground_sign(pred)
        pred = nn.functional.interpolate(pred, size=x_size, mode='bilinear')
        
        #refs = self.ref_net(x)
        #for ref in refs:
        #    print(ref.shape)

        #kernel = nn.functional.adaptive_avg_pool2d(kernel, (1, 1))
        #pl0 = torch.sum(x0 * kernel, dim=1, keepdim=True)
        #pl1 = torch.sum(x1u * kernel, dim=1, keepdim=True)
        #pl2 = torch.sum(x3u * kernel, dim=1, keepdim=True)
        #pl3 = torch.sum(x4u * kernel, dim=1, keepdim=True)
        
        
        #masked_x = (mask * base_feat).view(1, 64, -1, 1)
        #new_x = base_feat.view(1, 64, 1, -1)
        #print(masked_x.shape, new_x.shape)
        #pred = torch.max(torch.sum(masked_x * new_x, dim=1, keepdim=True), dim=2)[0]
        #print(pred.shape)
        #pred = pred.view(1, 1, 20, 20)
        
         
        
        #pred1 = nn.functional.interpolate(pred1, size=x_size, mode='bilinear')
        
        
        OutDict = {}
        OutDict['refine'] = [ref_pred, ] # [pl0, pl1, pl2, pl3] #
        OutDict['sal'] = [pred, x2p, x3p, x4p] # [pred, ] # 
        OutDict['final'] = pred
        
        return OutDict
    


class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        # encoder: backbone, forward function output 5 encoder features. details in methods/base/model.py
        # feat: length of encoder features. e.g.: VGG:[64, 128, 256, 512, 512]; Resnet:[64, 256, 512, 1024, 2048]
        super(Network, self).__init__()

        self.encoder = encoder # resnet(True) #
        
        #self.feat = feat[0] // 2
        #self.adapters = nn.ModuleList([nn.Sequential(*list(up_conv(in1, self.feat, False))) for in1 in feat])
        #feat = [feat[0] // 2, ] * 5
        self.decoder = decoder(config, encoder, feat)

    def forward(self, x, phase='test'):
        x_size = x.size()[2:]
        xs = self.encoder(x)
        
        w = self.encoder.fc.weight
        b = self.encoder.fc.bias
        
        #out = self.decoder(xs, x, x_size, phase)
        out = self.decoder(xs, [w, b], x_size, phase)
        return out
