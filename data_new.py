import os, random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch

#mean = np.array((104.00699, 116.66877, 122.67892)).reshape((1, 1, 3))
mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def get_image_list(name, config, phase):
    images = []
    gts = []
    
    image_root = os.path.join(config['data_path'], name, 'images')
    if phase == 'train' and name == 'MSB-TR':
        #tag = 'imagenet'
        tag = 'moco'
        #tag = 'segmentations'
        #gt_root = os.path.join('data', name, 'crf')
    else:
        tag = 'segmentations'
        
    print(tag)
    gt_root = os.path.join(config['data_path'], name, tag)
        
    
    '''
    if name == 'MSB-TR':
        with open(os.path.join(config['data_path'], name, '{}.txt'.format('train'))) as f:
            lines = f.readlines()
        images = sorted([os.path.join(image_root, '.'.join(line.split('.')[:-1]) + '.jpg') for line in lines])
        gts = sorted([os.path.join(gt_root, '.'.join(line.split('.')[:-1]) + '.png') for line in lines])
    else:
    '''
    
    images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')])
    gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')])
    
    
    #print(len(images), len(gts))
    '''
    if name in ('DUTS-TR', 'SOD', 'PASCAL-S', 'ECSSD', 'HKU-IS', 'DUTS-TE', 'DUT-OMRON'):
        image_root = os.path.join(config['data_path'], name, 'images')
        gt_root = os.path.join(config['data_path'], name, 'segmentations')
        #if os.path.exists(os.path.join('data/pseudo', name)):
        #    gt_root = os.path.join('data/pseudo', name)
        #else:
        #    gt_root = os.path.join(config['data_path'], name, 'segmentations')
            
        
        images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')])
        gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')])
        #print(len(images), len(gts))
    '''
    return images, gts

def get_loader(config):
    dataset = Train_Dataset(config['trset'], config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['batch'],
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=True)
    return data_loader

def random_light(x):
    contrast = np.random.rand(1)+0.5
    light = np.random.randint(-20,20)
    x = contrast*x + light
    return np.clip(x,0,255)

def rotate(img, gt):
    angle = np.random.randint(-25,25)
    img = img.rotate(angle)
    gt = gt.rotate(angle)
    return img, gt

class Train_Dataset(data.Dataset):
    def __init__(self, config):
        self.config = config
        self.images_list, self.gts_list = get_image_list(config['trset'], config, 'train')
        self.size = len(self.images_list)
        self.dataset_name = config['trset']
        
        self.images, self.gts = self.load_data(self.images_list, self.gts_list)
        #self.images, self.gts = self.load_data(self.images_list[:10], self.gts_list[:10])
        print(len(self.images), len(self.gts))

    def load_data(self, images_list, gts_list):
        images = []
        gts = []
        for image_path, gt_path in zip(images_list, gts_list):
            image = Image.open(image_path).convert('RGB')
            gt = Image.open(gt_path).convert('L')
            
            if self.config['data_aug']:
                image, gt = rotate(image, gt)
                image = random_light(image)
            
            img_size = self.config['size']
            image = image.resize((img_size, img_size))
            gt = gt.resize((img_size, img_size))
        
            image = np.array(image).astype(np.float32)
            gt = np.array(gt)
            
            #print(image.shape, gt.shape)
            if random.random() > 0.5:
                image = image[:, ::-1]
                gt = gt[:, ::-1]
            
            image = ((image / 255.) - mean) / std
            image = image.transpose((2, 0, 1))
            gt = np.expand_dims(gt / 255., axis=0)
            
            images.append(image)
            gts.append(gt)
        return torch.tensor(np.array(images)).float().cuda(), torch.tensor(np.array(gts)).float().cuda()
        # return torch.tensor(np.array(images)).float(), torch.tensor(np.array(gts)).float()

    def __len__(self):
        return self.size

class Test_Dataset:
    def __init__(self, name, config=None):
        self.config = config
        self.images, self.gts = get_image_list(name, config, 'test')
        self.size = len(self.images)
        self.dataset_name = name

    def load_data(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if not self.config['orig_size']:
            image = image.resize((self.config['size'], self.config['size']))
        image = np.array(image).astype(np.float32)
        gt = np.array(Image.open(self.gts[index]).convert('L'))
        name = self.images[index].split('/')[-1].split('.')[0]
        
        
        image = ((image / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        image = torch.tensor(np.expand_dims(image, 0)).float()
        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
        #gt = (gt > 0.5).astype(np.float32)
        #gt = (gt > 128).astype(np.float32)
        
        return image, gt, name

def test_data():
    config = {'orig_size': True, 'size': 288, 'data_path': '../dataset'}
    dataset = 'SOD'
    
    '''
    data_loader = Test_Dataset(dataset, config)
    #data_loader = Train_Dataset(dataset, config)
    data_size = data_loader.size
    
    for i in range(data_size):
        img, gt, name = data_loader.load_data(i)
        #img, gt = data_loader.__getitem__(i)
        new_img = (img * std + mean) * 255.
        #new_img = gt * 255
        print(np.min(new_img), np.max(new_img))
        new_img = (new_img).astype(np.uint8)
        #print(new_img.shape).astype(np.)
        im = Image.fromarray(new_img)
        #im.save('temp/' + name + '.jpg')
        im.save('temp/' + str(i) + '.jpg')
    
    '''
    
    data_loader = Val_Dataset(dataset, config)
    imgs, gts, names = data_loader.load_all_data()
    print(imgs.shape, gts.shape, len(names))
    

if __name__ == "__main__":
    test_data()