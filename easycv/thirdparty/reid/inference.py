# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math

from easycv.predictors.classifier import ClassificationPredictor

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
cfg = parser.parse_args()

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model, dataloaders):
    count = 0

    for iter, data in enumerate(dataloaders):
        img, _ = data
        n, c, h, w = img.size()
        count += n
        print(count)

        # outputs = torch.FloatTensor(n, 2048).zero_().cuda()
        # for i in range(2):
        #     if i == 0:
        #         input_img = img.cuda()
        #     else:
        #         input_img = fliplr(img).cuda()
        input_img =  img.cuda()
        outputs = model([input_img], mode='extract')
        # norm feature
        fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
        outputs = outputs.div(fnorm.expand_as(outputs))

        
        if iter == 0:
            features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
        #features = torch.cat((features,ff.data.cpu()), 0)
        start = iter*cfg.batchsize
        end = min( (iter+1)*cfg.batchsize, len(dataloaders.dataset))
        features[ start:end, :] = outputs
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

######################################################################
# prepare dataset
transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor(),
])
image_datasets = {x: datasets.ImageFolder(os.path.join(cfg.test_dir, x), transform) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=cfg.batchsize,
                                            shuffle=False, num_workers=16) for x in ['gallery','query']}
gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')

# Extract feature
config_file = 'configs/classification/imagenet/resnet/market1501_resnet50_jpg.py'
checkpoint = '/home/yunji.cjy/projects/reid/epoch_60.pth'
# img_path = '/home/yunji.cjy/0002_c1s1_000451_03.jpg'
model = ClassificationPredictor(
    model_path=checkpoint,
    config_file=config_file)
# results = model([img_path], mode='extract')
# print(results)
# exit()

since = time.time()
gallery_feature = extract_feature(model, dataloaders['gallery'])
query_feature = extract_feature(model, dataloaders['query'])
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)

print(opt.name)
result = './model/%s/result.txt'%opt.name
os.system('python evaluate_gpu.py | tee -a %s'%result)
