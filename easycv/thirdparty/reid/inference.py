# -*- coding: utf-8 -*-
import argparse
import torch
import numpy as np
import time
import os
import scipy.io

from easycv.predictors.classifier import ClassificationPredictor

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
cfg = parser.parse_args()

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def get_id(img_path):
    camera_id = []
    labels = []
    for path in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def file_name_walk(file_dir):
    image_path_list = []
    for root, dirs, files in os.walk(file_dir):
        for name in files:
            if name.endswith('.jpg'):
                image_path_list.append(os.path.join(root, name))
    return image_path_list

######################################################################
# prepare dataset
gallery_path = file_name_walk('/apsarapangu/disk2/yunji.cjy/Market1501/pytorch/gallery')
query_path = file_name_walk('/apsarapangu/disk2/yunji.cjy/Market1501/pytorch/query')

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

######################################################################
# extract features
print('-------test-----------')

# Extract feature
config_file = 'configs/classification/imagenet/resnet/market1501_resnet50_jpg.py'
checkpoint = '/home/yunji.cjy/projects/reid/epoch_60.pth'
model = ClassificationPredictor(
    model_path=checkpoint,
    config_file=config_file,
    batch_size=cfg.batchsize)

since = time.time()
gallery_feature = model(gallery_path, mode='extract')
query_feature = model(query_path, mode='extract')
gallery_feature = torch.cat(gallery_feature, 0)
query_feature = torch.cat(query_feature, 0)
gallery_feature_norm = torch.norm(gallery_feature, p=2, dim=1, keepdim=True)
gallery_feature = gallery_feature.div(gallery_feature_norm.expand_as(gallery_feature))
query_feature_norm = torch.norm(query_feature, p=2, dim=1, keepdim=True)
query_feature = query_feature.div(query_feature_norm.expand_as(query_feature))
print(gallery_feature.size(), query_feature.size())
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('easycv/thirdparty/reid/pytorch_result.mat',result)
