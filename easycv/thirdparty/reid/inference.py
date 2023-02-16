# -*- coding: utf-8 -*-
import argparse
import torch
import numpy as np
import time
import os
import scipy.io

from easycv.predictors.classifier import ClassificationPredictor

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--config', default=None, help='config file')
parser.add_argument('--checkpoint', help='checkpoint file')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
args = parser.parse_args()

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

def extract_feature(model, image_dir):
    image_path = file_name_walk(image_dir)
    image_cam, image_label = get_id(image_path)
    image_feature = model(image_path, mode='extract')
    image_feature = torch.cat(image_feature, 0)
    image_feature_norm = torch.norm(image_feature, p=2, dim=1, keepdim=True)
    image_feature = image_feature.div(image_feature_norm.expand_as(image_feature))
    return image_feature, image_cam, image_label

gallery_dir = os.path.join(args.test_dir, 'gallery')
query_dir = os.path.join(args.test_dir, 'query')

# build model
model = ClassificationPredictor(
    model_path=args.checkpoint,
    config_file=args.config,
    batch_size=args.batchsize)

# extract features
since = time.time()
gallery_feature, gallery_cam, gallery_label = extract_feature(model, gallery_dir)
query_feature, query_cam, query_label = extract_feature(model, query_dir)
print(gallery_feature.size(), query_feature.size())
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

# save inference result
inference_result_path = os.path.join(os.path.dirname(args.checkpoint), 'pytorch_result.mat')
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat(inference_result_path, result)

# evaluate
os.system(f'python easycv/thirdparty/reid/evaluate.py {inference_result_path}')