# Copyright (c) Alibaba, Inc. and its affiliates.

import os

import wget

# The location where downloaded data is stored
from easycv.utils.constant import CACHE_DIR

COCO_CFG = dict(
    coco2017=[
        'http://images.cocodataset.org/zips/train2017.zip',
        'http://images.cocodataset.org/zips/val2017.zip',
        'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
    ],
    detection=dict(
        train='instances_train2017.json',
        val='instances_val2017.json',
    ),
    train_dataset='train2017',
    val_dataset='val2017',
    pose=dict(
        train='person_keypoints_train2017.json',
        val='person_keypoints_val2017.json'))

VOC_CFG = dict(
    voc2007=
    'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    voc2012=
    'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
)


def download(link, target_dir=CACHE_DIR):
    file_name = wget.filename_from_url(link)
    # Check whether the compressed package exists. If no, download the compressed package
    if not os.path.exists(os.path.join(target_dir, file_name)):
        try:
            print(f'{file_name} is start downlaod........')
            file_name = wget.download(link, out=target_dir)
            print(f'{file_name} is download finished\n')
        except:
            print(f'{file_name} is download fail')
            exit()
    # The prevention of Ctrol + C
    if not os.path.exists(os.path.join(target_dir, file_name)):
        exit()
    return os.path.join(target_dir, file_name)


def check_path_exists(map_path):
    for value in map_path.values():
        assert os.path.exists(value), f'{value} is not exists'
    return map_path
