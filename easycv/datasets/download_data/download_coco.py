# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import os

from easycv.utils.constant import CACHE_DIR
from .commont import check_path_exists, download, extract

cfg = dict(
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


# return output dir
def process_coco(path, split='train', task='detection'):
    annotations_path = os.path.join(path, 'annotations')
    map_path = dict()
    map_path['ann_file'] = os.path.join(annotations_path, cfg[task][split])
    map_path['img_prefix'] = os.path.join(path, cfg[split + '_dataset'])
    return check_path_exists(map_path)


def regularization_path(name, target_dir=CACHE_DIR):
    file_list = glob.glob(os.path.join(target_dir, name.upper(), '*.json'))
    annotations_dir = os.path.join(target_dir, name.upper(), 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)
    for tmp in file_list:
        cmd = f'mv {tmp} {annotations_dir}'
        os.system(cmd)
    return os.path.join(target_dir, name.upper())


def download_coco(name, split='train', target_dir=CACHE_DIR, task='detection'):

    root_path = os.path.join(target_dir, name.upper())
    # 查看此根目录路径是否存在
    if os.path.exists(root_path):
        return process_coco(root_path, split=split, task=task)

    download_finished = list()
    for link in cfg.get(name):
        download_finished.append(download(link, target_dir=target_dir))

    for file_path in download_finished:
        extract(name, file_path, target_dir=target_dir)

    # 规范化路径
    path = regularization_path(name, target_dir=target_dir)

    return process_coco(path, split=split, task=task)
