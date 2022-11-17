# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import os

from easycv.utils.constant import CACHE_DIR
from .commont import COCO_CFG, check_path_exists, download


# extract the data
def extract(file, target_dir=CACHE_DIR):

    save_dir = os.path.join(target_dir, 'COCO2017')
    os.makedirs(save_dir, exist_ok=True)
    cmd = f'unzip -d {save_dir} {file}'
    print('begin Unpack.....................')
    os.system(cmd)
    print('Unpack is finished.....................')


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


# Check whether the data exists
def check_data_exists(target_dir, split, task):
    root_path = os.path.join(target_dir, 'COCO2017')
    if os.path.exists(root_path):
        return process_coco(root_path, split=split, task=task)
    else:
        return False


def download_coco(name,
                  split='train',
                  target_dir=CACHE_DIR,
                  task='detection',
                  **kwargs):
    # Declare a global
    global cfg
    # Use it for testing
    if kwargs.get('cfg'):
        cfg = kwargs.get('cfg')
    else:
        cfg = COCO_CFG

    if check_data_exists(target_dir, split, task):
        return check_data_exists(target_dir, split, task)

    download_finished = list()
    for link in cfg.get(name):
        download_finished.append(download(link, target_dir=target_dir))

    for file_path in download_finished:
        extract(file_path, target_dir=target_dir)

    # Path of regularization
    path = regularization_path(name, target_dir=target_dir)

    return process_coco(path, split=split, task=task)
