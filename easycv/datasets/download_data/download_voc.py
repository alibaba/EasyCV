# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from easycv.utils.constant import CACHE_DIR
from .commont import check_path_exists, download

cfg = dict(
    voc2007=
    'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    voc2012=
    'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
)


# Check whether the data exists
def check_data_exists(name, target_dir, split):
    root_path = os.path.join(target_dir, 'VOCdevkit', name.upper())
    if os.path.exists(root_path):
        return check_path_exists({'path': process_voc(root_path, split=split)})
    else:
        return False


# return abs path
def process_voc(path, split='train'):
    if split == 'train':
        return os.path.join(path, 'ImageSets/Main/train.txt')
    else:
        return os.path.join(path, 'ImageSets/Main/val.txt')


# xtract the data
def extract(name, file, target_dir=CACHE_DIR):
    cmd = f'tar -xvf {file} -C {target_dir}'
    print('begin Unpack.....................')
    os.system(cmd)
    print('Unpack is finished.....................')


def download_voc(name, split='train', target_dir=CACHE_DIR):

    if check_data_exists(name, target_dir, split):
        return check_data_exists(name, target_dir, split)

    # Start the download
    file_path = download(cfg.get(name), target_dir=target_dir)

    # began to unpack
    extract(name, file_path, target_dir=target_dir)

    return check_data_exists(name, target_dir, split)
