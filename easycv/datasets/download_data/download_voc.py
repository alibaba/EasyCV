# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from easycv.utils.constant import CACHE_DIR
from .commont import check_path_exists, download, extract

cfg = dict(
    voc2007=
    'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    voc2012=
    'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
)


# return abs path
def process_voc(path, split='train'):
    if split == 'train':
        return os.path.join(path, 'ImageSets/Main/train.txt')
    else:
        return os.path.join(path, 'ImageSets/Main/val.txt')


def download_voc(name, split='train', target_dir=CACHE_DIR):

    root_path = os.path.join(target_dir, 'VOCdevkit', name.upper())
    # 查看此根目录路径是否存在
    if os.path.exists(os.path.join(root_path)):
        return check_path_exists({'path': process_voc(root_path, split=split)})
    # 开始下载
    file_path = download(cfg.get(name), target_dir=target_dir)

    # 开始解压
    extract(name, file_path, target_dir=target_dir)

    return check_path_exists({'path': process_voc(root_path, split=split)})
