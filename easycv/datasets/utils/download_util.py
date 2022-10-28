# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import os

import wget

# The location where downloaded data is stored
from easycv.utils.constant import CACHE_DIR


def download(link, target_dir=CACHE_DIR):
    file_name = wget.filename_from_url(link)
    # 查看是否有压缩包，无压缩包的话下载压缩包
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


def extrack(name, file, target_dir=CACHE_DIR):
    if name == 'coco2017':
        save_dir = os.path.join(target_dir, name.upper())
        os.makedirs(save_dir, exist_ok=True)
        cmd = f'unzip -d {save_dir} {file}'
    else:
        cmd = f'tar -xvf {file} -C {target_dir}'
    print('begin Unpack.....................')
    os.system(cmd)
    print('Unpack is finished.....................')


def check_path_exists(map_path):
    for value in map_path.values():
        assert os.path.exists(value), f'{value} is not exists'
    return map_path


def download_voc(name, split='train', target_dir=CACHE_DIR):
    link = dict(
        voc2007=
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        voc2012=
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    )

    def process_voc(path, split='train'):
        if split == 'train':
            return os.path.join(path, 'ImageSets/Main/train.txt')
        else:
            return os.path.join(path, 'ImageSets/Main/val.txt')

    root_path = os.path.join(target_dir, 'VOCdevkit', name.upper())
    # 查看此根目录路径是否存在
    if os.path.exists(os.path.join(root_path)):
        return check_path_exists({'path': process_voc(root_path, split=split)})
    # 开始下载
    file_path = download(link.get(name), target_dir=target_dir)

    # 开始解压
    extrack(name, file_path, target_dir=target_dir)

    return check_path_exists({'path': process_voc(root_path, split=split)})


def download_coco(name, split='train', target_dir=CACHE_DIR, task='detection'):
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

    # output dir
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

    root_path = os.path.join(target_dir, name.upper())
    # 查看此根目录路径是否存在
    if os.path.exists(root_path):
        return process_coco(root_path, split=split, task=task)

    download_finished = list()
    for link in cfg.get(name):
        download_finished.append(download(link, target_dir=target_dir))

    for file_path in download_finished:
        extrack(name, file_path, target_dir=target_dir)

    # 规范化路径
    path = regularization_path(name, target_dir=target_dir)

    return process_coco(path, split=split, task=task)
