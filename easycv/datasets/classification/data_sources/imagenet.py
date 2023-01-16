# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from PIL import Image

from easycv.datasets.registry import DATASOURCES
from easycv.file.image import load_image


def generate_label_map(label_path):
    label_txt = open(label_path).readlines()
    assert len(label_txt) > 1, f'{label_path} is None'
    label_map = dict()
    for line in label_txt:
        deal_line = line.strip().split()
        label_map[deal_line[0]] = (deal_line[1], deal_line[2])
    return label_map


def get_images_list(txt_path):
    assert os.path.exists(txt_path), f'{txt_path} is not exists'
    train_txt = open(txt_path).readlines()
    assert len(txt_path) > 1, f'{txt_path} is None'
    return train_txt


@DATASOURCES.register_module
class ClsSourceImageNet1k(object):

    def __init__(self, root, split):
        """
        Args:
            root: The root directory of the data
                    example：if data/imagenet
                                └── train
                                    └── n01440764
                                    └── n01443537
                                    └── ...
                                └── val
                                    └── n01440764
                                    └── n01443537
                                    └── ...
                                └── meta
                                    ├── train.txt
                                    ├── val.txt
                                    ├── ...
                                has input root = data/imagenet
            split : train  or val
        """
        self.root = root
        assert split in ['train', 'test']
        if split == 'train':
            self.split = 'meta/train.txt'
        else:
            self.split = 'meta/val.txt'
        self.txt_path = get_images_list(os.path.join(root, self.split))
        self.label_json = generate_label_map(
            os.path.join(root, 'meta/label_name.txt'))

    def read_data(self, image_path):
        img_path = os.path.join(self.root, image_path.strip())
        assert os.path.exists(img_path), f'{img_path} is not exists'
        img = load_image(img_path, mode='RGB')
        img = Image.fromarray(img)
        label_key = image_path.split('/')[1]
        assert label_key in self.label_json, f'{label_key} label is not exists'
        label = self.label_json[label_key]
        return {'img': img, 'gt_labels': int(label[0].strip())}

    def __len__(self):
        return len(self.txt_path)

    def __getitem__(self, idx):
        return self.read_data(self.txt_path[idx])
