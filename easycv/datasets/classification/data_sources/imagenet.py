# Copyright (c) Alibaba, Inc. and its affiliates.

from PIL import Image
import os
from easycv.datasets.registry import DATASOURCES


@DATASOURCES.register_module
class ClsSourceImageNet1k(object):

    def generate_label_json(self, label_path):
        label_txt = open(label_path).readlines()
        assert len(label_txt) > 1, f"{label_path} is None"
        label_Json = dict()
        for line in label_txt:
            deal_line = line.strip().split()
            label_Json[deal_line[0]] = (deal_line[1], deal_line[2])

        return label_Json

    def read_train_data(self, train_path):
        assert os.path.exists(train_path), f"{train_path} is not exists"
        train_txt = open(train_path).readlines()
        assert len(train_path) > 1, f"{train_path} is None"
        return train_txt

    def read_image_data(self, image_path):
        img_path = os.path.join(self.root, image_path.strip())
        assert os.path.exists(img_path), f"{img_path} is not exists"
        img = Image.open(img_path, mode='r')  # img: HWC, RGB
        label_key = image_path.split('/')[1]
        assert label_key in self.label_json, f"{label_key} label is not exists"
        label = self.label_json[label_key]

        return {"img":img , "gt_labels": int(label[0].strip())}

    def __init__(self, root, split):
        self.root = root
        assert split in ['train', 'test']
        if split == 'train':
            self.split = "meta/train.txt"
        else:
            self.split = "meta/val.txt"
        self.train_path = self.read_train_data(os.path.join(root, self.split))
        self.label_json = self.generate_label_json(os.path.join(root, "meta/label_name.txt"))

    def __len__(self):
        return len(self.train_path)

    def __getitem__(self, idx):
        return self.read_image_data(self.train_path[idx])